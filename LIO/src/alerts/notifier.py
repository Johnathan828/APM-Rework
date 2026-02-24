# APM/LIO/src/alerts/notifier.py
from __future__ import annotations

import base64
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo

from ..apm_core.ssot import get_sensor
from ..apm_core.raw_pull import build_wide_frame


@dataclass
class PlotBundle:
    short_plot_b64: str
    long_plot_b64: str


class Notifier:
    """
    Legacy-like email formatting:
      - High priority headers
      - Emoji body + "Likely Cause" table
      - Two embedded plots (short + past 4 weeks)
    Uses:
      - Email_DEV when global_debug=True
      - Email when global_debug=False
    """

    def __init__(self, *, ssot: Dict[str, Any], sensor_name: str, logger, db, global_debug: bool):
        self.ssot = ssot
        self.sensor_name = sensor_name
        self.logger = logger
        self.db = db
        self.global_debug = global_debug

        self.email_section = "Email_DEV" if global_debug else "Email"
        s = db.settings  # IniSettings

        self.from_address = s.get(self.email_section, "from_address")
        self.password = s.get(self.email_section, "password")
        self.smtp_server = s.get(self.email_section, "smtp_server")
        self.smtp_port = int(s.get(self.email_section, "smtp_port", fallback="587"))
        self.generate_emails = str(s.get(self.email_section, "generate_emails", fallback="True")).lower() == "true"
        to_raw = s.get(self.email_section, "to_address", fallback="")
        self.to_addresses = [x.strip() for x in to_raw.split(",") if x.strip()]

        self.tz_local = ZoneInfo("Africa/Johannesburg")
        self.tz_utc = ZoneInfo("UTC")

    # ---------------------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------------------
    def _ts_strings(self, ts_utc_naive: datetime) -> Tuple[str, str]:
        # ts is stored UTC-naive in our runtime. Convert for display.
        ts_utc = ts_utc_naive.replace(tzinfo=self.tz_utc)
        ts_local = ts_utc.astimezone(self.tz_local)
        return ts_local.strftime("%Y-%m-%d %H:%M:%S %Z"), ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    def _build_likely_cause_table(self, feature_contrib: Optional[pd.DataFrame]) -> str:
        """
        Legacy-style table:
          - Uses max contribution per feature over the window
          - Displays Description + "🔴 Likely Cause" / "🟢 Not Likely"
        """
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        features = cfg.get("Features", {}) or {}

        if feature_contrib is None or feature_contrib.empty:
            df = pd.DataFrame([{"Tag": "(none)", "Description": "(none)", "Anomaly Cause": "🟢 Not Likely"}])
            return self._style_table(df)

        # feature_contrib columns are displaynames, values 0/1 (currently)
        max_vals = feature_contrib.max()

        rows = []
        for disp, v in max_vals.items():
            meta = features.get(disp, {}) if isinstance(features, dict) else {}
            desc = str(meta.get("description", disp)) if isinstance(meta, dict) else str(disp)
            rows.append(
                {
                    "Tag": disp,
                    "Description": desc,
                    "Anomaly Cause": ("🔴 Likely Cause" if float(v) > 0 else "🟢 Not Likely"),
                }
            )

        df = pd.DataFrame(rows)
        return self._style_table(df)

    def _style_table(self, df: pd.DataFrame) -> str:
        html = df.to_html(escape=False, index=False)
        css = """
        <style>
            table {
                width: 75%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
            }
            th, td {
                padding: 8px;
                text-align: left;
                border: 1px solid #ddd;
            }
            th {
                background-color: #2e3647;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #ddd;
            }
        </style>
        """
        return css + html

    # ---------------------------------------------------------------------
    # Plotting (legacy-ish)
    # ---------------------------------------------------------------------
    def _score_fixed_thresholds(
        self,
        *,
        df_wide: pd.DataFrame,
        sensor_cfg: Dict[str, Any],
        feature_displaynames_to_tags: Dict[str, str],
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Recompute FixedThresholds score purely for plotting (no MSSQL reads).
        Same logic as FixedThresholdsModel:
          score = 1 - mean( outside_bounds_flags )
        """
        method_cfg = ((sensor_cfg.get("Method", {}) or {}).get("FixedThresholds", {}) or {})
        high_map = method_cfg.get("high", {}) or {}
        low_map = method_cfg.get("low", {}) or {}

        contrib_cols: Dict[str, pd.Series] = {}
        viol_flags: List[pd.Series] = []

        for disp, tag in feature_displaynames_to_tags.items():
            if tag not in df_wide.columns:
                contrib_cols[disp] = pd.Series(0.0, index=df_wide.index)
                continue

            s = pd.to_numeric(df_wide[tag], errors="coerce")

            hi = high_map.get(tag, None)
            lo = low_map.get(tag, None)

            outside = pd.Series(False, index=df_wide.index)
            if hi is not None:
                outside = outside | (s > float(hi))
            if lo is not None:
                outside = outside | (s < float(lo))

            outside = outside.fillna(False)
            viol_flags.append(outside.astype(float))
            contrib_cols[disp] = outside.astype(float)

        if viol_flags:
            viol_df = pd.concat(viol_flags, axis=1)
            violation_ratio = viol_df.mean(axis=1)
        else:
            violation_ratio = pd.Series(0.0, index=df_wide.index)

        score = (1.0 - violation_ratio).clip(0.0, 1.0)
        feature_contrib = pd.DataFrame(contrib_cols, index=df_wide.index).fillna(0.0)
        return score, feature_contrib

    def _compute_anomaly_ranges(
        self,
        *,
        score: pd.Series,
        section: pd.Series,
        alarm_thresh: float,
        filter_value: float,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Legacy-ish: anomaly where score < alarm_thresh AND section == open.
        Returns contiguous ranges for shading.
        """
        if score is None or score.empty:
            return []
        score = score.sort_index()
        section = section.reindex(score.index).ffill().fillna(0.0)

        mask = (score < float(alarm_thresh)) & (section >= float(filter_value))
        if not mask.any():
            return []

        ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        start = None
        for ts, is_on in mask.items():
            if is_on and start is None:
                start = ts
            elif (not is_on) and start is not None:
                ranges.append((start, ts))
                start = None
        if start is not None:
            ranges.append((start, score.index.max()))
        return ranges

    def _compute_startup_ranges(
        self,
        *,
        section: pd.Series,
        filter_value: float,
        startup_minutes: int,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Mark periods right after CLOSED->OPEN transition as "startup".
        In legacy sample, they used 3 hours fixed; we use startup_period minutes from SSOT.
        """
        if startup_minutes <= 0 or section is None or section.empty:
            return []

        section = section.sort_index()
        open_mask = section >= float(filter_value)
        prev_open = open_mask.shift(1).fillna(False)
        transitions = (~prev_open) & (open_mask)  # closed -> open

        ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for t in section.index[transitions]:
            ranges.append((t, t + pd.Timedelta(minutes=int(startup_minutes))))
        return ranges

    def _fig_to_base64(self, fig) -> str:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        out = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return out

    def _generate_plots(
        self,
        *,
        latest_ts: datetime,
        alarm_thresh: float,
        filter_value: float,
    ) -> PlotBundle:
        """
        Short plot: last ~10 hours
        Long plot: past 4 weeks view (~672 hours)
        """
        sensor = get_sensor(self.ssot, self.sensor_name)
        sensor_cfg = sensor.cfg

        # Map displayname -> raw tag string
        feature_displaynames_to_tags: Dict[str, str] = {}
        for disp, meta in (sensor_cfg.get("Features", {}) or {}).items():
            if isinstance(meta, dict) and meta.get("tag"):
                feature_displaynames_to_tags[disp] = meta["tag"]

        # Legacy numbers (debug changes window lengths; we mirror the idea)
        short_hours = 10
        long_hours = 100 if self.global_debug else 672  # 4 weeks

        def build_plot(hours: int, title_suffix: str) -> str:
            end = latest_ts
            start = latest_ts - timedelta(hours=hours)

            df_wide, filter_tag_series = build_wide_frame(db=self.db, sensor=sensor, start=start, end=end)

            if df_wide is None or df_wide.empty:
                fig = plt.figure(figsize=(14, 4))
                ax = fig.add_subplot(111)
                ax.set_title(f"No raw data available ({title_suffix})")
                ax.grid(True)
                return self._fig_to_base64(fig)

            # Standardize index
            df_wide = df_wide.copy()
            df_wide.index = pd.to_datetime(df_wide.index, utc=True).tz_convert(None)
            df_wide = df_wide.sort_index()

            filter_tag_series = filter_tag_series.copy()
            filter_tag_series.index = pd.to_datetime(filter_tag_series.index, utc=True).tz_convert(None)
            filter_tag_series = filter_tag_series.sort_index()

            # Build score + contrib (FixedThresholds only for now)
            score, feature_contrib = self._score_fixed_thresholds(
                df_wide=df_wide,
                sensor_cfg=sensor_cfg,
                feature_displaynames_to_tags=feature_displaynames_to_tags,
            )

            # Section status (open/closed)
            section = filter_tag_series.reindex(score.index).ffill().fillna(0.0)

            # Startup and anomaly ranges for shading
            startup_minutes = int((sensor_cfg.get("Other", {}) or {}).get("startup_period", 0))
            startup_ranges = self._compute_startup_ranges(section=section, filter_value=filter_value, startup_minutes=startup_minutes)
            anomaly_ranges = self._compute_anomaly_ranges(score=score, section=section, alarm_thresh=alarm_thresh, filter_value=filter_value)

            # --- Figure layout: Section status strip + score + per-feature plots ---
            feature_tags = list(feature_displaynames_to_tags.values())
            n_feat = len(feature_tags)

            fig_h = 8 + 1.2 * max(1, n_feat)
            fig = plt.figure(figsize=(14, fig_h))
            gs = fig.add_gridspec(nrows=(2 + n_feat), ncols=1, height_ratios=[0.4, 1.0] + [1.0] * n_feat)

            # Section status strip
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.set_title("Section Status")
            ax0.set_yticks([])
            ax0.grid(True, axis="x")

            # draw bars
            for ts_i, val in section.fillna(0.0).items():
                # grey off, green on, orange startup, red anomaly
                color = "grey" if val < filter_value else "green"
                ax0.barh(0, 1, left=ts_i, height=1, color=color, align="edge")

            # overlay startup + anomaly shading on strip
            for a, b in startup_ranges:
                ax0.axvspan(a, b, alpha=0.25, color="orange")
            for a, b in anomaly_ranges:
                ax0.axvspan(a, b, alpha=0.25, color="red")

            ax0.set_xlim(section.index.min(), section.index.max())
            ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

            # Score plot
            ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
            ax1.plot(score.index, score.values, label="Health Score")
            ax1.axhline(float(alarm_thresh), linestyle="--", label="Alarm Threshold")
            for a, b in anomaly_ranges:
                ax1.axvspan(a, b, alpha=0.25, color="red")
            for a, b in startup_ranges:
                ax1.axvspan(a, b, alpha=0.15, color="orange")
            ax1.set_ylabel("Score")
            ax1.set_title(f"System Health Score: {sensor_cfg.get('Model_Description', self.sensor_name)} ({title_suffix})")
            ax1.grid(True)
            ax1.legend()

            # Feature plots (raw values) with threshold lines
            method_cfg = ((sensor_cfg.get("Method", {}) or {}).get("FixedThresholds", {}) or {})
            high_map = method_cfg.get("high", {}) or {}
            low_map = method_cfg.get("low", {}) or {}

            for i, (disp, tag) in enumerate(feature_displaynames_to_tags.items()):
                ax = fig.add_subplot(gs[2 + i, 0], sharex=ax0)
                if tag in df_wide.columns:
                    ax.plot(df_wide.index, pd.to_numeric(df_wide[tag], errors="coerce"), label=disp)
                ax.grid(True)

                hi = high_map.get(tag, None)
                lo = low_map.get(tag, None)
                if hi is not None:
                    ax.axhline(float(hi), linestyle="--", label="High")
                if lo is not None:
                    ax.axhline(float(lo), linestyle="--", label="Low")

                for a, b in anomaly_ranges:
                    ax.axvspan(a, b, alpha=0.20, color="red")
                for a, b in startup_ranges:
                    ax.axvspan(a, b, alpha=0.12, color="orange")

                meta = (sensor_cfg.get("Features", {}) or {}).get(disp, {}) or {}
                unit_desc = str(meta.get("unit_description", ""))
                unit = str(meta.get("unit", ""))
                ax.set_ylabel(f"{unit_desc} ({unit})".strip())
                ax.set_title(str(meta.get("description", disp)))
                ax.legend(loc="upper right")

            fig.autofmt_xdate()
            return self._fig_to_base64(fig)

        short_b64 = build_plot(short_hours, "Short View")
        long_b64 = build_plot(long_hours, "Past 4 Weeks View" if not self.global_debug else "Long View (Debug)")
        return PlotBundle(short_plot_b64=short_b64, long_plot_b64=long_b64)

    # ---------------------------------------------------------------------
    # Email build/send
    # ---------------------------------------------------------------------
    def _build_html(
        self,
        *,
        score: float,
        latest_ts: datetime,
        feature_contrib: Optional[pd.DataFrame],
        plots: PlotBundle,
    ) -> str:
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        model_desc = str(cfg.get("Model_Description", self.sensor_name))

        feature_table_html = self._build_likely_cause_table(feature_contrib)
        ts_local_s, _ts_utc_s = self._ts_strings(latest_ts)

        # Legacy-like body copy
        body = f"""🚨 Alert! 🚨<br><br>
        We have detected a sustained significant deviation in the operating conditions of the: <b>{model_desc}</b>.<br>
        Please review the system status to restore stability.<br><br>
        The table below highlights the importance of each measured detector:<br>
        {feature_table_html}<br><br>

        <img src="data:image/png;base64,{plots.short_plot_b64}" alt="Plot" /><br><br>

        <h3 style="
            background-color: white;
            color: #6a11cb;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-family: Arial, sans-serif;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            width: 30%;
        ">
            Past 4 Weeks View
        </h3>
        <img src="data:image/png;base64,{plots.long_plot_b64}" alt="Plot" /><br><br>

        <div style="font-family: Arial, sans-serif; color:#444; font-size:12px;">
          Timestamp (Local): {ts_local_s}
        </div>

        You can ask us any questions you might have by posting them on the
        <a href="https://neuromine.atlassian.net/servicedesk/customer/portal/5">Jira Portal</a>.
        """

        return body

    def _send_email(self, *, subject: str, html: str, use_high_priority: bool = True) -> None:
        if not self.generate_emails:
            self.logger.info(f"{self.sensor_name}: generate_emails=False, skipping email")
            return
        if not self.to_addresses:
            self.logger.warning(f"{self.sensor_name}: no to_address configured, skipping email")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_address
        msg["To"] = ", ".join(self.to_addresses)

        if use_high_priority:
            msg["X-Priority"] = "1"
            msg["X-MSMail-Priority"] = "High"
            msg["Importance"] = "High"

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.from_address, self.password)
            server.sendmail(self.from_address, self.to_addresses, msg.as_string())

        self.logger.info(f"{self.sensor_name}: email sent -> {', '.join(self.to_addresses)}")

    # ---------------------------------------------------------------------
    # Public API used by run_once.py
    # ---------------------------------------------------------------------
    def send_test_email(
        self,
        *,
        score: float,
        section_status: float,  # kept for signature compatibility (not required in legacy format)
        latest_ts: datetime,
        feature_contrib: Optional[pd.DataFrame],
        reason: str,  # kept for signature compatibility
        alarm_thresh: Optional[float] = None,
    ) -> None:
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        site = str(self.ssot.get("site", "SITE"))
        model_desc = str(cfg.get("Model_Description", self.sensor_name))

        subject = f"{site} DEBUG TEST EMAIL: {model_desc} model"

        # Use SSOT alarm thresh for plot threshold reference
        alarm_thresh = float(((cfg.get("Other", {}) or {}).get("alarm_thresh", 0.75)))
        filter_value = float(((cfg.get("filter_tag", {}) or {}).get("filter_value", 0.9)))

        plots = self._generate_plots(latest_ts=latest_ts, alarm_thresh=alarm_thresh, filter_value=filter_value)
        html = self._build_html(score=score, latest_ts=latest_ts, feature_contrib=feature_contrib, plots=plots)

        self._send_email(subject=subject, html=html, use_high_priority=False)

    def send_alert(
        self,
        *,
        score: float,
        section_status: float,  # kept for signature compatibility
        latest_ts: datetime,
        feature_contrib: Optional[pd.DataFrame],
        reason: str,  # kept for signature compatibility
        alarm_thresh: Optional[float] = None,
    ) -> None:
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        site = str(self.ssot.get("site", "SITE"))
        model_desc = str(cfg.get("Model_Description", self.sensor_name))

        subject = f"{site} Anomaly Detected by: {model_desc} model"

        alarm_thresh = float(((cfg.get("Other", {}) or {}).get("alarm_thresh", 0.75)))
        filter_value = float(((cfg.get("filter_tag", {}) or {}).get("filter_value", 0.9)))

        plots = self._generate_plots(latest_ts=latest_ts, alarm_thresh=alarm_thresh, filter_value=filter_value)
        html = self._build_html(score=score, latest_ts=latest_ts, feature_contrib=feature_contrib, plots=plots)

        self._send_email(subject=subject, html=html, use_high_priority=True)