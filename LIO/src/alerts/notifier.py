from __future__ import annotations

import base64
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
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

    Timezone behavior:
      - Plots are rendered in a "display timezone"
      - display_timezone precedence:
          config.ini [Email/Email_DEV] display_timezone -> ssot["timezone"] -> UTC
      - Email body includes a small footer with Display + UTC timestamps
    """

    def __init__(self, *, ssot: Dict[str, Any], sensor_name: str, logger, db, global_debug: bool):
        self.ssot = ssot
        self.sensor_name = sensor_name
        self.logger = logger
        self.db = db
        self.global_debug = global_debug

        self.email_section = "Email_DEV" if global_debug else "Email"
        s = db.settings  # IniSettings / ConfigParser-like

        self.from_address = s.get(self.email_section, "from_address")
        self.password = s.get(self.email_section, "password")
        self.smtp_server = s.get(self.email_section, "smtp_server")
        self.smtp_port = int(s.get(self.email_section, "smtp_port", fallback="587"))
        self.generate_emails = str(s.get(self.email_section, "generate_emails", fallback="True")).lower() == "true"
        to_raw = s.get(self.email_section, "to_address", fallback="")
        self.to_addresses = [x.strip() for x in to_raw.split(",") if x.strip()]

        # Display timezone precedence: config.ini -> SSOT -> UTC
        tz_name = (s.get(self.email_section, "display_timezone", fallback=None) or str(self.ssot.get("timezone", "UTC")))
        self.display_tz = ZoneInfo(tz_name)
        self.tz_utc = ZoneInfo("UTC")

    # ---------------------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------------------
    def _ts_strings(self, ts_utc_naive: datetime) -> Tuple[str, str]:
        ts_utc = ts_utc_naive.replace(tzinfo=self.tz_utc)
        ts_disp = ts_utc.astimezone(self.display_tz)
        return ts_disp.strftime("%Y-%m-%d %H:%M:%S %Z"), ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

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

    def _trigger_text(
        self,
        *,
        method_name: str,
        raw_tag: str,
        is_likely: bool,
        trigger_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Trigger column text rules (WINDOW-BASED):

        Threshold methods (FixedThresholds / StatisticalThresholds):
          - if ANY value in window > high -> "> High Threshold: <HIGH_THRESHOLD_VALUE>"
          - elif ANY value in window < low -> "< Low Threshold: <LOW_THRESHOLD_VALUE>"
          - else -> "Not Triggered"
        Only the THRESHOLD value is shown (not the raw value).

        Non-threshold methods:
          - default: "Triggered by Model" for Likely Cause rows
          - non-likely rows: "Not Triggered"
        """
        if not is_likely:
            return "Not Triggered"

        if method_name in ("FixedThresholds", "StatisticalThresholds"):
            ctx = trigger_context or {}

            raw_window_by_tag = (ctx.get("raw_window_by_tag") or {})
            s = raw_window_by_tag.get(raw_tag, None)

            thr = (ctx.get("thresholds") or {})
            thr_high = (thr.get("high") or {})
            thr_low = (thr.get("low") or {})

            hi = thr_high.get(raw_tag, None)
            lo = thr_low.get(raw_tag, None)

            try:
                if s is not None and hi is not None:
                    s_num = pd.to_numeric(s, errors="coerce")
                    if (s_num > float(hi)).fillna(False).any():
                        return f"> High Threshold: {float(hi):.2f}"

                if s is not None and lo is not None:
                    s_num = pd.to_numeric(s, errors="coerce")
                    if (s_num < float(lo)).fillna(False).any():
                        return f"< Low Threshold: {float(lo):.2f}"
            except Exception:
                pass

            return "Not Triggered"

        return "Triggered by Model"

    def _build_likely_cause_table(
        self,
        feature_contrib: Optional[pd.DataFrame],
        *,
        method_name: str,
        trigger_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Table with 4 columns:
          Tag | Description | Anomaly Cause | Trigger
        """
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        features = cfg.get("Features", {}) or {}

        if feature_contrib is None or feature_contrib.empty:
            df = pd.DataFrame(
                [
                    {
                        "Tag": "(none)",
                        "Description": "(none)",
                        "Anomaly Cause": "🟢 Not Likely",
                        "Trigger": "Not Triggered",
                    }
                ]
            )
            return self._style_table(df)

        max_vals = feature_contrib.max()

        rows = []
        for disp, v in max_vals.items():
            meta = features.get(disp, {}) if isinstance(features, dict) else {}
            raw_tag = str(meta.get("tag", disp)) if isinstance(meta, dict) else str(disp)
            desc = str(meta.get("description", disp)) if isinstance(meta, dict) else str(disp)

            is_likely = float(v) > 0.0
            cause = "🔴 Likely Cause" if is_likely else "🟢 Not Likely"

            trigger = self._trigger_text(
                method_name=method_name,
                raw_tag=raw_tag,
                is_likely=is_likely,
                trigger_context=trigger_context,
            )

            rows.append(
                {
                    "Tag": raw_tag,
                    "Description": desc,
                    "Anomaly Cause": cause,
                    "Trigger": trigger,
                }
            )

        df = pd.DataFrame(rows)
        return self._style_table(df)

    # ---------------------------------------------------------------------
    # Plotting helpers (FixedThresholds parity for now)
    # ---------------------------------------------------------------------
    def _score_fixed_thresholds(
        self,
        *,
        df_wide: pd.DataFrame,
        sensor_cfg: Dict[str, Any],
        feature_displaynames_to_tags: Dict[str, str],
    ) -> Tuple[pd.Series, pd.DataFrame]:
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

    def _decorate_section_status(
        self,
        *,
        score: pd.Series,
        section_raw: pd.Series,
        alarm_thresh: float,
        filter_value: float,
        startup_minutes: int,
    ) -> pd.Series:
        idx = score.index
        sec = section_raw.reindex(idx)

        out = pd.Series(index=idx, dtype=float)
        out[:] = -1  # unknown by default

        open_mask = sec >= float(filter_value)
        out[open_mask.fillna(False)] = 1
        out[(~open_mask).fillna(False)] = 0

        anomaly_mask = (score < float(alarm_thresh)) & (out == 1)
        out[anomaly_mask] = 2

        if startup_minutes and int(startup_minutes) > 0:
            prev_open = out.shift(1).isin([1, 2, 3])
            now_open = out.isin([1, 2])
            transitions = (~prev_open.fillna(False)) & (now_open.fillna(False))
            for t in out.index[transitions]:
                end = t + pd.Timedelta(minutes=int(startup_minutes))
                out.loc[(out.index >= t) & (out.index <= end)] = 3

        return out

    def _mask_to_ranges(self, mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        if mask is None or mask.empty:
            return []
        mask = mask.fillna(False).astype(bool)
        ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        start = None
        for ts, on in mask.items():
            if on and start is None:
                start = ts
            elif (not on) and start is not None:
                ranges.append((start, ts))
                start = None
        if start is not None:
            ranges.append((start, mask.index.max()))
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
        sensor = get_sensor(self.ssot, self.sensor_name)
        sensor_cfg = sensor.cfg

        feature_displaynames_to_tags: Dict[str, str] = {}
        for disp, meta in (sensor_cfg.get("Features", {}) or {}).items():
            if isinstance(meta, dict) and meta.get("tag"):
                feature_displaynames_to_tags[disp] = meta["tag"]

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

            # Convert to DISPLAY timezone for plotting (drop tz for matplotlib)
            df_wide = df_wide.copy()
            df_wide.index = pd.to_datetime(df_wide.index, utc=True).tz_convert(self.display_tz).tz_localize(None)
            df_wide = df_wide.sort_index()

            if filter_tag_series is None or len(getattr(filter_tag_series, "index", [])) == 0:
                filter_tag_series = pd.Series(index=df_wide.index, data=np.nan, dtype=float)
            else:
                filter_tag_series = filter_tag_series.copy()
                filter_tag_series.index = (
                    pd.to_datetime(filter_tag_series.index, utc=True)
                    .tz_convert(self.display_tz)
                    .tz_localize(None)
                )
                filter_tag_series = filter_tag_series.sort_index()

            score, _fc = self._score_fixed_thresholds(
                df_wide=df_wide,
                sensor_cfg=sensor_cfg,
                feature_displaynames_to_tags=feature_displaynames_to_tags,
            )

            section_raw = filter_tag_series.reindex(score.index).ffill()
            startup_minutes = int((sensor_cfg.get("Other", {}) or {}).get("startup_period", 0))

            section_status = self._decorate_section_status(
                score=score,
                section_raw=section_raw,
                alarm_thresh=float(alarm_thresh),
                filter_value=float(filter_value),
                startup_minutes=startup_minutes,
            )

            # Figure layout
            n_feat = len(feature_displaynames_to_tags)
            fig_h = 10 + 2 * max(1, n_feat)
            fig, axes = plt.subplots(
                n_feat + 2,
                1,
                figsize=(14, fig_h),
                gridspec_kw={"height_ratios": [0.2, 1] + [1] * n_feat},
            )
            fig.subplots_adjust(hspace=0.8)

            # Section Status
            ax0 = axes[0]
            for ts_i, val in section_status.fillna(-1).items():
                if val in (-1, 0):
                    color = "grey"
                elif val == 2:
                    color = "red"
                elif val == 3:
                    color = "orange"
                else:
                    color = "green"
                ax0.barh(0, 1, left=ts_i, height=1, color=color, align="edge")

            ax0.xaxis_date()
            ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H"))
            ax0.set_xlim(section_status.index.min(), section_status.index.max())
            ax0.set_yticks([])
            ax0.set_xlabel("Date")
            ax0.set_title("Section Status")
            ax0.grid(True)

            handles = [plt.Line2D([0], [0], color=c, lw=4) for c in ["grey", "red", "green", "orange"]]
            labels = ["Section off", "Aomaly", "No Anomaly", "Startrup"]
            ax0.legend(handles, labels, loc="best")

            # Health Score
            ax1 = axes[1]
            score_pct = score * 100.0
            ax1.plot(score_pct.index, score_pct.values, label="Health Score", color="blue")
            ax1.set_xlim(section_status.index.min(), section_status.index.max())
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Health Score %")
            ax1.set_title(f"System Health Score %: {sensor_cfg.get('Model_Description', self.sensor_name)}")
            ax1.grid(True)
            ax1.legend()

            anomaly_ranges = self._mask_to_ranges(section_status == 2)
            for a, b in anomaly_ranges:
                try:
                    if (section_status.loc[a:b] == 3).any():
                        continue
                except Exception:
                    pass
                ax1.axvspan(a, b, color="red", alpha=0.3)

            # Per-feature plots: highlight threshold breaches while section ON
            method_cfg = ((sensor_cfg.get("Method", {}) or {}).get("FixedThresholds", {}) or {})
            high_map = method_cfg.get("high", {}) or {}
            low_map = method_cfg.get("low", {}) or {}

            operating_mask = section_status.isin([1, 2])  # section ON
            filtered_df = df_wide.loc[operating_mask].copy()
            filtered_df = filtered_df.dropna(how="any")

            for i, (disp, tag) in enumerate(feature_displaynames_to_tags.items()):
                ax = axes[i + 2]

                s = pd.to_numeric(df_wide.get(tag, pd.Series(index=df_wide.index, dtype=float)), errors="coerce")
                ax.plot(df_wide.index, s.values, label=disp)

                meta = (sensor_cfg.get("Features", {}) or {}).get(disp, {}) or {}
                unit_desc = str(meta.get("unit_description", ""))
                unit = str(meta.get("unit", ""))
                ax.set_xlim(section_status.index.min(), section_status.index.max())
                ax.set_xlabel("Datetime")
                ax.set_ylabel(f"{unit_desc} :({unit})".strip())
                ax.set_title(str(meta.get("description", disp)))
                ax.grid(True)

                # trendline
                if tag in filtered_df.columns and len(filtered_df.index) >= 4:
                    y = pd.to_numeric(filtered_df[tag], errors="coerce").values
                    x = np.arange(len(filtered_df.index))
                    ok = np.isfinite(y)
                    if ok.sum() >= 4:
                        slope, intercept = np.polyfit(x[ok], y[ok], 1)
                        trend = slope * x + intercept
                        ax.plot(filtered_df.index, trend, label=f"{disp} Trendline", linestyle="--", color="orange")

                # highlight threshold breaches while section ON
                hi = high_map.get(tag, None)
                lo = low_map.get(tag, None)

                if (hi is not None or lo is not None) and not s.empty:
                    outside = pd.Series(False, index=s.index)
                    if hi is not None:
                        outside = outside | (s > float(hi))
                    if lo is not None:
                        outside = outside | (s < float(lo))
                    outside = outside.fillna(False)

                    sec_on = operating_mask.reindex(outside.index).fillna(False)
                    outside = outside & sec_on

                    ranges = self._mask_to_ranges(outside)
                    for a, b in ranges:
                        ax.axvspan(a, b, color="red", alpha=0.2)

                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            if self.global_debug:
                try:
                    plt.show()
                except Exception:
                    pass
            buf.seek(0)
            plt.close(fig)

            return base64.b64encode(buf.read()).decode("utf-8")

        short_b64 = build_plot(short_hours, "Short View")
        long_title = "Past 4 Weeks View" if not self.global_debug else "Long View (Debug)"
        long_b64 = build_plot(long_hours, long_title)
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
        alarm_thresh: float,
        method_name: str,
        trigger_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        model_desc = str(cfg.get("Model_Description", self.sensor_name))

        feature_table_html = self._build_likely_cause_table(
            feature_contrib,
            method_name=method_name,
            trigger_context=trigger_context,
        )

        ts_disp_s, ts_utc_s = self._ts_strings(latest_ts)
        pretty_method = (trigger_context or {}).get("pretty_method") or method_name

        # Detection Mode line BEFORE the table (as requested)
        body = f"""🚨 Alert! 🚨<br><br> 
                    We have detected a sustained significant deviation in the operating conditions of the: {model_desc}. 
                    Please review the system status to restore stability <br><br>                   
                    The table below highlights the importance of each measured detector <br><br>
                    <b>Detection Mode:</b> {pretty_method}<br>
                    {feature_table_html}<br><br>

                    <img src="data:image/png;base64,{plots.short_plot_b64}" alt="Plot" />
                    <br><br>
                        <h3 style="
                            background-color: white;
                            color: #6a11cb;
                            padding: 15px;
                            border-radius: 8px;
                            text-align: center;
                            font-family: Arial, sans-serif;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                            margin: 20px auto;
                            width: 20%;
                        ">
                            Past 4 Weeks View
                        </h3>
                        <br><br>
                    <img src="data:image/png;base64,{plots.long_plot_b64}" alt="Plot" />
                    <br><br>
                    <div style="font-family: Arial, sans-serif; color:#444; font-size:12px;">
                      Timestamp (Display): {ts_disp_s}<br>
                      Timestamp (UTC): {ts_utc_s}
                    </div>
                    <br><br>

                    You can ask us any questions you might have by posting them on the <a href="https://neuromine.atlassian.net/servicedesk/customer/portal/5">Jira Portal</a>."""
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
        section_status: float,
        latest_ts: datetime,
        feature_contrib: Optional[pd.DataFrame],
        reason: str,
        alarm_thresh: Optional[float] = None,
        filter_value: Optional[float] = None,
        method_name: str = "FixedThresholds",
        trigger_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        site = str(self.ssot.get("site", "SITE"))
        model_desc = str(cfg.get("Model_Description", self.sensor_name))

        subject = f"{site} DEBUG TEST EMAIL: {model_desc} model"

        if alarm_thresh is None:
            alarm_thresh = float(((cfg.get("Other", {}) or {}).get("alarm_thresh", 0.75)))
        if filter_value is None:
            filter_value = float(((cfg.get("filter_tag", {}) or {}).get("filter_value", 0.9)))

        plots = self._generate_plots(latest_ts=latest_ts, alarm_thresh=float(alarm_thresh), filter_value=float(filter_value))
        html = self._build_html(
            score=score,
            latest_ts=latest_ts,
            feature_contrib=feature_contrib,
            plots=plots,
            alarm_thresh=float(alarm_thresh),
            method_name=method_name,
            trigger_context=trigger_context,
        )

        self._send_email(subject=subject, html=html, use_high_priority=False)

    def send_alert(
        self,
        *,
        score: float,
        section_status: float,
        latest_ts: datetime,
        feature_contrib: Optional[pd.DataFrame],
        reason: str,
        alarm_thresh: Optional[float] = None,
        filter_value: Optional[float] = None,
        method_name: str = "FixedThresholds",
        trigger_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = self.ssot.get(self.sensor_name, {}) or {}
        site = str(self.ssot.get("site", "SITE"))
        model_desc = str(cfg.get("Model_Description", self.sensor_name))

        subject = f"{site} Anomaly Detected by: {model_desc} model"

        if alarm_thresh is None:
            alarm_thresh = float(((cfg.get("Other", {}) or {}).get("alarm_thresh", 0.75)))
        if filter_value is None:
            filter_value = float(((cfg.get("filter_tag", {}) or {}).get("filter_value", 0.9)))

        plots = self._generate_plots(latest_ts=latest_ts, alarm_thresh=float(alarm_thresh), filter_value=float(filter_value))
        html = self._build_html(
            score=score,
            latest_ts=latest_ts,
            feature_contrib=feature_contrib,
            plots=plots,
            alarm_thresh=float(alarm_thresh),
            method_name=method_name,
            trigger_context=trigger_context,
        )

        self._send_email(subject=subject, html=html, use_high_priority=True)