# APM/LIO/src/alerts/notifier.py
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd


class Notifier:
    """
    Uses:
      - Email_DEV when global_debug=True
      - Email when global_debug=False
    Sensor debug sends test email every run.
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

    def _build_html(self, *, title: str, ts: datetime, score: float, section_status: float, reason: str, feature_contrib: Optional[pd.DataFrame]):
        rows_html = ""
        if feature_contrib is not None and not feature_contrib.empty and ts in feature_contrib.index:
            r = feature_contrib.loc[ts]
            try:
                top = r.sort_values(ascending=False).head(10)
                for k, v in top.items():
                    rows_html += f"<tr><td>{k}</td><td>{float(v):.3f}</td></tr>"
            except Exception:
                pass

        if not rows_html:
            rows_html = "<tr><td>(none)</td><td>0.000</td></tr>"

        return f"""
        <html>
          <body>
            <h2>{title}</h2>
            <ul>
              <li><b>Timestamp:</b> {ts.isoformat()}</li>
              <li><b>Score:</b> {score:.3f}</li>
              <li><b>Section status:</b> {section_status:.3f}</li>
              <li><b>Reason:</b> {reason}</li>
            </ul>

            <h3>Top Contributors</h3>
            <table border="1" cellpadding="6" cellspacing="0">
              <tr><th>Feature</th><th>Contribution</th></tr>
              {rows_html}
            </table>
          </body>
        </html>
        """

    def _send_email(self, *, subject: str, html: str):
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
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.from_address, self.password)
            server.sendmail(self.from_address, self.to_addresses, msg.as_string())

        self.logger.info(f"{self.sensor_name}: email sent -> {', '.join(self.to_addresses)}")

    def send_test_email(self, *, score: float, section_status: float, latest_ts: datetime, feature_contrib: Optional[pd.DataFrame], reason: str):
        subject = f"{self.ssot.get('site','SITE')} | DEBUG TEST EMAIL: {self.sensor_name}"
        html = self._build_html(
            title="DEBUG TEST EMAIL",
            ts=latest_ts,
            score=score,
            section_status=section_status,
            reason=reason,
            feature_contrib=feature_contrib,
        )
        self._send_email(subject=subject, html=html)

    def send_alert(self, *, score: float, section_status: float, latest_ts: datetime, feature_contrib: Optional[pd.DataFrame], reason: str):
        subject = f"{self.ssot.get('site','SITE')} | ALERT: {self.sensor_name}"
        html = self._build_html(
            title="ALERT TRIGGERED",
            ts=latest_ts,
            score=score,
            section_status=section_status,
            reason=reason,
            feature_contrib=feature_contrib,
        )
        self._send_email(subject=subject, html=html)