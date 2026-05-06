import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str) -> None:
    """
    Send a plain-text email via SMTP to the configured recipient.

    Uses SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, and NOTIFY_EMAIL from
    config.py. Silently no-ops when SMTP_USER or NOTIFY_EMAIL are empty so the
    module is always safe to call in environments without credentials configured.

    Args:
        subject: Email subject line.
        body: Plain-text email body.
    """
    if not config.SMTP_USER or not config.NOTIFY_EMAIL:
        return
    msg = MIMEMultipart()
    msg["From"] = config.SMTP_USER
    msg["To"] = config.NOTIFY_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASS)
            server.send_message(msg)
        logger.debug("send_email: sent '%s' to %s", subject, config.NOTIFY_EMAIL)
    except Exception as exc:
        logger.warning("send_email: failed to send '%s' — %s", subject, exc)


class EmailErrorHandler(logging.Handler):
    """
    Logging handler that sends an email for every ERROR or CRITICAL record.

    Install on the root logger in main.py after configuring email credentials.
    Uses send_email() internally, so it silently no-ops when email is not
    configured — safe to always install.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Send an email containing this log record's formatted message.

        Args:
            record: The log record emitted at ERROR or CRITICAL level.
        """
        try:
            subject = f"[TradingBot ERROR] {record.name}: {record.getMessage()[:80]}"
            body = self.format(record)
            send_email(subject, body)
        except Exception:
            self.handleError(record)
