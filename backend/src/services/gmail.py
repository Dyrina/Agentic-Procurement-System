"""
services/gmail.py — Gmail API wrapper for sending RFQs and fetching replies.

Uses the Google API Python Client with OAuth2 credentials to:
  • Send RFQ (Request for Quotation) emails to suppliers.
  • Poll for and retrieve reply messages from supplier threads.

Credentials are loaded from file paths specified in ``core/config.py``.
"""

from __future__ import annotations

import base64
import os
from email.mime.text import MIMEText
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
]


def get_gmail_service(credentials_path: str, token_path: str):
    """Build and return an authenticated Gmail API service object."""
    creds: Credentials | None = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def send_email(service, to: str, subject: str, body_html: str) -> str:
    """Send an email. Returns the Gmail message ID."""
    msg = MIMEText(body_html, "html")
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return result["id"]


def fetch_replies(
    service,
    from_emails: list[str],
    since_timestamp: str,
) -> list[dict[str, Any]]:
    """
    Fetch reply emails from each address in from_emails sent after since_timestamp.

    since_timestamp: ISO 8601 string e.g. '2026-07-01T10:00:00Z'
    Returns list of:
        {
            "from": str,
            "subject": str,
            "body_text": str,
            "attachments": [{"filename": str, "data": bytes}]
        }
    """
    import email as email_lib
    from datetime import datetime, timezone

    # Convert ISO timestamp to Unix epoch for Gmail query
    dt = datetime.fromisoformat(since_timestamp.replace("Z", "+00:00"))
    epoch = int(dt.timestamp())

    replies = []
    for sender_email in from_emails:
        query = f"from:{sender_email} after:{epoch}"
        try:
            result = service.users().messages().list(userId="me", q=query).execute()
        except HttpError:
            continue

        messages = result.get("messages", [])
        for msg_ref in messages:
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="full"
            ).execute()

            headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
            body_text = _extract_body(msg["payload"])
            attachments = _extract_attachments(service, msg)

            replies.append({
                "from": headers.get("From", sender_email),
                "subject": headers.get("Subject", ""),
                "body_text": body_text,
                "attachments": attachments,
            })

    return replies


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text body from Gmail message payload."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""


def _extract_attachments(service, msg: dict) -> list[dict[str, Any]]:
    """Download all PDF attachments from a Gmail message."""
    attachments = []
    payload = msg["payload"]

    for part in payload.get("parts", []):
        filename = part.get("filename", "")
        if not filename.lower().endswith(".pdf"):
            continue
        attachment_id = part["body"].get("attachmentId")
        if not attachment_id:
            continue
        att = service.users().messages().attachments().get(
            userId="me", messageId=msg["id"], id=attachment_id
        ).execute()
        data = base64.urlsafe_b64decode(att["data"] + "==")
        attachments.append({"filename": filename, "data": data})

    return attachments