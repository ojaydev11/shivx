"""
Google API Integration (Gmail + Calendar)
==========================================
Safe Google API integration with OAuth2, scoped permissions, and approval workflows.

Features:
- OAuth2 authentication with user consent
- Gmail operations (read, send with approval)
- Calendar operations (read, create with approval)
- Scoped permissions (read-only by default)
- Write operations require EXECUTE permission
- Full audit logging
"""

import os
import time
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    Credentials = None  # type: ignore

from security.guardian_defense import get_guardian_defense
from utils.audit_chain import append_jsonl
from utils.policy_guard import get_policy_guard

logger = logging.getLogger(__name__)


class GoogleServiceType(Enum):
    """Google service types"""
    GMAIL = "gmail"
    CALENDAR = "calendar"


class GoogleOperationType(Enum):
    """Google operation types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


# Default scopes (read-only)
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
]

# Write scopes (require explicit approval)
WRITE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
]


class GoogleClient:
    """
    Safe Google API client (Gmail + Calendar) with OAuth2 and approval workflows.

    Features:
    - OAuth2 authentication with user consent
    - Read operations: No approval needed
    - Write operations: Require EXECUTE permission
    - Scoped permissions: Read-only by default
    - Audit logging: All operations logged
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        token_path: str = "var/tokens/google_token.json",
        audit_log_path: str = "var/audit/google_operations.jsonl",
        scopes: Optional[List[str]] = None
    ):
        """
        Initialize Google client.

        Args:
            credentials_path: Path to OAuth2 credentials JSON
            token_path: Path to store OAuth2 token
            audit_log_path: Path to audit log file
            scopes: OAuth2 scopes (uses DEFAULT_SCOPES if not provided)
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google API libraries not installed. "
                "Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH", "config/google_credentials.json")
        self.token_path = token_path
        self.audit_log_path = audit_log_path
        self.scopes = scopes or DEFAULT_SCOPES.copy()

        # Ensure token directory exists
        Path(token_path).parent.mkdir(parents=True, exist_ok=True)

        # Get integrations
        self.guardian = get_guardian_defense()
        self.policy_guard = get_policy_guard()

        # Authenticate
        self.creds = self._authenticate()

        # Build services
        self.gmail_service = None
        self.calendar_service = None

        logger.info(f"GoogleClient initialized with scopes: {self.scopes}")

    def _authenticate(self) -> Credentials:
        """Authenticate with OAuth2"""
        creds = None

        # Load existing token
        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
            except Exception as e:
                logger.warning(f"Failed to load token: {e}")

        # Refresh or create new token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Failed to refresh token: {e}")
                    creds = None

            if not creds:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Google credentials not found at {self.credentials_path}. "
                        "Download from Google Cloud Console."
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes
                )
                creds = flow.run_local_server(port=0)

            # Save token
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        return creds

    def _get_gmail_service(self):
        """Get or create Gmail service"""
        if not self.gmail_service:
            self.gmail_service = build('gmail', 'v1', credentials=self.creds)
        return self.gmail_service

    def _get_calendar_service(self):
        """Get or create Calendar service"""
        if not self.calendar_service:
            self.calendar_service = build('calendar', 'v3', credentials=self.creds)
        return self.calendar_service

    def _log_operation(
        self,
        service: GoogleServiceType,
        operation_type: GoogleOperationType,
        action: str,
        details: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log operation to audit chain"""
        try:
            operation_id = f"google_{int(time.time() * 1000)}"

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "integration": "google",
                "service": service.value,
                "operation_id": operation_id,
                "operation_type": operation_type.value,
                "action": action,
                "details": details,
                "success": success,
                "error": error,
            }

            append_jsonl(self.audit_log_path, log_entry)

        except Exception as e:
            logger.error(f"Failed to log operation: {e}")

    def _require_write_permission(self, action: str, details: Dict[str, Any]) -> bool:
        """Check if write operation is allowed"""
        # Check policy
        policy_result = self.policy_guard.evaluate({
            "action": "google.write",
            "operation": action,
            "details": details
        })

        if policy_result.decision == "deny":
            logger.error(
                f"Operation denied by policy: {', '.join(policy_result.reasons)}"
            )
            return False

        if policy_result.requires_approval:
            logger.warning(
                f"Operation requires approval: {action}\n"
                f"Details: {details}"
            )
            # In production, this would prompt user
            # For now, we auto-deny write operations without explicit permission
            return False

        return True

    # ========================================================================
    # Gmail Operations
    # ========================================================================

    def list_messages(
        self,
        query: Optional[str] = None,
        max_results: int = 10,
        label_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List Gmail messages.

        Args:
            query: Gmail search query (e.g., "from:user@example.com")
            max_results: Maximum number of messages to return
            label_ids: Filter by label IDs (e.g., ["INBOX", "UNREAD"])

        Returns:
            List of messages
        """
        try:
            service = self._get_gmail_service()

            results = service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results,
                labelIds=label_ids
            ).execute()

            messages = results.get('messages', [])

            result = []
            for msg in messages:
                msg_detail = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['From', 'Subject', 'Date']
                ).execute()

                headers = {h['name']: h['value'] for h in msg_detail.get('payload', {}).get('headers', [])}

                result.append({
                    "id": msg_detail['id'],
                    "thread_id": msg_detail['threadId'],
                    "from": headers.get('From'),
                    "subject": headers.get('Subject'),
                    "date": headers.get('Date'),
                    "snippet": msg_detail.get('snippet'),
                    "label_ids": msg_detail.get('labelIds', [])
                })

            self._log_operation(
                GoogleServiceType.GMAIL,
                GoogleOperationType.READ,
                "list_messages",
                {"query": query, "count": len(result)},
                True
            )

            return result

        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            self._log_operation(
                GoogleServiceType.GMAIL,
                GoogleOperationType.READ,
                "list_messages",
                {"query": query},
                False,
                str(e)
            )
            raise

    def get_message(self, message_id: str) -> Dict[str, Any]:
        """
        Get full message details.

        Args:
            message_id: Gmail message ID

        Returns:
            Message details
        """
        try:
            service = self._get_gmail_service()

            message = service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()

            headers = {h['name']: h['value'] for h in message.get('payload', {}).get('headers', [])}

            # Extract body
            body = ""
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        body = part.get('body', {}).get('data', '')
                        break
            else:
                body = message['payload'].get('body', {}).get('data', '')

            result = {
                "id": message['id'],
                "thread_id": message['threadId'],
                "from": headers.get('From'),
                "to": headers.get('To'),
                "subject": headers.get('Subject'),
                "date": headers.get('Date'),
                "body": body,
                "label_ids": message.get('labelIds', []),
                "snippet": message.get('snippet')
            }

            self._log_operation(
                GoogleServiceType.GMAIL,
                GoogleOperationType.READ,
                "get_message",
                {"message_id": message_id},
                True
            )

            return result

        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            self._log_operation(
                GoogleServiceType.GMAIL,
                GoogleOperationType.READ,
                "get_message",
                {"message_id": message_id},
                False,
                str(e)
            )
            raise

    def send_message(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        bcc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send email message.
        Requires write permission and approval.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (plain text)
            cc: CC recipients
            bcc: BCC recipients

        Returns:
            Sent message info
        """
        if not self._require_write_permission("send_message", {"to": to, "subject": subject}):
            raise PermissionError("Write permission required for sending emails")

        try:
            service = self._get_gmail_service()

            # Create message
            from email.mime.text import MIMEText
            import base64

            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            if cc:
                message['cc'] = cc
            if bcc:
                message['bcc'] = bcc

            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

            sent = service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()

            result = {
                "id": sent['id'],
                "thread_id": sent['threadId'],
                "to": to,
                "subject": subject,
            }

            self._log_operation(
                GoogleServiceType.GMAIL,
                GoogleOperationType.WRITE,
                "send_message",
                {"to": to, "subject": subject, "message_id": sent['id']},
                True
            )

            return result

        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            self._log_operation(
                GoogleServiceType.GMAIL,
                GoogleOperationType.WRITE,
                "send_message",
                {"to": to, "subject": subject},
                False,
                str(e)
            )
            raise

    # ========================================================================
    # Calendar Operations
    # ========================================================================

    def list_calendars(self) -> List[Dict[str, Any]]:
        """List all calendars"""
        try:
            service = self._get_calendar_service()

            calendar_list = service.calendarList().list().execute()

            result = [
                {
                    "id": cal['id'],
                    "summary": cal.get('summary'),
                    "description": cal.get('description'),
                    "time_zone": cal.get('timeZone'),
                    "primary": cal.get('primary', False),
                }
                for cal in calendar_list.get('items', [])
            ]

            self._log_operation(
                GoogleServiceType.CALENDAR,
                GoogleOperationType.READ,
                "list_calendars",
                {"count": len(result)},
                True
            )

            return result

        except HttpError as e:
            logger.error(f"Calendar API error: {e}")
            self._log_operation(
                GoogleServiceType.CALENDAR,
                GoogleOperationType.READ,
                "list_calendars",
                {},
                False,
                str(e)
            )
            raise

    def list_events(
        self,
        calendar_id: str = 'primary',
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List calendar events.

        Args:
            calendar_id: Calendar ID (use 'primary' for primary calendar)
            time_min: Minimum event time (defaults to now)
            time_max: Maximum event time
            max_results: Maximum number of events

        Returns:
            List of events
        """
        try:
            service = self._get_calendar_service()

            if not time_min:
                time_min = datetime.utcnow()

            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min.isoformat() + 'Z',
                timeMax=time_max.isoformat() + 'Z' if time_max else None,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            result = [
                {
                    "id": event['id'],
                    "summary": event.get('summary'),
                    "description": event.get('description'),
                    "start": event.get('start', {}).get('dateTime') or event.get('start', {}).get('date'),
                    "end": event.get('end', {}).get('dateTime') or event.get('end', {}).get('date'),
                    "location": event.get('location'),
                    "attendees": [a.get('email') for a in event.get('attendees', [])],
                    "html_link": event.get('htmlLink'),
                }
                for event in events
            ]

            self._log_operation(
                GoogleServiceType.CALENDAR,
                GoogleOperationType.READ,
                "list_events",
                {"calendar_id": calendar_id, "count": len(result)},
                True
            )

            return result

        except HttpError as e:
            logger.error(f"Calendar API error: {e}")
            self._log_operation(
                GoogleServiceType.CALENDAR,
                GoogleOperationType.READ,
                "list_events",
                {"calendar_id": calendar_id},
                False,
                str(e)
            )
            raise

    def create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        calendar_id: str = 'primary'
    ) -> Dict[str, Any]:
        """
        Create calendar event.
        Requires write permission and approval.

        Args:
            summary: Event title
            start_time: Event start time
            end_time: Event end time
            description: Event description
            location: Event location
            attendees: List of attendee email addresses
            calendar_id: Calendar ID

        Returns:
            Created event info
        """
        if not self._require_write_permission("create_event", {"summary": summary}):
            raise PermissionError("Write permission required for creating calendar events")

        try:
            service = self._get_calendar_service()

            event = {
                'summary': summary,
                'description': description,
                'location': location,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
            }

            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]

            created = service.events().insert(
                calendarId=calendar_id,
                body=event
            ).execute()

            result = {
                "id": created['id'],
                "summary": created.get('summary'),
                "start": created.get('start', {}).get('dateTime'),
                "end": created.get('end', {}).get('dateTime'),
                "html_link": created.get('htmlLink'),
            }

            self._log_operation(
                GoogleServiceType.CALENDAR,
                GoogleOperationType.WRITE,
                "create_event",
                {"summary": summary, "event_id": created['id']},
                True
            )

            return result

        except HttpError as e:
            logger.error(f"Calendar API error: {e}")
            self._log_operation(
                GoogleServiceType.CALENDAR,
                GoogleOperationType.WRITE,
                "create_event",
                {"summary": summary},
                False,
                str(e)
            )
            raise

    def get_auth_status(self) -> Dict[str, Any]:
        """Get authentication status"""
        return {
            "authenticated": self.creds is not None and self.creds.valid,
            "scopes": self.scopes,
            "has_write_scopes": any(scope in self.scopes for scope in WRITE_SCOPES),
            "token_path": self.token_path,
            "expires_at": self.creds.expiry.isoformat() if self.creds and self.creds.expiry else None,
        }
