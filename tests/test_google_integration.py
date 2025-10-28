"""
Comprehensive Tests for Google API Integration (Gmail + Calendar)
=================================================================
Tests all Google client operations with mocks for external API calls.

Test Coverage:
- OAuth2 authentication
- Gmail operations (list, read, send)
- Calendar operations (list calendars, events, create events)
- Permission checking
- Error handling
- Audit logging
"""

import pytest
import time
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, mock_open
from typing import Optional, List, Dict, Any

# Mock Google modules before import
import sys
sys.modules['google'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.credentials'] = MagicMock()
sys.modules['google.auth'] = MagicMock()
sys.modules['google.auth.transport'] = MagicMock()
sys.modules['google.auth.transport.requests'] = MagicMock()
sys.modules['google_auth_oauthlib'] = MagicMock()
sys.modules['google_auth_oauthlib.flow'] = MagicMock()
sys.modules['googleapiclient'] = MagicMock()
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.errors'] = MagicMock()


class MockCredentials:
    """Mock Google OAuth2 credentials"""
    def __init__(self, token: str = "test_token"):
        self.token = token
        self.valid = True
        self.expired = False
        self.refresh_token = "refresh_token"
        self.expiry = datetime.now() + timedelta(days=7)

    def refresh(self, request):
        self.valid = True
        self.expired = False

    def to_json(self):
        return '{"token": "test_token", "refresh_token": "refresh_token"}'

    @staticmethod
    def from_authorized_user_file(path: str, scopes: List[str]):
        return MockCredentials()


class MockGmailService:
    """Mock Gmail API service"""
    def __init__(self):
        self._messages = self._create_messages_api()

    def _create_messages_api(self):
        messages = MagicMock()
        messages.list = self._list_messages
        messages.get = self._get_message
        messages.send = self._send_message
        return messages

    def users(self):
        return self

    def messages(self):
        return self._messages

    def _list_messages(self, userId: str, q: Optional[str] = None, maxResults: int = 10, labelIds: Optional[List[str]] = None):
        execute = MagicMock()
        execute.execute = MagicMock(return_value={
            'messages': [
                {'id': 'msg_001', 'threadId': 'thread_001'},
                {'id': 'msg_002', 'threadId': 'thread_002'},
                {'id': 'msg_003', 'threadId': 'thread_003'},
            ]
        })
        return execute

    def _get_message(self, userId: str, id: str, format: str = 'full', metadataHeaders: Optional[List[str]] = None):
        execute = MagicMock()

        if format == 'metadata':
            execute.execute = MagicMock(return_value={
                'id': id,
                'threadId': f'thread_{id}',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'sender@example.com'},
                        {'name': 'Subject', 'value': 'Test Subject'},
                        {'name': 'Date', 'value': datetime.now().isoformat()},
                    ]
                },
                'snippet': 'This is a test email snippet',
                'labelIds': ['INBOX', 'UNREAD']
            })
        else:  # full format
            execute.execute = MagicMock(return_value={
                'id': id,
                'threadId': f'thread_{id}',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'sender@example.com'},
                        {'name': 'To', 'value': 'recipient@example.com'},
                        {'name': 'Subject', 'value': 'Test Subject'},
                        {'name': 'Date', 'value': datetime.now().isoformat()},
                    ],
                    'body': {
                        'data': base64.urlsafe_b64encode(b'Test email body').decode()
                    }
                },
                'snippet': 'This is a test email snippet',
                'labelIds': ['INBOX']
            })

        return execute

    def _send_message(self, userId: str, body: Dict[str, Any]):
        execute = MagicMock()
        execute.execute = MagicMock(return_value={
            'id': 'sent_msg_123',
            'threadId': 'thread_sent_123',
            'labelIds': ['SENT']
        })
        return execute


class MockCalendarService:
    """Mock Calendar API service"""
    def __init__(self):
        self._calendar_list = self._create_calendar_list_api()
        self._events = self._create_events_api()

    def _create_calendar_list_api(self):
        calendar_list = MagicMock()
        calendar_list.list = self._list_calendars
        return calendar_list

    def _create_events_api(self):
        events = MagicMock()
        events.list = self._list_events
        events.insert = self._insert_event
        return events

    def calendarList(self):
        return self._calendar_list

    def events(self):
        return self._events

    def _list_calendars(self):
        execute = MagicMock()
        execute.execute = MagicMock(return_value={
            'items': [
                {
                    'id': 'primary',
                    'summary': 'Primary Calendar',
                    'description': 'Main calendar',
                    'timeZone': 'America/New_York',
                    'primary': True
                },
                {
                    'id': 'cal_002',
                    'summary': 'Work Calendar',
                    'description': 'Work events',
                    'timeZone': 'America/New_York',
                    'primary': False
                }
            ]
        })
        return execute

    def _list_events(self, calendarId: str, timeMin: Optional[str] = None, timeMax: Optional[str] = None,
                     maxResults: int = 10, singleEvents: bool = True, orderBy: Optional[str] = None):
        execute = MagicMock()
        execute.execute = MagicMock(return_value={
            'items': [
                {
                    'id': 'event_001',
                    'summary': 'Team Meeting',
                    'description': 'Weekly team sync',
                    'start': {'dateTime': datetime.now().isoformat()},
                    'end': {'dateTime': (datetime.now() + timedelta(hours=1)).isoformat()},
                    'location': 'Conference Room A',
                    'attendees': [{'email': 'team@example.com'}],
                    'htmlLink': 'https://calendar.google.com/event?eid=event_001'
                },
                {
                    'id': 'event_002',
                    'summary': 'Project Review',
                    'start': {'dateTime': (datetime.now() + timedelta(days=1)).isoformat()},
                    'end': {'dateTime': (datetime.now() + timedelta(days=1, hours=2)).isoformat()},
                    'htmlLink': 'https://calendar.google.com/event?eid=event_002'
                }
            ]
        })
        return execute

    def _insert_event(self, calendarId: str, body: Dict[str, Any]):
        execute = MagicMock()
        execute.execute = MagicMock(return_value={
            'id': 'new_event_123',
            'summary': body.get('summary'),
            'start': body.get('start'),
            'end': body.get('end'),
            'htmlLink': 'https://calendar.google.com/event?eid=new_event_123'
        })
        return execute


def mock_build(service: str, version: str, credentials):
    """Mock googleapiclient.discovery.build"""
    if service == 'gmail':
        return MockGmailService()
    elif service == 'calendar':
        return MockCalendarService()
    return MagicMock()


# Patch modules
sys.modules['google.oauth2.credentials'].Credentials = MockCredentials
sys.modules['googleapiclient.discovery'].build = mock_build
sys.modules['googleapiclient.errors'].HttpError = Exception


@pytest.fixture
def mock_guardian():
    """Mock Guardian Defense"""
    with patch('integrations.google_client.get_guardian_defense') as mock:
        guardian = MagicMock()
        mock.return_value = guardian
        yield guardian


@pytest.fixture
def mock_policy_guard():
    """Mock Policy Guard"""
    with patch('integrations.google_client.get_policy_guard') as mock:
        policy = MagicMock()
        result = MagicMock()
        result.decision = "allow"
        result.requires_approval = False
        result.reasons = []
        policy.evaluate = MagicMock(return_value=result)
        mock.return_value = policy
        yield policy


@pytest.fixture
def mock_audit():
    """Mock audit logging"""
    with patch('integrations.google_client.append_jsonl') as mock:
        yield mock


@pytest.fixture
def mock_credentials_file(tmp_path):
    """Mock credentials file"""
    creds_path = tmp_path / "google_credentials.json"
    creds_path.write_text('{"client_id": "test", "client_secret": "test"}')
    return str(creds_path)


@pytest.fixture
def mock_token_file(tmp_path):
    """Mock token file"""
    token_path = tmp_path / "google_token.json"
    token_path.write_text('{"token": "test_token", "refresh_token": "refresh"}')
    return str(token_path)


@pytest.fixture
def google_client(mock_guardian, mock_policy_guard, mock_audit, mock_credentials_file, mock_token_file, tmp_path):
    """Create Google client with mocks"""
    from integrations.google_client import GoogleClient

    audit_path = tmp_path / "google_audit.jsonl"

    with patch('os.path.exists', return_value=True):
        client = GoogleClient(
            credentials_path=mock_credentials_file,
            token_path=mock_token_file,
            audit_log_path=str(audit_path)
        )

    return client


# ============================================================================
# Initialization Tests
# ============================================================================

def test_google_client_initialization(google_client):
    """Test Google client initialization"""
    assert google_client is not None
    assert google_client.creds is not None
    assert google_client.creds.valid is True


def test_google_client_scopes(google_client):
    """Test default scopes are read-only"""
    assert any('readonly' in scope for scope in google_client.scopes)


# ============================================================================
# Gmail Tests
# ============================================================================

def test_list_gmail_messages(google_client):
    """Test listing Gmail messages"""
    messages = google_client.list_messages(query="from:sender@example.com", max_results=10)

    assert len(messages) == 3
    assert all('id' in msg for msg in messages)
    assert all('from' in msg for msg in messages)
    assert all('subject' in msg for msg in messages)


def test_list_gmail_messages_with_labels(google_client):
    """Test listing Gmail messages with label filter"""
    messages = google_client.list_messages(label_ids=["INBOX", "UNREAD"])

    assert len(messages) == 3


def test_get_gmail_message(google_client):
    """Test getting Gmail message details"""
    message = google_client.get_message("msg_001")

    assert message['id'] == 'msg_001'
    assert 'from' in message
    assert 'to' in message
    assert 'subject' in message
    assert 'body' in message


def test_send_gmail_message_permission_denied(google_client, mock_policy_guard):
    """Test sending Gmail message without permission"""
    # Mock denial
    result_mock = MagicMock()
    result_mock.decision = "deny"
    result_mock.requires_approval = True
    result_mock.reasons = ["Write permission required"]
    mock_policy_guard.evaluate.return_value = result_mock

    with pytest.raises(PermissionError, match="Write permission required"):
        google_client.send_message(
            to="recipient@example.com",
            subject="Test Email",
            body="Test body"
        )


def test_send_gmail_message_with_permission(google_client, mock_policy_guard):
    """Test sending Gmail message with permission"""
    # Mock approval
    result_mock = MagicMock()
    result_mock.decision = "allow"
    result_mock.requires_approval = False
    mock_policy_guard.evaluate.return_value = result_mock

    # Patch _require_write_permission to return True
    with patch.object(google_client, '_require_write_permission', return_value=True):
        result = google_client.send_message(
            to="recipient@example.com",
            subject="Test Email",
            body="Test body"
        )

        assert 'id' in result
        assert result['to'] == 'recipient@example.com'
        assert result['subject'] == 'Test Email'


def test_send_gmail_message_with_cc_bcc(google_client, mock_policy_guard):
    """Test sending Gmail message with CC and BCC"""
    with patch.object(google_client, '_require_write_permission', return_value=True):
        result = google_client.send_message(
            to="recipient@example.com",
            subject="Test Email",
            body="Test body",
            cc="cc@example.com",
            bcc="bcc@example.com"
        )

        assert 'id' in result


# ============================================================================
# Calendar Tests
# ============================================================================

def test_list_calendars(google_client):
    """Test listing calendars"""
    calendars = google_client.list_calendars()

    assert len(calendars) == 2
    assert all('id' in cal for cal in calendars)
    assert all('summary' in cal for cal in calendars)
    assert calendars[0]['primary'] is True


def test_list_calendar_events(google_client):
    """Test listing calendar events"""
    events = google_client.list_events(
        calendar_id='primary',
        max_results=10
    )

    assert len(events) == 2
    assert all('id' in event for event in events)
    assert all('summary' in event for event in events)
    assert all('start' in event for event in events)


def test_list_calendar_events_with_time_range(google_client):
    """Test listing calendar events with time range"""
    time_min = datetime.now()
    time_max = datetime.now() + timedelta(days=7)

    events = google_client.list_events(
        calendar_id='primary',
        time_min=time_min,
        time_max=time_max
    )

    assert isinstance(events, list)


def test_create_calendar_event_permission_denied(google_client, mock_policy_guard):
    """Test creating calendar event without permission"""
    result_mock = MagicMock()
    result_mock.decision = "deny"
    result_mock.requires_approval = True
    mock_policy_guard.evaluate.return_value = result_mock

    with pytest.raises(PermissionError, match="Write permission required"):
        google_client.create_event(
            summary="Team Meeting",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )


def test_create_calendar_event_with_permission(google_client, mock_policy_guard):
    """Test creating calendar event with permission"""
    with patch.object(google_client, '_require_write_permission', return_value=True):
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)

        result = google_client.create_event(
            summary="Team Meeting",
            start_time=start_time,
            end_time=end_time,
            description="Weekly sync",
            location="Conference Room",
            attendees=["team@example.com"]
        )

        assert 'id' in result
        assert result['summary'] == "Team Meeting"
        assert 'html_link' in result


def test_create_calendar_event_without_optional_fields(google_client):
    """Test creating calendar event without optional fields"""
    with patch.object(google_client, '_require_write_permission', return_value=True):
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)

        result = google_client.create_event(
            summary="Simple Event",
            start_time=start_time,
            end_time=end_time
        )

        assert 'id' in result


# ============================================================================
# Authentication Tests
# ============================================================================

def test_get_auth_status(google_client):
    """Test getting authentication status"""
    status = google_client.get_auth_status()

    assert status['authenticated'] is True
    assert 'scopes' in status
    assert 'has_write_scopes' in status
    assert 'token_path' in status


def test_credentials_refresh(google_client):
    """Test credentials refresh"""
    # Make credentials expired
    google_client.creds.expired = True
    google_client.creds.valid = False

    # Refresh should work
    google_client.creds.refresh(None)

    assert google_client.creds.valid is True
    assert google_client.creds.expired is False


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_list_messages_api_error(google_client):
    """Test handling API errors when listing messages"""
    with patch.object(google_client, '_get_gmail_service', side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="API Error"):
            google_client.list_messages()


def test_get_message_not_found(google_client):
    """Test handling message not found"""
    mock_service = google_client._get_gmail_service()

    with patch.object(mock_service.users().messages(), 'get', side_effect=Exception("Not Found")):
        with pytest.raises(Exception):
            google_client.get_message("nonexistent_id")


def test_create_event_api_error(google_client):
    """Test handling API errors when creating event"""
    with patch.object(google_client, '_require_write_permission', return_value=True):
        with patch.object(google_client, '_get_calendar_service', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                google_client.create_event(
                    summary="Test Event",
                    start_time=datetime.now(),
                    end_time=datetime.now() + timedelta(hours=1)
                )


# ============================================================================
# Audit Logging Tests
# ============================================================================

def test_audit_logging_gmail_read(google_client, mock_audit):
    """Test audit logging for Gmail read operations"""
    google_client.list_messages()

    assert mock_audit.called
    call_args = mock_audit.call_args[0]
    log_entry = call_args[1]

    assert log_entry['integration'] == 'google'
    assert log_entry['service'] == 'gmail'
    assert log_entry['operation_type'] == 'read'
    assert log_entry['success'] is True


def test_audit_logging_calendar_read(google_client, mock_audit):
    """Test audit logging for Calendar read operations"""
    google_client.list_calendars()

    assert mock_audit.called
    call_args = mock_audit.call_args[0]
    log_entry = call_args[1]

    assert log_entry['integration'] == 'google'
    assert log_entry['service'] == 'calendar'
    assert log_entry['operation_type'] == 'read'


def test_audit_logging_write_operation(google_client, mock_audit):
    """Test audit logging for write operations"""
    with patch.object(google_client, '_require_write_permission', return_value=True):
        google_client.create_event(
            summary="Test Event",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )

    assert mock_audit.called
    call_args = mock_audit.call_args[0]
    log_entry = call_args[1]

    assert log_entry['operation_type'] == 'write'
    assert log_entry['action'] == 'create_event'


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_gmail_workflow(google_client):
    """Test complete Gmail workflow"""
    # List messages
    messages = google_client.list_messages()
    assert len(messages) > 0

    # Get specific message
    message = google_client.get_message(messages[0]['id'])
    assert message['id'] == messages[0]['id']


def test_full_calendar_workflow(google_client):
    """Test complete Calendar workflow"""
    # List calendars
    calendars = google_client.list_calendars()
    assert len(calendars) > 0

    # List events
    events = google_client.list_events(calendar_id='primary')
    assert isinstance(events, list)


# ============================================================================
# Performance Tests
# ============================================================================

def test_multiple_operations_performance(google_client):
    """Test performance with multiple operations"""
    start = time.time()

    for i in range(5):
        google_client.list_messages(max_results=10)
        google_client.list_calendars()

    duration = time.time() - start
    assert duration < 5.0
