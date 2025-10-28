# ShivX External Integrations

## Overview

ShivX provides secure, production-ready integrations with external services. All integrations implement comprehensive safety controls including:

- **Guardian Defense Integration**: All operations monitored for threats
- **Policy Guard Enforcement**: Permission checks before operations
- **Audit Logging**: Complete audit trail for compliance
- **Rate Limiting**: Prevents API abuse
- **DLP Scanning**: Prevents data leakage
- **Approval Workflows**: Write operations require explicit approval

## Available Integrations

### 1. GitHub Integration

**File**: `integrations/github_client.py`

#### Features
- List repositories (user, organization)
- Read issues, pull requests
- Create issues, PRs (with approval)
- Add comments (with approval)
- Rate limiting (5000 req/hour)
- Dry-run mode for testing

#### Setup

1. **Generate GitHub Personal Access Token**
   - Go to: https://github.com/settings/tokens
   - Generate new token with scopes: `repo`, `read:user`, `read:org`
   - Copy token

2. **Configure Environment**
   ```bash
   # .env
   GITHUB_ACCESS_TOKEN=ghp_your_token_here
   ```

3. **Initialize Client**
   ```python
   from integrations.github_client import GitHubClient

   # Read-only operations (no approval needed)
   client = GitHubClient()
   repos = client.list_repositories()
   issues = client.list_issues("owner/repo")

   # Write operations (require approval)
   client.create_issue(
       repo_full_name="owner/repo",
       title="Bug Report",
       body="Found a bug...",
       labels=["bug"]
   )
   ```

#### API Reference

**Read Operations** (No Approval Required)
```python
# List repositories
repos = client.list_repositories(user="octocat", org="github")

# Get repository details
repo = client.get_repository("owner/repo")

# List issues
issues = client.list_issues("owner/repo", state="open", labels=["bug"])

# Get issue details
issue = client.get_issue("owner/repo", 123)

# Check rate limit
status = client.get_rate_limit_status()
```

**Write Operations** (Require Approval)
```python
# Create issue
issue = client.create_issue(
    repo_full_name="owner/repo",
    title="New Issue",
    body="Description",
    labels=["enhancement"]
)

# Create pull request
pr = client.create_pull_request(
    repo_full_name="owner/repo",
    title="New Feature",
    head="feature-branch",
    base="main",
    body="PR description"
)

# Add comment
comment = client.add_comment(
    repo_full_name="owner/repo",
    issue_number=123,
    comment="LGTM!"
)
```

**Dry-Run Mode**
```python
# Test operations without executing
client = GitHubClient(dry_run=True)
result = client.create_issue(...)  # Returns preview, doesn't create
```

---

### 2. Google API Integration (Gmail + Calendar)

**File**: `integrations/google_client.py`

#### Features
- OAuth2 authentication with user consent
- Gmail: Read emails, send (with approval)
- Calendar: List events, create (with approval)
- Scoped permissions (read-only by default)
- Token refresh

#### Setup

1. **Create Google Cloud Project**
   - Go to: https://console.cloud.google.com/
   - Create new project
   - Enable Gmail API and Calendar API

2. **Create OAuth2 Credentials**
   - Go to: APIs & Services > Credentials
   - Create OAuth 2.0 Client ID
   - Application type: Desktop app
   - Download JSON as `config/google_credentials.json`

3. **Configure Environment**
   ```bash
   # .env
   GOOGLE_CREDENTIALS_PATH=config/google_credentials.json
   ```

4. **First-Time Authentication**
   ```python
   from integrations.google_client import GoogleClient

   # First run will open browser for OAuth consent
   client = GoogleClient()
   # Approve access, token saved to var/tokens/google_token.json
   ```

#### API Reference

**Gmail Operations**
```python
# List messages
messages = client.list_messages(
    query="from:sender@example.com is:unread",
    max_results=10,
    label_ids=["INBOX"]
)

# Get message details
message = client.get_message("message_id_123")

# Send email (requires approval)
result = client.send_message(
    to="recipient@example.com",
    subject="Hello",
    body="Email body",
    cc="cc@example.com"
)
```

**Calendar Operations**
```python
# List calendars
calendars = client.list_calendars()

# List events
events = client.list_events(
    calendar_id="primary",
    time_min=datetime.now(),
    time_max=datetime.now() + timedelta(days=7)
)

# Create event (requires approval)
event = client.create_event(
    summary="Team Meeting",
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(hours=1),
    description="Weekly sync",
    location="Conference Room A",
    attendees=["team@example.com"]
)
```

**Authentication Status**
```python
status = client.get_auth_status()
# Returns:
# {
#   'authenticated': True,
#   'scopes': [...],
#   'has_write_scopes': False,
#   'token_path': '...',
#   'expires_at': '...'
# }
```

---

### 3. Telegram Bot

**File**: `integrations/telegram_bot.py`

#### Features
- Command handlers (/start, /help, /status, /agents, /task)
- Message handling (text, voice, documents)
- Notifications and alerts
- User whitelist for security
- Rate limiting (20 msg/min per user)

#### Setup

1. **Create Telegram Bot**
   - Open Telegram, search for @BotFather
   - Send: `/newbot`
   - Follow instructions, get token

2. **Configure Environment**
   ```bash
   # .env
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_ALLOWED_USER_IDS=123456789,987654321  # Your user IDs
   ```

3. **Get Your User ID**
   - Open Telegram, search for @userinfobot
   - Send: `/start`
   - Copy your user ID

4. **Start Bot**
   ```python
   from integrations.telegram_bot import TelegramBot

   bot = TelegramBot()
   bot.start()  # Starts polling for messages
   ```

#### API Reference

**Bot Commands**
```
/start   - Initialize bot
/help    - Show help message
/status  - Show system status
/agents  - List active agents
/task    - Create new task
/cancel  - Cancel current task
```

**Sending Notifications**
```python
# Send to specific user
await bot.send_notification(
    user_id=123456789,
    message="Trade executed: BUY SOL/USDT @ $100"
)

# Send alert to all users
count = await bot.send_alert(
    message="High volatility detected!",
    user_ids=None  # None = all allowed users
)
```

**Bot Status**
```python
status = bot.get_status()
# Returns:
# {
#   'running': True,
#   'webhook_mode': False,
#   'allowed_users': 2,
#   'active_tasks': 1,
#   'rate_limit_per_minute': 20
# }
```

---

### 4. Browser Automation

**File**: `integrations/browser_automation.py`

#### Features
- Headless browser automation (Chromium, Firefox, WebKit)
- URL allowlist enforcement
- Navigate, extract text, screenshot
- Form filling, clicking
- Resource limits and timeouts

#### Setup

1. **Install Playwright Browsers**
   ```bash
   playwright install chromium
   ```

2. **Configure URL Allowlist**
   ```bash
   # .env
   BROWSER_URL_ALLOWLIST=example.com,*.github.com,docs.python.org
   ```

3. **Initialize Client**
   ```python
   from integrations.browser_automation import BrowserAutomation, BrowserType

   browser = BrowserAutomation(
       browser_type=BrowserType.CHROMIUM,
       headless=True,
       url_allowlist=["example.com", "*.github.com"]
   )
   ```

#### API Reference

**Session Management**
```python
# Create session
session_id = await browser.create_session()

# Close session
await browser.close_session(session_id)

# Cleanup all
await browser.cleanup()
```

**Navigation and Extraction**
```python
# Navigate to URL
result = await browser.navigate(
    session_id=session_id,
    url="https://example.com",
    timeout=30000  # 30 seconds
)

# Extract text
text = await browser.extract_text(
    session_id=session_id,
    url="https://example.com",
    selector="article.main"  # Optional CSS selector
)

# Take screenshot
screenshot = await browser.take_screenshot(
    session_id=session_id,
    url="https://example.com",
    output_path="var/screenshots/example.png",
    full_page=True
)
```

**Form Interaction**
```python
# Fill form
result = await browser.fill_form(
    session_id=session_id,
    url="https://example.com/form",
    form_data={
        "#username": "testuser",
        "#password": "testpass"
    }
)

# Click element
result = await browser.click_element(
    session_id=session_id,
    url="https://example.com",
    selector="button.submit"
)
```

---

### 5. LLM Client Bridges

**File**: `integrations/llm_client.py`

#### Features
- Unified interface for Claude and ChatGPT
- Prompt injection filtering
- DLP scanning of responses
- Cost tracking and limits
- Token usage monitoring

#### Setup

1. **Get API Keys**
   - Anthropic: https://console.anthropic.com/
   - OpenAI: https://platform.openai.com/api-keys

2. **Configure Environment**
   ```bash
   # .env
   ANTHROPIC_API_KEY=sk-ant-api03-...
   OPENAI_API_KEY=sk-...
   ```

3. **Initialize Client**
   ```python
   from integrations.llm_client import LLMClient, LLMProvider

   client = LLMClient(
       enable_safety_checks=True,
       max_tokens_per_day=1000000,
       max_cost_per_day=100.0
   )
   ```

#### API Reference

**Completions**
```python
# Claude completion
response = await client.complete(
    prompt="Explain quantum computing in simple terms",
    provider=LLMProvider.CLAUDE,
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    temperature=0.7
)

# ChatGPT completion
response = await client.complete(
    prompt="Write a Python function to sort a list",
    provider=LLMProvider.CHATGPT,
    model="gpt-4",
    max_tokens=512
)

# Response structure
{
    'provider': 'claude',
    'model': 'claude-3-sonnet-20240229',
    'content': 'Quantum computing is...',
    'usage': {
        'prompt_tokens': 15,
        'completion_tokens': 250,
        'total_tokens': 265,
        'estimated_cost': 0.001325
    },
    'finish_reason': 'end_turn',
    'safe': True,
    'safety_issues': []
}
```

**Usage Tracking**
```python
stats = client.get_usage_stats()
# Returns:
# {
#   'date': '2024-01-15',
#   'providers': {
#     'claude': {
#       'tokens_used': 50000,
#       'tokens_limit': 1000000,
#       'cost_used': 2.5,
#       'cost_limit': 100.0
#     },
#     'openai': { ... }
#   },
#   'total_tokens_used': 75000,
#   'total_cost_used': 4.25
# }
```

**Available Providers**
```python
providers = client.get_available_providers()
# Returns: ['claude', 'openai'] (based on configured API keys)
```

---

## Security Features

### 1. Guardian Defense Integration

All integrations route through Guardian Defense for:
- Threat detection
- Rate limit monitoring
- Resource abuse detection

### 2. Policy Guard Enforcement

Write operations checked against:
- User permissions
- Operation type
- Resource limits
- Business rules

### 3. Audit Logging

Complete audit trail:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "integration": "github",
  "operation_id": "gh_1705315800000",
  "operation_type": "write",
  "action": "create_issue",
  "repository": "owner/repo",
  "details": {"title": "Bug Report"},
  "success": true,
  "rate_limit_remaining": 4950
}
```

Audit logs stored in: `var/audit/<integration>_operations.jsonl`

### 4. Dry-Run Mode

Test operations without execution:
```python
# GitHub
client = GitHubClient(dry_run=True)
result = client.create_issue(...)  # Returns preview

# Shows what would happen without executing
```

---

## API Endpoints

All integrations exposed via FastAPI endpoints:

### GitHub
```
GET  /api/integrations/github/repos
GET  /api/integrations/github/repos/{owner}/{repo}
GET  /api/integrations/github/repos/{owner}/{repo}/issues
POST /api/integrations/github/issues
POST /api/integrations/github/pull-requests
```

### Gmail
```
GET  /api/integrations/gmail/messages
GET  /api/integrations/gmail/messages/{message_id}
POST /api/integrations/gmail/send
```

### Calendar
```
GET  /api/integrations/calendar/calendars
GET  /api/integrations/calendar/events
POST /api/integrations/calendar/events
```

### Browser
```
POST /api/integrations/browser/session
POST /api/integrations/browser/navigate
POST /api/integrations/browser/extract
POST /api/integrations/browser/screenshot
```

### LLM
```
POST /api/integrations/llm/complete
GET  /api/integrations/llm/usage
```

### General
```
GET  /api/integrations/status  # All integrations status
```

---

## Testing

Comprehensive test suite with mocks:

```bash
# Run all integration tests
pytest tests/test_github_integration.py -v
pytest tests/test_google_integration.py -v
pytest tests/test_telegram_bot.py -v
pytest tests/test_browser_automation.py -v
pytest tests/test_llm_client.py -v

# Run with coverage
pytest tests/ --cov=integrations --cov-report=html
```

---

## Troubleshooting

### GitHub Integration

**Issue**: `ValueError: GitHub access token required`
- **Solution**: Set `GITHUB_ACCESS_TOKEN` in `.env`

**Issue**: `RateLimitExceededException`
- **Solution**: Wait for rate limit reset, check `get_rate_limit_status()`

### Google Integration

**Issue**: `FileNotFoundError: Google credentials not found`
- **Solution**: Download OAuth2 credentials to `config/google_credentials.json`

**Issue**: `RefreshError: Token expired`
- **Solution**: Delete `var/tokens/google_token.json`, re-authenticate

### Telegram Bot

**Issue**: `ValueError: Telegram bot token required`
- **Solution**: Set `TELEGRAM_BOT_TOKEN` in `.env`

**Issue**: Bot not responding
- **Solution**: Check user ID in `TELEGRAM_ALLOWED_USER_IDS`

### Browser Automation

**Issue**: `ImportError: Playwright not installed`
- **Solution**: Run `pip install playwright && playwright install chromium`

**Issue**: `PermissionError: URL not in allowlist`
- **Solution**: Add domain to `BROWSER_URL_ALLOWLIST` in `.env`

### LLM Client

**Issue**: `RuntimeError: Daily token limit exceeded`
- **Solution**: Wait for daily reset or increase `max_tokens_per_day`

**Issue**: `ValueError: Unsafe prompt`
- **Solution**: Remove prompt injection patterns from input

---

## Best Practices

1. **Always Use Dry-Run First**: Test operations before executing
2. **Monitor Rate Limits**: Check limits before bulk operations
3. **Review Audit Logs**: Monitor `var/audit/` for security events
4. **Rotate API Keys**: Regular key rotation for security
5. **Use Read-Only Scopes**: Minimum necessary permissions
6. **Enable Safety Checks**: Always run with `enable_safety_checks=True`
7. **Set Resource Limits**: Configure appropriate limits for your use case

---

## Support

For issues or questions:
- Check audit logs: `var/audit/<integration>_operations.jsonl`
- Review error messages in logs: `var/logs/`
- Run diagnostics: `pytest tests/test_<integration>.py -v`

---

## License

ShivX External Integrations - Production Ready
Copyright © 2024 ShivX Team
