"""
Comprehensive Tests for GitHub Integration
===========================================
Tests all GitHub client operations with mocks for external API calls.

Test Coverage:
- Client initialization
- Read operations (repos, issues, PRs)
- Write operations (create issue, PR, comment)
- Rate limiting
- Error handling
- Audit logging
- Dry-run mode
"""

import pytest
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Optional

# Mock GitHub module before import
import sys
sys.modules['github'] = MagicMock()
sys.modules['github.Repository'] = MagicMock()
sys.modules['github.Issue'] = MagicMock()
sys.modules['github.PullRequest'] = MagicMock()


class MockRepo:
    """Mock GitHub repository"""
    def __init__(self, name: str):
        self.name = name
        self.full_name = f"owner/{name}"
        self.description = "Test repo"
        self.private = False
        self.language = "Python"
        self.stargazers_count = 100
        self.forks_count = 10
        self.html_url = f"https://github.com/owner/{name}"
        self.clone_url = f"https://github.com/owner/{name}.git"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.watchers_count = 50
        self.open_issues_count = 5
        self.default_branch = "main"

    def get_topics(self):
        return ["ai", "trading"]

    def get_issues(self, state="open", labels=None):
        return [MockIssue(1), MockIssue(2)]

    def get_issue(self, number: int):
        return MockIssue(number)

    def create_issue(self, title: str, body: str, labels: list):
        issue = MockIssue(123)
        issue.title = title
        issue.body = body
        return issue

    def create_pull(self, title: str, body: str, head: str, base: str):
        pr = MockPR(456)
        pr.title = title
        pr.body = body
        pr.head = head
        pr.base = base
        return pr


class MockIssue:
    """Mock GitHub issue"""
    def __init__(self, number: int):
        self.number = number
        self.title = f"Test Issue #{number}"
        self.body = "Test issue body"
        self.state = "open"
        self.user = MockUser("testuser")
        self.labels = []
        self.assignees = []
        self.comments = 3
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.closed_at = None
        self.html_url = f"https://github.com/owner/repo/issues/{number}"
        self.id = number * 1000

    def create_comment(self, body: str):
        comment = MockComment(self.number)
        comment.body = body
        return comment


class MockPR:
    """Mock GitHub pull request"""
    def __init__(self, number: int):
        self.number = number
        self.title = f"Test PR #{number}"
        self.body = "Test PR body"
        self.state = "open"
        self.user = MockUser("testuser")
        self.head = "feature-branch"
        self.base = "main"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.html_url = f"https://github.com/owner/repo/pull/{number}"


class MockComment:
    """Mock GitHub comment"""
    def __init__(self, issue_number: int):
        self.id = int(time.time() * 1000)
        self.body = "Test comment"
        self.user = MockUser("testuser")
        self.created_at = datetime.now()
        self.html_url = f"https://github.com/owner/repo/issues/{issue_number}#comment-{self.id}"


class MockUser:
    """Mock GitHub user"""
    def __init__(self, login: str):
        self.login = login
        self.name = f"{login.title()}"
        self.email = f"{login}@example.com"

    def get_repos(self):
        return [MockRepo("repo1"), MockRepo("repo2"), MockRepo("repo3")]


class MockOrg:
    """Mock GitHub organization"""
    def __init__(self, login: str):
        self.login = login

    def get_repos(self):
        return [MockRepo("org-repo1"), MockRepo("org-repo2")]


class MockRateLimit:
    """Mock GitHub rate limit"""
    def __init__(self):
        self.core = Mock()
        self.core.limit = 5000
        self.core.remaining = 4950
        self.core.reset = datetime.now()

        self.search = Mock()
        self.search.limit = 30
        self.search.remaining = 25
        self.search.reset = datetime.now()


class MockGithub:
    """Mock GitHub client"""
    def __init__(self, token: str):
        self.token = token

    def get_user(self, username: Optional[str] = None):
        if username:
            return MockUser(username)
        return MockUser("authenticated_user")

    def get_organization(self, org: str):
        return MockOrg(org)

    def get_repo(self, repo_full_name: str):
        parts = repo_full_name.split("/")
        return MockRepo(parts[-1] if len(parts) > 1 else repo_full_name)

    def get_rate_limit(self):
        return MockRateLimit()


# Patch the github module
sys.modules['github'].Github = MockGithub
sys.modules['github'].GithubException = Exception
sys.modules['github'].RateLimitExceededException = Exception


@pytest.fixture
def mock_guardian():
    """Mock Guardian Defense"""
    with patch('integrations.github_client.get_guardian_defense') as mock:
        guardian = MagicMock()
        guardian.detect_rate_limit_abuse = MagicMock()
        mock.return_value = guardian
        yield guardian


@pytest.fixture
def mock_policy_guard():
    """Mock Policy Guard"""
    with patch('integrations.github_client.get_policy_guard') as mock:
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
    with patch('integrations.github_client.append_jsonl') as mock:
        yield mock


@pytest.fixture
def github_client(mock_guardian, mock_policy_guard, mock_audit, tmp_path):
    """Create GitHub client with mocks"""
    from integrations.github_client import GitHubClient

    # Use temporary audit log path
    audit_path = tmp_path / "github_audit.jsonl"

    client = GitHubClient(
        access_token="test_token_12345",
        audit_log_path=str(audit_path),
        dry_run=False
    )
    return client


@pytest.fixture
def github_client_dry_run(mock_guardian, mock_policy_guard, mock_audit, tmp_path):
    """Create GitHub client in dry-run mode"""
    from integrations.github_client import GitHubClient

    audit_path = tmp_path / "github_audit.jsonl"

    client = GitHubClient(
        access_token="test_token_12345",
        audit_log_path=str(audit_path),
        dry_run=True
    )
    return client


# ============================================================================
# Initialization Tests
# ============================================================================

def test_github_client_initialization(github_client):
    """Test GitHub client initialization"""
    assert github_client is not None
    assert github_client.access_token == "test_token_12345"
    assert github_client.dry_run is False
    assert github_client.rate_limit_remaining == 5000


def test_github_client_initialization_dry_run(github_client_dry_run):
    """Test GitHub client initialization in dry-run mode"""
    assert github_client_dry_run.dry_run is True


def test_github_client_initialization_missing_token():
    """Test GitHub client fails without token"""
    from integrations.github_client import GitHubClient

    with pytest.raises(ValueError, match="GitHub access token required"):
        GitHubClient(access_token=None)


# ============================================================================
# Read Operations Tests
# ============================================================================

def test_list_repositories_authenticated_user(github_client):
    """Test listing repositories for authenticated user"""
    repos = github_client.list_repositories()

    assert len(repos) == 3
    assert all('name' in repo for repo in repos)
    assert all('full_name' in repo for repo in repos)
    assert all('url' in repo for repo in repos)


def test_list_repositories_specific_user(github_client):
    """Test listing repositories for specific user"""
    repos = github_client.list_repositories(user="octocat")

    assert len(repos) == 3
    assert all('stars' in repo for repo in repos)
    assert all('forks' in repo for repo in repos)


def test_list_repositories_organization(github_client):
    """Test listing repositories for organization"""
    repos = github_client.list_repositories(org="github")

    assert len(repos) == 2
    assert all('language' in repo for repo in repos)


def test_get_repository_details(github_client):
    """Test getting repository details"""
    repo = github_client.get_repository("owner/test-repo")

    assert repo['name'] == "test-repo"
    assert repo['full_name'] == "owner/test-repo"
    assert 'stars' in repo
    assert 'forks' in repo
    assert 'topics' in repo


def test_list_issues(github_client):
    """Test listing issues in repository"""
    issues = github_client.list_issues("owner/repo")

    assert len(issues) == 2
    assert all('number' in issue for issue in issues)
    assert all('title' in issue for issue in issues)
    assert all('state' in issue for issue in issues)


def test_list_issues_with_filters(github_client):
    """Test listing issues with filters"""
    issues = github_client.list_issues(
        "owner/repo",
        state="closed",
        labels=["bug", "urgent"]
    )

    assert isinstance(issues, list)


def test_get_issue_details(github_client):
    """Test getting issue details"""
    issue = github_client.get_issue("owner/repo", 123)

    assert issue['number'] == 123
    assert 'title' in issue
    assert 'body' in issue
    assert 'user' in issue
    assert 'labels' in issue
    assert 'comments' in issue


def test_rate_limit_check(github_client):
    """Test rate limit checking"""
    github_client._check_rate_limit()

    assert github_client.rate_limit_remaining == 4950


def test_rate_limit_status(github_client):
    """Test getting rate limit status"""
    status = github_client.get_rate_limit_status()

    assert 'core' in status
    assert 'search' in status
    assert status['core']['limit'] == 5000
    assert status['core']['remaining'] == 4950


# ============================================================================
# Write Operations Tests
# ============================================================================

def test_create_issue_dry_run(github_client_dry_run):
    """Test creating issue in dry-run mode"""
    result = github_client_dry_run.create_issue(
        repo_full_name="owner/repo",
        title="Test Issue",
        body="This is a test issue",
        labels=["bug", "enhancement"]
    )

    assert result['dry_run'] is True
    assert result['action'] == "create_issue"
    assert result['title'] == "Test Issue"
    assert 'Would create issue' in result['message']


def test_create_issue_with_approval(github_client, mock_policy_guard):
    """Test creating issue with approval"""
    # Mock approval
    result_mock = MagicMock()
    result_mock.decision = "allow"
    result_mock.requires_approval = False
    mock_policy_guard.evaluate.return_value = result_mock

    result = github_client.create_issue(
        repo_full_name="owner/repo",
        title="Test Issue",
        body="This is a test issue",
        labels=["bug"]
    )

    assert 'number' in result
    assert 'url' in result
    assert result['title'] == "Test Issue"


def test_create_issue_denied(github_client, mock_policy_guard):
    """Test creating issue denied by policy"""
    # Mock denial
    result_mock = MagicMock()
    result_mock.decision = "deny"
    result_mock.reasons = ["Insufficient permissions"]
    mock_policy_guard.evaluate.return_value = result_mock

    with pytest.raises(PermissionError):
        github_client.create_issue(
            repo_full_name="owner/repo",
            title="Test Issue"
        )


def test_create_pull_request_dry_run(github_client_dry_run):
    """Test creating PR in dry-run mode"""
    result = github_client_dry_run.create_pull_request(
        repo_full_name="owner/repo",
        title="Test PR",
        head="feature-branch",
        base="main",
        body="This is a test PR"
    )

    assert result['dry_run'] is True
    assert result['action'] == "create_pull_request"
    assert result['title'] == "Test PR"
    assert result['head'] == "feature-branch"
    assert result['base'] == "main"


def test_create_pull_request_with_approval(github_client, mock_policy_guard):
    """Test creating PR with approval"""
    result_mock = MagicMock()
    result_mock.decision = "allow"
    result_mock.requires_approval = False
    mock_policy_guard.evaluate.return_value = result_mock

    result = github_client.create_pull_request(
        repo_full_name="owner/repo",
        title="Test PR",
        head="feature-branch",
        base="main"
    )

    assert 'number' in result
    assert 'url' in result
    assert result['title'] == "Test PR"


def test_add_comment_dry_run(github_client_dry_run):
    """Test adding comment in dry-run mode"""
    result = github_client_dry_run.add_comment(
        repo_full_name="owner/repo",
        issue_number=123,
        comment="This is a test comment"
    )

    assert result['dry_run'] is True
    assert result['action'] == "add_comment"
    assert result['issue_number'] == 123
    assert result['comment'] == "This is a test comment"


def test_add_comment_with_approval(github_client, mock_policy_guard):
    """Test adding comment with approval"""
    result_mock = MagicMock()
    result_mock.decision = "allow"
    result_mock.requires_approval = False
    mock_policy_guard.evaluate.return_value = result_mock

    result = github_client.add_comment(
        repo_full_name="owner/repo",
        issue_number=123,
        comment="This is a test comment"
    )

    assert 'comment_id' in result
    assert 'url' in result
    assert result['issue_number'] == 123


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_list_repositories_api_error(github_client):
    """Test handling API errors when listing repositories"""
    with patch.object(github_client.client, 'get_user', side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="API Error"):
            github_client.list_repositories()


def test_get_repository_not_found(github_client):
    """Test handling repository not found"""
    with patch.object(github_client.client, 'get_repo', side_effect=Exception("Not Found")):
        with pytest.raises(Exception, match="Not Found"):
            github_client.get_repository("owner/nonexistent")


def test_create_issue_api_error(github_client, mock_policy_guard):
    """Test handling API errors when creating issue"""
    result_mock = MagicMock()
    result_mock.decision = "allow"
    result_mock.requires_approval = False
    mock_policy_guard.evaluate.return_value = result_mock

    with patch.object(github_client.client, 'get_repo', side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="API Error"):
            github_client.create_issue(
                repo_full_name="owner/repo",
                title="Test Issue"
            )


# ============================================================================
# Audit Logging Tests
# ============================================================================

def test_audit_logging_read_operation(github_client, mock_audit):
    """Test audit logging for read operations"""
    github_client.list_repositories()

    # Verify audit was logged
    assert mock_audit.called
    call_args = mock_audit.call_args[0]
    log_entry = call_args[1]

    assert log_entry['integration'] == 'github'
    assert log_entry['operation_type'] == 'read'
    assert log_entry['action'] == 'list_repositories'
    assert log_entry['success'] is True


def test_audit_logging_write_operation(github_client_dry_run, mock_audit):
    """Test audit logging for write operations"""
    github_client_dry_run.create_issue(
        repo_full_name="owner/repo",
        title="Test Issue"
    )

    # Verify audit was logged
    assert mock_audit.called
    call_args = mock_audit.call_args[0]
    log_entry = call_args[1]

    assert log_entry['integration'] == 'github'
    assert log_entry['operation_type'] == 'write'
    assert log_entry['action'] == 'create_issue'
    assert log_entry['dry_run'] is True


def test_audit_logging_error(github_client, mock_audit):
    """Test audit logging for errors"""
    with patch.object(github_client.client, 'get_repo', side_effect=Exception("API Error")):
        try:
            github_client.get_repository("owner/repo")
        except Exception:
            pass

    # Verify error was logged
    assert mock_audit.called
    call_args = mock_audit.call_args[0]
    log_entry = call_args[1]

    assert log_entry['success'] is False
    assert log_entry['error'] is not None


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_issue_workflow(github_client_dry_run):
    """Test complete issue workflow: list, get, create"""
    # List issues
    issues = github_client_dry_run.list_issues("owner/repo")
    assert len(issues) > 0

    # Get specific issue
    issue = github_client_dry_run.get_issue("owner/repo", issues[0]['number'])
    assert issue['number'] == issues[0]['number']

    # Create new issue (dry-run)
    new_issue = github_client_dry_run.create_issue(
        repo_full_name="owner/repo",
        title="New Issue",
        body="Issue body"
    )
    assert new_issue['dry_run'] is True


def test_full_pr_workflow(github_client_dry_run):
    """Test complete PR workflow: create and add comment"""
    # Create PR (dry-run)
    pr = github_client_dry_run.create_pull_request(
        repo_full_name="owner/repo",
        title="New PR",
        head="feature",
        base="main"
    )
    assert pr['dry_run'] is True

    # Add comment (dry-run)
    comment = github_client_dry_run.add_comment(
        repo_full_name="owner/repo",
        issue_number=123,  # PRs are issues too
        comment="LGTM!"
    )
    assert comment['dry_run'] is True


# ============================================================================
# Performance Tests
# ============================================================================

def test_multiple_operations_performance(github_client):
    """Test performance with multiple operations"""
    start = time.time()

    # Perform multiple operations
    for i in range(10):
        github_client.list_repositories()

    duration = time.time() - start

    # Should complete in reasonable time (< 5 seconds for mocked operations)
    assert duration < 5.0


def test_rate_limit_tracking(github_client):
    """Test rate limit is tracked across operations"""
    initial_remaining = github_client.rate_limit_remaining

    github_client.list_repositories()
    github_client.get_repository("owner/repo")
    github_client.list_issues("owner/repo")

    # Rate limit should be tracked (updated by _check_rate_limit)
    assert github_client.rate_limit_remaining == 4950
