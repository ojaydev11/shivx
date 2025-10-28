"""
Integrations API Router
========================
API endpoints for external service integrations.

Integrations:
- GitHub Operations
- Gmail/Calendar
- Telegram Bot
- Browser Automation
- LLM Bridges (Claude, ChatGPT)
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, require_permission, get_settings
from app.dependencies.auth import TokenData
from config.settings import Settings
from core.security.hardening import Permission

# Lazy load integration clients to avoid import errors
from integrations import (
    get_github_client,
    get_google_client,
    get_telegram_bot,
    get_browser_automation,
    get_llm_client,
    get_integration_status
)

router = APIRouter(
    prefix="/api/integrations",
    tags=["integrations"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

# GitHub Models
class GitHubRepoFilter(BaseModel):
    user: Optional[str] = None
    org: Optional[str] = None


class GitHubIssueCreate(BaseModel):
    repo: str = Field(..., description="Repository full name (owner/repo)")
    title: str
    body: Optional[str] = None
    labels: Optional[List[str]] = None


class GitHubPRCreate(BaseModel):
    repo: str = Field(..., description="Repository full name (owner/repo)")
    title: str
    head: str = Field(..., description="Branch to merge from")
    base: str = Field(..., description="Branch to merge into")
    body: Optional[str] = None


# Gmail Models
class GmailMessageFilter(BaseModel):
    query: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=100)
    label_ids: Optional[List[str]] = None


class GmailSendMessage(BaseModel):
    to: str
    subject: str
    body: str
    cc: Optional[str] = None
    bcc: Optional[str] = None


# Calendar Models
class CalendarEventCreate(BaseModel):
    summary: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: Optional[List[str]] = None
    calendar_id: str = "primary"


# Browser Models
class BrowserNavigate(BaseModel):
    url: str
    timeout: Optional[int] = None


class BrowserExtractText(BaseModel):
    url: str
    selector: Optional[str] = None


class BrowserScreenshot(BaseModel):
    url: str
    output_path: str
    full_page: bool = False


# LLM Models
class LLMCompleteRequest(BaseModel):
    prompt: str
    provider: str = Field(..., description="claude or chatgpt")
    model: Optional[str] = None
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)


# ============================================================================
# GitHub Endpoints
# ============================================================================

@router.get("/github/repos")
async def list_github_repos(
    user: Optional[str] = None,
    org: Optional[str] = None,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    List GitHub repositories.

    Requires: READ permission
    """
    try:
        GitHubClient = get_github_client()
        client = GitHubClient()

        repos = client.list_repositories(user=user, org=org)

        return {
            "repos": repos,
            "count": len(repos)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list repositories: {str(e)}"
        )


@router.get("/github/repos/{owner}/{repo}")
async def get_github_repo(
    owner: str,
    repo: str,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    Get GitHub repository details.

    Requires: READ permission
    """
    try:
        GitHubClient = get_github_client()
        client = GitHubClient()

        repo_info = client.get_repository(f"{owner}/{repo}")

        return repo_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get repository: {str(e)}"
        )


@router.get("/github/repos/{owner}/{repo}/issues")
async def list_github_issues(
    owner: str,
    repo: str,
    state: str = "open",
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    List GitHub issues.

    Requires: READ permission
    """
    try:
        GitHubClient = get_github_client()
        client = GitHubClient()

        issues = client.list_issues(f"{owner}/{repo}", state=state)

        return {
            "issues": issues,
            "count": len(issues)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list issues: {str(e)}"
        )


@router.post("/github/issues")
async def create_github_issue(
    issue: GitHubIssueCreate,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Create GitHub issue.

    Requires: EXECUTE permission (write operation requires approval)
    """
    try:
        GitHubClient = get_github_client()
        client = GitHubClient()

        result = client.create_issue(
            repo_full_name=issue.repo,
            title=issue.title,
            body=issue.body,
            labels=issue.labels
        )

        return result
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create issue: {str(e)}"
        )


@router.post("/github/pull-requests")
async def create_github_pr(
    pr: GitHubPRCreate,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Create GitHub pull request.

    Requires: EXECUTE permission (write operation requires approval)
    """
    try:
        GitHubClient = get_github_client()
        client = GitHubClient()

        result = client.create_pull_request(
            repo_full_name=pr.repo,
            title=pr.title,
            head=pr.head,
            base=pr.base,
            body=pr.body
        )

        return result
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pull request: {str(e)}"
        )


# ============================================================================
# Gmail Endpoints
# ============================================================================

@router.get("/gmail/messages")
async def list_gmail_messages(
    query: Optional[str] = None,
    max_results: int = 10,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    List Gmail messages.

    Requires: READ permission
    """
    try:
        GoogleClient = get_google_client()
        client = GoogleClient()

        messages = client.list_messages(query=query, max_results=max_results)

        return {
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list messages: {str(e)}"
        )


@router.get("/gmail/messages/{message_id}")
async def get_gmail_message(
    message_id: str,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    Get Gmail message details.

    Requires: READ permission
    """
    try:
        GoogleClient = get_google_client()
        client = GoogleClient()

        message = client.get_message(message_id)

        return message
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get message: {str(e)}"
        )


@router.post("/gmail/send")
async def send_gmail_message(
    message: GmailSendMessage,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Send Gmail message.

    Requires: EXECUTE permission (write operation requires approval)
    """
    try:
        GoogleClient = get_google_client()
        client = GoogleClient()

        result = client.send_message(
            to=message.to,
            subject=message.subject,
            body=message.body,
            cc=message.cc,
            bcc=message.bcc
        )

        return result
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send message: {str(e)}"
        )


# ============================================================================
# Calendar Endpoints
# ============================================================================

@router.get("/calendar/calendars")
async def list_calendars(
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    List calendars.

    Requires: READ permission
    """
    try:
        GoogleClient = get_google_client()
        client = GoogleClient()

        calendars = client.list_calendars()

        return {
            "calendars": calendars,
            "count": len(calendars)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list calendars: {str(e)}"
        )


@router.get("/calendar/events")
async def list_calendar_events(
    calendar_id: str = "primary",
    max_results: int = 10,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    List calendar events.

    Requires: READ permission
    """
    try:
        GoogleClient = get_google_client()
        client = GoogleClient()

        events = client.list_events(
            calendar_id=calendar_id,
            max_results=max_results
        )

        return {
            "events": events,
            "count": len(events)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list events: {str(e)}"
        )


@router.post("/calendar/events")
async def create_calendar_event(
    event: CalendarEventCreate,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Create calendar event.

    Requires: EXECUTE permission (write operation requires approval)
    """
    try:
        GoogleClient = get_google_client()
        client = GoogleClient()

        result = client.create_event(
            summary=event.summary,
            start_time=event.start_time,
            end_time=event.end_time,
            description=event.description,
            location=event.location,
            attendees=event.attendees,
            calendar_id=event.calendar_id
        )

        return result
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create event: {str(e)}"
        )


# ============================================================================
# Browser Automation Endpoints
# ============================================================================

@router.post("/browser/session")
async def create_browser_session(
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Create browser automation session.

    Requires: EXECUTE permission
    """
    try:
        BrowserAutomation = get_browser_automation()
        client = BrowserAutomation()

        session_id = await client.create_session()

        return {
            "session_id": session_id,
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@router.post("/browser/navigate")
async def browser_navigate(
    nav: BrowserNavigate,
    session_id: str,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Navigate browser to URL.

    Requires: EXECUTE permission
    """
    try:
        BrowserAutomation = get_browser_automation()
        client = BrowserAutomation()

        result = await client.navigate(
            session_id=session_id,
            url=nav.url,
            timeout=nav.timeout
        )

        return result
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to navigate: {str(e)}"
        )


@router.post("/browser/extract")
async def browser_extract_text(
    extract: BrowserExtractText,
    session_id: str,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Extract text from page.

    Requires: EXECUTE permission
    """
    try:
        BrowserAutomation = get_browser_automation()
        client = BrowserAutomation()

        result = await client.extract_text(
            session_id=session_id,
            url=extract.url,
            selector=extract.selector
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text: {str(e)}"
        )


@router.post("/browser/screenshot")
async def browser_screenshot(
    screenshot: BrowserScreenshot,
    session_id: str,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Take screenshot of page.

    Requires: EXECUTE permission
    """
    try:
        BrowserAutomation = get_browser_automation()
        client = BrowserAutomation()

        result = await client.take_screenshot(
            session_id=session_id,
            url=screenshot.url,
            output_path=screenshot.output_path,
            full_page=screenshot.full_page
        )

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to take screenshot: {str(e)}"
        )


# ============================================================================
# LLM Endpoints
# ============================================================================

@router.post("/llm/complete")
async def llm_complete(
    request: LLMCompleteRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
):
    """
    Complete prompt using LLM.

    Requires: EXECUTE permission

    Supports: claude, chatgpt
    """
    try:
        from integrations.llm_client import LLMProvider

        # Validate provider
        if request.provider not in ["claude", "chatgpt"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {request.provider}. Use 'claude' or 'chatgpt'"
            )

        provider = LLMProvider.CLAUDE if request.provider == "claude" else LLMProvider.CHATGPT

        LLMClient = get_llm_client()
        client = LLMClient()

        response = await client.complete(
            prompt=request.prompt,
            provider=provider,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        return {
            "provider": response.provider,
            "model": response.model,
            "content": response.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "estimated_cost": response.usage.estimated_cost
            },
            "finish_reason": response.finish_reason,
            "timestamp": response.timestamp,
            "safe": response.safe,
            "safety_issues": response.safety_issues
        }
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete prompt: {str(e)}"
        )


@router.get("/llm/usage")
async def llm_usage(
    current_user: TokenData = Depends(require_permission(Permission.READ)),
):
    """
    Get LLM usage statistics.

    Requires: READ permission
    """
    try:
        LLMClient = get_llm_client()
        client = LLMClient()

        usage = client.get_usage_stats()

        return usage
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage stats: {str(e)}"
        )


# ============================================================================
# General Integration Endpoints
# ============================================================================

@router.get("/status")
async def integrations_status():
    """
    Get status of all integrations (public endpoint).

    No authentication required
    """
    status = get_integration_status()

    return {
        "integrations": status,
        "timestamp": datetime.now().isoformat()
    }
