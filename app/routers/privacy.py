"""
Privacy and GDPR Compliance API Endpoints

Provides comprehensive privacy control APIs:
- Consent management (grant/revoke/check)
- Telemetry preferences
- GDPR rights (access, rectification, erasure)
- Offline mode status
- Data retention settings
"""

import logging
import hashlib
from typing import Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from core.privacy.consent import ConsentManager, check_analytics_consent
from core.privacy.gdpr import GDPRCompliance, DataExportFormat
from core.privacy.offline import get_offline_mode
from core.privacy.airgap import get_airgap_mode
from app.models.privacy import ConsentType, ConsentStatus, TelemetryMode
from config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ConsentRequest(BaseModel):
    """Request to grant/revoke consent"""
    consent_type: ConsentType
    grant: bool = Field(..., description="True to grant, False to revoke")


class ConsentResponse(BaseModel):
    """Consent status response"""
    user_id: str
    consent_type: str
    status: str
    granted_at: Optional[str] = None
    revoked_at: Optional[str] = None


class AllConsentResponse(BaseModel):
    """All consent statuses"""
    user_id: str
    consents: Dict[str, bool]


class TelemetryPreferenceRequest(BaseModel):
    """Telemetry preference update request"""
    telemetry_mode: TelemetryMode
    do_not_track: Optional[bool] = None
    collect_errors: Optional[bool] = None
    collect_performance: Optional[bool] = None
    collect_usage: Optional[bool] = None


class DataExportRequest(BaseModel):
    """Data export request"""
    format: DataExportFormat = DataExportFormat.JSON
    include_metadata: bool = True


class ForgetMeRequest(BaseModel):
    """Forget-me request (GDPR erasure)"""
    confirmation: str = Field(
        ...,
        description="Confirmation token (SHA256 of user_id, first 16 chars)"
    )


class DataRectificationRequest(BaseModel):
    """Data rectification request"""
    corrections: Dict[str, any] = Field(
        ...,
        description="Dictionary of fields to correct"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_current_user_id(request: Request) -> str:
    """
    Extract user ID from request

    In production, this would extract from JWT token or session
    """
    # TODO: Implement proper authentication
    # For now, use a test user ID or from header
    return request.headers.get("X-User-ID", "test-user")


def get_client_ip(request: Request) -> Optional[str]:
    """Get client IP address"""
    return request.client.host if request.client else None


def get_user_agent(user_agent: Optional[str] = Header(None)) -> Optional[str]:
    """Get user agent from header"""
    return user_agent


# ============================================================================
# Consent Management Endpoints
# ============================================================================

@router.post("/consent", response_model=ConsentResponse)
async def update_consent(
    consent_req: ConsentRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_agent: Optional[str] = Header(None),
):
    """
    Update user consent for a specific type

    Grants or revokes consent for:
    - Necessary (always granted, cannot revoke)
    - Functional (offline mode, caching)
    - Analytics (telemetry, metrics)
    - Marketing (future use)
    """
    user_id = get_current_user_id(request)
    ip_address = get_client_ip(request)

    consent_mgr = ConsentManager(db)

    if consent_req.grant:
        consent = await consent_mgr.grant_consent(
            user_id=user_id,
            consent_type=consent_req.consent_type,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    else:
        if consent_req.consent_type == ConsentType.NECESSARY:
            raise HTTPException(
                status_code=400,
                detail="Cannot revoke necessary consent - required for basic functionality"
            )

        consent = await consent_mgr.revoke_consent(
            user_id=user_id,
            consent_type=consent_req.consent_type,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        if not consent:
            raise HTTPException(
                status_code=404,
                detail=f"No consent record found for type: {consent_req.consent_type}"
            )

    await db.commit()

    return ConsentResponse(
        user_id=consent.user_id,
        consent_type=consent.consent_type.value,
        status=consent.status.value,
        granted_at=consent.granted_at.isoformat() if consent.granted_at else None,
        revoked_at=consent.revoked_at.isoformat() if consent.revoked_at else None,
    )


@router.get("/consent", response_model=AllConsentResponse)
async def get_consent(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Get all consent statuses for current user

    Returns consent status for all consent types
    """
    user_id = get_current_user_id(request)

    consent_mgr = ConsentManager(db)
    consents = await consent_mgr.get_all_consent(user_id)

    return AllConsentResponse(
        user_id=user_id,
        consents=consents,
    )


@router.delete("/consent")
async def revoke_all_consent(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_agent: Optional[str] = Header(None),
):
    """
    Revoke all non-necessary consent

    Revokes consent for:
    - Functional
    - Analytics
    - Marketing

    Necessary consent cannot be revoked
    """
    user_id = get_current_user_id(request)
    ip_address = get_client_ip(request)

    consent_mgr = ConsentManager(db)
    count = await consent_mgr.revoke_all_consent(
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    await db.commit()

    return {
        "user_id": user_id,
        "consents_revoked": count,
        "message": f"Revoked {count} consent(s) successfully",
    }


# ============================================================================
# Telemetry Preference Endpoints
# ============================================================================

@router.get("/telemetry/status")
async def get_telemetry_status(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Get current telemetry status

    Returns what data is being collected and why
    """
    settings = get_settings()
    user_id = get_current_user_id(request)

    # Check if analytics allowed
    dnt_header = request.headers.get("DNT")
    analytics_allowed = await check_analytics_consent(db, user_id, dnt_header)

    return {
        "telemetry_mode": settings.telemetry_mode,
        "offline_mode": settings.offline_mode,
        "airgap_mode": settings.airgap_mode,
        "respect_dnt": settings.respect_dnt,
        "dnt_header": dnt_header,
        "analytics_allowed": analytics_allowed,
        "collection_enabled": {
            "errors": analytics_allowed and settings.telemetry_mode != "disabled",
            "performance": analytics_allowed and settings.telemetry_mode in ("standard", "full"),
            "usage": analytics_allowed and settings.telemetry_mode == "full",
        },
        "message": (
            "Telemetry disabled" if not analytics_allowed
            else f"Telemetry mode: {settings.telemetry_mode}"
        ),
    }


@router.get("/telemetry/data")
async def get_telemetry_data(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    View collected telemetry data for current user

    Returns telemetry data collected for transparency
    """
    user_id = get_current_user_id(request)

    # TODO: Implement fetching user-specific telemetry
    # For now, return placeholder

    return {
        "user_id": user_id,
        "telemetry_data": [],
        "message": "Telemetry data retrieval not yet implemented",
    }


# ============================================================================
# GDPR Rights Endpoints
# ============================================================================

@router.get("/data-export")
async def export_user_data(
    request: Request,
    export_req: DataExportRequest = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    Export all user data (GDPR Right to Access - Article 15)

    Returns comprehensive export of all user data in machine-readable format
    """
    user_id = get_current_user_id(request)

    logger.info(f"Data export requested: user={user_id}, format={export_req.format}")

    gdpr = GDPRCompliance(db)

    try:
        data = await gdpr.export_user_data(
            user_id=user_id,
            format=export_req.format,
            include_metadata=export_req.include_metadata,
        )

        await db.commit()

        return data

    except Exception as e:
        logger.error(f"Data export failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Data export failed: {str(e)}")


@router.delete("/forget-me")
async def forget_user(
    forget_req: ForgetMeRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete all user data (GDPR Right to Erasure - Article 17)

    ⚠️ WARNING: This operation is IRREVERSIBLE!

    Requires confirmation token: first 16 characters of SHA256(user_id)

    Deletes:
    - User profile
    - All conversations and memory
    - Trading data
    - Consent records
    - Telemetry data
    - Files and artifacts
    """
    user_id = get_current_user_id(request)
    ip_address = get_client_ip(request)

    logger.warning(f"⚠️ FORGET ME request: user={user_id}, ip={ip_address}")

    gdpr = GDPRCompliance(db)

    try:
        result = await gdpr.forget_user(
            user_id=user_id,
            confirmation_token=forget_req.confirmation,
            requester_ip=ip_address,
        )

        return {
            "status": "success",
            "message": "All user data has been permanently deleted",
            "details": result.to_dict(),
        }

    except ValueError as e:
        logger.warning(f"Forget-me failed - invalid confirmation: user={user_id}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Forget-me failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Data deletion failed: {str(e)}")


@router.put("/data-correction")
async def rectify_user_data(
    rectify_req: DataRectificationRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Correct user data (GDPR Right to Rectification - Article 16)

    Allows users to correct inaccurate personal data
    """
    user_id = get_current_user_id(request)
    ip_address = get_client_ip(request)

    logger.info(f"Data rectification requested: user={user_id}")

    gdpr = GDPRCompliance(db)

    try:
        results = await gdpr.rectify_user_data(
            user_id=user_id,
            corrections=rectify_req.corrections,
            requester_ip=ip_address,
        )

        await db.commit()

        return {
            "status": "success",
            "results": results,
        }

    except Exception as e:
        logger.error(f"Data rectification failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Data rectification failed: {str(e)}")


# ============================================================================
# Offline Mode Endpoints
# ============================================================================

@router.get("/offline-status")
async def get_offline_status():
    """
    Get offline mode status

    Returns:
    - Whether offline mode is enabled
    - Network isolation status
    - Blocked requests count
    - Degraded features
    """
    offline = get_offline_mode()
    status = offline.get_status()

    return status


# ============================================================================
# Air-Gap Mode Endpoints
# ============================================================================

@router.get("/airgap-status")
async def get_airgap_status():
    """
    Get air-gap mode status

    Returns:
    - Whether air-gap mode is enabled
    - Network isolation verification
    - Active network interfaces
    - Violations (if any)
    """
    airgap = get_airgap_mode()
    status = airgap.get_status()

    return status


# ============================================================================
# Privacy Policy Endpoint
# ============================================================================

@router.get("/policy")
async def get_privacy_policy():
    """
    Get privacy policy

    Returns platform privacy policy
    """
    return {
        "platform": "ShivX",
        "version": "1.0",
        "last_updated": "2025-10-28",
        "policy": {
            "data_collection": {
                "what_we_collect": [
                    "User profile information",
                    "Trading activity (if trading features enabled)",
                    "Conversation history (with consent)",
                    "Technical logs (errors, performance)",
                ],
                "why_we_collect": [
                    "Provide platform functionality",
                    "Improve user experience",
                    "Ensure platform security",
                    "Comply with legal obligations",
                ],
            },
            "data_retention": {
                "conversations": "90 days (configurable)",
                "memory": "365 days (configurable)",
                "audit_logs": "90 days (compliance requirement)",
                "telemetry": "30 days (configurable)",
            },
            "user_rights": {
                "access": "Export all your data at any time",
                "rectification": "Correct inaccurate data",
                "erasure": "Request complete data deletion",
                "portability": "Export data in machine-readable format",
                "restrict_processing": "Opt-out of analytics and telemetry",
            },
            "privacy_features": {
                "offline_mode": "Block all external network requests",
                "airgap_mode": "Complete network isolation",
                "consent_management": "Granular control over data collection",
                "telemetry_modes": "disabled, minimal, standard, full",
                "dnt_support": "Respects Do Not Track headers",
            },
            "contact": {
                "email": "privacy@shivx.ai",
                "website": "https://shivx.ai/privacy",
            },
        },
    }


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get("/health")
async def privacy_health_check():
    """
    Privacy subsystem health check

    Returns status of all privacy components
    """
    settings = get_settings()
    offline = get_offline_mode()
    airgap = get_airgap_mode()

    return {
        "status": "healthy",
        "privacy_features": {
            "offline_mode": {
                "enabled": settings.offline_mode,
                "status": "active" if offline.is_enabled() else "disabled",
            },
            "airgap_mode": {
                "enabled": settings.airgap_mode,
                "status": "active" if airgap.is_enabled() else "disabled",
            },
            "telemetry": {
                "mode": settings.telemetry_mode,
                "respect_dnt": settings.respect_dnt,
            },
            "gdpr": {
                "enabled": settings.gdpr_mode,
                "features": ["data_export", "forget_me", "rectification"],
            },
            "consent_management": {
                "enabled": True,
                "types": ["necessary", "functional", "analytics", "marketing"],
            },
        },
    }
