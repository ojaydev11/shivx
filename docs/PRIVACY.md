# ShivX Privacy & GDPR Compliance

## Table of Contents

1. [Overview](#overview)
2. [Privacy Architecture](#privacy-architecture)
3. [Privacy Modes](#privacy-modes)
4. [GDPR Compliance](#gdpr-compliance)
5. [Consent Management](#consent-management)
6. [Telemetry Privacy](#telemetry-privacy)
7. [Data Retention](#data-retention)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Best Practices](#best-practices)

---

## Overview

ShivX implements comprehensive privacy controls that give users complete control over their data. Our privacy implementation is built on four pillars:

1. **User Consent** - Granular control over data collection
2. **Data Minimization** - Only collect what's necessary
3. **Transparency** - Clear visibility into what data is collected
4. **User Rights** - Full GDPR compliance including data export and erasure

### Key Features

- **Offline Mode** - Block all external network requests
- **Air-Gap Mode** - Maximum network isolation with verification
- **Consent Management** - Granular consent for different data types
- **Telemetry Controls** - Four-level telemetry mode (disabled/minimal/standard/full)
- **GDPR Rights** - Data export, rectification, erasure, portability
- **DNT Support** - Respects Do Not Track headers
- **Audit Trail** - Complete audit log of all privacy-sensitive operations

---

## Privacy Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Privacy Control Layer                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Offline    │  │   Air-Gap    │  │   Consent    │      │
│  │     Mode     │  │     Mode     │  │  Management  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Telemetry   │  │     GDPR     │  │     Data     │      │
│  │   Privacy    │  │  Compliance  │  │  Retention   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (All data collection goes through privacy controls)         │
└─────────────────────────────────────────────────────────────┘
```

### Privacy-First Design Principles

1. **Fail Closed** - If privacy checks fail, block the operation
2. **Opt-In by Default** - Users must explicitly consent to non-essential data collection
3. **Transparent** - All data collection is logged and visible to users
4. **Reversible** - Users can revoke consent and request data deletion at any time

---

## Privacy Modes

### 1. Offline Mode

**Purpose:** Block all external network requests while allowing localhost communication.

**Use Cases:**
- Data privacy compliance
- Development/testing without external dependencies
- Network-isolated environments

**Configuration:**
```bash
SHIVX_OFFLINE_MODE=true
```

**Behavior:**
- ✅ Allows: localhost, 127.0.0.1, ::1, 0.0.0.0
- ❌ Blocks: All external HTTP/HTTPS requests
- 📊 Telemetry: Disabled
- 🔍 Logging: All blocked requests are logged

**API Endpoint:**
```bash
GET /api/privacy/offline-status
```

**Example Response:**
```json
{
  "offline_mode": true,
  "status": "isolated",
  "blocked_requests": 42,
  "degraded_features": [
    "External API calls (DEX, price feeds, etc.)",
    "Real-time market data updates",
    "Telemetry and error reporting"
  ]
}
```

### 2. Air-Gap Mode

**Purpose:** Maximum network isolation with startup verification.

**Use Cases:**
- High-security deployments
- Compliance requirements (financial, healthcare)
- Zero-trust environments

**Configuration:**
```bash
SHIVX_AIRGAP_MODE=true
```

**Behavior:**
- 🔒 Verifies no external network interfaces on startup
- ❌ Fails startup if external interfaces detected
- 📊 Telemetry: Disabled
- 🔍 Logging: All connection attempts logged as violations

**Startup Verification:**
```
============================================================
AIR-GAP MODE ENABLED
Performing network isolation verification...
============================================================
✓ Air-gap verification successful
============================================================
```

**API Endpoint:**
```bash
GET /api/privacy/airgap-status
```

**Example Response:**
```json
{
  "airgap_mode": true,
  "status": "isolated",
  "verified": true,
  "violations": 0,
  "external_interfaces": []
}
```

### 3. Telemetry Modes

**Purpose:** Control what telemetry data is collected.

**Modes:**

| Mode     | Errors | Performance | Usage | Use Case                    |
|----------|--------|-------------|-------|----------------------------|
| disabled | ❌     | ❌          | ❌    | Maximum privacy            |
| minimal  | ✅     | ❌          | ❌    | Critical events only       |
| standard | ✅     | ✅          | ❌    | Balanced (recommended)     |
| full     | ✅     | ✅          | ✅    | Development/testing        |

**Configuration:**
```bash
SHIVX_TELEMETRY_MODE=standard
```

**Privacy Controls:**
- Respects user consent (analytics consent required)
- Respects DNT (Do Not Track) headers
- Disabled in offline/airgap modes
- User can opt-out via API

**API Endpoints:**
```bash
GET  /api/privacy/telemetry/status
POST /api/privacy/telemetry/opt-out
```

---

## GDPR Compliance

ShivX is fully GDPR-compliant, implementing all required data subject rights.

### Rights Implemented

#### 1. Right to Access (Article 15)

Export all user data in machine-readable format.

**API:**
```bash
GET /api/privacy/data-export?format=json&include_metadata=true
```

**Response:**
```json
{
  "user_id": "user123",
  "export_date": "2025-10-28T10:00:00Z",
  "data": {
    "profile": {...},
    "consents": [...],
    "telemetry_preferences": {...},
    "conversations": [...],
    "memory": [...],
    "audit_logs": [...],
    "trading": {...}
  },
  "metadata": {
    "version": "1.0",
    "platform": "shivx",
    "re_importable": true
  }
}
```

#### 2. Right to Erasure / "Right to be Forgotten" (Article 17)

Permanently delete all user data.

**API:**
```bash
DELETE /api/privacy/forget-me
```

**Request:**
```json
{
  "confirmation": "a1b2c3d4e5f6g7h8"
}
```

**Confirmation Token:** First 16 characters of SHA256(user_id)

**Example:**
```bash
# Generate confirmation token
echo -n "user123" | sha256sum | cut -c1-16
```

**Response:**
```json
{
  "status": "success",
  "message": "All user data has been permanently deleted",
  "details": {
    "tables_purged": {
      "user_consents": 3,
      "telemetry_preferences": 1,
      "conversations": 45,
      "memory_entries": 120,
      "audit_logs_anonymized": 89,
      "users": 1
    },
    "total_records_deleted": 259,
    "files_deleted": [
      "./data/users/user123"
    ],
    "total_files_deleted": 1,
    "duration_seconds": 1.234
  }
}
```

**What Gets Deleted:**
- ✅ User profile
- ✅ Conversations and messages
- ✅ Memory entries
- ✅ Trading data (positions, orders)
- ✅ Consent records
- ✅ Telemetry preferences
- ✅ API keys
- ✅ User files
- ✅ Vector store embeddings
- 📝 Audit logs (anonymized, not deleted)

#### 3. Right to Rectification (Article 16)

Correct inaccurate personal data.

**API:**
```bash
PUT /api/privacy/data-correction
```

**Request:**
```json
{
  "corrections": {
    "profile": {
      "email": "newemail@example.com",
      "username": "newusername"
    }
  }
}
```

#### 4. Right to Data Portability (Article 20)

Export data in portable format for transfer to another service.

**API:**
```bash
GET /api/privacy/data-export?format=json&include_metadata=true
```

---

## Consent Management

### Consent Types

ShivX implements granular consent management:

| Consent Type | Description                          | Can Revoke? |
|-------------|--------------------------------------|-------------|
| necessary   | Required for basic functionality     | ❌ No       |
| functional  | Offline mode, caching, preferences   | ✅ Yes      |
| analytics   | Telemetry, metrics, performance data | ✅ Yes      |
| marketing   | Future use (newsletters, etc.)       | ✅ Yes      |

### Consent Lifecycle

```
┌─────────────┐
│   PENDING   │ (Initial state)
└──────┬──────┘
       │
       ├─────► GRANTED (User grants consent)
       │
       └─────► DENIED  (User denies consent)
                 │
                 └─────► REVOKED (User revokes consent)
```

### API Endpoints

#### Grant/Revoke Consent

```bash
POST /api/privacy/consent
```

**Request:**
```json
{
  "consent_type": "analytics",
  "grant": true
}
```

#### Get All Consent Statuses

```bash
GET /api/privacy/consent
```

**Response:**
```json
{
  "user_id": "user123",
  "consents": {
    "necessary": true,
    "functional": true,
    "analytics": false,
    "marketing": false
  }
}
```

#### Revoke All Consent

```bash
DELETE /api/privacy/consent
```

**Response:**
```json
{
  "user_id": "user123",
  "consents_revoked": 2,
  "message": "Revoked 2 consent(s) successfully"
}
```

### Consent Enforcement

Consent is enforced at multiple levels:

1. **Decorator-based**
```python
@consent_required(ConsentType.ANALYTICS)
async def track_user_action(user_id: str, db: AsyncSession):
    # Only executes if user has granted analytics consent
    ...
```

2. **Function-based**
```python
has_consent = await consent_manager.check_consent(
    user_id=user_id,
    consent_type=ConsentType.ANALYTICS
)
if not has_consent:
    return  # Skip operation
```

3. **Middleware-based** (DNT + Telemetry Mode + Consent)
```python
if request.headers.get("DNT") == "1":
    return  # Skip analytics
```

---

## Telemetry Privacy

### Privacy Controls

Telemetry collection respects multiple privacy controls:

```
┌──────────────────────────────────────────────────┐
│ Should we collect telemetry for this event?      │
│                                                   │
│ 1. Is offline mode enabled?        → NO          │
│ 2. Is airgap mode enabled?         → NO          │
│ 3. Is telemetry_mode=disabled?     → NO          │
│ 4. Does DNT header = 1?            → NO          │
│ 5. Does user have analytics consent? → YES       │
│ 6. Does telemetry mode allow event? → YES        │
│                                                   │
│ Only if ALL checks pass → Collect telemetry      │
└──────────────────────────────────────────────────┘
```

### Telemetry Transparency

Users can view what telemetry data has been collected:

```bash
GET /api/privacy/telemetry/status
```

**Response:**
```json
{
  "telemetry_mode": "standard",
  "offline_mode": false,
  "airgap_mode": false,
  "respect_dnt": true,
  "dnt_header": "0",
  "analytics_allowed": true,
  "collection_enabled": {
    "errors": true,
    "performance": true,
    "usage": false
  }
}
```

---

## Data Retention

### Default Retention Periods

| Data Type       | Default Retention | Configurable? |
|----------------|-------------------|---------------|
| Conversations  | 90 days           | ✅ Yes        |
| Memory         | 365 days          | ✅ Yes        |
| Audit Logs     | 90 days           | ✅ Yes        |
| Telemetry      | 30 days           | ✅ Yes        |

### Configuration

**Global (via environment):**
```bash
SHIVX_CONVERSATION_RETENTION_DAYS=90
SHIVX_MEMORY_RETENTION_DAYS=365
SHIVX_AUDIT_LOG_RETENTION_DAYS=90
SHIVX_TELEMETRY_RETENTION_DAYS=30
SHIVX_AUTO_PURGE_ENABLED=true
```

**Per-User (via database):**
```sql
INSERT INTO data_retention (
    user_id,
    conversation_days,
    memory_days,
    audit_log_days,
    telemetry_days,
    auto_purge_enabled
) VALUES (
    'user123',
    30,   -- Shorter retention for conversations
    90,   -- Shorter retention for memory
    90,   -- Keep audit logs
    7,    -- Minimal telemetry retention
    true
);
```

### Auto-Purge

When `auto_purge_enabled=true`, data is automatically deleted after retention period expires.

**Purge Schedule:**
- Runs: Daily at 2:00 AM UTC
- What: Deletes data older than retention period
- Logs: Purge operations logged to audit log

---

## API Reference

### Privacy Router Endpoints

All endpoints are prefixed with `/api/privacy`.

#### Consent Management

| Method | Endpoint              | Description                      |
|--------|-----------------------|----------------------------------|
| POST   | `/consent`            | Grant or revoke consent          |
| GET    | `/consent`            | Get all consent statuses         |
| DELETE | `/consent`            | Revoke all non-necessary consent |

#### Telemetry

| Method | Endpoint                   | Description                  |
|--------|----------------------------|------------------------------|
| GET    | `/telemetry/status`        | Get telemetry status         |
| GET    | `/telemetry/data`          | View collected telemetry     |

#### GDPR Rights

| Method | Endpoint             | Description                    |
|--------|----------------------|--------------------------------|
| GET    | `/data-export`       | Export all user data           |
| DELETE | `/forget-me`         | Delete all user data (irreversible) |
| PUT    | `/data-correction`   | Correct user data              |

#### Privacy Modes

| Method | Endpoint            | Description              |
|--------|---------------------|--------------------------|
| GET    | `/offline-status`   | Get offline mode status  |
| GET    | `/airgap-status`    | Get air-gap mode status  |

#### Other

| Method | Endpoint   | Description          |
|--------|------------|----------------------|
| GET    | `/policy`  | Get privacy policy   |
| GET    | `/health`  | Privacy health check |

---

## Configuration

### Environment Variables

```bash
# Privacy Modes
SHIVX_OFFLINE_MODE=false
SHIVX_AIRGAP_MODE=false

# Telemetry
SHIVX_TELEMETRY_MODE=standard  # disabled/minimal/standard/full
SHIVX_RESPECT_DNT=true

# GDPR
SHIVX_GDPR_MODE=true

# Data Retention
SHIVX_DATA_RETENTION_DAYS=90
SHIVX_CONVERSATION_RETENTION_DAYS=90
SHIVX_MEMORY_RETENTION_DAYS=365
SHIVX_AUDIT_LOG_RETENTION_DAYS=90
SHIVX_TELEMETRY_RETENTION_DAYS=30
SHIVX_AUTO_PURGE_ENABLED=true
```

### Startup Verification

Privacy modes are verified on application startup:

```
======================================================================
ShivX AI Trading System v1.0.0 (production)
======================================================================
✓ Security hardening engine initialized
✓ Configuration loaded (env: production)
✓ Offline mode: disabled (full network access)
✓ Air-gap mode: disabled
✓ Privacy configuration: {'offline_mode': False, 'airgap_mode': False, 'telemetry_mode': 'standard', 'gdpr_mode': True, 'respect_dnt': True}
✓ Telemetry enabled (mode: standard)
======================================================================
```

---

## Best Practices

### For Users

1. **Review Consent Settings** - Regularly review what data you're consenting to
2. **Use Offline Mode for Privacy** - Enable offline mode when working with sensitive data
3. **Exercise Your Rights** - Request data export annually to verify what's stored
4. **Set DNT Header** - Configure your browser to send Do Not Track headers

### For Developers

1. **Check Consent Before Collection** - Always verify user consent before collecting analytics
2. **Respect Privacy Modes** - Check `offline_mode` and `airgap_mode` before network requests
3. **Log Privacy Operations** - All privacy-sensitive operations should be logged to audit trail
4. **Test with Privacy Modes** - Test your features with offline/airgap modes enabled
5. **Document Data Collection** - Clearly document what data is collected and why

### For Administrators

1. **Enable Privacy Modes in Production** - Use offline mode for sensitive deployments
2. **Monitor Audit Logs** - Regularly review audit logs for privacy violations
3. **Configure Retention Policies** - Set appropriate retention periods for your use case
4. **Regular Audits** - Conduct quarterly privacy audits
5. **User Training** - Train users on privacy features and their rights

---

## Compliance Checklist

- ✅ **Consent Management** - Granular consent with audit trail
- ✅ **Data Minimization** - Only collect necessary data
- ✅ **Transparency** - Clear privacy policy and data disclosure
- ✅ **User Rights** - All GDPR rights implemented
- ✅ **Data Security** - Encryption at rest and in transit
- ✅ **Data Retention** - Configurable retention with auto-purge
- ✅ **Audit Trail** - Complete audit log of privacy operations
- ✅ **Privacy by Design** - Privacy built into architecture
- ✅ **Privacy by Default** - Opt-in for non-essential data collection
- ✅ **Breach Notification** - Incident response procedures

---

## Legal Notice

This privacy implementation is designed to comply with:
- **GDPR** (General Data Protection Regulation - EU)
- **CCPA** (California Consumer Privacy Act)
- **PIPEDA** (Personal Information Protection and Electronic Documents Act - Canada)

**Disclaimer:** This implementation provides technical compliance mechanisms. Legal compliance also requires proper policies, procedures, and organizational measures. Consult with legal counsel for complete compliance.

---

## Support

For privacy-related questions or to exercise your rights:

- **Email:** privacy@shivx.ai
- **Documentation:** https://shivx.ai/privacy
- **Security Issues:** security@shivx.ai

---

Last Updated: 2025-10-28
