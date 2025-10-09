# ShivX 5-Minute Quickstart Guide

**Goal:** Get ShivX running from zero in ≤10 minutes (Gate G7)

**Prerequisites:**
- Windows 10/11 (PowerShell 5.1+)
- Python 3.10+ installed
- Git installed
- 2GB+ free RAM
- Internet connection (for initial setup only)

---

## Step 1: Clone & Navigate (30 seconds)

```powershell
git clone <your-repo-url> shivx
cd shivx
git checkout release/shivx-hardening-001
```

---

## Step 2: Bootstrap Environment (5-8 minutes)

**Run the automated bootstrap:**

```powershell
.\scripts\dev_bootstrap.ps1
```

**What this does:**
- ✅ Checks Python 3.10+ is installed
- ✅ Creates virtual environment (`.venv`)
- ✅ Installs all dependencies from `requirements.txt`
- ✅ Installs Playwright browsers (chromium)
- ✅ Creates `.env` from `env.example`
- ✅ Creates required directories (`var/`, `logs/`, `data/`, `memory/`)
- ✅ Runs readiness check

---

## Step 3: Configure Secrets (2 minutes)

**CRITICAL:** Update `.env` with secure values:

```powershell
# Open .env in your editor
notepad .env
```

**Replace these values:**

```bash
# Generate a 64-character random token
ADMIN_TOKEN=<your-64-char-token-here>

# Generate a base64-encoded 32-byte encryption key
ENCRYPTION_KEY=<your-base64-32-byte-key-here>

# Generate a 32+ character JWT secret
JWT_SECRET=<your-strong-secret-here>
```

**Quick generators (PowerShell):**

```powershell
# ADMIN_TOKEN (64 chars)
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 64 | % {[char]$_})

# ENCRYPTION_KEY (base64, 32 bytes)
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))

# JWT_SECRET (32 chars)
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_})
```

---

## Step 4: Launch ShivX (30 seconds)

**Option A: GUI Dashboard** (Recommended for first-time users)

```powershell
python shivx_runner.py --mode gui
```

Then open: **http://127.0.0.1:8051/dashboard**

**Option B: Main API Server** (For developers)

```powershell
python shivx_runner.py --mode prod
```

API at: **http://127.0.0.1:5099**

**Option C: Dev Mode** (With auto-reload)

```powershell
python shivx_runner.py --mode dev
```

---

## Step 5: Verify Installation (30 seconds)

**Run readiness check:**

```powershell
python shivx_runner.py --mode readiness
```

Expected output:
```json
{
  "boot_ok": true,
  "offline": false,
  "mode": "readiness",
  "readiness_exit": 0
}
```

**Run smoke tests:**

```powershell
pytest tests/test_smoke.py -v
```

---

## Common Issues & Fixes

### Issue: "Python not found"
**Fix:** Install Python 3.10+ from https://www.python.org/ and add to PATH

### Issue: "ModuleNotFoundError: No module named 'fastapi'"
**Fix:** Ensure virtual environment is activated:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Issue: "validate_env_or_die: missing ADMIN_TOKEN"
**Fix:** Edit `.env` and set secure values (see Step 3)

### Issue: "Playwright browsers not installed"
**Fix:**
```powershell
python -m playwright install chromium
```

### Issue: "Port 8051 already in use"
**Fix:** Change port in `.env`:
```bash
GUI_PORT=8052
```

---

## Next Steps

**1. Explore the Dashboard:**
- Navigate to http://127.0.0.1:8051/dashboard
- Authenticate with your `ADMIN_TOKEN`
- Explore: Queue Management, Agent Status, Logs, Settings

**2. Run Your First Goal:**
```python
from orchestrator.queue_manager import queue_manager, Priority

# Add a goal to the queue
goal_id = queue_manager.add_goal(
    goal="Draft tomorrow's plan",
    project="sewago",
    priority=Priority.NORMAL
)

# Start the agent service
from orchestrator.agent_service import AgentService
service = AgentService()
service.start()
```

**3. Enable Optional Features:**

Edit `.env` to enable:
```bash
ENABLE_VOICE=1          # Voice (STT/TTS)
ENABLE_BROWSER_AGENT=1  # Browser automation
ENABLE_AUTONOMY=1       # Autonomous mode
```

**4. Read Full Documentation:**
- `README.md` - System overview
- `ARCHITECTURE.md` - Technical deep-dive
- `release/STATUS.md` - Production hardening status
- `release/artifacts/wirecheck_report.md` - Dependency graph
- `release/runbooks/` - Troubleshooting guides

---

## Production Deployment

**For production environments:**

1. **Set strong secrets** (ADMIN_TOKEN, ENCRYPTION_KEY, JWT_SECRET)
2. **Enable HSTS** if using HTTPS: `ENABLE_HSTS=1`
3. **Configure firewall** if exposing API externally
4. **Enable offline mode** for maximum security: `USE_OFFLINE=1`
5. **Review feature flags** in `.env` and disable unused subsystems
6. **Run full test battery:** `.\scripts\run_all_tests.ps1`
7. **Review security report:** `.\scripts\security_scan.ps1`

---

## Support & Resources

- **Documentation:** `docs/`
- **Troubleshooting:** `release/runbooks/`
- **Architecture:** `ARCHITECTURE.md`
- **Security:** `SECURITY.md`
- **Operations:** `OPERATIONS.md`

**Need help?** Check `release/runbooks/troubleshooting.md`

---

**Total Time:** 5-10 minutes ✅

**Status:** Ready for development! 🚀

