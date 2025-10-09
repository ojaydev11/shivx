# ShivX Troubleshooting Runbook

**Last Updated:** October 9, 2025  
**Maintained by:** Release QA Team

---

## Table of Contents

1. [Service Won't Start](#service-wont-start)
2. [GPU/Voice Issues](#gpuvoice-issues)
3. [Browser Automation Fails](#browser-automation-fails)
4. [Memory/Database Corruption](#memorydatabase-corruption)
5. [Performance Degradation](#performance-degradation)
6. [Network/Offline Mode Issues](#networkoffline-mode-issues)

---

## Service Won't Start

### Symptom: Python errors on launch

**Diagnosis:**
```powershell
python shivx_runner.py --mode readiness
```

**Common Causes:**

**1. Missing environment variables**
```
Error: validate_env_or_die: missing ADMIN_TOKEN
```
**Fix:** Edit `.env` and set all required values:
- `ADMIN_TOKEN` (64+ chars, random)
- `ENCRYPTION_KEY` (base64, 32 bytes)
- `JWT_SECRET` (32+ chars, random)

**2. Virtual environment not activated**
```
ModuleNotFoundError: No module named 'fastapi'
```
**Fix:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**3. Port already in use**
```
Error: [Errno 10048] Only one usage of each socket address...
```
**Fix:** Change port in `.env`:
```bash
GUI_PORT=8052  # or any free port
AGI_ADMIN_PORT=5100
```

Or find and kill the process:
```powershell
netstat -ano | findstr :8051
taskkill /PID <PID> /F
```

---

## GPU/Voice Issues

### Symptom: Voice pipeline fails, STT/TTS errors

**Diagnosis:**
```powershell
python -c "import agents.voice.stt_vosk; print('STT OK')"
python -c "import pyttsx3; print('TTS OK')"
```

**Common Causes:**

**1. Vosk model not installed**
```
FileNotFoundError: models/vosk/en
```
**Fix:** Download Vosk model:
```powershell
# Create directory
mkdir models\vosk\en

# Download model (manual or script)
# See: https://alphacephei.com/vosk/models
# Extract to models/vosk/en/
```

**2. pyttsx3 not working (Windows)**
```
RuntimeError: Could not find any TTS engine
```
**Fix:** Ensure Windows SAPI voices installed:
- Open: Control Panel → Speech Recognition → Text to Speech
- Verify at least one voice is available
- Restart Python

**3. GPU driver issues (if using GPU acceleration)**
```
CUDA not available
```
**Fix:**
- Update GPU drivers
- Or: Disable GPU features in `.env`:
```bash
USE_CPU_ONLY=1
```

---

## Browser Automation Fails

### Symptom: Playwright errors, headless browser crashes

**Diagnosis:**
```powershell
python -m playwright install --help
python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"
```

**Common Causes:**

**1. Playwright browsers not installed**
```
Error: Executable doesn't exist at ...
```
**Fix:**
```powershell
python -m playwright install chromium
```

**2. Allowlist blocking navigation**
```
Error: Navigation blocked by allowlist
```
**Fix:** Add domain to `ALLOWLIST_DOMAINS` in `.env`:
```bash
ALLOWLIST_DOMAINS=github.com,docs.python.org,your-domain.com
```

**3. Headless mode failing on Windows**
```
Error: Browser closed unexpectedly
```
**Fix:** Try headed mode for debugging:
```python
# In browser_agent.py or test code
browser = playwright.chromium.launch(headless=False)
```

Or: Update Playwright:
```powershell
pip install --upgrade playwright
python -m playwright install chromium
```

---

## Memory/Database Corruption

### Symptom: SQLite errors, corrupted vector DB

**Diagnosis:**
```powershell
# Check DB files
python -c "import sqlite3; conn = sqlite3.connect('data/personal_memory.db'); print('DB OK')"
```

**Common Causes:**

**1. Locked database file**
```
OperationalError: database is locked
```
**Fix:**
```powershell
# Find and kill processes holding the file
# Windows Resource Monitor → CPU → Associated Handles → search "personal_memory.db"

# Or restart the system
```

**2. Corrupted ChromaDB**
```
Error: Chroma initialization failed
```
**Fix:** Rebuild vector index:
```powershell
# Backup first!
mv memory\chroma_db memory\chroma_db.backup

# Restart ShivX (will recreate index)
python shivx_runner.py --mode gui
```

**3. Disk full**
```
Error: No space left on device
```
**Fix:**
- Free up disk space (delete `logs/`, `var/runs/` old data)
- Or configure retention:
```bash
# In config/settings.yaml
retention:
  artifacts_days: 7
  logs_days: 14
```

---

## Performance Degradation

### Symptom: Slow responses, high CPU/RAM usage

**Diagnosis:**
```powershell
# Check resource usage
Get-Process python | Select-Object CPU, WorkingSet, Path

# Run profiling
.\scripts\profiling.ps1 --component orchestrator
```

**Common Causes:**

**1. Memory leak in long-running process**
**Fix:** Restart agent service periodically (via cron/Task Scheduler)

**2. Large queue backlog**
```powershell
# Check queue sizes
python -c "from orchestrator.queue_manager import queue_manager; print(queue_manager.get_all_queues_status())"
```
**Fix:** Clear old completed items:
```python
from orchestrator.queue_manager import queue_manager
queue_manager.clear_completed("sewago", older_than_days=7)
```

**3. Vector DB index too large**
**Fix:** Run memory cleanup:
```python
from core.vector_memory import get_vector_memory_system
vm = get_vector_memory_system()
vm.cleanup_old_memories(days_threshold=90, importance_threshold=0.5)
```

---

## Network/Offline Mode Issues

### Symptom: Egress blocked when not intended

**Diagnosis:**
```powershell
# Check offline mode status
python -c "import os; print('USE_OFFLINE=' + os.getenv('USE_OFFLINE', '0'))"
python -c "from core.security.net_guard import is_egress_blocking_active; print(is_egress_blocking_active())"
```

**Common Causes:**

**1. Offline mode enabled unintentionally**
**Fix:** Edit `.env`:
```bash
USE_OFFLINE=0
```

**2. Allowlist too restrictive**
**Fix:** Add required domains:
```bash
ALLOWLIST_DOMAINS=github.com,pypi.org,your-api.com
```

**3. Strict loopback blocking local services**
```
Error: Offline mode: loopback port 5432 not in allowlist
```
**Fix:** Add port to allowlist or disable strict mode:
```bash
STRICT_LOOPBACK=0
# OR
LOOPBACK_PORT_ALLOWLIST=8000,8080,5432,6379
```

---

## Emergency Procedures

### Kill Switch Activation

**When:** Runaway autonomous agent, security incident

**How:**
```powershell
# Create panic flag
echo $null > var\runtime\panic.flag

# Verify agent stops
python -c "from orchestrator.agent_service import AgentService; print(AgentService().get_status())"
```

**Deactivate:**
```powershell
rm var\runtime\panic.flag
```

### Full System Reset

**When:** Unrecoverable state, testing fresh install

**How:**
```powershell
# CAUTION: This deletes all data!

# Stop all services
taskkill /F /IM python.exe

# Backup if needed
mv data data.backup
mv memory memory.backup
mv logs logs.backup

# Clean slate
rm -r data, memory, logs, var

# Restart bootstrap
.\scripts\dev_bootstrap.ps1
```

---

## Getting Help

1. Check logs: `logs/`
2. Review audit trail: `var/security/`
3. Run diagnostics: `python shivx_runner.py --mode readiness`
4. Consult docs: `README.md`, `ARCHITECTURE.md`
5. Check known issues: `release/STATUS.md`

**Still stuck?** File an issue with:
- OS version
- Python version
- Error message (full traceback)
- Steps to reproduce
- Relevant log files

