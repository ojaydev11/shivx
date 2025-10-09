# ShivX Test Baseline Report – Phase II

**Generated:** October 9, 2025  
**Branch:** `release/shivx-hardening-001`  
**Phase:** II – Test Foundation  
**Engineer:** A2 (Unit/Integration Engineer)

---

## Executive Summary

**Test Collection Results:**
- ✅ **570 tests discovered** across test suite
- ⚠️ **81 collection errors** (import/module issues)
- **489 potentially runnable tests** (570 - 81)
- **Current pass/fail status:** NOT YET MEASURED (need to fix imports first)

**Test Distribution by Category:**
- Unit tests: ~350 (estimated)
- Integration tests: ~120 (estimated)  
- E2E tests: ~30 (estimated)
- Smoke tests: ~40 (estimated)
- Security tests: ~30 (estimated)

**Critical Finding:** Many tests fail to import due to missing modules or architectural drift. Immediate action required to align tests with current codebase structure.

---

## 1. Test Collection Issues

### 1.1 Missing Module Dependencies

**Impact: 81 test files cannot run**

| Missing Module | Affected Tests | Issue |
|----------------|----------------|-------|
| `fastapi` | 10+ test files | **CRITICAL** - FastAPI is in requirements.txt but tests can't import TestClient |
| `utils.jsonx` | 5+ test files | Module doesn't exist in current codebase |
| `src.core.*` modules | 30+ test files | Old architecture - tests reference `src/` instead of current structure |
| `ccxt` (crypto exchange lib) | 2 test files | Optional dependency not in requirements.txt |
| `agents.voice` | 1 test file | Import path mismatch |

### 1.2 pytest Configuration Issues

**Fixed in updated `pytest.ini`:**
- ✅ Registered custom markers (`asyncio`, `slow`, `security`, `integration`, etc.)
- ✅ Added asyncio_mode = auto
- ✅ Configured coverage reporting to `release/artifacts/coverage_report.html`
- ✅ Added test timeouts (300s default)
- ✅ Strict marker enforcement

### 1.3 Architectural Drift

**Pattern:** Many tests reference `src/` module structure that no longer exists.

**Examples:**
```python
# OLD (failing)
from src.core.observability.events import emit_event
from src.api.agi import create_app
from src.core.startup_orchestrator import start_all

# NEW (current architecture)
from app.obs.metrics import emit_event  # or similar
from app.main import app
from orchestrator.agent_service import AgentService
```

**Root Cause:** Codebase was refactored from `src/` structure to flat `app/`, `core/`, `orchestrator/` structure, but tests weren't updated.

---

## 2. Test Inventory by Module

### 2.1 Core Module Tests

| Module | Tests Found | Status | Coverage Goal |
|--------|-------------|--------|---------------|
| `core/agent/` (planner, goal_runner, blackboard) | 15 | ⚠️ Import errors (utils.jsonx) | 90% |
| `core/security/` (env_guard, net_guard) | 8 | ✅ Likely runnable | 95% |
| `core/personal_brain.py` | 5 | ✅ Likely runnable | 85% |
| `core/vector_memory.py` | 3 | ✅ Likely runnable | 80% |
| `core/skills/` (registry, tools) | 20 | ⚠️ Mixed (some import errors) | 90% |
| `core/obs/` | 5 | ❌ Missing src.core.observability | 80% |

### 2.2 Orchestrator Module Tests

| Module | Tests Found | Status | Coverage Goal |
|--------|-------------|--------|---------------|
| `orchestrator/agent_service.py` | 8 | ⚠️ Import errors | 90% |
| `orchestrator/queue_manager.py` | 10 | ⚠️ Import errors (utils.jsonx) | 90% |
| `orchestrator/chaos.py` | 5 | ✅ Likely runnable | 85% |
| `orchestrator/watchdog.py` | 6 | ⚠️ Import errors | 85% |

### 2.3 App/API Module Tests

| Module | Tests Found | Status | Coverage Goal |
|--------|-------------|--------|---------------|
| `app/main.py` (FastAPI app) | 15 | ❌ Can't import fastapi.TestClient | 80% |
| `app/routes/*` (all routers) | 50+ | ❌ Can't import fastapi.TestClient | 75% |
| `app/obs/*` (observability) | 10 | ❌ src.core.observability missing | 80% |
| `app/security/*` (auth, headers) | 12 | ⚠️ Mixed | 90% |
| `app/services/*` | 15 | ⚠️ Mixed | 80% |

### 2.4 Agent Module Tests

| Module | Tests Found | Status | Coverage Goal |
|--------|-------------|--------|---------------|
| `agents/voice/` | 5 | ⚠️ Import path issues | 75% |
| `agents/browser/` | 10 | ⚠️ ToolSchema parameter error | 80% |

### 2.5 Integration & E2E Tests

| Category | Tests Found | Status | Notes |
|----------|-------------|--------|-------|
| Integration tests | 30+ | ⚠️ src.* import errors | Cross-module workflows |
| E2E (Playwright) | 0 | ❌ **MISSING** | Need to create |
| Smoke tests | 40 | ⚠️ src.* import errors | Quick validation |

---

## 3. Root Cause Analysis

### Issue 1: FastAPI Import Failures

**Symptom:** `ModuleNotFoundError: No module named 'fastapi'`

**Diagnosis:** FastAPI IS in requirements.txt (v0.115.5), so this is likely a Python path or virtual environment issue.

**Fix:** Ensure tests run with virtual environment activated.

### Issue 2: utils.jsonx Missing

**Symptom:** `ModuleNotFoundError: No module named 'utils.jsonx'`

**Diagnosis:** Module doesn't exist in current codebase. Likely removed during refactoring.

**Fix Options:**
1. Create stub `utils/jsonx.py` with `dumps = json.dumps, loads = json.loads`
2. Update tests to use standard `json` module
3. Search codebase for any actual jsonx usage

**Recommendation:** Option 1 (quick stub) for immediate fix, then Option 2 (proper refactor).

### Issue 3: src.* Import Paths

**Symptom:** `ModuleNotFoundError: No module named 'src.core.*'`

**Diagnosis:** Old test files reference deprecated `src/` structure.

**Fix:** Bulk find-replace in test files:
```bash
# sed or PowerShell equivalent
src.api.agi -> app.main
src.core.observability -> app.obs
src.core.monitoring -> app.monitoring
src.core.startup_orchestrator -> orchestrator.agent_service
```

**Recommendation:** Create migration script to update all test imports.

### Issue 4: ToolSchema Parameter Error

**Symptom:** `TypeError: ToolSchema.__init__() got an unexpected keyword argument 'parameters'`

**Diagnosis:** ToolSchema signature changed but test/code not updated.

**Fix:** Review `core/skills/tool_base.py` and update `browser_tools.py`.

---

## 4. Immediate Action Plan

### Phase II-A: Fix Test Imports (Priority 1)

**Duration:** 1-2 hours

1. ✅ Create `utils/jsonx.py` stub
2. ✅ Create test import migration script
3. ✅ Run migration on all test files
4. ✅ Fix ToolSchema parameter issue in `browser_tools.py`
5. ✅ Verify virtual environment activation for pytest

### Phase II-B: Measure Baseline Coverage (Priority 1)

**Duration:** 30 minutes

1. Run tests with coverage: `pytest --cov --cov-report=html`
2. Identify modules with <50% coverage
3. Document critical path coverage gaps
4. Export baseline metrics to JSON

### Phase II-C: Expand Unit Tests (Priority 2)

**Duration:** 3-4 hours

**Focus areas (high-impact, low-coverage):**
1. `orchestrator/agent_service.py` - panic flag, queue processing, atomic writes
2. `orchestrator/queue_manager.py` - retry logic, priority ordering, persistence
3. `core/agent/goal_runner.py` - blackboard resolution, kill switch, artifact saving
4. `core/agent/planner.py` - plan validation, risk scoring, critical path
5. `core/personal_brain.py` - encryption, pattern tracking, bond meter
6. `core/security/net_guard.py` - egress blocking, loopback allowlist

**Target:** 90% coverage on these critical modules.

### Phase II-D: Create E2E Tests (Priority 2)

**Duration:** 2-3 hours

**Test scenarios:**
1. **GUI Dashboard Launch**
   - Start GUI mode → open dashboard → verify auth → shutdown
2. **Agent Workflow**
   - Add goal to queue → agent processes → verify artifacts → check audit log
3. **Kill Switch**
   - Start agent → activate panic flag → verify immediate stop
4. **Offline Mode**
   - Enable USE_OFFLINE=1 → attempt network request → verify blocked
5. **Voice Pipeline** (if Vosk available)
   - STT → process → TTS → verify audio output

**Tool:** Playwright for GUI, standard pytest for CLI

---

## 5. Coverage Targets

### Gate G1 Requirements

**Critical Path Coverage:** ≥90%

Critical paths defined as:
- Boot sequence (`shivx_runner.py` → `app.main` startup)
- Security guards (`env_guard`, `net_guard`)
- Agent execution loop (`agent_service` → `goal_runner` → `planner`)
- Queue management (add, get, retry, persist)
- Tool execution (`registry` → `tool.execute()`)
- Memory systems (personal brain, vector memory)

**Overall Coverage:** ≥75%

### Current Baseline (Estimated)

**NOTE:** Actual measurement pending import fixes.

**Estimated current coverage (based on test inventory):**
- **Critical path:** ~40-50% (NEEDS EXPANSION)
- **Overall:** ~55-60% (NEEDS EXPANSION)

**Gap to close:**
- Critical path: +40-50 percentage points
- Overall: +15-20 percentage points

---

## 6. Test Quality Standards

All new/updated tests must:

1. **Follow AAA pattern:** Arrange, Act, Assert
2. **Use fixtures:** No test-level setup/teardown spaghetti
3. **Mock external I/O:** No real network, filesystem (unless integration test)
4. **Have descriptive names:** `test_queue_manager_retries_failed_items_with_exponential_backoff`
5. **Include docstrings:** Explain what/why, not how
6. **Fast by default:** Unit tests <100ms, integration <1s, E2E <10s
7. **Deterministic:** No random failures, no time dependencies
8. **Isolated:** Tests can run in any order, in parallel

### Example: Good Test

```python
import pytest
from orchestrator.queue_manager import QueueManager, Priority

@pytest.fixture
def queue_manager(tmp_path):
    """Isolated queue manager with temp storage."""
    qm = QueueManager(base_dir=tmp_path)
    yield qm
    qm.close()

def test_queue_manager_retries_failed_items_with_exponential_backoff(queue_manager):
    """
    GIVEN a queue item that has failed twice
    WHEN retry_item() is called
    THEN retry_count increments and next_retry_after uses exponential backoff
    """
    # Arrange
    item_id = queue_manager.add_goal("test goal", "project1", Priority.NORMAL)
    queue_manager.mark_item_failed(item_id, "project1", "error1")
    queue_manager.retry_item(item_id, "project1")
    queue_manager.mark_item_failed(item_id, "project1", "error2")
    
    # Act
    queue_manager.retry_item(item_id, "project1")
    
    # Assert
    item = queue_manager.get_item_by_id(item_id, "project1")
    assert item.retry_count == 2
    assert item.next_retry_after > time.time() + 60  # At least 60s backoff
```

---

## 7. Test Scripts

### scripts/run_all_tests.ps1

**Purpose:** Execute full test battery with coverage

```powershell
# Run all tests with coverage
pytest --cov --cov-report=html --cov-report=term -v

# Check coverage thresholds
# (Will be added once baseline is established)
```

### scripts/run_unit_tests.ps1

**Purpose:** Fast unit test subset

```powershell
pytest tests/ -m "not slow and not integration and not e2e" -v
```

### scripts/run_integration_tests.ps1

**Purpose:** Integration tests only

```powershell
pytest tests/ -m integration -v
```

---

## 8. Next Steps

**Immediate (Phase II-A):**
1. ✅ Create `utils/jsonx.py` stub
2. ✅ Migrate test imports (src.* → current structure)
3. ✅ Fix ToolSchema issue
4. ✅ Re-run test collection → should see 0 import errors

**Short-term (Phase II-B/C):**
1. Run coverage baseline
2. Expand unit tests for critical paths
3. Target 90% coverage on orchestrator + core.agent modules

**Medium-term (Phase II-D):**
1. Create E2E test suite (Playwright)
2. Validate all user journeys work end-to-end

**Sign-off criteria for Phase II:**
- ✅ 0 collection errors
- ✅ ≥90% critical path coverage
- ✅ ≥75% overall coverage
- ✅ Coverage report exported to `release/artifacts/coverage_report.html`
- ✅ E2E report exported to `release/artifacts/e2e_report.html`

---

## 9. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Test import fixes break existing passing tests | MEDIUM | MEDIUM | Run tests before/after migration, compare |
| Coverage target too aggressive given codebase size | LOW | MEDIUM | Focus on critical path first, lower overall target if needed |
| E2E tests flaky due to timing issues | MEDIUM | LOW | Use explicit waits, retry logic, isolated test environments |
| Playwright setup complex on Windows | LOW | LOW | Document exact setup steps, provide troubleshooting guide |

---

## 10. Success Metrics

**Phase II Complete when:**
- Gate G1 Coverage: ✅ ≥90% critical, ≥75% overall
- All import errors: ✅ Fixed
- Test execution time: ✅ Full battery <10 minutes
- E2E scenarios: ✅ 5 critical journeys passing
- Documentation: ✅ Coverage report + E2E report published

---

**Prepared by:** A2 (Unit/Integration Engineer)  
**Reviewed by:** A0 (Release Captain)  
**Date:** October 9, 2025  
**Status:** ⚠️ IN PROGRESS - Import fixes needed before coverage measurement

