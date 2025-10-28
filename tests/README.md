# ShivX Test Suite

## Overview

Comprehensive test suite for the ShivX AI platform, ensuring reliability, security, and performance across all components.

## Test Coverage

### Security Tests (90%+ coverage)
- **Prompt Injection** (`test_prompt_injection.py`) - 50+ attack vectors
- **DLP** (`test_dlp.py`) - PII/secrets detection and redaction
- **Content Moderation** (`test_content_moderation.py`) - Harmful content filtering
- **Authentication** (`test_auth_comprehensive.py`) - JWT, permissions, MFA
- **Security Hardening** (`test_security_hardening.py`) - Production security
- **Penetration Testing** (`test_security_penetration.py`) - Attack simulations

### Multi-Agent Framework (85%+ coverage)
- **Intent Router** (`test_intent_router.py`) - 30+ tests for intent classification
- **Task Graph** (`test_task_graph.py`) - 40+ tests for orchestration
- **Agents** (`test_agents.py`) - 60+ tests for all agent types
  - Planner Agent
  - Researcher Agent
  - Coder Agent
  - Operator Agent
  - Finance Agent
  - Safety Agent
- **Agent Handoffs** - Cross-agent communication and state transfer
- **Resource Governance** - Quota enforcement and monitoring

### Memory & RAG (80%+ coverage)
- **Vector Store** (`test_vector_store.py`) - 40+ tests for semantic search
- **Long-term Memory** (`test_long_term_memory.py`) - Persistent storage
- **RAG Pipeline** - Retrieval-augmented generation

### Privacy Features (90%+ coverage)
- **GDPR Compliance** (`test_gdpr.py`) - All GDPR rights implemented
  - Right to Access (data export)
  - Right to Erasure (forget-me)
  - Right to Rectification
  - Data portability
- **Offline Mode** (`test_offline_mode.py`) - Network isolation
- **Air-gap Mode** (`test_airgap.py`) - Complete isolation
- **Consent Management** (`test_consent.py`) - User consent tracking

### API Tests (80%+ coverage)
- **AI API** (`test_ai_api.py`) - LLM integration endpoints
- **Trading API** (`test_trading_api.py`) - DEX trading operations
- **Analytics API** (`test_analytics_api.py`) - Metrics and reporting
- **WebSocket** (`test_websocket.py`) - Real-time communication

### Integration Tests
- **Voice** (`test_voice.py`) - STT/TTS integration
- **GitHub Integration** (`test_github_integration.py`)
- **Google Integration** (`test_google_integration.py`)
- **E2E Workflows** (`test_e2e_workflows.py`) - Complete user journeys

### Performance Tests
- **Cache Performance** (`test_cache_performance.py`) - Redis caching
- **Performance Benchmarks** (`test_performance.py`) - API latency, throughput
- **ML Models** (`test_ml_models.py`) - Model inference performance

## Running Tests

### All Tests
```bash
pytest
```

### With Coverage Report
```bash
pytest --cov=app --cov=core --cov=utils --cov=integrations --cov-report=html --cov-report=term-missing
```

### Specific Test Category
```bash
# Security tests only
pytest tests/test_security_*.py

# Multi-agent tests
pytest tests/test_intent_router.py tests/test_task_graph.py tests/test_agents.py

# Privacy tests
pytest tests/test_gdpr.py tests/test_offline_mode.py tests/test_consent.py

# E2E tests
pytest tests/test_e2e_*.py

# Performance tests
pytest tests/test_performance*.py -m performance
```

### By Marker
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Security tests
pytest -m security

# Performance tests
pytest -m performance

# E2E tests
pytest -m e2e

# Exclude slow tests
pytest -m "not slow"
```

### Parallel Execution
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run with 4 workers
pytest -n 4
```

### Watch Mode (Development)
```bash
# Install pytest-watch
pip install pytest-watch

# Watch for changes and re-run tests
ptw -- --tb=short
```

## Test Organization

### Directory Structure
```
tests/
├── conftest.py                     # Shared fixtures
├── __init__.py
│
├── test_ai_api.py                  # API endpoint tests
├── test_analytics_api.py
├── test_trading_api.py
├── test_websocket.py
│
├── test_auth_comprehensive.py      # Authentication & authorization
├── test_security_hardening.py      # Security controls
├── test_security_penetration.py    # Penetration tests
├── test_security_production.py
│
├── test_prompt_injection.py        # Security: Prompt injection
├── test_dlp.py                     # Security: Data loss prevention
├── test_content_moderation.py      # Security: Content moderation
├── test_guardian_defense.py        # Security: Runtime defense
│
├── test_intent_router.py           # Multi-agent: Intent routing
├── test_task_graph.py              # Multi-agent: Task orchestration
├── test_agents.py                  # Multi-agent: All agents
│
├── test_vector_store.py            # Memory: Vector database
├── test_long_term_memory.py        # Memory: Persistent storage
│
├── test_gdpr.py                    # Privacy: GDPR compliance
├── test_offline_mode.py            # Privacy: Offline mode
├── test_airgap.py                  # Privacy: Air-gap mode
├── test_consent.py                 # Privacy: Consent management
│
├── test_github_integration.py      # Integrations
├── test_google_integration.py
├── test_voice.py
│
├── test_e2e_workflows.py           # End-to-end tests
├── test_integration.py             # Integration tests
│
├── test_performance.py             # Performance benchmarks
├── test_cache_performance.py
├── test_ml_models.py
│
└── test_database.py                # Database operations
```

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (multiple components)
- `@pytest.mark.e2e` - End-to-end tests (full workflows)
- `@pytest.mark.security` - Security-focused tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.slow` - Slow tests (can be skipped in development)
- `@pytest.mark.asyncio` - Async tests (requires pytest-asyncio)

## Fixtures

### Common Fixtures (conftest.py)

**Settings & Configuration:**
- `test_settings` - Test settings with safe defaults
- `override_settings` - Override app settings for tests

**API Client:**
- `app` - FastAPI app instance
- `client` - Synchronous test client
- `auth_client` - Authenticated test client

**Authentication:**
- `test_token` - Valid JWT token (user permissions)
- `admin_token` - Admin JWT token
- `readonly_token` - Read-only JWT token
- `expired_token` - Expired JWT token
- `invalid_token` - Invalid JWT token

**Mock Data:**
- `mock_trade_signal` - Trading signal data
- `mock_position` - Trading position data
- `mock_market_data` - Market data
- `mock_ml_model` - ML model metadata

**Mock Services:**
- `mock_jupiter_client` - Mock Jupiter DEX client
- `mock_solana_client` - Mock Solana RPC client

**Utilities:**
- `freeze_time` - Freeze time for deterministic tests
- `benchmark_settings` - Performance benchmark configuration

## Writing Tests

### Test Structure

```python
import pytest
from typing import Dict, Any

@pytest.mark.unit  # Add appropriate marker
class TestFeatureName:
    """Test suite for FeatureName"""

    def test_basic_functionality(self, fixture_name):
        """Test: Basic functionality works correctly"""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result.success
        assert result.value == expected_value

    def test_error_handling(self, fixture_name):
        """Test: Errors are handled gracefully"""
        with pytest.raises(ExpectedException):
            function_under_test(invalid_input)

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test: Async functionality works"""
        result = await async_function()
        assert result is not None
```

### Best Practices

1. **Test Naming:**
   - Use descriptive names: `test_user_can_login_with_valid_credentials`
   - Follow pattern: `test_<what>_<when>_<expected>`

2. **Test Organization:**
   - Group related tests in classes
   - Use fixtures for common setup
   - One assertion per test (when possible)

3. **Test Independence:**
   - Tests should not depend on each other
   - Clean up after tests (use fixtures with yield)
   - Use isolated test data

4. **Async Tests:**
   - Mark with `@pytest.mark.asyncio`
   - Use `async def` for test functions
   - Use `await` for async calls

5. **Mock External Dependencies:**
   - Mock external APIs
   - Mock database calls in unit tests
   - Use real services only in integration tests

## Coverage Requirements

### Target Coverage
- **Overall:** ≥80%
- **Critical Paths:** ≥90% (security, auth, trading)
- **New Features:** 100%

### Checking Coverage
```bash
# Generate HTML report
pytest --cov=app --cov=core --cov=utils --cov-report=html

# Open report
open htmlcov/index.html

# Terminal report with missing lines
pytest --cov=app --cov=core --cov=utils --cov-report=term-missing

# Fail if coverage below 80%
pytest --cov=app --cov=core --cov=utils --cov-fail-under=80
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Nightly builds

### CI Configuration (.github/workflows/test.yml)
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      - name: Run tests
        run: pytest --cov --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install package in editable mode
pip install -e .
```

**Async Test Errors:**
```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Ensure asyncio marker is used
@pytest.mark.asyncio
async def test_async_function():
    pass
```

**Fixture Not Found:**
```bash
# Check conftest.py is in tests/ directory
# Ensure fixture is properly decorated with @pytest.fixture
```

**Test Timeout:**
```bash
# Increase timeout for slow tests
pytest --timeout=300

# Or mark specific tests
@pytest.mark.timeout(60)
def test_slow_operation():
    pass
```

## Performance Benchmarks

### API Latency Targets
- **p50:** <50ms
- **p95:** <200ms
- **p99:** <500ms

### Throughput Targets
- **Concurrent requests:** 100+ req/s
- **Vector search:** <100ms for 10k vectors
- **RAG pipeline:** <500ms end-to-end

### Running Benchmarks
```bash
# Run performance tests
pytest tests/test_performance*.py -m performance -v

# Generate performance report
pytest tests/test_performance.py --benchmark-json=benchmark.json
```

## Security Testing

### Penetration Testing
```bash
# Run security penetration tests
pytest tests/test_security_penetration.py -v

# Test prompt injection defenses
pytest tests/test_prompt_injection.py -v

# Test DLP
pytest tests/test_dlp.py -v
```

### Attack Vectors Tested
- **Prompt Injection:** 50+ attack patterns
- **SQL Injection:** Parameterized queries
- **XSS:** Input sanitization
- **CSRF:** Token validation
- **Rate Limiting:** Request throttling
- **Authentication:** JWT security
- **Authorization:** Permission checks

## Contributing

### Adding New Tests

1. **Create test file** following naming convention: `test_<feature>.py`
2. **Import necessary modules** and fixtures
3. **Organize tests** into classes by functionality
4. **Add appropriate markers** (@pytest.mark.unit, etc.)
5. **Document tests** with clear docstrings
6. **Run tests locally** before committing
7. **Check coverage** and add missing tests

### Test Review Checklist
- [ ] Tests are independent and isolated
- [ ] Tests have clear, descriptive names
- [ ] Tests cover happy path and edge cases
- [ ] Tests handle error conditions
- [ ] Tests use appropriate fixtures
- [ ] Tests are properly marked
- [ ] Coverage meets requirements (≥80%)
- [ ] Tests pass in CI

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## Support

For questions or issues with tests:
1. Check this README
2. Review conftest.py for available fixtures
3. Look at similar existing tests
4. Open an issue with test failure details
