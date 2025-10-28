# ShivX Testing Quick Start Guide

## Installation

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Or install specific testing packages
pip install pytest pytest-asyncio pytest-cov pytest-benchmark pytest-xdist pytest-timeout
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_trading_api.py

# Run specific test class
pytest tests/test_trading_api.py::TestTradingStrategies

# Run specific test function
pytest tests/test_trading_api.py::TestTradingStrategies::test_list_strategies_with_auth
```

### Coverage Reports

```bash
# Generate terminal coverage report
pytest tests/ --cov=app --cov=security --cov-report=term

# Generate HTML coverage report
pytest tests/ --cov=app --cov=security --cov-report=html
open htmlcov/index.html

# Generate coverage with missing lines
pytest tests/ --cov=app --cov=security --cov-report=term-missing
```

### Run by Category

```bash
# Unit tests only (fast)
pytest tests/ -m unit

# Integration tests
pytest tests/ -m integration

# Security tests
pytest tests/ -m security

# Performance tests
pytest tests/ -m performance

# E2E tests
pytest tests/ -m e2e

# Exclude slow tests
pytest tests/ -m "not slow"
```

### Parallel Execution

```bash
# Run tests in parallel (uses all CPU cores)
pytest tests/ -n auto

# Run with specific number of workers
pytest tests/ -n 4
```

### Debugging

```bash
# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ -l

# Verbose tracebacks
pytest tests/ --tb=long
```

## Test File Overview

| File | Purpose | Tests | Run Time |
|------|---------|-------|----------|
| `test_trading_api.py` | Trading endpoints | 45+ | ~10s |
| `test_analytics_api.py` | Analytics endpoints | 50+ | ~12s |
| `test_ai_api.py` | AI/ML endpoints | 40+ | ~8s |
| `test_auth_comprehensive.py` | Authentication | 35+ | ~15s |
| `test_guardian_defense.py` | Security system | 38+ | ~25s |
| `test_e2e_workflows.py` | End-to-end flows | 12+ | ~20s |
| `test_security_penetration.py` | Security testing | 25+ | ~15s |
| `test_performance.py` | Performance tests | 15+ | ~30s |

## Quick Test Suites

### Pre-Commit (Fast - ~30s)
```bash
pytest tests/test_auth_comprehensive.py tests/test_trading_api.py -m "not slow"
```

### Security Validation (~40s)
```bash
pytest tests/test_auth_comprehensive.py \
       tests/test_guardian_defense.py \
       tests/test_security_penetration.py
```

### Full API Coverage (~60s)
```bash
pytest tests/test_trading_api.py \
       tests/test_analytics_api.py \
       tests/test_ai_api.py
```

### Complete Suite (~3 minutes)
```bash
pytest tests/ --cov=app --cov=security --cov-report=html
```

## Continuous Integration

### GitHub Actions
Add to `.github/workflows/test.yml`:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/ --cov=app --cov=security --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Writing New Tests

### Test Structure
```python
import pytest
from fastapi import status

@pytest.mark.unit
class TestYourFeature:
    """Test suite for YourFeature"""

    def test_should_do_x_when_y(self, client, test_token):
        """Test that X happens when Y condition is met"""
        # Arrange
        data = {"key": "value"}

        # Act
        response = client.post(
            "/api/endpoint",
            json=data,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["result"] == "expected"
```

### Available Fixtures
- `client`: FastAPI TestClient
- `test_token`: Valid JWT token with READ, WRITE, EXECUTE permissions
- `admin_token`: JWT token with ADMIN permission
- `readonly_token`: JWT token with only READ permission
- `test_settings`: Test configuration
- `guardian_defense`: Guardian defense instance
- `mock_jupiter_client`: Mocked Jupiter DEX client
- `mock_solana_client`: Mocked Solana RPC client

## Coverage Goals

| Component | Target | Status |
|-----------|--------|--------|
| Authentication | >90% | ✅ 88.41% |
| Guardian Defense | >90% | ✅ 96.09% |
| API Routers | 100% | ✅ 100% |
| Database Models | >80% | ⚠️ 60% |
| Security | 100% | ✅ 100% |

## Troubleshooting

### Import Errors
```bash
# Ensure dependencies are installed
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

### Slow Tests
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only fast unit tests
pytest tests/ -m unit
```

### Failed Tests
```bash
# Show detailed output
pytest tests/ -vv

# Show failed test output only
pytest tests/ --tb=short

# Rerun only failed tests
pytest tests/ --lf
```

## Best Practices

1. **Run tests before committing**
   ```bash
   pytest tests/test_auth_comprehensive.py tests/test_trading_api.py -m "not slow"
   ```

2. **Check coverage before PR**
   ```bash
   pytest tests/ --cov=app --cov-report=term-missing
   ```

3. **Run security tests weekly**
   ```bash
   pytest tests/ -m security
   ```

4. **Run performance tests before releases**
   ```bash
   pytest tests/ -m performance
   ```

5. **Run full suite in CI/CD**
   ```bash
   pytest tests/ --cov=app --cov=security --cov-report=xml
   ```

## Maintenance

### Update Test Dependencies
```bash
pip install --upgrade pytest pytest-cov pytest-asyncio
pip freeze > requirements-dev.txt
```

### Clean Test Artifacts
```bash
rm -rf .pytest_cache htmlcov .coverage
```

### Generate Fresh Coverage Report
```bash
rm -f .coverage
pytest tests/ --cov=app --cov=security --cov-report=html
```

---

**For detailed test results, see TEST_SUITE_REPORT.md**
