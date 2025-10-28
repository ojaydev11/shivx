"""
Pytest configuration and fixtures for ShivX tests
Comprehensive test fixtures for API testing, database, authentication, and mocking
"""

import pytest
import os
import asyncio
from typing import Generator, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from fastapi.testclient import TestClient
from config.settings import Settings, TradingMode, Environment
from core.security.hardening import Permission
from app.dependencies.auth import create_access_token, TokenData


# ============================================================================
# Test Settings
# ============================================================================

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Provide test settings with safe defaults"""
    return Settings(
        env=Environment.LOCAL,
        skip_auth=False,  # Always test auth
        trading_mode=TradingMode.PAPER,
        secret_key="test_secret_key_32_chars_minimum_required_for_security",
        jwt_secret="test_jwt_secret_32_chars_minimum_required_for_security",
        jwt_expiration_minutes=60,
        database_url="sqlite:///:memory:",  # In-memory for fast tests
        redis_url="redis://localhost:6379/15",  # Test DB
        debug=False,
        log_level="ERROR",  # Reduce noise in tests
        feature_advanced_trading=True,
        feature_rl_trading=True,
        feature_sentiment_analysis=True,
        feature_dex_arbitrage=True,
    )


@pytest.fixture
def override_settings(test_settings):
    """Override app settings for tests"""
    from config.settings import get_settings
    from functools import lru_cache

    # Clear cache
    get_settings.cache_clear()

    def _get_test_settings():
        return test_settings

    return _get_test_settings


# ============================================================================
# FastAPI Test Client
# ============================================================================

@pytest.fixture(scope="session")
def app():
    """Create FastAPI app for testing"""
    from main import app as fastapi_app
    return fastapi_app


@pytest.fixture
def client(app, test_settings):
    """FastAPI test client with test settings"""
    from app.dependencies import get_settings

    # Override settings dependency
    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def auth_client(client, test_token):
    """Authenticated test client"""
    client.headers = {"Authorization": f"Bearer {test_token}"}
    return client


# ============================================================================
# Authentication Fixtures
# ============================================================================

@pytest.fixture
def test_user_id() -> str:
    """Test user ID"""
    return "test_user_123"


@pytest.fixture
def test_permissions() -> set:
    """Standard test user permissions"""
    return {Permission.READ, Permission.WRITE, Permission.EXECUTE}


@pytest.fixture
def admin_permissions() -> set:
    """Admin permissions"""
    return {Permission.ADMIN}


@pytest.fixture
def readonly_permissions() -> set:
    """Read-only permissions"""
    return {Permission.READ}


@pytest.fixture
def test_token(test_user_id, test_permissions, test_settings) -> str:
    """Generate valid JWT token for testing"""
    return create_access_token(test_user_id, test_permissions, test_settings)


@pytest.fixture
def admin_token(test_user_id, admin_permissions, test_settings) -> str:
    """Generate admin JWT token"""
    return create_access_token(f"{test_user_id}_admin", admin_permissions, test_settings)


@pytest.fixture
def readonly_token(test_user_id, readonly_permissions, test_settings) -> str:
    """Generate read-only JWT token"""
    return create_access_token(f"{test_user_id}_readonly", readonly_permissions, test_settings)


@pytest.fixture
def expired_token(test_user_id, test_permissions, test_settings) -> str:
    """Generate expired JWT token for testing"""
    from jose import jwt
    from datetime import datetime, timedelta

    expire = datetime.utcnow() - timedelta(hours=1)  # Already expired
    to_encode = {
        "sub": test_user_id,
        "permissions": [p.value for p in test_permissions],
        "exp": expire,
        "iat": datetime.utcnow() - timedelta(hours=2)
    }

    return jwt.encode(to_encode, test_settings.jwt_secret, algorithm=test_settings.jwt_algorithm)


@pytest.fixture
def invalid_token() -> str:
    """Invalid JWT token"""
    return "invalid.jwt.token.abc123"


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_trade_signal() -> Dict[str, Any]:
    """Mock AI trade signal"""
    return {
        "signal_id": "sig_test_123",
        "token": "SOL",
        "action": "buy",
        "confidence": 0.85,
        "price_target": 105.00,
        "strategy": "RL Trading (PPO)",
        "reasoning": "Strong bullish momentum detected",
        "generated_at": datetime.now().isoformat()
    }


@pytest.fixture
def mock_position() -> Dict[str, Any]:
    """Mock trading position"""
    return {
        "position_id": "pos_test_123",
        "token": "SOL",
        "size": 100.0,
        "entry_price": 100.00,
        "current_price": 105.00,
        "pnl": 5.00,
        "pnl_pct": 0.05,
        "opened_at": datetime.now().isoformat(),
        "status": "open"
    }


@pytest.fixture
def mock_market_data() -> Dict[str, Any]:
    """Mock market data"""
    return {
        "token": "SOL",
        "price": 102.50,
        "volume_24h": 450_000_000,
        "market_cap": 45_000_000_000,
        "price_change_24h": 0.025,
        "timestamp": datetime.now().isoformat()
    }


@pytest.fixture
def mock_ml_model() -> Dict[str, Any]:
    """Mock ML model info"""
    return {
        "model_id": "rl_ppo_test",
        "name": "RL Trading (PPO)",
        "version": "1.0.0",
        "type": "rl",
        "status": "deployed",
        "accuracy": 0.75,
        "performance_metrics": {
            "sharpe_ratio": 1.85,
            "win_rate": 0.68
        },
        "trained_on": datetime.now().isoformat(),
        "deployed_on": datetime.now().isoformat()
    }


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def db_session():
    """Database session for tests (in-memory SQLite)"""
    # TODO: Implement when database models are ready
    # from app.database import SessionLocal, Base, engine
    # Base.metadata.create_all(bind=engine)
    # session = SessionLocal()
    # yield session
    # session.close()
    # Base.metadata.drop_all(bind=engine)
    pass


# ============================================================================
# Mock External Services
# ============================================================================

@pytest.fixture
def mock_jupiter_client():
    """Mock Jupiter DEX client"""
    mock = Mock()
    mock.get_quote = AsyncMock(return_value={
        "inputMint": "So11111111111111111111111111111111111111112",
        "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        "inAmount": "1000000000",  # 1 SOL
        "outAmount": "102500000",  # 102.5 USDC
        "priceImpactPct": 0.1
    })
    mock.execute_swap = AsyncMock(return_value={
        "signature": "test_tx_sig_123",
        "status": "success"
    })
    return mock


@pytest.fixture
def mock_solana_client():
    """Mock Solana RPC client"""
    mock = Mock()
    mock.get_balance = AsyncMock(return_value=1000000000)  # 1 SOL
    mock.send_transaction = AsyncMock(return_value="test_tx_123")
    mock.confirm_transaction = AsyncMock(return_value=True)
    return mock


# ============================================================================
# Guardian Defense Fixtures
# ============================================================================

@pytest.fixture
def guardian_defense():
    """Guardian defense instance for testing"""
    from security.guardian_defense import GuardianDefense
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        guardian = GuardianDefense(log_dir=tmpdir)
        yield guardian
        guardian.stop()


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def freeze_time():
    """Freeze time for deterministic tests"""
    frozen = datetime(2025, 10, 28, 12, 0, 0)

    class FrozenTime:
        @staticmethod
        def now():
            return frozen

    return FrozenTime


@pytest.fixture
def test_username() -> str:
    """Fixture for test username - can be overridden via environment variable"""
    return os.getenv("SHIVX_TEST_USERNAME", "test_user_fixture")


@pytest.fixture
def test_password() -> str:
    """
    Fixture for test password - meets production security requirements.

    Password requirements (enforced by PasswordValidator):
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character
    - No common weak patterns
    - No sequential characters
    - Minimum character diversity

    Can be overridden via SHIVX_TEST_PASSWORD environment variable
    """
    return os.getenv("SHIVX_TEST_PASSWORD", "Test!Pass@2024#Secure")


@pytest.fixture
def wrong_password() -> str:
    """
    Fixture for wrong password for negative tests.
    Also meets security requirements to avoid validation errors.
    """
    return "Wrong!Pass@2024#Invalid"


@pytest.fixture
def test_user_cleanup():
    """Cleanup fixture to ensure test users are removed after tests"""
    created_users = []

    def register_user(user_id: str):
        created_users.append(user_id)

    yield register_user

    # Cleanup happens here after test completes
    # In a real implementation, would delete users from database
    pass


# ============================================================================
# Async Event Loop
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def benchmark_settings():
    """Settings for performance benchmarks"""
    return {
        "min_iterations": 10,
        "max_time": 1.0,  # seconds
        "warmup": True
    }


@pytest.fixture
def rate_limiter_config():
    """Rate limiter configuration for testing"""
    return {
        "requests_per_minute": 100,
        "burst_size": 10
    }


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def generate_test_trades():
    """Generate mock trade data for testing"""
    def _generator(count: int = 10):
        return [
            {
                "trade_id": f"trade_{i}",
                "token": "SOL",
                "action": "buy" if i % 2 == 0 else "sell",
                "amount": 100.0 + i,
                "price": 100.0 + (i * 0.5),
                "executed_at": (datetime.now() - timedelta(hours=i)).isoformat()
            }
            for i in range(count)
        ]
    return _generator


@pytest.fixture
def generate_test_signals():
    """Generate mock trading signals"""
    def _generator(count: int = 5):
        return [
            {
                "signal_id": f"sig_{i}",
                "token": ["SOL", "RAY", "ORCA"][i % 3],
                "action": ["buy", "sell", "hold"][i % 3],
                "confidence": 0.7 + (i * 0.05),
                "strategy": "RL Trading (PPO)",
                "generated_at": (datetime.now() - timedelta(minutes=i*5)).isoformat()
            }
            for i in range(count)
        ]
    return _generator


# ============================================================================
# Configuration Helpers
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(f"SHIVX_{key.upper()}", str(value))
    return _set_env


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
