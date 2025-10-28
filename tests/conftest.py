"""
Pytest configuration and fixtures for ShivX tests
"""

import pytest
import os
from typing import Generator


@pytest.fixture
def test_username() -> str:
    """Fixture for test username - can be overridden via environment variable"""
    return os.getenv("SHIVX_TEST_USERNAME", "test_user_fixture")


@pytest.fixture
def test_password() -> str:
    """Fixture for test password - can be overridden via environment variable"""
    return os.getenv("SHIVX_TEST_PASSWORD", "test_pass_fixture_secure_123")


@pytest.fixture
def wrong_password() -> str:
    """Fixture for wrong password for negative tests"""
    return "intentionally_wrong_password_xyz"


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
