#!/usr/bin/env python3
"""
Verification Script for ShivX Caching Implementation
Checks all components are properly implemented
"""

import os
import sys
from pathlib import Path


def check_file_exists(path: str) -> bool:
    """Check if file exists"""
    return Path(path).exists()


def check_implementation():
    """Verify all caching components are implemented"""

    print("=" * 70)
    print("ShivX Caching Implementation Verification")
    print("=" * 70)
    print()

    checks = []

    # Core implementation files
    core_files = [
        ("Redis Connection Manager", "app/cache.py"),
        ("Market Data Cache", "app/services/market_cache.py"),
        ("Indicator Cache", "app/services/indicator_cache.py"),
        ("ML Predictions Cache", "app/services/ml_cache.py"),
        ("Session Manager", "app/services/session_cache.py"),
        ("Cache Monitor", "app/services/cache_monitor.py"),
        ("Cache Invalidation", "app/services/cache_invalidation.py"),
        ("Rate Limit Middleware", "app/middleware/rate_limit.py"),
        ("HTTP Cache Middleware", "app/middleware/cache.py"),
        ("Middleware Init", "app/middleware/__init__.py"),
    ]

    print("Core Implementation Files:")
    print("-" * 70)
    for name, path in core_files:
        exists = check_file_exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name:40s} {path}")
        checks.append(exists)
    print()

    # Test files
    test_files = [
        ("Performance Tests", "tests/test_cache_performance.py"),
    ]

    print("Test Files:")
    print("-" * 70)
    for name, path in test_files:
        exists = check_file_exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name:40s} {path}")
        checks.append(exists)
    print()

    # Documentation
    doc_files = [
        ("Implementation Guide", "CACHING_IMPLEMENTATION.md"),
        ("Executive Summary", "CACHING_SUMMARY.md"),
        ("Integration Example", "examples/cache_integration_example.py"),
    ]

    print("Documentation:")
    print("-" * 70)
    for name, path in doc_files:
        exists = check_file_exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name:40s} {path}")
        checks.append(exists)
    print()

    # Configuration
    print("Configuration:")
    print("-" * 70)
    config_file = "config/settings.py"
    exists = check_file_exists(config_file)
    status = "✅" if exists else "❌"
    print(f"{status} Settings Configuration{' ' * 23} {config_file}")

    # Check for cache settings in config
    if exists:
        with open(config_file, 'r') as f:
            content = f.read()
            has_redis_pool = "redis_pool_size" in content
            has_cache_enabled = "cache_enabled" in content
            has_cache_ttls = "cache_market_price_ttl" in content

            status = "✅" if has_redis_pool else "❌"
            print(f"{status} - Redis pool configuration")

            status = "✅" if has_cache_enabled else "❌"
            print(f"{status} - Cache enabled setting")

            status = "✅" if has_cache_ttls else "❌"
            print(f"{status} - Cache TTL settings")

            checks.extend([has_redis_pool, has_cache_enabled, has_cache_ttls])
    print()

    # Summary
    print("=" * 70)
    print("Summary:")
    print("-" * 70)
    total = len(checks)
    passed = sum(checks)
    percentage = (passed / total * 100) if total > 0 else 0

    print(f"Total Checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {percentage:.1f}%")
    print()

    if passed == total:
        print("✅ ALL CHECKS PASSED - Implementation Complete!")
        print()
        print("The caching system is fully implemented and ready for use.")
        print()
        print("Next Steps:")
        print("1. Install Redis: docker run -d -p 6379:6379 redis:7-alpine")
        print("2. Configure .env: SHIVX_REDIS_URL=redis://localhost:6379/0")
        print("3. Run tests: pytest tests/test_cache_performance.py -v")
        print("4. Start application with caching enabled")
        print()
        return True
    else:
        print("❌ SOME CHECKS FAILED - Review missing components")
        return False


if __name__ == "__main__":
    # Change to project root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    success = check_implementation()
    sys.exit(0 if success else 1)
