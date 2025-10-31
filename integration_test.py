"""
Quick Integration Test
Tests that all new components can be imported and initialized
"""

import sys
import asyncio

def test_imports():
    """Test that all components can be imported"""
    print("✓ Testing imports...")

    try:
        # Database models
        from app.models.trading import Position, TradeSignal, TradeExecution, Strategy
        from app.models.ml import MLModel, TrainingJob, Prediction
        from app.models.user import User, APIKey
        print("  ✅ Database models imported successfully")

        # Services
        from app.services.trading_service import TradingService
        from app.services.ml_service import MLService
        from app.services.analytics_service import AnalyticsService
        print("  ✅ Services imported successfully")

        # AGI modules
        from core.agi.language import LanguageModule
        from core.agi.memory import MemoryModule
        from core.agi.perception import PerceptionModule
        from core.agi.planning import PlanningModule
        from core.agi.social import SocialIntelligence
        from core.agi.core import AGICore
        print("  ✅ AGI modules imported successfully")

        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_agi_initialization():
    """Test AGI core initialization"""
    print("\n✓ Testing AGI initialization...")

    try:
        from core.agi.core import AGICore

        agi = AGICore()
        status = agi.get_status()

        print(f"  ✅ AGI initialized successfully")
        print(f"  ✅ AGI Score: {status['agi_score']}/100")
        print(f"  ✅ Status: {status['status']}")

        return True
    except Exception as e:
        print(f"  ❌ AGI initialization failed: {e}")
        return False


async def test_agi_chat():
    """Test AGI chat functionality"""
    print("\n✓ Testing AGI chat...")

    try:
        from core.agi.core import AGICore

        agi = AGICore()
        response = await agi.chat("Hello, how are you?")

        print(f"  ✅ Chat response: {response[:100]}...")

        return True
    except Exception as e:
        print(f"  ❌ Chat test failed: {e}")
        return False


async def test_agi_modules():
    """Test individual AGI modules"""
    print("\n✓ Testing AGI modules...")

    try:
        from core.agi.language import LanguageModule
        from core.agi.memory import MemoryModule
        from core.agi.planning import PlanningModule
        from core.agi.social import SocialIntelligence

        # Language
        lang = LanguageModule()
        understanding = await lang.understand("Should I buy SOL?")
        print(f"  ✅ Language: Intent={understanding['intent']}")

        # Memory
        mem = MemoryModule()
        await mem.remember("test", "Test episode")
        print(f"  ✅ Memory: Episode stored")

        # Planning
        plan_mod = PlanningModule()
        capabilities = plan_mod.get_capabilities()
        print(f"  ✅ Planning: {capabilities['action_library_size']} actions")

        # Social
        social = SocialIntelligence()
        emotion = await social.recognize_emotion("I'm happy!")
        print(f"  ✅ Social: Emotion recognized")

        return True
    except Exception as e:
        print(f"  ❌ Module test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("="*70)
    print("SHIVX INTEGRATION TEST SUITE")
    print("="*70)

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: AGI Initialization
    results.append(("AGI Init", test_agi_initialization()))

    # Test 3: AGI Chat (async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results.append(("AGI Chat", loop.run_until_complete(test_agi_chat())))

    # Test 4: AGI Modules (async)
    results.append(("AGI Modules", loop.run_until_complete(test_agi_modules())))
    loop.close()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")

    print(f"\nTotal: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")

    if passed == total:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
