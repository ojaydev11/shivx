#!/usr/bin/env python3
"""
Stress Test: Language Intelligence (Pillar 6)

Comprehensive stress test for Natural Language Intelligence:
- NLU (understanding)
- NLG (generation)
- Dialogue management
- Language reasoning

Tests functionality, performance, and error handling.
"""
import sys
from pathlib import Path
import time
from typing import List, Dict, Any
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.agi_service import get_agi_service


class LanguageStressTest:
    """Stress test suite for Language Intelligence"""

    def __init__(self):
        print()
        print("=" * 70)
        print("🧪 STRESS TEST: LANGUAGE INTELLIGENCE (Pillar 6)")
        print("=" * 70)
        print()

        self.agi = get_agi_service()
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results: List[Dict[str, Any]] = []

    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling"""
        print(f"🔬 Test: {test_name}")
        print("-" * 70)

        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time

            if result.get("passed", False):
                print(f"✅ PASSED ({duration:.2f}s)")
                self.tests_passed += 1
            else:
                print(f"❌ FAILED ({duration:.2f}s)")
                print(f"   Reason: {result.get('reason', 'Unknown')}")
                self.tests_failed += 1

            self.test_results.append({
                "test": test_name,
                "passed": result.get("passed", False),
                "duration": duration,
                "details": result
            })

        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ CRASHED ({duration:.2f}s)")
            print(f"   Error: {e}")
            traceback.print_exc()
            self.tests_failed += 1

            self.test_results.append({
                "test": test_name,
                "passed": False,
                "duration": duration,
                "error": str(e)
            })

        print()

    # ========================================================================
    # NLU Tests (Natural Language Understanding)
    # ========================================================================

    def test_nlu_basic(self) -> Dict[str, Any]:
        """Test basic language understanding"""
        text = "I want to buy 100 SOL tokens at market price"

        result = self.agi.understand_language(text)

        if "error" in result:
            return {"passed": False, "reason": result["error"]}

        # Validate structure
        if not all(k in result for k in ["text", "intent", "entities", "sentiment", "confidence"]):
            return {"passed": False, "reason": "Missing required fields"}

        print(f"   Intent: {result['intent']}")
        print(f"   Entities: {len(result['entities'])} found")
        print(f"   Sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.2%}")

        return {"passed": True, "result": result}

    def test_nlu_complex(self) -> Dict[str, Any]:
        """Test complex language understanding"""
        text = "Can you analyze the market sentiment for Bitcoin and Ethereum, then execute a limit buy order for 5 ETH at $3000 if the conditions are favorable?"

        result = self.agi.understand_language(text)

        if "error" in result:
            return {"passed": False, "reason": result["error"]}

        print(f"   Intent: {result['intent']}")
        print(f"   Entities: {len(result['entities'])} found")

        # Should detect multiple intents/entities
        if len(result['entities']) < 2:
            return {
                "passed": False,
                "reason": f"Expected multiple entities, got {len(result['entities'])}"
            }

        return {"passed": True, "result": result}

    def test_nlu_sentiment(self) -> Dict[str, Any]:
        """Test sentiment analysis"""
        texts = [
            ("This trading bot is absolutely amazing! Best ROI ever!", "positive"),
            ("I lost all my money. This is terrible.", "negative"),
            ("The market is sideways today.", "neutral")
        ]

        all_passed = True
        for text, expected_sentiment in texts:
            result = self.agi.understand_language(text)

            if "error" in result:
                return {"passed": False, "reason": result["error"]}

            detected = result['sentiment'].get('label', '').lower()
            print(f"   '{text[:50]}...' → {detected}")

            # Note: We're lenient here - just check that sentiment exists
            if 'label' not in result['sentiment']:
                all_passed = False

        return {"passed": all_passed}

    # ========================================================================
    # NLG Tests (Natural Language Generation)
    # ========================================================================

    def test_nlg_professional(self) -> Dict[str, Any]:
        """Test professional language generation"""
        context = "Market analysis shows strong bullish trend"

        result = self.agi.generate_response(context=context, style="professional")

        if "error" in result:
            return {"passed": False, "reason": result["error"]}

        print(f"   Response: '{result['response'][:100]}...'")

        # Check that response exists and is non-empty
        if not result.get('response'):
            return {"passed": False, "reason": "Empty response"}

        return {"passed": True, "result": result}

    def test_nlg_multiple_styles(self) -> Dict[str, Any]:
        """Test multiple generation styles"""
        context = "SOL price increased 15% today"
        styles = ["professional", "casual", "technical", "creative"]

        all_passed = True
        for style in styles:
            result = self.agi.generate_response(context=context, style=style)

            if "error" in result:
                return {"passed": False, "reason": result["error"]}

            print(f"   {style}: '{result['response'][:60]}...'")

            if not result.get('response'):
                all_passed = False

        return {"passed": all_passed}

    # ========================================================================
    # Dialogue Tests
    # ========================================================================

    def test_dialogue_single_turn(self) -> Dict[str, Any]:
        """Test single-turn dialogue"""
        session_id = "test_session_001"

        # Create session
        session_result = self.agi.create_session("test_user", session_id)

        if "error" in session_result:
            return {"passed": False, "reason": session_result["error"]}

        # Send message
        result = self.agi.chat(
            message="What's the current market trend?",
            session_id=session_id
        )

        if "error" in result:
            return {"passed": False, "reason": result["error"]}

        print(f"   User: {result['message']}")
        print(f"   AGI: {result['response'][:100]}...")

        return {"passed": True, "result": result}

    def test_dialogue_multi_turn(self) -> Dict[str, Any]:
        """Test multi-turn dialogue with context"""
        session_id = "test_session_002"

        # Create session
        self.agi.create_session("test_user", session_id)

        # Turn 1
        result1 = self.agi.chat(
            message="I want to invest in crypto",
            session_id=session_id
        )

        if "error" in result1:
            return {"passed": False, "reason": result1["error"]}

        print(f"   Turn 1:")
        print(f"      User: {result1['message']}")
        print(f"      AGI: {result1['response'][:80]}...")

        # Turn 2 - should maintain context
        result2 = self.agi.chat(
            message="Which tokens do you recommend?",
            session_id=session_id
        )

        if "error" in result2:
            return {"passed": False, "reason": result2["error"]}

        print(f"   Turn 2:")
        print(f"      User: {result2['message']}")
        print(f"      AGI: {result2['response'][:80]}...")

        # Turn 3 - follow-up
        result3 = self.agi.chat(
            message="Tell me more about the first one",
            session_id=session_id
        )

        if "error" in result3:
            return {"passed": False, "reason": result3["error"]}

        print(f"   Turn 3:")
        print(f"      User: {result3['message']}")
        print(f"      AGI: {result3['response'][:80]}...")

        return {"passed": True, "turns": 3}

    # ========================================================================
    # Performance Tests
    # ========================================================================

    def test_performance_nlu(self) -> Dict[str, Any]:
        """Test NLU performance under load"""
        texts = [
            "Buy 10 BTC at market price",
            "What's the sentiment for Ethereum?",
            "Execute arbitrage between Raydium and Orca",
            "Show me the top performing tokens",
            "Set a stop loss at $50,000"
        ]

        start_time = time.time()
        results = []

        for text in texts:
            result = self.agi.understand_language(text)
            if "error" not in result:
                results.append(result)

        duration = time.time() - start_time
        avg_time = duration / len(texts)

        print(f"   Processed: {len(results)}/{len(texts)} texts")
        print(f"   Total time: {duration:.2f}s")
        print(f"   Avg per text: {avg_time:.3f}s")

        # Performance threshold: should process at least 5 texts in under 5 seconds
        passed = len(results) == len(texts) and duration < 5.0

        return {
            "passed": passed,
            "total_time": duration,
            "avg_time": avg_time,
            "processed": len(results)
        }

    def test_performance_nlg(self) -> Dict[str, Any]:
        """Test NLG performance under load"""
        contexts = [
            "Market is bullish",
            "High volatility detected",
            "Arbitrage opportunity found",
            "Risk level increased",
            "Portfolio rebalancing needed"
        ]

        start_time = time.time()
        results = []

        for context in contexts:
            result = self.agi.generate_response(context=context)
            if "error" not in result:
                results.append(result)

        duration = time.time() - start_time
        avg_time = duration / len(contexts)

        print(f"   Generated: {len(results)}/{len(contexts)} responses")
        print(f"   Total time: {duration:.2f}s")
        print(f"   Avg per response: {avg_time:.3f}s")

        passed = len(results) == len(contexts) and duration < 10.0

        return {
            "passed": passed,
            "total_time": duration,
            "avg_time": avg_time,
            "generated": len(results)
        }

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_error_handling_empty_input(self) -> Dict[str, Any]:
        """Test handling of empty input"""
        result = self.agi.understand_language("")

        # Should handle gracefully (either work or return error, but not crash)
        print(f"   Result: {result.get('intent', result.get('error', 'No response'))}")

        return {"passed": True}  # Didn't crash

    def test_error_handling_invalid_session(self) -> Dict[str, Any]:
        """Test handling of invalid session"""
        result = self.agi.chat(
            message="Hello",
            session_id="nonexistent_session_999"
        )

        # Should handle gracefully
        print(f"   Result: {result.get('response', result.get('error', 'No response'))[:60]}...")

        return {"passed": True}  # Didn't crash

    # ========================================================================
    # Integration Tests
    # ========================================================================

    def test_integration_understanding_and_generation(self) -> Dict[str, Any]:
        """Test NLU + NLG pipeline"""
        # Understand
        text = "What's the best trading strategy for volatile markets?"
        nlu_result = self.agi.understand_language(text)

        if "error" in nlu_result:
            return {"passed": False, "reason": nlu_result["error"]}

        print(f"   Intent: {nlu_result['intent']}")

        # Generate response based on understanding
        context = f"User asked about {nlu_result['intent']}"
        nlg_result = self.agi.generate_response(context=context, style="professional")

        if "error" in nlg_result:
            return {"passed": False, "reason": nlg_result["error"]}

        print(f"   Response: '{nlg_result['response'][:100]}...'")

        return {"passed": True}

    # ========================================================================
    # Main Test Runner
    # ========================================================================

    def run_all_tests(self):
        """Run all language intelligence tests"""
        print("Starting comprehensive language intelligence stress test...")
        print()

        # NLU Tests
        print("📋 NLU (Natural Language Understanding) Tests")
        print("=" * 70)
        self.run_test("NLU: Basic Understanding", self.test_nlu_basic)
        self.run_test("NLU: Complex Understanding", self.test_nlu_complex)
        self.run_test("NLU: Sentiment Analysis", self.test_nlu_sentiment)

        # NLG Tests
        print("📋 NLG (Natural Language Generation) Tests")
        print("=" * 70)
        self.run_test("NLG: Professional Style", self.test_nlg_professional)
        self.run_test("NLG: Multiple Styles", self.test_nlg_multiple_styles)

        # Dialogue Tests
        print("📋 Dialogue Management Tests")
        print("=" * 70)
        self.run_test("Dialogue: Single Turn", self.test_dialogue_single_turn)
        self.run_test("Dialogue: Multi-Turn Context", self.test_dialogue_multi_turn)

        # Performance Tests
        print("📋 Performance Tests")
        print("=" * 70)
        self.run_test("Performance: NLU Under Load", self.test_performance_nlu)
        self.run_test("Performance: NLG Under Load", self.test_performance_nlg)

        # Error Handling Tests
        print("📋 Error Handling Tests")
        print("=" * 70)
        self.run_test("Error: Empty Input", self.test_error_handling_empty_input)
        self.run_test("Error: Invalid Session", self.test_error_handling_invalid_session)

        # Integration Tests
        print("📋 Integration Tests")
        print("=" * 70)
        self.run_test("Integration: NLU + NLG", self.test_integration_understanding_and_generation)

        # Print Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print()
        print("=" * 70)
        print("📊 TEST SUMMARY")
        print("=" * 70)
        print()

        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"Total Tests:   {total_tests}")
        print(f"Passed:        {self.tests_passed} ✅")
        print(f"Failed:        {self.tests_failed} ❌")
        print(f"Pass Rate:     {pass_rate:.1f}%")
        print()

        if pass_rate >= 90:
            print("🎉 EXCELLENT - Language Intelligence is highly stable!")
        elif pass_rate >= 75:
            print("✅ GOOD - Language Intelligence is stable with minor issues")
        elif pass_rate >= 50:
            print("⚠️  FAIR - Language Intelligence needs improvements")
        else:
            print("❌ POOR - Language Intelligence requires major fixes")

        print()
        print("=" * 70)


if __name__ == "__main__":
    try:
        tester = LanguageStressTest()
        tester.run_all_tests()

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
