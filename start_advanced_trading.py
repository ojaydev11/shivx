"""
Start Advanced AI Trading System
Combines traditional arbitrage with cutting-edge AI
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/advanced_trading.log')
    ]
)

logger = logging.getLogger(__name__)

from core.income.advanced_trading_ai import AdvancedTradingAI
from core.income.jupiter_client import JupiterClient


class AdvancedTradingSystem:
    """
    Most Advanced Automated Trading System
    Combines multiple AI strategies with real market data
    """

    def __init__(self, config: dict):
        self.config = config
        self.ai = AdvancedTradingAI(config)
        self.jupiter = None
        self.running = False

        # Performance tracking
        self.start_time = None
        self.total_cycles = 0
        self.total_signals_generated = 0
        self.total_trades_executed = 0

    async def initialize(self):
        """Initialize the trading system"""
        logger.info("=" * 70)
        logger.info("ADVANCED AI TRADING SYSTEM - Initializing")
        logger.info("=" * 70)

        # Initialize Jupiter client
        self.jupiter = JupiterClient(timeout=30)
        await self.jupiter.__aenter__()

        logger.info("[OK] Jupiter client connected")
        logger.info("[OK] Advanced AI initialized with 5 strategies:")
        logger.info("  1. Deep Reinforcement Learning (PPO)")
        logger.info("  2. ML Price Prediction (LSTM-style)")
        logger.info("  3. Sentiment Analysis")
        logger.info("  4. Technical Analysis (RSI, MACD, BB)")
        logger.info("  5. DEX Arbitrage")

        self.start_time = datetime.utcnow()
        logger.info(f"[OK] System ready at {self.start_time}")
        logger.info("=" * 70)

    async def get_live_market_data(self):
        """Get real-time market data from Jupiter"""
        prices = {}
        volumes = {}

        tokens = ['SOL', 'USDC', 'USDT', 'RAY', 'ORCA', 'JUP', 'BONK']

        # Get SOL price in USDC as baseline
        sol_quote = await self.jupiter.get_quote(
            input_mint=self.jupiter.TOKENS['SOL'],
            output_mint=self.jupiter.TOKENS['USDC'],
            amount=1_000_000_000,  # 1 SOL
            slippage_bps=50
        )

        if sol_quote:
            sol_price = sol_quote.out_amount / 1_000_000  # Convert to USDC
            prices['SOL'] = sol_price
            volumes['SOL'] = 1000000  # Placeholder

        # USDC and USDT are stablecoins
        prices['USDC'] = 1.0
        prices['USDT'] = 1.0
        volumes['USDC'] = 5000000
        volumes['USDT'] = 4500000

        # Get other token prices (in USDC)
        for token in ['RAY', 'ORCA', 'JUP', 'BONK']:
            if token not in self.jupiter.TOKENS:
                continue

            try:
                quote = await self.jupiter.get_quote(
                    input_mint=self.jupiter.TOKENS[token],
                    output_mint=self.jupiter.TOKENS['USDC'],
                    amount=1_000_000,  # 1 token (assuming 6 decimals)
                    slippage_bps=100
                )

                if quote:
                    price = quote.out_amount / 1_000_000
                    prices[token] = price
                    volumes[token] = 500000

                await asyncio.sleep(0.1)  # Rate limiting

            except Exception as e:
                logger.warning(f"Failed to get price for {token}: {e}")

        return prices, volumes

    async def trading_cycle(self):
        """Single trading cycle"""
        self.total_cycles += 1

        logger.info(f"\n{'='*70}")
        logger.info(f"Trading Cycle #{self.total_cycles}")
        logger.info(f"{'='*70}\n")

        # Get real market data
        logger.info("[DATA] Fetching live market data...")
        prices, volumes = await self.get_live_market_data()

        logger.info(f"[OK] Market data retrieved:")
        for token, price in sorted(prices.items()):
            logger.info(f"  {token}: ${price:.4f}")

        # Analyze market with AI
        logger.info("\n[AI] Running AI analysis...")
        market_state = await self.ai.analyze_market(prices, volumes)

        # Generate trading signals
        pairs = self.config.get('monitored_pairs', [
            ('SOL', 'USDC'),
            ('SOL', 'USDT'),
            ('RAY', 'USDC'),
            ('ORCA', 'USDC'),
            ('JUP', 'USDC'),
            ('USDC', 'USDT')
        ])

        logger.info(f"[SIGNAL] Generating signals for {len(pairs)} pairs...")
        signals = await self.ai.generate_signals(market_state, pairs)

        self.total_signals_generated += len(signals)

        if signals:
            logger.info(f"\n[OK] Generated {len(signals)} trading signals:\n")

            for i, signal in enumerate(signals[:5], 1):  # Show top 5
                logger.info(f"Signal #{i}: {signal.pair} - {signal.action}")
                logger.info(f"  Confidence: {signal.confidence:.1%}")
                logger.info(f"  Expected Profit: {signal.expected_profit_pct:.3f}%")
                logger.info(f"  Sharpe Ratio: {signal.sharpe_ratio:.2f}")
                logger.info(f"  Position Size: {signal.position_size_pct:.1%}")
                logger.info(f"  Strategy: {signal.strategy}")
                logger.info(f"  Reasoning: {', '.join(signal.reasoning[:2])}")
                logger.info("")

            # Execute best signal (in paper trading mode)
            best_signal = signals[0]
            if best_signal.expected_profit_pct > 0.1:  # Min 0.1% profit
                logger.info(f"[TRADE] Executing signal: {best_signal.pair} {best_signal.action}")
                result = await self.ai.execute_signal(best_signal)

                if result['success']:
                    self.total_trades_executed += 1
                    logger.info(f"[OK] Trade executed successfully!")
                    logger.info(f"  Actual Profit: {result['actual_profit_pct']:.3f}%")
                    logger.info(f"  Slippage: {result['slippage_pct']:.3f}%")
                else:
                    logger.warning(f"[FAIL] Trade failed")
        else:
            logger.info("No profitable signals generated this cycle")

        # Show performance metrics
        logger.info(f"\n[METRICS] Performance Metrics:")
        metrics = self.ai.get_performance_metrics()
        logger.info(f"  Total Trades: {self.ai.total_trades}")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
        logger.info(f"  Total Profit: {self.ai.total_profit:.3f}%")

        # Save AI state
        await self.ai.save_state()

    async def run(self):
        """Main trading loop"""
        await self.initialize()

        self.running = True
        cycle_interval = self.config.get('cycle_interval_seconds', 30)

        logger.info(f"\n[START] Starting trading loop (cycle every {cycle_interval}s)...\n")

        try:
            while self.running:
                await self.trading_cycle()

                # Wait for next cycle
                logger.info(f"\n[WAIT] Waiting {cycle_interval}s until next cycle...\n")
                await asyncio.sleep(cycle_interval)

        except KeyboardInterrupt:
            logger.info("\n\n[STOP] Trading stopped by user")

        except Exception as e:
            logger.error(f"\n\n[ERROR] Error in trading loop: {e}", exc_info=True)

        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("\n" + "=" * 70)
        logger.info("SHUTTING DOWN - Final Statistics")
        logger.info("=" * 70)

        if self.start_time:
            runtime = datetime.utcnow() - self.start_time
            logger.info(f"Runtime: {runtime}")

        logger.info(f"Total Cycles: {self.total_cycles}")
        logger.info(f"Signals Generated: {self.total_signals_generated}")
        logger.info(f"Trades Executed: {self.total_trades_executed}")

        metrics = self.ai.get_performance_metrics()
        logger.info(f"\nFinal Performance:")
        logger.info(f"  Total Profit: {self.ai.total_profit:.3f}%")
        logger.info(f"  Win Rate: {metrics.win_rate:.1%}")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Expectancy: {metrics.expectancy:.3f}%")

        # Save final state
        await self.ai.save_state()
        logger.info("\n[OK] AI state saved")

        if self.jupiter:
            await self.jupiter.__aexit__(None, None, None)
            logger.info("[OK] Jupiter client closed")

        logger.info("\n" + "=" * 70)
        logger.info("SHUTDOWN COMPLETE")
        logger.info("=" * 70)


async def main():
    """Entry point"""
    # Configuration - OPTIMIZED FOR MICRO-OPPORTUNITIES
    config = {
        'min_profit_pct': 0.02,  # 🔴 CRITICAL FIX #1: 5x more sensitive (was 0.1%)
        'max_position_size_usd': 200,
        'cycle_interval_seconds': 15,  # 🟡 HIGH FIX #4: Faster cycles (was 30s)
        'monitored_pairs': [
            # Original 7 pairs
            ('SOL', 'USDC'),
            ('SOL', 'USDT'),
            ('RAY', 'USDC'),
            ('ORCA', 'USDC'),
            ('JUP', 'USDC'),
            ('BONK', 'USDC'),
            ('USDC', 'USDT'),
            # 🟡 HIGH FIX #5: Added 13 more pairs for 20 total
            ('BONK', 'SOL'),
            ('JUP', 'SOL'),
            ('RAY', 'USDT'),
            ('ORCA', 'USDT'),
            ('BONK', 'USDT'),
            ('JUP', 'USDT'),
            ('RAY', 'ORCA'),
            ('SOL', 'JUP'),
            ('RAY', 'SOL'),
            ('ORCA', 'SOL'),
            ('JUP', 'RAY'),
            ('BONK', 'RAY'),
            ('BONK', 'ORCA'),
        ]
    }

    system = AdvancedTradingSystem(config)
    await system.run()


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("ADVANCED AI TRADING SYSTEM")
    print("   Most Sophisticated Trading AI on the Internet")
    print("=" * 70)
    print("\nFeatures:")
    print("  [OK] Deep Reinforcement Learning (PPO)")
    print("  [OK] ML Price Prediction with LSTM-style models")
    print("  [OK] Real-time Sentiment Analysis")
    print("  [OK] Advanced Technical Indicators (RSI, MACD, Bollinger)")
    print("  [OK] DEX Arbitrage Detection")
    print("  [OK] Ensemble AI Strategy (5 models voting)")
    print("  [OK] Kelly Criterion Position Sizing")
    print("  [OK] Risk-Adjusted Sharpe Optimization")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 70 + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTrading stopped by user")
