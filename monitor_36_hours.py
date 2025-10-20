"""
36-Hour Trading System Monitor
Validates signal consistency and system stability
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path

def save_snapshot(snapshot_num, data):
    """Save monitoring snapshot"""
    snapshot_file = Path(f"logs/snapshot_{snapshot_num:02d}.json")
    with open(snapshot_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Snapshot saved to: {snapshot_file}")

def monitor_36_hours():
    """Monitor trading system for 36 hours with 6-hour snapshots"""

    print("\n" + "="*70)
    print("36-HOUR TRADING SYSTEM VALIDATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will end: {(datetime.now() + timedelta(hours=36)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nObjectives:")
    print("  1. Validate signal consistency over extended period")
    print("  2. Test system stability (no crashes)")
    print("  3. Analyze performance across market conditions")
    print("  4. Prepare data for live trading decision")
    print("\nMonitoring schedule:")
    print("  - Snapshot every 6 hours (6 total)")
    print("  - Quick check every 30 minutes")
    print("  - Alert on errors or anomalies")
    print("="*70 + "\n")

    start_time = datetime.now()
    end_time = start_time + timedelta(hours=36)
    snapshot_interval = timedelta(hours=6)
    check_interval = timedelta(minutes=30)

    next_snapshot = start_time + snapshot_interval
    next_check = start_time + check_interval
    snapshot_num = 0

    # Get baseline
    try:
        with open('data/income/advanced_ai/ai_state.json') as f:
            baseline_state = json.load(f)
        baseline_trades = baseline_state.get('total_trades', 0)
        baseline_profit = baseline_state.get('total_profit', 0.0)
    except:
        baseline_trades = 0
        baseline_profit = 0.0

    print(f"Baseline: {baseline_trades} trades, ${baseline_profit:.4f} profit\n")

    try:
        while datetime.now() < end_time:
            current_time = datetime.now()

            # Snapshot time?
            if current_time >= next_snapshot:
                snapshot_num += 1
                elapsed = (current_time - start_time).total_seconds() / 3600
                remaining = (end_time - current_time).total_seconds() / 3600

                print("\n" + "="*70)
                print(f"SNAPSHOT #{snapshot_num}/6 ({elapsed:.1f}h elapsed, {remaining:.1f}h remaining)")
                print("="*70)

                # Read current state
                try:
                    with open('data/income/advanced_ai/ai_state.json') as f:
                        ai_state = json.load(f)

                    with open('logs/optimized_trading_test.log') as f:
                        lines = f.readlines()
                        cycles = [l for l in lines if 'Trading Cycle #' in l]
                        current_cycle = len(cycles)

                        # Check for errors
                        recent_errors = [l for l in lines[-1000:] if 'ERROR' in l or 'Error:' in l]

                    total_trades = ai_state.get('total_trades', 0)
                    winning = ai_state.get('winning_trades', 0)
                    losing = ai_state.get('losing_trades', 0)
                    profit = ai_state.get('total_profit', 0.0)

                    snapshot_data = {
                        'snapshot_num': snapshot_num,
                        'timestamp': current_time,
                        'elapsed_hours': elapsed,
                        'cycles_completed': current_cycle,
                        'total_trades': total_trades,
                        'winning_trades': winning,
                        'losing_trades': losing,
                        'total_profit': profit,
                        'win_rate': (winning / total_trades * 100) if total_trades > 0 else 0,
                        'avg_profit_per_trade': (profit / total_trades) if total_trades > 0 else 0,
                        'trades_since_start': total_trades - baseline_trades,
                        'profit_since_start': profit - baseline_profit,
                        'errors_detected': len(recent_errors),
                        'system_status': 'ERROR' if recent_errors else 'RUNNING'
                    }

                    # Display
                    print(f"\nTime: {current_time.strftime('%H:%M:%S')}")
                    print(f"Cycles: {current_cycle}")
                    print(f"Total Trades: {total_trades} ({snapshot_data['trades_since_start']} new)")
                    print(f"Win Rate: {snapshot_data['win_rate']:.1f}%")
                    print(f"Total Profit: ${profit:.4f} (${snapshot_data['profit_since_start']:.4f} new)")
                    print(f"Avg/Trade: ${snapshot_data['avg_profit_per_trade']:.4f}")

                    if recent_errors:
                        print(f"\n⚠️  ERRORS DETECTED: {len(recent_errors)}")
                        for err in recent_errors[-3:]:
                            print(f"  {err.strip()}")
                    else:
                        print(f"\n✅ System Status: HEALTHY (no errors)")

                    # Save snapshot
                    save_snapshot(snapshot_num, snapshot_data)

                    print("="*70 + "\n")

                except Exception as e:
                    print(f"ERROR reading snapshot data: {e}\n")

                next_snapshot = current_time + snapshot_interval

            # Quick health check?
            elif current_time >= next_check:
                try:
                    with open('data/income/advanced_ai/ai_state.json') as f:
                        ai_state = json.load(f)

                    total_trades = ai_state.get('total_trades', 0)
                    profit = ai_state.get('total_profit', 0.0)

                    elapsed = (current_time - start_time).total_seconds() / 3600
                    print(f"[{current_time.strftime('%H:%M:%S')}] Health check: {total_trades} trades, ${profit:.4f} profit ({elapsed:.1f}h elapsed)")

                except Exception as e:
                    print(f"[{current_time.strftime('%H:%M:%S')}] Health check failed: {e}")

                next_check = current_time + check_interval

            # Sleep a bit
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    # Final report
    print("\n" + "="*70)
    print("36-HOUR MONITORING COMPLETE")
    print("="*70)

    actual_duration = (datetime.now() - start_time).total_seconds() / 3600

    try:
        with open('data/income/advanced_ai/ai_state.json') as f:
            final_state = json.load(f)

        final_trades = final_state.get('total_trades', 0)
        final_profit = final_state.get('total_profit', 0.0)
        final_winning = final_state.get('winning_trades', 0)

        print(f"\nActual Duration: {actual_duration:.1f} hours")
        print(f"Final Stats:")
        print(f"  Total Trades: {final_trades} ({final_trades - baseline_trades} new)")
        print(f"  Win Rate: {(final_winning/final_trades*100):.1f}%" if final_trades > 0 else "  Win Rate: N/A")
        print(f"  Total Profit: ${final_profit:.4f} (${final_profit - baseline_profit:.4f} new)")
        print(f"  Hourly Rate: {(final_trades - baseline_trades) / actual_duration:.1f} trades/hour")
        print(f"  Hourly Profit: ${(final_profit - baseline_profit) / actual_duration:.2f}/hour")

        # Save final report
        final_report = {
            'duration_hours': actual_duration,
            'baseline_trades': baseline_trades,
            'final_trades': final_trades,
            'new_trades': final_trades - baseline_trades,
            'baseline_profit': baseline_profit,
            'final_profit': final_profit,
            'new_profit': final_profit - baseline_profit,
            'trades_per_hour': (final_trades - baseline_trades) / actual_duration,
            'profit_per_hour': (final_profit - baseline_profit) / actual_duration,
            'win_rate': (final_winning / final_trades * 100) if final_trades > 0 else 0
        }

        with open('logs/36hour_final_report.json', 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        print(f"\nFinal report saved to: logs/36hour_final_report.json")

    except Exception as e:
        print(f"Error generating final report: {e}")

    print("="*70 + "\n")

if __name__ == "__main__":
    monitor_36_hours()
