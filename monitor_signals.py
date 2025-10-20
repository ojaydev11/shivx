import time
import json
import subprocess
from datetime import datetime

print("\n" + "="*70)
print("ACTIVE SIGNAL MONITORING - Waiting for First Signals")
print("="*70 + "\n")

last_cycle = 0
signal_detected = False

while True:
    try:
        # Read logs to check for signals
        with open('logs/optimized_trading_test.log', 'r') as f:
            lines = f.readlines()
        
        # Count cycles
        cycles = [l for l in lines if 'Trading Cycle #' in l]
        current_cycle = len(cycles)
        
        # Check for signal generation
        signal_lines = [l for l in lines if 'Generated' in l and 'trading signals' in l]
        
        if current_cycle != last_cycle:
            progress = current_cycle
            bar = '#' * (progress // 5) + '-' * (20 - progress // 5)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle #{current_cycle} | Progress: [{bar}] {progress}%", end='')
            
            # Check if we got signals this cycle
            recent_signals = [l for l in lines[-50:] if 'Generated' in l and 'trading signals' in l]
            if recent_signals and 'No profitable' not in lines[-30:]:
                print(" | SIGNAL DETECTED!")
                signal_detected = True
                # Print signal details
                for line in lines[-50:]:
                    if 'Signal #' in line or 'Confidence:' in line or 'Expected Profit:' in line:
                        print(f"    {line.strip()}")
            else:
                print(" | No signals (warmup)")
            
            last_cycle = current_cycle
            
            # Check for trades
            if current_cycle > 25:
                trade_lines = [l for l in lines[-100:] if 'Executing signal' in l or 'Trade executed' in l]
                if trade_lines:
                    print("\n" + "!"*70)
                    print("TRADE DETECTED!")
                    print("!"*70)
                    for line in trade_lines[-5:]:
                        print(line.strip())
                    break
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        break
    except Exception as e:
        print(f"\nError: {e}")
        time.sleep(5)

print("\n" + "="*70)
print("Monitoring complete")
print("="*70)
