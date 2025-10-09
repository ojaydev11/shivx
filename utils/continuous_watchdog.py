#!/usr/bin/env python3
"""
Continuous Watchdog Process - Handles automated snapshot collection and reporting
"""

import time
import schedule
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
import signal
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.watchdog_reporter import WatchdogReporter

class ContinuousWatchdog:
    """Manage continuous watchdog operations"""
    
    def __init__(self):
        self.reporter = WatchdogReporter()
        self.start_time = datetime.now()
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum=None, frame=None):
        """Graceful shutdown"""
        print(f"\n📛 Shutting down continuous watchdog (signal: {signum})")
        self.running = False
        
    def take_snapshot_job(self):
        """Job to take snapshots every 10 minutes"""
        try:
            print(f"📸 Taking snapshot at {datetime.now().strftime('%H:%M:%S')}")
            snapshot = self.reporter.take_snapshot()
            self.reporter.save_raw_snapshot(snapshot)
            
            # Check for anomalies
            anomalies = self.reporter.detect_anomalies(snapshot)
            if anomalies:
                self.reporter.log_incident(anomalies, snapshot)
                print(f"🚨 Anomalies detected: {len(anomalies)}")
                
        except Exception as e:
            print(f"❌ Snapshot job failed: {e}")
    
    def heartbeat_job(self):
        """Job to check panic flag every hour"""
        try:
            print(f"💓 Heartbeat check at {datetime.now().strftime('%H:%M:%S')}")
            heartbeat = self.reporter.check_panic_flag()
            
            if heartbeat['status'] != 'operational':
                print(f"⚠️ Non-operational status: {heartbeat['status']}")
                
        except Exception as e:
            print(f"❌ Heartbeat job failed: {e}")
    
    def generate_t6h_report(self):
        """Generate T+6h report"""
        try:
            print("📊 Generating T+6h status report...")
            report = self.reporter.generate_interim_report(6)
            
            report_path = Path("docs/staging/T+6h_status.md")
            report_path.write_text(report, encoding='utf-8')
            
            print(f"✅ T+6h report saved: {report_path}")
            
        except Exception as e:
            print(f"❌ T+6h report failed: {e}")
    
    def generate_t24h_report(self):
        """Generate T+24h report"""
        try:
            print("📊 Generating T+24h status report...")
            report = self.reporter.generate_interim_report(24)
            
            report_path = Path("docs/staging/T+24h_status.md")
            report_path.write_text(report, encoding='utf-8')
            
            print(f"✅ T+24h report saved: {report_path}")
            
        except Exception as e:
            print(f"❌ T+24h report failed: {e}")
    
    def generate_final_report(self):
        """Generate final T+48h report"""
        try:
            print("📊 Generating FINAL production promotion report...")
            
            # Use the existing report generator
            import subprocess
            result = subprocess.run([
                sys.executable, 'utils/generate_promotion_report.py'
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                print("✅ Final production promotion report generated")
            else:
                print(f"❌ Final report generation failed: {result.stderr}")
            
            # Generate machine summary
            summary_data = {
                'watchdog_completed': True,
                'completion_time': datetime.now().isoformat(),
                'duration_hours': 48,
                'final_metrics': self.reporter.calculate_metrics(),
                'anomalies_detected': len(list(Path('var/verification/incidents').glob('*'))),
                'recommendation': 'See PRODUCTION_PROMOTION_REPORT.md'
            }
            
            summary_path = Path('var/verification/watch/summary.json')
            summary_path.write_text(json.dumps(summary_data, indent=2), encoding='utf-8')
            
            print("📋 Watchdog period completed successfully!")
            self.running = False
            
        except Exception as e:
            print(f"❌ Final report failed: {e}")
    
    def setup_schedule(self):
        """Setup scheduled jobs"""
        # Snapshots every 10 minutes
        schedule.every(10).minutes.do(self.take_snapshot_job)
        
        # Heartbeat every hour
        schedule.every().hour.do(self.heartbeat_job)
        
        # T+6h report (schedule for 6 hours from start)
        target_6h = self.start_time + timedelta(hours=6)
        schedule.every().day.at(target_6h.strftime('%H:%M')).do(self.generate_t6h_report).tag('t6h')
        
        # T+24h report  
        target_24h = self.start_time + timedelta(hours=24)
        schedule.every().day.at(target_24h.strftime('%H:%M')).do(self.generate_t24h_report).tag('t24h')
        
        # Final report at T+48h
        target_48h = self.start_time + timedelta(hours=48)
        schedule.every().day.at(target_48h.strftime('%H:%M')).do(self.generate_final_report).tag('final')
        
        print(f"📅 Schedule configured:")
        print(f"   • Snapshots: Every 10 minutes")
        print(f"   • Heartbeat: Every hour") 
        print(f"   • T+6h report: {target_6h.strftime('%Y-%m-%d %H:%M')}")
        print(f"   • T+24h report: {target_24h.strftime('%Y-%m-%d %H:%M')}")
        print(f"   • Final report: {target_48h.strftime('%Y-%m-%d %H:%M')}")
    
    def run(self):
        """Main execution loop"""
        print("🎯 Starting continuous watchdog monitoring...")
        print(f"⏰ Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.setup_schedule()
        
        # Take initial snapshot
        self.take_snapshot_job()
        self.heartbeat_job()
        
        # Main monitoring loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
                # Check if 48 hours elapsed (fallback)
                elapsed = datetime.now() - self.start_time
                if elapsed >= timedelta(hours=48):
                    print("⏰ 48-hour period completed - generating final report")
                    self.generate_final_report()
                    break
                    
            except KeyboardInterrupt:
                print("\n⚡ Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait a minute before retry
        
        print("📛 Continuous watchdog monitoring stopped")

def main():
    """Main entry point"""
    watchdog = ContinuousWatchdog()
    
    print("=" * 60)
    print("🎯 ShivX 48-Hour Continuous Watchdog")
    print("=" * 60)
    
    try:
        watchdog.run()
    except Exception as e:
        print(f"💥 Fatal error: {e}")
    finally:
        print("🏁 Watchdog monitoring session ended")

if __name__ == "__main__":
    main()