#!/usr/bin/env python3
"""
ShivX Staging Environment Monitor
Continuously monitors staging environment and logs anomalies for production promotion assessment
"""

import json
import time
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import psutil
from typing import Dict, List, Any
import threading
import signal
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('var/staging/watchdog.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('staging_monitor')

class StagingMonitor:
    """Monitor staging environment for anomalies during 48-hour watchdog period"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.watchdog_duration = timedelta(hours=48)
        self.health_endpoint = "http://localhost:8080/health"
        self.status_endpoint = "http://localhost:8080/status"
        self.running = True
        self.anomalies = []
        
        # Monitoring thresholds
        self.thresholds = {
            'response_time_ms': 5000,  # 5 second max response time
            'error_rate_percent': 1.0,  # 1% max error rate
            'memory_usage_percent': 85,  # 85% max memory usage
            'cpu_usage_percent': 90,   # 90% max CPU usage
            'disk_usage_percent': 90,  # 90% max disk usage
            'consecutive_failures': 3   # Max consecutive health check failures
        }
        
        self.stats = {
            'total_checks': 0,
            'failed_checks': 0,
            'anomalies_detected': 0,
            'consecutive_failures': 0,
            'max_response_time': 0,
            'avg_response_time': 0,
            'uptime_seconds': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def check_health_endpoint(self) -> Dict[str, Any]:
        """Check health endpoint and return metrics"""
        try:
            start_time = time.time()
            response = requests.get(self.health_endpoint, timeout=10)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            self.stats['total_checks'] += 1
            self.stats['max_response_time'] = max(self.stats['max_response_time'], response_time)
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_checks'] - 1) + response_time) 
                / self.stats['total_checks']
            )
            
            if response.status_code == 200:
                self.stats['consecutive_failures'] = 0
                data = response.json()
                return {
                    'status': 'healthy',
                    'response_time_ms': response_time,
                    'data': data
                }
            else:
                self.stats['failed_checks'] += 1
                self.stats['consecutive_failures'] += 1
                return {
                    'status': 'unhealthy',
                    'response_time_ms': response_time,
                    'status_code': response.status_code,
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            self.stats['failed_checks'] += 1
            self.stats['consecutive_failures'] += 1
            return {
                'status': 'failed',
                'response_time_ms': None,
                'error': str(e)
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory_available_gb': memory.available / (1024**3)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_growth_protocol(self) -> Dict[str, Any]:
        """Test growth protocol functionality"""
        try:
            result = subprocess.run(
                [sys.executable, 'cli/shivx.py', 'growth:status'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                return {
                    'status': 'operational',
                    'output': result.stdout[:500]  # Limit output size
                }
            else:
                return {
                    'status': 'failed',
                    'error': result.stderr[:500]
                }
                
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'error': 'Growth protocol check timed out'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def detect_anomalies(self, health_result: Dict[str, Any], resources: Dict[str, Any]) -> List[str]:
        """Detect anomalies based on monitoring data"""
        anomalies = []
        
        # Check response time
        if health_result.get('response_time_ms', 0) > self.thresholds['response_time_ms']:
            anomalies.append(f"High response time: {health_result['response_time_ms']:.1f}ms")
        
        # Check consecutive failures
        if self.stats['consecutive_failures'] >= self.thresholds['consecutive_failures']:
            anomalies.append(f"Consecutive health check failures: {self.stats['consecutive_failures']}")
        
        # Check error rate
        if self.stats['total_checks'] > 0:
            error_rate = (self.stats['failed_checks'] / self.stats['total_checks']) * 100
            if error_rate > self.thresholds['error_rate_percent']:
                anomalies.append(f"High error rate: {error_rate:.1f}%")
        
        # Check resource usage
        if resources.get('cpu_percent', 0) > self.thresholds['cpu_usage_percent']:
            anomalies.append(f"High CPU usage: {resources['cpu_percent']:.1f}%")
            
        if resources.get('memory_percent', 0) > self.thresholds['memory_usage_percent']:
            anomalies.append(f"High memory usage: {resources['memory_percent']:.1f}%")
            
        if resources.get('disk_percent', 0) > self.thresholds['disk_usage_percent']:
            anomalies.append(f"High disk usage: {resources['disk_percent']:.1f}%")
        
        return anomalies
    
    def log_anomaly(self, anomaly: str, context: Dict[str, Any]):
        """Log anomaly and update checklist"""
        timestamp = datetime.now().isoformat()
        anomaly_record = {
            'timestamp': timestamp,
            'anomaly': anomaly,
            'context': context
        }
        
        self.anomalies.append(anomaly_record)
        self.stats['anomalies_detected'] += 1
        
        logger.warning(f"ANOMALY DETECTED: {anomaly}")
        
        # Update production checklist with anomaly
        try:
            self.update_checklist_with_anomaly(anomaly_record)
        except Exception as e:
            logger.error(f"Failed to update checklist: {e}")
    
    def update_checklist_with_anomaly(self, anomaly_record: Dict[str, Any]):
        """Update production checklist with detected anomaly"""
        checklist_path = Path('PRODUCTION_PROMOTION_CHECKLIST.md')
        
        if not checklist_path.exists():
            return
        
        content = checklist_path.read_text(encoding='utf-8')
        
        # Add staging watchdog notes section if it doesn't exist
        if "## 🔍 **Staging Watchdog Notes**" not in content:
            watchdog_section = """
---

## 🔍 **Staging Watchdog Notes**

*Auto-generated during 48-hour staging monitoring period*

### **Anomalies Detected**
"""
            content = content.replace("---\n\n**🎉 ShivX 2.0.0 is PRODUCTION-READY!**", 
                                    watchdog_section + "\n---\n\n**🎉 ShivX 2.0.0 is PRODUCTION-READY!**")
        
        # Add the new anomaly
        anomaly_entry = f"""
**{anomaly_record['timestamp']}**
- ⚠️ {anomaly_record['anomaly']}
- Context: {json.dumps(anomaly_record['context'], indent=2)}
"""
        
        # Insert before the final section
        content = content.replace("---\n\n**🎉 ShivX 2.0.0 is PRODUCTION-READY!**",
                                f"{anomaly_entry}\n---\n\n**🎉 ShivX 2.0.0 is PRODUCTION-READY!**")
        
        checklist_path.write_text(content, encoding='utf-8')
        logger.info("Updated production checklist with anomaly")
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate current monitoring status report"""
        elapsed = datetime.now() - self.start_time
        remaining = self.watchdog_duration - elapsed
        
        return {
            'watchdog_status': {
                'started': self.start_time.isoformat(),
                'elapsed_hours': elapsed.total_seconds() / 3600,
                'remaining_hours': max(0, remaining.total_seconds() / 3600),
                'completion_percent': min(100, (elapsed.total_seconds() / self.watchdog_duration.total_seconds()) * 100)
            },
            'health_stats': self.stats,
            'anomalies_detected': len(self.anomalies),
            'current_status': 'monitoring' if remaining.total_seconds() > 0 else 'completed'
        }
    
    def run_monitoring_cycle(self):
        """Run a single monitoring cycle"""
        logger.info("Running monitoring cycle...")
        
        # Check health endpoint
        health_result = self.check_health_endpoint()
        
        # Check system resources  
        resources = self.check_system_resources()
        
        # Check growth protocol (every 10th cycle to avoid spam)
        growth_result = None
        if self.stats['total_checks'] % 10 == 0:
            growth_result = self.check_growth_protocol()
        
        # Detect anomalies
        anomalies = self.detect_anomalies(health_result, resources)
        
        # Log any anomalies found
        for anomaly in anomalies:
            context = {
                'health': health_result,
                'resources': resources,
                'growth': growth_result,
                'stats': self.stats
            }
            self.log_anomaly(anomaly, context)
        
        # Log normal status periodically
        if self.stats['total_checks'] % 60 == 0:  # Every hour
            status = self.generate_status_report()
            logger.info(f"Hourly status: {status['watchdog_status']['completion_percent']:.1f}% complete, "
                       f"{len(self.anomalies)} anomalies detected")
    
    def run(self):
        """Run the staging monitor for the full watchdog period"""
        logger.info("Starting 48-hour staging monitoring...")
        logger.info(f"Monitoring will complete at: {self.start_time + self.watchdog_duration}")
        
        try:
            while self.running:
                current_time = datetime.now()
                elapsed = current_time - self.start_time
                
                # Check if watchdog period is complete
                if elapsed >= self.watchdog_duration:
                    logger.info("48-hour watchdog period completed successfully!")
                    break
                
                # Run monitoring cycle
                self.run_monitoring_cycle()
                
                # Update uptime stats
                self.stats['uptime_seconds'] = elapsed.total_seconds()
                
                # Sleep for 60 seconds between cycles
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        except Exception as e:
            logger.error(f"Monitoring failed with error: {e}")
            self.log_anomaly(f"Monitor crashed: {str(e)}", {'exception': str(e)})
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of monitoring"""
        logger.info("Shutting down staging monitor...")
        
        # Generate final status report
        final_status = self.generate_status_report()
        
        # Save final monitoring report
        report_path = Path('var/staging/final_monitoring_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'monitoring_period': {
                'start': self.start_time.isoformat(),
                'end': datetime.now().isoformat(),
                'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            },
            'final_status': final_status,
            'anomalies': self.anomalies,
            'statistics': self.stats
        }
        
        report_path.write_text(json.dumps(report_data, indent=2), encoding='utf-8')
        logger.info(f"Final monitoring report saved to: {report_path}")
        
        # Log summary
        logger.info(f"MONITORING SUMMARY:")
        logger.info(f"  Total checks: {self.stats['total_checks']}")
        logger.info(f"  Failed checks: {self.stats['failed_checks']}")
        logger.info(f"  Anomalies detected: {len(self.anomalies)}")
        logger.info(f"  Average response time: {self.stats['avg_response_time']:.1f}ms")
        logger.info(f"  Uptime: {self.stats['uptime_seconds'] / 3600:.1f} hours")

def main():
    """Main entry point"""
    monitor = StagingMonitor()
    monitor.run()

if __name__ == "__main__":
    main()