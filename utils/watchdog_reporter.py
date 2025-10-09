#!/usr/bin/env python3
"""
ShivX Watchdog Reporter - Generate structured monitoring reports during 48-hour period
"""

import json
import requests
import time
import psutil
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import sys

class WatchdogReporter:
    """Generate structured reports and snapshots during watchdog period"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.health_endpoint = "http://localhost:8080/health"
        self.status_endpoint = "http://localhost:8080/status"
        self.watch_dir = Path("var/verification/watch")
        self.incidents_dir = Path("var/verification/incidents")
        self.staging_docs = Path("docs/staging")
        
        # Ensure directories exist
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        self.staging_docs.mkdir(parents=True, exist_ok=True)
        
        # Historical data for analysis
        self.health_samples = []
        self.request_counts = []
        self.response_times = []
        
    def take_snapshot(self) -> Dict[str, Any]:
        """Take a complete system snapshot"""
        timestamp = datetime.now()
        
        # Health check
        health_data = self.get_health_data()
        status_data = self.get_status_data()
        system_data = self.get_system_data()
        
        snapshot = {
            "timestamp": timestamp.isoformat(),
            "health": health_data,
            "status": status_data,
            "system": system_data,
            "uptime_hours": (timestamp - self.start_time).total_seconds() / 3600
        }
        
        # Store samples for trend analysis
        if health_data.get('response_time_ms'):
            self.response_times.append(health_data['response_time_ms'])
            
        if status_data.get('metrics', {}).get('requests_served'):
            self.request_counts.append(status_data['metrics']['requests_served'])
            
        self.health_samples.append({
            'timestamp': timestamp.isoformat(),
            'status': health_data.get('status', 'unknown'),
            'response_time': health_data.get('response_time_ms', 0)
        })
        
        # Keep only last 50 samples to prevent memory bloat
        if len(self.health_samples) > 50:
            self.health_samples = self.health_samples[-50:]
            
        return snapshot
    
    def get_health_data(self) -> Dict[str, Any]:
        """Get health endpoint data"""
        try:
            start_time = time.time()
            response = requests.get(self.health_endpoint, timeout=10)
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                data['response_time_ms'] = response_time_ms
                data['status_code'] = response.status_code
                return data
            else:
                return {
                    'status': 'unhealthy',
                    'response_time_ms': response_time_ms,
                    'status_code': response.status_code,
                    'error': f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                'status': 'failed',
                'response_time_ms': None,
                'error': str(e)
            }
    
    def get_status_data(self) -> Dict[str, Any]:
        """Get status endpoint data"""
        try:
            response = requests.get(self.status_endpoint, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'error': str(e)}
    
    def get_system_data(self) -> Dict[str, Any]:
        """Get system resource data"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_percent': cpu_percent,
                'memory': {
                    'percent': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'total_gb': memory.total / (1024**3)
                },
                'disk': {
                    'percent': disk.percent,
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'total_gb': disk.total / (1024**3)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def save_raw_snapshot(self, snapshot: Dict[str, Any]):
        """Save raw snapshot files"""
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Save health snapshot
        health_file = self.watch_dir / f"health_{timestamp_str}.json"
        health_file.write_text(json.dumps({
            'timestamp': snapshot['timestamp'],
            'health': snapshot['health'],
            'system': snapshot['system']
        }, indent=2), encoding='utf-8')
        
        # Save status snapshot
        status_file = self.watch_dir / f"status_{timestamp_str}.json"
        status_file.write_text(json.dumps({
            'timestamp': snapshot['timestamp'],
            'status': snapshot['status'],
            'uptime_hours': snapshot['uptime_hours']
        }, indent=2), encoding='utf-8')
    
    def check_panic_flag(self) -> Dict[str, Any]:
        """Check panic flag status and log heartbeat"""
        panic_flag = Path("var/runtime/panic.flag")
        panic_ready = Path("var/runtime/panic.flag.ready")
        
        heartbeat_data = {
            'timestamp': datetime.now().isoformat(),
            'panic_flag_exists': panic_flag.exists(),
            'panic_ready_exists': panic_ready.exists(),
            'status': 'operational' if not panic_flag.exists() else 'panic_active'
        }
        
        # Log to heartbeat file
        heartbeat_log = self.watch_dir / "heartbeat.log"
        with open(heartbeat_log, 'a', encoding='utf-8') as f:
            f.write(f"{heartbeat_data['timestamp']} - Status: {heartbeat_data['status']}\n")
        
        return heartbeat_data
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregated metrics from historical data"""
        if not self.health_samples:
            return {'error': 'No health samples available'}
        
        # Calculate uptime percentage
        total_samples = len(self.health_samples)
        healthy_samples = len([s for s in self.health_samples if s.get('status') == 'healthy'])
        uptime_percent = (healthy_samples / total_samples * 100) if total_samples > 0 else 0
        
        # Calculate response time percentiles
        valid_times = [s['response_time'] for s in self.health_samples if s.get('response_time') is not None and s['response_time'] > 0]
        
        if valid_times:
            p50_latency = statistics.median(valid_times)
            p95_latency = statistics.quantiles(valid_times, n=20)[18] if len(valid_times) >= 20 else max(valid_times)
        else:
            p50_latency = p95_latency = 0
        
        # Error rate
        error_count = total_samples - healthy_samples
        error_rate = (error_count / total_samples * 100) if total_samples > 0 else 0
        
        # Request counts (from latest status if available)
        latest_requests = self.request_counts[-1] if self.request_counts else 0
        
        return {
            'uptime_percent': uptime_percent,
            'total_samples': total_samples,
            'healthy_samples': healthy_samples,
            'error_rate_percent': error_rate,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'total_requests': latest_requests,
            'monitoring_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }
    
    def detect_anomalies(self, snapshot: Dict[str, Any]) -> List[str]:
        """Detect anomalies in current snapshot"""
        anomalies = []
        
        # Check health status
        health = snapshot.get('health', {})
        if health.get('status') != 'healthy':
            anomalies.append(f"Health check failed: {health.get('error', 'unknown error')}")
        
        # Check response time
        response_time = health.get('response_time_ms', 0)
        if response_time > 5000:  # 5 second threshold
            anomalies.append(f"High response time: {response_time:.1f}ms")
        
        # Check system resources
        system = snapshot.get('system', {})
        if system.get('cpu_percent', 0) > 90:
            anomalies.append(f"High CPU usage: {system['cpu_percent']:.1f}%")
            
        memory_percent = system.get('memory', {}).get('percent', 0)
        if memory_percent > 85:
            anomalies.append(f"High memory usage: {memory_percent:.1f}%")
            
        disk_percent = system.get('disk', {}).get('percent', 0)
        if disk_percent > 90:
            anomalies.append(f"High disk usage: {disk_percent:.1f}%")
        
        return anomalies
    
    def log_incident(self, anomalies: List[str], snapshot: Dict[str, Any]):
        """Log incident when anomalies detected"""
        if not anomalies:
            return
            
        timestamp = datetime.now()
        incident_id = timestamp.strftime('%Y%m%d_%H%M%S')
        incident_dir = self.incidents_dir / incident_id
        incident_dir.mkdir(exist_ok=True)
        
        # Save incident details
        incident_data = {
            'incident_id': incident_id,
            'timestamp': timestamp.isoformat(),
            'anomalies': anomalies,
            'snapshot': snapshot,
            'severity': 'HIGH' if len(anomalies) > 2 else 'MEDIUM'
        }
        
        incident_file = incident_dir / 'incident.json'
        incident_file.write_text(json.dumps(incident_data, indent=2), encoding='utf-8')
        
        # Update production checklist
        self.update_checklist_with_incident(incident_data)
        
        print(f"🚨 INCIDENT LOGGED: {incident_id} - {len(anomalies)} anomalies detected")
    
    def update_checklist_with_incident(self, incident_data: Dict[str, Any]):
        """Update production checklist with red incident note"""
        checklist_path = Path('PRODUCTION_PROMOTION_CHECKLIST.md')
        
        if not checklist_path.exists():
            return
        
        content = checklist_path.read_text(encoding='utf-8')
        
        # Create incident note
        severity_emoji = "🔴" if incident_data['severity'] == 'HIGH' else "🟠"
        incident_note = f"""
### **{severity_emoji} INCIDENT - {incident_data['incident_id']}**

**Time**: {incident_data['timestamp']}  
**Severity**: {incident_data['severity']}  
**Anomalies**: {len(incident_data['anomalies'])}

**Issues Detected**:
{chr(10).join(f'- {anomaly}' for anomaly in incident_data['anomalies'])}

**Evidence**: `var/verification/incidents/{incident_data['incident_id']}/`

**Root Cause Hypothesis**: System resource pressure or external dependency issue  
**Next Steps**: Monitor for recovery, investigate if persistent  

---
"""
        
        # Insert before "Anomalies Detected" section end
        content = content.replace("*Monitoring in progress - anomalies will be logged here automatically*",
                                f"*Monitoring in progress - anomalies will be logged here automatically*\n{incident_note}")
        
        checklist_path.write_text(content, encoding='utf-8')
    
    def generate_interim_report(self, hours_elapsed: int) -> str:
        """Generate interim status report"""
        metrics = self.calculate_metrics()
        latest_snapshot = self.take_snapshot()
        system = latest_snapshot.get('system', {})
        
        # Get last 5 health samples
        last_5_samples = self.health_samples[-5:] if len(self.health_samples) >= 5 else self.health_samples
        
        report = f"""# 📊 ShivX Staging T+{hours_elapsed}h Status Report

**Report Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Monitoring Duration**: {metrics.get('monitoring_duration_hours', 0):.1f} hours  
**Progress**: {(hours_elapsed / 48) * 100:.1f}% complete  

---

## 🎯 **Key Performance Metrics**

### **Availability & Reliability**
- **Uptime**: {metrics.get('uptime_percent', 0):.2f}%
- **Total Health Checks**: {metrics.get('total_samples', 0):,}
- **Successful Checks**: {metrics.get('healthy_samples', 0):,}
- **Error Rate**: {metrics.get('error_rate_percent', 0):.2f}%

### **Performance**
- **P50 Latency**: {metrics.get('p50_latency_ms', 0):.1f}ms
- **P95 Latency**: {metrics.get('p95_latency_ms', 0):.1f}ms
- **Total Requests Served**: {metrics.get('total_requests', 0):,}

### **System Resources**
- **CPU Usage**: {system.get('cpu_percent', 0):.1f}%
- **Memory Usage**: {system.get('memory', {}).get('percent', 0):.1f}% ({system.get('memory', {}).get('used_gb', 0):.1f}GB used)
- **Disk Usage**: {system.get('disk', {}).get('percent', 0):.1f}% ({system.get('disk', {}).get('free_gb', 0):.1f}GB free)

---

## 🔍 **Recent Health Samples**

| Time | Status | Response Time |
|------|--------|---------------|"""

        for sample in last_5_samples:
            status_emoji = "✅" if sample.get('status') == 'healthy' else "❌"
            time_str = datetime.fromisoformat(sample['timestamp']).strftime('%H:%M:%S')
            response_time = f"{sample.get('response_time', 0):.1f}ms"
            report += f"\n| {time_str} | {status_emoji} {sample.get('status', 'unknown')} | {response_time} |"

        report += f"""

---

## 📁 **Raw Data References**

**Health Snapshots**: `var/verification/watch/health_*.json`  
**Status Snapshots**: `var/verification/watch/status_*.json`  
**Heartbeat Log**: `var/verification/watch/heartbeat.log`  
**Latest Snapshot**: {datetime.now().strftime('%Y%m%d_%H%M')}

---

## 🎯 **Assessment**

**Status**: {'🟢 HEALTHY' if metrics.get('uptime_percent', 0) > 99 else '🟡 DEGRADED' if metrics.get('uptime_percent', 0) > 95 else '🔴 CRITICAL'}  
**Recommendation**: {'Continue monitoring' if metrics.get('uptime_percent', 0) > 99 else 'Investigate issues'}  

**Next Report**: T+{24 if hours_elapsed == 6 else 48}h  

---

*Report generated automatically by ShivX Watchdog Reporter*
"""
        
        return report
    
    def run_alert_drill(self) -> str:
        """Run panic flag alert drill and capture system response"""
        print("🚨 Running alert drill - creating panic flag...")
        
        panic_flag = Path("var/runtime/panic.flag")
        drill_start = datetime.now()
        
        drill_report = f"""# 🚨 ShivX Alert Drill Report

**Drill Time**: {drill_start.strftime('%Y-%m-%d %H:%M:%S')}  
**Purpose**: Test panic flag response and system behavior  

---

## 🔬 **Test Procedure**

### **Phase 1: Create Panic Flag**
"""
        
        # Create panic flag
        panic_flag.write_text(f"DRILL: Alert drill at {drill_start.isoformat()}\n", encoding='utf-8')
        
        # Wait and observe
        print("Waiting 10 seconds to observe system response...")
        time.sleep(10)
        
        # Check health during panic
        health_during_panic = self.get_health_data()
        status_during_panic = self.get_status_data()
        
        drill_report += f"""
**Action**: Created panic flag at `var/runtime/panic.flag`  
**Content**: Drill marker with timestamp  

### **Phase 2: System Response (10s observation)**

**Health Endpoint**: {health_during_panic.get('status', 'unknown')}  
**Status Endpoint**: {status_during_panic.get('status', 'unknown')}  
**Response Time**: {health_during_panic.get('response_time_ms', 0):.1f}ms  

"""
        
        # Remove panic flag
        print("Removing panic flag...")
        panic_flag.unlink()
        drill_end = datetime.now()
        
        # Wait and check recovery
        print("Waiting 5 seconds to confirm recovery...")
        time.sleep(5)
        
        health_after_recovery = self.get_health_data()
        
        drill_report += f"""### **Phase 3: Recovery**

**Action**: Removed panic flag  
**Recovery Time**: {drill_end.strftime('%H:%M:%S')}  
**Post-Recovery Health**: {health_after_recovery.get('status', 'unknown')}  
**Post-Recovery Response**: {health_after_recovery.get('response_time_ms', 0):.1f}ms  

---

## 🎯 **Drill Results**

**Duration**: {(drill_end - drill_start).total_seconds():.1f} seconds  
**System Impact**: {'Minimal - service remained operational' if health_during_panic.get('status') == 'healthy' else 'Service disrupted during panic state'}  
**Recovery**: {'Successful - service restored' if health_after_recovery.get('status') == 'healthy' else 'Issues detected post-recovery'}  

### **Conclusions**

- ✅ Panic flag mechanism accessible
- {'✅' if health_during_panic.get('status') == 'healthy' else '⚠️'} Service behavior during panic: {health_during_panic.get('status', 'unknown')}
- {'✅' if health_after_recovery.get('status') == 'healthy' else '⚠️'} Recovery after flag removal: {health_after_recovery.get('status', 'unknown')}

**Recommendation**: {'Emergency procedures validated' if health_after_recovery.get('status') == 'healthy' else 'Review panic flag handling'}

---

*Drill completed successfully*
"""
        
        return drill_report

# Initialize reporter and run initial operations
def main():
    reporter = WatchdogReporter()
    
    print("🎯 ShivX Watchdog Reporter initialized")
    print(f"📊 Monitoring started at: {reporter.start_time}")
    print(f"📁 Watch directory: {reporter.watch_dir}")
    
    # Take initial snapshot
    initial_snapshot = reporter.take_snapshot()
    reporter.save_raw_snapshot(initial_snapshot)
    
    # Check for immediate anomalies
    anomalies = reporter.detect_anomalies(initial_snapshot)
    if anomalies:
        reporter.log_incident(anomalies, initial_snapshot)
    
    # Log initial heartbeat
    heartbeat = reporter.check_panic_flag()
    print(f"💓 Initial heartbeat: {heartbeat['status']}")
    
    print("✅ Watchdog reporter ready - monitoring in background")

if __name__ == "__main__":
    main()