#!/usr/bin/env python3
"""
Generate final production promotion report after 48-hour watchdog period
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

def load_monitoring_data() -> Dict[str, Any]:
    """Load final monitoring data from staging monitor"""
    report_path = Path('var/staging/final_monitoring_report.json')
    
    if not report_path.exists():
        return {'error': 'Final monitoring report not found'}
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {'error': f'Failed to load monitoring report: {e}'}

def analyze_anomalies(monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze anomalies and categorize by severity"""
    if 'anomalies' not in monitoring_data:
        return {'total': 0, 'critical': [], 'high': [], 'medium': [], 'low': []}
    
    anomalies = monitoring_data['anomalies']
    categorized = {'total': len(anomalies), 'critical': [], 'high': [], 'medium': [], 'low': []}
    
    for anomaly in anomalies:
        description = anomaly.get('anomaly', '').lower()
        
        # Critical: system crashes, consecutive failures > 5
        if 'crash' in description or 'consecutive failures' in description:
            context = anomaly.get('context', {})
            if context.get('stats', {}).get('consecutive_failures', 0) > 5:
                categorized['critical'].append(anomaly)
            else:
                categorized['high'].append(anomaly)
        
        # High: high error rates, long response times
        elif 'high error rate' in description or 'high response time' in description:
            categorized['high'].append(anomaly)
        
        # Medium: resource usage issues
        elif 'high cpu' in description or 'high memory' in description or 'high disk' in description:
            categorized['medium'].append(anomaly)
        
        # Low: everything else
        else:
            categorized['low'].append(anomaly)
    
    return categorized

def calculate_uptime_percentage(monitoring_data: Dict[str, Any]) -> float:
    """Calculate uptime percentage from monitoring data"""
    stats = monitoring_data.get('statistics', {})
    total_checks = stats.get('total_checks', 0)
    failed_checks = stats.get('failed_checks', 0)
    
    if total_checks == 0:
        return 0.0
    
    return ((total_checks - failed_checks) / total_checks) * 100

def determine_promotion_readiness(monitoring_data: Dict[str, Any], anomalies: Dict[str, Any]) -> Dict[str, Any]:
    """Determine if system is ready for production promotion"""
    
    # Success criteria
    criteria = {
        'uptime_min': 99.0,          # 99% uptime minimum
        'critical_anomalies_max': 0, # 0 critical anomalies
        'high_anomalies_max': 5,     # Max 5 high severity anomalies
        'avg_response_time_max': 2000, # 2 second max average response
        'error_rate_max': 1.0        # 1% max error rate
    }
    
    # Calculate metrics
    uptime = calculate_uptime_percentage(monitoring_data)
    stats = monitoring_data.get('statistics', {})
    avg_response_time = stats.get('avg_response_time', 0)
    total_checks = stats.get('total_checks', 0)
    failed_checks = stats.get('failed_checks', 0)
    error_rate = (failed_checks / total_checks * 100) if total_checks > 0 else 0
    
    # Check each criterion
    checks = {
        'uptime': {
            'value': uptime,
            'threshold': criteria['uptime_min'],
            'passed': uptime >= criteria['uptime_min'],
            'description': f"Uptime {uptime:.2f}% >= {criteria['uptime_min']}%"
        },
        'critical_anomalies': {
            'value': len(anomalies['critical']),
            'threshold': criteria['critical_anomalies_max'],
            'passed': len(anomalies['critical']) <= criteria['critical_anomalies_max'],
            'description': f"Critical anomalies {len(anomalies['critical'])} <= {criteria['critical_anomalies_max']}"
        },
        'high_anomalies': {
            'value': len(anomalies['high']),
            'threshold': criteria['high_anomalies_max'],
            'passed': len(anomalies['high']) <= criteria['high_anomalies_max'],
            'description': f"High anomalies {len(anomalies['high'])} <= {criteria['high_anomalies_max']}"
        },
        'avg_response_time': {
            'value': avg_response_time,
            'threshold': criteria['avg_response_time_max'],
            'passed': avg_response_time <= criteria['avg_response_time_max'],
            'description': f"Avg response {avg_response_time:.1f}ms <= {criteria['avg_response_time_max']}ms"
        },
        'error_rate': {
            'value': error_rate,
            'threshold': criteria['error_rate_max'],
            'passed': error_rate <= criteria['error_rate_max'],
            'description': f"Error rate {error_rate:.2f}% <= {criteria['error_rate_max']}%"
        }
    }
    
    # Overall decision
    all_passed = all(check['passed'] for check in checks.values())
    blocking_failures = [name for name, check in checks.items() if not check['passed']]
    
    return {
        'ready_for_production': all_passed,
        'blocking_issues': blocking_failures,
        'criteria_checks': checks,
        'recommendation': 'APPROVE' if all_passed else 'REJECT',
        'next_steps': 'Production deployment authorized' if all_passed else 'Address blocking issues before promotion'
    }

def generate_report() -> str:
    """Generate the final production promotion report"""
    
    # Load monitoring data
    monitoring_data = load_monitoring_data()
    
    if 'error' in monitoring_data:
        return f"ERROR: {monitoring_data['error']}"
    
    # Analyze data
    anomalies = analyze_anomalies(monitoring_data)
    readiness = determine_promotion_readiness(monitoring_data, anomalies)
    
    # Get time data
    monitoring_period = monitoring_data.get('monitoring_period', {})
    start_time = monitoring_period.get('start', 'Unknown')
    end_time = monitoring_period.get('end', 'Unknown')
    duration_hours = monitoring_period.get('duration_hours', 0)
    
    # Get statistics
    stats = monitoring_data.get('statistics', {})
    
    # Generate report
    status_icon = "✅" if readiness['ready_for_production'] else "❌"
    recommendation = readiness['recommendation']
    
    report = f"""# 🎯 ShivX Final Production Promotion Report

**Assessment Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Monitoring Period**: {start_time} → {end_time}  
**Duration**: {duration_hours:.1f} hours  
**Status**: {status_icon} **{recommendation}**  

---

## 📊 **48-Hour Watchdog Summary**

### **System Performance Metrics**
- **Total Health Checks**: {stats.get('total_checks', 0):,}
- **Failed Checks**: {stats.get('failed_checks', 0):,}
- **Uptime**: {calculate_uptime_percentage(monitoring_data):.2f}%
- **Average Response Time**: {stats.get('avg_response_time', 0):.1f}ms
- **Maximum Response Time**: {stats.get('max_response_time', 0):.1f}ms
- **Error Rate**: {(stats.get('failed_checks', 0) / stats.get('total_checks', 1) * 100):.2f}%

### **Anomaly Analysis**
- **Total Anomalies Detected**: {anomalies['total']}
- **Critical Severity**: {len(anomalies['critical'])} 🔴
- **High Severity**: {len(anomalies['high'])} 🟠  
- **Medium Severity**: {len(anomalies['medium'])} 🟡
- **Low Severity**: {len(anomalies['low'])} 🟢

---

## 🎯 **Production Readiness Assessment**

| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|"""

    # Add criteria table
    for name, check in readiness['criteria_checks'].items():
        status_emoji = "✅" if check['passed'] else "❌"
        report += f"\n| {name.replace('_', ' ').title()} | {check['value']:.2f} | {check['threshold']} | {status_emoji} |"

    # Continue report
    report += f"""

### **Overall Assessment**: {status_icon} **{recommendation}**

{readiness['next_steps']}

"""

    # Add blocking issues if any
    if readiness['blocking_issues']:
        report += f"""### **⚠️ Blocking Issues**

The following criteria must be addressed before production promotion:

"""
        for issue in readiness['blocking_issues']:
            check = readiness['criteria_checks'][issue]
            report += f"- **{issue.replace('_', ' ').title()}**: {check['description']} ❌\n"

    # Add anomaly details if significant
    if anomalies['critical'] or len(anomalies['high']) > 0:
        report += f"""
---

## 🔍 **Significant Anomalies**

"""
        # Critical anomalies
        if anomalies['critical']:
            report += """### **🔴 Critical Anomalies**

"""
            for i, anomaly in enumerate(anomalies['critical'][:5], 1):  # Limit to 5 most recent
                report += f"""**{i}. {anomaly['timestamp']}**
- {anomaly['anomaly']}
- Context: System impact detected

"""

        # High severity anomalies  
        if anomalies['high']:
            report += """### **🟠 High Severity Anomalies**

"""
            for i, anomaly in enumerate(anomalies['high'][:3], 1):  # Limit to 3 most recent
                report += f"""**{i}. {anomaly['timestamp']}**
- {anomaly['anomaly']}

"""

    # Production recommendation
    if readiness['ready_for_production']:
        report += f"""
---

## 🚀 **Production Deployment Authorization**

### **✅ APPROVED FOR PRODUCTION**

**Justification:**
- 48-hour watchdog period completed successfully
- All production readiness criteria met
- System demonstrated stability and reliability
- No critical anomalies detected
- Performance within acceptable thresholds

### **Recommended Next Steps:**
1. ✅ **Schedule Production Deployment**
2. ✅ **Prepare Production Environment** 
3. ✅ **Execute Go-Live Checklist**
4. ✅ **Monitor First 24 Hours**
5. ✅ **Complete Post-Deployment Verification**

### **Production Deployment Cleared**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    else:
        report += f"""
---

## ⚠️ **Production Deployment Hold**

### **❌ NOT READY FOR PRODUCTION**

**Issues Identified:**
{chr(10).join(f"- {issue.replace('_', ' ').title()}" for issue in readiness['blocking_issues'])}

### **Required Actions Before Promotion:**
1. 🔧 **Address all blocking issues listed above**
2. 🔍 **Investigate and resolve anomalies**
3. 🧪 **Re-run watchdog period if critical issues found**
4. 📊 **Verify all metrics meet production criteria**
5. 📋 **Update this assessment after fixes**

### **Next Assessment**: After issues are resolved
"""

    # Footer
    report += f"""
---

## 📋 **Documentation References**

- **Staging Environment**: http://localhost:8080/
- **Monitoring Logs**: `var/staging/watchdog.log`
- **Rollback Procedures**: `ROLLBACK_GUIDE.md`
- **Production Checklist**: `PRODUCTION_PROMOTION_CHECKLIST.md`

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Assessment By**: Automated Staging Monitor  
**Next Review**: Post-production deployment  

**🎉 ShivX Ready for Production!** 🚀 """ if readiness['ready_for_production'] else "**⚠️ Production Deployment Requires Attention**"

    return report

def main():
    """Main entry point"""
    report = generate_report()
    
    # Save report
    report_path = Path('PRODUCTION_PROMOTION_REPORT.md')
    report_path.write_text(report, encoding='utf-8')
    
    print(f"Production promotion report generated: {report_path}")
    print(f"Report length: {len(report.split())} words")

if __name__ == "__main__":
    main()