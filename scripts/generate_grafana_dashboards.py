#!/usr/bin/env python3
"""
ShivX Grafana Dashboard Generator
==================================
Generates comprehensive Grafana dashboards for ShivX monitoring.

Dashboards created:
1. System Health (CPU, memory, disk, network)
2. API Performance (latency, throughput, errors)
3. Trading Metrics (positions, PnL, trades, signals)
4. Security Monitoring (auth failures, rate limits, Guardian events)
5. Database Performance (queries, connections, slow queries)
6. ML Model Performance (inference time, predictions, accuracy)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


class GrafanaDashboardGenerator:
    """Generates Grafana dashboard JSON files"""

    def __init__(self, output_dir: str = "./deploy/grafana/dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self):
        """Generate all dashboards"""
        print("Generating Grafana dashboards...")

        dashboards = [
            ("system-health", self.create_system_health_dashboard()),
            ("api-performance", self.create_api_performance_dashboard()),
            ("trading-metrics", self.create_trading_dashboard()),
            ("security-monitoring", self.create_security_dashboard()),
            ("database-performance", self.create_database_dashboard()),
            ("ml-model-performance", self.create_ml_dashboard()),
        ]

        for name, dashboard in dashboards:
            self.save_dashboard(name, dashboard)

        print(f"✓ Generated {len(dashboards)} dashboards in {self.output_dir}")

    def create_system_health_dashboard(self) -> Dict:
        """Create System Health dashboard"""
        return {
            "dashboard": {
                "title": "ShivX - System Health",
                "tags": ["shivx", "system", "health"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "30s",
                "panels": [
                    self.create_panel(
                        title="CPU Usage",
                        gridPos={"x": 0, "y": 0, "w": 8, "h": 8},
                        targets=[{
                            "expr": "rate(process_cpu_seconds_total{job='shivx'}[5m]) * 100",
                            "legendFormat": "CPU %",
                        }],
                        unit="percent",
                    ),
                    self.create_panel(
                        title="Memory Usage",
                        gridPos={"x": 8, "y": 0, "w": 8, "h": 8},
                        targets=[{
                            "expr": "process_resident_memory_bytes{job='shivx'} / 1024 / 1024",
                            "legendFormat": "Memory MB",
                        }],
                        unit="decmbytes",
                    ),
                    self.create_panel(
                        title="Disk Usage",
                        gridPos={"x": 16, "y": 0, "w": 8, "h": 8},
                        targets=[{
                            "expr": "(1 - node_filesystem_avail_bytes{mountpoint='/'} / node_filesystem_size_bytes{mountpoint='/'}) * 100",
                            "legendFormat": "Disk %",
                        }],
                        unit="percent",
                    ),
                    self.create_panel(
                        title="Network I/O",
                        gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                        targets=[
                            {
                                "expr": "rate(node_network_receive_bytes_total[5m])",
                                "legendFormat": "Receive {{device}}",
                            },
                            {
                                "expr": "rate(node_network_transmit_bytes_total[5m])",
                                "legendFormat": "Transmit {{device}}",
                            },
                        ],
                        unit="Bps",
                    ),
                    self.create_panel(
                        title="Container Restarts",
                        gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(container_restart_total[15m])) by (container)",
                            "legendFormat": "{{container}}",
                        }],
                        unit="short",
                    ),
                ],
            }
        }

    def create_api_performance_dashboard(self) -> Dict:
        """Create API Performance dashboard"""
        return {
            "dashboard": {
                "title": "ShivX - API Performance",
                "tags": ["shivx", "api", "performance"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "10s",
                "panels": [
                    self.create_panel(
                        title="Request Rate",
                        gridPos={"x": 0, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(http_requests_total{job='shivx'}[5m])) by (method, endpoint)",
                            "legendFormat": "{{method}} {{endpoint}}",
                        }],
                        unit="reqps",
                    ),
                    self.create_panel(
                        title="Error Rate",
                        gridPos={"x": 12, "y": 0, "w": 12, "h": 8},
                        targets=[
                            {
                                "expr": "sum(rate(http_requests_total{job='shivx',status=~'5..'}[5m]))",
                                "legendFormat": "5xx errors",
                            },
                            {
                                "expr": "sum(rate(http_requests_total{job='shivx',status=~'4..'}[5m]))",
                                "legendFormat": "4xx errors",
                            },
                        ],
                        unit="reqps",
                    ),
                    self.create_panel(
                        title="P50/P95/P99 Latency",
                        gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                        targets=[
                            {
                                "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job='shivx'}[5m])) by (le))",
                                "legendFormat": "P50",
                            },
                            {
                                "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job='shivx'}[5m])) by (le))",
                                "legendFormat": "P95",
                            },
                            {
                                "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job='shivx'}[5m])) by (le))",
                                "legendFormat": "P99",
                            },
                        ],
                        unit="s",
                    ),
                    self.create_panel(
                        title="Status Code Distribution",
                        gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(http_requests_total{job='shivx'}[5m])) by (status)",
                            "legendFormat": "{{status}}",
                        }],
                        unit="reqps",
                        type="piechart",
                    ),
                ],
            }
        }

    def create_trading_dashboard(self) -> Dict:
        """Create Trading Metrics dashboard"""
        return {
            "dashboard": {
                "title": "ShivX - Trading Metrics",
                "tags": ["shivx", "trading"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "15s",
                "panels": [
                    self.create_panel(
                        title="Cumulative PnL (USD)",
                        gridPos={"x": 0, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "trading_pnl_total_usd{job='shivx'}",
                            "legendFormat": "Total PnL",
                        }],
                        unit="currencyUSD",
                    ),
                    self.create_panel(
                        title="Active Positions",
                        gridPos={"x": 12, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "trading_active_positions{job='shivx'}",
                            "legendFormat": "Positions",
                        }],
                        unit="short",
                    ),
                    self.create_panel(
                        title="Trade Success Rate",
                        gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(trades_total{job='shivx',status='success'}[5m])) / sum(rate(trades_total{job='shivx'}[5m])) * 100",
                            "legendFormat": "Success %",
                        }],
                        unit="percent",
                    ),
                    self.create_panel(
                        title="Trade Volume (24h)",
                        gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(increase(trade_volume_usd{job='shivx'}[24h]))",
                            "legendFormat": "Volume USD",
                        }],
                        unit="currencyUSD",
                    ),
                    self.create_panel(
                        title="Trading Signals",
                        gridPos={"x": 0, "y": 16, "w": 24, "h": 8},
                        targets=[
                            {
                                "expr": "sum(rate(trading_signals_total{job='shivx',signal='buy'}[5m]))",
                                "legendFormat": "Buy signals",
                            },
                            {
                                "expr": "sum(rate(trading_signals_total{job='shivx',signal='sell'}[5m]))",
                                "legendFormat": "Sell signals",
                            },
                        ],
                        unit="short",
                    ),
                ],
            }
        }

    def create_security_dashboard(self) -> Dict:
        """Create Security Monitoring dashboard"""
        return {
            "dashboard": {
                "title": "ShivX - Security Monitoring",
                "tags": ["shivx", "security"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "10s",
                "panels": [
                    self.create_panel(
                        title="Failed Authentication Attempts",
                        gridPos={"x": 0, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(auth_failures_total{job='shivx'}[5m]))",
                            "legendFormat": "Failed auth/s",
                        }],
                        unit="short",
                    ),
                    self.create_panel(
                        title="Rate Limit Violations",
                        gridPos={"x": 12, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(rate_limit_exceeded_total{job='shivx'}[5m]))",
                            "legendFormat": "Rate limits/s",
                        }],
                        unit="short",
                    ),
                    self.create_panel(
                        title="Guardian Defense Status",
                        gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "guardian_defense_lockdown_active{job='shivx'}",
                            "legendFormat": "Lockdown Active",
                        }],
                        unit="bool",
                    ),
                    self.create_panel(
                        title="Security Incidents",
                        gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(security_incident_total{job='shivx'}[5m])) by (type)",
                            "legendFormat": "{{type}}",
                        }],
                        unit="short",
                    ),
                ],
            }
        }

    def create_database_dashboard(self) -> Dict:
        """Create Database Performance dashboard"""
        return {
            "dashboard": {
                "title": "ShivX - Database Performance",
                "tags": ["shivx", "database"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "30s",
                "panels": [
                    self.create_panel(
                        title="Database Connections",
                        gridPos={"x": 0, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "pg_stat_database_numbackends{job='postgres'}",
                            "legendFormat": "Active connections",
                        }],
                        unit="short",
                    ),
                    self.create_panel(
                        title="Query Rate",
                        gridPos={"x": 12, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "rate(pg_stat_database_xact_commit{job='postgres'}[5m])",
                            "legendFormat": "Commits/s",
                        }],
                        unit="short",
                    ),
                    self.create_panel(
                        title="Slow Queries (>1s)",
                        gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "pg_slow_queries_total{job='postgres'}",
                            "legendFormat": "Slow queries",
                        }],
                        unit="short",
                    ),
                    self.create_panel(
                        title="Database Size",
                        gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "pg_database_size_bytes{job='postgres'} / 1024 / 1024 / 1024",
                            "legendFormat": "Size GB",
                        }],
                        unit="decgbytes",
                    ),
                ],
            }
        }

    def create_ml_dashboard(self) -> Dict:
        """Create ML Model Performance dashboard"""
        return {
            "dashboard": {
                "title": "ShivX - ML Model Performance",
                "tags": ["shivx", "ml", "ai"],
                "timezone": "browser",
                "schemaVersion": 16,
                "version": 0,
                "refresh": "30s",
                "panels": [
                    self.create_panel(
                        title="Model Inference Time",
                        gridPos={"x": 0, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "histogram_quantile(0.95, sum(rate(ml_inference_duration_seconds_bucket{job='shivx'}[5m])) by (le, model))",
                            "legendFormat": "P95 {{model}}",
                        }],
                        unit="s",
                    ),
                    self.create_panel(
                        title="Predictions per Second",
                        gridPos={"x": 12, "y": 0, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(ml_predictions_total{job='shivx'}[5m])) by (model)",
                            "legendFormat": "{{model}}",
                        }],
                        unit="short",
                    ),
                    self.create_panel(
                        title="Model Accuracy",
                        gridPos={"x": 0, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "ml_model_accuracy{job='shivx'}",
                            "legendFormat": "{{model}}",
                        }],
                        unit="percentunit",
                    ),
                    self.create_panel(
                        title="Prediction Errors",
                        gridPos={"x": 12, "y": 8, "w": 12, "h": 8},
                        targets=[{
                            "expr": "sum(rate(ml_prediction_errors_total{job='shivx'}[5m])) by (model, error_type)",
                            "legendFormat": "{{model}} - {{error_type}}",
                        }],
                        unit="short",
                    ),
                ],
            }
        }

    def create_panel(self, title: str, gridPos: Dict, targets: List[Dict],
                    unit: str = "short", type: str = "graph") -> Dict:
        """Create a dashboard panel"""
        return {
            "title": title,
            "type": type,
            "gridPos": gridPos,
            "targets": [
                {
                    **target,
                    "refId": chr(65 + i),  # A, B, C, ...
                }
                for i, target in enumerate(targets)
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "color": {"mode": "palette-classic"},
                },
            },
            "options": {
                "legend": {"displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "none"},
            },
        }

    def save_dashboard(self, name: str, dashboard: Dict):
        """Save dashboard to JSON file"""
        filepath = self.output_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(dashboard, f, indent=2)
        print(f"  ✓ {name}.json")


def main():
    """Main entry point"""
    generator = GrafanaDashboardGenerator()
    generator.generate_all()
    print("\n✓ All Grafana dashboards generated successfully!")
    print("\nDashboards will be automatically loaded by Grafana on startup.")


if __name__ == "__main__":
    main()
