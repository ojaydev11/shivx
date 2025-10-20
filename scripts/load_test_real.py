"""
ShivX Real Load Test Harness
============================
Purpose: Execute real load tests against running ShivX instance
Uses: httpx + asyncio for actual HTTP requests
"""

import asyncio
import httpx
import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse
import psutil
import sys

@dataclass
class LoadTestMetrics:
    """Metrics collected during load test"""
    profile: str
    name: str
    config: Dict[str, Any]
    started_at: str
    completed_at: str
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies_ms: List[float]
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    success_rate: float
    error_rate: float
    requests_per_second: float
    cpu_samples: List[float]
    cpu_avg_percent: float
    cpu_max_percent: float
    ram_samples_mb: List[float]
    ram_avg_mb: float
    ram_max_mb: float
    errors: List[Dict[str, Any]]
    status: str  # PASS/FAIL

@dataclass
class LoadTestProfile:
    """Load test profile configuration"""
    key: str
    name: str
    agents: int
    tasks_per_min: int
    duration_min: int
    description: str
    endpoints: List[str]
    cycles: int = 1

# Define test profiles (same as baseline)
PROFILES = {
    "P1": LoadTestProfile(
        key="P1",
        name="Baseline",
        agents=2,
        tasks_per_min=10,
        duration_min=1,  # Reduced to 1 min for faster testing
        description="Baseline performance measurement",
        endpoints=["/api/health/live", "/api/health/status", "/api/health/details"]
    ),
    "P2": LoadTestProfile(
        key="P2",
        name="Concurrency",
        agents=15,
        tasks_per_min=100,
        duration_min=2,  # Reduced to 2 min
        description="High concurrency stress test",
        endpoints=["/api/health/live", "/api/health/status", "/api/health/ready", "/api/health/check"]
    ),
    "P3": LoadTestProfile(
        key="P3",
        name="Spike",
        agents=50,
        tasks_per_min=500,
        duration_min=1,  # Short burst
        description="Spike test 0-100 tasks in 10s 5x cycles",
        endpoints=["/api/health/live", "/api/health/status"],
        cycles=5
    ),
    "P4": LoadTestProfile(
        key="P4",
        name="Soak",
        agents=10,
        tasks_per_min=30,
        duration_min=5,  # Reduced to 5 min (abbreviated from 120 min)
        description="Soak test extended duration (abbreviated)",
        endpoints=["/api/health/live", "/api/health/status", "/api/health/details"]
    ),
    "P5": LoadTestProfile(
        key="P5",
        name="GPU Mix",
        agents=5,
        tasks_per_min=20,
        duration_min=2,  # Reduced to 2 min
        description="GPU-intensive mix (simulated with mixed endpoints)",
        endpoints=["/api/health/check", "/api/health/ready", "/api/health/details"]
    ),
}

class ResourceMonitor:
    """Monitor CPU/RAM during test"""
    def __init__(self):
        self.cpu_samples = []
        self.ram_samples = []
        self.running = False

    async def start(self):
        """Start monitoring resources"""
        self.running = True
        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=0.1)
                ram = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                self.cpu_samples.append(cpu)
                self.ram_samples.append(ram)
                await asyncio.sleep(1.0)  # Sample every second
            except Exception:
                break

    def stop(self):
        """Stop monitoring"""
        self.running = False

    def get_metrics(self):
        """Get collected metrics"""
        return {
            "cpu_samples": self.cpu_samples,
            "cpu_avg": statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0,
            "cpu_max": max(self.cpu_samples) if self.cpu_samples else 0.0,
            "ram_samples": self.ram_samples,
            "ram_avg": statistics.mean(self.ram_samples) if self.ram_samples else 0.0,
            "ram_max": max(self.ram_samples) if self.ram_samples else 0.0,
        }

class LoadTester:
    """Real load tester using httpx"""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout
        self.latencies = []
        self.errors = []
        self.successful = 0
        self.failed = 0

    async def make_request(self, endpoint: str, client: httpx.AsyncClient) -> Dict[str, Any]:
        """Make single HTTP request and measure latency"""
        start = time.perf_counter()
        try:
            response = await client.get(f"{self.base_url}{endpoint}")
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code < 400:
                self.successful += 1
                self.latencies.append(latency_ms)
                return {"success": True, "latency_ms": latency_ms, "status": response.status_code}
            else:
                self.failed += 1
                error = {
                    "endpoint": endpoint,
                    "status": response.status_code,
                    "timestamp": datetime.now().isoformat()
                }
                self.errors.append(error)
                return {"success": False, "error": error}

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self.failed += 1
            error = {
                "endpoint": endpoint,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.errors.append(error)
            return {"success": False, "error": error}

    async def run_agent(self, agent_id: int, profile: LoadTestProfile, duration_sec: int, client: httpx.AsyncClient):
        """Simulate single agent making requests"""
        interval = 60.0 / profile.tasks_per_min  # Seconds between requests
        end_time = time.time() + duration_sec

        while time.time() < end_time:
            # Pick endpoint from profile (round-robin)
            endpoint = profile.endpoints[agent_id % len(profile.endpoints)]
            await self.make_request(endpoint, client)
            await asyncio.sleep(interval)

    async def run_profile(self, profile: LoadTestProfile) -> LoadTestMetrics:
        """Execute load test profile"""
        print(f"\nRunning Profile {profile.key}: {profile.name}")
        print(f"Description: {profile.description}")
        print(f"Config: {profile.agents} agents, {profile.tasks_per_min} tasks/min, {profile.duration_min} min")
        print("")

        # Reset metrics
        self.latencies = []
        self.errors = []
        self.successful = 0
        self.failed = 0

        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor_task = asyncio.create_task(monitor.start())

        started_at = datetime.now().isoformat()
        start_time = time.time()

        # Run test
        duration_sec = profile.duration_min * 60

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Create agent tasks
            agent_tasks = [
                self.run_agent(i, profile, duration_sec, client)
                for i in range(profile.agents)
            ]

            # Wait for all agents to complete
            await asyncio.gather(*agent_tasks)

        # Stop monitoring
        monitor.stop()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        completed_at = datetime.now().isoformat()
        elapsed = time.time() - start_time

        # Calculate metrics
        resource_metrics = monitor.get_metrics()

        total = self.successful + self.failed
        success_rate = self.successful / total if total > 0 else 0.0
        error_rate = self.failed / total if total > 0 else 0.0
        rps = total / elapsed if elapsed > 0 else 0.0

        # Calculate latency percentiles
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
            p90 = sorted_latencies[int(len(sorted_latencies) * 0.90)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            min_lat = min(sorted_latencies)
            max_lat = max(sorted_latencies)
            mean_lat = statistics.mean(sorted_latencies)
        else:
            p50 = p90 = p99 = min_lat = max_lat = mean_lat = 0.0

        # Determine pass/fail
        status = "PASS" if success_rate >= 0.95 and p99 < 2000 else "FAIL"

        metrics = LoadTestMetrics(
            profile=profile.key,
            name=profile.name,
            config=asdict(profile),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=elapsed,
            total_requests=total,
            successful_requests=self.successful,
            failed_requests=self.failed,
            latencies_ms=self.latencies[:100],  # Sample for file size
            p50_latency_ms=p50,
            p90_latency_ms=p90,
            p99_latency_ms=p99,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            mean_latency_ms=mean_lat,
            success_rate=success_rate,
            error_rate=error_rate,
            requests_per_second=rps,
            cpu_samples=resource_metrics["cpu_samples"][:100],  # Sample
            cpu_avg_percent=resource_metrics["cpu_avg"],
            cpu_max_percent=resource_metrics["cpu_max"],
            ram_samples_mb=resource_metrics["ram_samples"][:100],  # Sample
            ram_avg_mb=resource_metrics["ram_avg"],
            ram_max_mb=resource_metrics["ram_max"],
            errors=self.errors[:20],  # Sample
            status=status
        )

        # Print results
        print(f"Profile {profile.key} complete - {status}")
        print(f"   Total requests: {total} ({self.successful} success, {self.failed} failed)")
        print(f"   Success rate: {success_rate*100:.2f}%")
        print(f"   Latency: P50={p50:.1f}ms | P90={p90:.1f}ms | P99={p99:.1f}ms")
        print(f"   RPS: {rps:.1f}")
        print(f"   CPU: avg={resource_metrics['cpu_avg']:.1f}% max={resource_metrics['cpu_max']:.1f}%")
        print(f"   RAM: avg={resource_metrics['ram_avg']:.1f}MB max={resource_metrics['ram_max']:.1f}MB")
        print("")

        return metrics

    async def run_all_profiles(self, profile_keys: List[str]) -> List[LoadTestMetrics]:
        """Run multiple profiles"""
        results = []
        for key in profile_keys:
            if key not in PROFILES:
                print(f"Warning: Unknown profile {key}, skipping")
                continue

            profile = PROFILES[key]
            metrics = await self.run_profile(profile)
            results.append(metrics)

            # Export results
            output_dir = Path("release/artifacts/load_test_results")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{key}_results.json"

            with open(output_path, "w") as f:
                json.dump(asdict(metrics), f, indent=2)

            print(f"Results exported to: {output_path}")
            print("")

        return results

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ShivX Real Load Test Harness")
    parser.add_argument(
        "--profile",
        "-p",
        choices=list(PROFILES.keys()) + ["ALL"],
        default="P1",
        help="Load test profile to run"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of ShivX instance"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds"
    )

    args = parser.parse_args()

    print("ShivX Real Load Test Harness")
    print("=============================")
    print(f"Target: {args.base_url}")
    print(f"Timeout: {args.timeout}s")
    print("")

    # Check if server is reachable
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{args.base_url}/api/health/live")
            if response.status_code != 200:
                print(f"ERROR: Server not healthy (status {response.status_code})")
                sys.exit(1)
        print("✓ Server is reachable")
        print("")
    except Exception as e:
        print(f"ERROR: Cannot reach server: {e}")
        print("Make sure ShivX is running on", args.base_url)
        sys.exit(1)

    # Run tests
    tester = LoadTester(base_url=args.base_url, timeout=args.timeout)

    if args.profile == "ALL":
        profile_keys = ["P1", "P2", "P3", "P4", "P5"]
    else:
        profile_keys = [args.profile]

    results = await tester.run_all_profiles(profile_keys)

    # Print summary
    print("=" * 60)
    print("Load Test Summary")
    print("=" * 60)
    for r in results:
        status_symbol = "✓" if r.status == "PASS" else "✗"
        print(f"{status_symbol} {r.profile}: {r.name}")
        print(f"   Latency P99: {r.p99_latency_ms:.1f}ms")
        print(f"   Success: {r.success_rate*100:.1f}%")
        print(f"   Requests: {r.total_requests} ({r.requests_per_second:.1f} RPS)")
        print(f"   Status: {r.status}")
        print("")

    # Overall pass/fail
    all_pass = all(r.status == "PASS" for r in results)
    print("=" * 60)
    if all_pass:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_pass else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
