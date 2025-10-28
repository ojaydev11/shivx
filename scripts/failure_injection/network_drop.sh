#!/usr/bin/env bash
# Network Drop Failure Injection Script
# Simulates network connectivity loss to test resilience

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/shivx_network_drop_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"
}

# Check if running as root (required for iptables)
if [[ $EUID -ne 0 ]]; then
   error "This script must be run as root (for iptables manipulation)"
   echo "  Usage: sudo $0"
   exit 1
fi

# Parse arguments
DURATION=${1:-30}  # Default 30 seconds
TARGET_HOST=${2:-api.mainnet-beta.solana.com}  # Default Solana RPC

log "Network Drop Failure Injection Test"
log "===================================="
log "Target Host: $TARGET_HOST"
log "Duration: ${DURATION}s"
log ""

# Backup existing iptables rules
log "Backing up iptables rules..."
iptables-save > /tmp/iptables_backup_$(date +%Y%m%d_%H%M%S).rules

# Check baseline connectivity
log "Checking baseline connectivity..."
if ping -c 3 -W 5 $TARGET_HOST &> /dev/null; then
    log "✓ Baseline connectivity OK to $TARGET_HOST"
else
    warn "Baseline connectivity check failed (may already be blocked)"
fi

# Inject network failure (block outbound traffic to target)
log "Injecting network failure..."
log "Blocking outbound traffic to $TARGET_HOST"

# Block DNS resolution
iptables -A OUTPUT -p udp --dport 53 -d 8.8.8.8 -j DROP
iptables -A OUTPUT -p tcp --dport 53 -d 8.8.8.8 -j DROP

# Block HTTP/HTTPS to target
iptables -A OUTPUT -p tcp --dport 443 -d $TARGET_HOST -j DROP
iptables -A OUTPUT -p tcp --dport 80 -d $TARGET_HOST -j DROP

log "✓ Network failure injected"

# Wait for duration
log "Waiting ${DURATION}s..."
log "During this time, the application should:"
log "  1. Detect network failure via circuit breaker"
log "  2. Trigger graceful degradation"
log "  3. Use cached data if available"
log "  4. Log errors without crashing"
log ""

sleep "$DURATION"

# Restore network
log "Restoring network connectivity..."
iptables -D OUTPUT -p udp --dport 53 -d 8.8.8.8 -j DROP
iptables -D OUTPUT -p tcp --dport 53 -d 8.8.8.8 -j DROP
iptables -D OUTPUT -p tcp --dport 443 -d $TARGET_HOST -j DROP
iptables -D OUTPUT -p tcp --dport 80 -d $TARGET_HOST -j DROP

log "✓ Network connectivity restored"

# Verify recovery
log "Verifying connectivity recovery..."
sleep 5
if ping -c 3 -W 5 $TARGET_HOST &> /dev/null; then
    log "✓ Connectivity recovered successfully"
else
    error "Connectivity not recovered (may need manual intervention)"
fi

# Check application health
log "Checking application health..."
if curl -s http://localhost:8000/api/health/ready > /dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8000/api/health/ready | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
    log "✓ Application health: $HEALTH"
else
    error "Application health check failed"
fi

log ""
log "Failure injection test complete"
log "Log file: $LOG_FILE"
log ""
log "Next steps:"
log "  1. Check application logs for circuit breaker activation"
log "  2. Verify graceful degradation occurred"
log "  3. Confirm no data loss"
log "  4. Review metrics for recovery time"
