#!/usr/bin/env bash
# Create System Snapshot Script
# Creates a point-in-time snapshot of the ShivX system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-./snapshots}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_NAME="shivx_snapshot_${TIMESTAMP}"
SNAPSHOT_PATH="${SNAPSHOT_DIR}/${SNAPSHOT_NAME}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log "ShivX Snapshot Creation"
log "======================"
log "Snapshot: $SNAPSHOT_NAME"
log "Path: $SNAPSHOT_PATH"
log ""

# Create snapshot directory
mkdir -p "$SNAPSHOT_PATH"

# 1. Snapshot database
log "Snapshotting database..."
if [ -f "data/shivx.db" ]; then
    mkdir -p "$SNAPSHOT_PATH/database"
    sqlite3 data/shivx.db ".backup '$SNAPSHOT_PATH/database/shivx.db'"
    log "✓ Database snapshot created (SQLite)"
elif command -v pg_dump &> /dev/null; then
    warn "PostgreSQL detected but no credentials provided"
    warn "Skipping database snapshot (set DB_URL for PostgreSQL backup)"
else
    warn "No database found to snapshot"
fi

# 2. Snapshot configuration
log "Snapshotting configuration..."
mkdir -p "$SNAPSHOT_PATH/config"
cp .env "$SNAPSHOT_PATH/config/.env" 2>/dev/null || warn "No .env file found"
cp config/*.env "$SNAPSHOT_PATH/config/" 2>/dev/null || true
log "✓ Configuration snapshot created"

# 3. Snapshot ML models
log "Snapshotting ML models..."
if [ -d "models/checkpoints" ]; then
    mkdir -p "$SNAPSHOT_PATH/models"
    cp -r models/checkpoints "$SNAPSHOT_PATH/models/" 2>/dev/null || warn "No model checkpoints found"
    log "✓ ML models snapshot created"
else
    warn "No models directory found"
fi

# 4. Snapshot audit logs
log "Snapshotting audit logs..."
if [ -d "var/resilience" ]; then
    mkdir -p "$SNAPSHOT_PATH/audit"
    cp -r var/resilience "$SNAPSHOT_PATH/audit/" 2>/dev/null || true
    log "✓ Audit logs snapshot created"
fi

# 5. Snapshot application data
log "Snapshotting application data..."
if [ -d "data" ]; then
    mkdir -p "$SNAPSHOT_PATH/data"
    cp -r data/* "$SNAPSHOT_PATH/data/" 2>/dev/null || true
    log "✓ Application data snapshot created"
fi

# 6. Create metadata
log "Creating snapshot metadata..."
cat > "$SNAPSHOT_PATH/metadata.json" <<EOF
{
  "snapshot_name": "$SNAPSHOT_NAME",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "version": "$(grep 'version' pyproject.toml | head -1 | cut -d '"' -f 2 2>/dev/null || echo 'unknown')",
  "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "created_by": "$USER",
  "hostname": "$(hostname)",
  "components": {
    "database": $([ -f "$SNAPSHOT_PATH/database/shivx.db" ] && echo "true" || echo "false"),
    "config": $([ -f "$SNAPSHOT_PATH/config/.env" ] && echo "true" || echo "false"),
    "models": $([ -d "$SNAPSHOT_PATH/models" ] && echo "true" || echo "false"),
    "audit_logs": $([ -d "$SNAPSHOT_PATH/audit" ] && echo "true" || echo "false"),
    "app_data": $([ -d "$SNAPSHOT_PATH/data" ] && echo "true" || echo "false")
  }
}
EOF
log "✓ Metadata created"

# 7. Calculate checksums
log "Calculating checksums..."
(cd "$SNAPSHOT_PATH" && find . -type f -exec sha256sum {} \; > checksums.sha256)
log "✓ Checksums calculated"

# 8. Compress snapshot
log "Compressing snapshot..."
tar -czf "${SNAPSHOT_PATH}.tar.gz" -C "$SNAPSHOT_DIR" "$SNAPSHOT_NAME"
SNAPSHOT_SIZE=$(du -h "${SNAPSHOT_PATH}.tar.gz" | cut -f1)
log "✓ Snapshot compressed (${SNAPSHOT_SIZE})"

# Cleanup uncompressed snapshot
rm -rf "$SNAPSHOT_PATH"

log ""
log "Snapshot created successfully!"
log "  File: ${SNAPSHOT_PATH}.tar.gz"
log "  Size: $SNAPSHOT_SIZE"
log ""
log "To restore this snapshot, run:"
log "  ./scripts/snapshot/restore_snapshot.sh ${SNAPSHOT_NAME}.tar.gz"
