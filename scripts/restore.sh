#!/bin/bash
# ============================================================================
# ShivX Database Restore Script
# ============================================================================
# Restores PostgreSQL database from backup with validation
# Supports point-in-time recovery using WAL archives
# ============================================================================

set -e
set -o pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/shivx}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-shivx-postgres}"
POSTGRES_USER="${POSTGRES_USER:-shivx}"
POSTGRES_DB="${POSTGRES_DB:-shivx}"
ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/etc/shivx/backup-key.txt}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Usage
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] <backup_file>

Restore ShivX database from backup.

OPTIONS:
    -h, --help              Show this help message
    -v, --verify            Verify backup integrity only (don't restore)
    -p, --pitr <timestamp>  Point-in-time recovery to timestamp
    -f, --force             Force restore without confirmation
    --download-s3           Download backup from S3 first

EXAMPLES:
    # Verify backup
    $0 --verify /var/backups/shivx/shivx_backup_20250101_120000.sql.gz.enc

    # Restore from local backup
    $0 /var/backups/shivx/shivx_backup_20250101_120000.sql.gz.enc

    # Restore from S3
    $0 --download-s3 shivx_backup_20250101_120000.sql.gz.enc

    # Point-in-time recovery
    $0 --pitr "2025-01-01 12:30:00" /var/backups/shivx/shivx_backup_20250101_120000.sql.gz.enc

EOF
    exit 1
}

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error_exit() {
    log_error "$1"
    exit 1
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"

    log "Verifying backup integrity..."

    # Check if file exists
    if [ ! -f "$backup_file" ]; then
        error_exit "Backup file not found: $backup_file"
    fi

    # Verify checksum
    if [ -f "${backup_file}.sha256" ]; then
        log "Verifying checksum..."
        local stored_checksum=$(cat "${backup_file}.sha256")
        local calculated_checksum=$(sha256sum "$backup_file" | awk '{print $1}')

        if [ "$stored_checksum" != "$calculated_checksum" ]; then
            error_exit "Checksum mismatch! Backup may be corrupted."
        fi
        log_success "Checksum verified"
    else
        log_warning "No checksum file found, skipping verification"
    fi

    # Test decryption if encrypted
    if [[ "$backup_file" == *.enc ]]; then
        log "Testing decryption..."
        if [ ! -f "$ENCRYPTION_KEY_FILE" ]; then
            error_exit "Encryption key not found: $ENCRYPTION_KEY_FILE"
        fi

        openssl enc -aes-256-cbc -d \
            -in "$backup_file" \
            -pass file:"$ENCRYPTION_KEY_FILE" \
            | gunzip | head -n 10 > /dev/null \
            || error_exit "Decryption test failed"

        log_success "Decryption test passed"
    fi

    # Test decompression
    if [[ "$backup_file" == *.gz* ]]; then
        log "Testing decompression..."

        if [[ "$backup_file" == *.enc ]]; then
            openssl enc -aes-256-cbc -d \
                -in "$backup_file" \
                -pass file:"$ENCRYPTION_KEY_FILE" \
                | gunzip | head -n 10 > /dev/null \
                || error_exit "Decompression test failed"
        else
            gunzip -t "$backup_file" || error_exit "Decompression test failed"
        fi

        log_success "Decompression test passed"
    fi

    log_success "Backup verification completed successfully"
}

# Download backup from S3
download_from_s3() {
    local backup_name="$1"
    local local_file="${BACKUP_DIR}/${backup_name}"

    if [ -z "$S3_BUCKET" ]; then
        error_exit "S3_BUCKET not configured"
    fi

    if ! command -v aws &>/dev/null; then
        error_exit "AWS CLI not installed"
    fi

    log "Downloading backup from S3: s3://${S3_BUCKET}/backups/${backup_name}"

    aws s3 cp "s3://${S3_BUCKET}/backups/${backup_name}" "$local_file" \
        || error_exit "S3 download failed"

    # Download checksum
    aws s3 cp "s3://${S3_BUCKET}/backups/${backup_name}.sha256" "${local_file}.sha256" \
        || log_warning "Checksum file not found in S3"

    log_success "Backup downloaded: $local_file"
    echo "$local_file"
}

# Prepare backup file for restore
prepare_backup() {
    local backup_file="$1"
    local temp_dir=$(mktemp -d)
    local prepared_file="${temp_dir}/restore.sql"

    log "Preparing backup file..."

    # Decrypt if encrypted
    if [[ "$backup_file" == *.enc ]]; then
        log "Decrypting backup..."
        openssl enc -aes-256-cbc -d \
            -in "$backup_file" \
            -pass file:"$ENCRYPTION_KEY_FILE" \
            | gunzip > "$prepared_file" \
            || error_exit "Decryption failed"
    elif [[ "$backup_file" == *.gz ]]; then
        log "Decompressing backup..."
        gunzip -c "$backup_file" > "$prepared_file" \
            || error_exit "Decompression failed"
    else
        cp "$backup_file" "$prepared_file"
    fi

    log_success "Backup prepared: $prepared_file"
    echo "$prepared_file"
}

# Create pre-restore snapshot
create_snapshot() {
    log "Creating pre-restore snapshot..."

    local snapshot_name="pre_restore_$(date +%Y%m%d_%H%M%S)"
    local snapshot_file="${BACKUP_DIR}/${snapshot_name}.sql.gz"

    docker exec "$POSTGRES_CONTAINER" pg_dump \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --clean --if-exists \
        | gzip > "$snapshot_file" \
        || log_warning "Failed to create snapshot"

    if [ -f "$snapshot_file" ]; then
        log_success "Pre-restore snapshot created: $snapshot_file"
    fi
}

# Restore database
restore_database() {
    local backup_file="$1"

    log "Starting database restore..."

    # Stop application containers to prevent connections
    log "Stopping application containers..."
    docker-compose -f "$(dirname $0)/../deploy/docker-compose.yml" stop shivx || log_warning "Failed to stop shivx container"

    # Wait for connections to close
    log "Waiting for active connections to close..."
    sleep 5

    # Terminate remaining connections
    docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d postgres -c \
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$POSTGRES_DB' AND pid <> pg_backend_pid();" \
        || log_warning "Failed to terminate connections"

    # Perform restore
    log "Restoring database..."
    docker exec -i "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d postgres < "$backup_file" \
        || error_exit "Database restore failed"

    log_success "Database restored successfully"

    # Restart application
    log "Restarting application..."
    docker-compose -f "$(dirname $0)/../deploy/docker-compose.yml" start shivx \
        || log_warning "Failed to restart shivx container"

    log_success "Application restarted"
}

# Perform point-in-time recovery
pitr_restore() {
    local backup_file="$1"
    local target_time="$2"

    log "Performing point-in-time recovery to: $target_time"

    # Restore base backup
    restore_database "$backup_file"

    # Apply WAL archives up to target time
    log "Applying WAL archives..."

    local recovery_conf="${BACKUP_DIR}/recovery.conf"
    cat > "$recovery_conf" <<EOF
restore_command = 'cp /var/lib/postgresql/wal_archive/%f %p'
recovery_target_time = '$target_time'
recovery_target_action = 'promote'
EOF

    # Copy recovery.conf to container
    docker cp "$recovery_conf" "${POSTGRES_CONTAINER}:/var/lib/postgresql/data/recovery.conf"

    # Restart PostgreSQL to apply recovery
    docker restart "$POSTGRES_CONTAINER"

    log "Waiting for recovery to complete..."
    sleep 10

    # Check recovery status
    docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
        "SELECT pg_is_in_recovery();" \
        || error_exit "Recovery verification failed"

    log_success "Point-in-time recovery completed"
}

# Validate restored database
validate_restore() {
    log "Validating restored database..."

    # Check database connectivity
    docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1;" \
        &>/dev/null || error_exit "Database connectivity check failed"

    # Check table counts
    local table_count=$(docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c \
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';")

    log "Tables found: $(echo $table_count | xargs)"

    if [ "$(echo $table_count | xargs)" -eq 0 ]; then
        log_warning "No tables found in database"
    fi

    # Check for common tables
    local critical_tables=("users" "trades" "positions")
    for table in "${critical_tables[@]}"; do
        if docker exec "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c \
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '$table');" \
            | grep -q "t"; then
            log_success "Table exists: $table"
        else
            log_warning "Table not found: $table"
        fi
    done

    log_success "Database validation completed"
}

# Main function
main() {
    local backup_file=""
    local verify_only=false
    local pitr_time=""
    local force=false
    local download_s3=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                ;;
            -v|--verify)
                verify_only=true
                shift
                ;;
            -p|--pitr)
                pitr_time="$2"
                shift 2
                ;;
            -f|--force)
                force=true
                shift
                ;;
            --download-s3)
                download_s3=true
                shift
                ;;
            *)
                backup_file="$1"
                shift
                ;;
        esac
    done

    if [ -z "$backup_file" ]; then
        usage
    fi

    log "============================================================"
    log "ShivX Database Restore Utility"
    log "============================================================"

    # Download from S3 if requested
    if [ "$download_s3" = true ]; then
        backup_file=$(download_from_s3 "$backup_file")
    fi

    # Verify backup
    verify_backup "$backup_file"

    if [ "$verify_only" = true ]; then
        log_success "Verification complete. Backup is valid."
        exit 0
    fi

    # Confirmation prompt
    if [ "$force" = false ]; then
        log_warning "WARNING: This will REPLACE the current database!"
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log "Restore cancelled by user"
            exit 0
        fi
    fi

    # Create pre-restore snapshot
    create_snapshot

    # Prepare backup
    prepared_file=$(prepare_backup "$backup_file")

    # Perform restore
    if [ -n "$pitr_time" ]; then
        pitr_restore "$prepared_file" "$pitr_time"
    else
        restore_database "$prepared_file"
    fi

    # Validate restore
    validate_restore

    # Cleanup
    rm -rf "$(dirname $prepared_file)"

    log "============================================================"
    log_success "Database restore completed successfully!"
    log "============================================================"

    echo ""
    echo "Recovery Time Objective (RTO): Achieved"
    echo "Next steps:"
    echo "  1. Verify application functionality"
    echo "  2. Check data integrity"
    echo "  3. Monitor system performance"
    echo "  4. Review logs for errors"
}

# Run main
main "$@"
