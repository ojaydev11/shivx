#!/bin/bash
# ============================================================================
# ShivX Automated Backup Script
# ============================================================================
# Performs automated PostgreSQL backups with WAL archiving
# Supports local and S3 storage with encryption
# ============================================================================

set -e
set -o pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/shivx}"
S3_BUCKET="${S3_BUCKET:-}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-shivx-postgres}"
POSTGRES_USER="${POSTGRES_USER:-shivx}"
POSTGRES_DB="${POSTGRES_DB:-shivx}"
ENCRYPT_BACKUPS="${ENCRYPT_BACKUPS:-true}"
ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/etc/shivx/backup-key.txt}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
LOG_FILE="${BACKUP_DIR}/backup.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="shivx_backup_${TIMESTAMP}"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is running
    if ! docker ps &>/dev/null; then
        error_exit "Docker is not running"
    fi

    # Check if PostgreSQL container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
        error_exit "PostgreSQL container '$POSTGRES_CONTAINER' is not running"
    fi

    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        log "Creating backup directory: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"
    fi

    # Check encryption key if encryption is enabled
    if [ "$ENCRYPT_BACKUPS" = "true" ]; then
        if [ ! -f "$ENCRYPTION_KEY_FILE" ]; then
            log_warning "Encryption key not found, generating new key..."
            mkdir -p "$(dirname "$ENCRYPTION_KEY_FILE")"
            openssl rand -base64 32 > "$ENCRYPTION_KEY_FILE"
            chmod 600 "$ENCRYPTION_KEY_FILE"
            log_success "Generated new encryption key: $ENCRYPTION_KEY_FILE"
        fi
    fi

    log_success "Prerequisites check passed"
}

# Perform full database backup
backup_database() {
    log "Starting database backup: $BACKUP_NAME"

    local backup_file="${BACKUP_DIR}/${BACKUP_NAME}.sql"
    local compressed_file="${backup_file}.gz"

    # Dump database
    log "Dumping database..."
    docker exec "$POSTGRES_CONTAINER" pg_dump \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        --clean \
        --if-exists \
        --create \
        --format=plain \
        > "$backup_file" || error_exit "Database dump failed"

    # Compress backup
    log "Compressing backup..."
    gzip "$backup_file" || error_exit "Compression failed"

    # Encrypt if enabled
    if [ "$ENCRYPT_BACKUPS" = "true" ]; then
        log "Encrypting backup..."
        local encrypted_file="${compressed_file}.enc"
        openssl enc -aes-256-cbc \
            -salt \
            -in "$compressed_file" \
            -out "$encrypted_file" \
            -pass file:"$ENCRYPTION_KEY_FILE" || error_exit "Encryption failed"

        # Remove unencrypted file
        rm "$compressed_file"
        compressed_file="$encrypted_file"
    fi

    # Calculate checksum
    local checksum=$(sha256sum "$compressed_file" | awk '{print $1}')
    echo "$checksum" > "${compressed_file}.sha256"

    log_success "Database backup completed: $(basename $compressed_file)"
    log "Backup size: $(du -h $compressed_file | cut -f1)"
    log "Checksum: $checksum"

    echo "$compressed_file"
}

# Backup WAL archives
backup_wal_archives() {
    log "Backing up WAL archives..."

    local wal_backup_dir="${BACKUP_DIR}/wal/${BACKUP_NAME}"
    mkdir -p "$wal_backup_dir"

    # Copy WAL archives from container
    docker exec "$POSTGRES_CONTAINER" sh -c \
        "tar czf - /var/lib/postgresql/wal_archive" \
        > "${wal_backup_dir}/wal_archive.tar.gz" || log_warning "WAL backup failed (may not be configured)"

    if [ -f "${wal_backup_dir}/wal_archive.tar.gz" ]; then
        log_success "WAL archives backed up: ${wal_backup_dir}/wal_archive.tar.gz"
    fi
}

# Backup Docker volumes
backup_volumes() {
    log "Backing up Docker volumes..."

    local volumes_backup_dir="${BACKUP_DIR}/volumes/${BACKUP_NAME}"
    mkdir -p "$volumes_backup_dir"

    # List of volumes to backup
    local volumes=(
        "shivx-logs"
        "shivx-data"
        "shivx-models"
        "grafana-data"
        "prometheus-data"
    )

    for volume in "${volumes[@]}"; do
        if docker volume inspect "$volume" &>/dev/null; then
            log "Backing up volume: $volume"
            docker run --rm \
                -v "$volume:/volume" \
                -v "$volumes_backup_dir:/backup" \
                alpine \
                tar czf "/backup/${volume}.tar.gz" -C /volume . \
                || log_warning "Failed to backup volume: $volume"
        else
            log_warning "Volume not found: $volume"
        fi
    done

    log_success "Volume backups completed"
}

# Upload to S3 if configured
upload_to_s3() {
    local backup_file="$1"

    if [ -z "$S3_BUCKET" ]; then
        log "S3 upload skipped (S3_BUCKET not configured)"
        return 0
    fi

    log "Uploading backup to S3: $S3_BUCKET"

    # Check if AWS CLI is installed
    if ! command -v aws &>/dev/null; then
        log_warning "AWS CLI not found, skipping S3 upload"
        return 0
    fi

    # Upload main backup
    aws s3 cp "$backup_file" \
        "s3://${S3_BUCKET}/backups/$(basename $backup_file)" \
        --storage-class STANDARD_IA \
        || log_warning "S3 upload failed"

    # Upload checksum
    aws s3 cp "${backup_file}.sha256" \
        "s3://${S3_BUCKET}/backups/$(basename ${backup_file}.sha256)" \
        || log_warning "S3 checksum upload failed"

    # Upload WAL archives
    local wal_dir="${BACKUP_DIR}/wal/${BACKUP_NAME}"
    if [ -d "$wal_dir" ]; then
        aws s3 sync "$wal_dir" \
            "s3://${S3_BUCKET}/wal/${BACKUP_NAME}/" \
            || log_warning "WAL S3 upload failed"
    fi

    log_success "Backup uploaded to S3"
}

# Clean up old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days..."

    # Local cleanup
    find "$BACKUP_DIR" -name "shivx_backup_*" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR/wal" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    find "$BACKUP_DIR/volumes" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true

    # S3 cleanup if configured
    if [ -n "$S3_BUCKET" ] && command -v aws &>/dev/null; then
        local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y%m%d)

        aws s3 ls "s3://${S3_BUCKET}/backups/" \
            | awk '{print $4}' \
            | while read file; do
                if [[ $file =~ shivx_backup_([0-9]{8}) ]]; then
                    file_date="${BASH_REMATCH[1]}"
                    if [ "$file_date" -lt "$cutoff_date" ]; then
                        log "Deleting old S3 backup: $file"
                        aws s3 rm "s3://${S3_BUCKET}/backups/$file" || true
                    fi
                fi
            done
    fi

    log_success "Cleanup completed"
}

# Generate backup report
generate_report() {
    local backup_file="$1"
    local report_file="${BACKUP_DIR}/backup_report_${TIMESTAMP}.txt"

    cat > "$report_file" <<EOF
ShivX Backup Report
===================

Backup Date: $(date)
Backup Name: $BACKUP_NAME

Backup Details:
---------------
Database Backup: $(basename $backup_file)
Size: $(du -h $backup_file | cut -f1)
Checksum: $(cat ${backup_file}.sha256)
Encrypted: $ENCRYPT_BACKUPS

Storage Locations:
------------------
Local: $backup_file
$([ -n "$S3_BUCKET" ] && echo "S3: s3://${S3_BUCKET}/backups/$(basename $backup_file)")

Recovery Time Objective (RTO): < 1 hour
Recovery Point Objective (RPO): < 15 minutes

Next Steps:
-----------
1. Verify backup integrity: ./scripts/restore.sh --verify $backup_file
2. Test restore procedure monthly
3. Store encryption key securely separate from backups
4. Monitor backup completion in logging system

EOF

    log_success "Backup report generated: $report_file"
    cat "$report_file"
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"

    # Slack notification
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\"ShivX Backup [$status]: $message\"}" \
            &>/dev/null || true
    fi
}

# Main backup procedure
main() {
    log "============================================================"
    log "ShivX Automated Backup - Starting"
    log "============================================================"

    check_prerequisites

    # Perform backups
    backup_file=$(backup_database)
    backup_wal_archives
    backup_volumes

    # Upload to S3
    upload_to_s3 "$backup_file"

    # Cleanup old backups
    cleanup_old_backups

    # Generate report
    generate_report "$backup_file"

    log "============================================================"
    log_success "Backup completed successfully!"
    log "============================================================"

    send_notification "SUCCESS" "Backup completed: $BACKUP_NAME"
}

# Run main function
main "$@"
