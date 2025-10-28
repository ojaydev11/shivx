#!/bin/bash
# ============================================================================
# ShivX Secrets Generation Script
# ============================================================================
# Generates cryptographically secure secrets for production deployment
# ============================================================================

set -e

SECRETS_DIR="${1:-./deploy/secrets}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Generate a random secret
generate_secret() {
    local length="${1:-32}"
    python3 -c "import secrets; print(secrets.token_urlsafe($length))"
}

# Generate a random password (alphanumeric + special chars)
generate_password() {
    local length="${1:-24}"
    python3 -c "import secrets, string; chars = string.ascii_letters + string.digits + '!@#$%^&*()_+-='; print(''.join(secrets.choice(chars) for _ in range($length)))"
}

# Main script
main() {
    log_info "ShivX Secrets Generation Script"
    echo "=================================================="
    echo ""

    # Check if secrets directory already exists
    if [ -d "$SECRETS_DIR" ]; then
        log_warning "Secrets directory already exists: $SECRETS_DIR"
        read -p "Do you want to regenerate secrets? This will OVERWRITE existing secrets! (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_error "Aborted by user"
            exit 1
        fi
        log_warning "Backing up existing secrets..."
        mv "$SECRETS_DIR" "${SECRETS_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    # Create secrets directory
    log_info "Creating secrets directory: $SECRETS_DIR"
    mkdir -p "$SECRETS_DIR/postgres"
    chmod 700 "$SECRETS_DIR"

    # Generate application secrets
    log_info "Generating application secrets..."

    generate_secret 32 > "$SECRETS_DIR/shivx_secret_key.txt"
    log_success "Generated: shivx_secret_key.txt"

    generate_secret 32 > "$SECRETS_DIR/jwt_secret.txt"
    log_success "Generated: jwt_secret.txt"

    generate_secret 32 > "$SECRETS_DIR/api_key.txt"
    log_success "Generated: api_key.txt"

    # Generate database password
    log_info "Generating database secrets..."

    generate_password 24 > "$SECRETS_DIR/db_password.txt"
    log_success "Generated: db_password.txt"

    # Generate Grafana password
    log_info "Generating monitoring secrets..."

    generate_password 16 > "$SECRETS_DIR/grafana_admin_password.txt"
    log_success "Generated: grafana_admin_password.txt"

    # Set permissions
    log_info "Setting secure permissions..."
    chmod 600 "$SECRETS_DIR"/*.txt
    chmod 700 "$SECRETS_DIR/postgres"

    # Generate PostgreSQL SSL certificates (self-signed for dev)
    log_info "Generating PostgreSQL SSL certificates (self-signed)..."

    # Generate CA certificate
    openssl req -new -x509 -days 3650 -nodes \
        -out "$SECRETS_DIR/postgres/ca.crt" \
        -keyout "$SECRETS_DIR/postgres/ca.key" \
        -subj "/CN=ShivX-PostgreSQL-CA" \
        2>/dev/null

    # Generate server certificate
    openssl req -new -nodes \
        -out "$SECRETS_DIR/postgres/server.csr" \
        -keyout "$SECRETS_DIR/postgres/server.key" \
        -subj "/CN=postgres" \
        2>/dev/null

    # Sign server certificate with CA
    openssl x509 -req -days 3650 \
        -in "$SECRETS_DIR/postgres/server.csr" \
        -CA "$SECRETS_DIR/postgres/ca.crt" \
        -CAkey "$SECRETS_DIR/postgres/ca.key" \
        -CAcreateserial \
        -out "$SECRETS_DIR/postgres/server.crt" \
        2>/dev/null

    # Set certificate permissions
    chmod 600 "$SECRETS_DIR/postgres/server.key"
    chmod 644 "$SECRETS_DIR/postgres/server.crt"
    chmod 644 "$SECRETS_DIR/postgres/ca.crt"

    # Clean up CSR
    rm -f "$SECRETS_DIR/postgres/server.csr"

    log_success "Generated: PostgreSQL SSL certificates"

    # Create secrets summary file
    cat > "$SECRETS_DIR/README.txt" <<EOF
ShivX Production Secrets
========================

Generated: $(date)

CRITICAL SECURITY WARNINGS:
---------------------------
1. NEVER commit these files to version control
2. NEVER share these secrets via email or chat
3. NEVER log secret values
4. Store backups in a secure secret manager (AWS Secrets Manager, HashiCorp Vault, etc.)
5. Rotate secrets regularly (every 90 days recommended)
6. Use different secrets for each environment (dev, staging, production)

Files Generated:
----------------
- shivx_secret_key.txt: Application encryption key
- jwt_secret.txt: JWT token signing key
- api_key.txt: External API authentication key
- db_password.txt: PostgreSQL database password
- grafana_admin_password.txt: Grafana admin password
- postgres/ca.crt: PostgreSQL CA certificate
- postgres/server.crt: PostgreSQL server certificate
- postgres/server.key: PostgreSQL server private key

PostgreSQL SSL Certificates:
-----------------------------
The certificates generated are SELF-SIGNED and suitable for development/testing.
For production, obtain certificates from a trusted CA or use your organization's PKI.

Certificate validity: 10 years from generation date
Certificate subject: CN=postgres

Next Steps:
-----------
1. Review all generated secrets
2. Back up secrets to secure storage
3. Deploy using: docker-compose -f docker-compose.yml -f docker-compose.secrets.yml up -d
4. Test all services with new secrets
5. Document secret rotation procedures
6. Set up monitoring for secret expiration

EOF

    # Display summary
    echo ""
    echo "=================================================="
    log_success "All secrets generated successfully!"
    echo "=================================================="
    echo ""
    log_info "Secrets location: $SECRETS_DIR"
    log_info "Total secrets: $(find $SECRETS_DIR -name '*.txt' | wc -l) files"
    echo ""

    if [ "$VERBOSE" = "true" ]; then
        log_warning "Displaying secret lengths (NOT values):"
        echo "  - shivx_secret_key: $(wc -c < $SECRETS_DIR/shivx_secret_key.txt) bytes"
        echo "  - jwt_secret: $(wc -c < $SECRETS_DIR/jwt_secret.txt) bytes"
        echo "  - api_key: $(wc -c < $SECRETS_DIR/api_key.txt) bytes"
        echo "  - db_password: $(wc -c < $SECRETS_DIR/db_password.txt) bytes"
        echo "  - grafana_admin_password: $(wc -c < $SECRETS_DIR/grafana_admin_password.txt) bytes"
        echo ""
    fi

    log_warning "SECURITY REMINDER:"
    echo "  1. These secrets are stored in plaintext files"
    echo "  2. For production, use Docker Swarm secrets or external secret manager"
    echo "  3. Add $SECRETS_DIR to .gitignore"
    echo "  4. Set up secret rotation schedule"
    echo "  5. Restrict access to secrets directory"
    echo ""

    log_info "To deploy with secrets:"
    echo "  docker-compose -f docker-compose.yml -f docker-compose.secrets.yml up -d"
    echo ""

    log_info "Read full documentation:"
    echo "  cat $SECRETS_DIR/README.txt"
    echo ""
}

# Run main function
main "$@"
