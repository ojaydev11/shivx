#!/bin/bash
# ============================================================================
# ShivX SSL/TLS Certificate Setup Script
# ============================================================================
# Sets up SSL certificates for nginx reverse proxy
# Supports both self-signed (dev) and Let's Encrypt (production)
# ============================================================================

set -e

# Configuration
SSL_DIR="${1:-./deploy/nginx/ssl}"
DOMAIN="${2:-shivx.local}"
EMAIL="${3:-admin@shivx.local}"
MODE="${4:-selfsigned}"  # selfsigned or letsencrypt

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Function to generate self-signed certificates
generate_selfsigned() {
    log_info "Generating self-signed SSL certificates..."

    # Create SSL directory
    mkdir -p "$SSL_DIR"
    chmod 700 "$SSL_DIR"

    # Generate private key
    openssl genrsa -out "$SSL_DIR/privkey.pem" 4096

    # Generate certificate signing request
    openssl req -new -key "$SSL_DIR/privkey.pem" \
        -out "$SSL_DIR/cert.csr" \
        -subj "/C=US/ST=State/L=City/O=ShivX/OU=IT/CN=$DOMAIN/emailAddress=$EMAIL"

    # Create config for SAN (Subject Alternative Names)
    cat > "$SSL_DIR/openssl.cnf" <<EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = ShivX
OU = IT Department
CN = $DOMAIN
emailAddress = $EMAIL

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = *.$DOMAIN
DNS.3 = localhost
DNS.4 = api.$DOMAIN
DNS.5 = grafana.$DOMAIN
DNS.6 = prometheus.$DOMAIN
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

    # Generate self-signed certificate (valid for 365 days)
    openssl x509 -req -days 365 \
        -in "$SSL_DIR/cert.csr" \
        -signkey "$SSL_DIR/privkey.pem" \
        -out "$SSL_DIR/fullchain.pem" \
        -extfile "$SSL_DIR/openssl.cnf" \
        -extensions v3_req

    # Copy fullchain as chain (for consistency with Let's Encrypt structure)
    cp "$SSL_DIR/fullchain.pem" "$SSL_DIR/chain.pem"

    # Set permissions
    chmod 600 "$SSL_DIR/privkey.pem"
    chmod 644 "$SSL_DIR/fullchain.pem"
    chmod 644 "$SSL_DIR/chain.pem"

    # Clean up
    rm -f "$SSL_DIR/cert.csr" "$SSL_DIR/openssl.cnf"

    log_success "Self-signed certificate generated successfully!"
    log_warning "These certificates are for DEVELOPMENT ONLY!"
    log_warning "For production, use Let's Encrypt or certificates from a trusted CA."
}

# Function to setup Let's Encrypt certificates
setup_letsencrypt() {
    log_info "Setting up Let's Encrypt certificates..."

    # Check if running as root (required for certbot)
    if [ "$EUID" -ne 0 ]; then
        log_error "Let's Encrypt setup requires root privileges"
        log_info "Run with: sudo $0 $SSL_DIR $DOMAIN $EMAIL letsencrypt"
        exit 1
    fi

    # Check if certbot is installed
    if ! command -v certbot &> /dev/null; then
        log_error "certbot not found. Installing..."

        # Detect OS and install certbot
        if [ -f /etc/debian_version ]; then
            apt-get update
            apt-get install -y certbot
        elif [ -f /etc/redhat-release ]; then
            yum install -y certbot
        else
            log_error "Unsupported OS. Please install certbot manually."
            exit 1
        fi
    fi

    # Create webroot directory for ACME challenge
    mkdir -p /var/www/certbot

    # Obtain certificate
    log_info "Obtaining certificate for $DOMAIN (this may take a few minutes)..."

    certbot certonly --webroot \
        -w /var/www/certbot \
        -d "$DOMAIN" \
        -d "api.$DOMAIN" \
        -d "grafana.$DOMAIN" \
        -d "prometheus.$DOMAIN" \
        --email "$EMAIL" \
        --agree-tos \
        --non-interactive \
        --staple-ocsp

    # Create symbolic links to Let's Encrypt certificates
    mkdir -p "$SSL_DIR"
    ln -sf "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$SSL_DIR/fullchain.pem"
    ln -sf "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$SSL_DIR/privkey.pem"
    ln -sf "/etc/letsencrypt/live/$DOMAIN/chain.pem" "$SSL_DIR/chain.pem"

    log_success "Let's Encrypt certificate obtained successfully!"

    # Setup auto-renewal cron job
    log_info "Setting up automatic certificate renewal..."

    # Create renewal script
    cat > /etc/cron.daily/certbot-renew <<'EOF'
#!/bin/bash
certbot renew --quiet --post-hook "docker-compose -f /path/to/docker-compose.yml restart nginx"
EOF

    chmod +x /etc/cron.daily/certbot-renew

    log_success "Auto-renewal configured (runs daily)"
}

# Function to verify certificates
verify_certificates() {
    log_info "Verifying SSL certificates..."

    if [ ! -f "$SSL_DIR/privkey.pem" ] || [ ! -f "$SSL_DIR/fullchain.pem" ]; then
        log_error "Certificate files not found in $SSL_DIR"
        return 1
    fi

    # Check private key
    if ! openssl rsa -in "$SSL_DIR/privkey.pem" -check -noout 2>/dev/null; then
        log_error "Invalid private key"
        return 1
    fi

    # Check certificate
    if ! openssl x509 -in "$SSL_DIR/fullchain.pem" -text -noout &>/dev/null; then
        log_error "Invalid certificate"
        return 1
    fi

    # Display certificate info
    log_info "Certificate details:"
    openssl x509 -in "$SSL_DIR/fullchain.pem" -noout -subject -issuer -dates

    # Check if certificate matches private key
    cert_modulus=$(openssl x509 -noout -modulus -in "$SSL_DIR/fullchain.pem" | openssl md5)
    key_modulus=$(openssl rsa -noout -modulus -in "$SSL_DIR/privkey.pem" | openssl md5)

    if [ "$cert_modulus" = "$key_modulus" ]; then
        log_success "Certificate and private key match!"
    else
        log_error "Certificate and private key do NOT match!"
        return 1
    fi

    log_success "SSL certificates are valid!"
}

# Function to test SSL configuration
test_ssl() {
    log_info "Testing SSL configuration..."

    if ! command -v openssl &> /dev/null; then
        log_warning "OpenSSL not found, skipping SSL test"
        return
    fi

    # Test SSL handshake (if nginx is running)
    if curl -k -s https://localhost/ &> /dev/null; then
        log_info "Testing SSL connection..."
        echo | openssl s_client -connect localhost:443 -servername $DOMAIN 2>/dev/null | \
            openssl x509 -noout -text | grep -E "(Subject:|Issuer:|Not After)"
        log_success "SSL connection test passed!"
    else
        log_warning "nginx not running, skipping connection test"
    fi
}

# Main script
main() {
    log_info "ShivX SSL/TLS Certificate Setup"
    echo "=================================================="
    echo "  Mode: $MODE"
    echo "  Domain: $DOMAIN"
    echo "  Email: $EMAIL"
    echo "  SSL Directory: $SSL_DIR"
    echo "=================================================="
    echo ""

    case "$MODE" in
        selfsigned)
            generate_selfsigned
            verify_certificates
            ;;
        letsencrypt)
            setup_letsencrypt
            verify_certificates
            ;;
        verify)
            verify_certificates
            ;;
        test)
            test_ssl
            ;;
        *)
            log_error "Invalid mode: $MODE"
            echo "Usage: $0 [ssl_dir] [domain] [email] [mode]"
            echo "Modes:"
            echo "  selfsigned   - Generate self-signed certificates (default)"
            echo "  letsencrypt  - Obtain Let's Encrypt certificates"
            echo "  verify       - Verify existing certificates"
            echo "  test         - Test SSL configuration"
            exit 1
            ;;
    esac

    echo ""
    log_info "Next steps:"
    echo "  1. Update nginx configuration with your domain"
    echo "  2. Start nginx: docker-compose up -d nginx"
    echo "  3. Test SSL: curl -k https://$DOMAIN/api/health/live"
    echo "  4. For Let's Encrypt, ensure DNS points to your server"
    echo "  5. Run SSL Labs test: https://www.ssllabs.com/ssltest/"
    echo ""
}

# Run main
main "$@"
