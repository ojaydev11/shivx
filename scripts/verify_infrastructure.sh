#!/bin/bash
# ============================================================================
# ShivX Infrastructure Verification Script
# ============================================================================
# Comprehensive verification of production-ready infrastructure
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Functions
check_passed() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_failed() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
}

section_header() {
    echo ""
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Main verification
main() {
    echo ""
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║   ShivX Infrastructure Verification Script                ║${NC}"
    echo -e "${BOLD}${BLUE}║   Production Readiness Assessment                         ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # 1. File Structure
    section_header "1. File Structure Verification"

    if [ -f "deploy/docker-compose.yml" ]; then check_passed "docker-compose.yml exists"; else check_failed "docker-compose.yml missing"; fi
    if [ -f "deploy/docker-compose.secrets.yml" ]; then check_passed "docker-compose.secrets.yml exists"; else check_failed "docker-compose.secrets.yml missing"; fi
    if [ -f "deploy/secrets.example.yml" ]; then check_passed "secrets.example.yml exists"; else check_failed "secrets.example.yml missing"; fi
    if [ -f "deploy/postgres/postgresql.conf" ]; then check_passed "PostgreSQL SSL config exists"; else check_failed "PostgreSQL SSL config missing"; fi
    if [ -f "deploy/postgres/pg_hba.conf" ]; then check_passed "PostgreSQL auth config exists"; else check_failed "PostgreSQL auth config missing"; fi
    if [ -f "deploy/nginx/nginx.conf" ]; then check_passed "Nginx config exists"; else check_failed "Nginx config missing"; fi
    if [ -f "deploy/alerting-rules.yml" ]; then check_passed "Alert rules exist"; else check_failed "Alert rules missing"; fi
    if [ -f "deploy/alertmanager.yml" ]; then check_passed "Alertmanager config exists"; else check_failed "Alertmanager config missing"; fi
    if [ -f "deploy/loki/loki-config.yml" ]; then check_passed "Loki config exists"; else check_failed "Loki config missing"; fi
    if [ -f "deploy/promtail/promtail-config.yml" ]; then check_passed "Promtail config exists"; else check_failed "Promtail config missing"; fi

    # 2. Scripts
    section_header "2. Scripts Verification"

    if [ -x "scripts/generate_secrets.sh" ]; then check_passed "generate_secrets.sh is executable"; else check_failed "generate_secrets.sh not executable"; fi
    if [ -x "scripts/setup_ssl.sh" ]; then check_passed "setup_ssl.sh is executable"; else check_failed "setup_ssl.sh not executable"; fi
    if [ -x "scripts/validate_env.py" ]; then check_passed "validate_env.py is executable"; else check_failed "validate_env.py not executable"; fi
    if [ -x "scripts/backup.sh" ]; then check_passed "backup.sh is executable"; else check_failed "backup.sh not executable"; fi
    if [ -x "scripts/restore.sh" ]; then check_passed "restore.sh is executable"; else check_failed "restore.sh not executable"; fi
    if [ -x "scripts/generate_grafana_dashboards.py" ]; then check_passed "generate_grafana_dashboards.py is executable"; else check_failed "generate_grafana_dashboards.py not executable"; fi

    # 3. Dashboards
    section_header "3. Grafana Dashboards Verification"

    if [ -f "deploy/grafana/dashboards/system-health.json" ]; then check_passed "System Health dashboard exists"; else check_failed "System Health dashboard missing"; fi
    if [ -f "deploy/grafana/dashboards/api-performance.json" ]; then check_passed "API Performance dashboard exists"; else check_failed "API Performance dashboard missing"; fi
    if [ -f "deploy/grafana/dashboards/trading-metrics.json" ]; then check_passed "Trading Metrics dashboard exists"; else check_failed "Trading Metrics dashboard missing"; fi
    if [ -f "deploy/grafana/dashboards/security-monitoring.json" ]; then check_passed "Security Monitoring dashboard exists"; else check_failed "Security Monitoring dashboard missing"; fi
    if [ -f "deploy/grafana/dashboards/database-performance.json" ]; then check_passed "Database Performance dashboard exists"; else check_failed "Database Performance dashboard missing"; fi
    if [ -f "deploy/grafana/dashboards/ml-model-performance.json" ]; then check_passed "ML Model Performance dashboard exists"; else check_failed "ML Model Performance dashboard missing"; fi

    # 4. Documentation
    section_header "4. Documentation Verification"

    if [ -f "docs/disaster-recovery-runbook.md" ]; then check_passed "Disaster Recovery Runbook exists"; else check_failed "DR Runbook missing"; fi
    if [ -f "docs/security-checklist.md" ]; then check_passed "Security Checklist exists"; else check_failed "Security Checklist missing"; fi
    if [ -f "docs/INFRASTRUCTURE_DEPLOYMENT_REPORT.md" ]; then check_passed "Deployment Report exists"; else check_failed "Deployment Report missing"; fi
    if [ -f ".env.production.example" ]; then check_passed "Production env template exists"; else check_failed "Production env template missing"; fi

    # 5. Docker Compose Validation
    section_header "5. Docker Compose Configuration"

    if docker-compose -f deploy/docker-compose.yml config >/dev/null 2>&1; then
        check_passed "docker-compose.yml syntax valid"
    else
        check_failed "docker-compose.yml syntax invalid"
    fi

    # Check for hardcoded passwords
    if grep -q "POSTGRES_PASSWORD.*shivx_password" deploy/docker-compose.yml 2>/dev/null; then
        check_failed "Hardcoded PostgreSQL password found"
    else
        check_passed "No hardcoded PostgreSQL password"
    fi

    if grep -q "GF_SECURITY_ADMIN_PASSWORD.*admin[^_]" deploy/docker-compose.yml 2>/dev/null; then
        check_failed "Hardcoded Grafana password found"
    else
        check_passed "No hardcoded Grafana password"
    fi

    # 6. Security Configuration
    section_header "6. Security Configuration"

    # Check PostgreSQL SSL config
    if grep -q "ssl = on" deploy/postgres/postgresql.conf; then
        check_passed "PostgreSQL SSL enabled"
    else
        check_failed "PostgreSQL SSL not enabled"
    fi

    # Check for SSL certificates config
    if grep -q "ssl_cert_file" deploy/postgres/postgresql.conf; then
        check_passed "PostgreSQL SSL certificates configured"
    else
        check_failed "PostgreSQL SSL certificates not configured"
    fi

    # Check nginx SSL config
    if grep -q "ssl_protocols TLSv1.2 TLSv1.3" deploy/nginx/nginx.conf; then
        check_passed "Nginx TLS 1.2/1.3 configured"
    else
        check_warning "Nginx TLS configuration may need review"
    fi

    # Check security headers
    if grep -q "Strict-Transport-Security" deploy/nginx/nginx.conf; then
        check_passed "HSTS header configured"
    else
        check_failed "HSTS header missing"
    fi

    # 7. Monitoring Configuration
    section_header "7. Monitoring & Alerting"

    # Count alert rules
    alert_count=$(grep -c "alert:" deploy/alerting-rules.yml 2>/dev/null || echo 0)
    if [ "$alert_count" -gt 20 ]; then
        check_passed "Alert rules defined ($alert_count rules)"
    else
        check_warning "Only $alert_count alert rules found (expected 28+)"
    fi

    # Check Alertmanager receivers
    if grep -q "receivers:" deploy/alertmanager.yml; then
        check_passed "Alertmanager receivers configured"
    else
        check_failed "Alertmanager receivers not configured"
    fi

    # 8. Backup Configuration
    section_header "8. Backup & Recovery"

    # Check backup script features
    if grep -q "ENCRYPTION" scripts/backup.sh; then
        check_passed "Backup encryption configured"
    else
        check_warning "Backup encryption not configured"
    fi

    if grep -q "S3_BUCKET" scripts/backup.sh; then
        check_passed "S3 backup support available"
    else
        check_warning "S3 backup not configured"
    fi

    # Check restore script features
    if grep -q "pitr" scripts/restore.sh; then
        check_passed "Point-in-time recovery supported"
    else
        check_warning "PITR not available"
    fi

    # 9. Health Checks
    section_header "9. Health Check Endpoints"

    if grep -q "check_database" app/services/readiness.py; then
        check_passed "Database health check implemented"
    else
        check_failed "Database health check missing"
    fi

    if grep -q "check_redis" app/services/readiness.py; then
        check_failed "Redis health check implemented"
    else
        check_failed "Redis health check missing"
    fi

    if grep -q "check_disk_space" app/services/readiness.py; then
        check_passed "Disk space health check implemented"
    else
        check_failed "Disk space health check missing"
    fi

    if grep -q "/metrics" app/routes/health.py; then
        check_passed "Prometheus metrics endpoint implemented"
    else
        check_failed "Prometheus metrics endpoint missing"
    fi

    # 10. Logging Configuration
    section_header "10. Centralized Logging"

    if [ -f "deploy/loki/loki-config.yml" ]; then
        check_passed "Loki configuration exists"
    else
        check_failed "Loki configuration missing"
    fi

    if [ -f "deploy/promtail/promtail-config.yml" ]; then
        check_passed "Promtail configuration exists"
    else
        check_failed "Promtail configuration missing"
    fi

    if grep -q "retention_enabled: true" deploy/loki/loki-config.yml; then
        check_passed "Log retention configured"
    else
        check_warning "Log retention not configured"
    fi

    # Summary
    section_header "Verification Summary"

    echo ""
    echo -e "${BOLD}Results:${NC}"
    echo -e "  ${GREEN}Passed:${NC}   $PASSED_CHECKS"
    echo -e "  ${YELLOW}Warnings:${NC} $WARNING_CHECKS"
    echo -e "  ${RED}Failed:${NC}   $FAILED_CHECKS"
    echo -e "  ${BLUE}Total:${NC}    $TOTAL_CHECKS"
    echo ""

    # Calculate score
    SCORE=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))

    echo -e "${BOLD}Score: $SCORE%${NC}"
    echo ""

    # Determine status
    if [ $FAILED_CHECKS -eq 0 ]; then
        if [ $WARNING_CHECKS -eq 0 ]; then
            echo -e "${GREEN}${BOLD}✓ PERFECT! Infrastructure is production-ready!${NC}"
            echo ""
            return 0
        else
            echo -e "${YELLOW}${BOLD}⚠ GOOD! Infrastructure is production-ready with minor warnings.${NC}"
            echo -e "${YELLOW}Review warnings before deployment.${NC}"
            echo ""
            return 0
        fi
    else
        echo -e "${RED}${BOLD}✗ ISSUES FOUND! Fix failures before production deployment.${NC}"
        echo ""
        return 1
    fi
}

# Run verification
main "$@"
