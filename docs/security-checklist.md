# ShivX Production Security Hardening Checklist

## Overview

This comprehensive checklist ensures the ShivX trading platform meets security best practices before production deployment.

**CRITICAL:** All items marked with ⚠️ MUST be completed before production deployment.

**Last Updated:** 2025-01-28
**Version:** 1.0

---

## Table of Contents

1. [Secrets Management](#secrets-management)
2. [Authentication & Authorization](#authentication--authorization)
3. [Network Security](#network-security)
4. [Data Protection](#data-protection)
5. [Infrastructure Security](#infrastructure-security)
6. [Application Security](#application-security)
7. [Monitoring & Incident Response](#monitoring--incident-response)
8. [Compliance & Audit](#compliance--audit)
9. [OWASP Top 10 Verification](#owasp-top-10-verification)

---

## Secrets Management

### Encryption Keys

- [ ] ⚠️ All secrets are at least 32 bytes (256 bits)
- [ ] ⚠️ `SHIVX_SECRET_KEY` is randomly generated and unique
- [ ] ⚠️ `SHIVX_JWT_SECRET` is randomly generated and unique
- [ ] ⚠️ SECRET_KEY ≠ JWT_SECRET (different values)
- [ ] ⚠️ No placeholder values (REPLACE_WITH_, CHANGEME, etc.)
- [ ] ⚠️ No test/development secrets in production

**Verification:**
```bash
python3 scripts/validate_env.py --env-file .env.production --strict
```

### Secret Storage

- [ ] ⚠️ No secrets in version control (.env in .gitignore)
- [ ] ⚠️ No secrets in Docker images
- [ ] ⚠️ No secrets in logs
- [ ] ⚠️ Secrets stored in secure vault (AWS Secrets Manager, HashiCorp Vault)
- [ ] Docker secrets or environment variables used (not hardcoded)
- [ ] Backup encryption key stored separately from backups

**Verification:**
```bash
# Check for secrets in git history
git log -p | grep -E "(SECRET_KEY|JWT_SECRET|PASSWORD)" || echo "No secrets found"

# Check Docker images
docker history shivx-app | grep -E "(SECRET|PASSWORD)" || echo "No secrets in image"
```

### Secret Rotation

- [ ] Secret rotation policy documented (90 days recommended)
- [ ] Secret rotation tested in staging
- [ ] Database passwords rotated after initial setup
- [ ] API keys have expiration dates

---

## Authentication & Authorization

### Configuration

- [ ] ⚠️ `SKIP_AUTH` is set to `false`
- [ ] ⚠️ `DEBUG` mode is disabled
- [ ] ⚠️ `SHIVX_DEV` is set to `false`
- [ ] ⚠️ Strong password policy enforced (min 12 chars, complexity)
- [ ] Multi-factor authentication enabled for admin accounts
- [ ] Session timeout configured (default: 1 hour)

**Verification:**
```bash
grep -E "SKIP_AUTH|DEBUG|SHIVX_DEV" .env.production
# All should be false
```

### Rate Limiting

- [ ] ⚠️ Rate limiting enabled on all API endpoints
- [ ] Authentication endpoints have stricter limits (5 req/s)
- [ ] General API endpoints limited (10 req/s)
- [ ] Failed login attempts trigger progressive delays
- [ ] Account lockout after 5 failed attempts

**Verification:**
```bash
# Test rate limiting
for i in {1..20}; do curl -w "\n" https://api.shivx.io/api/auth/login; done
# Should see 429 Too Many Requests after limit
```

### Access Control

- [ ] ⚠️ Principle of least privilege applied
- [ ] Role-based access control (RBAC) implemented
- [ ] Admin access requires separate authentication
- [ ] Service accounts have minimal permissions
- [ ] No default credentials in use

---

## Network Security

### SSL/TLS Configuration

- [ ] ⚠️ SSL certificates valid and not expired
- [ ] ⚠️ SSL certificates from trusted CA (or valid self-signed for dev)
- [ ] ⚠️ HTTP → HTTPS redirect configured
- [ ] ⚠️ Minimum TLS version 1.2 (TLS 1.3 preferred)
- [ ] Strong cipher suites configured (no weak ciphers)
- [ ] HSTS header configured (max-age=31536000)
- [ ] Certificate auto-renewal configured (Let's Encrypt)

**Verification:**
```bash
# Test SSL configuration
./scripts/setup_ssl.sh verify

# SSL Labs test (for public servers)
# Visit: https://www.ssllabs.com/ssltest/

# Check TLS version
openssl s_client -connect api.shivx.io:443 -tls1_2
```

### Firewall & Network

- [ ] ⚠️ Firewall configured (UFW or iptables)
- [ ] ⚠️ Only necessary ports exposed (80, 443, 22)
- [ ] SSH port changed from default 22 (optional but recommended)
- [ ] SSH key-based authentication only (password auth disabled)
- [ ] fail2ban installed and configured
- [ ] DDoS protection enabled (Cloudflare or equivalent)

**Verification:**
```bash
# Check firewall status
sudo ufw status verbose

# Check open ports
sudo netstat -tulpn | grep LISTEN

# Check fail2ban
sudo fail2ban-client status
```

### CORS Configuration

- [ ] ⚠️ No wildcard (`*`) in CORS_ORIGINS
- [ ] ⚠️ Only specific domains whitelisted
- [ ] ⚠️ No `localhost` in production CORS_ORIGINS
- [ ] HTTPS-only origins
- [ ] Credentials properly handled

**Verification:**
```bash
grep CORS_ORIGINS .env.production
# Should NOT contain * or localhost
```

---

## Data Protection

### Database Security

- [ ] ⚠️ PostgreSQL SSL/TLS enabled (`sslmode=require`)
- [ ] ⚠️ Strong database password (not default)
- [ ] ⚠️ Database not exposed to public internet
- [ ] Database authentication uses SCRAM-SHA-256
- [ ] Connection pooling configured
- [ ] SQL injection protection (parameterized queries)
- [ ] Database backups encrypted

**Verification:**
```bash
# Check SSL mode
grep DATABASE_URL .env.production | grep -o "sslmode=[^&]*"
# Should be "sslmode=require" or "sslmode=verify-full"

# Test database SSL
docker exec shivx-postgres psql -U shivx -d shivx -c "SELECT ssl_is_used();"
# Should return true
```

### Encryption

- [ ] ⚠️ Data encrypted in transit (SSL/TLS everywhere)
- [ ] ⚠️ Sensitive data encrypted at rest
- [ ] Backup files encrypted
- [ ] API keys encrypted in database
- [ ] Wallet private keys encrypted (never plaintext)
- [ ] Strong encryption algorithms (AES-256, RSA-4096)

### Data Retention

- [ ] Data retention policy documented
- [ ] PII deletion procedures in place
- [ ] Audit logs retained for 1 year minimum
- [ ] Backup retention: 30 days

---

## Infrastructure Security

### Docker Security

- [ ] ⚠️ Docker daemon socket not exposed
- [ ] ⚠️ Containers run as non-root user
- [ ] ⚠️ No privileged containers
- [ ] Base images from trusted sources only
- [ ] Images scanned for vulnerabilities (Trivy, Snyk)
- [ ] Resource limits set (CPU, memory)
- [ ] Unnecessary capabilities dropped

**Verification:**
```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image shivx-app

# Check running as non-root
docker exec shivx-app whoami
# Should NOT be root
```

### System Hardening

- [ ] ⚠️ OS packages up to date
- [ ] ⚠️ Kernel security modules enabled (AppArmor/SELinux)
- [ ] Automatic security updates enabled
- [ ] Unnecessary services disabled
- [ ] Core dumps disabled
- [ ] Kernel parameters hardened (`sysctl.conf`)

**Verification:**
```bash
# Check updates
sudo apt update && sudo apt list --upgradable

# Check SELinux/AppArmor
sudo aa-status || sestatus
```

### Access Control

- [ ] ⚠️ SSH root login disabled
- [ ] ⚠️ SSH key-based auth only (no passwords)
- [ ] sudo requires password
- [ ] Separate user accounts (no shared logins)
- [ ] Privileged operations logged

---

## Application Security

### Input Validation

- [ ] ⚠️ All user inputs validated and sanitized
- [ ] ⚠️ SQL injection protection (parameterized queries)
- [ ] ⚠️ XSS protection enabled
- [ ] ⚠️ CSRF tokens on all state-changing operations
- [ ] File upload restrictions (size, type)
- [ ] JSON parsing limits enforced

### Security Headers

- [ ] ⚠️ `X-Frame-Options: DENY`
- [ ] ⚠️ `X-Content-Type-Options: nosniff`
- [ ] ⚠️ `X-XSS-Protection: 1; mode=block`
- [ ] ⚠️ `Strict-Transport-Security` (HSTS)
- [ ] ⚠️ `Content-Security-Policy` configured
- [ ] `Referrer-Policy` set

**Verification:**
```bash
# Check security headers
curl -I https://api.shivx.io | grep -E "X-Frame|X-Content|X-XSS|Strict-Transport|Content-Security"
```

### Dependency Security

- [ ] ⚠️ No known vulnerabilities in dependencies
- [ ] Dependency versions pinned
- [ ] Regular dependency updates scheduled
- [ ] Security advisories monitored

**Verification:**
```bash
# Python dependencies
safety check

# Or using Snyk
snyk test

# Check for outdated packages
pip list --outdated
```

### Error Handling

- [ ] ⚠️ No sensitive information in error messages
- [ ] ⚠️ Stack traces not exposed in production
- [ ] Errors logged securely (no sensitive data)
- [ ] Custom error pages configured

---

## Monitoring & Incident Response

### Monitoring

- [ ] ⚠️ Metrics collection enabled (`SHIVX_ENABLE_METRICS=true`)
- [ ] ⚠️ Centralized logging configured (Loki)
- [ ] ⚠️ All critical services monitored
- [ ] Health checks passing
- [ ] Grafana dashboards configured
- [ ] Prometheus scraping all targets

**Verification:**
```bash
# Check health endpoints
curl https://api.shivx.io/api/health/ready
curl https://api.shivx.io/api/health/live

# Check Prometheus targets
curl http://prometheus.shivx.io:9090/api/v1/targets
```

### Alerting

- [ ] ⚠️ Critical alerts go to PagerDuty
- [ ] ⚠️ Alert on high error rate (>1%)
- [ ] ⚠️ Alert on Guardian Defense lockdown
- [ ] ⚠️ Alert on failed authentication spike
- [ ] ⚠️ Alert on service down
- [ ] ⚠️ Alert on low disk space (<10%)
- [ ] Alert rules tested and firing correctly

**Verification:**
```bash
# Test alert
curl -X POST http://alertmanager:9093/api/v1/alerts -d '[{
  "labels": {"alertname":"test","severity":"warning"},
  "annotations": {"summary":"Test alert"}
}]'
```

### Incident Response

- [ ] Incident response plan documented
- [ ] Security incident contacts defined
- [ ] Guardian Defense configured and tested
- [ ] Lockdown procedures documented
- [ ] Disaster recovery runbook tested

### Logging

- [ ] ⚠️ Structured JSON logging enabled (`SHIVX_JSON_LOGGING=true`)
- [ ] Security events logged (auth failures, access changes)
- [ ] Logs include request IDs for tracing
- [ ] Log retention: 30 days minimum
- [ ] Logs sent to centralized system (Loki)
- [ ] No sensitive data in logs (passwords, tokens)

---

## Compliance & Audit

### Audit Logging

- [ ] ⚠️ Audit logging enabled (`SHIVX_AUDIT_ENABLED=true`)
- [ ] All privileged operations logged
- [ ] Failed access attempts logged
- [ ] Data access logged
- [ ] Configuration changes logged
- [ ] Audit logs tamper-proof
- [ ] Audit logs retained for 1 year

### Compliance

- [ ] Data protection regulations reviewed (GDPR, CCPA)
- [ ] Privacy policy updated
- [ ] Terms of service updated
- [ ] User consent mechanisms implemented
- [ ] Right to deletion implemented
- [ ] Data breach notification procedure documented

### Regular Audits

- [ ] Security audit scheduled (quarterly)
- [ ] Penetration testing scheduled (annually)
- [ ] Vulnerability scanning automated
- [ ] Dependency audits automated
- [ ] Access reviews quarterly

---

## OWASP Top 10 Verification

### A01:2021 – Broken Access Control

- [ ] ⚠️ Access control enforced on all endpoints
- [ ] ⚠️ No direct object references without authorization
- [ ] JWT tokens validated on every request
- [ ] Proper session management

### A02:2021 – Cryptographic Failures

- [ ] ⚠️ Strong encryption algorithms (AES-256, RSA-4096)
- [ ] ⚠️ TLS 1.2+ for all connections
- [ ] Passwords hashed with bcrypt/argon2
- [ ] No hardcoded keys or secrets

### A03:2021 – Injection

- [ ] ⚠️ SQL injection prevention (parameterized queries)
- [ ] ⚠️ Input validation on all user inputs
- [ ] ORM used properly
- [ ] Command injection prevention

### A04:2021 – Insecure Design

- [ ] Threat modeling completed
- [ ] Security requirements defined
- [ ] Secure architecture reviewed
- [ ] Rate limiting implemented

### A05:2021 – Security Misconfiguration

- [ ] ⚠️ Default credentials changed
- [ ] ⚠️ Unnecessary features disabled
- [ ] ⚠️ Error messages don't leak information
- [ ] Security headers configured

### A06:2021 – Vulnerable and Outdated Components

- [ ] ⚠️ Dependencies up to date
- [ ] ⚠️ No known CVEs
- [ ] Automated scanning in place
- [ ] Patch management process

### A07:2021 – Identification and Authentication Failures

- [ ] ⚠️ Strong password requirements
- [ ] ⚠️ MFA available for sensitive operations
- [ ] Session timeout configured
- [ ] Account lockout implemented

### A08:2021 – Software and Data Integrity Failures

- [ ] ⚠️ Code signing implemented
- [ ] ⚠️ Integrity checks on critical data
- [ ] CI/CD pipeline secured
- [ ] Supply chain security reviewed

### A09:2021 – Security Logging and Monitoring Failures

- [ ] ⚠️ Security events logged
- [ ] ⚠️ Real-time alerting configured
- [ ] ⚠️ Logs protected from tampering
- [ ] SIEM integration (optional)

### A10:2021 – Server-Side Request Forgery (SSRF)

- [ ] URL validation implemented
- [ ] Whitelist of allowed hosts
- [ ] Network segmentation
- [ ] No user-controlled URLs

---

## Automated Security Scanning

### Run Security Scans

```bash
# 1. Environment validation
python3 scripts/validate_env.py --env-file .env.production --strict

# 2. Dependency vulnerabilities
safety check
pip-audit

# 3. Container scanning
docker scan shivx-app

# 4. Trivy vulnerability scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image shivx-app

# 5. Python code security
bandit -r app/ -f json -o security-report.json

# 6. OWASP Dependency Check
dependency-check --project ShivX --scan .

# 7. SSL/TLS configuration
./scripts/setup_ssl.sh test
```

---

## Pre-Deployment Final Checks

### Critical Path

1. [ ] ⚠️ Run environment validation: `./scripts/validate_env.py --strict`
2. [ ] ⚠️ Verify all secrets are production-grade
3. [ ] ⚠️ Test SSL certificates: `./scripts/setup_ssl.sh verify`
4. [ ] ⚠️ Verify database SSL: Check `sslmode=require`
5. [ ] ⚠️ Test backup and restore: `./scripts/backup.sh && ./scripts/restore.sh --verify`
6. [ ] ⚠️ Verify health checks: `curl https://api.shivx.io/api/health/ready`
7. [ ] ⚠️ Test monitoring: Check Grafana dashboards
8. [ ] ⚠️ Test alerting: Fire test alert
9. [ ] ⚠️ Review security scan results
10. [ ] ⚠️ Sign off on security checklist

### Trading-Specific

- [ ] ⚠️ `SHIVX_TRADING_MODE` set to `paper` initially
- [ ] ⚠️ Guardian Defense enabled (`SHIVX_FEATURE_GUARDRAILS=true`)
- [ ] ⚠️ Circuit breakers configured
- [ ] Position limits configured
- [ ] Loss limits configured
- [ ] Test with small amounts first

---

## Sign-Off

**Security Review Completed By:**

| Name | Role | Signature | Date |
|------|------|-----------|------|
| | Infrastructure Lead | | |
| | Security Engineer | | |
| | Engineering Manager | | |
| | CTO | | |

**Approval for Production Deployment:**

- [ ] All ⚠️ CRITICAL items completed
- [ ] Security scan results reviewed
- [ ] Penetration test passed
- [ ] Disaster recovery tested
- [ ] Monitoring operational
- [ ] On-call team briefed

**Deployed By:** ___________________
**Date:** ___________________
**Deployment ID:** ___________________

---

## Post-Deployment Validation

Within 24 hours of deployment:

- [ ] Monitor error rates
- [ ] Monitor latency
- [ ] Verify backups running
- [ ] Verify alerts firing correctly
- [ ] Review security logs
- [ ] Test trading functionality (paper mode)

Within 7 days of deployment:

- [ ] Full security audit
- [ ] Performance review
- [ ] User acceptance testing
- [ ] Switch to live trading (if applicable)

---

## Appendix: Security Contact Information

### Internal Team

- **Security Team:** security@shivx.io
- **On-Call:** +1-XXX-XXX-XXXX
- **PagerDuty:** https://yourorg.pagerduty.com

### External Resources

- **OWASP:** https://owasp.org/
- **NIST Cybersecurity Framework:** https://www.nist.gov/cyberframework
- **CIS Controls:** https://www.cisecurity.org/controls

### Vulnerability Disclosure

- **Email:** security@shivx.io
- **PGP Key:** [Key ID]
- **Bug Bounty:** [Program URL]

---

**Document Version:** 1.0
**Last Updated:** 2025-01-28
**Next Review:** 2025-02-28
**Owner:** Security & Infrastructure Team
