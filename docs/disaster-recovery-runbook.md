# ShivX Disaster Recovery Runbook

## Overview

This runbook provides step-by-step procedures for recovering the ShivX trading platform from various disaster scenarios.

**Recovery Objectives:**
- **RTO (Recovery Time Objective):** < 1 hour
- **RPO (Recovery Point Objective):** < 15 minutes

---

## Table of Contents

1. [Backup Strategy](#backup-strategy)
2. [Recovery Scenarios](#recovery-scenarios)
3. [Step-by-Step Recovery Procedures](#step-by-step-recovery-procedures)
4. [Validation and Testing](#validation-and-testing)
5. [Contact Information](#contact-information)

---

## Backup Strategy

### Automated Backups

- **Frequency:** Daily at 2:00 AM UTC
- **Retention:** 30 days
- **Components Backed Up:**
  - PostgreSQL database (full dump + WAL archives)
  - Docker volumes (logs, data, models, Grafana, Prometheus)
  - Configuration files
  - SSL certificates

### Backup Locations

- **Primary:** Local server `/var/backups/shivx`
- **Secondary:** S3 bucket `s3://shivx-production-backups`
- **Tertiary:** Off-site encrypted backup (weekly)

### Backup Verification

Run weekly verification:
```bash
./scripts/restore.sh --verify /var/backups/shivx/latest_backup.sql.gz.enc
```

---

## Recovery Scenarios

### Scenario 1: Database Corruption

**Symptoms:**
- Application unable to connect to database
- PostgreSQL errors in logs
- Data inconsistencies

**Recovery Time:** 15-30 minutes

### Scenario 2: Complete Server Failure

**Symptoms:**
- Server unreachable
- Hardware failure
- Cloud instance terminated

**Recovery Time:** 30-60 minutes

### Scenario 3: Data Loss (Accidental Deletion)

**Symptoms:**
- Missing data
- Accidental table drop
- User reports data loss

**Recovery Time:** 20-40 minutes

### Scenario 4: Security Breach / Ransomware

**Symptoms:**
- Unauthorized access detected
- Encrypted files
- Guardian Defense lockdown activated

**Recovery Time:** 1-2 hours

---

## Step-by-Step Recovery Procedures

### Procedure 1: Database Recovery from Backup

#### Prerequisites
- SSH access to server
- Backup encryption key
- PostgreSQL container running

#### Steps

1. **Identify Latest Valid Backup**
   ```bash
   ls -lh /var/backups/shivx/ | grep shivx_backup
   ```

2. **Verify Backup Integrity**
   ```bash
   ./scripts/restore.sh --verify /var/backups/shivx/shivx_backup_YYYYMMDD_HHMMSS.sql.gz.enc
   ```

3. **Stop Trading Operations**
   ```bash
   # Activate circuit breaker
   curl -X POST https://api.shivx.io/api/trading/circuit-breaker/enable
   ```

4. **Create Pre-Restore Snapshot**
   ```bash
   docker exec shivx-postgres pg_dump -U shivx shivx > /tmp/pre_restore_snapshot.sql
   ```

5. **Perform Restore**
   ```bash
   ./scripts/restore.sh /var/backups/shivx/shivx_backup_YYYYMMDD_HHMMSS.sql.gz.enc
   ```

6. **Validate Restore**
   ```bash
   # Check database connectivity
   docker exec shivx-postgres psql -U shivx -d shivx -c "SELECT COUNT(*) FROM trades;"

   # Verify recent data
   docker exec shivx-postgres psql -U shivx -d shivx -c "SELECT MAX(created_at) FROM trades;"
   ```

7. **Resume Trading Operations**
   ```bash
   curl -X POST https://api.shivx.io/api/trading/circuit-breaker/disable
   ```

8. **Monitor System**
   - Check Grafana dashboards
   - Review application logs
   - Verify trading functionality

#### Rollback Procedure
If restore fails:
```bash
./scripts/restore.sh /tmp/pre_restore_snapshot.sql
```

---

### Procedure 2: Complete System Recovery (New Server)

#### Prerequisites
- New server or cloud instance
- Access to backups (S3 or off-site)
- DNS access to update records
- SSL certificates

#### Steps

1. **Provision New Server**
   ```bash
   # Minimum specs: 4 CPU, 16GB RAM, 200GB SSD
   # OS: Ubuntu 22.04 LTS
   ```

2. **Install Prerequisites**
   ```bash
   # Docker & Docker Compose
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER

   # Git
   sudo apt update && sudo apt install -y git
   ```

3. **Clone Repository**
   ```bash
   git clone https://github.com/yourorg/shivx.git
   cd shivx
   git checkout production
   ```

4. **Download Backups**
   ```bash
   aws s3 sync s3://shivx-production-backups/backups/ /var/backups/shivx/
   ```

5. **Restore Secrets**
   ```bash
   # Copy secrets from secure vault
   aws secretsmanager get-secret-value --secret-id shivx-production-secrets \
       --query SecretString --output text > deploy/secrets/.env.production

   # Generate new secrets if needed
   ./scripts/generate_secrets.sh deploy/secrets
   ```

6. **Setup SSL Certificates**
   ```bash
   # For Let's Encrypt
   ./scripts/setup_ssl.sh deploy/nginx/ssl your-domain.com admin@your-domain.com letsencrypt

   # OR use existing certificates
   cp /path/to/certificates/* deploy/nginx/ssl/
   ```

7. **Start Infrastructure**
   ```bash
   docker-compose -f deploy/docker-compose.yml up -d postgres redis
   # Wait for databases to be ready
   sleep 30
   ```

8. **Restore Database**
   ```bash
   BACKUP_FILE=$(ls -t /var/backups/shivx/shivx_backup_*.sql.gz.enc | head -1)
   ./scripts/restore.sh --force $BACKUP_FILE
   ```

9. **Restore Volumes**
   ```bash
   cd /var/backups/shivx/volumes/latest/
   for volume in *.tar.gz; do
       volume_name=${volume%.tar.gz}
       docker run --rm -v $volume_name:/volume -v $(pwd):/backup alpine \
           tar xzf /backup/$volume -C /volume
   done
   ```

10. **Start All Services**
    ```bash
    docker-compose -f deploy/docker-compose.yml up -d
    ```

11. **Update DNS**
    ```bash
    # Point domain to new server IP
    # Wait for DNS propagation (up to 5 minutes)
    ```

12. **Validate System**
    ```bash
    # Health checks
    curl https://api.shivx.io/api/health/ready

    # Check metrics
    curl https://api.shivx.io/metrics

    # Test trading functionality (paper mode)
    curl -X POST https://api.shivx.io/api/trading/test
    ```

13. **Enable Monitoring**
    - Verify Grafana dashboards: https://grafana.shivx.io
    - Check Prometheus targets: https://prometheus.shivx.io/targets
    - Confirm alerts are flowing to PagerDuty/Slack

14. **Resume Production Trading**
    ```bash
    # Only after thorough validation
    curl -X POST https://api.shivx.io/api/trading/resume
    ```

---

### Procedure 3: Point-in-Time Recovery

Use when you need to recover to a specific point in time (e.g., before data corruption).

#### Steps

1. **Identify Target Time**
   ```bash
   # Example: 2025-01-15 14:30:00
   TARGET_TIME="2025-01-15 14:30:00"
   ```

2. **Find Appropriate Backup**
   ```bash
   # Find backup taken BEFORE target time
   ls -lt /var/backups/shivx/ | grep -B 5 "Jan 15"
   ```

3. **Perform PITR**
   ```bash
   ./scripts/restore.sh --pitr "$TARGET_TIME" /var/backups/shivx/shivx_backup_20250115_120000.sql.gz.enc
   ```

4. **Validate Recovery Point**
   ```bash
   docker exec shivx-postgres psql -U shivx -d shivx -c \
       "SELECT MAX(created_at) FROM trades WHERE created_at <= '$TARGET_TIME';"
   ```

---

### Procedure 4: Security Incident Recovery

#### Steps

1. **Isolate System**
   ```bash
   # Enable Guardian Defense lockdown
   docker exec shivx-app curl -X POST http://localhost:8000/api/security/lockdown

   # Block all external access
   sudo ufw deny from any to any
   ```

2. **Assess Damage**
   ```bash
   # Check for unauthorized changes
   git status
   git diff HEAD

   # Review audit logs
   docker logs shivx-app | grep -i "security"

   # Check for malware
   sudo clamscan -r /home/shivx/
   ```

3. **Rebuild from Clean Backup**
   ```bash
   # Use backup from BEFORE incident
   CLEAN_BACKUP="/var/backups/shivx/shivx_backup_20250114_020000.sql.gz.enc"
   ./scripts/restore.sh --force $CLEAN_BACKUP
   ```

4. **Rotate All Secrets**
   ```bash
   # Generate new secrets
   ./scripts/generate_secrets.sh deploy/secrets

   # Update environment
   source deploy/secrets/.env.production

   # Restart all services
   docker-compose -f deploy/docker-compose.yml restart
   ```

5. **Harden Security**
   ```bash
   # Update all packages
   sudo apt update && sudo apt upgrade -y

   # Update firewall rules
   sudo ufw default deny incoming
   sudo ufw allow 22/tcp  # SSH
   sudo ufw allow 80/tcp  # HTTP
   sudo ufw allow 443/tcp # HTTPS
   sudo ufw enable

   # Enable fail2ban
   sudo apt install -y fail2ban
   sudo systemctl enable fail2ban
   ```

6. **Security Audit**
   ```bash
   # Run security scan
   ./scripts/security_scan_real.py

   # Check for vulnerabilities
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy shivx-app
   ```

7. **Notify Stakeholders**
   - Security team
   - Compliance officer
   - Affected users (if applicable)
   - Regulatory authorities (if required)

---

## Validation and Testing

### Monthly DR Test Checklist

- [ ] Verify all backups are completing successfully
- [ ] Test database restore on staging environment
- [ ] Validate backup encryption/decryption
- [ ] Test S3 download and restore
- [ ] Verify point-in-time recovery
- [ ] Test complete system rebuild
- [ ] Validate monitoring and alerting
- [ ] Review and update runbook
- [ ] Train on-call engineers
- [ ] Document lessons learned

### Automated Tests

```bash
# Run DR validation suite
./scripts/test_disaster_recovery.sh

# Verify backup integrity
./scripts/validate_backups.sh

# Test restore performance
time ./scripts/restore.sh --verify /var/backups/shivx/latest_backup.sql.gz.enc
```

---

## Post-Recovery Checklist

After any recovery procedure:

- [ ] System health checks passing
- [ ] All services running and healthy
- [ ] Database connectivity verified
- [ ] Trading functionality tested (paper mode first)
- [ ] Monitoring dashboards operational
- [ ] Alerts flowing to correct channels
- [ ] SSL certificates valid
- [ ] Backups resuming normally
- [ ] Performance metrics normal
- [ ] Security scans completed
- [ ] Incident report filed
- [ ] Runbook updated with lessons learned

---

## Contact Information

### On-Call Team

- **Primary On-Call:** +1-XXX-XXX-XXXX
- **Secondary On-Call:** +1-XXX-XXX-XXXX
- **PagerDuty:** https://yourorg.pagerduty.com

### Escalation

- **Engineering Manager:** manager@shivx.io
- **CTO:** cto@shivx.io
- **Security Team:** security@shivx.io

### External Support

- **AWS Support:** 1-866-221-0634 (Enterprise)
- **DNS Provider:** support@cloudflare.com
- **SSL Certificate Authority:** support@letsencrypt.org

---

## Appendix

### Backup File Naming Convention

```
shivx_backup_YYYYMMDD_HHMMSS.sql.gz.enc
```

Example: `shivx_backup_20250115_020000.sql.gz.enc`

### Recovery Time Estimates

| Scenario | Recovery Time | Complexity |
|----------|--------------|------------|
| Database restore only | 15-30 min | Low |
| Complete system rebuild | 30-60 min | Medium |
| Point-in-time recovery | 20-40 min | Medium |
| Security incident recovery | 1-2 hours | High |

### Important Paths

- Backups: `/var/backups/shivx`
- Secrets: `/etc/shivx/secrets`
- Logs: `/var/log/shivx`
- Configuration: `/home/shivx/deploy`
- SSL Certificates: `/etc/letsencrypt/live`

---

**Document Version:** 1.0
**Last Updated:** 2025-01-28
**Next Review:** 2025-02-28
**Owner:** Infrastructure Team
