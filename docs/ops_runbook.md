# ShivX Operations Runbook

> **Version:** 2.0.0
> **Last Updated:** 2025-10-28
> **Maintained by:** ShivX DevOps Team

## Table of Contents

- [Overview](#overview)
- [Emergency Contacts](#emergency-contacts)
- [System Architecture](#system-architecture)
- [Deployment Procedures](#deployment-procedures)
- [Rollback Procedures](#rollback-procedures)
- [Scaling Operations](#scaling-operations)
- [Backup & Restore](#backup--restore)
- [Monitoring & Alerts](#monitoring--alerts)
- [Troubleshooting](#troubleshooting)
- [Maintenance Tasks](#maintenance-tasks)
- [Security Incidents](#security-incidents)

---

## Overview

This runbook provides step-by-step procedures for operating and maintaining the ShivX AI Trading System. It covers common operational tasks, incident response, and troubleshooting procedures.

### System Components

- **API Server**: FastAPI application (port 8000)
- **Database**: PostgreSQL (port 5432)
- **Cache**: Redis (port 6379)
- **Monitoring**: Prometheus (port 9090) + Grafana (port 3000)
- **Worker Processes**: Background task executors

### Key Metrics

- **Uptime SLA**: 99.9%
- **Response Time**: < 200ms (p95)
- **Error Rate**: < 0.1%
- **Data Loss Tolerance**: Zero (with backups)

---

## Emergency Contacts

| Role | Name | Contact | Availability |
|------|------|---------|--------------|
| Primary On-Call | DevOps Team | devops@shivx.ai | 24/7 |
| Secondary On-Call | Engineering Lead | engineering@shivx.ai | 24/7 |
| Database Admin | DBA Team | dba@shivx.ai | Business Hours |
| Security Team | Security Team | security@shivx.ai | 24/7 |
| Management | CTO | cto@shivx.ai | Emergency Only |

### Escalation Path

1. **Level 1**: On-call Engineer (Response: 15 min)
2. **Level 2**: Engineering Lead (Response: 30 min)
3. **Level 3**: CTO (Response: 1 hour)

---

## System Architecture

```
┌─────────────────────────────────────────┐
│         Load Balancer (HAProxy)         │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼─────┐   ┌─────▼──────┐
│  API Node  │   │  API Node  │
│  (Docker)  │   │  (Docker)  │
└─────┬──────┘   └──────┬─────┘
      │                 │
      └────────┬────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼──┐  ┌───▼────┐
│ Redis │  │ DB  │  │ Worker │
└───────┘  └─────┘  └────────┘
```

---

## Deployment Procedures

### 1. Standard Deployment (Zero-Downtime)

**Duration:** ~15-20 minutes
**Risk:** Low
**Rollback Time:** < 5 minutes

#### Pre-Deployment Checklist

- [ ] Code reviewed and approved
- [ ] CI/CD pipeline passed (all tests green)
- [ ] Database migrations tested in staging
- [ ] Rollback plan prepared
- [ ] Monitoring dashboards open
- [ ] Team notified in Slack #deployments

#### Deployment Steps

```bash
# 1. Verify staging deployment
ssh staging-server
cd /opt/shivx
docker-compose ps
# Verify all services healthy

# 2. Pull latest changes on production
ssh production-server
cd /opt/shivx
git fetch origin
git checkout v2.0.0  # Use specific tag

# 3. Backup database
./scripts/backup_db.sh
# Verify backup: ls -lh /backup/db/

# 4. Run database migrations (if any)
source venv/bin/activate
alembic upgrade head
# Check migration status: alembic current

# 5. Build new Docker images
docker-compose build

# 6. Rolling update (one node at a time)
# Update Node 1
docker-compose up -d --no-deps --build api-node-1
sleep 30  # Wait for health checks
curl http://localhost:8001/health  # Verify healthy

# Update Node 2
docker-compose up -d --no-deps --build api-node-2
sleep 30
curl http://localhost:8002/health

# 7. Update worker processes
docker-compose up -d --no-deps --build worker

# 8. Verify deployment
./scripts/verify_deployment.sh

# 9. Monitor for 15 minutes
# Watch logs: docker-compose logs -f
# Check metrics: http://grafana.shivx.ai

# 10. Update deployment record
echo "$(date): Deployed v2.0.0" >> /var/log/deployments.log
```

#### Post-Deployment Verification

```bash
# Health check
curl http://api.shivx.ai/health
# Expected: {"status": "healthy", "version": "2.0.0"}

# Test critical endpoints
curl -X POST http://api.shivx.ai/api/v1/trades/test
curl http://api.shivx.ai/api/v1/analytics/health

# Check error rates (should be < 0.1%)
curl http://prometheus:9090/api/v1/query?query=rate(http_errors_total[5m])

# Verify database connections
docker exec shivx-api python -c "from config.database import test_connection; test_connection()"
```

---

## Rollback Procedures

### Emergency Rollback (< 5 minutes)

**Use when:** Critical bug, high error rate, system instability

```bash
# 1. Immediate rollback to previous version
cd /opt/shivx
git checkout v1.9.0  # Previous stable version

# 2. Restart services with old version
docker-compose down
docker-compose up -d

# 3. Verify rollback
curl http://api.shivx.ai/health
# Expected: {"status": "healthy", "version": "1.9.0"}

# 4. Notify team
# Post in #incidents: "Rolled back to v1.9.0 due to [reason]"

# 5. Monitor
# Watch metrics for 30 minutes to ensure stability
```

### Database Rollback

**Use when:** Migration causes issues

```bash
# 1. Check current migration
alembic current

# 2. Rollback one migration
alembic downgrade -1

# 3. Verify database state
psql -U shivx -d shivx_prod -c "SELECT version_num FROM alembic_version;"

# 4. Restart services
docker-compose restart api-node-1 api-node-2
```

---

## Scaling Operations

### Horizontal Scaling (Add More Nodes)

#### Scale Up (Add API Node)

```bash
# 1. Update docker-compose.yml
# Add new service:
cat >> docker-compose.yml << EOF
  api-node-3:
    build: .
    environment:
      - NODE_ID=3
    ports:
      - "8003:8000"
EOF

# 2. Start new node
docker-compose up -d api-node-3

# 3. Add to load balancer
# Edit /etc/haproxy/haproxy.cfg
sudo nano /etc/haproxy/haproxy.cfg
# Add: server node3 localhost:8003 check

# 4. Reload load balancer
sudo systemctl reload haproxy

# 5. Verify
curl http://api.shivx.ai/health
# Check all nodes receiving traffic
```

#### Scale Down (Remove API Node)

```bash
# 1. Remove from load balancer
sudo nano /etc/haproxy/haproxy.cfg
# Remove: server node3 localhost:8003 check
sudo systemctl reload haproxy

# 2. Wait for active connections to drain (5 minutes)
watch 'docker stats api-node-3'

# 3. Stop node
docker-compose stop api-node-3
docker-compose rm -f api-node-3

# 4. Remove from docker-compose.yml
```

### Vertical Scaling (Increase Resources)

#### Increase CPU/Memory

```bash
# 1. Update docker-compose.yml
# Modify resource limits:
services:
  api-node-1:
    deploy:
      resources:
        limits:
          cpus: '4.0'     # Was: 2.0
          memory: 8G      # Was: 4G

# 2. Recreate containers
docker-compose up -d --force-recreate api-node-1

# 3. Monitor performance
docker stats
```

### Database Scaling

#### Increase Connection Pool

```python
# Edit config/database.py
SQLALCHEMY_POOL_SIZE = 50  # Was: 20
SQLALCHEMY_MAX_OVERFLOW = 100  # Was: 40
```

#### Add Read Replicas

```bash
# 1. Set up PostgreSQL streaming replication
# On primary:
psql -U postgres -c "CREATE USER replicator WITH REPLICATION PASSWORD 'repl_password';"

# 2. Configure pg_hba.conf
echo "host replication replicator replica_ip/32 md5" >> /etc/postgresql/15/main/pg_hba.conf

# 3. Update application config
# Edit .env:
DATABASE_READ_URL=postgresql://user:pass@replica_ip:5432/shivx_prod

# 4. Deploy changes
docker-compose up -d
```

---

## Backup & Restore

### Automated Backups

Backups run automatically at:
- **Database**: Daily at 2:00 AM UTC
- **Redis**: Daily at 3:00 AM UTC
- **Configuration**: On every deployment
- **Models**: Weekly on Sundays at 1:00 AM UTC

Retention:
- Daily backups: 7 days
- Weekly backups: 4 weeks
- Monthly backups: 12 months

### Manual Database Backup

```bash
# Full backup
./scripts/backup_db.sh

# Or manually:
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U shivx -h localhost -d shivx_prod \
    | gzip > /backup/db/shivx_${BACKUP_DATE}.sql.gz

# Verify backup
gunzip -c /backup/db/shivx_${BACKUP_DATE}.sql.gz | head -20

# Upload to S3 (if configured)
aws s3 cp /backup/db/shivx_${BACKUP_DATE}.sql.gz \
    s3://shivx-backups/database/
```

### Database Restore

```bash
# 1. Stop application (to prevent writes)
docker-compose stop api-node-1 api-node-2 worker

# 2. Create restore point
pg_dump -U shivx -h localhost -d shivx_prod \
    > /backup/db/before_restore_$(date +%Y%m%d_%H%M%S).sql

# 3. Drop existing database (DESTRUCTIVE!)
psql -U postgres -c "DROP DATABASE shivx_prod;"
psql -U postgres -c "CREATE DATABASE shivx_prod OWNER shivx;"

# 4. Restore from backup
BACKUP_FILE="/backup/db/shivx_20251028_020000.sql.gz"
gunzip -c $BACKUP_FILE | psql -U shivx -d shivx_prod

# 5. Verify restore
psql -U shivx -d shivx_prod -c "SELECT COUNT(*) FROM trades;"
psql -U shivx -d shivx_prod -c "SELECT MAX(created_at) FROM trades;"

# 6. Restart application
docker-compose up -d

# 7. Verify application
curl http://api.shivx.ai/health
```

### Redis Backup & Restore

```bash
# Backup
redis-cli --rdb /backup/redis/dump_$(date +%Y%m%d_%H%M%S).rdb

# Restore
# 1. Stop Redis
docker-compose stop redis

# 2. Replace dump file
cp /backup/redis/dump_20251028_030000.rdb /var/lib/redis/dump.rdb

# 3. Start Redis
docker-compose up -d redis

# 4. Verify
redis-cli PING
redis-cli DBSIZE
```

---

## Monitoring & Alerts

### Key Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| System Overview | http://grafana.shivx.ai/d/overview | Overall health |
| API Performance | http://grafana.shivx.ai/d/api | Request/response metrics |
| Database | http://grafana.shivx.ai/d/db | DB queries, connections |
| Trading Activity | http://grafana.shivx.ai/d/trading | Trades, P&L |
| Error Tracking | http://grafana.shivx.ai/d/errors | Errors by type |

### Critical Alerts

#### High Error Rate (>1%)

```bash
# 1. Check error logs
docker-compose logs --tail=100 api-node-1 | grep ERROR

# 2. Check specific error types
curl http://prometheus:9090/api/v1/query?query=http_errors_total

# 3. Common causes:
#    - Database connection issues
#    - External API failures
#    - Memory exhaustion

# 4. Quick fixes:
docker-compose restart api-node-1  # Try restart
./scripts/clear_redis_cache.sh     # Clear cache if stale
```

#### High Response Time (>500ms p95)

```bash
# 1. Check slow queries
docker exec shivx-db psql -U shivx -d shivx_prod -c \
    "SELECT query, mean_time, calls FROM pg_stat_statements \
     ORDER BY mean_time DESC LIMIT 10;"

# 2. Check connection pool
curl http://api.shivx.ai/metrics | grep db_pool

# 3. Quick fixes:
# Increase connection pool
# Add database indexes
# Enable query caching
```

#### Database Connection Failures

```bash
# 1. Check database status
docker-compose ps db
pg_isready -h localhost -p 5432

# 2. Check connections
psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
psql -U postgres -c "SELECT max_conn FROM pg_settings WHERE name = 'max_connections';"

# 3. Kill idle connections
psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
    WHERE datname = 'shivx_prod' AND state = 'idle' AND state_change < NOW() - INTERVAL '10 minutes';"

# 4. Restart if necessary
docker-compose restart db
```

#### Out of Memory

```bash
# 1. Check memory usage
docker stats --no-stream

# 2. Identify memory hogs
ps aux --sort=-%mem | head -20

# 3. Quick fixes:
# Restart high-memory container
docker-compose restart api-node-1

# Clear Redis cache
redis-cli FLUSHDB

# Restart Redis
docker-compose restart redis
```

---

## Troubleshooting

### Application Won't Start

```bash
# 1. Check logs
docker-compose logs api-node-1

# 2. Common issues:
# - Port already in use
sudo lsof -i :8000

# - Missing environment variables
docker-compose config | grep -i error

# - Database not ready
docker-compose logs db | grep "ready to accept connections"

# 3. Start with verbose logging
docker-compose up api-node-1  # Without -d to see output
```

### Database Connection Issues

```bash
# 1. Test connectivity
psql -U shivx -h localhost -d shivx_prod -c "SELECT 1;"

# 2. Check DATABASE_URL in .env
grep DATABASE_URL .env

# 3. Check pg_hba.conf
sudo cat /etc/postgresql/15/main/pg_hba.conf | grep shivx

# 4. Restart PostgreSQL
sudo systemctl restart postgresql
```

### Redis Connection Issues

```bash
# 1. Test connectivity
redis-cli -h localhost -p 6379 PING

# 2. Check Redis status
docker-compose ps redis

# 3. Check logs
docker-compose logs redis

# 4. Restart Redis
docker-compose restart redis
```

### High CPU Usage

```bash
# 1. Identify process
top -o %CPU

# 2. Check for runaway tasks
docker stats

# 3. Check for infinite loops in code
docker-compose logs | grep -i "loop\|infinite\|stuck"

# 4. Profile the application
py-spy top --pid $(docker inspect -f '{{.State.Pid}}' shivx-api)
```

### Disk Space Issues

```bash
# 1. Check disk usage
df -h
du -sh /var/lib/docker/*
du -sh /opt/shivx/*

# 2. Clean Docker
docker system prune -a --volumes

# 3. Clean logs
find /var/log -name "*.log" -mtime +30 -delete
docker-compose logs --tail=0 > /dev/null  # Truncate logs

# 4. Clean old backups
find /backup -mtime +7 -delete
```

---

## Maintenance Tasks

### Weekly Tasks

```bash
# 1. Review error logs
docker-compose logs --since 7d | grep ERROR > /tmp/weekly_errors.log
# Review /tmp/weekly_errors.log

# 2. Check disk space
df -h
# Alert if > 80%

# 3. Review slow queries
docker exec shivx-db psql -U shivx -d shivx_prod -f /scripts/slow_queries.sql

# 4. Update dependencies (if needed)
# Check for security updates: pip list --outdated
```

### Monthly Tasks

```bash
# 1. Vacuum database
docker exec shivx-db psql -U shivx -d shivx_prod -c "VACUUM ANALYZE;"

# 2. Rotate logs
./scripts/rotate_logs.sh

# 3. Review and optimize indexes
docker exec shivx-db psql -U shivx -d shivx_prod -f /scripts/index_review.sql

# 4. Test backup restore
./scripts/test_backup_restore.sh

# 5. Update SSL certificates (if expiring soon)
certbot renew --dry-run
```

### Quarterly Tasks

```bash
# 1. Review and update runbook
# Update this document with new procedures

# 2. Disaster recovery drill
# Test full restore from backup

# 3. Security audit
./scripts/security_audit.sh

# 4. Performance review
# Analyze metrics and optimize bottlenecks

# 5. Dependency updates
# Update to latest stable versions
pip list --outdated
# Test in staging, then deploy
```

---

## Security Incidents

### Suspected Breach

```bash
# 1. IMMEDIATE ACTIONS
# Disconnect affected systems
sudo iptables -A INPUT -j DROP  # Block all incoming
sudo iptables -A OUTPUT -j DROP  # Block all outgoing

# 2. Preserve evidence
tar -czf /forensics/logs_$(date +%Y%m%d_%H%M%S).tar.gz /var/log/
docker-compose logs > /forensics/docker_logs_$(date +%Y%m%d_%H%M%S).log

# 3. Notify security team
# Email: security@shivx.ai
# Slack: #security-incidents

# 4. Change all credentials
# Database passwords
# API keys
# SSH keys
# Encryption keys

# 5. Review audit logs
# Check for unauthorized access
grep -i "unauthorized\|failed\|deny" /var/log/auth.log
```

### DDoS Attack

```bash
# 1. Enable rate limiting
# Edit nginx.conf or HAProxy config
# Limit to 100 req/sec per IP

# 2. Block attack IPs
sudo iptables -A INPUT -s ATTACK_IP -j DROP

# 3. Enable Cloudflare DDoS protection (if configured)
# Log into Cloudflare dashboard
# Enable "I'm Under Attack" mode

# 4. Scale up resources temporarily
./scripts/scale_up.sh

# 5. Monitor
watch 'netstat -an | grep :8000 | wc -l'
```

---

## Appendix

### Useful Commands

```bash
# View all running containers
docker-compose ps

# View resource usage
docker stats

# View logs
docker-compose logs -f [service]

# Execute command in container
docker-compose exec api bash

# Restart service
docker-compose restart [service]

# View configuration
docker-compose config
```

### Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| .env | Environment variables | /opt/shivx/.env |
| docker-compose.yml | Docker config | /opt/shivx/docker-compose.yml |
| nginx.conf | Web server config | /etc/nginx/nginx.conf |
| haproxy.cfg | Load balancer config | /etc/haproxy/haproxy.cfg |
| postgresql.conf | Database config | /etc/postgresql/15/main/postgresql.conf |

### Log Locations

| Service | Log Path |
|---------|----------|
| Application | /opt/shivx/logs/app.log |
| Docker | docker-compose logs |
| Nginx | /var/log/nginx/ |
| PostgreSQL | /var/log/postgresql/ |
| System | /var/log/syslog |

---

**Document Version:** 2.0.0
**Last Review:** 2025-10-28
**Next Review:** 2026-01-28

For questions or updates to this runbook, contact: devops@shivx.ai
