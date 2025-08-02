# Media Server Container Security Implementation

This directory contains a comprehensive container security strategy for the media server infrastructure, implementing defense-in-depth with multiple security layers.

## 🛡️ Security Components

### 1. **Secure Docker Compose Configuration**
- `docker-compose-secure.yml` - Production-ready secure configuration
- Non-root users for all containers
- Read-only root filesystems where possible
- Capability restrictions and security options
- Network segmentation with isolated zones

### 2. **AppArmor Profiles**
- `apparmor/media-server-profile` - Profile for media servers (Jellyfin, Plex)
- `apparmor/arr-suite-profile` - Profile for Arr suite applications
- `apparmor/download-client-profile` - Profile for download clients
- `apparmor/monitoring-profile` - Profile for monitoring tools

### 3. **Seccomp Profiles**
- `seccomp/media-server.json` - Syscall filtering for media servers
- `seccomp/arr-suite.json` - Syscall filtering for Arr suite
- `seccomp/download-client.json` - Syscall filtering for download clients
- `seccomp/traefik.json` - Syscall filtering for reverse proxy

### 4. **Security Scripts**
- `scripts/security-scan.sh` - Automated vulnerability scanning with Trivy
- `scripts/rotate-secrets.sh` - Automated secret rotation
- `scripts/setup-firewall.sh` - Network segmentation and firewall rules
- `scripts/incident-response.sh` - Automated incident response

### 5. **Secret Management**
- Docker secrets for all sensitive data
- Automated rotation capabilities
- Encrypted storage
- Zero-exposure in environment variables

## 🚀 Quick Start

### Initial Setup

1. **Initialize secrets:**
```bash
sudo ./scripts/rotate-secrets.sh init
```

2. **Deploy secure stack:**
```bash
docker-compose -f security/docker-compose-secure.yml up -d
```

3. **Configure firewall:**
```bash
sudo ./scripts/setup-firewall.sh setup
```

4. **Run security scan:**
```bash
./scripts/security-scan.sh
```

## 📋 Security Checklist

- [ ] Enable Docker user namespace remapping
- [ ] Deploy AppArmor profiles to `/etc/apparmor.d/`
- [ ] Configure Docker daemon for content trust
- [ ] Initialize and distribute secrets
- [ ] Apply network firewall rules
- [ ] Schedule regular security scans
- [ ] Configure log aggregation
- [ ] Set up monitoring alerts
- [ ] Document incident response procedures
- [ ] Schedule secret rotation

## 🔒 Security Features

### Container Hardening
- **Non-root execution** - All containers run as non-privileged users
- **Read-only filesystems** - Immutable container filesystems
- **Capability dropping** - Minimal Linux capabilities
- **No new privileges** - Prevent privilege escalation
- **Security profiles** - AppArmor and Seccomp enforcement

### Network Security
- **Network segmentation** - Isolated security zones
- **Firewall rules** - iptables-based traffic control
- **VPN kill switch** - Download traffic isolation
- **Rate limiting** - DDoS protection
- **TLS encryption** - End-to-end encryption

### Secret Management
- **Docker secrets** - Native secret management
- **Automated rotation** - Zero-downtime secret updates
- **Encrypted storage** - Secrets encrypted at rest
- **Access control** - Service-specific secret access
- **Audit logging** - Secret access tracking

### Vulnerability Management
- **Automated scanning** - Trivy integration
- **SBOM generation** - Software inventory tracking
- **CVE monitoring** - Vulnerability tracking
- **Patch management** - Update procedures
- **Compliance checking** - CIS benchmark validation

## 🎯 Usage Examples

### Run Security Scan
```bash
# Full security scan
./scripts/security-scan.sh

# Scan specific image
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image jellyfin/jellyfin:latest
```

### Rotate Secrets
```bash
# Rotate all secrets
sudo ./scripts/rotate-secrets.sh rotate-all

# Rotate specific category
sudo ./scripts/rotate-secrets.sh rotate-api
sudo ./scripts/rotate-secrets.sh rotate-db
sudo ./scripts/rotate-secrets.sh rotate-auth
```

### Monitor Security
```bash
# Check container security settings
docker inspect jellyfin | jq '.[0].HostConfig.SecurityOpt'

# View firewall rules
sudo ./scripts/setup-firewall.sh status

# Check secret status
docker secret ls
```

## 📊 Security Architecture

```
┌─────────────────┐
│   DMZ Network   │ ← Public Internet
│   (Traefik)     │
└────────┬────────┘
         │ TLS
┌────────┴────────┐
│ Frontend Network│ ← Authelia Authentication
│ (Web Services)  │
└────────┬────────┘
         │
┌────────┴────────┐
│ Backend Network │ ← Isolated Databases
│ (DB, Cache)     │
└─────────────────┘

┌─────────────────┐
│Downloads Network│ ← VPN Only
│ (Torrents)      │
└─────────────────┘
```

## 🚨 Incident Response

### Detection
1. Falco runtime monitoring
2. Prometheus security metrics
3. Log aggregation alerts
4. Network anomaly detection

### Response Procedures
1. **Isolate** - Network disconnection
2. **Investigate** - Forensic analysis
3. **Contain** - Stop malicious activity
4. **Eradicate** - Remove threats
5. **Recover** - Restore services
6. **Review** - Post-incident analysis

### Emergency Commands
```bash
# Stop suspicious container
docker stop <container_name>

# Disconnect from network
docker network disconnect <network> <container>

# Create forensic snapshot
docker commit <container> incident-<timestamp>

# View container processes
docker top <container>
```

## 📈 Monitoring & Alerts

### Security Metrics
- Container escape attempts
- Unauthorized file access
- Network policy violations
- Failed authentication attempts
- Privilege escalation attempts
- Resource abuse patterns

### Alert Configuration
Configure alerts in Grafana for:
- Critical CVEs detected
- Security profile violations
- Firewall rule breaches
- Secret access anomalies
- Container runtime errors

## 🔧 Maintenance

### Daily Tasks
- Review security alerts
- Check container health
- Verify backup completion

### Weekly Tasks
- Run vulnerability scans
- Review access logs
- Update security signatures

### Monthly Tasks
- Rotate secrets
- Security audit
- Update documentation
- Patch containers

### Quarterly Tasks
- Penetration testing
- Disaster recovery drill
- Security training
- Architecture review

## 📚 Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Container Security](https://owasp.org/www-project-docker-security/)

## ⚠️ Important Notes

1. **Test in staging** - Always test security changes in a non-production environment
2. **Backup first** - Create backups before applying security updates
3. **Monitor impact** - Watch for performance impacts from security controls
4. **Document changes** - Keep security configuration under version control
5. **Stay updated** - Regularly update security tools and signatures

## 🆘 Support

For security incidents or questions:
1. Check logs in `/var/log/docker/` and container logs
2. Review security reports in `security/reports/`
3. Consult incident response procedures
4. Escalate critical issues immediately

Remember: Security is an ongoing process, not a one-time configuration!