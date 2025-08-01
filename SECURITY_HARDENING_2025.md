# Media Server Security Hardening Guide - 2025

## 🛡️ Security Overview

This guide provides comprehensive security hardening for your 2025 media server setup, implementing defense-in-depth strategies with current best practices.

## 🔐 Security Architecture

### **Security Layers**
1. **Network Level**: Cloudflare protection + VPN
2. **Application Level**: Traefik reverse proxy with SSL
3. **Container Level**: Security-focused Docker configuration
4. **Access Level**: Authentication and authorization
5. **Monitoring Level**: Intrusion detection and logging

## 🌐 Network Security

### **Cloudflare Security Features**

**Essential Settings:**
```bash
# Cloudflare Dashboard Settings
Security Level: High
Browser Integrity Check: On
Challenge Passage: 30 minutes
Security Events: Monitor and block
```

**Firewall Rules:**
```javascript
// Block non-essential countries (adjust as needed)
(ip.geoip.country ne "US" and ip.geoip.country ne "CA" and ip.geoip.country ne "GB")

// Rate limiting
(http.request.uri.path contains "/api/") and (rate(1m) > 30)

// Block common attack patterns
(http.request.uri.query contains "../../" or 
 http.request.uri.query contains "<script" or
 http.request.uri.query contains "SELECT * FROM")
```

**Page Rules:**
```
1. *.yourdomain.com/admin/* → Security Level: High, Cache Level: Bypass
2. jellyfin.yourdomain.com → Security Level: Medium, SSL: Full (Strict)
3. *.yourdomain.com → SSL: Full (Strict), Always Use HTTPS: On
```

### **Server Firewall (UFW)**

```bash
# Reset and configure UFW
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow essential services
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# Block direct access to services
sudo ufw deny 8096/tcp comment 'Block Jellyfin direct'
sudo ufw deny 8080/tcp comment 'Block qBittorrent direct'
sudo ufw deny 9696/tcp comment 'Block Prowlarr direct'
sudo ufw deny 8989/tcp comment 'Block Sonarr direct'
sudo ufw deny 7878/tcp comment 'Block Radarr direct'

# Enable firewall
sudo ufw enable

# Show status
sudo ufw status verbose
```

## 🔒 Authentication Security

### **Multi-Factor Authentication Setup**

**Authelia Configuration** (Optional Advanced Setup):
```yaml
# config/authelia/configuration.yml
server:
  host: 0.0.0.0
  port: 9091

log:
  level: warn
  file_path: /config/authelia.log

theme: dark

jwt_secret: your_jwt_secret_here

default_redirection_url: https://home.yourdomain.com

authentication_backend:
  password_reset:
    disable: false
  refresh_interval: 5m
  file:
    path: /config/users_database.yml
    password:
      algorithm: argon2id
      iterations: 1
      salt_length: 16
      parallelism: 8
      memory: 64

access_control:
  default_policy: deny
  rules:
    - domain: jellyfin.yourdomain.com
      policy: one_factor
    - domain: home.yourdomain.com  
      policy: one_factor
    - domain: requests.yourdomain.com
      policy: one_factor
    - domain: "*.yourdomain.com"
      policy: two_factor

session:
  name: authelia_session
  secret: your_session_secret_here
  expiration: 3600
  inactivity: 300
  remember_me_duration: 1M

regulation:
  max_retries: 3
  find_time: 120
  ban_time: 300

storage:
  local:
    path: /config/db.sqlite3

notifier:
  filesystem:
    filename: /config/notification.txt

totp:
  issuer: yourdomain.com
  period: 30
  skew: 1
```

### **Strong Password Policy**

```bash
# Generate secure passwords
openssl rand -base64 32

# Example .env passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
AUTHELIA_JWT_SECRET=$(openssl rand -base64 32)
AUTHELIA_SESSION_SECRET=$(openssl rand -base64 32)
```

## 🐳 Container Security

### **Enhanced Docker Compose Security**

```yaml
# Security-hardened service example
jellyfin:
  image: jellyfin/jellyfin:latest
  container_name: jellyfin
  restart: unless-stopped
  
  # Security configurations
  security_opt:
    - no-new-privileges:true
    - apparmor:docker-default
  read_only: false  # Jellyfin needs write access
  tmpfs:
    - /tmp:noexec,nosuid,size=1g
    - /var/tmp:noexec,nosuid,size=1g
  
  # Resource limits
  deploy:
    resources:
      limits:
        memory: 4G
        cpus: '2.0'
      reservations:
        memory: 1G
        cpus: '0.5'
  
  # User mapping
  user: "${PUID}:${PGID}"
  
  # Network isolation
  networks:
    - media_network
  
  # Minimal capabilities
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - DAC_OVERRIDE
    - SETGID
    - SETUID
```

### **Docker Security Scanning**

```bash
# Install Docker Scout
curl -sSfL https://raw.githubusercontent.com/docker/scout-cli/main/install.sh | sh -s --

# Scan images for vulnerabilities
docker scout cves jellyfin/jellyfin:latest
docker scout cves lscr.io/linuxserver/sonarr:latest

# Automated scanning script
cat > scan-images.sh << 'EOF'
#!/bin/bash
images=(
  "jellyfin/jellyfin:latest"
  "lscr.io/linuxserver/sonarr:latest"
  "lscr.io/linuxserver/radarr:latest"
  "lscr.io/linuxserver/prowlarr:latest"
  "lscr.io/linuxserver/qbittorrent:latest"
  "traefik:v3.0"
)

for image in "${images[@]}"; do
  echo "Scanning $image..."
  docker scout cves "$image"
done
EOF
chmod +x scan-images.sh
```

## 📊 Monitoring & Intrusion Detection

### **Fail2Ban Setup**

```bash
# Install Fail2Ban
sudo apt install fail2ban

# Configure Fail2Ban for SSH and web services
sudo tee /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = 22
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[traefik-auth]
enabled = true
filter = traefik-auth
logpath = /var/log/traefik/access.log
port = http,https
maxretry = 5
bantime = 86400

[jellyfin]
enabled = true
filter = jellyfin
logpath = /path/to/jellyfin/logs/*.log
port = http,https
maxretry = 5
EOF

# Create Traefik filter
sudo tee /etc/fail2ban/filter.d/traefik-auth.conf << EOF
[Definition]
failregex = ^<HOST> \- \S+ \[\] "(GET|POST|HEAD)" [\d]+ (401|403) \d+ ".*?" ".*?" \d+ms$
ignoreregex =
EOF

sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### **Log Monitoring with Loki and Grafana** (Optional)

```yaml
# Add to docker-compose-2025-enhanced.yml
  loki:
    image: grafana/loki:latest
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./config/loki:/etc/loki
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - media_network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3001:3000"
    volumes:
      - ./config/grafana:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
    networks:
      - media_network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.grafana.rule=Host(`monitoring.${DOMAIN}`)"
      - "traefik.http.routers.grafana.entrypoints=websecure"
      - "traefik.http.routers.grafana.tls.certresolver=cloudflare"
      - "traefik.http.routers.grafana.middlewares=auth"

  promtail:
    image: grafana/promtail:latest
    container_name: promtail
    volumes:
      - ./config/promtail:/etc/promtail
      - /var/log:/var/log:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command: -config.file=/etc/promtail/config.yml
    networks:
      - media_network
```

## 🔍 Security Auditing

### **Regular Security Checks**

```bash
#!/bin/bash
# security-audit.sh

echo "🔒 Media Server Security Audit - $(date)"
echo "=========================================="

# Check for rootkits
echo "1. Checking for rootkits..."
sudo rkhunter --update
sudo rkhunter --check --sk

# Check for malware
echo "2. Scanning for malware..."
sudo clamscan -r /home --infected --log=/tmp/clamscan.log

# Check open ports
echo "3. Checking open ports..."
sudo netstat -tulpn | grep LISTEN

# Check failed logins
echo "4. Recent failed logins..."
sudo journalctl -u ssh -n 50 | grep "Failed password"

# Check Docker security
echo "5. Docker security status..."
docker run --rm --net host --pid host --userns host --cap-add audit_control \
  -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
  -v /etc:/etc:ro \
  -v /usr/bin/containerd:/usr/bin/containerd:ro \
  -v /usr/bin/runc:/usr/bin/runc:ro \
  -v /usr/lib/systemd:/usr/lib/systemd:ro \
  -v /var/lib:/var/lib:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  --label docker_bench_security \
  docker/docker-bench-security

# Check SSL certificates
echo "6. SSL certificate status..."
for domain in jellyfin sonarr radarr prowlarr; do
  echo "Checking $domain.yourdomain.com..."
  openssl s_client -connect $domain.yourdomain.com:443 -servername $domain.yourdomain.com < /dev/null 2>/dev/null | openssl x509 -noout -dates
done

echo "Audit complete. Check /tmp/clamscan.log for malware scan results."
```

### **Automated Security Updates**

```bash
# Setup unattended upgrades
sudo apt install unattended-upgrades

# Configure automatic security updates
sudo tee /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF

# Configure what gets updated
sudo tee /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id} ESMApps:\${distro_codename}-apps-security";
    "\${distro_id} ESM:\${distro_codename}-infra-security";
};

Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
Unattended-Upgrade::Mail "your-email@example.com";
EOF
```

## 🚨 Incident Response

### **Security Incident Checklist**

**If you suspect a security breach:**

1. **Immediate Actions**
   ```bash
   # Stop all services
   docker compose -f docker-compose-2025-enhanced.yml down
   
   # Block suspicious IPs
   sudo ufw insert 1 deny from SUSPICIOUS_IP
   
   # Change all passwords
   ```

2. **Investigation**
   ```bash
   # Check access logs
   docker logs traefik | grep "unusual_pattern"
   
   # Check container integrity
   docker diff container_name
   
   # Review authentication logs
   journalctl -u ssh -f
   ```

3. **Recovery**
   ```bash
   # Restore from backup
   tar -xzf backup-YYYYMMDD.tar.gz
   
   # Update all images
   docker compose pull
   
   # Restart with clean state
   docker compose up -d
   ```

### **Backup Strategy for Security**

```bash
#!/bin/bash
# secure-backup.sh

BACKUP_DIR="/encrypted-backup"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="media-server-$DATE.tar.gz"

# Create encrypted backup
tar -czf - ./config ./data/*.db .env | \
gpg --symmetric --cipher-algo AES256 --output "$BACKUP_DIR/$BACKUP_FILE.gpg"

# Upload to secure cloud storage
rclone copy "$BACKUP_DIR/$BACKUP_FILE.gpg" remote:backups/

# Keep only 30 days of backups
find "$BACKUP_DIR" -name "*.gpg" -mtime +30 -delete

echo "Secure backup created: $BACKUP_FILE.gpg"
```

## 📋 Security Maintenance Schedule

### **Daily**
- Monitor failed authentication attempts
- Check Cloudflare security events
- Review container resource usage

### **Weekly**
- Update Docker images
- Review access logs
- Check SSL certificate status
- Run malware scan

### **Monthly**
- Security audit script
- Update all system packages
- Review and rotate secrets
- Test backup restoration

### **Quarterly**
- Penetration testing
- Review and update firewall rules
- Security configuration review
- Update incident response plan

## 🔧 Security Tools Installation

```bash
#!/bin/bash
# install-security-tools.sh

# ClamAV for malware scanning
sudo apt install clamav clamav-daemon
sudo freshclam

# RKHunter for rootkit detection
sudo apt install rkhunter
sudo rkhunter --update

# Lynis for security auditing
sudo apt install lynis

# Chkrootkit for additional rootkit detection
sudo apt install chkrootkit

# AIDE for file integrity monitoring
sudo apt install aide
sudo aideinit
sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

echo "Security tools installed successfully"
```

## ⚠️ Security Compliance

### **Data Protection**
- Ensure compliance with local data protection laws
- Implement data retention policies
- Regular data backup and encryption
- Access logging and audit trails

### **Network Security Standards**
- TLS 1.2+ for all communications
- Strong cipher suites only
- Regular certificate rotation
- Network segmentation

### **Access Control**
- Principle of least privilege
- Regular access reviews
- Strong authentication requirements
- Session management

---

## 📞 Emergency Contacts

**In case of security incident:**
1. Change all passwords immediately
2. Isolate affected systems
3. Contact your ISP if needed
4. Document everything
5. Report to authorities if required

**Security Resources:**
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Cloudflare Security Center](https://www.cloudflare.com/security-center/)

---

**Last Updated**: 2025-07-27  
**Security Version**: 2025.1  
**Review Schedule**: Quarterly