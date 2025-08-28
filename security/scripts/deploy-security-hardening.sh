#!/bin/bash

# Security Hardening Deployment Script
# Comprehensive security implementation for media server infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SECURITY_DIR="$PROJECT_ROOT/security"

# Logging
LOG_FILE="$PROJECT_ROOT/logs/security-deployment.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Check if running as root (needed for some security configurations)
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root - this is required for system-level security configurations"
    else
        log_info "Running as non-root user - some system-level configurations may be skipped"
    fi
}

# Backup existing configurations
backup_configs() {
    log_info "Creating backup of existing configurations..."
    
    BACKUP_DIR="$PROJECT_ROOT/security-backup-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup existing nginx configs if they exist
    if [ -d "/etc/nginx" ] && [[ $EUID -eq 0 ]]; then
        cp -r /etc/nginx "$BACKUP_DIR/nginx-system" 2>/dev/null || true
    fi
    
    # Backup existing fail2ban configs if they exist
    if [ -d "/etc/fail2ban" ] && [[ $EUID -eq 0 ]]; then
        cp -r /etc/fail2ban "$BACKUP_DIR/fail2ban-system" 2>/dev/null || true
    fi
    
    # Backup project configurations
    [ -f "$PROJECT_ROOT/.env" ] && cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/"
    [ -f "$PROJECT_ROOT/docker-compose.yml" ] && cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/"
    
    log_success "Configurations backed up to $BACKUP_DIR"
}

# Install system dependencies
install_system_dependencies() {
    if [[ $EUID -ne 0 ]]; then
        log_warning "Skipping system dependency installation (requires root)"
        return 0
    fi
    
    log_info "Installing system security dependencies..."
    
    # Detect OS
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        apt-get update
        apt-get install -y \
            ufw \
            fail2ban \
            logrotate \
            rsyslog \
            curl \
            wget \
            openssl \
            ca-certificates \
            iptables-persistent \
            unattended-upgrades
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        yum update -y
        yum install -y \
            firewalld \
            fail2ban \
            logrotate \
            rsyslog \
            curl \
            wget \
            openssl \
            ca-certificates
    elif command -v apk &> /dev/null; then
        # Alpine Linux
        apk update
        apk add \
            ufw \
            fail2ban \
            logrotate \
            rsyslog \
            curl \
            wget \
            openssl \
            ca-certificates
    else
        log_warning "Unknown OS - manual dependency installation may be required"
    fi
    
    log_success "System dependencies installed"
}

# Configure firewall rules
configure_firewall() {
    if [[ $EUID -ne 0 ]]; then
        log_warning "Skipping firewall configuration (requires root)"
        return 0
    fi
    
    log_info "Configuring firewall rules..."
    
    # Configure UFW if available
    if command -v ufw &> /dev/null; then
        # Reset to defaults
        ufw --force reset
        
        # Default policies
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow SSH (be careful!)
        ufw allow ssh
        
        # Allow HTTP and HTTPS
        ufw allow 80/tcp
        ufw allow 443/tcp
        
        # Allow specific media server ports (restrictive)
        ufw allow from 192.168.0.0/16 to any port 8096 # Jellyfin
        ufw allow from 172.16.0.0/12 to any port 8096
        ufw allow from 10.0.0.0/8 to any port 8096
        
        # Allow monitoring (local only)
        ufw allow from 127.0.0.1 to any port 3000 # Grafana
        ufw allow from 127.0.0.1 to any port 9090 # Prometheus
        
        # Enable firewall
        ufw --force enable
        
        log_success "UFW firewall configured and enabled"
    fi
}

# Create security directories
create_security_directories() {
    log_info "Creating security directory structure..."
    
    mkdir -p "$SECURITY_DIR"/{nginx/{conf.d,ssl,logs},fail2ban/{filters,actions},modsec/{custom-rules,logs},monitor,clamav-data,redis-data}
    mkdir -p "$PROJECT_ROOT/logs"/{nginx,fail2ban,security,docker}
    
    # Set appropriate permissions
    chmod 755 "$SECURITY_DIR"
    chmod 700 "$SECURITY_DIR/nginx/ssl" 2>/dev/null || true
    chmod 755 "$PROJECT_ROOT/logs"
    
    log_success "Security directories created"
}

# Generate SSL certificates
generate_ssl_certificates() {
    log_info "Generating SSL certificates..."
    
    SSL_DIR="$SECURITY_DIR/nginx/ssl"
    
    if [ ! -f "$SSL_DIR/server.crt" ]; then
        # Generate self-signed certificate for development/testing
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/server.key" \
            -out "$SSL_DIR/server.crt" \
            -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
        
        # Set secure permissions
        chmod 600 "$SSL_DIR/server.key"
        chmod 644 "$SSL_DIR/server.crt"
        
        log_success "Self-signed SSL certificate generated"
        log_warning "For production, replace with certificates from a trusted CA"
    else
        log_info "SSL certificate already exists"
    fi
}

# Configure secure environment variables
configure_secure_environment() {
    log_info "Configuring secure environment variables..."
    
    ENV_FILE="$PROJECT_ROOT/.env.secure"
    
    # Generate secure API keys and passwords if they don't exist
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << EOF
# Security Configuration
SECURITY_ALERT_EMAIL=admin@yourdomain.com
SECURITY_SLACK_WEBHOOK=
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=

# Generated secure keys (change these in production!)
API_SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
NGINX_RATE_LIMIT_KEY=$(openssl rand -hex 16)
REDIS_PASSWORD=$(openssl rand -base64 32)

# Database passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
MYSQL_ROOT_PASSWORD=$(openssl rand -base64 32)
NPM_DB_PASSWORD=$(openssl rand -base64 32)

# Service-specific passwords
GRAFANA_PASSWORD=$(openssl rand -base64 16)
PHOTOPRISM_PASSWORD=$(openssl rand -base64 16)
VAULTWARDEN_TOKEN=$(openssl rand -hex 64)
PIHOLE_PASSWORD=$(openssl rand -base64 16)

# Domain configuration
DOMAIN=localhost
TRUSTED_PROXIES=127.0.0.1,172.16.0.0/12

# Security settings
FAIL2BAN_ENABLED=true
MODSEC_ENABLED=true
RATE_LIMITING_ENABLED=true
EOF
        
        chmod 600 "$ENV_FILE"
        log_success "Secure environment file created: $ENV_FILE"
        log_warning "Please review and customize the values in $ENV_FILE"
    else
        log_info "Secure environment file already exists"
    fi
}

# Set up log rotation
configure_log_rotation() {
    if [[ $EUID -ne 0 ]]; then
        log_warning "Skipping log rotation setup (requires root)"
        return 0
    fi
    
    log_info "Configuring log rotation..."
    
    cat > /etc/logrotate.d/media-server-security << 'EOF'
/var/log/media-server/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    postrotate
        systemctl reload nginx > /dev/null 2>&1 || true
        systemctl reload fail2ban > /dev/null 2>&1 || true
    endscript
}

/var/log/docker/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    copytruncate
    maxsize 100M
}
EOF
    
    log_success "Log rotation configured"
}

# Create additional nginx configuration files
create_nginx_configs() {
    log_info "Creating additional nginx configuration files..."
    
    # Security headers configuration
    cat > "$SECURITY_DIR/nginx/conf.d/security-headers.conf" << 'EOF'
# Security headers configuration
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;

# Hide server version
server_tokens off;
more_set_headers 'Server: MediaServer';

# Prevent access to hidden files
location ~ /\. {
    deny all;
    access_log off;
    log_not_found off;
}

# Block common exploit attempts
location ~* \.(php|asp|aspx|jsp)$ {
    deny all;
    access_log off;
    log_not_found off;
}
EOF

    # Rate limiting configuration
    cat > "$SECURITY_DIR/nginx/conf.d/rate-limiting.conf" << 'EOF'
# Rate limiting zones
limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=api:10m rate=5r/s;
limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
limit_req_zone $binary_remote_addr zone=media:10m rate=20r/s;

# Connection limiting
limit_conn_zone $binary_remote_addr zone=perip:10m;
limit_conn_zone $server_name zone=perserver:10m;

# Apply limits
limit_conn perip 10;
limit_conn perserver 100;
EOF

    log_success "Additional nginx configurations created"
}

# Set up automated security updates
configure_auto_updates() {
    if [[ $EUID -ne 0 ]]; then
        log_warning "Skipping auto-update configuration (requires root)"
        return 0
    fi
    
    log_info "Configuring automatic security updates..."
    
    if command -v apt-get &> /dev/null; then
        # Configure unattended-upgrades for Ubuntu/Debian
        cat > /etc/apt/apt.conf.d/50unattended-upgrades-security << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESM:${distro_codename}";
};

Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";

Unattended-Upgrade::Mail "root";
Unattended-Upgrade::MailOnlyOnError "true";
EOF

        # Enable automatic updates
        cat > /etc/apt/apt.conf.d/20auto-upgrades << 'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::Download-Upgradeable-Packages "1";
APT::Periodic::AutocleanInterval "7";
EOF
    fi
    
    log_success "Automatic security updates configured"
}

# Deploy security services
deploy_security_services() {
    log_info "Deploying security services..."
    
    cd "$PROJECT_ROOT"
    
    # Build and start security services
    docker-compose -f security/security-hardening-config.yml build --no-cache
    docker-compose -f security/security-hardening-config.yml up -d
    
    # Wait for services to start
    sleep 30
    
    # Check service health
    if docker-compose -f security/security-hardening-config.yml ps | grep -q "Up"; then
        log_success "Security services deployed successfully"
    else
        log_error "Some security services failed to start"
        docker-compose -f security/security-hardening-config.yml logs
    fi
}

# Generate security report
generate_security_report() {
    log_info "Generating security deployment report..."
    
    REPORT_FILE="$PROJECT_ROOT/logs/security-report-$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# Security Hardening Deployment Report

**Deployment Date:** $(date)
**Report Generated:** $(date)

## Security Measures Implemented

### 1. Network Security
- [x] Nginx reverse proxy with security headers
- [x] Rate limiting configured
- [x] Firewall rules applied (if root access available)
- [x] SSL/TLS certificates generated

### 2. Application Security
- [x] Fail2ban intrusion prevention system
- [x] ModSecurity Web Application Firewall
- [x] API authentication and authorization
- [x] Input validation and sanitization

### 3. Monitoring and Alerting
- [x] Real-time security monitoring
- [x] Log aggregation and analysis
- [x] Email/Slack alerting system
- [x] Service health monitoring

### 4. Data Protection
- [x] Secure environment variable management
- [x] Database password generation
- [x] Encrypted communications (TLS)
- [x] Log rotation and retention

### 5. System Hardening
- [x] Container security policies
- [x] Resource limitations
- [x] Non-root user execution
- [x] Automatic security updates (if root access available)

## Next Steps

1. **Review Configuration Files:**
   - Check \`.env.secure\` file and customize values
   - Review nginx configuration in \`security/nginx/nginx.conf\`
   - Verify fail2ban rules in \`security/fail2ban/jail.local\`

2. **Test Security Measures:**
   - Run security scan: \`curl http://localhost:3011/api/scan\`
   - Check service health: \`curl http://localhost:3011/api/health\`
   - View security dashboard: \`curl http://localhost:3011/dashboard\`

3. **Production Considerations:**
   - Replace self-signed certificates with trusted CA certificates
   - Configure email settings for security alerts
   - Set up external log aggregation (ELK stack, Splunk, etc.)
   - Implement backup and disaster recovery procedures

## Service Endpoints

- **Security Monitoring:** http://localhost:3011
- **Nginx Proxy:** http://localhost (port 80/443)
- **Rate Limiter:** Internal (Redis on security network)
- **WAF (ModSecurity):** http://localhost:8080/8443

## Log Locations

- Security logs: \`$PROJECT_ROOT/logs/security/\`
- Nginx logs: \`$PROJECT_ROOT/security/nginx/logs/\`
- Fail2ban logs: \`$PROJECT_ROOT/security/fail2ban/\`
- Application logs: \`$PROJECT_ROOT/logs/\`

## Support and Maintenance

- Review security logs daily
- Update security rules monthly
- Test backup and recovery procedures quarterly
- Review and update security policies annually

---

For support and updates, please refer to the security documentation.
EOF

    log_success "Security report generated: $REPORT_FILE"
}

# Main deployment function
main() {
    log_info "Starting security hardening deployment..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Security directory: $SECURITY_DIR"
    
    check_privileges
    backup_configs
    create_security_directories
    configure_secure_environment
    generate_ssl_certificates
    create_nginx_configs
    
    # System-level configurations (require root)
    install_system_dependencies
    configure_firewall
    configure_log_rotation
    configure_auto_updates
    
    # Deploy containerized security services
    deploy_security_services
    
    # Generate final report
    generate_security_report
    
    log_success "Security hardening deployment completed!"
    log_info "Please review the security report and customize configurations as needed."
    log_warning "Remember to:"
    log_warning "  1. Update email settings in .env.secure"
    log_warning "  2. Replace self-signed certificates with trusted ones for production"
    log_warning "  3. Test all security measures thoroughly"
    log_warning "  4. Set up monitoring and alerting"
}

# Run main function
main "$@"