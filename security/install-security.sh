#!/bin/bash

# Security Installation Script
# Fixes: Automated deployment of security measures
# Author: Security Manager Agent
# Date: 2025-08-03

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECURITY_DIR="${PROJECT_ROOT}/security"
SECRETS_DIR="${SECURITY_DIR}/secrets"
LOGS_DIR="${SECURITY_DIR}/logs"
BACKUP_DIR="${SECURITY_DIR}/backups"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if running as root (should not be)
check_user() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
    fi
    log "✅ User check passed"
}

# Create secure directory structure
create_directories() {
    log "Creating secure directory structure..."
    
    # Create directories with proper permissions
    mkdir -p "${SECRETS_DIR}" && chmod 700 "${SECRETS_DIR}"
    mkdir -p "${LOGS_DIR}" && chmod 750 "${LOGS_DIR}"
    mkdir -p "${BACKUP_DIR}" && chmod 750 "${BACKUP_DIR}"
    mkdir -p "${SECURITY_DIR}/ssl" && chmod 750 "${SECURITY_DIR}/ssl"
    mkdir -p "${SECURITY_DIR}/config" && chmod 750 "${SECURITY_DIR}/config"
    
    log "✅ Directory structure created"
}

# Install Node.js dependencies
install_dependencies() {
    log "Installing Node.js security dependencies..."
    
    cd "${PROJECT_ROOT}"
    
    # Check if package.json exists
    if [[ ! -f "package.json" ]]; then
        log "Creating package.json..."
        cat > package.json << 'EOF'
{
  "name": "media-server-security",
  "version": "1.0.0",
  "description": "Secure media server with comprehensive security measures",
  "main": "index.js",
  "scripts": {
    "security:test": "node security/test-security.js",
    "security:migrate": "node security/secure-env-manager.js migrate .env",
    "security:generate": "node security/secrets-manager.js generate",
    "security:backup": "node security/secrets-manager.js backup"
  },
  "dependencies": {},
  "devDependencies": {}
}
EOF
    fi
    
    # Install required packages
    local packages=(
        "express"
        "helmet"
        "express-rate-limit"
        "bcrypt"
        "jsonwebtoken"
        "joi"
        "xss"
        "validator"
        "isomorphic-dompurify"
        "redis"
        "crypto"
    )
    
    if command -v npm &> /dev/null; then
        npm install --save "${packages[@]}"
        log "✅ Node.js dependencies installed"
    else
        warn "npm not found, please install Node.js dependencies manually"
    fi
}

# Set up file permissions
secure_permissions() {
    log "Setting secure file permissions..."
    
    # Security scripts executable
    chmod +x "${SECURITY_DIR}"/*.js 2>/dev/null || true
    chmod +x "${SECURITY_DIR}"/*.sh 2>/dev/null || true
    
    # Secure sensitive files
    find "${SECURITY_DIR}" -name "*.key" -exec chmod 600 {} \; 2>/dev/null || true
    find "${SECURITY_DIR}" -name "*.pem" -exec chmod 600 {} \; 2>/dev/null || true
    find "${SECURITY_DIR}" -name "*.p12" -exec chmod 600 {} \; 2>/dev/null || true
    
    # Configuration files
    find "${SECURITY_DIR}/config" -type f -exec chmod 640 {} \; 2>/dev/null || true
    
    # Log files
    find "${LOGS_DIR}" -type f -exec chmod 640 {} \; 2>/dev/null || true
    
    log "✅ File permissions secured"
}

# Generate SSL certificates for development
generate_ssl_certs() {
    log "Generating SSL certificates for development..."
    
    SSL_DIR="${SECURITY_DIR}/ssl"
    
    if [[ ! -f "${SSL_DIR}/server.key" ]]; then
        # Generate private key
        openssl genrsa -out "${SSL_DIR}/server.key" 2048
        chmod 600 "${SSL_DIR}/server.key"
        
        # Generate certificate signing request
        openssl req -new -key "${SSL_DIR}/server.key" -out "${SSL_DIR}/server.csr" -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        # Generate self-signed certificate
        openssl x509 -req -days 365 -in "${SSL_DIR}/server.csr" -signkey "${SSL_DIR}/server.key" -out "${SSL_DIR}/server.crt"
        chmod 644 "${SSL_DIR}/server.crt"
        
        # Generate combined PEM file
        cat "${SSL_DIR}/server.crt" "${SSL_DIR}/server.key" > "${SSL_DIR}/server.pem"
        chmod 600 "${SSL_DIR}/server.pem"
        
        # Clean up CSR
        rm "${SSL_DIR}/server.csr"
        
        log "✅ SSL certificates generated"
    else
        log "SSL certificates already exist"
    fi
}

# Initialize secrets management
initialize_secrets() {
    log "Initializing secrets management..."
    
    if [[ ! -f "${SECRETS_DIR}/vault.encrypted" ]]; then
        # Create initial secrets vault (will prompt for master password)
        node "${SECURITY_DIR}/secrets-manager.js" store INIT_SECRET "$(openssl rand -hex 32)" general
        log "✅ Secrets vault initialized"
    else
        log "Secrets vault already exists"
    fi
}

# Migrate existing environment files
migrate_environment() {
    log "Migrating existing environment files..."
    
    # Find and migrate .env files
    local env_files
    env_files=$(find "${PROJECT_ROOT}" -maxdepth 2 -name ".env" -not -path "*/security/*" 2>/dev/null || true)
    
    for env_file in $env_files; do
        if [[ -f "$env_file" && -s "$env_file" ]]; then
            log "Migrating $env_file..."
            node "${SECURITY_DIR}/secure-env-manager.js" migrate "$env_file"
        fi
    done
    
    log "✅ Environment files migrated"
}

# Create secure Docker configuration
create_docker_config() {
    log "Creating secure Docker configuration..."
    
    # Create secrets directory for Docker
    mkdir -p "${SECRETS_DIR}/docker" && chmod 700 "${SECRETS_DIR}/docker"
    
    # Export secrets for Docker Compose
    if [[ -f "${SECRETS_DIR}/vault.encrypted" ]]; then
        node "${SECURITY_DIR}/secrets-manager.js" export "${SECRETS_DIR}/docker"
    fi
    
    # Copy secure Docker compose file
    if [[ -f "${SECURITY_DIR}/docker-security-config.yml" ]]; then
        cp "${SECURITY_DIR}/docker-security-config.yml" "${PROJECT_ROOT}/docker-compose.secure.yml"
        log "✅ Secure Docker configuration created"
    fi
}

# Set up security monitoring
setup_monitoring() {
    log "Setting up security monitoring..."
    
    # Create monitoring configuration
    cat > "${SECURITY_DIR}/config/monitoring.json" << EOF
{
  "enabled": true,
  "logLevel": "INFO",
  "alertThresholds": {
    "failedLoginAttempts": 5,
    "rateLimitViolations": 10,
    "maliciousRequestsPerMinute": 20,
    "privilegeEscalationAttempts": 1
  },
  "retention": {
    "logRetentionDays": 30,
    "metricsRetentionDays": 7
  },
  "notifications": {
    "email": {
      "enabled": false,
      "smtp": {
        "host": "",
        "port": 587,
        "secure": false,
        "auth": {
          "user": "",
          "pass": ""
        }
      }
    },
    "webhook": {
      "enabled": false,
      "url": ""
    }
  }
}
EOF
    
    chmod 640 "${SECURITY_DIR}/config/monitoring.json"
    log "✅ Security monitoring configured"
}

# Create systemd service for security monitoring (Linux only)
create_systemd_service() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log "Creating systemd service for security monitoring..."
        
        cat > "/tmp/media-server-security.service" << EOF
[Unit]
Description=Media Server Security Monitor
After=network.target
Wants=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${PROJECT_ROOT}
ExecStart=/usr/bin/node ${SECURITY_DIR}/security-monitor.js
Restart=always
RestartSec=10
Environment=NODE_ENV=production

[Install]
WantedBy=multi-user.target
EOF
        
        log "Systemd service file created at /tmp/media-server-security.service"
        log "To install: sudo mv /tmp/media-server-security.service /etc/systemd/system/ && sudo systemctl enable media-server-security"
    fi
}

# Create security test script
create_test_script() {
    log "Creating security test script..."
    
    cat > "${SECURITY_DIR}/test-security.js" << 'EOF'
#!/usr/bin/env node

/**
 * Security Test Suite
 */

const fs = require('fs');
const path = require('path');

async function runSecurityTests() {
    console.log('🔒 Running security tests...\n');
    
    let passed = 0;
    let failed = 0;
    
    // Test 1: Check file permissions
    try {
        const secretsDir = './security/secrets';
        const stats = fs.statSync(secretsDir);
        const mode = (stats.mode & parseInt('777', 8)).toString(8);
        
        if (mode === '700') {
            console.log('✅ Test 1: Secrets directory permissions correct (700)');
            passed++;
        } else {
            console.log(`❌ Test 1: Secrets directory permissions incorrect (${mode}), should be 700`);
            failed++;
        }
    } catch (error) {
        console.log('❌ Test 1: Secrets directory not found');
        failed++;
    }
    
    // Test 2: Check SSL certificates
    try {
        const sslCert = './security/ssl/server.crt';
        if (fs.existsSync(sslCert)) {
            console.log('✅ Test 2: SSL certificate exists');
            passed++;
        } else {
            console.log('❌ Test 2: SSL certificate not found');
            failed++;
        }
    } catch (error) {
        console.log('❌ Test 2: SSL certificate check failed');
        failed++;
    }
    
    // Test 3: Check secrets vault
    try {
        const vaultFile = './security/secrets/vault.encrypted';
        if (fs.existsSync(vaultFile)) {
            console.log('✅ Test 3: Secrets vault exists');
            passed++;
        } else {
            console.log('❌ Test 3: Secrets vault not found');
            failed++;
        }
    } catch (error) {
        console.log('❌ Test 3: Secrets vault check failed');
        failed++;
    }
    
    // Test 4: Check Docker secrets
    try {
        const dockerSecretsDir = './security/secrets/docker';
        if (fs.existsSync(dockerSecretsDir)) {
            const files = fs.readdirSync(dockerSecretsDir);
            if (files.length > 0) {
                console.log(`✅ Test 4: Docker secrets exported (${files.length} files)`);
                passed++;
            } else {
                console.log('❌ Test 4: No Docker secrets found');
                failed++;
            }
        } else {
            console.log('❌ Test 4: Docker secrets directory not found');
            failed++;
        }
    } catch (error) {
        console.log('❌ Test 4: Docker secrets check failed');
        failed++;
    }
    
    console.log(`\n📊 Test Results: ${passed} passed, ${failed} failed`);
    
    if (failed === 0) {
        console.log('🎉 All security tests passed!');
        process.exit(0);
    } else {
        console.log('⚠️  Some security tests failed. Please review the configuration.');
        process.exit(1);
    }
}

runSecurityTests().catch(console.error);
EOF
    
    chmod +x "${SECURITY_DIR}/test-security.js"
    log "✅ Security test script created"
}

# Create backup script
create_backup_script() {
    log "Creating backup script..."
    
    cat > "${SECURITY_DIR}/backup-security.sh" << 'EOF'
#!/bin/bash

# Security Backup Script
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/security/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/security_backup_${TIMESTAMP}.tar.gz"

echo "🔒 Creating security backup..."

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Create backup
tar -czf "${BACKUP_FILE}" \
    --exclude="*.log" \
    --exclude="node_modules" \
    -C "${PROJECT_ROOT}" \
    security/

echo "✅ Security backup created: ${BACKUP_FILE}"

# Keep only last 10 backups
ls -t "${BACKUP_DIR}"/security_backup_*.tar.gz | tail -n +11 | xargs rm -f 2>/dev/null || true

echo "📊 Backup completed successfully"
EOF
    
    chmod +x "${SECURITY_DIR}/backup-security.sh"
    log "✅ Backup script created"
}

# Main installation function
main() {
    log "🔒 Starting security installation..."
    log "Project root: ${PROJECT_ROOT}"
    
    # Run installation steps
    check_user
    create_directories
    install_dependencies
    secure_permissions
    generate_ssl_certs
    # initialize_secrets  # Skip interactive step for now
    migrate_environment
    create_docker_config
    setup_monitoring
    create_systemd_service
    create_test_script
    create_backup_script
    
    log "🎉 Security installation completed!"
    log ""
    log "Next steps:"
    log "1. Initialize secrets vault: node security/secrets-manager.js store MASTER_KEY \$(openssl rand -hex 32)"
    log "2. Run security tests: npm run security:test"
    log "3. Start services with: docker-compose -f docker-compose.secure.yml up -d"
    log "4. Monitor security logs in: ${LOGS_DIR}"
    log ""
    log "⚠️  IMPORTANT: Keep your master password secure and create backups!"
}

# Run main function
main "$@"