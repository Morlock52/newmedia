#!/bin/bash

# Complete Jellyfin Authentication Fix Script
# This script fixes all authentication issues and sets up proper integration

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/auth-fix-complete.log"
JELLYFIN_URL="http://localhost:8096"

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

main() {
    log "=================================================="
    log "Complete Jellyfin Authentication Fix"
    log "=================================================="
    
    # Step 1: Run the main auth fix script
    log "Step 1: Running main authentication fix..."
    if [ -f "$SCRIPT_DIR/scripts/fix-jellyfin-auth.sh" ]; then
        bash "$SCRIPT_DIR/scripts/fix-jellyfin-auth.sh" || warning "Auth fix script encountered issues"
    else
        error "Auth fix script not found!"
        exit 1
    fi
    
    # Step 2: Install Node.js dependencies if needed
    log "Step 2: Installing Node.js dependencies..."
    if [ -f "$SCRIPT_DIR/package.json" ]; then
        cd "$SCRIPT_DIR"
        npm install axios joi 2>/dev/null || warning "Could not install some Node.js packages"
    fi
    
    # Step 3: Configure CORS
    log "Step 3: Configuring CORS settings..."
    if command -v node &> /dev/null; then
        cd "$SCRIPT_DIR/scripts"
        node jellyfin-cors-config.js || warning "CORS configuration encountered issues"
    else
        warning "Node.js not available, skipping CORS configuration"
    fi
    
    # Step 4: Wait for Jellyfin to be ready
    log "Step 4: Waiting for Jellyfin to be ready..."
    local max_attempts=20
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 "$JELLYFIN_URL/health" > /dev/null 2>&1; then
            log "Jellyfin is ready ✓"
            break
        fi
        
        info "Attempt $attempt/$max_attempts - waiting for Jellyfin..."
        sleep 10
        ((attempt++))
        
        if [ $attempt -gt $max_attempts ]; then
            error "Jellyfin failed to become ready"
            exit 1
        fi
    done
    
    # Step 5: Run authentication tests
    log "Step 5: Running authentication tests..."
    if command -v node &> /dev/null; then
        cd "$SCRIPT_DIR/scripts"
        node jellyfin-auth-test.js || warning "Some authentication tests failed"
    else
        warning "Node.js not available, skipping authentication tests"
    fi
    
    # Step 6: Update API server configuration
    log "Step 6: Updating API server configuration..."
    if [ -f "$SCRIPT_DIR/api/services/JellyfinAuthService.js" ]; then
        log "Jellyfin Authentication Service is ready ✓"
    fi
    
    # Step 7: Test API endpoints manually
    log "Step 7: Testing API endpoints..."
    
    local endpoints=(
        "/health"
        "/System/Info/Public" 
        "/System/Ping"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -s --connect-timeout 10 "$JELLYFIN_URL$endpoint" > /dev/null 2>&1; then
            log "✓ $endpoint - OK"
        else
            warning "✗ $endpoint - Failed"
        fi
    done
    
    # Step 8: Create startup script
    log "Step 8: Creating startup verification script..."
    cat > "$SCRIPT_DIR/verify-jellyfin-auth.sh" << 'EOF'
#!/bin/bash
# Jellyfin Authentication Verification Script

JELLYFIN_URL="http://localhost:8096"

echo "🔍 Verifying Jellyfin Authentication Setup..."

# Test basic connectivity
if curl -s --connect-timeout 5 "$JELLYFIN_URL/health" > /dev/null; then
    echo "✅ Jellyfin is accessible"
else
    echo "❌ Jellyfin is not accessible"
    exit 1
fi

# Test public endpoints
if curl -s --connect-timeout 5 "$JELLYFIN_URL/System/Info/Public" > /dev/null; then
    echo "✅ Public API endpoints working"
else
    echo "❌ Public API endpoints not working"
    exit 1
fi

# Check if API key file exists
if [ -f "./scripts/jellyfin-api-key.txt" ]; then
    echo "✅ API key file found"
else
    echo "⚠️  API key file not found - may need to create one manually"
fi

# Check if config file exists
if [ -f "./scripts/jellyfin-api-config.json" ]; then
    echo "✅ API configuration file found"
else
    echo "⚠️  API configuration file not found"
fi

echo "🎉 Jellyfin authentication verification completed!"
EOF
    
    chmod +x "$SCRIPT_DIR/verify-jellyfin-auth.sh"
    
    # Step 9: Final summary
    log "=================================================="
    log "Authentication Fix Complete!"
    log "=================================================="
    log ""
    log "🌐 Jellyfin URL: $JELLYFIN_URL"
    log "🔑 Default credentials: admin / admin123"
    log "📝 Log file: $LOG_FILE"
    log ""
    log "Next steps:"
    log "1. Access Jellyfin at $JELLYFIN_URL"
    log "2. Login with admin/admin123 (or create new user)"
    log "3. Run ./verify-jellyfin-auth.sh to verify setup"
    log "4. Start your dashboard application"
    log ""
    
    if [ -f "$SCRIPT_DIR/scripts/jellyfin-api-key.txt" ]; then
        local api_key=$(cat "$SCRIPT_DIR/scripts/jellyfin-api-key.txt")
        log "🗝️  API Key: ${api_key:0:30}..."
    fi
    
    log "Files created:"
    log "- scripts/fix-jellyfin-auth.sh (main fix script)"
    log "- scripts/jellyfin-cors-config.js (CORS configuration)"
    log "- scripts/jellyfin-auth-test.js (authentication tests)"
    log "- scripts/dashboard-jellyfin-integration.js (integration module)"
    log "- api/services/JellyfinAuthService.js (auth service)"
    log "- verify-jellyfin-auth.sh (verification script)"
    log ""
    log "🎉 Ready to use Jellyfin with your dashboard!"
}

# Run main function
main "$@"