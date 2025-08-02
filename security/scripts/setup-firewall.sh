#!/bin/bash
# Network segmentation and firewall setup for media server

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_DIR="$(dirname "$SCRIPT_DIR")"

# Network bridge names (from docker-compose)
DMZ_BRIDGE="br-dmz"
FRONTEND_BRIDGE="br-frontend"
BACKEND_BRIDGE="br-backend"
DOWNLOADS_BRIDGE="br-downloads"
MONITORING_BRIDGE="br-monitoring"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Save current iptables rules
backup_iptables() {
    local backup_file="/tmp/iptables-backup-$(date +%Y%m%d-%H%M%S)"
    log_info "Backing up current iptables rules to $backup_file"
    
    iptables-save > "$backup_file"
    log_success "Backup created: $backup_file"
}

# Function to check if Docker is using iptables
check_docker_iptables() {
    if ! iptables -L DOCKER-USER >/dev/null 2>&1; then
        log_error "DOCKER-USER chain not found. Is Docker running with iptables enabled?"
        exit 1
    fi
}

# Clear existing DOCKER-USER rules (preserving Docker's own rules)
clear_custom_rules() {
    log_info "Clearing existing custom firewall rules..."
    
    # Remove only our custom rules (marked with comments)
    while iptables -L DOCKER-USER --line-numbers | grep -q "media-server-fw"; do
        local line=$(iptables -L DOCKER-USER --line-numbers | grep "media-server-fw" | head -1 | awk '{print $1}')
        iptables -D DOCKER-USER "$line"
    done
    
    log_success "Custom rules cleared"
}

# Setup DMZ network rules (public-facing services)
setup_dmz_rules() {
    log_info "Setting up DMZ network rules..."
    
    # Allow only HTTP/HTTPS traffic to DMZ
    iptables -I DOCKER-USER -i "$DMZ_BRIDGE" -p tcp --dport 80 -j ACCEPT -m comment --comment "media-server-fw: DMZ HTTP"
    iptables -I DOCKER-USER -i "$DMZ_BRIDGE" -p tcp --dport 443 -j ACCEPT -m comment --comment "media-server-fw: DMZ HTTPS"
    
    # Allow established connections
    iptables -I DOCKER-USER -i "$DMZ_BRIDGE" -m state --state ESTABLISHED,RELATED -j ACCEPT -m comment --comment "media-server-fw: DMZ established"
    
    # Drop all other incoming traffic to DMZ
    iptables -A DOCKER-USER -i "$DMZ_BRIDGE" -j DROP -m comment --comment "media-server-fw: DMZ drop all"
    
    # Prevent DMZ from initiating connections to other networks
    iptables -I DOCKER-USER -o "$FRONTEND_BRIDGE" -i "$DMZ_BRIDGE" -j DROP -m comment --comment "media-server-fw: DMZ to frontend blocked"
    iptables -I DOCKER-USER -o "$BACKEND_BRIDGE" -i "$DMZ_BRIDGE" -j DROP -m comment --comment "media-server-fw: DMZ to backend blocked"
    iptables -I DOCKER-USER -o "$DOWNLOADS_BRIDGE" -i "$DMZ_BRIDGE" -j DROP -m comment --comment "media-server-fw: DMZ to downloads blocked"
    
    log_success "DMZ rules configured"
}

# Setup Frontend network rules (web interfaces)
setup_frontend_rules() {
    log_info "Setting up Frontend network rules..."
    
    # Allow specific service ports
    local frontend_ports=(
        "8096"   # Jellyfin
        "32400"  # Plex
        "8989"   # Sonarr
        "7878"   # Radarr
        "8686"   # Lidarr
        "9696"   # Prowlarr
        "6767"   # Bazarr
        "5055"   # Overseerr
        "8181"   # Tautulli
        "3000"   # Grafana
    )
    
    for port in "${frontend_ports[@]}"; do
        iptables -I DOCKER-USER -i "$FRONTEND_BRIDGE" -p tcp --dport "$port" -j ACCEPT -m comment --comment "media-server-fw: Frontend port $port"
    done
    
    # Allow frontend to communicate with backend
    iptables -I DOCKER-USER -i "$FRONTEND_BRIDGE" -o "$BACKEND_BRIDGE" -j ACCEPT -m comment --comment "media-server-fw: Frontend to backend"
    
    # Block frontend from downloads network
    iptables -I DOCKER-USER -i "$FRONTEND_BRIDGE" -o "$DOWNLOADS_BRIDGE" -j DROP -m comment --comment "media-server-fw: Frontend to downloads blocked"
    
    # Allow established connections
    iptables -I DOCKER-USER -i "$FRONTEND_BRIDGE" -m state --state ESTABLISHED,RELATED -j ACCEPT -m comment --comment "media-server-fw: Frontend established"
    
    # Drop all other traffic
    iptables -A DOCKER-USER -i "$FRONTEND_BRIDGE" -j DROP -m comment --comment "media-server-fw: Frontend drop all"
    
    log_success "Frontend rules configured"
}

# Setup Backend network rules (databases, cache)
setup_backend_rules() {
    log_info "Setting up Backend network rules..."
    
    # Allow only internal subnet access
    iptables -I DOCKER-USER -i "$BACKEND_BRIDGE" -s 172.30.0.0/16 -j ACCEPT -m comment --comment "media-server-fw: Backend internal only"
    
    # Block all external access
    iptables -A DOCKER-USER -i "$BACKEND_BRIDGE" -j DROP -m comment --comment "media-server-fw: Backend drop external"
    
    # Prevent backend from initiating external connections
    iptables -I DOCKER-USER -o "$DMZ_BRIDGE" -i "$BACKEND_BRIDGE" -j DROP -m comment --comment "media-server-fw: Backend to DMZ blocked"
    
    log_success "Backend rules configured"
}

# Setup Downloads network rules (isolated, VPN only)
setup_downloads_rules() {
    log_info "Setting up Downloads network rules..."
    
    # Allow only VPN tunnel traffic (tun0 interface)
    iptables -I DOCKER-USER -i "$DOWNLOADS_BRIDGE" -o tun0 -j ACCEPT -m comment --comment "media-server-fw: Downloads to VPN"
    iptables -I DOCKER-USER -i tun0 -o "$DOWNLOADS_BRIDGE" -j ACCEPT -m comment --comment "media-server-fw: VPN to downloads"
    
    # Allow DNS queries for VPN
    iptables -I DOCKER-USER -i "$DOWNLOADS_BRIDGE" -p udp --dport 53 -j ACCEPT -m comment --comment "media-server-fw: Downloads DNS"
    iptables -I DOCKER-USER -i "$DOWNLOADS_BRIDGE" -p tcp --dport 53 -j ACCEPT -m comment --comment "media-server-fw: Downloads DNS TCP"
    
    # Block all other outbound traffic (kill switch)
    iptables -A DOCKER-USER -i "$DOWNLOADS_BRIDGE" -j DROP -m comment --comment "media-server-fw: Downloads kill switch"
    
    # Prevent downloads network from accessing other internal networks
    iptables -I DOCKER-USER -o "$FRONTEND_BRIDGE" -i "$DOWNLOADS_BRIDGE" -j DROP -m comment --comment "media-server-fw: Downloads to frontend blocked"
    iptables -I DOCKER-USER -o "$BACKEND_BRIDGE" -i "$DOWNLOADS_BRIDGE" -j DROP -m comment --comment "media-server-fw: Downloads to backend blocked"
    iptables -I DOCKER-USER -o "$MONITORING_BRIDGE" -i "$DOWNLOADS_BRIDGE" -j DROP -m comment --comment "media-server-fw: Downloads to monitoring blocked"
    
    log_success "Downloads rules configured (VPN kill switch active)"
}

# Setup Monitoring network rules
setup_monitoring_rules() {
    log_info "Setting up Monitoring network rules..."
    
    # Allow monitoring to access all networks for metrics collection
    iptables -I DOCKER-USER -i "$MONITORING_BRIDGE" -o "$FRONTEND_BRIDGE" -j ACCEPT -m comment --comment "media-server-fw: Monitoring to frontend"
    iptables -I DOCKER-USER -i "$MONITORING_BRIDGE" -o "$BACKEND_BRIDGE" -j ACCEPT -m comment --comment "media-server-fw: Monitoring to backend"
    
    # Allow specific monitoring ports
    iptables -I DOCKER-USER -i "$MONITORING_BRIDGE" -p tcp --dport 9090 -j ACCEPT -m comment --comment "media-server-fw: Prometheus"
    iptables -I DOCKER-USER -i "$MONITORING_BRIDGE" -p tcp --dport 3000 -j ACCEPT -m comment --comment "media-server-fw: Grafana"
    iptables -I DOCKER-USER -i "$MONITORING_BRIDGE" -p tcp --dport 8080 -j ACCEPT -m comment --comment "media-server-fw: Trivy"
    
    # Block monitoring from downloads network
    iptables -I DOCKER-USER -i "$MONITORING_BRIDGE" -o "$DOWNLOADS_BRIDGE" -j DROP -m comment --comment "media-server-fw: Monitoring to downloads blocked"
    
    log_success "Monitoring rules configured"
}

# Setup rate limiting for DDoS protection
setup_rate_limiting() {
    log_info "Setting up rate limiting..."
    
    # Rate limit new connections to prevent DDoS
    iptables -I DOCKER-USER -p tcp --syn -m limit --limit 100/second --limit-burst 200 -j ACCEPT -m comment --comment "media-server-fw: Rate limit SYN"
    iptables -I DOCKER-USER -p tcp --syn -j DROP -m comment --comment "media-server-fw: Drop excess SYN"
    
    # Rate limit ICMP
    iptables -I DOCKER-USER -p icmp -m limit --limit 10/second --limit-burst 20 -j ACCEPT -m comment --comment "media-server-fw: Rate limit ICMP"
    iptables -I DOCKER-USER -p icmp -j DROP -m comment --comment "media-server-fw: Drop excess ICMP"
    
    log_success "Rate limiting configured"
}

# Setup logging for security events
setup_logging() {
    log_info "Setting up security logging..."
    
    # Log dropped packets (limited to prevent log flooding)
    iptables -I DOCKER-USER -m limit --limit 10/min -j LOG --log-prefix "DOCKER-FW-DROP: " --log-level 4 -m comment --comment "media-server-fw: Log drops"
    
    # Log potential port scans
    iptables -I DOCKER-USER -p tcp --tcp-flags ALL NONE -m limit --limit 3/min -j LOG --log-prefix "NULL-SCAN: " --log-level 4 -m comment --comment "media-server-fw: Log null scan"
    iptables -I DOCKER-USER -p tcp --tcp-flags ALL ALL -m limit --limit 3/min -j LOG --log-prefix "XMAS-SCAN: " --log-level 4 -m comment --comment "media-server-fw: Log xmas scan"
    
    log_success "Security logging configured"
}

# Save firewall rules persistently
save_rules() {
    log_info "Saving firewall rules..."
    
    # For Debian/Ubuntu
    if command -v netfilter-persistent >/dev/null 2>&1; then
        netfilter-persistent save
        log_success "Rules saved with netfilter-persistent"
    # For RHEL/CentOS
    elif command -v iptables-save >/dev/null 2>&1; then
        iptables-save > /etc/sysconfig/iptables 2>/dev/null || \
        iptables-save > /etc/iptables/rules.v4 2>/dev/null || \
        log_warning "Could not save rules automatically"
    fi
}

# Display current firewall rules
display_rules() {
    log_info "Current DOCKER-USER firewall rules:"
    echo ""
    iptables -L DOCKER-USER -v -n --line-numbers | grep -E "(media-server-fw|Chain)"
    echo ""
}

# Main execution
main() {
    check_root
    check_docker_iptables
    
    case "${1:-setup}" in
        setup)
            log_info "Setting up media server firewall rules..."
            backup_iptables
            clear_custom_rules
            
            setup_dmz_rules
            setup_frontend_rules
            setup_backend_rules
            setup_downloads_rules
            setup_monitoring_rules
            setup_rate_limiting
            setup_logging
            
            save_rules
            display_rules
            
            log_success "Firewall setup completed!"
            log_warning "Remember to test all services after applying firewall rules"
            ;;
            
        clear)
            log_info "Clearing media server firewall rules..."
            backup_iptables
            clear_custom_rules
            display_rules
            log_success "Firewall rules cleared"
            ;;
            
        status)
            display_rules
            ;;
            
        test)
            log_info "Testing firewall rules..."
            # Add connectivity tests here
            log_warning "Test mode not fully implemented"
            ;;
            
        *)
            echo "Usage: $0 {setup|clear|status|test}"
            echo ""
            echo "  setup  - Configure firewall rules"
            echo "  clear  - Remove custom firewall rules"
            echo "  status - Display current rules"
            echo "  test   - Test connectivity"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"