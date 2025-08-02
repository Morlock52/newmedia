#!/usr/bin/env bash
set -euo pipefail

# Fallback interactive setup for systems without GUI dialog tools
# This script provides the same functionality as setup-gui.sh but uses text prompts

# Determine project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration variables with defaults
DOMAIN=""
EMAIL=""
VPN_PROVIDER=""
VPN_TYPE="wireguard"
VPN_PORT_FORWARDING="on"
VPN_PORT_FORWARDING_PORT="6881"
PUID="$(id -u)"
PGID="$(id -g)"
TZ=""
UMASK="002"
DATA_ROOT="./data"
CONFIG_ROOT="./config"
DEPLOY_MONITORING="true"
SLACK_WEBHOOK=""

# Available VPN providers
VPN_PROVIDERS=("mullvad" "nordvpn" "protonvpn" "surfshark" "pia" "cyberghost" "expressvpn" "ipvanish" "purevpn" "windscribe")

# Common timezones
TIMEZONES=("UTC" "America/New_York" "America/Chicago" "America/Denver" "America/Los_Angeles" "Europe/London" "Europe/Paris" "Europe/Berlin" "Asia/Tokyo" "Asia/Singapore" "Australia/Sydney")

log() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}✅ $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" >&2
}

error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}" >&2
}

prompt() {
    echo -e "${PURPLE}🔵 $1${NC}" >&2
}

# Detect current timezone
detect_timezone() {
    if [[ -f /etc/timezone ]]; then
        cat /etc/timezone
    elif [[ -L /etc/localtime ]]; then
        readlink /etc/localtime | sed 's|.*/zoneinfo/||'
    else
        echo "UTC"
    fi
}

# Validate email format
validate_email() {
    local email="$1"
    if [[ "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        return 0
    else
        return 1
    fi
}

# Validate domain format
validate_domain() {
    local domain="$1"
    if [[ "$domain" =~ ^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?\.[a-zA-Z]{2,}$ ]]; then
        return 0
    else
        return 1
    fi
}

# Show welcome screen
show_welcome() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                    🎬 Media Server Stack Setup                      ║"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                      ║"
    echo "║  Welcome to the Media Server Stack Interactive Setup!               ║"
    echo "║                                                                      ║"
    echo "║  This setup will configure your Docker-based media server with:     ║"
    echo "║                                                                      ║"
    echo "║  • Jellyfin Media Server                                            ║"
    echo "║  • Sonarr/Radarr for TV/Movies                                      ║"
    echo "║  • qBittorrent with VPN Integration                                 ║"
    echo "║  • Traefik Reverse Proxy with SSL                                   ║"
    echo "║  • Prometheus/Grafana Monitoring                                    ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    read -p "Press Enter to continue..." -r
}

# Get input with default value
get_input() {
    local prompt_text="$1"
    local default_value="$2"
    local var_name="$3"
    local validation_func="${4:-}"
    
    while true; do
        if [[ -n "$default_value" ]]; then
            read -p "$prompt_text [$default_value]: " -r input
            input="${input:-$default_value}"
        else
            read -p "$prompt_text: " -r input
        fi
        
        if [[ -n "$validation_func" ]]; then
            if "$validation_func" "$input"; then
                declare -g "$var_name"="$input"
                break
            else
                error "Invalid input format. Please try again."
            fi
        else
            declare -g "$var_name"="$input"
            break
        fi
    done
}

# Show menu selection
show_menu() {
    local title="$1"
    local description="$2"
    shift 2
    local options=("$@")
    
    echo
    info "$title"
    echo "$description"
    echo
    
    local i=1
    for option in "${options[@]}"; do
        echo "  $i) $option"
        ((i++))
    done
    echo
    
    while true; do
        read -p "Select option (1-${#options[@]}): " -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
            echo "${options[$((choice-1))]}"
            return
        else
            error "Invalid selection. Please choose 1-${#options[@]}."
        fi
    done
}

# Get yes/no input
get_yes_no() {
    local prompt_text="$1"
    local default_value="${2:-y}"
    
    while true; do
        if [[ "$default_value" == "y" ]]; then
            read -p "$prompt_text [Y/n]: " -r response
        else
            read -p "$prompt_text [y/N]: " -r response
        fi
        
        response="${response:-$default_value}"
        case "$response" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo]) return 1 ;;
            *) error "Please answer yes or no." ;;
        esac
    done
}

# Main configuration steps
configure_domain() {
    echo
    info "🌐 Domain Configuration"
    echo "Your domain will be used for SSL certificates and service access:"
    echo "  • jellyfin.yourdomain.com"
    echo "  • sonarr.yourdomain.com"
    echo "  • radarr.yourdomain.com"
    echo
    
    get_input "Enter your domain name" "morloksmaze.com" "DOMAIN" "validate_domain"
    success "Domain: $DOMAIN"
}

configure_email() {
    echo
    info "📧 Email Configuration"
    echo "Your email will be used for Let's Encrypt certificate registration."
    echo
    
    get_input "Enter your email address" "admin@$DOMAIN" "EMAIL" "validate_email"
    success "Email: $EMAIL"
}

configure_vpn() {
    echo
    info "🔒 VPN Provider Selection"
    echo "Choose your VPN provider for secure downloading:"
    echo
    
    VPN_PROVIDER=$(show_menu "VPN Provider" "Select from supported providers:" "${VPN_PROVIDERS[@]}")
    success "VPN Provider: $VPN_PROVIDER"
}

configure_timezone() {
    echo
    info "🕐 Timezone Configuration"
    echo "This affects log timestamps and scheduled tasks."
    echo
    
    local current_tz
    current_tz=$(detect_timezone)
    info "Detected timezone: $current_tz"
    echo
    
    if get_yes_no "Use detected timezone ($current_tz)?"; then
        TZ="$current_tz"
    else
        TZ=$(show_menu "Timezone" "Select your timezone:" "${TIMEZONES[@]}")
    fi
    success "Timezone: $TZ"
}

configure_user() {
    echo
    info "👤 User Configuration"
    echo "Configure user and group IDs for proper file permissions."
    echo "Current user ID: $(id -u), group ID: $(id -g)"
    echo
    
    get_input "Enter User ID (PUID)" "$PUID" "PUID"
    get_input "Enter Group ID (PGID)" "$PGID" "PGID"
    success "User ID: $PUID, Group ID: $PGID"
}

configure_storage() {
    echo
    info "💾 Storage Configuration"
    echo "Configure paths for data and configuration storage."
    echo
    
    get_input "Data storage path (for movies, TV, music)" "$DATA_ROOT" "DATA_ROOT"
    get_input "Configuration storage path (for app settings)" "$CONFIG_ROOT" "CONFIG_ROOT"
    success "Data: $DATA_ROOT, Config: $CONFIG_ROOT"
}

configure_monitoring() {
    echo
    info "📊 Monitoring Configuration"
    echo "Prometheus/Grafana provides system metrics, dashboards, and alerting."
    echo
    
    if get_yes_no "Enable monitoring stack?"; then
        DEPLOY_MONITORING="true"
        echo
        info "🔔 Slack Notifications (Optional)"
        echo "Enter Slack webhook URL for alerts, or leave empty to skip."
        echo
        get_input "Slack webhook URL (optional)" "" "SLACK_WEBHOOK"
    else
        DEPLOY_MONITORING="false"
    fi
    success "Monitoring: $DEPLOY_MONITORING"
}

show_summary() {
    echo
    info "📋 Configuration Summary"
    echo "══════════════════════════════════════════════════════════════"
    echo "🌐 Domain:          $DOMAIN"
    echo "📧 Email:           $EMAIL"
    echo "🔒 VPN Provider:    $VPN_PROVIDER"
    echo "🕐 Timezone:        $TZ"
    echo "👤 User ID:         $PUID"
    echo "👥 Group ID:        $PGID"
    echo "💾 Data Path:       $DATA_ROOT"
    echo "⚙️ Config Path:     $CONFIG_ROOT"
    echo "📊 Monitoring:      $DEPLOY_MONITORING"
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        echo "🔔 Slack:           Configured"
    fi
    echo "══════════════════════════════════════════════════════════════"
    echo
    
    get_yes_no "Proceed with this configuration?"
}

write_env_file() {
    cat > "$ROOT/.env" << EOF
# VPN Configuration
VPN_PROVIDER=$VPN_PROVIDER
VPN_TYPE=$VPN_TYPE
VPN_PORT_FORWARDING=$VPN_PORT_FORWARDING
VPN_PORT_FORWARDING_PORT=$VPN_PORT_FORWARDING_PORT

# Domain and SSL
DOMAIN=$DOMAIN
EMAIL=$EMAIL

# User Configuration
PUID=$PUID
PGID=$PGID
TZ=$TZ
UMASK=$UMASK

# Storage Paths
DATA_ROOT=$DATA_ROOT
CONFIG_ROOT=$CONFIG_ROOT

# Database Configuration
POSTGRES_USER=mediaserver
POSTGRES_DB=mediaserver
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password

# API Keys (Use Docker Secrets)
JELLYFIN_API_KEY_FILE=/run/secrets/jellyfin_api_key
SONARR_API_KEY_FILE=/run/secrets/sonarr_api_key
RADARR_API_KEY_FILE=/run/secrets/radarr_api_key
LIDARR_API_KEY_FILE=/run/secrets/lidarr_api_key
READARR_API_KEY_FILE=/run/secrets/readarr_api_key
BAZARR_API_KEY_FILE=/run/secrets/bazarr_api_key
TAUTULLI_API_KEY_FILE=/run/secrets/tautulli_api_key

# PhotoPrism Admin Password (Use Docker Secret)
PHOTOPRISM_ADMIN_PASSWORD_FILE=/run/secrets/photoprism_admin_password

# Online Video Downloader (YouTubeDL-Material)
YTDL_MATERIAL_IMAGE=ghcr.io/iv-org/youtube-dl-material:latest
YTDL_MATERIAL_PORT=17442

# Monitoring Configuration
DEPLOY_MONITORING=$DEPLOY_MONITORING
SLACK_WEBHOOK=$SLACK_WEBHOOK
EOF
    
    success "Configuration saved to .env file"
}

show_next_steps() {
    echo
    success "🎉 Setup Complete!"
    echo
    info "Next steps:"
    echo "1. Add your VPN WireGuard private key:"
    echo "   echo \"YOUR_KEY\" > secrets/wg_private_key.txt"
    echo
    echo "2. Deploy the stack:"
    echo "   ./scripts/deploy.sh"
    echo
    echo "3. Access your services:"
    echo "   • Jellyfin: https://jellyfin.$DOMAIN"
    echo "   • Sonarr: https://sonarr.$DOMAIN"
    echo "   • Radarr: https://radarr.$DOMAIN"
    if [[ "$DEPLOY_MONITORING" == "true" ]]; then
        echo "   • Grafana: https://grafana.$DOMAIN"
    fi
    echo
}

main() {
    log "Starting Media Server Stack Interactive Setup"
    
    # Load existing config if present
    if [[ -f "$ROOT/.env" ]]; then
        warning "Existing .env file found - loading current settings"
        source "$ROOT/.env" 2>/dev/null || true
    fi
    
    show_welcome
    configure_domain
    configure_email
    configure_vpn
    configure_timezone
    configure_user
    configure_storage
    configure_monitoring
    
    if show_summary; then
        write_env_file
        show_next_steps
        
        if get_yes_no "Run security setup now?"; then
            "$ROOT/scripts/setup-security.sh" setup
        fi
        
        success "Setup completed successfully!"
    else
        warning "Setup cancelled"
        exit 0
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Setup cancelled by user${NC}"; exit 0' INT

main "$@"