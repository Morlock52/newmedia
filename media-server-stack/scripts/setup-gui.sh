#!/usr/bin/env bash
set -euo pipefail

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

# Check if we have dialog capabilities
DIALOG_CMD=""
if command -v whiptail >/dev/null 2>&1; then
    DIALOG_CMD="whiptail"
elif command -v dialog >/dev/null 2>&1; then
    DIALOG_CMD="dialog"
elif command -v zenity >/dev/null 2>&1; then
    DIALOG_CMD="zenity"
else
    echo -e "${YELLOW}⚠️  No GUI dialog command found. Falling back to interactive text mode.${NC}"
    echo -e "${BLUE}💡 For GUI mode, install: whiptail, dialog, or zenity${NC}"
    echo -e "${BLUE}💡 On Ubuntu/Debian: sudo apt install whiptail dialog${NC}"
    echo -e "${BLUE}💡 On macOS: brew install dialog newt${NC}"
    echo -e "${BLUE}💡 On CentOS/RHEL: sudo yum install dialog newt${NC}"
    echo -e "${BLUE}💡 On Fedora: sudo dnf install dialog newt${NC}"
    echo -e "${BLUE}💡 On Alpine: sudo apk add dialog newt${NC}"
    echo
    sleep 2
    # Fall back to interactive mode
    exec "$ROOT/scripts/setup-interactive.sh" "$@"
fi

# Configuration variables with defaults
DOMAIN=""
EMAIL=""
VPN_PROVIDER=""
VPN_TYPE="wireguard"
VPN_PORT_FORWARDING="on"
VPN_PORT_FORWARDING_PORT="6881"
PUID="1000"
PGID="1000"
TZ=""
UMASK="002"
DATA_ROOT="./data"
CONFIG_ROOT="./config"
DEPLOY_MONITORING="true"
SLACK_WEBHOOK=""

# Available VPN providers
VPN_PROVIDERS=("mullvad" "nordvpn" "protonvpn" "surfshark" "pia" "cyberghost" "expressvpn" "ipvanish" "purevpn" "windscribe" "airvpn" "ivpn")

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

# Detect current timezone
detect_timezone() {
    if [[ -f /etc/timezone ]]; then
        cat /etc/timezone
    elif [[ -L /etc/localtime ]]; then
        readlink /etc/localtime | sed 's|.*/zoneinfo/||'
    else
        # Default fallback
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
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        whiptail --title "🎬 Media Server Stack Setup" --msgbox "Welcome to the Media Server Stack Setup Wizard!\n\nThis interactive setup will guide you through configuring your Docker-based media server with:\n\n• Jellyfin Media Server\n• Sonarr/Radarr for TV/Movies\n• qBittorrent with VPN\n• Traefik Reverse Proxy\n• Prometheus/Grafana Monitoring\n\nPress OK to continue..." 18 70
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        dialog --title "🎬 Media Server Stack Setup" --msgbox "Welcome to the Media Server Stack Setup Wizard!\n\nThis interactive setup will guide you through configuring your Docker-based media server with:\n\n• Jellyfin Media Server\n• Sonarr/Radarr for TV/Movies\n• qBittorrent with VPN\n• Traefik Reverse Proxy\n• Prometheus/Grafana Monitoring\n\nPress OK to continue..." 18 70
        clear
    fi
}

# Get domain name
get_domain() {
    while true; do
        if [[ "$DIALOG_CMD" == "whiptail" ]]; then
            DOMAIN=$(whiptail --title "🌐 Domain Configuration" --inputbox "Enter your domain name for SSL certificates:\n\nThis will be used for:\n• jellyfin.yourdomain.com\n• sonarr.yourdomain.com\n• radarr.yourdomain.com\n\nExample: mydomain.com" 14 70 "${DOMAIN:-morloksmaze.com}" 3>&1 1>&2 2>&3)
        elif [[ "$DIALOG_CMD" == "dialog" ]]; then
            DOMAIN=$(dialog --title "🌐 Domain Configuration" --inputbox "Enter your domain name for SSL certificates:\n\nThis will be used for:\n• jellyfin.yourdomain.com\n• sonarr.yourdomain.com\n• radarr.yourdomain.com\n\nExample: mydomain.com" 14 70 "${DOMAIN:-morloksmaze.com}" 3>&1 1>&2 2>&3)
            clear
        fi
        
        if [[ $? -eq 0 && -n "$DOMAIN" ]]; then
            if validate_domain "$DOMAIN"; then
                success "Domain: $DOMAIN"
                break
            else
                if [[ "$DIALOG_CMD" == "whiptail" ]]; then
                    whiptail --title "❌ Invalid Domain" --msgbox "Invalid domain format. Please enter a valid domain like 'morloksmaze.com'" 8 50
                elif [[ "$DIALOG_CMD" == "dialog" ]]; then
                    dialog --title "❌ Invalid Domain" --msgbox "Invalid domain format. Please enter a valid domain like 'morloksmaze.com'" 8 50
                    clear
                fi
            fi
        else
            exit 0
        fi
    done
}

# Get email address
get_email() {
    while true; do
        if [[ "$DIALOG_CMD" == "whiptail" ]]; then
            EMAIL=$(whiptail --title "📧 Email Configuration" --inputbox "Enter your email address for SSL certificates:\n\nThis will be used for Let's Encrypt certificate registration and renewal notifications.\n\nExample: admin@${DOMAIN}" 12 70 "${EMAIL:-admin@${DOMAIN}}" 3>&1 1>&2 2>&3)
        elif [[ "$DIALOG_CMD" == "dialog" ]]; then
            EMAIL=$(dialog --title "📧 Email Configuration" --inputbox "Enter your email address for SSL certificates:\n\nThis will be used for Let's Encrypt certificate registration and renewal notifications.\n\nExample: admin@${DOMAIN}" 12 70 "${EMAIL:-admin@${DOMAIN}}" 3>&1 1>&2 2>&3)
            clear
        fi
        
        if [[ $? -eq 0 && -n "$EMAIL" ]]; then
            if validate_email "$EMAIL"; then
                success "Email: $EMAIL"
                break
            else
                if [[ "$DIALOG_CMD" == "whiptail" ]]; then
                    whiptail --title "❌ Invalid Email" --msgbox "Invalid email format. Please enter a valid email address." 8 50
                elif [[ "$DIALOG_CMD" == "dialog" ]]; then
                    dialog --title "❌ Invalid Email" --msgbox "Invalid email format. Please enter a valid email address." 8 50
                    clear
                fi
            fi
        else
            exit 0
        fi
    done
}

# Get VPN provider
get_vpn_provider() {
    local menu_items=()
    local i=1
    for provider in "${VPN_PROVIDERS[@]}"; do
        menu_items+=("$i" "$provider")
        ((i++))
    done
    
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        choice=$(whiptail --title "🔒 VPN Provider Selection" --menu "Choose your VPN provider:\n\nSupported providers with WireGuard support:" 18 70 10 "${menu_items[@]}" 3>&1 1>&2 2>&3)
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        choice=$(dialog --title "🔒 VPN Provider Selection" --menu "Choose your VPN provider:\n\nSupported providers with WireGuard support:" 18 70 10 "${menu_items[@]}" 3>&1 1>&2 2>&3)
        clear
    fi
    
    if [[ $? -eq 0 && -n "$choice" ]]; then
        VPN_PROVIDER="${VPN_PROVIDERS[$((choice-1))]}"
        success "VPN Provider: $VPN_PROVIDER"
    else
        exit 0
    fi
}

# Get timezone
get_timezone() {
    local current_tz
    current_tz=$(detect_timezone)
    
    local menu_items=()
    local i=1
    for tz in "${TIMEZONES[@]}"; do
        if [[ "$tz" == "$current_tz" ]]; then
            menu_items+=("$i" "$tz (detected)")
        else
            menu_items+=("$i" "$tz")
        fi
        ((i++))
    done
    
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        choice=$(whiptail --title "🕐 Timezone Configuration" --menu "Select your timezone:\n\nThis affects log timestamps and scheduled tasks." 18 70 10 "${menu_items[@]}" 3>&1 1>&2 2>&3)
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        choice=$(dialog --title "🕐 Timezone Configuration" --menu "Select your timezone:\n\nThis affects log timestamps and scheduled tasks." 18 70 10 "${menu_items[@]}" 3>&1 1>&2 2>&3)
        clear
    fi
    
    if [[ $? -eq 0 && -n "$choice" ]]; then
        TZ="${TIMEZONES[$((choice-1))]}"
        success "Timezone: $TZ"
    else
        exit 0
    fi
}

# Get user configuration
get_user_config() {
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        PUID=$(whiptail --title "👤 User Configuration" --inputbox "Enter the User ID (PUID) for container processes:\n\nThis should match your host user ID to ensure proper file permissions.\n\nCurrent user ID: $(id -u)" 12 70 "${PUID:-$(id -u)}" 3>&1 1>&2 2>&3)
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        PUID=$(dialog --title "👤 User Configuration" --inputbox "Enter the User ID (PUID) for container processes:\n\nThis should match your host user ID to ensure proper file permissions.\n\nCurrent user ID: $(id -u)" 12 70 "${PUID:-$(id -u)}" 3>&1 1>&2 2>&3)
        clear
    fi
    
    if [[ $? -eq 0 && -n "$PUID" ]]; then
        if [[ "$DIALOG_CMD" == "whiptail" ]]; then
            PGID=$(whiptail --title "👥 Group Configuration" --inputbox "Enter the Group ID (PGID) for container processes:\n\nThis should match your host group ID.\n\nCurrent group ID: $(id -g)" 10 70 "${PGID:-$(id -g)}" 3>&1 1>&2 2>&3)
        elif [[ "$DIALOG_CMD" == "dialog" ]]; then
            PGID=$(dialog --title "👥 Group Configuration" --inputbox "Enter the Group ID (PGID) for container processes:\n\nThis should match your host group ID.\n\nCurrent group ID: $(id -g)" 10 70 "${PGID:-$(id -g)}" 3>&1 1>&2 2>&3)
            clear
        fi
        
        if [[ $? -eq 0 && -n "$PGID" ]]; then
            success "User ID: $PUID, Group ID: $PGID"
        else
            exit 0
        fi
    else
        exit 0
    fi
}

# Get storage paths
get_storage_paths() {
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        DATA_ROOT=$(whiptail --title "💾 Data Storage Path" --inputbox "Enter the path for media data storage:\n\nThis will store your movies, TV shows, music, etc.\nUse absolute path for production, relative for testing.\n\nRecommended: /data or ./data" 12 70 "${DATA_ROOT}" 3>&1 1>&2 2>&3)
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        DATA_ROOT=$(dialog --title "💾 Data Storage Path" --inputbox "Enter the path for media data storage:\n\nThis will store your movies, TV shows, music, etc.\nUse absolute path for production, relative for testing.\n\nRecommended: /data or ./data" 12 70 "${DATA_ROOT}" 3>&1 1>&2 2>&3)
        clear
    fi
    
    if [[ $? -eq 0 && -n "$DATA_ROOT" ]]; then
        if [[ "$DIALOG_CMD" == "whiptail" ]]; then
            CONFIG_ROOT=$(whiptail --title "⚙️ Configuration Storage Path" --inputbox "Enter the path for service configuration storage:\n\nThis will store application settings and databases.\n\nRecommended: /config or ./config" 10 70 "${CONFIG_ROOT}" 3>&1 1>&2 2>&3)
        elif [[ "$DIALOG_CMD" == "dialog" ]]; then
            CONFIG_ROOT=$(dialog --title "⚙️ Configuration Storage Path" --inputbox "Enter the path for service configuration storage:\n\nThis will store application settings and databases.\n\nRecommended: /config or ./config" 10 70 "${CONFIG_ROOT}" 3>&1 1>&2 2>&3)
            clear
        fi
        
        if [[ $? -eq 0 && -n "$CONFIG_ROOT" ]]; then
            success "Data: $DATA_ROOT, Config: $CONFIG_ROOT"
        else
            exit 0
        fi
    else
        exit 0
    fi
}

# Get monitoring configuration
get_monitoring_config() {
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        if whiptail --title "📊 Monitoring Setup" --yesno "Enable Prometheus/Grafana monitoring?\n\nThis includes:\n• System metrics collection\n• Service health monitoring\n• Performance dashboards\n• Alerting capabilities\n\nRecommended: Yes" 12 70; then
            DEPLOY_MONITORING="true"
            
            SLACK_WEBHOOK=$(whiptail --title "🔔 Slack Notifications (Optional)" --inputbox "Enter Slack webhook URL for alerts (optional):\n\nLeave empty to skip Slack notifications.\nYou can add this later in the .env file.\n\nExample: https://hooks.slack.com/services/..." 12 70 "" 3>&1 1>&2 2>&3)
        else
            DEPLOY_MONITORING="false"
        fi
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        if dialog --title "📊 Monitoring Setup" --yesno "Enable Prometheus/Grafana monitoring?\n\nThis includes:\n• System metrics collection\n• Service health monitoring\n• Performance dashboards\n• Alerting capabilities\n\nRecommended: Yes" 12 70; then
            DEPLOY_MONITORING="true"
            
            SLACK_WEBHOOK=$(dialog --title "🔔 Slack Notifications (Optional)" --inputbox "Enter Slack webhook URL for alerts (optional):\n\nLeave empty to skip Slack notifications.\nYou can add this later in the .env file.\n\nExample: https://hooks.slack.com/services/..." 12 70 "" 3>&1 1>&2 2>&3)
            clear
        else
            DEPLOY_MONITORING="false"
        fi
        clear
    fi
    
    success "Monitoring: $DEPLOY_MONITORING"
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        success "Slack notifications configured"
    fi
}

# Show configuration summary
show_summary() {
    local summary="Configuration Summary:\n\n"
    summary+="🌐 Domain: $DOMAIN\n"
    summary+="📧 Email: $EMAIL\n"
    summary+="🔒 VPN Provider: $VPN_PROVIDER\n"
    summary+="🕐 Timezone: $TZ\n"
    summary+="👤 User ID: $PUID\n"
    summary+="👥 Group ID: $PGID\n"
    summary+="💾 Data Path: $DATA_ROOT\n"
    summary+="⚙️ Config Path: $CONFIG_ROOT\n"
    summary+="📊 Monitoring: $DEPLOY_MONITORING\n"
    
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        summary+="🔔 Slack: Configured\n"
    fi
    
    summary+="\nProceed with this configuration?"
    
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        if whiptail --title "📋 Configuration Summary" --yesno "$summary" 20 70; then
            return 0
        else
            return 1
        fi
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        if dialog --title "📋 Configuration Summary" --yesno "$summary" 20 70; then
            clear
            return 0
        else
            clear
            return 1
        fi
    fi
}

# Write configuration to .env file
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

# Show next steps
show_next_steps() {
    local next_steps="🎉 Setup Complete!\n\n"
    next_steps+="Next steps:\n\n"
    next_steps+="1. Add your VPN WireGuard private key:\n"
    next_steps+="   echo \"YOUR_KEY\" > secrets/wg_private_key.txt\n\n"
    next_steps+="2. Deploy the stack:\n"
    next_steps+="   ./scripts/deploy.sh\n\n"
    next_steps+="3. Access your services:\n"
    next_steps+="   • Jellyfin: https://jellyfin.$DOMAIN\n"
    next_steps+="   • Sonarr: https://sonarr.$DOMAIN\n"
    next_steps+="   • Radarr: https://radarr.$DOMAIN\n"
    
    if [[ "$DEPLOY_MONITORING" == "true" ]]; then
        next_steps+="   • Grafana: https://grafana.$DOMAIN\n"
    fi
    
    if [[ "$DIALOG_CMD" == "whiptail" ]]; then
        whiptail --title "🚀 Ready to Deploy!" --msgbox "$next_steps" 20 70
    elif [[ "$DIALOG_CMD" == "dialog" ]]; then
        dialog --title "🚀 Ready to Deploy!" --msgbox "$next_steps" 20 70
        clear
    fi
}

# Main execution
main() {
    log "Starting Media Server Stack GUI Setup"
    info "Using dialog command: $DIALOG_CMD"
    
    # Load existing config if present
    if [[ -f "$ROOT/.env" ]]; then
        warning "Existing .env file found - loading current settings"
        source "$ROOT/.env" 2>/dev/null || true
    fi
    
    show_welcome
    get_domain
    get_email
    get_vpn_provider
    get_timezone
    get_user_config
    get_storage_paths
    get_monitoring_config
    
    # Show summary and confirm
    if show_summary; then
        write_env_file
        show_next_steps
        success "Setup completed successfully!"
        
        # Offer to run security setup
        if [[ "$DIALOG_CMD" == "whiptail" ]]; then
            if whiptail --title "🔐 Security Setup" --yesno "Run security setup now?\n\nThis will generate secrets and configure security settings." 10 70; then
                "$ROOT/scripts/setup-security.sh" setup
                
                # Offer PIA VPN setup if using PIA
                if [[ "$VPN_PROVIDER" == "pia" ]]; then
                    if whiptail --title "🔒 PIA VPN Setup" --yesno "Set up PIA WireGuard automatically?\n\nThis will:\n• Download PIA official scripts\n• Generate WireGuard config\n• Extract and install private key\n• Restart VPN services\n\nYou'll need your PIA username/password." 14 70; then
                        "$ROOT/scripts/setup-pia-vpn.sh"
                    else
                        whiptail --title "ℹ️ Manual PIA Setup" --msgbox "To set up PIA manually later:\n\n1. Run: ./scripts/setup-pia-vpn.sh\n2. Or add your WireGuard key to:\n   secrets/wg_private_key.txt" 10 60
                    fi
                fi
            fi
        elif [[ "$DIALOG_CMD" == "dialog" ]]; then
            if dialog --title "🔐 Security Setup" --yesno "Run security setup now?\n\nThis will generate secrets and configure security settings." 10 70; then
                clear
                "$ROOT/scripts/setup-security.sh" setup
                
                # Offer PIA VPN setup if using PIA
                if [[ "$VPN_PROVIDER" == "pia" ]]; then
                    if dialog --title "🔒 PIA VPN Setup" --yesno "Set up PIA WireGuard automatically?\n\nThis will:\n• Download PIA official scripts\n• Generate WireGuard config\n• Extract and install private key\n• Restart VPN services\n\nYou'll need your PIA username/password." 14 70; then
                        clear
                        "$ROOT/scripts/setup-pia-vpn.sh"
                    else
                        dialog --title "ℹ️ Manual PIA Setup" --msgbox "To set up PIA manually later:\n\n1. Run: ./scripts/setup-pia-vpn.sh\n2. Or add your WireGuard key to:\n   secrets/wg_private_key.txt" 10 60
                        clear
                    fi
                fi
            fi
        fi
        
    else
        warning "Setup cancelled by user"
        exit 0
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Setup cancelled by user${NC}"; exit 0' INT

main "$@"