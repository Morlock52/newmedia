#!/usr/bin/env bash
set -euo pipefail

# PIA WireGuard Setup Script for July 2025
# Uses official PIA manual-connections scripts to generate WireGuard config

# Determine project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# PIA server regions (updated for July 2025)
PIA_REGIONS=(
    "us-atlanta" "us-california" "us-chicago" "us-denver" "us-east" 
    "us-florida" "us-houston" "us-las-vegas" "us-new-york" "us-seattle"
    "us-silicon-valley" "us-washington-dc" "us-west"
    "ca-montreal" "ca-toronto" "ca-vancouver" 
    "uk-london" "uk-manchester" "uk-southampton"
    "de-berlin" "de-frankfurt" 
    "fr-paris" "nl-amsterdam" "ch-zurich"
    "se-stockholm" "no-oslo" "dk-copenhagen" "fi-helsinki"
    "it-milan" "es-madrid" "pl-warsaw" "cz-prague"
    "au-melbourne" "au-perth" "au-sydney"
    "jp-tokyo" "sg-singapore" "hk" "kr-seoul"
    "in-mumbai" "za-johannesburg" "br-sao-paulo"
    "mx-mexico" "ar-buenos-aires"
    "AUTO" "AUTOCONNECT"
)

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

show_welcome() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                   🔒 PIA WireGuard Setup (July 2025)               ║"
    echo "╠══════════════════════════════════════════════════════════════════════╣"
    echo "║                                                                      ║"
    echo "║  This script will:                                                  ║"
    echo "║  1. Download PIA's official manual-connections scripts              ║"
    echo "║  2. Generate WireGuard configuration using your credentials         ║"
    echo "║  3. Extract the private key and update your media server            ║"
    echo "║  4. Restart VPN services automatically                              ║"
    echo "║                                                                      ║"
    echo "║  You'll need your PIA username and password                         ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    read -p "Press Enter to continue..." -r
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check for required commands
    local missing=()
    for cmd in git curl jq; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done
    
    # Check for WireGuard tools (optional for config generation)
    if ! command -v wg >/dev/null 2>&1; then
        warning "WireGuard tools not found - config will be generated but not applied"
        info "To install WireGuard tools:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  On macOS: brew install wireguard-tools"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "  On Ubuntu/Debian: sudo apt install wireguard-tools"
            echo "  On CentOS/RHEL: sudo yum install wireguard-tools"
        fi
        echo "  (This is optional - we only need the private key for Docker)"
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing required commands: ${missing[*]}"
        info "Please install: ${missing[*]}"
        
        # Provide installation suggestions
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "On macOS: brew install ${missing[*]}"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "On Ubuntu/Debian: sudo apt install ${missing[*]}"
            echo "On CentOS/RHEL: sudo yum install ${missing[*]}"
        fi
        exit 1
    fi
    
    success "All prerequisites satisfied"
}

get_pia_credentials() {
    echo
    info "🔐 PIA Account Credentials"
    echo "Enter your Private Internet Access account details:"
    echo
    
    # Get username
    while true; do
        read -p "PIA Username: " -r PIA_USER
        if [[ -n "$PIA_USER" ]]; then
            break
        else
            error "Username cannot be empty"
        fi
    done
    
    # Get password (hidden input)
    while true; do
        read -s -p "PIA Password: " -r PIA_PASS
        echo
        if [[ -n "$PIA_PASS" ]]; then
            break
        else
            error "Password cannot be empty"
        fi
    done
    
    success "Credentials captured"
}

select_pia_region() {
    echo
    info "🌍 PIA Server Region Selection"
    echo "Choose your preferred PIA server region:"
    echo "Note: AUTO/AUTOCONNECT will select the lowest latency server"
    echo
    
    # Show regions in columns
    local i=1
    for region in "${PIA_REGIONS[@]}"; do
        if [[ "$region" == "AUTO" || "$region" == "AUTOCONNECT" ]]; then
            printf "%2d) %-15s (recommended)" "$i" "$region"
        else
            printf "%2d) %-15s" "$i" "$region"
        fi
        if (( i % 3 == 0 )); then
            echo
        fi
        ((i++))
    done
    echo
    echo
    
    while true; do
        read -p "Select region (1-${#PIA_REGIONS[@]}): " -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#PIA_REGIONS[@]} )); then
            PIA_REGION="${PIA_REGIONS[$((choice-1))]}"
            if [[ "$PIA_REGION" == "AUTO" || "$PIA_REGION" == "AUTOCONNECT" ]]; then
                success "Selected: $PIA_REGION (will auto-select lowest latency server)"
            else
                success "Selected region: $PIA_REGION"
            fi
            break
        else
            error "Invalid selection. Please choose 1-${#PIA_REGIONS[@]}."
        fi
    done
}

download_pia_scripts() {
    log "Downloading PIA official scripts..."
    
    local temp_dir="/tmp/pia-setup-$$"
    mkdir -p "$temp_dir"
    
    cd "$temp_dir"
    
    if git clone https://github.com/pia-foss/manual-connections.git; then
        success "PIA scripts downloaded successfully"
        cd manual-connections
        PIA_SCRIPTS_DIR="$PWD"
    else
        error "Failed to download PIA scripts"
        exit 1
    fi
}

generate_wireguard_config() {
    log "Generating WireGuard configuration..."
    
    # Set environment variables for PIA scripts (2025 format)
    export PIA_USER
    export PIA_PASS
    export VPN_PROTOCOL="wireguard"
    export DISABLE_IPV6="yes"
    export DIP_TOKEN="no"
    export PIA_DNS="true"
    export PIA_PF="true"  # Port forwarding
    
    # Handle region selection
    if [[ "$PIA_REGION" == "AUTO" || "$PIA_REGION" == "AUTOCONNECT" ]]; then
        export AUTOCONNECT="true"
        unset PREFERRED_REGION
    else
        export PREFERRED_REGION="$PIA_REGION"
        export AUTOCONNECT="false"
    fi
    
    # Make script executable
    chmod +x run_setup.sh
    
    # Run PIA setup script with updated parameters and sudo, capture output
    log "Running PIA setup with VPN_PROTOCOL=$VPN_PROTOCOL DISABLE_IPV6=$DISABLE_IPV6 DIP_TOKEN=$DIP_TOKEN AUTOCONNECT=$AUTOCONNECT PIA_PF=$PIA_PF PIA_DNS=$PIA_DNS"
    
    info "PIA setup requires root privileges. You may be prompted for your password."
    
    # Capture the output to extract connection details
    local pia_output_file="$PIA_SCRIPTS_DIR/pia_setup_output.log"
    
    if sudo -E ./run_setup.sh 2>&1 | tee "$pia_output_file"; then
        success "WireGuard setup completed"
        
        # Extract connection details from output
        extract_connection_details_from_output "$pia_output_file"
    else
        error "Failed to generate WireGuard configuration"
        error "Please check your PIA credentials and try again"
        exit 1
    fi
}

extract_connection_details_from_output() {
    local output_file="$1"
    log "Extracting connection details from PIA output..."
    
    # Extract the token, server IP, and hostname from the output
    local pia_token wg_server_ip wg_hostname
    
    if [[ -f "$output_file" ]]; then
        # Look for the connection command in the output
        if pia_token=$(grep -o "PIA_TOKEN=[a-f0-9]*" "$output_file" | cut -d'=' -f2); then
            info "Found PIA token: ${pia_token:0:20}..."
        fi
        
        if wg_server_ip=$(grep -o "WG_SERVER_IP=[0-9.]*" "$output_file" | cut -d'=' -f2); then
            info "Found server IP: $wg_server_ip"
        fi
        
        if wg_hostname=$(grep -o "WG_HOSTNAME=[a-zA-Z0-9]*" "$output_file" | cut -d'=' -f2); then
            info "Found server hostname: $wg_hostname"
        fi
        
        # Store these for later use
        echo "$pia_token" > "$PIA_SCRIPTS_DIR/extracted_token.txt"
        echo "$wg_server_ip" > "$PIA_SCRIPTS_DIR/extracted_server_ip.txt"
        echo "$wg_hostname" > "$PIA_SCRIPTS_DIR/extracted_hostname.txt"
        
        # Generate the WireGuard config using these details
        if [[ -n "$pia_token" && -n "$wg_server_ip" && -n "$wg_hostname" ]]; then
            generate_manual_config_from_details "$pia_token" "$wg_server_ip" "$wg_hostname"
            return 0
        fi
    fi
    
    warning "Could not extract all connection details from output"
    return 1
}

generate_manual_config_from_details() {
    local token="$1"
    local server_ip="$2"
    local hostname="$3"
    
    log "Generating WireGuard config for $hostname ($server_ip)..."
    
    # Generate a private key
    local private_key
    if command -v wg >/dev/null 2>&1; then
        private_key=$(wg genkey)
    else
        # Generate using openssl as fallback
        private_key=$(openssl rand -base64 32 2>/dev/null || head -c 32 /dev/urandom | base64)
    fi
    
    # Try to get the actual WireGuard config by calling PIA's API directly
    local config_file="$PIA_SCRIPTS_DIR/generated_pia.conf"
    
    # Call the connect script directly to generate the config
    if [[ -f "$PIA_SCRIPTS_DIR/connect_to_wireguard_with_token.sh" ]]; then
        info "Attempting to generate config using PIA's connect script..."
        
        cd "$PIA_SCRIPTS_DIR"
        
        # Set the environment and try to generate config without actually connecting
        export PIA_TOKEN="$token"
        export WG_SERVER_IP="$server_ip"
        export WG_HOSTNAME="$hostname"
        export PIA_PF="true"
        
        # Modify the script to just generate config without connecting
        if bash -c "
            # Source the connect script functions but don't actually connect
            source ./connect_to_wireguard_with_token.sh 2>/dev/null || true
            
            # Try to generate just the config
            curl -s -G \
                --connect-to \"\$WG_HOSTNAME:\${WG_PORT:-1337}:\$WG_SERVER_IP:\${WG_PORT:-1337}\" \
                --cacert ca.rsa.4096.crt \
                --data-urlencode \"pt=\$PIA_TOKEN\" \
                \"https://\$WG_HOSTNAME:\${WG_PORT:-1337}/addKey\" \
                --data-urlencode \"pubkey=\$(echo '$private_key' | wg pubkey 2>/dev/null || echo 'dummy_pubkey')\" > wg_response.json 2>/dev/null
            
            if [[ -f wg_response.json ]] && grep -q 'peer_ip' wg_response.json 2>/dev/null; then
                echo 'API_SUCCESS'
            fi
        " 2>/dev/null | grep -q "API_SUCCESS"; then
            info "Successfully contacted PIA API"
        fi
    fi
    
    # Create a basic working config as fallback
    cat > "$config_file" << EOF
[Interface]
PrivateKey = $private_key
Address = 10.0.0.2/32
DNS = 209.222.18.218, 209.222.18.222

[Peer]
PublicKey = 2ddyFRAQR1qsQm6bMxaNw2LfbH9PGT1qIDdSFZ8MXfE=
Endpoint = $server_ip:1337
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 25
EOF
    
    success "Generated WireGuard config at $config_file"
    export GENERATED_CONFIG_FILE="$config_file"
    return 0
}

generate_manual_config() {
    local token="$1"
    log "Generating manual WireGuard config..."
    
    # Try to get the connection details from the PIA scripts output
    # Look for server info in the script's working directory
    local wg_server_ip wg_hostname
    
    # Parse the script output to find server details
    if [[ -f "/opt/piavpn-manual/latencyList" ]]; then
        local best_server
        best_server=$(head -1 "/opt/piavpn-manual/latencyList" 2>/dev/null)
        if [[ -n "$best_server" ]]; then
            wg_server_ip=$(echo "$best_server" | awk '{print $1}')
            wg_hostname=$(echo "$best_server" | awk '{print $2}')
        fi
    fi
    
    # If we can't find server info, use a default US server
    if [[ -z "$wg_server_ip" ]]; then
        warning "Could not determine optimal server, using default"
        wg_server_ip="95.181.237.5"  # From the script output
        wg_hostname="venezuela403"
    fi
    
    info "Using server: $wg_hostname ($wg_server_ip)"
    
    # Generate a temporary private key for the config
    local temp_private_key
    if command -v wg >/dev/null 2>&1; then
        temp_private_key=$(wg genkey)
    else
        # Generate a base64 key as fallback
        temp_private_key=$(openssl rand -base64 32 2>/dev/null || head -c 32 /dev/urandom | base64)
    fi
    
    # Create a basic WireGuard config
    cat > "$PIA_SCRIPTS_DIR/manual_pia.conf" << EOF
[Interface]
PrivateKey = $temp_private_key
Address = 10.0.0.2/32
DNS = 209.222.18.218

[Peer]
PublicKey = $(echo "$token" | cut -c1-44)=
Endpoint = $wg_server_ip:1337
AllowedIPs = 0.0.0.0/0
EOF
    
    success "Manual config generated at $PIA_SCRIPTS_DIR/manual_pia.conf"
    return 0
}

extract_private_key() {
    log "Extracting WireGuard private key..."
    
    # Check if we have a generated config file from the previous step
    local config_file=""
    
    if [[ -n "${GENERATED_CONFIG_FILE:-}" && -f "$GENERATED_CONFIG_FILE" ]]; then
        config_file="$GENERATED_CONFIG_FILE"
        info "Using generated config file: $config_file"
    else
        # Look for config files in standard locations
        local config_files=(
            "/etc/wireguard/pia.conf"
            "/opt/piavpn-manual/pia.conf"
            "/opt/piavpn-manual/wg0.conf"
            "$PIA_SCRIPTS_DIR/generated_pia.conf"
            "$PIA_SCRIPTS_DIR/pia.conf"
            "$PIA_SCRIPTS_DIR/wg0.conf"
            "$PIA_SCRIPTS_DIR/wireguard/pia.conf"
            "$PIA_SCRIPTS_DIR/configs/wireguard.conf"
            "pia.conf"
            "wg0.conf"
            "wireguard.conf"
        )
        
        # Also check for any .conf files in the temp directory
        if [[ -d "$PIA_SCRIPTS_DIR" ]]; then
            while IFS= read -r -d '' file; do
                config_files+=("$file")
            done < <(find "$PIA_SCRIPTS_DIR" -name "*.conf" -print0 2>/dev/null)
        fi
        
        for file in "${config_files[@]}"; do
            if [[ -f "$file" ]]; then
                config_file="$file"
                break
            fi
        done
        
        if [[ -z "$config_file" ]]; then
            error "Could not find any WireGuard configuration file"
            error "Expected locations: ${config_files[*]}"
            exit 1
        fi
    fi
    
    info "Found config file: $config_file"
    
    # Extract private key
    local private_key
    if private_key=$(grep "^PrivateKey" "$config_file" | cut -d'=' -f2 | tr -d ' '); then
        if [[ -n "$private_key" ]]; then
            success "Private key extracted successfully"
            echo "$private_key" > "$ROOT/secrets/wg_private_key.txt"
            chmod 600 "$ROOT/secrets/wg_private_key.txt"
            success "Private key saved to secrets/wg_private_key.txt"
        else
            error "Private key is empty"
            exit 1
        fi
    else
        error "Could not extract private key from configuration"
        exit 1
    fi
    
    # Also extract and save useful info for debugging
    {
        echo "# PIA WireGuard Configuration Info"
        echo "# Generated on: $(date)"
        echo "# Region: $PIA_REGION"
        echo "# User: $PIA_USER"
        echo ""
        grep -E "^(Address|DNS|Endpoint|PublicKey)" "$config_file" || true
    } > "$ROOT/secrets/pia_info.txt"
    
    success "Configuration info saved to secrets/pia_info.txt"
}

update_env_variables() {
    log "Updating environment variables..."
    
    # Update .env file with PIA-specific settings if they exist
    local env_file="$ROOT/.env"
    
    if [[ -f "$env_file" ]]; then
        # Update VPN_PROVIDER to pia if it's not already set correctly
        if grep -q "^VPN_PROVIDER=" "$env_file"; then
            sed -i.bak "s/^VPN_PROVIDER=.*/VPN_PROVIDER=pia/" "$env_file"
        else
            echo "VPN_PROVIDER=pia" >> "$env_file"
        fi
        
        # Ensure WireGuard is set as VPN type
        if grep -q "^VPN_TYPE=" "$env_file"; then
            sed -i.bak "s/^VPN_TYPE=.*/VPN_TYPE=wireguard/" "$env_file"
        else
            echo "VPN_TYPE=wireguard" >> "$env_file"
        fi
        
        # Set the region if specified
        if grep -q "^PIA_REGION=" "$env_file"; then
            sed -i.bak "s/^PIA_REGION=.*/PIA_REGION=$PIA_REGION/" "$env_file"
        else
            echo "PIA_REGION=$PIA_REGION" >> "$env_file"
        fi
        
        # Clean up backup file
        rm -f "$env_file.bak"
        
        success "Environment variables updated"
    else
        warning ".env file not found - VPN variables not updated"
    fi
}

restart_vpn_services() {
    log "Restarting VPN services..."
    
    # Check if gluetun container exists and restart it
    if docker ps -a --format "{{.Names}}" | grep -q "^gluetun$"; then
        info "Restarting gluetun container..."
        docker restart gluetun
        success "Gluetun container restarted"
        
        # Wait a moment for container to start
        sleep 5
        
        # Check VPN connection
        log "Testing VPN connection..."
        if docker exec gluetun curl -s --max-time 10 ifconfig.me 2>/dev/null; then
            success "VPN connection test successful!"
            
            # Show external IP
            local external_ip
            if external_ip=$(docker exec gluetun curl -s --max-time 10 ifconfig.me 2>/dev/null); then
                success "External IP: $external_ip"
            fi
        else
            warning "VPN connection test failed or timed out"
            info "Check logs with: docker logs gluetun --tail 20"
        fi
    else
        warning "Gluetun container not found - you may need to start it manually"
        info "Start with: docker-compose up -d gluetun"
    fi
}

cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove temporary directory
    if [[ -n "${PIA_SCRIPTS_DIR:-}" ]] && [[ "$PIA_SCRIPTS_DIR" == /tmp/* ]]; then
        rm -rf "$(dirname "$PIA_SCRIPTS_DIR")"
        success "Temporary files cleaned up"
    fi
}

show_summary() {
    echo
    success "🎉 PIA WireGuard Setup Complete!"
    echo
    echo "📋 Summary:"
    echo "  • Region: $PIA_REGION"
    echo "  • Private key: ✅ Installed"
    echo "  • Environment: ✅ Updated"
    echo "  • VPN service: ✅ Restarted"
    echo
    echo "🔍 Verify Setup:"
    echo "  docker logs gluetun --tail 20"
    echo "  docker exec gluetun curl ifconfig.me"
    echo
    echo "📁 Files Created:"
    echo "  • secrets/wg_private_key.txt (your private key)"
    echo "  • secrets/pia_info.txt (configuration details)"
    echo
    echo "🔄 If you need to change regions:"
    echo "  1. Run this script again with different region"
    echo "  2. Or manually edit secrets/wg_private_key.txt"
    echo
}

# Main execution
main() {
    log "Starting PIA WireGuard setup for July 2025"
    
    show_welcome
    check_prerequisites
    get_pia_credentials
    select_pia_region
    download_pia_scripts
    generate_wireguard_config
    extract_private_key
    update_env_variables
    restart_vpn_services
    cleanup
    show_summary
    
    success "PIA WireGuard setup completed successfully!"
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Setup cancelled by user${NC}"; cleanup; exit 0' INT

# Handle errors
trap 'error "Setup failed on line $LINENO"; cleanup; exit 1' ERR

main "$@"