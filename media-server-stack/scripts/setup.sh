#!/usr/bin/env bash
set -euo pipefail

# Smart setup script that automatically chooses the best available setup method
# This is the main entry point for new users - replaces the old basic setup

# Determine project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info() {
    echo -e "${BLUE}ℹ️  $1${NC}" >&2
}

success() {
    echo -e "${GREEN}✅ $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" >&2
}

# Show available setup options
show_setup_options() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                    🎬 Media Server Stack Setup                      ║"
    echo "║                         Choose Setup Method                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo
    echo "Available setup methods:"
    echo
    echo "1. 🎨 Modern GUI Setup (Recommended)"
    echo "   └─ Interactive dialogs with validation and smart defaults"
    echo
    echo "2. 📝 Text-Based Interactive Setup"
    echo "   └─ Terminal prompts with the same features as GUI mode"
    echo
    echo "3. ⚡ Quick Manual Setup"
    echo "   └─ Copy template and edit .env file manually"
    echo
    echo "4. 🔄 Automatic Detection"
    echo "   └─ Let the script choose the best available method"
    echo
}

# Check for existing configuration
check_existing_config() {
    if [[ -f "$ROOT/.env" ]]; then
        warning "Existing configuration found!"
        echo "This will update your current settings."
        echo
        read -p "Continue? [y/N]: " -r response
        case "$response" in
            [Yy]|[Yy][Ee][Ss]) ;;
            *) echo "Setup cancelled."; exit 0 ;;
        esac
        echo
    fi
}

# Auto-detect best setup method
auto_detect_setup() {
    info "Auto-detecting best setup method..."
    
    # Check for GUI capabilities
    if command -v whiptail >/dev/null 2>&1; then
        success "Found whiptail - using modern GUI setup"
        exec "$ROOT/scripts/setup-gui.sh" "$@"
    elif command -v dialog >/dev/null 2>&1; then
        success "Found dialog - using modern GUI setup"
        exec "$ROOT/scripts/setup-gui.sh" "$@"
    elif command -v zenity >/dev/null 2>&1; then
        success "Found zenity - using modern GUI setup"
        exec "$ROOT/scripts/setup-gui.sh" "$@"
    else
        success "Using interactive text setup"
        exec "$ROOT/scripts/setup-interactive.sh" "$@"
    fi
}

# Manual setup option
manual_setup() {
    info "Setting up manual configuration..."
    
    if [[ ! -f "$ROOT/.env" ]]; then
        cp "$ROOT/.env.example" "$ROOT/.env"
        success "Copied .env.example to .env"
    else
        warning ".env file already exists"
    fi
    
    echo
    echo "Manual setup completed! Next steps:"
    echo
    echo "1. Edit the .env file with your settings:"
    echo "   nano .env  # or use your preferred editor"
    echo
    echo "2. Key settings to configure:"
    echo "   • DOMAIN=yourdomain.com"
    echo "   • EMAIL=admin@yourdomain.com"  
    echo "   • VPN_PROVIDER=your_vpn_provider"
    echo
    echo "3. Add your VPN key:"
    echo "   echo \"YOUR_KEY\" > secrets/wg_private_key.txt"
    echo
    echo "4. Deploy the stack:"
    echo "   ./scripts/deploy.sh"
    echo
}

# Main setup logic
main() {
    clear
    
    # Handle command line arguments
    case "${1:-}" in
        --gui|gui)
            check_existing_config
            exec "$ROOT/scripts/setup-gui.sh" "${@:2}"
            ;;
        --interactive|interactive)
            check_existing_config
            exec "$ROOT/scripts/setup-interactive.sh" "${@:2}"
            ;;
        --manual|manual)
            check_existing_config
            manual_setup
            exit 0
            ;;
        --auto|auto|"")
            check_existing_config
            auto_detect_setup "${@:2}"
            ;;
        --help|-h|help)
            echo "Media Server Stack Setup"
            echo
            echo "Usage: $0 [METHOD]"
            echo
            echo "Methods:"
            echo "  gui          Use GUI setup (whiptail/dialog/zenity)"
            echo "  interactive  Use text-based interactive setup"
            echo "  manual       Copy template for manual editing"
            echo "  auto         Auto-detect best method (default)"
            echo "  help         Show this help"
            echo
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 help' for available options."
            exit 1
            ;;
    esac
}

main "$@"