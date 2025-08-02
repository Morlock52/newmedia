#!/bin/bash

# HoloMedia Hub Update Script
# Version: 1.0.0
# Description: Automated update system with rollback capability

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# Update configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
UPDATE_LOG="$PROJECT_ROOT/logs/update.log"
BACKUP_BEFORE_UPDATE=${BACKUP_BEFORE_UPDATE:-true}
UPDATE_BRANCH=${UPDATE_BRANCH:-main}
ROLLBACK_DIR="$PROJECT_ROOT/.rollback"
VERSION_FILE="$PROJECT_ROOT/version.txt"
CURRENT_VERSION=""
NEW_VERSION=""

# Update sources
UPDATE_SOURCE=${UPDATE_SOURCE:-github}
GITHUB_REPO=${GITHUB_REPO:-"holomedia/holomedia-hub"}
UPDATE_URL=${UPDATE_URL:-"https://api.github.com/repos/$GITHUB_REPO/releases/latest"}

# Load environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# ASCII Art Banner
show_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
   __  __          __      __     
  / / / /___  ____/ /___ _/ /____ 
 / / / / __ \/ __  / __ `/ __/ _ \
/ /_/ / /_/ / /_/ / /_/ / /_/  __/
\____/ .___/\__,_/\__,_/\__/\___/ 
    /_/                           
EOF
    echo -e "${NC}"
}

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$UPDATE_LOG"
}

# Progress spinner
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Error handler
error_exit() {
    echo -e "\n${RED}Error: $1${NC}" >&2
    log "ERROR: $1"
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}✓ $1${NC}"
    log "SUCCESS: $1"
}

# Warning message
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    log "WARNING: $1"
}

# Info message
info() {
    echo -e "${BLUE}ℹ $1${NC}"
    log "INFO: $1"
}

# Get current version
get_current_version() {
    if [ -f "$VERSION_FILE" ]; then
        CURRENT_VERSION=$(cat "$VERSION_FILE")
    else
        # Try to get from package.json
        if [ -f "$PROJECT_ROOT/package.json" ]; then
            CURRENT_VERSION=$(jq -r '.version // "unknown"' "$PROJECT_ROOT/package.json")
        else
            CURRENT_VERSION="unknown"
        fi
    fi
    
    info "Current version: $CURRENT_VERSION"
}

# Check for updates
check_for_updates() {
    echo -e "\n${CYAN}Checking for updates...${NC}"
    
    case "$UPDATE_SOURCE" in
        github)
            check_github_updates
            ;;
        npm)
            check_npm_updates
            ;;
        custom)
            check_custom_updates
            ;;
        *)
            error_exit "Unknown update source: $UPDATE_SOURCE"
            ;;
    esac
}

# Check GitHub releases
check_github_updates() {
    if ! command -v curl &> /dev/null; then
        error_exit "curl is required for GitHub updates"
    fi
    
    # Get latest release info
    RELEASE_INFO=$(curl -s "$UPDATE_URL")
    
    if [ $? -ne 0 ]; then
        error_exit "Failed to check for updates"
    fi
    
    NEW_VERSION=$(echo "$RELEASE_INFO" | jq -r '.tag_name // .name // "unknown"' | sed 's/^v//')
    RELEASE_NOTES=$(echo "$RELEASE_INFO" | jq -r '.body // "No release notes available"')
    DOWNLOAD_URL=$(echo "$RELEASE_INFO" | jq -r '.tarball_url // ""')
    
    if [ "$NEW_VERSION" = "unknown" ] || [ -z "$NEW_VERSION" ]; then
        info "No updates found"
        exit 0
    fi
    
    # Compare versions
    if [ "$CURRENT_VERSION" = "$NEW_VERSION" ]; then
        success "Already running the latest version ($CURRENT_VERSION)"
        exit 0
    else
        echo -e "\n${GREEN}Update available!${NC}"
        echo -e "Current version: ${YELLOW}$CURRENT_VERSION${NC}"
        echo -e "New version: ${GREEN}$NEW_VERSION${NC}"
        echo -e "\n${CYAN}Release Notes:${NC}"
        echo "$RELEASE_NOTES" | head -20
        echo ""
    fi
}

# Check npm updates
check_npm_updates() {
    if ! command -v npm &> /dev/null; then
        error_exit "npm is required for npm updates"
    fi
    
    # Check outdated packages
    info "Checking npm packages..."
    npm outdated --json > /tmp/npm_updates.json
    
    if [ -s /tmp/npm_updates.json ]; then
        echo -e "\n${YELLOW}Package updates available:${NC}"
        jq -r 'to_entries[] | "  \(.key): \(.value.current) → \(.value.latest)"' /tmp/npm_updates.json
    else
        success "All npm packages are up to date"
    fi
}

# Pre-update checks
pre_update_checks() {
    echo -e "\n${CYAN}Running pre-update checks...${NC}"
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 2 ]; then
        error_exit "Insufficient disk space. At least 2GB required."
    fi
    success "Disk space check passed (${AVAILABLE_SPACE}GB available)"
    
    # Check if services are running
    if docker ps --format "{{.Names}}" | grep -q "holomedia"; then
        warning "HoloMedia services are running. They will be stopped during update."
    fi
    
    # Check for uncommitted changes
    if [ -d "$PROJECT_ROOT/.git" ]; then
        if [ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]; then
            warning "Uncommitted changes detected in the repository"
            read -p "Continue anyway? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    success "Pre-update checks completed"
}

# Create update backup
create_update_backup() {
    if [ "$BACKUP_BEFORE_UPDATE" != "true" ]; then
        warning "Skipping backup (not recommended)"
        return
    fi
    
    echo -e "\n${CYAN}Creating backup before update...${NC}"
    
    # Run backup script
    if [ -x "$SCRIPT_DIR/backup.sh" ]; then
        "$SCRIPT_DIR/backup.sh" --full
    else
        warning "Backup script not found. Creating minimal backup..."
        
        # Create rollback directory
        mkdir -p "$ROLLBACK_DIR"
        
        # Backup critical files
        cp -r "$PROJECT_ROOT/.env" "$ROLLBACK_DIR/" 2>/dev/null || true
        cp -r "$PROJECT_ROOT/package.json" "$ROLLBACK_DIR/" 2>/dev/null || true
        cp -r "$PROJECT_ROOT/docker-compose.yml" "$ROLLBACK_DIR/" 2>/dev/null || true
        
        # Save current version
        echo "$CURRENT_VERSION" > "$ROLLBACK_DIR/version.txt"
    fi
    
    success "Backup completed"
}

# Stop services
stop_services() {
    echo -e "\n${CYAN}Stopping services...${NC}"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$PROJECT_ROOT/docker/docker-compose.full.yml" down
    elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
        docker compose -f "$PROJECT_ROOT/docker/docker-compose.full.yml" down
    fi
    
    # Stop systemd service if exists
    if systemctl is-active --quiet holomedia.service; then
        sudo systemctl stop holomedia.service
    fi
    
    success "Services stopped"
}

# Download update
download_update() {
    echo -e "\n${CYAN}Downloading update...${NC}"
    
    case "$UPDATE_SOURCE" in
        github)
            download_github_update
            ;;
        git)
            download_git_update
            ;;
        *)
            error_exit "Unsupported update method"
            ;;
    esac
}

# Download from GitHub
download_github_update() {
    local temp_dir="/tmp/holomedia_update_$$"
    mkdir -p "$temp_dir"
    
    info "Downloading version $NEW_VERSION..."
    
    # Download release
    curl -L "$DOWNLOAD_URL" -o "$temp_dir/update.tar.gz" &
    spinner $!
    
    # Extract update
    info "Extracting update..."
    tar -xzf "$temp_dir/update.tar.gz" -C "$temp_dir"
    
    # Find extracted directory
    UPDATE_DIR=$(find "$temp_dir" -maxdepth 1 -type d -name "*holomedia*" | head -1)
    
    if [ -z "$UPDATE_DIR" ]; then
        error_exit "Failed to extract update"
    fi
    
    # Apply update
    apply_update "$UPDATE_DIR"
    
    # Cleanup
    rm -rf "$temp_dir"
}

# Download via git
download_git_update() {
    if [ ! -d "$PROJECT_ROOT/.git" ]; then
        error_exit "Not a git repository"
    fi
    
    info "Fetching latest changes..."
    git -C "$PROJECT_ROOT" fetch origin
    
    # Check if fast-forward is possible
    LOCAL=$(git -C "$PROJECT_ROOT" rev-parse @)
    REMOTE=$(git -C "$PROJECT_ROOT" rev-parse @{u})
    BASE=$(git -C "$PROJECT_ROOT" merge-base @ @{u})
    
    if [ "$LOCAL" = "$REMOTE" ]; then
        success "Already up to date"
        exit 0
    elif [ "$LOCAL" = "$BASE" ]; then
        info "Pulling latest changes..."
        git -C "$PROJECT_ROOT" pull origin "$UPDATE_BRANCH"
        NEW_VERSION=$(git -C "$PROJECT_ROOT" describe --tags --abbrev=0 2>/dev/null || echo "git-$(git rev-parse --short HEAD)")
    else
        error_exit "Local changes conflict with remote. Please resolve manually."
    fi
}

# Apply update
apply_update() {
    local update_dir=$1
    
    echo -e "\n${CYAN}Applying update...${NC}"
    
    # Preserve configuration files
    local preserve_files=(".env" "config/local.json" "ssl/*")
    
    for file in "${preserve_files[@]}"; do
        if [ -e "$PROJECT_ROOT/$file" ]; then
            cp -r "$PROJECT_ROOT/$file" "/tmp/holomedia_preserve_$$_$(basename $file)"
        fi
    done
    
    # Copy update files
    rsync -av --exclude='.git' --exclude='node_modules' --exclude='data' \
        --exclude='logs' --exclude='uploads' --exclude='.env' \
        "$update_dir/" "$PROJECT_ROOT/"
    
    # Restore preserved files
    for file in "${preserve_files[@]}"; do
        local preserved="/tmp/holomedia_preserve_$$_$(basename $file)"
        if [ -e "$preserved" ]; then
            cp -r "$preserved" "$PROJECT_ROOT/$file"
            rm -rf "$preserved"
        fi
    done
    
    success "Update files applied"
}

# Update dependencies
update_dependencies() {
    echo -e "\n${CYAN}Updating dependencies...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Update npm packages
    if [ -f "package.json" ]; then
        info "Installing npm dependencies..."
        npm install --production &
        spinner $!
        success "npm dependencies updated"
    fi
    
    # Update Docker images
    if [ -f "docker/docker-compose.full.yml" ]; then
        info "Pulling latest Docker images..."
        docker-compose -f docker/docker-compose.full.yml pull &
        spinner $!
        success "Docker images updated"
    fi
}

# Run migrations
run_migrations() {
    echo -e "\n${CYAN}Running database migrations...${NC}"
    
    # Check if migrations exist
    if [ -d "$PROJECT_ROOT/migrations" ] || [ -f "$PROJECT_ROOT/migrate.js" ]; then
        info "Applying database migrations..."
        
        # Run migrations based on your setup
        if [ -f "$PROJECT_ROOT/migrate.js" ]; then
            node "$PROJECT_ROOT/migrate.js" up
        elif command -v npm &> /dev/null && grep -q "migrate" "$PROJECT_ROOT/package.json"; then
            npm run migrate
        fi
        
        success "Migrations completed"
    else
        info "No migrations to run"
    fi
}

# Post-update tasks
post_update_tasks() {
    echo -e "\n${CYAN}Running post-update tasks...${NC}"
    
    # Clear caches
    if [ -d "$PROJECT_ROOT/cache" ]; then
        rm -rf "$PROJECT_ROOT/cache/*"
        info "Cache cleared"
    fi
    
    # Rebuild assets if needed
    if [ -f "$PROJECT_ROOT/package.json" ] && grep -q "build" "$PROJECT_ROOT/package.json"; then
        info "Building assets..."
        npm run build &
        spinner $!
        success "Assets built"
    fi
    
    # Update version file
    echo "$NEW_VERSION" > "$VERSION_FILE"
    
    # Set correct permissions
    chmod 600 "$PROJECT_ROOT/.env" 2>/dev/null || true
    chmod +x "$PROJECT_ROOT/scripts/*.sh" 2>/dev/null || true
    
    success "Post-update tasks completed"
}

# Start services
start_services() {
    echo -e "\n${CYAN}Starting services...${NC}"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$PROJECT_ROOT/docker/docker-compose.full.yml" up -d
    elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
        docker compose -f "$PROJECT_ROOT/docker/docker-compose.full.yml" up -d
    fi
    
    # Start systemd service if exists
    if systemctl is-enabled --quiet holomedia.service; then
        sudo systemctl start holomedia.service
    fi
    
    # Wait for services to be ready
    info "Waiting for services to be ready..."
    sleep 10
    
    # Run health check
    if [ -x "$SCRIPT_DIR/health-check.sh" ]; then
        "$SCRIPT_DIR/health-check.sh"
    fi
    
    success "Services started"
}

# Rollback function
rollback() {
    echo -e "\n${RED}Rolling back update...${NC}"
    
    if [ ! -d "$ROLLBACK_DIR" ]; then
        error_exit "No rollback data found"
    fi
    
    # Stop services
    stop_services
    
    # Restore files
    if [ -f "$ROLLBACK_DIR/.env" ]; then
        cp "$ROLLBACK_DIR/.env" "$PROJECT_ROOT/.env"
    fi
    
    if [ -f "$ROLLBACK_DIR/package.json" ]; then
        cp "$ROLLBACK_DIR/package.json" "$PROJECT_ROOT/package.json"
        npm install --production
    fi
    
    # Restore version
    if [ -f "$ROLLBACK_DIR/version.txt" ]; then
        cp "$ROLLBACK_DIR/version.txt" "$VERSION_FILE"
    fi
    
    # Start services
    start_services
    
    success "Rollback completed"
    
    # Clean rollback directory
    rm -rf "$ROLLBACK_DIR"
}

# Auto-update setup
setup_auto_update() {
    echo -e "\n${CYAN}Setting up automatic updates...${NC}"
    
    # Create systemd timer for auto-updates
    cat > /tmp/holomedia-update.service << EOF
[Unit]
Description=HoloMedia Hub Auto Update
After=network.target

[Service]
Type=oneshot
ExecStart=$SCRIPT_DIR/update.sh --auto
User=$(whoami)
StandardOutput=journal
StandardError=journal
EOF

    cat > /tmp/holomedia-update.timer << EOF
[Unit]
Description=HoloMedia Hub Auto Update Timer
Requires=holomedia-update.service

[Timer]
OnCalendar=weekly
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Install systemd units
    if command -v systemctl &> /dev/null; then
        sudo cp /tmp/holomedia-update.{service,timer} /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable holomedia-update.timer
        sudo systemctl start holomedia-update.timer
        
        success "Automatic updates enabled (weekly)"
        info "To disable: sudo systemctl disable holomedia-update.timer"
    else
        warning "systemd not available. Cannot setup automatic updates."
    fi
    
    # Cleanup
    rm -f /tmp/holomedia-update.{service,timer}
}

# Check update status
check_update_status() {
    echo -e "${CYAN}Update Status${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    get_current_version
    
    # Check last update
    if [ -f "$UPDATE_LOG" ]; then
        LAST_UPDATE=$(grep "Update completed" "$UPDATE_LOG" | tail -1 | cut -d']' -f1 | sed 's/\[//')
        if [ -n "$LAST_UPDATE" ]; then
            echo "Last update: $LAST_UPDATE"
        fi
    fi
    
    # Check auto-update status
    if command -v systemctl &> /dev/null && systemctl is-enabled --quiet holomedia-update.timer; then
        echo -e "\nAutomatic updates: ${GREEN}Enabled${NC}"
        systemctl status holomedia-update.timer --no-pager | grep -E "Trigger:|Active:"
    else
        echo -e "\nAutomatic updates: ${YELLOW}Disabled${NC}"
    fi
    
    # Check for available updates
    check_for_updates
}

# Show help
show_help() {
    echo "HoloMedia Hub Update Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo "  --check             Check for updates only"
    echo "  --auto              Run update automatically (no prompts)"
    echo "  --rollback          Rollback to previous version"
    echo "  --status            Show update status"
    echo "  --setup-auto        Setup automatic updates"
    echo "  --force             Force update even if already up to date"
    echo "  --branch BRANCH     Update from specific branch (default: main)"
    echo "  --no-backup         Skip backup before update (not recommended)"
    echo ""
    echo "Environment Variables:"
    echo "  UPDATE_SOURCE       Update source (github/git/npm)"
    echo "  GITHUB_REPO         GitHub repository (owner/repo)"
    echo "  UPDATE_BRANCH       Git branch to update from"
}

# Main update flow
main() {
    local auto_mode=false
    local check_only=false
    local force_update=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                exit 0
                ;;
            --check)
                check_only=true
                shift
                ;;
            --auto)
                auto_mode=true
                shift
                ;;
            --rollback)
                rollback
                exit 0
                ;;
            --status)
                check_update_status
                exit 0
                ;;
            --setup-auto)
                setup_auto_update
                exit 0
                ;;
            --force)
                force_update=true
                shift
                ;;
            --branch)
                UPDATE_BRANCH="$2"
                shift 2
                ;;
            --no-backup)
                BACKUP_BEFORE_UPDATE=false
                shift
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    show_banner
    
    # Initialize
    mkdir -p "$(dirname "$UPDATE_LOG")"
    log "Update process started"
    
    # Get current version
    get_current_version
    
    # Check for updates
    check_for_updates
    
    if [ "$check_only" = true ]; then
        exit 0
    fi
    
    # Confirm update
    if [ "$auto_mode" != true ] && [ "$force_update" != true ]; then
        echo -e "\n${YELLOW}This will update HoloMedia Hub to version $NEW_VERSION${NC}"
        read -p "Do you want to continue? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Update cancelled"
            exit 0
        fi
    fi
    
    # Run update process
    pre_update_checks
    create_update_backup
    stop_services
    download_update
    update_dependencies
    run_migrations
    post_update_tasks
    start_services
    
    # Summary
    echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Update completed successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}\n"
    
    echo -e "${CYAN}Update Summary:${NC}"
    echo "• Previous version: $CURRENT_VERSION"
    echo "• New version: $NEW_VERSION"
    echo "• Update log: $UPDATE_LOG"
    
    if [ -d "$ROLLBACK_DIR" ]; then
        echo -e "\n${YELLOW}Rollback data saved. To rollback:${NC}"
        echo "  $0 --rollback"
    fi
    
    log "Update completed: $CURRENT_VERSION → $NEW_VERSION"
}

# Run main function
main "$@"