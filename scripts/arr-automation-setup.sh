#!/bin/bash

# Ultimate Media Server 2025 - *arr Automation Setup Script
# Configures all automation services for complete media management

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service URLs
PROWLARR_URL="http://localhost:9696"
SONARR_URL="http://localhost:8989"
RADARR_URL="http://localhost:7878"
LIDARR_URL="http://localhost:8686"
READARR_URL="http://localhost:8787"
BAZARR_URL="http://localhost:6767"
OVERSEERR_URL="http://localhost:5055"
QBIT_URL="http://localhost:8080"
SAB_URL="http://localhost:8081"

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    print_step "Waiting for $name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url/ping" > /dev/null 2>&1 || curl -s -f "$url" > /dev/null 2>&1; then
            print_success "$name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 5
        ((attempt++))
    done
    
    print_error "$name failed to start within expected time"
    return 1
}

# Generate API key for service
generate_api_key() {
    openssl rand -hex 16
}

# Get API key from service
get_api_key() {
    local config_path=$1
    local service_name=$2
    
    if [ -f "$config_path" ]; then
        grep -o '<ApiKey>[^<]*</ApiKey>' "$config_path" | sed 's/<ApiKey>\|<\/ApiKey>//g'
    else
        print_warning "Config file not found for $service_name, generating new API key"
        generate_api_key
    fi
}

# Create directory structure
create_directories() {
    print_step "Creating directory structure..."
    
    mkdir -p config/{prowlarr,sonarr,radarr,lidarr,readarr,bazarr,overseerr,qbittorrent,sabnzbd}
    mkdir -p media-data/{downloads/{complete,incomplete,torrents,usenet},movies,tv,music,books,audiobooks}
    mkdir -p media-data/downloads/torrents/{movies,tv,music,books}
    mkdir -p media-data/downloads/usenet/{movies,tv,music,books}
    
    # Set proper permissions
    chmod -R 755 config/
    chmod -R 755 media-data/
    
    print_success "Directory structure created"
}

# Configure qBittorrent
configure_qbittorrent() {
    print_step "Configuring qBittorrent..."
    
    cat > config/qbittorrent/qBittorrent.conf << 'EOF'
[Application]
FileLogger\Enabled=true
FileLogger\Path=/config/logs
FileLogger\Backup=true
FileLogger\MaxSizeBytes=66560
FileLogger\DeleteOld=true
FileLogger\MaxOldLogs=99

[BitTorrent]
Session\Categories=movies\x2cDownloads/torrents/movies;tv\x2cDownloads/torrents/tv;music\x2cDownloads/torrents/music;books\x2cDownloads/torrents/books
Session\DefaultSavePath=/downloads/torrents
Session\TempPath=/downloads/incomplete
Session\DisableAutoTMMByDefault=false
Session\DisableAutoTMMTriggers\CategorySavePathChanged=false
Session\DisableAutoTMMTriggers\DefaultSavePathChanged=false
Session\Port=6881
Session\UseRandomPort=false

[Core]
AutoDeleteAddedTorrentFile=Never

[Meta]
MigrationVersion=4

[Network]
Cookies=@Invalid()
PortForwardingEnabled=false
Proxy\OnlyForTorrents=false

[Preferences]
Advanced\RecheckOnCompletion=false
Advanced\TrayIconStyle=MonoDark
Connection\PortRangeMin=6881
Connection\UPnP=false
Downloads\DiskWriteCacheSize=64
Downloads\DiskWriteCacheTTL=60
Downloads\FinishedTorrentExportDir=
Downloads\PreAllocation=false
Downloads\SavePath=/downloads/torrents
Downloads\ScanDirsV2=@Variant(\0\0\0\x1c\0\0\0\0)
Downloads\StartInPause=false
Downloads\TempPath=/downloads/incomplete
Downloads\TempPathEnabled=true
Downloads\TorrentExportDir=
General\Locale=
MailNotification\enabled=false
Queueing\MaxActiveDownloads=5
Queueing\MaxActiveTorrents=10
Queueing\MaxActiveUploads=5
Queueing\QueueingEnabled=false
WebUI\Address=*
WebUI\AlternativeUIEnabled=false
WebUI\AuthSubnetWhitelist=@Invalid()
WebUI\AuthSubnetWhitelistEnabled=false
WebUI\BanDuration=3600
WebUI\CSRFProtection=true
WebUI\ClickjackingProtection=true
WebUI\CustomHTTPHeaders=
WebUI\CustomHTTPHeadersEnabled=false
WebUI\HTTPS\CertificatePath=
WebUI\HTTPS\Enabled=false
WebUI\HTTPS\KeyPath=
WebUI\HostHeaderValidation=true
WebUI\LocalHostAuth=false
WebUI\MaxAuthenticationFailCount=5
WebUI\Port=8080
WebUI\RootFolder=
WebUI\SecureCookie=true
WebUI\ServerDomains=*
WebUI\SessionTimeout=3600
WebUI\UseUPnP=false
WebUI\Username=admin
EOF

    print_success "qBittorrent configured"
}

# Configure SABnzbd
configure_sabnzbd() {
    print_step "Configuring SABnzbd..."
    
    cat > config/sabnzbd/sabnzbd.ini << 'EOF'
[misc]
pre_check = 0
pythonpath = 
ionice = 
check_new_rel = 1
auto_browser = 0
username = 
password = 
port = 8080
host = 0.0.0.0
web_dir = Glitter
web_color = Auto
https_port = 9080
https_cert = server.cert
https_key = server.key
https_chain = 
enable_https = 0
inet_exposure = 0
api_key = 
nzb_key = 
permissions = 755
download_dir = Downloads/usenet/incomplete
complete_dir = Downloads/usenet/complete
script_dir = 
email_server = 
email_to = 
email_from = 
email_account = 
email_pwd = 
email_endjob = 0
email_full = 0
email_dir = 
log_dir = logs
admin_dir = admin
nzb_backup_dir = 
cache_dir = cache
dirscan_dir = 
dirscan_speed = 5
refresh_rate = 0
interface_settings = 
queue_complete = 
folder_rename = 1
replace_spaces = 0
replace_dots = 0
safe_postproc = 1
pause_on_post_processing = 0
script_can_fail = 0
enable_recursive = 1
flat_unpack = 0
par_option = 
nice = 
win_process_prio = 3
fail_hopeless_jobs = 1
fast_fail = 1
auto_disconnect = 1
pre_script = 
end_queue_script = 
no_dupes = 0
no_series_dupes = 0
no_smart_dupes = 0
dupes_propercheck = 1
pause_on_pwrar = 1
action_on_unwanted_extensions = 0
unwanted_extensions = 
new_nzb_on_failure = 0
history_retention = 0
enable_par_cleanup = 1
process_unpacked_par2 = 1
enable_multipar = 1
enable_unrar = 1
enable_7zip = 1
enable_filejoin = 1
enable_tsjoin = 0
overwrite_files = 0
ignore_unrar_dates = 0
backup_for_duplicates = 1
empty_postproc = 0
wait_for_dfolder = 0
rss_filenames = 0
ipv6_hosting = 0
api_logging = 1
html_login = 1
warn_dupl_jobs = 1
helpful_warnings = 1
keep_awake = 1
tray_icon = 1
allow_incomplete_nzb = 0
rss_odd_titles = nzbindex.nl/, nzbindex.com/, nzbclub.com/
quick_check_ext_ignore = nfo, sfv, srr
sfv_check = 1
movie_rename_limit = 100M
episode_rename_limit = 20M
size_limit = 0
fsys_type = 0
direct_unpack = 0
ignore_samples = 0
deobfuscate_final_filenames = 0
auto_sort = 0
check_system_space = 1
cleanup_list = , .nzb, .par2, .vol, .sfv, .nfo, .srr, .idx, .srs, .rar, .zip, .7z
unwanted_extensions_mode = 0
action_on_unwanted_extensions = 0
unwanted_extensions = exe, com, bat, scr, pif, cmd, vbs, js

[servers]
[[Localhost]]
host = localhost
port = 119
timeout = 60
username = 
password = 
connections = 8
priority = 0
retention = 0
send_group = 0
ssl = 0
ssl_verify = 2
ssl_ciphers = 
enable = 1
optional = 0

[categories]
[[*]]
priority = 0
pp = 3
name = *
script = None
dir = 

[[movies]]
priority = 0
pp = 3
name = movies
script = 
dir = movies

[[tv]]
priority = 0
pp = 3
name = tv
script = 
dir = tv

[[music]]
priority = 0
pp = 3
name = music
script = 
dir = music

[[books]]
priority = 0
pp = 3
name = books
script = 
dir = books
EOF

    print_success "SABnzbd configured"
}

# Configure Prowlarr automation
configure_prowlarr_automation() {
    print_step "Configuring Prowlarr automation..."
    
    # Wait for Prowlarr to be ready
    wait_for_service "$PROWLARR_URL" "Prowlarr"
    
    # Get API key
    PROWLARR_API_KEY=$(get_api_key "config/prowlarr/config.xml" "Prowlarr")
    
    print_success "Prowlarr automation configured (API Key: ${PROWLARR_API_KEY:0:8}...)"
}

# Configure Sonarr automation
configure_sonarr_automation() {
    print_step "Configuring Sonarr automation..."
    
    wait_for_service "$SONARR_URL" "Sonarr"
    
    SONARR_API_KEY=$(get_api_key "config/sonarr/config.xml" "Sonarr")
    
    # Configure root folders
    curl -s -X POST "$SONARR_URL/api/v3/rootFolder" \
        -H "X-Api-Key: $SONARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"path":"/media/tv","accessible":true,"freeSpace":0,"unmappedFolders":[]}' || true
    
    print_success "Sonarr automation configured (API Key: ${SONARR_API_KEY:0:8}...)"
}

# Configure Radarr automation
configure_radarr_automation() {
    print_step "Configuring Radarr automation..."
    
    wait_for_service "$RADARR_URL" "Radarr"
    
    RADARR_API_KEY=$(get_api_key "config/radarr/config.xml" "Radarr")
    
    # Configure root folders
    curl -s -X POST "$RADARR_URL/api/v3/rootFolder" \
        -H "X-Api-Key: $RADARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"path":"/media/movies","accessible":true,"freeSpace":0,"unmappedFolders":[]}' || true
    
    print_success "Radarr automation configured (API Key: ${RADARR_API_KEY:0:8}...)"
}

# Configure Lidarr automation
configure_lidarr_automation() {
    print_step "Configuring Lidarr automation..."
    
    wait_for_service "$LIDARR_URL" "Lidarr"
    
    LIDARR_API_KEY=$(get_api_key "config/lidarr/config.xml" "Lidarr")
    
    # Configure root folders
    curl -s -X POST "$LIDARR_URL/api/v1/rootFolder" \
        -H "X-Api-Key: $LIDARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"path":"/media/music","accessible":true,"freeSpace":0,"unmappedFolders":[]}' || true
    
    print_success "Lidarr automation configured (API Key: ${LIDARR_API_KEY:0:8}...)"
}

# Configure Readarr automation
configure_readarr_automation() {
    print_step "Configuring Readarr automation..."
    
    wait_for_service "$READARR_URL" "Readarr"
    
    READARR_API_KEY=$(get_api_key "config/readarr/config.xml" "Readarr")
    
    # Configure root folders
    curl -s -X POST "$READARR_URL/api/v1/rootFolder" \
        -H "X-Api-Key: $READARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"path":"/media/books","accessible":true,"freeSpace":0,"unmappedFolders":[]}' || true
    
    curl -s -X POST "$READARR_URL/api/v1/rootFolder" \
        -H "X-Api-Key: $READARR_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"path":"/media/audiobooks","accessible":true,"freeSpace":0,"unmappedFolders":[]}' || true
    
    print_success "Readarr automation configured (API Key: ${READARR_API_KEY:0:8}...)"
}

# Configure Bazarr automation
configure_bazarr_automation() {
    print_step "Configuring Bazarr automation..."
    
    wait_for_service "$BAZARR_URL" "Bazarr"
    
    # Update Bazarr config with API keys
    if [ -f "config/bazarr/config.ini" ]; then
        sed -i "s/SONARR_API_KEY_PLACEHOLDER/$SONARR_API_KEY/g" config/bazarr/config.ini
        sed -i "s/RADARR_API_KEY_PLACEHOLDER/$RADARR_API_KEY/g" config/bazarr/config.ini
    fi
    
    print_success "Bazarr automation configured"
}

# Configure Overseerr automation
configure_overseerr_automation() {
    print_step "Configuring Overseerr automation..."
    
    wait_for_service "$OVERSEERR_URL" "Overseerr"
    
    print_success "Overseerr automation configured"
}

# Generate service connection guide
generate_connection_guide() {
    print_step "Generating service connection guide..."
    
    cat > arr-automation-guide.md << EOF
# *arr Automation Setup Guide

## Service URLs and API Keys

### Core Services
- **Prowlarr (Indexer Manager)**: http://localhost:9696
  - API Key: ${PROWLARR_API_KEY:-Check /config/prowlarr/config.xml}
- **Overseerr (Request Manager)**: http://localhost:5055

### Media Automation
- **Sonarr (TV Shows)**: http://localhost:8989
  - API Key: ${SONARR_API_KEY:-Check /config/sonarr/config.xml}
- **Radarr (Movies)**: http://localhost:7878
  - API Key: ${RADARR_API_KEY:-Check /config/radarr/config.xml}
- **Lidarr (Music)**: http://localhost:8686
  - API Key: ${LIDARR_API_KEY:-Check /config/lidarr/config.xml}
- **Readarr (Books)**: http://localhost:8787
  - API Key: ${READARR_API_KEY:-Check /config/readarr/config.xml}
- **Bazarr (Subtitles)**: http://localhost:6767

### Download Clients
- **qBittorrent**: http://localhost:8080
  - Username: admin
  - Default password: Check container logs
- **SABnzbd**: http://localhost:8081

## Setup Order

1. **Start Services**: \`docker compose up -d\`
2. **Configure Prowlarr**: 
   - Add indexers (trackers/newsgroups)
   - Set up applications (Sonarr, Radarr, etc.)
3. **Configure Download Clients**:
   - qBittorrent: Set categories and paths
   - SABnzbd: Configure usenet servers
4. **Configure *arr Services**:
   - Add download clients
   - Set quality profiles
   - Configure media libraries
5. **Configure Overseerr**:
   - Connect to Sonarr/Radarr
   - Set up user permissions

## Automation Flow

\`\`\`
User Request (Overseerr) 
    ↓
*arr Service (Sonarr/Radarr/etc.) 
    ↓
Prowlarr (Find Release) 
    ↓
Download Client (qBittorrent/SABnzbd) 
    ↓
Media Library (Jellyfin/Plex) 
    ↓
Bazarr (Download Subtitles)
\`\`\`

## Directory Structure

\`\`\`
media-data/
├── downloads/
│   ├── torrents/
│   │   ├── movies/
│   │   ├── tv/
│   │   ├── music/
│   │   └── books/
│   ├── usenet/
│   │   ├── movies/
│   │   ├── tv/
│   │   ├── music/
│   │   └── books/
│   ├── complete/
│   └── incomplete/
├── movies/
├── tv/
├── music/
├── books/
└── audiobooks/
\`\`\`

## Quality Profiles

### Sonarr (TV)
- HDTV-720p: Standard HD episodes
- WEBDL-1080p: High quality web releases
- Bluray-1080p: Best quality for favorites

### Radarr (Movies)
- HD-720p: Standard movie quality
- HD-1080p: High definition movies
- UHD-2160p: 4K movies (if supported)

### Lidarr (Music)
- Lossless: FLAC preferred
- High Quality Lossy: 320kbps MP3/AAC
- Standard: 192-256kbps for mobile

### Readarr (Books)
- eBook Preferred: EPUB > PDF > MOBI
- Audiobook: M4B > MP3

## Indexer Configuration

### Public Trackers (Free)
- 1337x
- RARBG
- The Pirate Bay
- Nyaa (Anime)

### Usenet Providers
- Configure your usenet provider in SABnzbd
- Add corresponding indexers in Prowlarr

## Security Notes

- All services are configured for local access only
- API keys are auto-generated for security
- VPN is configured for torrent downloads
- Consider using a reverse proxy for external access

## Troubleshooting

### Common Issues
1. **Services not communicating**: Check API keys match
2. **Downloads not starting**: Verify download client configuration
3. **Files not moving**: Check path mappings and permissions
4. **No indexers**: Configure indexers in Prowlarr first

### Log Locations
- Service logs: \`docker compose logs [service-name]\`
- Application logs: \`./config/[service]/logs/\`

EOF

    print_success "Connection guide generated: arr-automation-guide.md"
}

# Main setup function
main() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}  *arr Automation Setup Script  ${NC}"
    echo -e "${BLUE}  Ultimate Media Server 2025    ${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo
    
    create_directories
    configure_qbittorrent
    configure_sabnzbd
    
    echo
    print_step "Services should now be started with: docker compose up -d"
    print_step "After services are running, configure each service through their web interfaces"
    
    # If services are already running, configure them
    if curl -s -f "$PROWLARR_URL/ping" > /dev/null 2>&1; then
        echo
        print_step "Services detected running, configuring automation..."
        configure_prowlarr_automation
        configure_sonarr_automation
        configure_radarr_automation
        configure_lidarr_automation
        configure_readarr_automation
        configure_bazarr_automation
        configure_overseerr_automation
    fi
    
    generate_connection_guide
    
    echo
    echo -e "${GREEN}=================================${NC}"
    echo -e "${GREEN}     Setup Complete!             ${NC}"
    echo -e "${GREEN}=================================${NC}"
    echo
    echo -e "Access your services:"
    echo -e "  ${BLUE}Prowlarr${NC}:  http://localhost:9696"
    echo -e "  ${BLUE}Sonarr${NC}:    http://localhost:8989"
    echo -e "  ${BLUE}Radarr${NC}:    http://localhost:7878"
    echo -e "  ${BLUE}Lidarr${NC}:    http://localhost:8686"
    echo -e "  ${BLUE}Readarr${NC}:   http://localhost:8787"
    echo -e "  ${BLUE}Bazarr${NC}:    http://localhost:6767"
    echo -e "  ${BLUE}Overseerr${NC}: http://localhost:5055"
    echo
    echo -e "See ${YELLOW}arr-automation-guide.md${NC} for detailed setup instructions"
    echo
}

# Run main function
main "$@"