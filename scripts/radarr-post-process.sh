#!/bin/bash
# Radarr Post-Processing Webhook Script
# Handles movie download completion events

set -euo pipefail

# Configuration
JELLYFIN_API_URL="${JELLYFIN_URL:-http://jellyfin:8096}/Library/Refresh"
JELLYFIN_API_KEY="${JELLYFIN_API_KEY:-}"
LOG_FILE="/var/log/webhooks/radarr-post-process.log"

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to send notification
send_notification() {
    local title="$1"
    local message="$2"
    local priority="${3:-normal}"
    
    # Send to Grafana webhook (optional)
    if [[ -n "${GRAFANA_WEBHOOK_URL:-}" ]]; then
        curl -s -X POST "$GRAFANA_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"title\":\"$title\",\"message\":\"$message\",\"priority\":\"$priority\"}" || true
    fi
    
    # Log notification
    log "NOTIFICATION: [$priority] $title - $message"
}

# Function to refresh Jellyfin library
refresh_jellyfin() {
    if [[ -n "$JELLYFIN_API_KEY" ]]; then
        log "Refreshing Jellyfin movie library..."
        curl -s -X POST "$JELLYFIN_API_URL" \
            -H "X-Emby-Token: $JELLYFIN_API_KEY" \
            -H "Content-Type: application/json" || {
            log "ERROR: Failed to refresh Jellyfin library"
            return 1
        }
        log "Jellyfin library refresh triggered successfully"
    else
        log "WARNING: JELLYFIN_API_KEY not set, skipping library refresh"
    fi
}

# Function to trigger FileFlows processing
trigger_fileflows() {
    local file_path="$1"
    
    if [[ -n "${FILEFLOWS_API_URL:-}" && -n "${FILEFLOWS_API_KEY:-}" ]]; then
        log "Triggering FileFlows processing for: $file_path"
        curl -s -X POST "${FILEFLOWS_API_URL}/api/flow/process" \
            -H "Authorization: Bearer $FILEFLOWS_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"file\":\"$file_path\"}" || {
            log "ERROR: Failed to trigger FileFlows processing"
            return 1
        }
        log "FileFlows processing triggered successfully"
    else
        log "INFO: FileFlows not configured, skipping processing"
    fi
}

# Function to update movie metadata
update_movie_metadata() {
    local movie_title="$1"
    local file_path="$2"
    
    # Extract movie information
    local year
    year=$(echo "$movie_title" | grep -oE '\([0-9]{4}\)' | tr -d '()' || echo "")
    
    log "Updating metadata for: $movie_title ($year)"
    
    # Trigger Radarr to refresh movie metadata
    if [[ -n "${RADARR_API_KEY:-}" ]]; then
        curl -s -X POST "${RADARR_URL:-http://radarr:7878}/api/v3/command" \
            -H "X-Api-Key: $RADARR_API_KEY" \
            -H "Content-Type: application/json" \
            -d '{"name":"RefreshMovie"}' || {
            log "WARNING: Failed to trigger Radarr metadata refresh"
        }
    fi
}

# Main processing function
main() {
    local event_type="${1:-unknown}"
    local movie_title="${2:-unknown}"
    local file_path="${3:-unknown}"
    
    log "=== Radarr Post-Process Started ==="
    log "Event Type: $event_type"
    log "Movie: $movie_title"
    log "File: $file_path"
    
    case "$event_type" in
        "Download")
            log "Processing movie download completion..."
            
            # Send success notification
            send_notification \
                "Movie Downloaded" \
                "Successfully downloaded: $movie_title" \
                "normal"
            
            # Wait a moment for file to be fully written
            sleep 5
            
            # Update movie metadata
            update_movie_metadata "$movie_title" "$file_path"
            
            # Refresh Jellyfin library
            refresh_jellyfin
            
            # Trigger post-processing if configured
            if [[ "$file_path" != "unknown" && -f "$file_path" ]]; then
                trigger_fileflows "$file_path"
            fi
            ;;
            
        "Test")
            log "Test webhook received"
            send_notification \
                "Radarr Webhook Test" \
                "Radarr webhook is working correctly" \
                "low"
            ;;
            
        *)
            log "Unknown event type: $event_type"
            ;;
    esac
    
    log "=== Radarr Post-Process Completed ==="
    
    # Return JSON response
    echo '{"status":"success","message":"Post-processing completed"}'
}

# Error handling
trap 'log "ERROR: Script failed at line $LINENO"' ERR

# Execute main function with all arguments
main "$@"