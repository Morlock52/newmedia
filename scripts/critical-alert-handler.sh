#!/bin/bash
# Critical Alert Handler Script
# Responds to critical system alerts with automated recovery actions

set -euo pipefail

# Configuration
LOG_FILE="/var/log/webhooks/critical-alerts.log"
MAX_RESTART_ATTEMPTS=3
RESTART_COOLDOWN=300  # 5 minutes

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Function to send emergency notification
send_emergency_notification() {
    local title="$1"
    local message="$2"
    
    # Email notification
    if [[ -n "${SMTP_SERVER:-}" && -n "${ADMIN_EMAIL:-}" ]]; then
        {
            echo "Subject: [CRITICAL ALERT] $title"
            echo "To: $ADMIN_EMAIL"
            echo "From: $ALERT_EMAIL_FROM"
            echo ""
            echo "CRITICAL ALERT TRIGGERED"
            echo "========================"
            echo ""
            echo "Alert: $title"
            echo "Details: $message"
            echo "Time: $(date)"
            echo "Server: $(hostname)"
            echo ""
            echo "Automated recovery actions may have been triggered."
            echo "Please check the system status immediately."
        } | sendmail "$ADMIN_EMAIL" 2>/dev/null || log "Failed to send email notification"
    fi
    
    # Slack notification (if configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"text\": \"🚨 CRITICAL ALERT: $title\",
                \"attachments\": [{
                    \"color\": \"danger\",
                    \"fields\": [{
                        \"title\": \"Details\",
                        \"value\": \"$message\",
                        \"short\": false
                    }]
                }]
            }" || log "Failed to send Slack notification"
    fi
    
    # Discord notification (if configured)
    if [[ -n "${DISCORD_WEBHOOK_URL:-}" ]]; then
        curl -s -X POST "$DISCORD_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"embeds\": [{
                    \"title\": \"🚨 CRITICAL ALERT\",
                    \"description\": \"**$title**\\n\\n$message\",
                    \"color\": 15158332,
                    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\"
                }]
            }" || log "Failed to send Discord notification"
    fi
}

# Function to restart container with retry logic
restart_container() {
    local container_name="$1"
    local attempt_file="/tmp/.restart_attempts_$container_name"
    
    # Check if container exists
    if ! docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        log "Container $container_name does not exist, cannot restart"
        return 1
    fi
    
    # Check restart attempts
    local attempts=0
    if [[ -f "$attempt_file" ]]; then
        local last_attempt
        last_attempt=$(stat -c %Y "$attempt_file" 2>/dev/null || echo 0)
        local current_time
        current_time=$(date +%s)
        
        if (( current_time - last_attempt < RESTART_COOLDOWN )); then
            attempts=$(cat "$attempt_file" 2>/dev/null || echo 0)
            if (( attempts >= MAX_RESTART_ATTEMPTS )); then
                log "Maximum restart attempts ($MAX_RESTART_ATTEMPTS) reached for $container_name, skipping"
                return 1
            fi
        else
            # Reset attempts if cooldown period has passed
            attempts=0
        fi
    fi
    
    # Increment attempt counter
    ((attempts++))
    echo "$attempts" > "$attempt_file"
    
    log "Attempting to restart $container_name (attempt $attempts/$MAX_RESTART_ATTEMPTS)"
    
    # Restart container
    if docker restart "$container_name"; then
        log "Successfully restarted $container_name"
        
        # Wait for container to be healthy
        local wait_time=0
        local max_wait=60
        
        while (( wait_time < max_wait )); do
            local status
            status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")
            
            if [[ "$status" == "healthy" ]] || [[ "$status" == "none" ]]; then
                # If no healthcheck or healthy, check if running
                local state
                state=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "unknown")
                if [[ "$state" == "running" ]]; then
                    log "$container_name is running and healthy after restart"
                    # Reset attempts on successful restart
                    rm -f "$attempt_file"
                    return 0
                fi
            fi
            
            sleep 5
            ((wait_time += 5))
        done
        
        log "WARNING: $container_name restarted but may not be healthy"
        return 0
    else
        log "ERROR: Failed to restart $container_name"
        return 1
    fi
}

# Function to handle VPN connection loss
handle_vpn_alert() {
    log "Handling VPN connection alert"
    
    # Stop download clients to prevent IP leaks
    local download_clients=("qbittorrent" "transmission" "sabnzbd" "nzbget")
    
    for client in "${download_clients[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${client}$"; then
            log "Stopping $client to prevent IP leak"
            docker stop "$client" || log "Failed to stop $client"
        fi
    done
    
    # Attempt to restart VPN
    if restart_container "gluetun"; then
        log "VPN restarted successfully, restarting download clients"
        
        # Wait for VPN to establish connection
        sleep 30
        
        # Restart download clients
        for client in "${download_clients[@]}"; do
            if docker ps -a --format "{{.Names}}" | grep -q "^${client}$"; then
                local state
                state=$(docker inspect --format='{{.State.Status}}' "$client" 2>/dev/null || echo "unknown")
                if [[ "$state" != "running" ]]; then
                    log "Restarting $client"
                    docker start "$client" || log "Failed to restart $client"
                fi
            fi
        done
    else
        log "CRITICAL: Failed to restart VPN, download clients remain stopped"
        send_emergency_notification \
            "VPN Recovery Failed" \
            "Could not restart VPN connection. Download clients have been stopped to prevent IP leaks. Manual intervention required."
    fi
}

# Function to handle disk space alerts
handle_disk_space_alert() {
    local mount_point="${1:-/}"
    
    log "Handling disk space alert for $mount_point"
    
    # Clean up common temporary locations
    if [[ "$mount_point" == "/" ]]; then
        log "Cleaning system temporary files"
        find /tmp -type f -atime +7 -delete 2>/dev/null || true
        find /var/tmp -type f -atime +7 -delete 2>/dev/null || true
        
        # Clean Docker system
        log "Cleaning Docker system"
        docker system prune -f --volumes || log "Docker cleanup failed"
    fi
    
    # Clean download directory if it's the affected mount
    if [[ "$mount_point" == "/downloads" ]] || [[ "$mount_point" == *"download"* ]]; then
        log "Cleaning old downloads"
        find /downloads -name "*.part" -mtime +1 -delete 2>/dev/null || true
        find /downloads -name "*.tmp" -mtime +1 -delete 2>/dev/null || true
        
        # Trigger cleanup in qBittorrent
        if docker ps --format "{{.Names}}" | grep -q "qbittorrent"; then
            log "Triggering qBittorrent cleanup"
            # Could implement qBittorrent API call to remove completed downloads
        fi
    fi
    
    # Check if space was freed
    local usage_after
    usage_after=$(df "$mount_point" | awk 'NR==2 {print $5}' | sed 's/%//')
    log "Disk usage after cleanup: ${usage_after}%"
    
    if (( usage_after > 90 )); then
        send_emergency_notification \
            "Disk Space Critical" \
            "Disk space on $mount_point is still at ${usage_after}% after cleanup. Manual intervention required."
    fi
}

# Function to handle service down alerts
handle_service_down() {
    local service_name="$1"
    
    log "Handling service down alert for $service_name"
    
    case "$service_name" in
        "jellyfin"|"sonarr"|"radarr"|"prowlarr"|"qbittorrent"|"transmission")
            restart_container "$service_name"
            ;;
        "postgres"|"redis"|"mariadb")
            log "Database service $service_name is down, attempting restart"
            restart_container "$service_name"
            ;;
        "traefik")
            log "Reverse proxy is down, this is critical"
            restart_container "traefik"
            send_emergency_notification \
                "Reverse Proxy Down" \
                "Traefik reverse proxy was down and restart was attempted. All services may be inaccessible."
            ;;
        *)
            log "Unknown service: $service_name, attempting generic restart"
            restart_container "$service_name"
            ;;
    esac
}

# Main alert handler
main() {
    local alert_status="${1:-unknown}"
    local alert_name="${2:-unknown}"
    local alert_summary="${3:-unknown}"
    
    log "=== Critical Alert Handler Started ==="
    log "Status: $alert_status"
    log "Alert: $alert_name"
    log "Summary: $alert_summary"
    
    # Only handle firing alerts
    if [[ "$alert_status" != "firing" ]]; then
        log "Alert is not firing, skipping automated response"
        echo '{"status":"skipped","message":"Alert not firing"}'
        return 0
    fi
    
    # Handle specific alert types
    case "$alert_name" in
        "VPNConnectionLost"|"VPNIPLeak")
            handle_vpn_alert
            ;;
        "DiskSpaceCritical"|"DiskSpaceLow")
            # Extract mount point from summary if possible
            local mount_point
            mount_point=$(echo "$alert_summary" | grep -oE '/[a-zA-Z0-9/_-]+' | head -1 || echo "/")
            handle_disk_space_alert "$mount_point"
            ;;
        "ServiceDown"|"ContainerDown")
            # Extract service name from summary
            local service_name
            service_name=$(echo "$alert_summary" | grep -oE '[a-zA-Z0-9_-]+' | head -1 || echo "unknown")
            handle_service_down "$service_name"
            ;;
        "CriticalCPUUsage"|"CriticalMemoryUsage")
            log "System resource critical, checking for problematic containers"
            # Could implement logic to restart high-resource containers
            send_emergency_notification \
                "System Resources Critical" \
                "$alert_summary - System may become unresponsive."
            ;;
        *)
            log "Unhandled alert type: $alert_name"
            send_emergency_notification \
                "Unhandled Critical Alert" \
                "$alert_summary"
            ;;
    esac
    
    log "=== Critical Alert Handler Completed ==="
    
    echo '{"status":"success","message":"Alert handled"}'
}

# Error handling
trap 'log "ERROR: Critical alert handler failed at line $LINENO"' ERR

# Execute main function
main "$@"