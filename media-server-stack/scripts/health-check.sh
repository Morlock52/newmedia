#!/usr/bin/env bash
set -euo pipefail

# Determine project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load environment variables
if [[ -f "$ROOT/.env" ]]; then
  source "$ROOT/.env"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
  log "${RED}ERROR: $1${NC}"
}

log_success() {
  log "${GREEN}SUCCESS: $1${NC}"
}

log_warning() {
  log "${YELLOW}WARNING: $1${NC}"
}

# Function to check Docker service health
check_service() {
  local service_name="$1"
  local health_endpoint="$2"
  
  log "Checking $service_name..."
  
  # Check if container is running
  if ! docker-compose -f "$ROOT/compose/docker-compose.yml" ps "$service_name" | grep -q "Up"; then
    log_error "$service_name is not running"
    return 1
  fi
  
  # Check health endpoint if provided
  if [[ -n "$health_endpoint" ]]; then
    if curl -s -f "$health_endpoint" > /dev/null 2>&1; then
      log_success "$service_name is healthy"
    else
      log_error "$service_name health check failed"
      return 1
    fi
  else
    log_success "$service_name is running"
  fi
  
  return 0
}

# Function to check VPN connectivity
check_vpn() {
  log "Checking VPN connectivity..."
  
  if ! check_service "gluetun" ""; then
    return 1
  fi
  
  # Check VPN IP
  local vpn_ip=$(docker exec gluetun wget -qO- https://ipinfo.io/ip 2>/dev/null || echo "failed")
  
  if [[ "$vpn_ip" == "failed" ]]; then
    log_error "VPN connectivity test failed"
    return 1
  fi
  
  log_success "VPN is active (IP: $vpn_ip)"
  return 0
}

# Function to check disk space
check_disk_space() {
  log "Checking disk space..."
  
  local data_root="${DATA_ROOT:-$ROOT/data}"
  local disk_usage=$(df -h "$data_root" | awk 'NR==2 {print $5}' | sed 's/%//')
  
  if [[ $disk_usage -lt 80 ]]; then
    log_success "Disk usage: ${disk_usage}%"
  elif [[ $disk_usage -lt 90 ]]; then
    log_warning "Disk usage high: ${disk_usage}%"
  else
    log_error "Disk usage critical: ${disk_usage}%"
    return 1
  fi
  
  return 0
}

# Main health check
main() {
  log "Starting comprehensive health check..."
  
  local failed_checks=0
  
  # Core services
  check_service "traefik" "" || ((failed_checks++))
  check_service "gluetun" "" || ((failed_checks++))
  check_service "qbittorrent" "" || ((failed_checks++))
  check_service "prowlarr" "" || ((failed_checks++))
  check_service "sonarr" "" || ((failed_checks++))
  check_service "radarr" "" || ((failed_checks++))
  check_service "jellyfin" "" || ((failed_checks++))
  check_service "overseerr" "" || ((failed_checks++))
  
  # System checks
  check_vpn || ((failed_checks++))
  check_disk_space || ((failed_checks++))
  
  if [[ $failed_checks -eq 0 ]]; then
    log_success "All health checks passed!"
    exit 0
  else
    log_error "$failed_checks health checks failed"
    exit 1
  fi
}

main "$@"