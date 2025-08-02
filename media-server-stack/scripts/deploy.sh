 #!/usr/bin/env bash
 set -euo pipefail

# Determine project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load environment variables from .env, or copy .env.example if missing
if [[ -f "$ROOT/.env" ]]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ROOT/.env"
  set +o allexport
else
  echo "WARNING: .env not found, copying .env.example to .env"
  cp "$ROOT/.env.example" "$ROOT/.env"
  set -o allexport
  source "$ROOT/.env"
  set +o allexport
fi

# Auto-create symlinks for Docker Compose CLI convenience
ln -sf "$ROOT/compose/docker-compose.yml" "$ROOT/docker-compose.yml"
ln -sf "$ROOT/compose/compose.production.yml" "$ROOT/docker-compose.override.yml"

# Data and config roots (override via .env)
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
CONFIG_ROOT="${CONFIG_ROOT:-$ROOT/config}"

# Configuration
COMPOSE_FILES="-f compose/docker-compose.yml -f compose/compose.production.yml"
MONITORING_COMPOSE="-f compose/compose.monitoring.yml"
HEALTH_CHECK_TIMEOUT=300
BACKUP_RETENTION=30

 log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

check_prerequisites() {
    log "Checking prerequisites..."
    command -v docker >/dev/null 2>&1 || { log "Docker not installed"; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { log "Docker Compose not installed"; exit 1; }
}

create_directories() {
    log "Creating project data and config directories under $DATA_ROOT and $CONFIG_ROOT"
    mkdir -p "$DATA_ROOT"/{media,torrents,usenet}/{movies,tv,music,online-videos}
    
    # Create config directories, but don't fail if some can't be created due to permissions
    for dir in jellyfin sonarr radarr lidarr readarr prowlarr qbittorrent overseerr bazarr homarr mylar podgrab photoprism tautulli youtube-dl-material traefik gluetun prometheus grafana alertmanager; do
        mkdir -p "$CONFIG_ROOT/$dir" 2>/dev/null || log "Warning: Could not create $CONFIG_ROOT/$dir (may already exist with different ownership)"
    done
    
    mkdir -p secrets
    # Ensure proper permissions; ignore failures on platforms where chown may be restricted (e.g., macOS)
    chown -R 1000:1000 "$DATA_ROOT" "$CONFIG_ROOT" 2>/dev/null || log "Warning: unable to chown data/config directories"
    chmod -R 755 "$DATA_ROOT" "$CONFIG_ROOT" 2>/dev/null || true
}

setup_networks() {
    log "Setting up Docker networks..."
    docker network create traefik_network 2>/dev/null || log "traefik_network already exists"
    docker network create monitoring_network 2>/dev/null || log "monitoring_network already exists"
}

generate_secrets() {
    log "Generating secrets if they don't exist..."
    
    # Generate Traefik dashboard auth if it doesn't exist
    if [[ ! -f "secrets/traefik_dashboard_auth.txt" ]]; then
        echo "admin:$(openssl passwd -apr1 changeme)" > secrets/traefik_dashboard_auth.txt
        log "Generated traefik dashboard auth (default: admin/changeme)"
    fi
    
    # Generate placeholder for WireGuard key
    if [[ ! -f "secrets/wg_private_key.txt" ]]; then
        echo "# Place your WireGuard private key here" > secrets/wg_private_key.txt
        log "Created placeholder for WireGuard private key"
    fi
    
    # Generate Grafana admin password
    if [[ ! -f "secrets/grafana_admin_password.txt" ]]; then
        openssl rand -base64 32 > secrets/grafana_admin_password.txt
        log "Generated Grafana admin password"
    fi
    
    chmod 600 secrets/*
}

deploy_stack() {
    log "Deploying media server stack..."
    pushd "$ROOT" >/dev/null
    
    # Copy monitoring configs (use root directory if config subdirs not accessible)
    if [[ -d "config/prometheus" ]]; then
        cp prometheus.yml config/prometheus/ 2>/dev/null || true
        cp alert_rules.yml config/prometheus/ 2>/dev/null || true
    else
        log "Using monitoring configs from project root"
    fi
    
    # Deploy main stack
    docker-compose $COMPOSE_FILES up -d
    
    # Deploy monitoring stack if requested
    if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
        log "Deploying monitoring stack..."
        docker-compose $MONITORING_COMPOSE up -d
    fi
    
    popd >/dev/null

  log "Waiting for services to be healthy..."
  sleep 30  # Give services time to start
}

verify_deployment() {
  log "Verifying deployment..."
  local failed_services=()
  local core_services=("traefik" "gluetun" "jellyfin" "sonarr" "radarr" "prowlarr" "qbittorrent" "overseerr" "bazarr" "homarr")

    pushd "$ROOT" >/dev/null
    for service in "${core_services[@]}"; do
        if ! docker-compose $COMPOSE_FILES ps "$service" | grep -q "Up"; then
            failed_services+=("$service")
        fi
    done
    popd >/dev/null

  if [ ${#failed_services[@]} -gt 0 ]; then
    log "Failed services: ${failed_services[*]}"
    log "Checking logs for failed services..."
    for service in "${failed_services[@]}"; do
        log "Logs for $service:"
        docker-compose $COMPOSE_FILES logs --tail 20 "$service" || true
    done
    exit 1
  fi

  log "All services deployed successfully!"
  log "Access your services at:"
  log "  - Jellyfin: https://jellyfin.${DOMAIN:-localhost}"
  log "  - Sonarr: https://sonarr.${DOMAIN:-localhost}"
  log "  - Radarr: https://radarr.${DOMAIN:-localhost}"
  log "  - Prowlarr: https://prowlarr.${DOMAIN:-localhost}"
  log "  - Overseerr: https://overseerr.${DOMAIN:-localhost}"
  log "  - Traefik Dashboard: https://traefik.${DOMAIN:-localhost}"
  
  if [[ "${DEPLOY_MONITORING:-true}" == "true" ]]; then
      log "  - Prometheus: https://prometheus.${DOMAIN:-localhost}"
      log "  - Grafana: https://grafana.${DOMAIN:-localhost}"
  fi
}

# Main execution
main() {
  log "Starting media server deployment..."
  check_prerequisites
  setup_networks
  create_directories
  generate_secrets
  deploy_stack
  verify_deployment
  log "Deployment completed successfully!"
}

main "$@"