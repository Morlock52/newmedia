#!/usr/bin/env bash
set -euo pipefail

# Determine project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

generate_password() {
  openssl rand -base64 32
}

generate_api_key() {
  openssl rand -hex 32
}

setup_secrets() {
  log "Setting up secrets management..."
  
  mkdir -p "$ROOT/secrets"
  chmod 700 "$ROOT/secrets"
  
  # Generate Traefik dashboard authentication
  if [[ ! -f "$ROOT/secrets/traefik_dashboard_auth.txt" ]]; then
    log "Generating Traefik dashboard authentication..."
    read -p "Enter username for Traefik dashboard [admin]: " traefik_user
    traefik_user=${traefik_user:-admin}
    
    read -s -p "Enter password for Traefik dashboard [random]: " traefik_pass
    echo
    traefik_pass=${traefik_pass:-$(generate_password)}
    
    echo "$traefik_user:$(openssl passwd -apr1 "$traefik_pass")" > "$ROOT/secrets/traefik_dashboard_auth.txt"
    log "Traefik dashboard: user=$traefik_user, pass=$traefik_pass"
  fi
  
  # Generate Grafana admin password
  if [[ ! -f "$ROOT/secrets/grafana_admin_password.txt" ]]; then
    log "Generating Grafana admin password..."
    grafana_pass=$(generate_password)
    echo "$grafana_pass" > "$ROOT/secrets/grafana_admin_password.txt"
    log "Grafana admin password: $grafana_pass"
  fi
  
  # Generate API keys for services
  for service in jellyfin sonarr radarr prowlarr lidarr readarr bazarr tautulli; do
    if [[ ! -f "$ROOT/secrets/${service}_api_key.txt" ]]; then
      log "Generating API key for $service..."
      generate_api_key > "$ROOT/secrets/${service}_api_key.txt"
    fi
  done
  
  # Generate PhotoPrism admin password
  if [[ ! -f "$ROOT/secrets/photoprism_admin_password.txt" ]]; then
    log "Generating PhotoPrism admin password..."
    generate_password > "$ROOT/secrets/photoprism_admin_password.txt"
  fi
  
  # Create placeholder for VPN key
  if [[ ! -f "$ROOT/secrets/wg_private_key.txt" ]]; then
    log "Creating placeholder for WireGuard private key..."
    cat > "$ROOT/secrets/wg_private_key.txt" << 'EOF'
# Place your WireGuard private key here
# Example: aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890=
# 
# To obtain your WireGuard key:
# 1. Log into your VPN provider's dashboard
# 2. Navigate to WireGuard configuration
# 3. Copy the private key
# 4. Replace this content with your actual key
EOF
  fi
  
  # Set secure permissions
  chmod 600 "$ROOT/secrets"/*
  
  log "Secrets generated successfully!"
  log "Important: Review and update $ROOT/secrets/wg_private_key.txt with your actual VPN key"
}

setup_network_security() {
  log "Setting up network security..."
  
  # Create Docker networks with specific subnets
  docker network create --driver bridge \
    --subnet=172.20.0.0/16 \
    --gateway=172.20.0.1 \
    traefik_network 2>/dev/null || log "traefik_network already exists"
  
  docker network create --driver bridge \
    --subnet=172.21.0.0/16 \
    --gateway=172.21.0.1 \
    --internal \
    download_network 2>/dev/null || log "download_network already exists"
  
  docker network create --driver bridge \
    --subnet=172.22.0.0/16 \
    --gateway=172.22.0.1 \
    monitoring_network 2>/dev/null || log "monitoring_network already exists"
  
  docker network create --driver bridge \
    --subnet=172.23.0.0/16 \
    --gateway=172.23.0.1 \
    media_network 2>/dev/null || log "media_network already exists"
}

setup_file_permissions() {
  log "Setting up file permissions..."
  
  # Ensure proper ownership of data directories
  if [[ -d "$ROOT/data" ]]; then
    sudo chown -R 1000:1000 "$ROOT/data" 2>/dev/null || log "Unable to chown data directory"
    chmod -R 755 "$ROOT/data"
  fi
  
  if [[ -d "$ROOT/config" ]]; then
    sudo chown -R 1000:1000 "$ROOT/config" 2>/dev/null || log "Unable to chown config directory"
    chmod -R 755 "$ROOT/config"
  fi
  
  # Make scripts executable
  chmod +x "$ROOT/scripts"/*.sh
}

verify_security_setup() {
  log "Verifying security setup..."
  
  # Check network isolation
  if docker network inspect download_network --format '{{.Internal}}' | grep -q true; then
    log "✓ Download network is properly isolated"
  else
    log "⚠ Warning: Download network is not isolated"
  fi
  
  # Check secrets permissions
  if [[ -d "$ROOT/secrets" ]]; then
    secret_perms=$(stat -c "%a" "$ROOT/secrets" 2>/dev/null || stat -f "%A" "$ROOT/secrets" 2>/dev/null)
    if [[ "$secret_perms" == "700" ]]; then
      log "✓ Secrets directory has correct permissions"
    else
      log "⚠ Warning: Secrets directory permissions are $secret_perms (should be 700)"
    fi
  fi
  
  # Check if VPN key is configured
  if [[ -f "$ROOT/secrets/wg_private_key.txt" ]] && grep -q "aBcDeFgHiJkLmNoPqRsTuVwXyZ" "$ROOT/secrets/wg_private_key.txt"; then
    log "⚠ Warning: VPN private key is still placeholder - update with real key"
  elif [[ -f "$ROOT/secrets/wg_private_key.txt" ]] && ! grep -q "^#" "$ROOT/secrets/wg_private_key.txt"; then
    log "✓ VPN private key appears to be configured"
  fi
  
  log "Security setup verification completed"
}

audit_container_security() {
  log "Performing container security audit..."
  
  # Check for running containers with root user
  log "Checking for containers running as root..."
  docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | while read -r line; do
    if [[ "$line" == "NAMES"* ]]; then continue; fi
    container_name=$(echo "$line" | awk '{print $1}')
    if docker exec "$container_name" id -u 2>/dev/null | grep -q "^0$"; then
      log "⚠ Warning: Container $container_name is running as root"
    fi
  done
  
  # Check for containers with excessive capabilities
  log "Checking container capabilities..."
  docker ps --format "{{.Names}}" | while read -r container; do
    caps=$(docker inspect "$container" --format '{{.HostConfig.CapAdd}}' 2>/dev/null || echo "[]")
    if [[ "$caps" != "[]" ]] && [[ "$caps" != "<no value>" ]]; then
      log "Info: Container $container has additional capabilities: $caps"
    fi
  done
  
  log "Container security audit completed"
}

main() {
  case "${1:-setup}" in
    setup)
      setup_secrets
      setup_network_security
      setup_file_permissions
      verify_security_setup
      log "Security setup completed successfully!"
      ;;
    audit)
      audit_container_security
      ;;
    verify)
      verify_security_setup
      ;;
    *)
      log "Usage: $0 [setup|audit|verify]"
      exit 1
      ;;
  esac
}

main "$@"