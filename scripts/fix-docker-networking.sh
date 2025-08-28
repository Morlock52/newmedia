#!/bin/bash

# Docker Network Fixes and Optimization Script
# Fixes common networking issues and optimizes container communication

set -euo pipefail

echo "🔧 Docker Network Fix and Optimization Script"
echo "=============================================="

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verify Docker is installed and running
if ! command_exists docker; then
    log "❌ Docker is not installed"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    log "❌ Docker daemon is not running"
    exit 1
fi

# Clean up existing problematic networks
log "🧹 Cleaning up existing networks..."
docker network prune -f

# Remove orphaned networks that might cause conflicts
for network in media-net downloads-net vpn-net monitoring-net management-net; do
    if docker network ls | grep -q "$network"; then
        log "📡 Removing existing network: $network"
        docker network rm "$network" 2>/dev/null || true
    fi
done

# Create optimized networks with proper configuration
log "🏗️  Creating optimized networks..."

# Media services network - optimized for high-throughput
docker network create media-net \
    --driver bridge \
    --subnet=172.30.0.0/16 \
    --gateway=172.30.0.1 \
    --ip-range=172.30.1.0/24 \
    --opt com.docker.network.bridge.name=media-br0 \
    --opt com.docker.network.bridge.enable_icc=true \
    --opt com.docker.network.bridge.enable_ip_masquerade=true \
    --opt com.docker.network.bridge.host_binding_ipv4=0.0.0.0 \
    --opt com.docker.network.mtu=1500 \
    --label com.media-server.network.tier=media || log "⚠️  Network media-net already exists"

# Downloads network - isolated for security
docker network create downloads-net \
    --driver bridge \
    --subnet=172.31.0.0/16 \
    --gateway=172.31.0.1 \
    --ip-range=172.31.1.0/24 \
    --opt com.docker.network.bridge.name=downloads-br0 \
    --opt com.docker.network.bridge.enable_icc=true \
    --opt com.docker.network.bridge.enable_ip_masquerade=true \
    --opt com.docker.network.mtu=1500 \
    --label com.media-server.network.tier=downloads || log "⚠️  Network downloads-net already exists"

# VPN network - secure tunnel
docker network create vpn-net \
    --driver bridge \
    --subnet=172.32.0.0/16 \
    --gateway=172.32.0.1 \
    --ip-range=172.32.1.0/24 \
    --opt com.docker.network.bridge.name=vpn-br0 \
    --opt com.docker.network.bridge.enable_icc=false \
    --opt com.docker.network.bridge.enable_ip_masquerade=false \
    --opt com.docker.network.mtu=1436 \
    --label com.media-server.network.tier=vpn || log "⚠️  Network vpn-net already exists"

# Monitoring network
docker network create monitoring-net \
    --driver bridge \
    --subnet=172.33.0.0/16 \
    --gateway=172.33.0.1 \
    --ip-range=172.33.1.0/24 \
    --opt com.docker.network.bridge.name=monitoring-br0 \
    --opt com.docker.network.bridge.enable_icc=true \
    --opt com.docker.network.mtu=1500 \
    --label com.media-server.network.tier=monitoring || log "⚠️  Network monitoring-net already exists"

# Management network
docker network create management-net \
    --driver bridge \
    --subnet=172.34.0.0/16 \
    --gateway=172.34.0.1 \
    --ip-range=172.34.1.0/24 \
    --opt com.docker.network.bridge.name=mgmt-br0 \
    --opt com.docker.network.bridge.enable_icc=true \
    --opt com.docker.network.mtu=1500 \
    --label com.media-server.network.tier=management || log "⚠️  Network management-net already exists"

log "🎉 Docker networking fixes completed!"
log ""
log "Next steps:"
log "1. Run 'docker-compose down && docker-compose up -d' to recreate containers"
log "2. Verify services are accessible on their configured ports"