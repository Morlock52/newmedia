#!/bin/bash

# Media Server Stack 2025 - Optimized Deployment Script
# Implements all agent recommendations for security, accessibility, and performance

set -e

echo "🎬 Media Server Stack 2025 - Optimized Deployment"
echo "=================================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Stop current deployment if running
echo "🛑 Stopping current deployment..."
docker compose -f docker-compose-2025-fixed.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose-optimized.yml down --remove-orphans 2>/dev/null || true

# Clean up networks
echo "🧹 Cleaning up networks..."
docker network prune -f

# Remove problematic containers
echo "🗑️ Removing problematic containers..."
docker rm -f gluetun cadvisor 2>/dev/null || true

# Deploy optimized stack
echo "🚀 Deploying optimized media server stack..."
docker compose -f docker-compose-optimized.yml up -d --pull always

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check service status
echo "📊 Service Status:"
docker compose -f docker-compose-optimized.yml ps --format "table {{.Service}}\t{{.State}}\t{{.Ports}}"

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "🌐 Service Access URLs:"
echo "========================"
echo "🎬 Jellyfin Media Server:    http://localhost:8096"
echo "📚 AudioBookshelf:            http://localhost:13378"
echo "🎵 Navidrome Music:           http://localhost:4533"
echo "📸 Immich Photos:             http://localhost:2283"
echo "📥 qBittorrent:               http://localhost:8080"
echo "📰 SABnzbd:                   http://localhost:8081"
echo "🎭 Radarr:                    http://localhost:7878"
echo "📺 Sonarr:                    http://localhost:8989"
echo "🔍 Prowlarr:                  http://localhost:9696"
echo "📊 Grafana:                   http://localhost:3000"
echo "🎯 Prometheus:                http://localhost:9090"
echo "🏠 Homepage Dashboard:        http://localhost:3001"
echo "🐳 Portainer:                 http://localhost:9000"
echo "🔧 Traefik Dashboard:         http://localhost:8090"
echo ""
echo "🔐 Credentials: Check .generated_passwords.txt"
echo ""
echo "🎉 All services are optimized for macOS and include:"
echo "   ✅ Direct port access (no proxy issues)"
echo "   ✅ Latest security hardening"
echo "   ✅ All missing media applications"
echo "   ✅ Comprehensive monitoring"
echo "   ✅ Resource optimization"
echo ""
echo "📖 Open service-access.html for a visual dashboard"