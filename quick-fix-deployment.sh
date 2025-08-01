#!/bin/bash

# Quick Fix for Media Server Deployment Issues
# Addresses immediate accessibility and performance problems

set -e

echo "🔧 Quick Fix: Media Server Deployment Issues"
echo "============================================="

# Stop current deployment
echo "⏹️  Stopping current deployment..."
docker-compose -f docker-compose-2025-fixed.yml down 2>/dev/null || true

# Check Docker resources
echo "📊 Checking Docker resources..."
docker system df

# Clean up if needed
echo "🧹 Cleaning up Docker resources..."
docker system prune -f

# Deploy optimized configuration
echo "🚀 Deploying optimized configuration..."
./deploy-macos-optimized.sh

echo ""
echo "✅ Quick fix completed!"
echo ""
echo "🌐 Updated Service Access:"
echo "  Main Dashboard: http://localhost:3000"
echo "  Jellyfin:       http://localhost:8096"
echo "  Downloads:      http://localhost:8081 (Torrent), http://localhost:8082 (Usenet)"
echo "  Movies:         http://localhost:7878 (Radarr)"
echo "  TV Shows:       http://localhost:8989 (Sonarr)"
echo "  Music:          http://localhost:4533 (Navidrome)"
echo "  Audiobooks:     http://localhost:13378"
echo "  Indexers:       http://localhost:9696 (Prowlarr)"
echo "  Requests:       http://localhost:5055 (Overseerr)"
echo "  Docker:         http://localhost:9000 (Portainer)"
echo ""
echo "⚡ Key Improvements:"
echo "  ✅ Removed problematic VPN configuration"
echo "  ✅ Fixed service accessibility issues"
echo "  ✅ Optimized for macOS Docker performance"
echo "  ✅ Added direct port access for all services"
echo "  ✅ Simplified network architecture"
echo ""