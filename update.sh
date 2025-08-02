#!/bin/bash
# Update script for Ultimate Media Server 2025

set -euo pipefail

echo "🔄 Ultimate Media Server 2025 - Update Process"
echo "============================================="

# Backup before update
echo "📦 Creating backup..."
./backup.sh

# Pull latest images
echo "🐳 Pulling latest Docker images..."
docker-compose pull

# Update containers
echo "🚀 Updating containers..."
docker-compose up -d --remove-orphans

# Clean up
echo "🧹 Cleaning up old images..."
docker image prune -f

# Run health check
echo "🏥 Running health check..."
sleep 30
./health-check.sh

echo "✅ Update complete!"
