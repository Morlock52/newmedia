#!/bin/bash

# Start media stack inside container
set -e

echo "🚀 Starting Media Stack in Container"
echo "===================================="

# Wait for Docker daemon to be ready
echo "⏳ Waiting for Docker daemon..."
while ! docker info >/dev/null 2>&1; do
    sleep 2
done
echo "✅ Docker daemon is ready"

# Change to media directory
cd /media

# Start the media stack
echo "🐳 Starting media services..."
docker compose up -d

# Monitor services
echo "📊 Media stack started! Monitoring services..."

# Keep the script running and monitor
while true; do
    echo "$(date): Checking service health..."
    docker compose ps
    sleep 300  # Check every 5 minutes
done