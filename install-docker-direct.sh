#!/bin/bash

# Direct Docker Desktop installation for macOS
echo "🐳 Docker Installation for macOS"
echo "================================"

# Check if Docker is already installed
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
    echo "✅ Docker is already installed and running!"
    docker --version
    exit 0
fi

echo "📥 Downloading Docker Desktop..."

# Download Docker Desktop for Mac (Intel)
curl -L -o /tmp/Docker.dmg "https://desktop.docker.com/mac/main/amd64/Docker.dmg"

echo "📦 Installing Docker Desktop..."

# Mount the DMG
hdiutil mount /tmp/Docker.dmg

# Copy Docker to Applications
cp -R "/Volumes/Docker/Docker.app" "/Applications/"

# Unmount the DMG
hdiutil unmount "/Volumes/Docker"

# Clean up
rm /tmp/Docker.dmg

echo "🚀 Starting Docker Desktop..."
open /Applications/Docker.app

echo "⏳ Waiting for Docker to start..."
echo "Please wait for the Docker whale icon to appear in your menu bar."
echo "This may take 1-2 minutes on first startup."

# Wait for Docker to be ready
timeout=120
while [ $timeout -gt 0 ]; do
    if docker info &> /dev/null; then
        echo "✅ Docker is ready!"
        docker --version
        exit 0
    fi
    sleep 5
    ((timeout-=5))
    echo -n "."
done

echo ""
echo "⚠️  Docker is still starting up."
echo "Please wait for Docker Desktop to finish starting, then run:"
echo "  ./deploy-simple.sh"