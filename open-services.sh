#!/bin/bash

# Ultimate Media Server 2025 - Quick Access Script
# Opens all main services in your default browser

echo "🚀 Opening Ultimate Media Server 2025 Services..."

# Main Services
echo "Opening main dashboard..."
open "http://localhost:3001" # Homepage Dashboard

sleep 2

echo "Opening media services..."
open "http://localhost:8096" # Jellyfin
open "http://localhost:5055" # Overseerr

sleep 2

echo "Opening management tools..."
open "http://localhost:9000" # Portainer

sleep 2

echo "Opening automation services..."
open "http://localhost:8989" # Sonarr
open "http://localhost:7878"  # Radarr
open "http://localhost:9696"  # Prowlarr

sleep 2

echo "Opening download clients..."
open "http://localhost:8080"  # qBittorrent

sleep 2

echo "Opening monitoring..."
open "http://localhost:3000"  # Grafana

echo ""
echo "🎉 All services opened! Check your browser tabs."
echo ""
echo "📋 Quick Reference:"
echo "   🏠 Main Dashboard: http://localhost:3001"
echo "   🎬 Jellyfin Media: http://localhost:8096"
echo "   🎯 Request Media:  http://localhost:5055"
echo "   🐳 Portainer:      http://localhost:9000"
echo ""