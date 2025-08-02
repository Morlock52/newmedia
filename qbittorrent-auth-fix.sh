#!/bin/bash

# qBittorrent Authentication Fix Script
# This script helps configure qBittorrent to allow ARR services to connect

echo "=== qBittorrent Authentication Configuration Helper ==="
echo ""
echo "The ARR services are failing to authenticate with qBittorrent."
echo "This is expected behavior and requires manual configuration in qBittorrent."
echo ""

echo "🔧 STEP 1: Access qBittorrent Web UI"
echo "1. Open your browser and go to: http://localhost:8090"
echo "2. Login with:"
echo "   - Username: admin"
echo "   - Password: adminadmin"
echo ""

echo "🔧 STEP 2: Configure Authentication Settings"
echo "1. In qBittorrent Web UI, click 'Tools' → 'Options'"
echo "2. Click on 'Web UI' tab"
echo "3. In the 'Authentication' section:"
echo "   - ✅ ENABLE: 'Bypass authentication for clients on localhost'"
echo "   - ✅ ENABLE: 'Bypass authentication for clients in whitelisted IP subnets'"
echo "   - Add to IP subnet whitelist: '172.20.0.0/16' (Docker network range)"
echo "4. Click 'Save'"
echo ""

echo "🔧 STEP 3: Verify Docker Network Range"
echo "Checking Docker network configuration..."

# Get the actual Docker network subnet
network_info=$(docker network inspect newmedia_media-net 2>/dev/null | jq -r '.[0].IPAM.Config[0].Subnet' 2>/dev/null || echo "172.20.0.0/16")
echo "Docker network subnet: $network_info"
echo ""
echo "⚠️  IMPORTANT: Add this subnet to qBittorrent's IP whitelist: $network_info"
echo ""

echo "🔧 STEP 4: Alternative Configuration (if bypass doesn't work)"
echo "If bypass authentication doesn't work, you can:"
echo "1. Change qBittorrent username/password to something simpler"
echo "2. Or disable authentication entirely (NOT recommended)"
echo "3. Or add specific Docker container IPs to whitelist"
echo ""

echo "🔧 STEP 5: Get Container IP Addresses"
echo "Getting container IP addresses for manual whitelist configuration..."
echo ""

# Get container IPs
qb_ip=$(docker inspect qbittorrent 2>/dev/null | jq -r '.[0].NetworkSettings.Networks["newmedia_media-net"].IPAddress' 2>/dev/null || echo "not found")
sonarr_ip=$(docker inspect sonarr 2>/dev/null | jq -r '.[0].NetworkSettings.Networks["newmedia_media-net"].IPAddress' 2>/dev/null || echo "not found")
radarr_ip=$(docker inspect radarr 2>/dev/null | jq -r '.[0].NetworkSettings.Networks["newmedia_media-net"].IPAddress' 2>/dev/null || echo "not found")
lidarr_ip=$(docker inspect lidarr 2>/dev/null | jq -r '.[0].NetworkSettings.Networks["newmedia_media-net"].IPAddress' 2>/dev/null || echo "not found")

echo "Container IP Addresses:"
echo "  - qBittorrent: $qb_ip"
echo "  - Sonarr: $sonarr_ip"
echo "  - Radarr: $radarr_ip"
echo "  - Lidarr: $lidarr_ip"
echo ""
echo "Add these IPs to qBittorrent whitelist if needed."
echo ""

echo "🔧 STEP 6: Test Configuration"
echo "After configuring qBittorrent authentication, run:"
echo "  ./configure-download-clients-final.sh"
echo ""
echo "Or manually test from container:"
echo "  docker exec sonarr curl -s http://qbittorrent:8080/api/v2/app/version"
echo ""

echo "🔧 STEP 7: Manual ARR Configuration (Web UI)"
echo "If the script still fails, you can manually add qBittorrent in each ARR service:"
echo ""
echo "For Sonarr (http://localhost:8989):"
echo "  Settings → Download Clients → Add qBittorrent"
echo "  - Host: qbittorrent"
echo "  - Port: 8080"
echo "  - Username: admin"
echo "  - Password: adminadmin"
echo "  - Category: sonarr"
echo ""
echo "For Radarr (http://localhost:7878):"
echo "  Settings → Download Clients → Add qBittorrent"  
echo "  - Host: qbittorrent"
echo "  - Port: 8080"
echo "  - Username: admin"
echo "  - Password: adminadmin"
echo "  - Category: radarr"
echo ""
echo "For Lidarr (http://localhost:8686):"
echo "  Settings → Download Clients → Add qBittorrent"
echo "  - Host: qbittorrent"
echo "  - Port: 8080" 
echo "  - Username: admin"
echo "  - Password: adminadmin"
echo "  - Category: lidarr"
echo ""

echo "✅ Summary:"
echo "1. Configure qBittorrent authentication bypass for localhost/Docker network"
echo "2. Re-run the configuration script"
echo "3. Or manually configure through each ARR service's Web UI"
echo "4. Test downloads to ensure everything works"
echo ""
echo "Open qBittorrent now: http://localhost:8090"