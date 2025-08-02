#!/bin/bash

# Cloudflare DNS Setup Script
# This script provides the exact API calls needed to set up DNS records

CF_API_TOKEN="kDsjTUt52knOrj2jPwjAi7wJNIJptNhoEBmN23Kc"
ZONE_ID="5d9c72066e5f2b8944253353f67b1b61"
TUNNEL_ID="43efadba-2cd1-4e2e-8174-c1686f0d5544"
TUNNEL_HOSTNAME="${TUNNEL_ID}.cfargotunnel.com"

echo "=== Cloudflare DNS Setup ==="
echo "Zone ID: ${ZONE_ID}"
echo "Tunnel ID: ${TUNNEL_ID}"
echo "Tunnel Hostname: ${TUNNEL_HOSTNAME}"
echo ""

# Check if we need to create DNS records
echo "=== Step 1: Create DNS Records ==="
echo "The following DNS records need to be created in Cloudflare:"
echo ""

SUBDOMAINS=("home" "jellyfin" "auth" "sonarr" "radarr" "prowlarr" "overseerr" "traefik" "lidarr")

for subdomain in "${SUBDOMAINS[@]}"; do
  echo "CNAME: ${subdomain}.morloksmaze.com → ${TUNNEL_HOSTNAME} (Proxied)"
done

echo ""
echo "=== Manual DNS Creation Instructions ==="
echo "1. Go to https://dash.cloudflare.com"
echo "2. Select domain: morloksmaze.com"
echo "3. Go to DNS > Records"
echo "4. Add the following CNAME records:"
echo ""

for subdomain in "${SUBDOMAINS[@]}"; do
  echo "   Type: CNAME"
  echo "   Name: ${subdomain}"
  echo "   Target: ${TUNNEL_HOSTNAME}"
  echo "   Proxy: Enabled (Orange Cloud)"
  echo "   TTL: Auto"
  echo ""
done

echo "=== Step 2: Configure Tunnel Routes ==="
echo "After DNS records are created, configure tunnel routes:"
echo ""
echo "1. Go to https://one.dash.cloudflare.com"
echo "2. Go to Zero Trust > Access > Tunnels"
echo "3. Click on tunnel: ${TUNNEL_ID}"
echo "4. Click Configure"
echo "5. Add these Public hostnames:"
echo ""

TRAEFIK_IP="172.24.0.2"
for subdomain in "${SUBDOMAINS[@]}"; do
  echo "   Subdomain: ${subdomain}.morloksmaze.com"
  echo "   Service: http://${TRAEFIK_IP}:80"
  echo "   Path: (leave empty)"
  echo ""
done

echo "=== Step 3: Test Configuration ==="
echo "After setup, test with:"
echo "curl -I https://home.morloksmaze.com"
echo ""

# Let's try to create the DNS records anyway
echo "=== Attempting to create DNS records via API ==="
for subdomain in "${SUBDOMAINS[@]}"; do
  echo "Attempting to create ${subdomain}.morloksmaze.com..."
  
  response=$(curl -s -X POST "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records" \
    -H "Authorization: Bearer ${CF_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{
      \"type\": \"CNAME\",
      \"name\": \"${subdomain}\",
      \"content\": \"${TUNNEL_HOSTNAME}\",
      \"ttl\": 1,
      \"proxied\": true
    }")
  
  success=$(echo "$response" | jq -r '.success // false')
  if [ "$success" = "true" ]; then
    echo "✅ Created ${subdomain}.morloksmaze.com"
  else
    error=$(echo "$response" | jq -r '.errors[0].message // "API call failed"')
    echo "❌ ${subdomain}.morloksmaze.com: ${error}"
  fi
done