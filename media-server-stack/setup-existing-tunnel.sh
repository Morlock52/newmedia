#!/usr/bin/env bash
set -euo pipefail

echo "🔐 Setting up Cloudflare Authentication for Existing Tunnel"
echo "=========================================================="
echo ""
echo "Using your existing tunnel: morloksmaze-media-tunnel"
echo "Domain: morloksmaze.com"
echo "Email: admin@morloksmaze.com"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment
source .env

echo -e "${GREEN}Step 1: Check Tunnel Status${NC}"
echo "First, let's make sure your tunnel is running..."

# Check if cloudflared container is running
if docker ps | grep -q cloudflared; then
    echo "✅ Cloudflared container is running"
    echo ""
    echo "Recent tunnel logs:"
    docker logs cloudflared --tail 5
    echo ""
else
    echo "⚠️  Cloudflared container not running. Starting it..."
    docker-compose up -d cloudflared
    sleep 10
    echo "Tunnel started. Checking connection..."
    docker logs cloudflared --tail 5
fi

echo ""
read -p "Press ENTER to continue to Zero Trust setup..."

echo -e "${GREEN}Step 2: Access Your Tunnel in Cloudflare Dashboard${NC}"
echo "Opening Cloudflare Zero Trust dashboard..."

# Open the tunnel management page directly
osascript -e 'tell application "Google Chrome" to open location "https://one.dash.cloudflare.com/"'

echo ""
echo "In the Cloudflare Zero Trust dashboard:"
echo "1. Go to Networks → Tunnels"
echo "2. Find your tunnel: 'morloksmaze-media-tunnel'"
echo "3. If it shows as 'Healthy' - great!"
echo "4. If not, we may need to reconnect it"
echo ""
read -p "Is your tunnel showing as 'Healthy'? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "If tunnel is not healthy, try:"
    echo "1. Stop and restart: docker-compose restart cloudflared"
    echo "2. Check logs: docker logs cloudflared"
    echo "3. The tunnel token might be expired"
    echo ""
    read -p "Press ENTER after checking tunnel status..."
fi

echo ""
echo -e "${GREEN}Step 3: Configure Public Hostnames for Your Tunnel${NC}"
echo "Now let's set up the public hostnames..."

echo ""
echo "In the Zero Trust dashboard:"
echo "1. Click on your tunnel: 'morloksmaze-media-tunnel'"
echo "2. Go to the 'Public Hostnames' tab"
echo "3. Add these hostnames (click 'Add a public hostname' for each):"
echo ""

# Services configuration
declare -A services=(
    ["jellyfin"]="8096"
    ["sonarr"]="8989"
    ["radarr"]="7878" 
    ["prowlarr"]="9696"
    ["overseerr"]="5055"
    ["traefik"]="8080"
)

echo -e "${BLUE}Public Hostnames to Add:${NC}"
echo "=========================="
echo ""

for service in "${!services[@]}"; do
    port="${services[$service]}"
    echo "Hostname ${service}:"
    echo "  Subdomain: ${service}"
    echo "  Domain: morloksmaze.com"
    echo "  Service: HTTP"
    echo "  URL: traefik:80"
    echo "  Additional headers:"
    echo "    Host: ${service}.morloksmaze.com"
    echo ""
done

echo "⚠️  Important: Use 'traefik:80' as the service URL, NOT localhost!"
echo "This connects to your Traefik container which routes to the right service."
echo ""
read -p "Press ENTER after adding all public hostnames..."

echo ""
echo -e "${GREEN}Step 4: Set Up Access Applications${NC}"
echo "Now let's protect each service with authentication..."

echo ""
echo "In Zero Trust dashboard:"
echo "1. Go to Access → Applications"
echo "2. Click 'Add an application'"
echo "3. Choose 'Self-hosted'"
echo "4. Create one application for each service:"
echo ""

for service in "${!services[@]}"; do
    echo -e "${BLUE}Application for ${service}:${NC}"
    echo "  Application name: $(echo ${service^})"
    echo "  Application domain: ${service}.morloksmaze.com"
    echo "  Session duration: 24 hours"
    echo ""
    echo "  Policy configuration:"
    echo "    Policy name: ${service^} Access"
    echo "    Action: Allow"
    echo "    Include → Emails: admin@morloksmaze.com"
    echo ""
done

echo "Create all 6 applications before continuing."
echo ""
read -p "Press ENTER after creating all Access applications..."

echo ""
echo -e "${GREEN}Step 5: Configure Authentication Method${NC}"
echo "Setting up email PIN authentication..."

echo ""
echo "In Zero Trust dashboard:"
echo "1. Go to Settings → Authentication"
echo "2. Click 'Add new'"
echo "3. Select 'One-time PIN'"
echo "4. Save configuration"
echo ""
read -p "Press ENTER after configuring One-time PIN authentication..."

echo ""
echo -e "${GREEN}Step 6: Verify DNS Records${NC}"
echo "Let's check your DNS configuration..."

# Open DNS management
osascript -e 'tell application "Google Chrome" to open location "https://dash.cloudflare.com/"'

echo ""
echo "In Cloudflare DNS management:"
echo "1. Go to your domain: morloksmaze.com"
echo "2. Click on 'DNS' → 'Records'"
echo "3. Verify these CNAME records exist and are PROXIED (🧡):"
echo ""

for service in "${!services[@]}"; do
    echo "  ${service} → morloksmaze.com (Proxied 🧡)"
done

echo ""
echo "If any records are missing, add them as CNAME records pointing to morloksmaze.com"
echo "Make sure they are PROXIED (orange cloud icon)"
echo ""
read -p "Press ENTER after verifying DNS records..."

echo ""
echo -e "${GREEN}Step 7: Test Your Configuration${NC}"
echo "Let's test each service with authentication..."

# First, make sure our services are running
echo "Checking local services..."
docker-compose up -d jellyfin sonarr radarr prowlarr overseerr traefik

echo "Waiting for services to start..."
sleep 15

echo ""
echo "Testing each service with Cloudflare authentication:"
echo ""

for service in "${!services[@]}"; do
    url="https://${service}.morloksmaze.com"
    
    echo -e "${BLUE}🧪 Testing: ${service^}${NC}"
    echo "Opening: $url"
    
    # Open the service URL
    osascript -e "tell application \"Google Chrome\" to open location \"$url\""
    
    echo ""
    echo "Expected flow:"
    echo "1. Cloudflare Access login page appears"
    echo "2. Enter: admin@morloksmaze.com"
    echo "3. Check email for PIN code"
    echo "4. Enter PIN to access ${service^}"
    echo ""
    
    read -p "Did ${service^} authentication work correctly? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✅ ${service^} working correctly${NC}"
    else
        echo -e "${RED}❌ ${service^} authentication failed${NC}"
        echo "Troubleshooting ${service}:"
        echo "- Check Access application exists for ${service}.morloksmaze.com"
        echo "- Verify public hostname is configured in tunnel"
        echo "- Check DNS record exists and is proxied"
        echo "- Try incognito mode to clear cache"
    fi
    echo ""
done

echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "==================="
echo ""
echo "Your media server is now protected by Cloudflare Zero Trust authentication!"
echo ""
echo -e "${BLUE}📱 Your Secure Services:${NC}"
for service in "${!services[@]}"; do
    echo "• ${service^}: https://${service}.morloksmaze.com"
done

echo ""
echo -e "${BLUE}🔑 How to Access:${NC}"
echo "1. Visit any service URL"
echo "2. Enter email: admin@morloksmaze.com" 
echo "3. Check email for PIN code"
echo "4. Enter PIN to access service"
echo ""
echo -e "${BLUE}🛠️  Management:${NC}"
echo "• Zero Trust Dashboard: https://one.dash.cloudflare.com/"
echo "• Add users: Access → Applications → Edit policies"
echo "• View access logs: Logs → Access"
echo "• Manage tunnel: Networks → Tunnels"
echo ""
echo -e "${BLUE}🔧 Troubleshooting:${NC}"
echo "• Check tunnel: docker logs cloudflared"
echo "• Check services: docker-compose ps"
echo "• Test internal: curl -I http://localhost:8096"
echo ""

# Final status check
echo -e "${YELLOW}Final Status Check:${NC}"
echo "=================="

# Check tunnel
if docker logs cloudflared --tail 5 | grep -q "Registered tunnel connection"; then
    echo "✅ Tunnel connected"
else
    echo "⚠️  Tunnel may have connection issues"
fi

# Check services  
running_services=0
for service in "${!services[@]}"; do
    if docker ps | grep -q "$service"; then
        ((running_services++))
    fi
done

echo "✅ $running_services/${#services[@]} services running"
echo ""
echo "🎊 Enjoy your secure media server!"
