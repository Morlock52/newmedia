#!/usr/bin/env bash
set -euo pipefail

echo "🔐 Cloudflare Zero Trust Authentication Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load environment
if [[ -f ".env" ]]; then
    source .env
else
    echo -e "${RED}❌ .env file not found${NC}"
    exit 1
fi

echo -e "${BLUE}Domain:${NC} $DOMAIN"
echo -e "${BLUE}Email:${NC} $EMAIL"
echo ""

# Function to open URL and wait for user
open_and_wait() {
    local url="$1"
    local description="$2"
    
    echo -e "${YELLOW}📱 Opening: $description${NC}"
    echo -e "${BLUE}URL: $url${NC}"
    
    # Open in Chrome
    osascript -e "tell application \"Google Chrome\" to open location \"$url\""
    
    echo ""
    read -p "Press ENTER when you've completed this step..."
    echo ""
}

echo -e "${GREEN}🚀 Step 1: Access Zero Trust Dashboard${NC}"
open_and_wait "https://one.dash.cloudflare.com/" "Cloudflare Zero Trust Dashboard"

echo "Complete these steps:"
echo "1. Select your domain: $DOMAIN"
echo "2. Navigate to 'Zero Trust' in the sidebar"
echo "3. Choose the Free Plan if prompted"
echo "4. Complete any onboarding steps"
echo ""
read -p "Press ENTER when Zero Trust is set up..."

echo -e "${GREEN}🔑 Step 2: Configure Authentication Method${NC}"
echo "In the Zero Trust dashboard:"
echo "1. Go to Settings → Authentication"
echo "2. Click 'Add new' authentication method"
echo "3. Select 'One-time PIN'"
echo "4. Save the configuration"
echo ""
read -p "Press ENTER when authentication method is configured..."

echo -e "${GREEN}🌐 Step 3: Create Access Applications${NC}"
echo "Now we'll create applications for each service..."

# Services to configure
declare -A services=(
    ["jellyfin"]="8096"
    ["sonarr"]="8989" 
    ["radarr"]="7878"
    ["prowlarr"]="9696"
    ["overseerr"]="5055"
    ["traefik"]="8080"
)

echo "For each service, create an application with these settings:"
echo ""

for service in "${!services[@]}"; do
    port="${services[$service]}"
    echo -e "${BLUE}📺 $service.${DOMAIN}${NC}"
    echo "  Application name: $(echo "$service" | sed 's/.*/\u&/')"
    echo "  Application domain: $service.$DOMAIN"
    echo "  Policy name: $(echo "$service" | sed 's/.*/\u&/') Access"
    echo "  Include rule: Emails → $EMAIL"
    echo ""
done

open_and_wait "https://one.dash.cloudflare.com/" "Create Access Applications"

echo -e "${GREEN}🔗 Step 4: Configure Tunnel Public Hostnames${NC}"
echo "In Zero Trust → Networks → Tunnels:"
echo "1. Find your tunnel: morloksmaze-media-tunnel"
echo "2. Click 'Configure'"
echo "3. Go to 'Public Hostnames' tab"
echo "4. Add these hostnames:"
echo ""

for service in "${!services[@]}"; do
    port="${services[$service]}"
    echo "  Subdomain: $service"
    echo "  Domain: $DOMAIN" 
    echo "  Service: HTTP://traefik:80"
    echo "  Host Header: $service.$DOMAIN"
    echo ""
done

open_and_wait "https://one.dash.cloudflare.com/" "Configure Tunnel Hostnames"

echo -e "${GREEN}🌍 Step 5: Verify DNS Records${NC}"
echo "Check that these DNS records exist and are proxied (🧡):"
echo ""

for service in "${!services[@]}"; do
    echo "  CNAME $service → $DOMAIN (Proxied)"
done

open_and_wait "https://dash.cloudflare.com/" "Verify DNS Records"

echo -e "${GREEN}🧪 Step 6: Test Your Setup${NC}"
echo "Let's test each service..."

# Check if Docker containers are running
echo "Checking Docker containers..."
if ! docker ps | grep -q "jellyfin\|sonarr\|radarr"; then
    echo -e "${YELLOW}⚠️  Some containers may not be running. Starting them...${NC}"
    docker-compose up -d
    echo "Waiting for containers to start..."
    sleep 30
fi

echo ""
echo "Testing external access to your services:"
echo ""

for service in "${!services[@]}"; do
    url="https://$service.$DOMAIN"
    echo -e "${BLUE}🔗 Testing: $url${NC}"
    
    # Open the URL
    osascript -e "tell application \"Google Chrome\" to open location \"$url\"" &
    
    echo "  Expected: Cloudflare Access login page"
    echo "  1. Enter your email: $EMAIL"
    echo "  2. Check email for PIN code"
    echo "  3. Enter PIN to access $service"
    echo ""
    
    read -p "  Does $service work correctly? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "  ${GREEN}✅ $service working${NC}"
    else
        echo -e "  ${RED}❌ $service not working${NC}"
        echo "  Check:"
        echo "    - Application configuration in Zero Trust"
        echo "    - Tunnel public hostname settings"
        echo "    - DNS record configuration"
        echo "    - Docker container status"
    fi
    echo ""
done

echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""
echo "Your media server is now protected by Cloudflare Zero Trust!"
echo ""
echo -e "${BLUE}📋 Quick Reference:${NC}"
echo "• Jellyfin: https://jellyfin.$DOMAIN"
echo "• Sonarr: https://sonarr.$DOMAIN" 
echo "• Radarr: https://radarr.$DOMAIN"
echo "• Prowlarr: https://prowlarr.$DOMAIN"
echo "• Overseerr: https://overseerr.$DOMAIN"
echo ""
echo -e "${BLUE}🔧 Management:${NC}"
echo "• Zero Trust Dashboard: https://one.dash.cloudflare.com/"
echo "• Add users: Settings → Authentication → Add emails to policies"
echo "• View logs: Logs → Access"
echo ""
echo -e "${BLUE}🆘 Troubleshooting:${NC}"
echo "• Check container logs: docker logs [container-name]"
echo "• Verify tunnel: docker logs cloudflared"
echo "• Test internal: curl -I http://localhost:8096"
