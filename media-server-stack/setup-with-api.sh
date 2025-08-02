#!/usr/bin/env bash
set -euo pipefail

echo "🔐 Enhanced Cloudflare Authentication Setup with API Integration"
echo "=============================================================="
echo ""
echo "Using your existing configuration:"
echo "• Domain: morloksmaze.com"
echo "• Email: admin@morloksmaze.com"
echo "• Tunnel: morloksmaze-media-tunnel"
echo "• API Integration: Available"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load environment
source .env

# Check if API key is configured
if [[ "$CF_API_KEY" == "your-cloudflare-api-key" ]]; then
    echo -e "${YELLOW}⚠️  API Key not configured in .env${NC}"
    echo "Please update CF_API_KEY in your .env file with your real Cloudflare Global API key"
    echo ""
    read -p "Do you want to enter it now? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Enter your Cloudflare Global API key:"
        read -s api_key
        
        # Update .env file
        sed -i.bak "s/CF_API_KEY=your-cloudflare-api-key/CF_API_KEY=$api_key/" .env
        source .env
        echo "✅ API key updated in .env"
    else
        echo "You can update it later in .env and re-run this script"
    fi
    echo ""
fi

# Function to call Cloudflare API
call_cf_api() {
    local endpoint="$1"
    local method="${2:-GET}"
    local data="${3:-}"
    
    local curl_args=(-s -X "$method" \
                    -H "X-Auth-Email: $CF_API_EMAIL" \
                    -H "X-Auth-Key: $CF_API_KEY" \
                    -H "Content-Type: application/json")
    
    if [[ -n "$data" ]]; then
        curl_args+=(-d "$data")
    fi
    
    curl "${curl_args[@]}" "https://api.cloudflare.com/v4/$endpoint"
}

echo -e "${GREEN}Step 1: Verify API Access and Get Zone Info${NC}"

if [[ "$CF_API_KEY" != "your-cloudflare-api-key" ]]; then
    echo "Testing Cloudflare API access..."
    
    # Get zone information
    zone_response=$(call_cf_api "zones?name=$DOMAIN")
    
    if echo "$zone_response" | jq -e '.success' >/dev/null 2>&1; then
        zone_id=$(echo "$zone_response" | jq -r '.result[0].id')
        zone_name=$(echo "$zone_response" | jq -r '.result[0].name')
        
        echo "✅ API access verified"
        echo "✅ Zone found: $zone_name (ID: $zone_id)"
        
        # Update zone ID in .env if it's placeholder
        if [[ "$CLOUDFLARE_ZONE_ID" == "your-cloudflare-zone-id" ]]; then
            sed -i.bak "s/CLOUDFLARE_ZONE_ID=your-cloudflare-zone-id/CLOUDFLARE_ZONE_ID=$zone_id/" .env
            echo "✅ Zone ID updated in .env"
        fi
        
        # Check existing DNS records
        echo ""
        echo "Checking existing DNS records..."
        dns_response=$(call_cf_api "zones/$zone_id/dns_records?type=CNAME")
        
        services=("jellyfin" "sonarr" "radarr" "prowlarr" "overseerr" "traefik")
        missing_records=()
        
        for service in "${services[@]}"; do
            if echo "$dns_response" | jq -e ".result[] | select(.name == \"$service.$DOMAIN\")" >/dev/null; then
                echo "✅ DNS record exists: $service.$DOMAIN"
            else
                echo "❌ Missing DNS record: $service.$DOMAIN"
                missing_records+=("$service")
            fi
        done
        
        # Offer to create missing records
        if [[ ${#missing_records[@]} -gt 0 ]]; then
            echo ""
            read -p "Create missing DNS records automatically? (y/n): " -n 1 -r
            echo ""
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                for service in "${missing_records[@]}"; do
                    echo "Creating CNAME record for $service.$DOMAIN..."
                    
                    record_data=$(cat <<EOF
{
  "type": "CNAME",
  "name": "$service",
  "content": "$DOMAIN",
  "proxied": true,
  "ttl": 1
}
EOF
)
                    
                    create_response=$(call_cf_api "zones/$zone_id/dns_records" "POST" "$record_data")
                    
                    if echo "$create_response" | jq -e '.success' >/dev/null; then
                        echo "✅ Created: $service.$DOMAIN"
                    else
                        echo "❌ Failed to create: $service.$DOMAIN"
                        echo "Error: $(echo "$create_response" | jq -r '.errors[0].message // "Unknown error"')"
                    fi
                done
            fi
        fi
        
    else
        echo "❌ API access failed"
        echo "Error: $(echo "$zone_response" | jq -r '.errors[0].message // "Unknown error"')"
        echo "Please check your API key and email in .env"
    fi
else
    echo "⚠️  Skipping API checks - API key not configured"
fi

echo ""
read -p "Press ENTER to continue to tunnel setup..."

echo -e "${GREEN}Step 2: Check Tunnel Status${NC}"

# Check tunnel container
if docker ps | grep -q cloudflared; then
    echo "✅ Cloudflared container is running"
    
    # Show recent logs
    echo ""
    echo "Recent tunnel logs:"
    echo "=================="
    docker logs cloudflared --tail 10
    echo ""
    
    if docker logs cloudflared --tail 20 | grep -q "Registered tunnel connection"; then
        echo "✅ Tunnel connection established"
    else
        echo "⚠️  Tunnel may not be fully connected"
        echo "Restarting tunnel..."
        docker-compose restart cloudflared
        sleep 10
    fi
else
    echo "❌ Cloudflared container not running"
    echo "Starting tunnel..."
    docker-compose up -d cloudflared
    sleep 15
fi

echo ""
read -p "Press ENTER to continue to Zero Trust configuration..."

echo -e "${GREEN}Step 3: Configure Zero Trust (Manual Steps)${NC}"
echo "Opening Cloudflare Zero Trust dashboard..."

# Open Zero Trust dashboard
osascript -e 'tell application "Google Chrome" to open location "https://one.dash.cloudflare.com/"'

echo ""
echo "In the Zero Trust dashboard, complete these steps:"
echo ""
echo "🔐 Authentication Setup:"
echo "1. Go to Settings → Authentication"
echo "2. Click 'Add new' if no auth method exists"
echo "3. Select 'One-time PIN'"
echo "4. Save configuration"
echo ""
read -p "Press ENTER after configuring authentication..."

echo ""
echo "🔗 Tunnel Public Hostnames:"
echo "1. Go to Networks → Tunnels"
echo "2. Find and click: 'morloksmaze-media-tunnel'"
echo "3. Go to 'Public Hostnames' tab"
echo "4. Add these hostnames:"
echo ""

# Service configurations
declare -A services=(
    ["jellyfin"]="Media Server"
    ["sonarr"]="TV Shows"
    ["radarr"]="Movies"
    ["prowlarr"]="Indexers"
    ["overseerr"]="Requests"
    ["traefik"]="Dashboard"
)

for service in "${!services[@]}"; do
    echo "   ${service}.morloksmaze.com → HTTP://traefik:80"
    echo "   (Host header: ${service}.morloksmaze.com)"
done

echo ""
echo "⚠️  Important: Use 'traefik:80' as the service URL!"
echo ""
read -p "Press ENTER after adding all public hostnames..."

echo ""
echo "🛡️  Access Applications:"
echo "1. Go to Access → Applications"
echo "2. Click 'Add an application' → 'Self-hosted'"
echo "3. Create applications for each service:"
echo ""

for service in "${!services[@]}"; do
    description="${services[$service]}"
    echo "   Application: ${service^} ($description)"
    echo "   Domain: ${service}.morloksmaze.com"
    echo "   Policy: Allow email → admin@morloksmaze.com"
    echo ""
done

read -p "Press ENTER after creating all Access applications..."

echo ""
echo -e "${GREEN}Step 4: Start Services and Test${NC}"

echo "Starting all services..."
docker-compose up -d

echo "Waiting for services to start..."
sleep 20

echo ""
echo "Testing services locally first..."

declare -A service_ports=(
    ["jellyfin"]="8096"
    ["sonarr"]="8989"
    ["radarr"]="7878"
    ["prowlarr"]="9696"
    ["overseerr"]="5055"
    ["traefik"]="8080"
)

for service in "${!service_ports[@]}"; do
    port="${service_ports[$service]}"
    echo -n "Testing $service (localhost:$port)... "
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" | grep -q "200\|302\|401"; then
        echo "✅ OK"
    else
        echo "❌ Not responding"
    fi
done

echo ""
echo -e "${GREEN}Step 5: Test External Access with Authentication${NC}"

echo "Now testing external access through Cloudflare..."
echo ""

for service in "${!services[@]}"; do
    url="https://${service}.morloksmaze.com"
    
    echo -e "${BLUE}🧪 Testing: ${service^} (${services[$service]})${NC}"
    echo "URL: $url"
    
    # Open the service URL
    osascript -e "tell application \"Google Chrome\" to open location \"$url\""
    
    echo ""
    echo "Expected authentication flow:"
    echo "1. Cloudflare Access login page"
    echo "2. Enter: admin@morloksmaze.com"
    echo "3. Check email for PIN code"
    echo "4. Enter PIN to access ${service^}"
    echo ""
    
    read -p "Did ${service^} authentication work? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✅ ${service^} working correctly${NC}"
    else
        echo -e "${RED}❌ ${service^} authentication failed${NC}"
        echo ""
        echo "Troubleshooting checklist:"
        echo "• Access application exists for ${service}.morloksmaze.com"
        echo "• Public hostname configured in tunnel"
        echo "• DNS record exists and proxied (🧡)"
        echo "• Service running locally on port ${service_ports[$service]}"
        echo "• Try incognito/private browsing"
    fi
    echo ""
done

echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "==================="

# Final summary
echo ""
echo -e "${BLUE}📱 Your Secure Media Server:${NC}"
for service in "${!services[@]}"; do
    echo "• ${service^}: https://${service}.morloksmaze.com"
done

echo ""
echo -e "${BLUE}🔑 Access Instructions:${NC}"
echo "1. Visit any service URL"
echo "2. Cloudflare Access login appears"
echo "3. Enter: admin@morloksmaze.com"
echo "4. Check email for PIN code"
echo "5. Enter PIN to access service"

echo ""
echo -e "${BLUE}🛠️  Management Links:${NC}"
echo "• Zero Trust: https://one.dash.cloudflare.com/"
echo "• DNS Management: https://dash.cloudflare.com/morloksmaze.com/dns"
echo "• Tunnel Management: https://one.dash.cloudflare.com/ → Networks → Tunnels"

echo ""
echo -e "${BLUE}📊 Current Status:${NC}"

# Check tunnel status
if docker logs cloudflared --tail 5 | grep -q "Registered tunnel connection"; then
    echo "✅ Tunnel: Connected"
else
    echo "⚠️  Tunnel: Check connection"
fi

# Check services
running_count=0
for service in "${!service_ports[@]}"; do
    if docker ps | grep -q "$service"; then
        ((running_count++))
    fi
done

echo "✅ Services: $running_count/${#service_ports[@]} running"

echo ""
echo "🎊 Your media server is now secured with Cloudflare Zero Trust!"
echo "📧 Remember: You'll get a PIN code via email for each login."
