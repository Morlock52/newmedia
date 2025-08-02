#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Auto-Configure Cloudflare Tunnel Hostnames with API"
echo "======================================================"

# Load environment
source .env

echo "Domain: $DOMAIN"
echo "API Email: $CF_API_EMAIL"
echo "Using API to configure tunnel hostnames..."
echo ""

# Check if API key is configured
if [[ "$CF_API_KEY" == "your-cloudflare-api-key" ]]; then
    echo "❌ API key not configured in .env"
    echo "Please update CF_API_KEY in your .env file first"
    exit 1
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
    
    curl "${curl_args[@]}" "https://api.cloudflare.com/client/v4/$endpoint"
}

echo "1. 🔍 Finding your tunnel..."

# Get account ID and tunnel ID from the token
if command -v base64 >/dev/null && command -v jq >/dev/null; then
    token_data=$(echo "$CLOUDFLARE_TUNNEL_TOKEN" | base64 -d 2>/dev/null | jq . 2>/dev/null || echo "Failed")
    
    if [[ "$token_data" != "Failed" ]]; then
        account_id=$(echo "$token_data" | jq -r '.a')
        tunnel_id=$(echo "$token_data" | jq -r '.t')
        
        echo "✅ Account ID: $account_id"
        echo "✅ Tunnel ID: $tunnel_id"
        echo ""
    else
        echo "❌ Could not decode tunnel token"
        exit 1
    fi
else
    echo "❌ Missing required tools (base64, jq)"
    exit 1
fi

echo "2. 🌐 Getting current tunnel configuration..."

# Get current tunnel config
tunnel_config=$(call_cf_api "accounts/$account_id/cfd_tunnel/$tunnel_id/configurations")

if echo "$tunnel_config" | jq -e '.success' >/dev/null 2>&1; then
    echo "✅ Retrieved current tunnel configuration"
else
    echo "❌ Failed to get tunnel configuration"
    echo "Error: $(echo "$tunnel_config" | jq -r '.errors[0].message // "Unknown error"')"
    exit 1
fi

echo ""
echo "3. ➕ Adding public hostnames..."

# Services to configure
services=(
    "prowlarr:9696"
    "jellyfin:8096"
    "sonarr:8989"
    "radarr:7878"
    "overseerr:5055"
    "lidarr:8686"
    "readarr:8787"
    "bazarr:6767"
    "tautulli:8181"
    "mylar:8090"
    "podgrab:8080"
    "youtube-dl:17442"
    "photoprism:2342"
    "traefik:8080"
)

# Create ingress rules for each service
ingress_rules=""
for service_port in "${services[@]}"; do
    service="${service_port%:*}"
    echo "  Adding: $service.$DOMAIN"
    
    if [[ -n "$ingress_rules" ]]; then
        ingress_rules+=","
    fi
    
    ingress_rules+="{
        \"hostname\": \"$service.$DOMAIN\",
        \"service\": \"http://traefik:80\",
        \"originRequest\": {
            \"httpHostHeader\": \"$service.$DOMAIN\"
        }
    }"
done

# Add catch-all rule (required)
if [[ -n "$ingress_rules" ]]; then
    ingress_rules+=","
fi
ingress_rules+="{
    \"service\": \"http_status:404\"
}"

# Create the complete tunnel configuration
config_data=$(cat <<EOF
{
    "config": {
        "ingress": [$ingress_rules]
    }
}
EOF
)

echo ""
echo "4. 🔧 Updating tunnel configuration..."

# Update tunnel configuration
update_response=$(call_cf_api "accounts/$account_id/cfd_tunnel/$tunnel_id/configurations" "PUT" "$config_data")

if echo "$update_response" | jq -e '.success' >/dev/null 2>&1; then
    echo "✅ Tunnel configuration updated successfully!"
    
    echo ""
    echo "📋 Configured hostnames:"
    for service_port in "${services[@]}"; do
        service="${service_port%:*}"
        echo "  ✅ https://$service.$DOMAIN"
    done
    
else
    echo "❌ Failed to update tunnel configuration"
    echo "Error: $(echo "$update_response" | jq -r '.errors[0].message // "Unknown error"')"
    echo ""
    echo "Response: $update_response"
    exit 1
fi

echo ""
echo "5. 🔄 Restarting tunnel to apply changes..."

# Restart cloudflared to pick up new config
docker-compose restart cloudflared

echo "⏳ Waiting for tunnel to reconnect..."
sleep 15

echo ""
echo "6. 🧪 Testing configuration..."

# Test prowlarr specifically
echo "Testing https://prowlarr.$DOMAIN..."

test_response=$(curl -s -I "https://prowlarr.$DOMAIN" 2>/dev/null || echo "FAILED")

if echo "$test_response" | grep -q "HTTP/[12]"; then
    echo "🎉 SUCCESS! Prowlarr is responding!"
    echo ""
    echo "✅ Error 1033 should be fixed!"
    echo ""
    echo "🌐 All your services are now accessible:"
    for service_port in "${services[@]}"; do
        service="${service_port%:*}"
        echo "  • https://$service.$DOMAIN"
    done
    
    echo ""
    echo "🚀 Opening Prowlarr for you..."
    osascript -e 'tell application "Google Chrome" to open location "https://prowlarr.morloksmaze.com"'
    
else
    echo "⏳ Services may still be starting up..."
    echo "Wait 30 seconds and try again"
fi

echo ""
echo "🎯 Next Steps:"
echo "============="
echo "1. ✅ Tunnel hostnames configured"
echo "2. 🔐 Set up Cloudflare Zero Trust authentication"
echo "3. ⚙️ Configure your media management services"
echo "4. 🧪 Test all services: ./test-complete-stack.sh"

echo ""
echo "🎉 Your media server should now be fully accessible!"
