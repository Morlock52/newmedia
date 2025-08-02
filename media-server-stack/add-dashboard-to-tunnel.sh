#!/usr/bin/env bash
set -euo pipefail

echo "🏠 Adding Homarr Dashboard to Cloudflare Tunnel"
echo "=============================================="

# Load environment
source .env

echo "Domain: $DOMAIN"
echo "Dashboard URLs:"
echo "• Main: https://$DOMAIN (root domain)"
echo "• Alt: https://dashboard.$DOMAIN"
echo "• Alt: https://homarr.$DOMAIN"
echo ""

# Check if API key is configured
if [[ "$CF_API_KEY" == "your-cloudflare-api-key" ]]; then
    echo "⚠️  API key not configured. You'll need to add manually:"
    echo ""
    echo "In Cloudflare Zero Trust → Networks → Tunnels → morloksmaze-media-tunnel:"
    echo "Add these public hostnames (all point to HTTP://traefik:80):"
    echo "• $DOMAIN → HTTP://traefik:80"
    echo "• dashboard.$DOMAIN → HTTP://traefik:80"
    echo "• homarr.$DOMAIN → HTTP://traefik:80"
    exit 0
fi

echo "🔧 Using API to add dashboard hostnames..."

# Get account and tunnel IDs
token_data=$(echo "$CLOUDFLARE_TUNNEL_TOKEN" | base64 -d 2>/dev/null | jq . 2>/dev/null)
account_id=$(echo "$token_data" | jq -r '.a')
tunnel_id=$(echo "$token_data" | jq -r '.t')

echo "Account ID: $account_id"
echo "Tunnel ID: $tunnel_id"

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

echo ""
echo "📋 Getting current tunnel configuration..."

# Get current tunnel config
tunnel_config=$(call_cf_api "accounts/$account_id/cfd_tunnel/$tunnel_id/configurations")

if ! echo "$tunnel_config" | jq -e '.success' >/dev/null 2>&1; then
    echo "❌ Failed to get tunnel configuration"
    exit 1
fi

echo "✅ Retrieved current configuration"

# Add dashboard hostnames to existing configuration
additional_hostnames=(
    "$DOMAIN"
    "dashboard.$DOMAIN"
    "homarr.$DOMAIN"
)

echo ""
echo "➕ Adding dashboard hostnames..."

# Get existing ingress rules
existing_ingress=$(echo "$tunnel_config" | jq -r '.result.config.ingress')

# Add new hostnames to ingress rules
new_rules=""
for hostname in "${additional_hostnames[@]}"; do
    echo "  Adding: $hostname"
    
    if [[ -n "$new_rules" ]]; then
        new_rules+=","
    fi
    
    new_rules+="{
        \"hostname\": \"$hostname\",
        \"service\": \"http://traefik:80\",
        \"originRequest\": {
            \"httpHostHeader\": \"$hostname\"
        }
    }"
done

# Combine with existing rules (remove the catch-all first)
existing_without_catchall=$(echo "$existing_ingress" | jq '.[:-1]')
combined_rules=$(echo "$existing_without_catchall" | jq --argjson new "[$new_rules]" '. + $new | . + [{"service": "http_status:404"}]')

# Create updated configuration
config_data=$(jq -n --argjson ingress "$combined_rules" '{
    "config": {
        "ingress": $ingress
    }
}')

echo ""
echo "🔄 Updating tunnel configuration..."

# Update tunnel configuration
update_response=$(call_cf_api "accounts/$account_id/cfd_tunnel/$tunnel_id/configurations" "PUT" "$config_data")

if echo "$update_response" | jq -e '.success' >/dev/null 2>&1; then
    echo "✅ Tunnel configuration updated!"
    
    echo ""
    echo "📋 Dashboard URLs added:"
    for hostname in "${additional_hostnames[@]}"; do
        echo "  ✅ https://$hostname"
    done
    
    echo ""
    echo "🔄 Restarting tunnel..."
    docker-compose restart cloudflared
    
    echo "⏳ Waiting for tunnel to reconnect..."
    sleep 15
    
    echo ""
    echo "🧪 Testing main dashboard..."
    
    # Test the main domain
    if curl -s -I "https://$DOMAIN" | grep -q "HTTP/[12]"; then
        echo "🎉 SUCCESS! Dashboard is accessible!"
        echo ""
        echo "🌐 Your dashboard is now available at:"
        echo "• https://$DOMAIN (main)"
        echo "• https://dashboard.$DOMAIN"
        echo "• https://homarr.$DOMAIN"
        
        echo ""
        echo "🚀 Opening your dashboard..."
        osascript -e "tell application \"Google Chrome\" to open location \"https://$DOMAIN\""
        
    else
        echo "⏳ Dashboard may still be loading..."
        echo "Try again in 30 seconds: https://$DOMAIN"
    fi
    
else
    echo "❌ Failed to update tunnel configuration"
    echo "Error: $(echo "$update_response" | jq -r '.errors[0].message // "Unknown error"')"
fi

echo ""
echo "🎯 What's Next:"
echo "=============="
echo "1. ✅ Homarr dashboard deployed"
echo "2. ✅ Root domain configured"
echo "3. 🔐 Set up Cloudflare Zero Trust for all services"
echo "4. 🎨 Customize your dashboard layout"
echo "5. 📊 Add service integrations for live stats"
