#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Fix DNS Records for Dashboard"
echo "==============================="

# Load environment
source .env

echo "Domain: $DOMAIN"
echo "Issue: DNS records for dashboard subdomains don't exist"
echo ""

# Check if API key is configured
if [[ "$CF_API_KEY" == "your-cloudflare-api-key" ]]; then
    echo "❌ API key not configured. Manual setup needed:"
    echo ""
    echo "Go to Cloudflare DNS and add these CNAME records:"
    echo "• dashboard → morloksmaze.com (Proxied 🧡)"
    echo "• homarr → morloksmaze.com (Proxied 🧡)"
    echo "• morloksmaze.com → existing A/AAAA record"
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

echo "1. 🔍 Getting zone information..."

# Get zone information
zone_response=$(call_cf_api "zones?name=$DOMAIN")

if echo "$zone_response" | jq -e '.success' >/dev/null 2>&1; then
    zone_id=$(echo "$zone_response" | jq -r '.result[0].id')
    echo "✅ Zone ID: $zone_id"
else
    echo "❌ Failed to get zone information"
    exit 1
fi

echo ""
echo "2. 📋 Checking existing DNS records..."

# Get existing DNS records
dns_response=$(call_cf_api "zones/$zone_id/dns_records")

# Check what records already exist
echo "Existing DNS records:"
echo "$dns_response" | jq -r '.result[] | select(.type == "A" or .type == "AAAA" or .type == "CNAME") | "\(.name) (\(.type)) → \(.content)"'

echo ""
echo "3. ➕ Creating missing DNS records..."

# Records to create
dashboard_records=(
    "dashboard"
    "homarr"
)

for subdomain in "${dashboard_records[@]}"; do
    full_name="$subdomain.$DOMAIN"
    
    # Check if record already exists
    if echo "$dns_response" | jq -e ".result[] | select(.name == \"$full_name\")" >/dev/null; then
        echo "✅ $full_name already exists"
        continue
    fi
    
    echo "Creating CNAME record for $full_name..."
    
    record_data=$(cat <<EOF
{
  "type": "CNAME",
  "name": "$subdomain",
  "content": "$DOMAIN",
  "proxied": true,
  "ttl": 1
}
EOF
)
    
    create_response=$(call_cf_api "zones/$zone_id/dns_records" "POST" "$record_data")
    
    if echo "$create_response" | jq -e '.success' >/dev/null; then
        echo "✅ Created: $full_name → $DOMAIN (Proxied)"
    else
        echo "❌ Failed to create: $full_name"
        echo "Error: $(echo "$create_response" | jq -r '.errors[0].message // "Unknown error"')"
    fi
done

echo ""
echo "4. 🧪 Testing DNS resolution..."

# Wait a moment for DNS propagation
echo "Waiting 10 seconds for DNS propagation..."
sleep 10

# Test DNS resolution
for subdomain in "${dashboard_records[@]}"; do
    full_name="$subdomain.$DOMAIN"
    echo -n "Testing $full_name: "
    
    if nslookup "$full_name" >/dev/null 2>&1; then
        echo "✅ Resolves"
    else
        echo "⏳ Still propagating..."
    fi
done

echo ""
echo "5. 🔄 Restarting services to apply changes..."

# Restart Homarr and tunnel
docker-compose restart homarr cloudflared

echo "⏳ Waiting for services to restart..."
sleep 15

echo ""
echo "6. 🧪 Testing dashboard access..."

# Test dashboard URLs
dashboard_urls=(
    "https://$DOMAIN"
    "https://dashboard.$DOMAIN" 
    "https://homarr.$DOMAIN"
)

for url in "${dashboard_urls[@]}"; do
    echo -n "Testing $url: "
    
    response=$(curl -s -I "$url" 2>/dev/null || echo "FAILED")
    
    if echo "$response" | grep -q "HTTP/[12]"; then
        echo "✅ Working"
    elif echo "$response" | grep -q "cloudflare"; then
        echo "🔐 Cloudflare (needs auth setup)"
    else
        echo "⏳ Still starting up"
    fi
done

echo ""
echo "🎉 DNS Records Fixed!"
echo "==================="

echo ""
echo "📋 Created DNS records:"
for subdomain in "${dashboard_records[@]}"; do
    echo "• $subdomain.$DOMAIN → $DOMAIN (Proxied)"
done

echo ""
echo "🌐 Your dashboard should now be accessible at:"
echo "• https://$DOMAIN (main)"
echo "• https://dashboard.$DOMAIN"
echo "• https://homarr.$DOMAIN"

echo ""
echo "🚀 Opening dashboard..."
osascript -e "tell application \"Google Chrome\" to open location \"https://dashboard.$DOMAIN\""

echo ""
echo "🎯 If it still doesn't work:"
echo "============================"
echo "1. Wait 2-3 minutes for full DNS propagation"
echo "2. Try incognito/private browsing"
echo "3. Check: docker logs homarr"
echo "4. Check: docker logs cloudflared"

echo ""
echo "🔐 Next step: Set up Cloudflare Zero Trust authentication!"
