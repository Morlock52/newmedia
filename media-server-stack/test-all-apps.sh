#!/usr/bin/env bash
set -euo pipefail

echo "🧪 Comprehensive Media Server Testing Suite"
echo "==========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

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

# Define services with their details
declare -A services=(
    ["jellyfin"]="8096:Media Server:Jellyfin"
    ["sonarr"]="8989:TV Management:Sonarr"
    ["radarr"]="7878:Movie Management:Radarr"
    ["prowlarr"]="9696:Indexer Management:Prowlarr"
    ["overseerr"]="5055:Request Management:Overseerr"
    ["traefik"]="8080:Reverse Proxy:Traefik Dashboard"
)

# Utility functions
check_service_health() {
    local service="$1"
    local port="$2"
    
    # Check if container is running
    if ! docker ps | grep -q "$service"; then
        echo "❌ Container not running"
        return 1
    fi
    
    # Check if port responds
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" 2>/dev/null || echo "000")
    
    case $http_code in
        200|302|401) echo "✅ Healthy ($http_code)" ;;
        000) echo "❌ No response" ;;
        *) echo "⚠️  Response: $http_code" ;;
    esac
}

test_external_access() {
    local service="$1"
    local url="https://$service.$DOMAIN"
    
    echo -n "  External access: "
    
    # Try to access the URL and check for Cloudflare Access or service response
    local response
    response=$(curl -s -I "$url" 2>/dev/null || echo "FAILED")
    
    if echo "$response" | grep -q "cloudflare"; then
        echo "🔐 Protected by Cloudflare"
    elif echo "$response" | grep -q "HTTP/[12].[01] [23]"; then
        echo "📡 Accessible (may not be protected)"
    else
        echo "❌ No response"
    fi
}

check_dns_record() {
    local service="$1"
    local hostname="$service.$DOMAIN"
    
    echo -n "  DNS resolution: "
    
    if nslookup "$hostname" >/dev/null 2>&1; then
        echo "✅ Resolves"
    else
        echo "❌ No DNS record"
    fi
}

# Test 1: Docker Environment
echo -e "${GREEN}🐳 Test 1: Docker Environment${NC}"
echo "=============================="

echo -n "Docker daemon: "
if docker info >/dev/null 2>&1; then
    echo "✅ Running"
else
    echo "❌ Not running"
    exit 1
fi

echo -n "Docker Compose: "
if command -v docker-compose >/dev/null 2>&1; then
    echo "✅ Available"
else
    echo "❌ Not installed"
    exit 1
fi

echo -n "Project containers: "
container_count=$(docker-compose ps -q 2>/dev/null | wc -l | tr -d ' ')
echo "$container_count containers"

echo ""

# Test 2: Service Health Check
echo -e "${GREEN}🏥 Test 2: Local Service Health${NC}"
echo "==============================="

for service_key in "${!services[@]}"; do
    IFS=':' read -r port description name <<< "${services[$service_key]}"
    
    echo -e "${BLUE}$name${NC}"
    echo -n "  Container status: "
    
    if docker ps | grep -q "$service_key"; then
        echo "✅ Running"
        
        echo -n "  Health check: "
        check_service_health "$service_key" "$port"
        
        echo -n "  Port $port: "
        if netstat -an 2>/dev/null | grep -q ":$port.*LISTEN" || lsof -i ":$port" >/dev/null 2>&1; then
            echo "✅ Listening"
        else
            echo "❌ Not listening"
        fi
    else
        echo "❌ Not running"
    fi
    echo ""
done

# Test 3: Network Connectivity
echo -e "${GREEN}🌐 Test 3: Network & DNS${NC}"
echo "========================="

echo -n "Internet connectivity: "
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    echo "✅ Online"
else
    echo "❌ Offline"
fi

echo -n "Domain resolution: "
if nslookup "$DOMAIN" >/dev/null 2>&1; then
    echo "✅ $DOMAIN resolves"
else
    echo "❌ $DOMAIN does not resolve"
fi

echo ""
echo "Service DNS records:"
for service_key in "${!services[@]}"; do
    IFS=':' read -r port description name <<< "${services[$service_key]}"
    echo -e "${BLUE}$service_key.$DOMAIN${NC}"
    check_dns_record "$service_key"
done

echo ""

# Test 4: Cloudflare Tunnel
echo -e "${GREEN}🔗 Test 4: Cloudflare Tunnel${NC}"
echo "============================="

echo -n "Cloudflared container: "
if docker ps | grep -q cloudflared; then
    echo "✅ Running"
    
    echo ""
    echo "Recent tunnel logs:"
    echo "=================="
    docker logs cloudflared --tail 10
    echo ""
    
    echo -n "Tunnel connection: "
    if docker logs cloudflared --tail 20 | grep -q "Registered tunnel connection"; then
        echo "✅ Connected"
    else
        echo "❌ Not connected"
    fi
    
    echo -n "Tunnel health: "
    if docker logs cloudflared --tail 20 | grep -q "Tunnel started"; then
        echo "✅ Healthy"
    else
        echo "⚠️  Check logs"
    fi
else
    echo "❌ Not running"
    echo "Starting cloudflared..."
    docker-compose up -d cloudflared
    sleep 10
fi

echo ""

# Test 5: External Access
echo -e "${GREEN}🌍 Test 5: External Access${NC}"
echo "=========================="

for service_key in "${!services[@]}"; do
    IFS=':' read -r port description name <<< "${services[$service_key]}"
    
    echo -e "${BLUE}$name (https://$service_key.$DOMAIN)${NC}"
    test_external_access "$service_key"
done

echo ""

# Test 6: Interactive Authentication Test
echo -e "${GREEN}🔐 Test 6: Authentication Flow${NC}"
echo "=============================="

echo "Testing Cloudflare Zero Trust authentication..."
echo ""

read -p "Do you want to test the authentication flow interactively? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "I'll open each service for you to test the authentication flow."
    echo "For each service, you should see:"
    echo "1. Cloudflare Access login page"
    echo "2. Enter your email: $EMAIL"
    echo "3. Check email for PIN code"
    echo "4. Enter PIN to access the service"
    echo ""
    
    for service_key in "${!services[@]}"; do
        IFS=':' read -r port description name <<< "${services[$service_key]}"
        url="https://$service_key.$DOMAIN"
        
        echo -e "${PURPLE}Testing: $name${NC}"
        echo "URL: $url"
        
        # Open the service
        osascript -e "tell application \"Google Chrome\" to open location \"$url\""
        
        echo ""
        read -p "Did the authentication work and can you access $name? (y/n): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}✅ $name authentication working${NC}"
        else
            echo -e "${RED}❌ $name authentication failed${NC}"
            
            echo "Troubleshooting steps for $name:"
            echo "• Check Zero Trust → Access → Applications for $service_key.$DOMAIN"
            echo "• Verify tunnel public hostname for $service_key.$DOMAIN"
            echo "• Check DNS record exists and is proxied"
            echo "• Try incognito/private browsing mode"
            echo "• Check container logs: docker logs $service_key"
        fi
        echo ""
    done
fi

# Test 7: Port Conflicts
echo -e "${GREEN}🚨 Test 7: Port Conflicts${NC}"
echo "========================"

ports=("80" "443" "8080" "8096" "8989" "7878" "9696" "5055")

for port in "${ports[@]}"; do
    echo -n "Port $port: "
    
    if lsof -i ":$port" >/dev/null 2>&1; then
        process=$(lsof -i ":$port" 2>/dev/null | tail -n +2 | head -n 1 | awk '{print $1}')
        echo "🔶 In use by $process"
    else
        echo "✅ Available"
    fi
done

echo ""

# Test 8: Logs Check
echo -e "${GREEN}📋 Test 8: Recent Error Logs${NC}"
echo "============================"

echo "Checking for recent errors in container logs..."
echo ""

for service_key in "${!services[@]}"; do
    if docker ps | grep -q "$service_key"; then
        echo -e "${BLUE}$service_key logs:${NC}"
        
        # Get last 5 lines and check for errors
        logs=$(docker logs "$service_key" --tail 5 2>&1)
        
        if echo "$logs" | grep -i "error\|failed\|exception" >/dev/null; then
            echo -e "${RED}⚠️  Errors found:${NC}"
            echo "$logs" | grep -i "error\|failed\|exception" | tail -3
        else
            echo "✅ No recent errors"
        fi
        echo ""
    fi
done

# Summary Report
echo -e "${GREEN}📊 Test Summary Report${NC}"
echo "======================"
echo ""

# Count working services
working_local=0
working_containers=0
total_services=${#services[@]}

for service_key in "${!services[@]}"; do
    if docker ps | grep -q "$service_key"; then
        ((working_containers++))
    fi
    
    IFS=':' read -r port description name <<< "${services[$service_key]}"
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port" 2>/dev/null || echo "000")
    if [[ "$http_code" =~ ^[23] ]]; then
        ((working_local++))
    fi
done

echo "📦 Containers: $working_containers/$total_services running"
echo "🏥 Services: $working_local/$total_services responding locally"

if docker ps | grep -q cloudflared; then
    echo "🔗 Tunnel: Running"
else
    echo "🔗 Tunnel: Not running"
fi

echo ""
echo -e "${BLUE}🔗 Your Service URLs:${NC}"
for service_key in "${!services[@]}"; do
    IFS=':' read -r port description name <<< "${services[$service_key]}"
    echo "• $name: https://$service_key.$DOMAIN"
done

echo ""
echo -e "${BLUE}🛠️  Quick Commands:${NC}"
echo "• View all containers: docker-compose ps"
echo "• Restart service: docker-compose restart [service]"
echo "• View logs: docker logs [service]"
echo "• Stop all: docker-compose down"
echo "• Start all: docker-compose up -d"

echo ""
echo -e "${BLUE}🆘 Support Resources:${NC}"
echo "• Zero Trust Dashboard: https://one.dash.cloudflare.com/"
echo "• Tunnel Management: Zero Trust → Networks → Tunnels"
echo "• DNS Records: https://dash.cloudflare.com/$DOMAIN/dns"
echo "• Access Apps: Zero Trust → Access → Applications"

echo ""
if [[ $working_containers -eq $total_services ]] && docker ps | grep -q cloudflared; then
    echo -e "${GREEN}🎉 All systems operational!${NC}"
else
    echo -e "${YELLOW}⚠️  Some services need attention. Check the details above.${NC}"
fi
