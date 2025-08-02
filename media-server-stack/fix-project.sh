#!/usr/bin/env bash
set -euo pipefail

echo "=== Media Server Stack Diagnostics & Fix ==="
echo "Checking current status..."

# Change to project directory
cd "$(dirname "$0")"

echo "1. Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

echo -e "\n2. Checking if containers are running..."
if docker ps -q | grep -q .; then
    echo "Current running containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    echo "No containers currently running"
fi

echo -e "\n3. Checking for existing containers from this project..."
if docker-compose ps -q | grep -q .; then
    echo "Containers from this project:"
    docker-compose ps
else
    echo "No containers from this project are running"
fi

echo -e "\n4. Checking environment configuration..."
if [[ -f ".env" ]]; then
    echo "✅ .env file exists"
    # Check critical settings without exposing secrets
    if grep -q "DOMAIN=morloksmaze.com" .env; then
        echo "✅ Domain configured: morloksmaze.com"
    fi
    if grep -q "VPN_PROVIDER=pia" .env; then
        echo "✅ VPN provider set to PIA"
    fi
else
    echo "❌ .env file missing"
fi

echo -e "\n5. Checking secrets..."
if [[ -f "secrets/wg_private_key.txt" ]]; then
    if [[ -s "secrets/wg_private_key.txt" ]] && ! grep -q "Place your" secrets/wg_private_key.txt; then
        echo "✅ WireGuard private key appears to be configured"
    else
        echo "⚠️  WireGuard private key is placeholder/empty"
    fi
else
    echo "❌ WireGuard private key missing"
fi

echo -e "\n6. Checking network connectivity..."
if ping -c 1 google.com &> /dev/null; then
    echo "✅ Internet connectivity working"
else
    echo "❌ No internet connectivity"
fi

echo -e "\n7. Checking Docker daemon..."
if docker info &> /dev/null; then
    echo "✅ Docker daemon is running"
else
    echo "❌ Docker daemon not running"
    exit 1
fi

echo -e "\n8. Checking for port conflicts..."
if lsof -i :80 &> /dev/null; then
    echo "⚠️  Port 80 is in use:"
    lsof -i :80
fi

if lsof -i :443 &> /dev/null; then
    echo "⚠️  Port 443 is in use:"
    lsof -i :443
fi

echo -e "\n=== RECOMMENDATIONS ==="

# Check what needs to be fixed
needs_restart=false
needs_config=false

if ! docker-compose ps -q | grep -q .; then
    echo "🔧 Containers are not running. Try: docker-compose up -d"
    needs_restart=true
fi

if [[ ! -f "secrets/wg_private_key.txt" ]] || grep -q "Place your" secrets/wg_private_key.txt 2>/dev/null; then
    echo "🔧 Need to configure VPN private key"
    needs_config=true
fi

if grep -q "your-cloudflare-api-key" .env 2>/dev/null; then
    echo "🔧 Need to configure real Cloudflare API key"
    needs_config=true
fi

if [[ "$needs_restart" == "true" ]]; then
    echo -e "\n🚀 To start the stack:"
    echo "   ./scripts/deploy.sh"
fi

if [[ "$needs_config" == "true" ]]; then
    echo -e "\n⚙️  To configure missing settings:"
    echo "   ./scripts/setup.sh"
fi

echo -e "\n📋 Quick Commands:"
echo "   View logs: docker-compose logs [service_name]"
echo "   Restart service: docker-compose restart [service_name]"
echo "   Stop all: docker-compose down"
echo "   Start all: docker-compose up -d"
echo "   Health check: ./scripts/health-check.sh"

echo -e "\n=== End Diagnostics ==="
