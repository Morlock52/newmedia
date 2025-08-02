#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Media Server Stack Auto-Fix Script"
echo "======================================"

# Change to project directory
cd "$(dirname "$0")"

echo "1. Stopping any running containers..."
docker-compose down 2>/dev/null || echo "No containers to stop"

echo "2. Fixing environment configuration..."
if [[ -f ".env.fixed" ]]; then
    cp .env .env.backup
    cp .env.fixed .env
    echo "✅ Updated .env file (backup saved as .env.backup)"
fi

echo "3. Creating required directories..."
mkdir -p data/{media,torrents,usenet}/{movies,tv,music,online-videos}
mkdir -p config/{jellyfin,sonarr,radarr,prowlarr,overseerr,qbittorrent,bazarr,traefik,gluetun}
mkdir -p secrets

echo "4. Setting proper permissions..."
chmod -R 755 data config 2>/dev/null || true
chmod -R 600 secrets/* 2>/dev/null || true

echo "5. Generating missing secrets..."
if [[ ! -f "secrets/traefik_dashboard_auth.txt" ]]; then
    echo "admin:\$(openssl passwd -apr1 changeme)" > secrets/traefik_dashboard_auth.txt
    echo "✅ Generated Traefik auth (admin/changeme)"
fi

if [[ ! -f "secrets/authelia_jwt_secret.txt" ]]; then
    openssl rand -base64 32 > secrets/authelia_jwt_secret.txt
    echo "✅ Generated Authelia JWT secret"
fi

if [[ ! -f "secrets/authelia_session_secret.txt" ]]; then
    openssl rand -base64 32 > secrets/authelia_session_secret.txt
    echo "✅ Generated Authelia session secret"
fi

if [[ ! -f "secrets/authelia_encryption_key.txt" ]]; then
    openssl rand -base64 32 > secrets/authelia_encryption_key.txt
    echo "✅ Generated Authelia encryption key"
fi

if [[ ! -f "secrets/authelia_smtp_password.txt" ]]; then
    echo "dummy-smtp-password" > secrets/authelia_smtp_password.txt
    echo "✅ Generated dummy SMTP password"
fi

chmod 600 secrets/*

echo "6. Creating Docker networks..."
docker network create traefik_network 2>/dev/null || echo "traefik_network already exists"
docker network create monitoring_network 2>/dev/null || echo "monitoring_network already exists"

echo "7. Starting essential services only..."
# Start just core services to test
docker-compose up -d traefik jellyfin sonarr radarr prowlarr

echo "8. Waiting for services to start..."
sleep 30

echo "9. Checking service health..."
services=("traefik" "jellyfin" "sonarr" "radarr" "prowlarr")
for service in "${services[@]}"; do
    if docker-compose ps "$service" | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
        echo "Logs for $service:"
        docker-compose logs --tail 10 "$service"
    fi
done

echo ""
echo "🌟 Quick Fix Complete!"
echo "====================="
echo ""
echo "🌐 Your services should be accessible at:"
echo "   • Jellyfin: http://localhost:8096 (or https://jellyfin.morloksmaze.com if tunnel works)"
echo "   • Sonarr: http://localhost:8989"
echo "   • Radarr: http://localhost:7878"
echo "   • Prowlarr: http://localhost:9696"
echo "   • Traefik Dashboard: http://localhost:8080"
echo ""
echo "🔧 Still need to configure:"
echo "   1. Real Cloudflare API key in .env"
echo "   2. Verify Cloudflare tunnel token is valid"
echo "   3. Add VPN if needed (currently disabled)"
echo ""
echo "📋 Next steps:"
echo "   • Test local access first"
echo "   • Configure Cloudflare properly for external access"
echo "   • Add qBittorrent + VPN back when basic stack works"
echo ""
echo "🆘 If something is broken:"
echo "   docker-compose logs [service-name]"
echo "   docker-compose restart [service-name]"
