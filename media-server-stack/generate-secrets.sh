#!/usr/bin/env bash
set -euo pipefail

echo "🔐 Generating Missing Secrets for Complete Media Stack"
echo "===================================================="

# Create secrets directory if it doesn't exist
mkdir -p secrets

echo "Generating missing secrets..."

# PhotoPrism admin password
if [[ ! -f "secrets/photoprism_admin_password.txt" ]]; then
    openssl rand -base64 32 > secrets/photoprism_admin_password.txt
    echo "✅ Generated PhotoPrism admin password"
fi

# Ensure all existing secrets exist with defaults if missing
if [[ ! -f "secrets/traefik_dashboard_auth.txt" ]]; then
    echo "admin:$(openssl passwd -apr1 changeme)" > secrets/traefik_dashboard_auth.txt
    echo "✅ Generated Traefik dashboard auth (admin/changeme)"
fi

if [[ ! -f "secrets/wg_private_key.txt" ]]; then
    echo "# Place your WireGuard private key here" > secrets/wg_private_key.txt
    echo "✅ Created WireGuard private key placeholder"
fi

# Set proper permissions
chmod 600 secrets/*

echo ""
echo "🎯 Secret Status:"
echo "=================="
echo "• Traefik Dashboard: admin/changeme"
echo "• PhotoPrism Admin: $(cat secrets/photoprism_admin_password.txt)"
echo "• WireGuard Key: $(if grep -q "Place your" secrets/wg_private_key.txt 2>/dev/null; then echo "⚠️  Placeholder - needs real key"; else echo "✅ Configured"; fi)"

echo ""
echo "🔒 Secrets generated successfully!"
