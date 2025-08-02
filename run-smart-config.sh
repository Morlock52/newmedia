#!/bin/bash

# Non-interactive smart configuration script
echo "🚀 Running Smart Auto-Configuration for Media Server"
echo "===================================================="

# Check if smart-auto-configure.sh exists
if [ ! -f "scripts/smart-auto-configure.sh" ]; then
    echo "❌ Error: scripts/smart-auto-configure.sh not found"
    exit 1
fi

# Make it executable
chmod +x scripts/smart-auto-configure.sh

# Run the smart auto-configure script
echo "Starting configuration of existing services..."
./scripts/smart-auto-configure.sh

echo ""
echo "✅ Configuration complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Check Prowlarr at http://localhost:9696 for configured indexers"
echo "2. Verify ARR apps are connected at:"
echo "   - Sonarr: http://localhost:8989"
echo "   - Radarr: http://localhost:7878"
echo "3. Check download clients:"
echo "   - qBittorrent: http://localhost:8090 (admin/adminadmin)"
echo "4. Access main dashboard at http://localhost:7575"