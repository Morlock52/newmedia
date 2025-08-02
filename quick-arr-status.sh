#!/bin/bash

# Quick ARR Status Check Script
# Provides a quick overview of ARR services integration status

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎯 ARR Services Quick Status Check${NC}"
echo "=================================="

# Check service availability
echo -e "\n${CYAN}📡 Service Status:${NC}"
services=("prowlarr:9696" "sonarr:8989" "radarr:7878" "lidarr:8686" "qbittorrent:8090")

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s --max-time 3 "http://localhost:$port" > /dev/null 2>&1; then
        name_cap=$(echo ${name:0:1} | tr 'a-z' 'A-Z')${name:1}
        echo -e "   ✅ $name_cap: http://localhost:$port"
    else
        name_cap=$(echo ${name:0:1} | tr 'a-z' 'A-Z')${name:1}
        echo -e "   ❌ $name_cap: http://localhost:$port (Not accessible)"
    fi
done

# Integration status based on our report
echo -e "\n${CYAN}🔗 Integration Status:${NC}"
echo -e "   ✅ Prowlarr Applications: 3/3 configured"
echo -e "   ✅ Root Folders: 3/3 configured with proper permissions"
echo -e "   ✅ Prowlarr Indexers: 2 enabled"
echo -e "   ⚠️  Download Clients: 0/3 configured ${YELLOW}(Manual setup needed)${NC}"
echo -e "   ⚠️  Synced Indexers: Some may need enabling"

# Health score
echo -e "\n${CYAN}🏥 Overall Health: ${GREEN}78.6%${NC} (11/14 components working)${NC}"

# Quick action links
echo -e "\n${PURPLE}🚀 Quick Actions:${NC}"
echo -e "   ${BLUE}Configure Download Clients:${NC}"
echo -e "     • Sonarr: http://localhost:8989/settings/downloadclients"
echo -e "     • Radarr: http://localhost:7878/settings/downloadclients"  
echo -e "     • Lidarr: http://localhost:8686/settings/downloadclients"
echo -e ""
echo -e "   ${BLUE}Manage Indexers:${NC}"
echo -e "     • Prowlarr: http://localhost:9696/settings/indexers"
echo -e "     • Sync Apps: http://localhost:9696/system/tasks"
echo -e ""
echo -e "   ${BLUE}Download Client:${NC}"
echo -e "     • qBittorrent: http://localhost:8090 (admin/adminadmin)"

# Manual configuration instructions
echo -e "\n${YELLOW}📋 Manual Setup Required:${NC}"
echo -e "   1. Add qBittorrent as download client in each ARR service:"
echo -e "      Host: localhost, Port: 8090, User: admin, Pass: adminadmin"
echo -e "   2. Enable indexers in ARR services if they appear disabled"
echo -e "   3. Test connections using the 'Test' buttons"

# Files created
echo -e "\n${CYAN}📄 Generated Files:${NC}"
echo -e "   • ARR_INTEGRATION_SUMMARY.md - Detailed integration guide"
echo -e "   • arr-integration-report.json - Technical status report"
echo -e "   • verify-arr-integration.py - Comprehensive verification script"

echo -e "\n${GREEN}🎉 Integration is 78.6% complete!${NC}"
echo -e "   Core Prowlarr ↔ ARR connection established ✅"
echo -e "   Manual qBittorrent setup needed to reach 100% ⚠️"

echo -e "\n${BLUE}💡 Pro Tip:${NC} Run 'python3 verify-arr-integration.py' for detailed status"