#!/bin/bash

# Cloudflare Cache Purge Helper for PSScript
# Helps clear CDN cache after deployment

echo "☁️  Cloudflare Cache Purge Helper"
echo "================================"

# Configuration (you'll need to add your details)
CLOUDFLARE_EMAIL="your-email@example.com"
CLOUDFLARE_API_KEY="your-global-api-key"
CLOUDFLARE_ZONE_ID="your-zone-id"
DOMAIN="psscript.morloksmaze.com"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}⚠️  Before using this script:${NC}"
echo "1. Get your Cloudflare API credentials:"
echo "   - Email: Your Cloudflare account email"
echo "   - Global API Key: Profile → API Tokens → View Global API Key"
echo "   - Zone ID: Overview tab for your domain"
echo ""
echo "2. Update the configuration variables in this script"
echo ""

read -p "Have you updated the configuration? (y/n): " CONFIGURED

if [ "$CONFIGURED" != "y" ]; then
    echo -e "${RED}Please update the configuration first.${NC}"
    echo ""
    echo "Alternatively, you can purge manually:"
    echo "1. Log in to Cloudflare Dashboard"
    echo "2. Select your domain"
    echo "3. Go to Caching → Configuration"
    echo "4. Click 'Purge Everything' or use Custom Purge"
    exit 0
fi

echo ""
echo -e "${YELLOW}🎯 Purge Options:${NC}"
echo "1. Purge everything (recommended after major updates)"
echo "2. Purge specific files only"
echo "3. Show manual instructions"
echo ""
read -p "Select option (1-3): " PURGE_OPTION

case $PURGE_OPTION in
    1)
        echo -e "${YELLOW}🧹 Purging all cache...${NC}"
        
        response=$(curl -s -X POST \
            "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/purge_cache" \
            -H "X-Auth-Email: $CLOUDFLARE_EMAIL" \
            -H "X-Auth-Key: $CLOUDFLARE_API_KEY" \
            -H "Content-Type: application/json" \
            --data '{"purge_everything":true}')
        
        if echo "$response" | grep -q '"success":true'; then
            echo -e "${GREEN}✅ Cache purged successfully!${NC}"
            echo "Wait 30 seconds for changes to propagate."
        else
            echo -e "${RED}❌ Error purging cache${NC}"
            echo "$response"
        fi
        ;;
        
    2)
        echo -e "${YELLOW}📝 Purging specific files...${NC}"
        
        # Files to purge
        FILES_TO_PURGE=(
            "https://$DOMAIN/index.html"
            "https://$DOMAIN/js/navigation-fix.js"
            "https://$DOMAIN/js/button-fix.js"
            "https://$DOMAIN/js/service-validator.js"
            "https://$DOMAIN/js/error-handler.js"
            "https://$DOMAIN/js/navigation-manager.js"
        )
        
        # Convert array to JSON
        FILES_JSON=$(printf '"%s",' "${FILES_TO_PURGE[@]}")
        FILES_JSON="[${FILES_JSON%,}]"
        
        response=$(curl -s -X POST \
            "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/purge_cache" \
            -H "X-Auth-Email: $CLOUDFLARE_EMAIL" \
            -H "X-Auth-Key: $CLOUDFLARE_API_KEY" \
            -H "Content-Type: application/json" \
            --data "{\"files\":$FILES_JSON}")
        
        if echo "$response" | grep -q '"success":true'; then
            echo -e "${GREEN}✅ Specific files purged!${NC}"
        else
            echo -e "${RED}❌ Error purging files${NC}"
            echo "$response"
        fi
        ;;
        
    3)
        echo -e "${YELLOW}📋 Manual Cache Purge Instructions:${NC}"
        echo ""
        echo "1. Go to: https://dash.cloudflare.com"
        echo "2. Select your domain: psscript.morloksmaze.com"
        echo "3. Navigate to: Caching → Configuration"
        echo ""
        echo "Option A - Purge Everything:"
        echo "- Click 'Purge Everything'"
        echo "- Confirm the action"
        echo ""
        echo "Option B - Custom Purge (Specific Files):"
        echo "- Click 'Custom Purge'"
        echo "- Select 'URL' tab"
        echo "- Enter these URLs one by one:"
        echo "  • https://psscript.morloksmaze.com/index.html"
        echo "  • https://psscript.morloksmaze.com/js/navigation-fix.js"
        echo "  • https://psscript.morloksmaze.com/js/button-fix.js"
        echo "  • https://psscript.morloksmaze.com/js/service-validator.js"
        echo "  • https://psscript.morloksmaze.com/js/error-handler.js"
        echo "  • https://psscript.morloksmaze.com/js/navigation-manager.js"
        echo "- Click 'Purge'"
        ;;
        
    *)
        echo -e "${RED}Invalid option.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}⏱️  Cache Propagation:${NC}"
echo "- Changes typically take 30-60 seconds"
echo "- Test in incognito/private mode"
echo "- Check cf-cache-status header in DevTools"

echo ""
echo -e "${YELLOW}🧪 Quick Test Commands:${NC}"
echo "# Check if cache was cleared:"
echo "curl -I https://psscript.morloksmaze.com | grep cf-cache-status"
echo ""
echo "# Test specific file:"
echo "curl -I https://psscript.morloksmaze.com/js/navigation-fix.js | grep cf-cache-status"
echo ""
echo "Status meanings:"
echo "- HIT = Served from cache (old version)"
echo "- MISS = Served from origin (new version)"
echo "- EXPIRED = Cache expired, fetching new"