#!/bin/bash

# PSScript Website Deployment Validation Script
# Tests if the fixes have been successfully deployed

echo "🧪 PSScript Website Deployment Test"
echo "==================================="

WEBSITE_URL="https://psscript.morloksmaze.com"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Testing website: $WEBSITE_URL${NC}"
echo ""

# Function to test URL
test_url() {
    local url=$1
    local description=$2
    
    echo -n "Testing $description... "
    
    # Test with curl
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ OK (200)${NC}"
        return 0
    elif [ "$response" = "404" ]; then
        echo -e "${RED}✗ 404 Not Found${NC}"
        return 1
    else
        echo -e "${YELLOW}⚠ Status: $response${NC}"
        return 1
    fi
}

# Test main pages
echo -e "${YELLOW}📄 Testing main pages:${NC}"
test_url "$WEBSITE_URL" "Homepage"
test_url "$WEBSITE_URL/dashboard" "Dashboard"
test_url "$WEBSITE_URL/movies" "Movies"
test_url "$WEBSITE_URL/series" "Series"
test_url "$WEBSITE_URL/music" "Music"
test_url "$WEBSITE_URL/live" "Live"
test_url "$WEBSITE_URL/analytics" "Analytics"

echo ""
echo -e "${YELLOW}📁 Testing JavaScript files:${NC}"
test_url "$WEBSITE_URL/js/navigation-fix.js" "navigation-fix.js"
test_url "$WEBSITE_URL/js/button-fix.js" "button-fix.js"
test_url "$WEBSITE_URL/js/service-validator.js" "service-validator.js"
test_url "$WEBSITE_URL/js/error-handler.js" "error-handler.js"
test_url "$WEBSITE_URL/js/navigation-manager.js" "navigation-manager.js"

echo ""
echo -e "${YELLOW}🔍 Checking for common issues:${NC}"

# Check if index.html contains fix script references
echo -n "Checking if index.html loads fix scripts... "
if curl -s "$WEBSITE_URL" | grep -q "navigation-fix.js"; then
    echo -e "${GREEN}✓ Yes${NC}"
else
    echo -e "${RED}✗ No - Fix scripts may not be loaded${NC}"
fi

# Test 404 handling
echo -n "Testing 404 error handling... "
response=$(curl -s -o /dev/null -w "%{http_code}" "$WEBSITE_URL/this-page-does-not-exist")
if [ "$response" = "404" ] || [ "$response" = "200" ]; then
    echo -e "${GREEN}✓ Handled${NC}"
else
    echo -e "${YELLOW}⚠ Unexpected status: $response${NC}"
fi

echo ""
echo -e "${YELLOW}🌐 CDN Cache Status:${NC}"
echo "To check Cloudflare cache status:"
echo "1. Open browser DevTools (F12)"
echo "2. Go to Network tab"
echo "3. Reload the page"
echo "4. Look for 'cf-cache-status' header"
echo "   - HIT = Served from cache (needs purge)"
echo "   - MISS = Fresh from server"

echo ""
echo -e "${YELLOW}📱 Manual Testing Required:${NC}"
echo "Please manually test these features:"
echo "- [ ] Navigation buttons are clickable (not disabled)"
echo "- [ ] Keyboard shortcuts work (press 1-6)"
echo "- [ ] Ctrl+K opens quick navigation"
echo "- [ ] Service status panel appears"
echo "- [ ] Error messages show user-friendly text"
echo "- [ ] Mobile navigation works properly"

echo ""
echo -e "${YELLOW}🔧 Troubleshooting Tips:${NC}"
echo "1. If files return 404:"
echo "   - Check file permissions (should be 644)"
echo "   - Verify correct upload path"
echo "   - Check .htaccess rules"
echo ""
echo "2. If fixes don't work:"
echo "   - Clear browser cache completely"
echo "   - Test in incognito/private mode"
echo "   - Check browser console for errors"
echo "   - Verify CDN cache was purged"
echo ""
echo "3. Browser Console Commands:"
echo "   - Check if fixes loaded: typeof NavigationFix"
echo "   - Check if buttons fixed: typeof ButtonFix"
echo "   - View service status: ServiceValidator.showStatus()"