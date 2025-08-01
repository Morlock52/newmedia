#!/bin/bash

# PSScript Website Fix Deployment Script
# This script helps deploy the fixes to psscript.morloksmaze.com

echo "🚀 PSScript Website Fix Deployment"
echo "=================================="

# Configuration
REMOTE_HOST="psscript.morloksmaze.com"
REMOTE_USER="your-username"  # Update this with your SSH username
REMOTE_PATH="/var/www/psscript"  # Update this with your web root path
LOCAL_PATH="/Users/morlock/fun/newmedia/holographic-dashboard"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}📋 Files to deploy:${NC}"
echo "- js/navigation-fix.js"
echo "- js/button-fix.js"
echo "- js/service-validator.js"
echo "- js/error-handler.js"
echo "- js/navigation-manager.js"
echo "- index.html"
echo ""

# Check if files exist locally
echo -e "${YELLOW}🔍 Checking local files...${NC}"
FILES_TO_DEPLOY=(
    "js/navigation-fix.js"
    "js/button-fix.js"
    "js/service-validator.js"
    "js/error-handler.js"
    "js/navigation-manager.js"
    "index.html"
)

MISSING_FILES=0
for file in "${FILES_TO_DEPLOY[@]}"; do
    if [ -f "$LOCAL_PATH/$file" ]; then
        echo -e "${GREEN}✓${NC} Found: $file"
    else
        echo -e "${RED}✗${NC} Missing: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${RED}❌ Some files are missing. Please check your local files.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}📤 Deployment Options:${NC}"
echo "1. Deploy via SSH/SCP (requires SSH access)"
echo "2. Deploy via FTP (requires FTP credentials)"
echo "3. Generate deployment package (manual upload)"
echo ""
read -p "Select deployment method (1-3): " DEPLOY_METHOD

case $DEPLOY_METHOD in
    1)
        # SSH/SCP Deployment
        echo -e "${YELLOW}🔐 SSH Deployment Selected${NC}"
        echo "Please update the script with your SSH credentials first."
        echo ""
        read -p "Have you updated REMOTE_USER and REMOTE_PATH? (y/n): " CONFIRM
        
        if [ "$CONFIRM" = "y" ]; then
            echo -e "${YELLOW}📤 Uploading files via SCP...${NC}"
            
            # Create backup directory on remote
            ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_PATH/backup-$(date +%Y%m%d)"
            
            # Backup existing files
            echo -e "${YELLOW}📦 Creating backup...${NC}"
            ssh $REMOTE_USER@$REMOTE_HOST "cp -r $REMOTE_PATH/js $REMOTE_PATH/backup-$(date +%Y%m%d)/ 2>/dev/null || true"
            ssh $REMOTE_USER@$REMOTE_HOST "cp $REMOTE_PATH/index.html $REMOTE_PATH/backup-$(date +%Y%m%d)/ 2>/dev/null || true"
            
            # Upload new files
            for file in "${FILES_TO_DEPLOY[@]}"; do
                echo -e "Uploading $file..."
                scp "$LOCAL_PATH/$file" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$file"
            done
            
            echo -e "${GREEN}✅ Files uploaded successfully!${NC}"
        else
            echo -e "${RED}❌ Please update the credentials and run again.${NC}"
            exit 1
        fi
        ;;
        
    2)
        # FTP Deployment
        echo -e "${YELLOW}📁 FTP Deployment${NC}"
        echo "Creating FTP upload script..."
        
        cat > "$LOCAL_PATH/ftp-upload.txt" << EOF
# FTP Upload Commands
# Use with your FTP client

# Connect to your FTP server
# Upload these files to your web root:

put js/navigation-fix.js js/navigation-fix.js
put js/button-fix.js js/button-fix.js
put js/service-validator.js js/service-validator.js
put js/error-handler.js js/error-handler.js
put js/navigation-manager.js js/navigation-manager.js
put index.html index.html

# Make sure to backup existing files first!
EOF
        
        echo -e "${GREEN}✅ FTP commands saved to: ftp-upload.txt${NC}"
        echo "Use these commands with your FTP client."
        ;;
        
    3)
        # Generate deployment package
        echo -e "${YELLOW}📦 Creating deployment package...${NC}"
        
        PACKAGE_NAME="psscript-fixes-$(date +%Y%m%d-%H%M%S).zip"
        
        cd "$LOCAL_PATH"
        zip -r "$PACKAGE_NAME" "${FILES_TO_DEPLOY[@]}"
        
        echo -e "${GREEN}✅ Deployment package created: $PACKAGE_NAME${NC}"
        echo "Upload this zip file to your server and extract it in your web root."
        ;;
        
    *)
        echo -e "${RED}❌ Invalid option selected.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${YELLOW}🧹 Post-Deployment Steps:${NC}"
echo "1. Clear CDN cache (if using Cloudflare):"
echo "   - Go to Cloudflare Dashboard"
echo "   - Select your domain"
echo "   - Go to Caching → Configuration"
echo "   - Click 'Purge Everything' or purge specific URLs"
echo ""
echo "2. Clear browser cache:"
echo "   - Chrome/Edge: Ctrl+Shift+Delete (Cmd+Shift+Delete on Mac)"
echo "   - Select 'Cached images and files'"
echo "   - Clear for 'Last hour' or 'All time'"
echo ""
echo "3. Test the website:"
echo "   - Visit https://psscript.morloksmaze.com"
echo "   - Test all navigation buttons"
echo "   - Try keyboard shortcuts (1-6, Ctrl+K)"
echo "   - Check for 404 errors"
echo "   - Test external service links"
echo ""

# Create test checklist
cat > "$LOCAL_PATH/deployment-test-checklist.md" << EOF
# PSScript Deployment Test Checklist

## Pre-Deployment
- [ ] Backup existing files
- [ ] Verify all fix files are present
- [ ] Check server credentials

## Deployment
- [ ] Upload all JavaScript files to /js/ directory
- [ ] Upload updated index.html
- [ ] Verify file permissions (644 for files)

## Post-Deployment
- [ ] Clear CDN cache (Cloudflare)
- [ ] Clear browser cache
- [ ] Test in incognito/private mode

## Functionality Tests
- [ ] Navigation buttons work (no disabled state)
- [ ] All menu items are clickable
- [ ] Keyboard shortcuts work (1-6 for navigation)
- [ ] Ctrl+K opens quick navigation
- [ ] Invalid URLs redirect to dashboard
- [ ] 404 page shows with suggestions
- [ ] External service links open correctly
- [ ] Error notifications appear for failures
- [ ] Service status panel displays

## Browser Testing
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari
- [ ] Mobile browsers

## Performance
- [ ] Page loads without JavaScript errors
- [ ] Console shows fix modules loaded
- [ ] No broken resources in Network tab

## Rollback Plan
- [ ] Backup files are accessible
- [ ] Know how to restore previous version
- [ ] Have server access ready

## Monitoring
- [ ] Check error logs after deployment
- [ ] Monitor for user complaints
- [ ] Verify analytics still working

EOF

echo -e "${GREEN}✅ Test checklist created: deployment-test-checklist.md${NC}"
echo ""
echo -e "${YELLOW}📞 Need help?${NC}"
echo "- Check browser console for errors"
echo "- Review WEBSITE_FIXES_SUMMARY.md for details"
echo "- Test in incognito mode to avoid cache issues"