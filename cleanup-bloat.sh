#!/bin/bash

# Ultimate Media Server - Bloat Cleanup Script
# This removes unnecessary files for single container deployment

echo "🧹 Ultimate Media Server - Cleanup Script"
echo "========================================"
echo "This will remove unnecessary files for single container deployment"
echo ""

# Confirmation
read -p "Are you sure you want to clean up bloat? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

echo ""
echo "📊 Current disk usage:"
du -sh . 2>/dev/null || echo "Unable to calculate"

# Create backup directory for important files
echo ""
echo "📦 Backing up essential files..."
mkdir -p .backup-essential
cp -p Dockerfile.multi-service .backup-essential/ 2>/dev/null
cp -p modern-landing.html dashboard-enhanced.html .backup-essential/ 2>/dev/null
cp -p mobile-ui.css social-share.js .backup-essential/ 2>/dev/null
cp -p .env .env.example .backup-essential/ 2>/dev/null
cp -p install-media-server.sh .backup-essential/ 2>/dev/null

# Remove bloat
echo ""
echo "🗑️ Removing bloat..."

# Remove excess docker-compose files (keep only essential ones)
echo "- Removing 20+ docker-compose files..."
mkdir -p .backup-compose
mv docker-compose*.yml .backup-compose/ 2>/dev/null
# Keep only the main one if needed
cp .backup-compose/docker-compose.yml . 2>/dev/null

# Remove duplicate/old HTML dashboards
echo "- Removing duplicate HTML dashboards..."
mkdir -p .backup-html
mv *dashboard*.html .backup-html/ 2>/dev/null
mv *DASHBOARD*.html .backup-html/ 2>/dev/null
mv service-status.html env-settings*.html .backup-html/ 2>/dev/null
# Restore the modern ones
cp .backup-essential/dashboard-enhanced.html . 2>/dev/null

# Remove test files and reports
echo "- Removing test files and reports..."
rm -rf TEST_REPORTS/ TEST_RESULTS/ 2>/dev/null
rm -f *test*.js *test*.sh 2>/dev/null

# Remove duplicate guides and docs
echo "- Cleaning up duplicate documentation..."
mkdir -p .backup-docs
mv DEPLOYMENT_GUIDE*.md .backup-docs/ 2>/dev/null
mv *ARCHITECTURE*.md .backup-docs/ 2>/dev/null
mv *GUIDE*.md .backup-docs/ 2>/dev/null
# Keep only essential docs
cp .backup-docs/RUN_SINGLE_CONTAINER.md . 2>/dev/null

# Remove temporary and log files
echo "- Removing temporary files..."
rm -f *.log *.tmp .*.swp 2>/dev/null
rm -rf .roo .roomodes 2>/dev/null

# Remove duplicate Dockerfiles
echo "- Cleaning duplicate Dockerfiles..."
mkdir -p .backup-dockerfiles
mv Dockerfile.* .backup-dockerfiles/ 2>/dev/null
# Keep only the multi-service one
cp .backup-dockerfiles/Dockerfile.multi-service . 2>/dev/null

# Remove unused directories
echo "- Removing unused directories..."
rm -rf memory scripts/old scripts/test 2>/dev/null
rm -rf holographic-* quantum-* web3-* blockchain-* 2>/dev/null
rm -rf ai-media-features ai-ml-nexus ar-vr-media 2>/dev/null

# Clean up node modules if present
echo "- Cleaning node artifacts..."
rm -rf node_modules package-lock.json 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 New disk usage:"
du -sh . 2>/dev/null || echo "Unable to calculate"

echo ""
echo "📁 Essential files kept:"
echo "- Dockerfile.multi-service (single container)"
echo "- modern-landing.html (modern UI)"
echo "- dashboard-enhanced.html (dashboard)"
echo "- mobile-ui.css, social-share.js (UI assets)"
echo "- RUN_SINGLE_CONTAINER.md (instructions)"
echo "- install-media-server.sh (installer)"

echo ""
echo "💾 Backups created in:"
echo "- .backup-essential/ (core files)"
echo "- .backup-compose/ (docker-compose files)"
echo "- .backup-html/ (old dashboards)"
echo "- .backup-docs/ (old documentation)"

echo ""
echo "🚀 To run the single container:"
echo "docker build -t mediaserver-aio -f Dockerfile.multi-service ."
echo "docker run -d --name mediaserver -p 80:80 -p 8096:8096 [...] mediaserver-aio"

echo ""
echo "🗑️ To permanently delete backups:"
echo "rm -rf .backup-*"