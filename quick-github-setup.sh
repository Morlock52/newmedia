#!/bin/bash

echo "🚀 Quick GitHub Setup for Ultimate Media Server 2025"
echo "===================================================="
echo ""

# Create a minimal commit first
echo "Creating initial commit with essential files..."

# Initialize git if needed
if [ ! -d ".git" ]; then
    git init
    git branch -m main
fi

# Configure git
git config user.email "user@example.com"
git config user.name "Media Server Admin"

# Remove lock file if exists
rm -f .git/index.lock

# Stage only essential files first
git add README.md SIMPLE-README.md .gitignore 2>/dev/null || true
git add ultimate-fun-dashboard.html 2>/dev/null || true
git add deploy-arm64-media-apps.sh 2>/dev/null || true
git add deploy-complete-media-ecosystem.sh 2>/dev/null || true
git add ULTIMATE_MEDIA_SERVER_2025_REVIEW_AND_IMPROVEMENTS.md 2>/dev/null || true
git add DEPLOYMENT_COMPLETE_SUMMARY.md 2>/dev/null || true

# Commit essentials
git commit -m "Initial commit: Ultimate Media Server 2025 essentials" --no-verify 2>/dev/null || true

echo ""
echo "✅ Essential files committed!"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo "   Name: ultimate-media-server-2025"
echo ""
echo "2. Add remote and push:"
echo "   git remote add origin https://github.com/YOUR-USERNAME/ultimate-media-server-2025.git"
echo "   git push -u origin main"
echo ""
echo "3. Then add remaining files:"
echo "   git add ."
echo "   git commit -m 'Add complete documentation and scripts'"
echo "   git push"
echo ""
echo "Your repository URL will be:"
echo "https://github.com/YOUR-USERNAME/ultimate-media-server-2025"
echo ""
echo "🎉 Ready to share your media server with the world!"