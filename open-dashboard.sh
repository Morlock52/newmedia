#!/bin/bash

# Smart Dashboard Launcher - 2025 Optimized
# One-click access to your media server dashboard

echo "🎬 Opening Media Server Dashboard..."

# Try to open the optimized dashboard first
if [ -f "service-access-optimized.html" ]; then
    echo "✅ Opening optimized dashboard..."
    open service-access-optimized.html
elif [ -f "service-access.html" ]; then
    echo "✅ Opening standard dashboard..."
    open service-access.html
else
    echo "❌ Dashboard files not found. Opening Homepage service..."
    open http://localhost:3001
fi

echo "🌐 Dashboard should open in your default browser"
echo "📱 If nothing opens, go to: http://localhost:3001"