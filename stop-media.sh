#!/bin/bash

# Stop and clean up media server stack

echo "🛑 Stopping Media Server Stack"
echo "==============================="

# Stop all services
echo "📊 Stopping services..."
docker-compose down

# Optional: Remove volumes (uncomment to reset all data)
# echo "🗑️  Removing volumes..."
# docker-compose down -v

# Optional: Remove images (uncomment to free space)
# echo "🖼️  Removing images..."
# docker-compose down --rmi all

echo "✅ Media server stopped"
echo ""
echo "💡 To restart: ./quick-deploy.sh"
echo "💡 To remove all data: docker-compose down -v"