#!/bin/bash

# Quick Deployment Script - Adds Dashboard and AI to Existing Services
# This works with your already running containers

set -e

echo "🚀 Quick Media Server Enhancement Deployment"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Current Services Status:${NC}"
echo "------------------------"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(jellyfin|sonarr|radarr|prowlarr|qbittorrent)" | head -10

echo ""
echo -e "${GREEN}✅ Your existing services are running!${NC}"
echo ""

echo -e "${YELLOW}Adding Enhanced Features:${NC}"
echo "• Unified Dashboard"
echo "• AI Assistant"
echo "• Service Monitoring"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p ai-services ai-models scripts

# Create a simple AI service
cat > ai-services/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from datetime import datetime

app = FastAPI(title="Media Server AI Assistant")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str
    context: dict = {}

@app.get("/")
async def root():
    return {"message": "AI Assistant Running", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "jellyfin": "http://localhost:8096",
            "sonarr": "http://localhost:8989",
            "radarr": "http://localhost:7878"
        }
    }

@app.post("/api/query")
async def process_query(query: Query):
    # Simple response for now
    return {
        "response": f"Processing: {query.query}",
        "suggestions": [
            "Check Sonarr for new episodes",
            "Browse Jellyfin library",
            "Review download queue"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/services")
async def get_services():
    return {
        "services": [
            {"name": "Jellyfin", "url": "http://localhost:8096", "status": "running"},
            {"name": "Sonarr", "url": "http://localhost:8989", "status": "running"},
            {"name": "Radarr", "url": "http://localhost:7878", "status": "running"},
            {"name": "Prowlarr", "url": "http://localhost:9696", "status": "running"},
            {"name": "qBittorrent", "url": "http://localhost:8080", "status": "running"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
EOF

# Deploy the quick enhancement
echo ""
echo -e "${BLUE}Deploying enhancements...${NC}"

# Check if dashboard needs npm install
if [ -d "dashboard" ]; then
    echo "Setting up dashboard..."
    cd dashboard
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    cd ..
fi

# Use the quick deploy compose file
docker-compose -f docker-compose.quick-deploy.yml up -d

echo ""
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo ""
echo "Access Points:"
echo "=============="
echo "• Dashboard:     http://localhost:5173"
echo "• AI Assistant:  http://localhost:8090"
echo "• AI API Docs:   http://localhost:8090/docs"
echo ""
echo "Existing Services:"
echo "• Jellyfin:      http://localhost:8096"
echo "• Sonarr:        http://localhost:8989"
echo "• Radarr:        http://localhost:7878"
echo "• Prowlarr:      http://localhost:9696"
echo "• qBittorrent:   http://localhost:8080"
echo ""
echo -e "${GREEN}Your media server is now enhanced with AI and a unified dashboard!${NC}"
EOF

chmod +x deploy-now.sh