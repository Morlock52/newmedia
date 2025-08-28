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
