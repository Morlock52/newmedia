"""
Archon Server - Knowledge and Task Management Backend
Version: 1.0.0
Date: August 2025
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import socketio
import asyncio
import asyncpg
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import os
import json
from loguru import logger
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logger.add("logs/archon-server.log", rotation="10 MB", retention="30 days")

# Pydantic models
class Project(BaseModel):
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    github_repo: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "active"
    metadata: Dict[str, Any] = {}

class Task(BaseModel):
    id: Optional[str] = None
    project_id: str
    title: str
    description: Optional[str] = None
    status: str = "todo"  # todo, doing, review, done
    priority: int = 5
    feature: Optional[str] = None
    assigned_to: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

class Document(BaseModel):
    id: Optional[str] = None
    project_id: Optional[str] = None
    title: str
    content: str
    doc_type: str = "text"  # text, pdf, url, code
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}
    created_at: Optional[datetime] = None

class RAGQuery(BaseModel):
    query: str
    project_id: Optional[str] = None
    match_count: int = 5
    include_metadata: bool = True

# Database connection pool
class Database:
    def __init__(self):
        self.pool = None
        self.redis_client = None
        
    async def connect(self):
        """Initialize database connections"""
        db_url = os.getenv("DATABASE_URL", "postgresql://archon:archon_secure_pass_2025@localhost:5432/archon")
        self.pool = await asyncpg.create_pool(db_url, min_size=5, max_size=20)
        
        # Initialize Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = await redis.from_url(redis_url)
        
        # Create tables if they don't exist
        await self.init_schema()
        
    async def init_schema(self):
        """Initialize database schema"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Projects table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    github_repo VARCHAR(500),
                    status VARCHAR(50) DEFAULT 'active',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tasks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    description TEXT,
                    status VARCHAR(50) DEFAULT 'todo',
                    priority INTEGER DEFAULT 5,
                    feature VARCHAR(255),
                    assigned_to VARCHAR(255),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Documents table with vector embeddings
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
                    title VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    doc_type VARCHAR(50) DEFAULT 'text',
                    embedding vector(768),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id)")
            
    async def disconnect(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
        if self.redis_client:
            await self.redis_client.close()

# Initialize database
db = Database()

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    await db.connect()
    logger.info("Archon Server started successfully")
    yield
    await db.disconnect()
    logger.info("Archon Server shutdown")

# Initialize FastAPI app
app = FastAPI(
    title="Archon Server",
    description="Knowledge and Task Management Backend for AI Coding Assistants",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*'
)
socket_app = socketio.ASGIApp(sio, app)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "archon-server",
        "timestamp": datetime.utcnow().isoformat()
    }

# Projects endpoints
@app.post("/api/projects", response_model=Project)
async def create_project(project: Project):
    """Create a new project"""
    async with db.pool.acquire() as conn:
        result = await conn.fetchrow("""
            INSERT INTO projects (title, description, github_repo, status, metadata)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id, title, description, github_repo, status, metadata, created_at, updated_at
        """, project.title, project.description, project.github_repo, project.status, json.dumps(project.metadata))
        
        return Project(**dict(result))

@app.get("/api/projects", response_model=List[Project])
async def list_projects(status: Optional[str] = None):
    """List all projects"""
    async with db.pool.acquire() as conn:
        if status:
            results = await conn.fetch("""
                SELECT * FROM projects WHERE status = $1 ORDER BY created_at DESC
            """, status)
        else:
            results = await conn.fetch("""
                SELECT * FROM projects ORDER BY created_at DESC
            """)
        
        return [Project(**dict(r)) for r in results]

@app.get("/api/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get project by ID"""
    async with db.pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT * FROM projects WHERE id = $1
        """, project_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return Project(**dict(result))

# Tasks endpoints
@app.post("/api/tasks", response_model=Task)
async def create_task(task: Task):
    """Create a new task"""
    async with db.pool.acquire() as conn:
        result = await conn.fetchrow("""
            INSERT INTO tasks (project_id, title, description, status, priority, feature, assigned_to, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, project_id, title, description, status, priority, feature, assigned_to, metadata, created_at, updated_at
        """, task.project_id, task.title, task.description, task.status, task.priority, 
            task.feature, task.assigned_to, json.dumps(task.metadata))
        
        # Emit WebSocket event
        await sio.emit('task_created', dict(result))
        
        return Task(**dict(result))

@app.get("/api/tasks", response_model=List[Task])
async def list_tasks(
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    feature: Optional[str] = None
):
    """List tasks with optional filters"""
    async with db.pool.acquire() as conn:
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        
        if project_id:
            params.append(project_id)
            query += f" AND project_id = ${len(params)}"
        
        if status:
            params.append(status)
            query += f" AND status = ${len(params)}"
        
        if feature:
            params.append(feature)
            query += f" AND feature = ${len(params)}"
        
        query += " ORDER BY priority DESC, created_at DESC"
        
        results = await conn.fetch(query, *params)
        return [Task(**dict(r)) for r in results]

@app.put("/api/tasks/{task_id}", response_model=Task)
async def update_task(task_id: str, updates: Dict[str, Any]):
    """Update task status or other fields"""
    async with db.pool.acquire() as conn:
        # Build dynamic update query
        set_clauses = []
        params = [task_id]
        
        for key, value in updates.items():
            if key not in ['id', 'created_at']:
                params.append(value)
                set_clauses.append(f"{key} = ${len(params)}")
        
        if not set_clauses:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        query = f"""
            UPDATE tasks 
            SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            RETURNING *
        """
        
        result = await conn.fetchrow(query, *params)
        
        if not result:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Emit WebSocket event
        await sio.emit('task_updated', dict(result))
        
        return Task(**dict(result))

# Document/Knowledge endpoints
@app.post("/api/documents")
async def add_document(doc: Document):
    """Add document to knowledge base"""
    # Generate embedding
    embedding = embedder.encode(doc.content).tolist()
    
    async with db.pool.acquire() as conn:
        result = await conn.fetchrow("""
            INSERT INTO documents (project_id, title, content, doc_type, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id, project_id, title, doc_type, metadata, created_at
        """, doc.project_id, doc.title, doc.content, doc.doc_type, 
            embedding, json.dumps(doc.metadata))
        
        return {"id": str(result['id']), "status": "indexed", "title": doc.title}

@app.post("/api/rag/query")
async def perform_rag_query(query: RAGQuery):
    """Perform RAG query on knowledge base"""
    # Generate query embedding
    query_embedding = embedder.encode(query.query).tolist()
    
    async with db.pool.acquire() as conn:
        # Perform vector similarity search
        base_query = """
            SELECT id, title, content, doc_type, metadata,
                   1 - (embedding <=> $1::vector) as similarity
            FROM documents
        """
        
        if query.project_id:
            base_query += " WHERE project_id = $2"
            base_query += f" ORDER BY similarity DESC LIMIT {query.match_count}"
            results = await conn.fetch(base_query, query_embedding, query.project_id)
        else:
            base_query += f" ORDER BY embedding <=> $1::vector LIMIT {query.match_count}"
            results = await conn.fetch(base_query, query_embedding)
        
        return {
            "query": query.query,
            "matches": [
                {
                    "id": str(r['id']),
                    "title": r['title'],
                    "content": r['content'][:500],  # Truncate for response
                    "similarity": float(r['similarity']),
                    "metadata": r['metadata'] if query.include_metadata else {}
                }
                for r in results
            ]
        }

# WebSocket endpoints
@sio.event
async def connect(sid, environ):
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {sid}")
    await sio.emit('connected', {'sid': sid}, to=sid)

@sio.event
async def disconnect(sid):
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def subscribe_project(sid, data):
    """Subscribe to project updates"""
    project_id = data.get('project_id')
    await sio.enter_room(sid, f"project_{project_id}")
    await sio.emit('subscribed', {'project_id': project_id}, to=sid)

# Agent collaboration endpoints
@app.post("/api/agents/register")
async def register_agent(agent_info: Dict[str, Any]):
    """Register an AI agent with Archon"""
    agent_id = agent_info.get('id')
    capabilities = agent_info.get('capabilities', [])
    
    # Store agent info in Redis
    await db.redis_client.setex(
        f"agent:{agent_id}",
        3600,  # 1 hour TTL
        json.dumps(agent_info)
    )
    
    return {"status": "registered", "agent_id": agent_id}

@app.get("/api/agents/active")
async def list_active_agents():
    """List all active agents"""
    keys = await db.redis_client.keys("agent:*")
    agents = []
    
    for key in keys:
        agent_data = await db.redis_client.get(key)
        if agent_data:
            agents.append(json.loads(agent_data))
    
    return {"agents": agents, "count": len(agents)}

# Export the socket app for uvicorn
app = socket_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8181, reload=True)