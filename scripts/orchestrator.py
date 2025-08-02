#!/usr/bin/env python3
"""
Media Server Orchestrator
Central management system for all automation components
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import signal
from typing import Dict, List, Optional
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiofiles
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import schedule
import threading

# Import our automation modules
sys.path.append(str(Path(__file__).parent))
from media_processing import MediaProcessor
from content_discovery import ContentDiscovery, ContentRequest, ContentType, Quality
from organization_workflows import MediaOrganizer
from user_experience import UserExperienceManager
from maintenance_automation import MaintenanceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
request_counter = Counter('media_requests_total', 'Total media requests', ['type', 'status'])
processing_time = Histogram('media_processing_seconds', 'Media processing time')
active_tasks = Gauge('active_tasks', 'Number of active tasks')
system_health = Gauge('system_health_score', 'Overall system health score')

# API models
class MediaRequest(BaseModel):
    title: str
    media_type: str
    quality: Optional[str] = "1080p"
    priority: Optional[int] = 5
    user_id: Optional[str] = "anonymous"

class OrganizeRequest(BaseModel):
    path: str
    recursive: Optional[bool] = True
    dry_run: Optional[bool] = False

class BackupRequest(BaseModel):
    backup_type: Optional[str] = "manual"
    compress: Optional[bool] = True

class Orchestrator:
    def __init__(self, config_path='/config/orchestrator.json'):
        self.config = self.load_config(config_path)
        self.app = FastAPI(title="Media Server Orchestrator")
        self.setup_middleware()
        self.setup_routes()
        self.init_components()
        self.background_tasks = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 10))
        
    def load_config(self, config_path):
        """Load orchestrator configuration"""
        default_config = {
            'api_port': 8003,
            'max_workers': 10,
            'enable_components': {
                'media_processing': True,
                'content_discovery': True,
                'organization': True,
                'user_experience': True,
                'maintenance': True
            },
            'automation_rules': {
                'auto_organize': True,
                'auto_process': True,
                'auto_discover': True
            },
            'monitoring': {
                'prometheus_enabled': True,
                'health_check_interval': 60
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def init_components(self):
        """Initialize automation components"""
        self.components = {}
        
        if self.config['enable_components']['media_processing']:
            self.components['media_processor'] = MediaProcessor()
            
        if self.config['enable_components']['content_discovery']:
            self.components['content_discovery'] = ContentDiscovery()
            
        if self.config['enable_components']['organization']:
            self.components['media_organizer'] = MediaOrganizer()
            
        if self.config['enable_components']['user_experience']:
            self.components['user_experience'] = UserExperienceManager()
            
        if self.config['enable_components']['maintenance']:
            self.components['maintenance'] = MaintenanceManager()
        
        logger.info(f"Initialized {len(self.components)} components")
    
    def setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = datetime.now()
            response = await call_next(request)
            process_time = (datetime.now() - start_time).total_seconds()
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return HTMLResponse(content=self.generate_dashboard())
        
        @self.app.get("/health")
        async def health_check():
            health_status = await self.check_health()
            return JSONResponse(content=health_status)
        
        @self.app.get("/metrics")
        async def metrics():
            return generate_latest()
        
        # Media requests
        @self.app.post("/api/request")
        async def create_media_request(request: MediaRequest, background_tasks: BackgroundTasks):
            try:
                # Track metric
                request_counter.labels(type=request.media_type, status='received').inc()
                
                # Create content request
                content_request = ContentRequest(
                    title=request.title,
                    content_type=ContentType(request.media_type),
                    quality=Quality(request.quality),
                    priority=request.priority
                )
                
                # Process in background
                background_tasks.add_task(self.process_media_request, content_request, request.user_id)
                
                return {"status": "accepted", "message": f"Request for {request.title} accepted"}
                
            except Exception as e:
                request_counter.labels(type=request.media_type, status='failed').inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        # Organization
        @self.app.post("/api/organize")
        async def organize_media(request: OrganizeRequest, background_tasks: BackgroundTasks):
            try:
                background_tasks.add_task(
                    self.organize_media,
                    Path(request.path),
                    request.recursive,
                    request.dry_run
                )
                return {"status": "started", "message": "Organization task started"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Processing
        @self.app.post("/api/process/{file_path:path}")
        async def process_file(file_path: str, background_tasks: BackgroundTasks):
            try:
                background_tasks.add_task(self.process_media_file, file_path)
                return {"status": "started", "message": f"Processing {file_path}"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Maintenance
        @self.app.post("/api/backup")
        async def create_backup(request: BackupRequest, background_tasks: BackgroundTasks):
            try:
                background_tasks.add_task(
                    self.create_backup,
                    request.backup_type
                )
                return {"status": "started", "message": "Backup task started"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/stats")
        async def get_statistics():
            return await self.get_system_statistics()
        
        @self.app.get("/api/tasks")
        async def get_active_tasks():
            return {"active_tasks": active_tasks._value.get(), "tasks": self.background_tasks}
        
        @self.app.post("/api/scan")
        async def scan_media(background_tasks: BackgroundTasks):
            background_tasks.add_task(self.full_media_scan)
            return {"status": "started", "message": "Full media scan started"}
    
    async def process_media_request(self, request: ContentRequest, user_id: str):
        """Process a media request through the full pipeline"""
        try:
            active_tasks.inc()
            
            # Step 1: User experience - create request
            if 'user_experience' in self.components:
                await self.components['user_experience'].create_request(user_id, {
                    'title': request.title,
                    'media_type': request.content_type.value,
                    'priority': request.priority
                })
            
            # Step 2: Content discovery - search and add
            if 'content_discovery' in self.components:
                success = self.components['content_discovery'].add_content_request(request)
                if success:
                    request_counter.labels(type=request.content_type.value, status='added').inc()
                else:
                    request_counter.labels(type=request.content_type.value, status='failed').inc()
            
            logger.info(f"Processed request for {request.title}")
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            request_counter.labels(type=request.content_type.value, status='error').inc()
        finally:
            active_tasks.dec()
    
    async def organize_media(self, path: Path, recursive: bool, dry_run: bool):
        """Organize media files"""
        try:
            active_tasks.inc()
            
            if 'media_organizer' in self.components:
                self.components['media_organizer'].organize_directory(path, recursive, dry_run)
                logger.info(f"Organized media in {path}")
            
        except Exception as e:
            logger.error(f"Error organizing media: {e}")
        finally:
            active_tasks.dec()
    
    async def process_media_file(self, file_path: str):
        """Process a single media file"""
        try:
            active_tasks.inc()
            
            with processing_time.time():
                if 'media_processor' in self.components:
                    success = self.components['media_processor'].process_file(file_path)
                    if success:
                        logger.info(f"Processed file: {file_path}")
                        
                        # Auto-organize if enabled
                        if self.config['automation_rules']['auto_organize']:
                            if 'media_organizer' in self.components:
                                self.components['media_organizer'].organize_file(Path(file_path))
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
        finally:
            active_tasks.dec()
    
    async def create_backup(self, backup_type: str):
        """Create system backup"""
        try:
            active_tasks.inc()
            
            if 'maintenance' in self.components:
                success = self.components['maintenance'].perform_backup(backup_type)
                if success:
                    logger.info(f"Backup completed: {backup_type}")
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
        finally:
            active_tasks.dec()
    
    async def full_media_scan(self):
        """Perform full media library scan"""
        try:
            active_tasks.inc()
            logger.info("Starting full media scan")
            
            media_paths = [
                '/media/movies',
                '/media/tv',
                '/media/music'
            ]
            
            for media_path in media_paths:
                if not os.path.exists(media_path):
                    continue
                
                # Process new files
                if self.config['automation_rules']['auto_process']:
                    if 'media_processor' in self.components:
                        self.components['media_processor'].process_directory(media_path)
                
                # Organize files
                if self.config['automation_rules']['auto_organize']:
                    if 'media_organizer' in self.components:
                        self.components['media_organizer'].organize_directory(Path(media_path))
            
            logger.info("Full media scan completed")
            
        except Exception as e:
            logger.error(f"Error in media scan: {e}")
        finally:
            active_tasks.dec()
    
    async def check_health(self) -> Dict:
        """Check overall system health"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'metrics': {
                'active_tasks': active_tasks._value.get(),
                'total_requests': sum(request_counter._metrics.values())
            }
        }
        
        # Check each component
        component_scores = []
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'monitor_health'):
                    component_health = component.monitor_health()
                    health_status['components'][name] = 'healthy'
                    component_scores.append(1.0)
                else:
                    health_status['components'][name] = 'unknown'
                    component_scores.append(0.5)
            except Exception as e:
                health_status['components'][name] = 'unhealthy'
                component_scores.append(0.0)
                logger.error(f"Error checking {name} health: {e}")
        
        # Calculate overall health score
        if component_scores:
            health_score = sum(component_scores) / len(component_scores)
            system_health.set(health_score)
            
            if health_score < 0.5:
                health_status['status'] = 'unhealthy'
            elif health_score < 0.8:
                health_status['status'] = 'degraded'
        
        return health_status
    
    async def get_system_statistics(self) -> Dict:
        """Get system statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'system': {
                'active_tasks': active_tasks._value.get(),
                'total_requests': {}
            }
        }
        
        # Get request statistics
        for metric, value in request_counter._metrics.items():
            media_type = metric[0][1]  # Extract media type from labels
            status = metric[0][3]  # Extract status from labels
            
            if media_type not in stats['system']['total_requests']:
                stats['system']['total_requests'][media_type] = {}
            
            stats['system']['total_requests'][media_type][status] = value
        
        # Get component-specific stats
        if 'content_discovery' in self.components:
            # This would call component-specific stats methods
            pass
        
        return stats
    
    def generate_dashboard(self) -> str:
        """Generate HTML dashboard"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Media Server Orchestrator</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .status { display: inline-block; padding: 5px 10px; border-radius: 4px; }
                .healthy { background-color: #4CAF50; color: white; }
                .degraded { background-color: #FF9800; color: white; }
                .unhealthy { background-color: #F44336; color: white; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-value { font-size: 24px; font-weight: bold; }
                button { background-color: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px; }
                button:hover { background-color: #1976D2; }
            </style>
            <script>
                async function makeRequest(endpoint, method = 'GET', body = null) {
                    const options = { method };
                    if (body) {
                        options.headers = { 'Content-Type': 'application/json' };
                        options.body = JSON.stringify(body);
                    }
                    const response = await fetch(endpoint, options);
                    return response.json();
                }
                
                async function refreshStatus() {
                    const health = await makeRequest('/health');
                    document.getElementById('health-status').innerHTML = `
                        <span class="status ${health.status}">${health.status.toUpperCase()}</span>
                    `;
                    
                    const stats = await makeRequest('/api/stats');
                    document.getElementById('active-tasks').textContent = stats.system.active_tasks;
                }
                
                async function startScan() {
                    const result = await makeRequest('/api/scan', 'POST');
                    alert(result.message);
                }
                
                async function createBackup() {
                    const result = await makeRequest('/api/backup', 'POST', { backup_type: 'manual' });
                    alert(result.message);
                }
                
                setInterval(refreshStatus, 5000);
                window.onload = refreshStatus;
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Media Server Orchestrator</h1>
                
                <div class="card">
                    <h2>System Status</h2>
                    <div id="health-status">Loading...</div>
                    <div class="metric">
                        <div>Active Tasks</div>
                        <div class="metric-value" id="active-tasks">-</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Quick Actions</h2>
                    <button onclick="startScan()">Start Media Scan</button>
                    <button onclick="createBackup()">Create Backup</button>
                    <button onclick="window.location.href='/metrics'">View Metrics</button>
                </div>
                
                <div class="card">
                    <h2>API Endpoints</h2>
                    <ul>
                        <li>POST /api/request - Create media request</li>
                        <li>POST /api/organize - Organize media files</li>
                        <li>POST /api/process/{path} - Process media file</li>
                        <li>POST /api/backup - Create backup</li>
                        <li>GET /api/stats - Get statistics</li>
                        <li>GET /health - Health check</li>
                        <li>GET /metrics - Prometheus metrics</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def start_background_scheduler(self):
        """Start background task scheduler"""
        def run_scheduler():
            # Schedule periodic tasks
            if self.config['automation_rules']['auto_discover']:
                schedule.every(30).minutes.do(
                    lambda: asyncio.create_task(self.check_new_releases())
                )
            
            if self.config['monitoring']['prometheus_enabled']:
                schedule.every(self.config['monitoring']['health_check_interval']).seconds.do(
                    lambda: asyncio.create_task(self.check_health())
                )
            
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    async def check_new_releases(self):
        """Check for new content releases"""
        if 'content_discovery' in self.components:
            self.components['content_discovery'].monitor_releases()
    
    def run(self):
        """Run the orchestrator"""
        logger.info("Starting Media Server Orchestrator")
        
        # Start background scheduler
        self.start_background_scheduler()
        
        # Start maintenance daemon if enabled
        if 'maintenance' in self.components and self.config['enable_components']['maintenance']:
            maintenance_thread = threading.Thread(
                target=self.components['maintenance'].run,
                daemon=True
            )
            maintenance_thread.start()
        
        # Run FastAPI server
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.config['api_port'],
            log_level="info"
        )

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Shutting down orchestrator...")
    sys.exit(0)

def main():
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run orchestrator
    orchestrator = Orchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()