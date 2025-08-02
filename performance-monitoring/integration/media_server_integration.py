#!/usr/bin/env python3
"""
Media Server Integration Module
==============================

Integrates the performance monitoring system with existing media server
infrastructure, providing specialized monitoring for Jellyfin, Sonarr,
Radarr, and other media applications.
"""

import asyncio
import json
import logging
import time
import aiohttp
import subprocess
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import docker
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MediaServerMetrics:
    """Media server specific metrics"""
    timestamp: float
    service_name: str
    status: str
    response_time: float
    active_streams: int
    transcoding_sessions: int
    disk_space_gb: float
    bandwidth_mbps: float
    cpu_usage: float
    memory_usage_mb: int
    error_count: int
    queue_size: int
    health_score: float

@dataclass
class TranscodingMetrics:
    """Transcoding performance metrics"""
    session_id: str
    media_file: str
    source_codec: str
    target_codec: str
    resolution_source: str
    resolution_target: str
    fps: float
    bitrate_kbps: int
    progress_percent: float
    estimated_time_remaining: int
    cpu_usage: float
    gpu_usage: float
    temperature_celsius: int

@dataclass
class ContentMetrics:
    """Content library metrics"""
    library_name: str
    total_items: int
    recently_added: int
    storage_size_gb: float
    indexing_status: str
    scan_progress: float
    missing_metadata: int
    corrupt_files: int
    duplicate_files: int

class JellyfinMonitor:
    """Jellyfin media server monitoring"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_system_info(self) -> Dict[str, Any]:
        """Get Jellyfin system information"""
        try:
            url = f"{self.base_url}/System/Info"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get system info: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
            
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get active streaming sessions"""
        try:
            url = f"{self.base_url}/Sessions"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get sessions: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting sessions: {e}")
            return []
            
    async def get_transcoding_jobs(self) -> List[TranscodingMetrics]:
        """Get active transcoding jobs"""
        sessions = await self.get_active_sessions()
        transcoding_jobs = []
        
        for session in sessions:
            play_state = session.get('PlayState', {})
            transcode_info = session.get('TranscodingInfo')
            
            if transcode_info:
                job = TranscodingMetrics(
                    session_id=session.get('Id', ''),
                    media_file=session.get('NowPlayingItem', {}).get('Name', ''),
                    source_codec=transcode_info.get('VideoCodec', ''),
                    target_codec=transcode_info.get('TranscodingVideoCodec', ''),
                    resolution_source=f"{transcode_info.get('Width', 0)}x{transcode_info.get('Height', 0)}",
                    resolution_target=f"{transcode_info.get('TranscodingWidth', 0)}x{transcode_info.get('TranscodingHeight', 0)}",
                    fps=transcode_info.get('Framerate', 0.0),
                    bitrate_kbps=transcode_info.get('Bitrate', 0) // 1000,
                    progress_percent=play_state.get('PositionTicks', 0) / play_state.get('RuntimeTicks', 1) * 100,
                    estimated_time_remaining=0,  # Would need calculation
                    cpu_usage=0.0,  # Would need system monitoring
                    gpu_usage=0.0,  # Would need GPU monitoring
                    temperature_celsius=0  # Would need hardware monitoring
                )
                transcoding_jobs.append(job)
                
        return transcoding_jobs
        
    async def get_library_stats(self) -> List[ContentMetrics]:
        """Get library statistics"""
        try:
            url = f"{self.base_url}/Library/VirtualFolders"
            headers = self._get_headers()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    libraries = await response.json()
                    stats = []
                    
                    for library in libraries:
                        # Get detailed stats for each library
                        library_stats = await self._get_library_details(library['ItemId'])
                        stats.append(library_stats)
                        
                    return stats
                else:
                    logger.error(f"Failed to get libraries: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting library stats: {e}")
            return []
            
    async def _get_library_details(self, library_id: str) -> ContentMetrics:
        """Get detailed library information"""
        try:
            url = f"{self.base_url}/Items/Counts"
            headers = self._get_headers()
            params = {'ParentId': library_id}
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return ContentMetrics(
                        library_name=library_id,  # Would need to get actual name
                        total_items=data.get('ItemCount', 0),
                        recently_added=data.get('RecentlyAddedItemCount', 0),
                        storage_size_gb=0.0,  # Would need storage calculation
                        indexing_status='complete',
                        scan_progress=100.0,
                        missing_metadata=0,
                        corrupt_files=0,
                        duplicate_files=0
                    )
        except Exception as e:
            logger.error(f"Error getting library details: {e}")
            
        return ContentMetrics(
            library_name=library_id,
            total_items=0,
            recently_added=0,
            storage_size_gb=0.0,
            indexing_status='unknown',
            scan_progress=0.0,
            missing_metadata=0,
            corrupt_files=0,
            duplicate_files=0
        )
        
    async def collect_metrics(self) -> MediaServerMetrics:
        """Collect comprehensive Jellyfin metrics"""
        start_time = time.time()
        
        try:
            # Get basic system info
            system_info = await self.get_system_info()
            sessions = await self.get_active_sessions()
            transcoding_jobs = await self.get_transcoding_jobs()
            
            response_time = (time.time() - start_time) * 1000
            
            # Calculate metrics
            active_streams = len(sessions)
            transcoding_sessions = len(transcoding_jobs)
            
            # Get bandwidth usage (sum of all session bitrates)
            total_bandwidth = sum(
                session.get('TranscodingInfo', {}).get('Bitrate', 0) 
                for session in sessions
            ) / 1_000_000  # Convert to Mbps
            
            return MediaServerMetrics(
                timestamp=time.time(),
                service_name='jellyfin',
                status='up' if system_info else 'down',
                response_time=response_time,
                active_streams=active_streams,
                transcoding_sessions=transcoding_sessions,
                disk_space_gb=0.0,  # Would need disk monitoring
                bandwidth_mbps=total_bandwidth,
                cpu_usage=0.0,  # Would need container metrics
                memory_usage_mb=0,  # Would need container metrics
                error_count=0,
                queue_size=0,
                health_score=100.0 if system_info else 0.0
            )
            
        except Exception as e:
            logger.error(f"Error collecting Jellyfin metrics: {e}")
            return MediaServerMetrics(
                timestamp=time.time(),
                service_name='jellyfin',
                status='error',
                response_time=-1,
                active_streams=0,
                transcoding_sessions=0,
                disk_space_gb=0.0,
                bandwidth_mbps=0.0,
                cpu_usage=0.0,
                memory_usage_mb=0,
                error_count=1,
                queue_size=0,
                health_score=0.0
            )
            
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers"""
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers['X-MediaBrowser-Token'] = self.api_key
            
        return headers

class ArrServiceMonitor:
    """Monitor Sonarr/Radarr/Prowlarr services"""
    
    def __init__(self, service_name: str, base_url: str, api_key: str):
        self.service_name = service_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            api_version = 'v3' if self.service_name in ['sonarr', 'radarr'] else 'v1'
            url = f"{self.base_url}/api/{api_version}/system/status"
            headers = {'X-Api-Key': self.api_key}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get {self.service_name} status: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting {self.service_name} status: {e}")
            return {}
            
    async def get_queue(self) -> List[Dict[str, Any]]:
        """Get download queue"""
        try:
            api_version = 'v3' if self.service_name in ['sonarr', 'radarr'] else 'v1'
            url = f"{self.base_url}/api/{api_version}/queue"
            headers = {'X-Api-Key': self.api_key}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('records', []) if isinstance(data, dict) else data
                else:
                    logger.error(f"Failed to get {self.service_name} queue: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting {self.service_name} queue: {e}")
            return []
            
    async def get_disk_space(self) -> List[Dict[str, Any]]:
        """Get disk space information"""
        try:
            api_version = 'v3' if self.service_name in ['sonarr', 'radarr'] else 'v1'
            url = f"{self.base_url}/api/{api_version}/diskspace"
            headers = {'X-Api-Key': self.api_key}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get {self.service_name} disk space: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting {self.service_name} disk space: {e}")
            return []
            
    async def get_calendar(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get upcoming releases"""
        try:
            api_version = 'v3' if self.service_name in ['sonarr', 'radarr'] else 'v1'
            url = f"{self.base_url}/api/{api_version}/calendar"
            headers = {'X-Api-Key': self.api_key}
            
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days)
            
            params = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get {self.service_name} calendar: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting {self.service_name} calendar: {e}")
            return []
            
    async def collect_metrics(self) -> MediaServerMetrics:
        """Collect comprehensive metrics"""
        start_time = time.time()
        
        try:
            status = await self.get_system_status()
            queue = await self.get_queue()
            disk_space = await self.get_disk_space()
            
            response_time = (time.time() - start_time) * 1000
            
            # Calculate disk space
            total_free_space = sum(
                disk.get('freeSpace', 0) for disk in disk_space
            ) / (1024**3)  # Convert to GB
            
            return MediaServerMetrics(
                timestamp=time.time(),
                service_name=self.service_name,
                status='up' if status else 'down',
                response_time=response_time,
                active_streams=0,  # Not applicable
                transcoding_sessions=0,  # Not applicable
                disk_space_gb=total_free_space,
                bandwidth_mbps=0.0,  # Would need network monitoring
                cpu_usage=0.0,  # Would need container metrics
                memory_usage_mb=0,  # Would need container metrics
                error_count=0,
                queue_size=len(queue),
                health_score=100.0 if status else 0.0
            )
            
        except Exception as e:
            logger.error(f"Error collecting {self.service_name} metrics: {e}")
            return MediaServerMetrics(
                timestamp=time.time(),
                service_name=self.service_name,
                status='error',
                response_time=-1,
                active_streams=0,
                transcoding_sessions=0,
                disk_space_gb=0.0,
                bandwidth_mbps=0.0,
                cpu_usage=0.0,
                memory_usage_mb=0,
                error_count=1,
                queue_size=0,
                health_score=0.0
            )

class TautulliMonitor:
    """Monitor Tautulli (Plex monitoring)"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_activity(self) -> Dict[str, Any]:
        """Get current activity"""
        try:
            url = f"{self.base_url}/api/v2"
            params = {
                'apikey': self.api_key,
                'cmd': 'get_activity'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', {}).get('data', {})
                else:
                    logger.error(f"Failed to get Tautulli activity: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting Tautulli activity: {e}")
            return {}
            
    async def collect_metrics(self) -> MediaServerMetrics:
        """Collect Tautulli metrics"""
        start_time = time.time()
        
        try:
            activity = await self.get_activity()
            response_time = (time.time() - start_time) * 1000
            
            stream_count = activity.get('stream_count', 0)
            total_bandwidth = activity.get('total_bandwidth', 0)
            
            return MediaServerMetrics(
                timestamp=time.time(),
                service_name='tautulli',
                status='up' if activity else 'down',
                response_time=response_time,
                active_streams=stream_count,
                transcoding_sessions=activity.get('stream_count_transcode', 0),
                disk_space_gb=0.0,
                bandwidth_mbps=total_bandwidth / 1000,  # Convert Kbps to Mbps
                cpu_usage=0.0,
                memory_usage_mb=0,
                error_count=0,
                queue_size=0,
                health_score=100.0 if activity else 0.0
            )
            
        except Exception as e:
            logger.error(f"Error collecting Tautulli metrics: {e}")
            return MediaServerMetrics(
                timestamp=time.time(),
                service_name='tautulli',
                status='error',
                response_time=-1,
                active_streams=0,
                transcoding_sessions=0,
                disk_space_gb=0.0,
                bandwidth_mbps=0.0,
                cpu_usage=0.0,
                memory_usage_mb=0,
                error_count=1,
                queue_size=0,
                health_score=0.0
            )

class DockerContainerMonitor:
    """Monitor Docker containers for media services"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.enabled = True
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.enabled = False
            
    def get_container_metrics(self, container_name: str) -> Dict[str, Any]:
        """Get metrics for a specific container"""
        if not self.enabled:
            return {}
            
        try:
            container = self.client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_usage = self._calculate_cpu_usage(stats)
            
            # Calculate memory usage
            memory_usage = self._calculate_memory_usage(stats)
            
            # Get network I/O
            network_io = self._calculate_network_io(stats)
            
            return {
                'container_name': container_name,
                'status': container.status,
                'cpu_usage': cpu_usage,
                'memory_usage_mb': memory_usage['usage'] / (1024*1024),
                'memory_limit_mb': memory_usage['limit'] / (1024*1024),
                'network_rx_mb': network_io['rx_bytes'] / (1024*1024),
                'network_tx_mb': network_io['tx_bytes'] / (1024*1024),
                'started_at': container.attrs['State']['StartedAt'],
                'image': container.image.tags[0] if container.image.tags else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting container metrics for {container_name}: {e}")
            return {}
            
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage"""
        cpu_stats = stats.get('cpu_stats', {})
        precpu_stats = stats.get('precpu_stats', {})
        
        cpu_usage = cpu_stats.get('cpu_usage', {})
        precpu_usage = precpu_stats.get('cpu_usage', {})
        
        cpu_delta = cpu_usage.get('total_usage', 0) - precpu_usage.get('total_usage', 0)
        system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
        
        if system_delta > 0 and cpu_delta > 0:
            cpu_count = cpu_stats.get('online_cpus', 1)
            return (cpu_delta / system_delta) * cpu_count * 100.0
        return 0.0
        
    def _calculate_memory_usage(self, stats: Dict) -> Dict[str, int]:
        """Calculate memory usage"""
        memory_stats = stats.get('memory_stats', {})
        return {
            'usage': memory_stats.get('usage', 0),
            'limit': memory_stats.get('limit', 0)
        }
        
    def _calculate_network_io(self, stats: Dict) -> Dict[str, int]:
        """Calculate network I/O"""
        networks = stats.get('networks', {})
        
        total_rx = sum(net.get('rx_bytes', 0) for net in networks.values())
        total_tx = sum(net.get('tx_bytes', 0) for net in networks.values())
        
        return {
            'rx_bytes': total_rx,
            'tx_bytes': total_tx
        }

class MediaServerIntegrator:
    """Main integration orchestrator"""
    
    def __init__(self, config_path: str = "config/monitoring.yml"):
        self.config = self._load_config(config_path)
        self.docker_monitor = DockerContainerMonitor()
        
        # Initialize service monitors
        self.service_monitors = {}
        self._initialize_monitors()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'media_services': {
                'jellyfin': {
                    'enabled': True,
                    'url': 'http://jellyfin:8096',
                    'api_key': None,
                    'container_name': 'jellyfin'
                },
                'sonarr': {
                    'enabled': True,
                    'url': 'http://sonarr:8989',
                    'api_key': 'your-api-key',
                    'container_name': 'sonarr'
                },
                'radarr': {
                    'enabled': True,
                    'url': 'http://radarr:7878',
                    'api_key': 'your-api-key',
                    'container_name': 'radarr'
                },
                'prowlarr': {
                    'enabled': True,
                    'url': 'http://prowlarr:9696',
                    'api_key': 'your-api-key',
                    'container_name': 'prowlarr'
                },
                'tautulli': {
                    'enabled': False,
                    'url': 'http://tautulli:8181',
                    'api_key': 'your-api-key',
                    'container_name': 'tautulli'
                }
            }
        }
        
    def _initialize_monitors(self):
        """Initialize service monitors"""
        media_services = self.config.get('media_services', {})
        
        for service_name, service_config in media_services.items():
            if not service_config.get('enabled', False):
                continue
                
            try:
                if service_name == 'jellyfin':
                    self.service_monitors[service_name] = JellyfinMonitor(
                        service_config['url'],
                        service_config.get('api_key')
                    )
                elif service_name in ['sonarr', 'radarr', 'prowlarr']:
                    self.service_monitors[service_name] = ArrServiceMonitor(
                        service_name,
                        service_config['url'],
                        service_config['api_key']
                    )
                elif service_name == 'tautulli':
                    self.service_monitors[service_name] = TautulliMonitor(
                        service_config['url'],
                        service_config['api_key']
                    )
                    
                logger.info(f"Initialized monitor for {service_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize monitor for {service_name}: {e}")
                
    async def collect_all_metrics(self) -> List[MediaServerMetrics]:
        """Collect metrics from all configured services"""
        metrics = []
        
        # Collect metrics from each service
        for service_name, monitor in self.service_monitors.items():
            try:
                async with monitor:
                    service_metrics = await monitor.collect_metrics()
                    
                    # Enhance with Docker container metrics
                    container_name = self.config.get('media_services', {}).get(
                        service_name, {}
                    ).get('container_name', service_name)
                    
                    container_metrics = self.docker_monitor.get_container_metrics(container_name)
                    
                    if container_metrics:
                        service_metrics.cpu_usage = container_metrics.get('cpu_usage', 0.0)
                        service_metrics.memory_usage_mb = container_metrics.get('memory_usage_mb', 0)
                        
                    metrics.append(service_metrics)
                    
            except Exception as e:
                logger.error(f"Error collecting metrics for {service_name}: {e}")
                
                # Create error metric
                error_metric = MediaServerMetrics(
                    timestamp=time.time(),
                    service_name=service_name,
                    status='error',
                    response_time=-1,
                    active_streams=0,
                    transcoding_sessions=0,
                    disk_space_gb=0.0,
                    bandwidth_mbps=0.0,
                    cpu_usage=0.0,
                    memory_usage_mb=0,
                    error_count=1,
                    queue_size=0,
                    health_score=0.0
                )
                metrics.append(error_metric)
                
        return metrics
        
    async def get_service_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary of all services"""
        metrics = await self.collect_all_metrics()
        
        total_services = len(metrics)
        healthy_services = len([m for m in metrics if m.status == 'up'])
        total_streams = sum(m.active_streams for m in metrics)
        total_transcoding = sum(m.transcoding_sessions for m in metrics)
        total_bandwidth = sum(m.bandwidth_mbps for m in metrics)
        average_response_time = sum(m.response_time for m in metrics if m.response_time > 0) / max(1, len([m for m in metrics if m.response_time > 0]))
        
        overall_health = (healthy_services / total_services) * 100 if total_services > 0 else 0
        
        return {
            'timestamp': time.time(),
            'total_services': total_services,
            'healthy_services': healthy_services,
            'overall_health_percent': overall_health,
            'total_active_streams': total_streams,
            'total_transcoding_sessions': total_transcoding,
            'total_bandwidth_mbps': total_bandwidth,
            'average_response_time_ms': average_response_time,
            'service_details': [asdict(m) for m in metrics]
        }
        
    async def get_transcoding_summary(self) -> Dict[str, Any]:
        """Get transcoding performance summary"""
        transcoding_jobs = []
        
        # Get transcoding info from Jellyfin
        jellyfin_monitor = self.service_monitors.get('jellyfin')
        if jellyfin_monitor:
            try:
                async with jellyfin_monitor:
                    jobs = await jellyfin_monitor.get_transcoding_jobs()
                    transcoding_jobs.extend(jobs)
            except Exception as e:
                logger.error(f"Error getting transcoding jobs: {e}")
                
        return {
            'timestamp': time.time(),
            'total_jobs': len(transcoding_jobs),
            'active_jobs': len([j for j in transcoding_jobs if j.progress_percent < 100]),
            'average_cpu_usage': sum(j.cpu_usage for j in transcoding_jobs) / max(1, len(transcoding_jobs)),
            'jobs': [asdict(j) for j in transcoding_jobs]
        }
        
    async def start_monitoring_loop(self, interval: int = 30):
        """Start continuous monitoring loop"""
        logger.info(f"Starting media server monitoring loop (interval: {interval}s)")
        
        while True:
            try:
                start_time = time.time()
                
                # Collect metrics
                metrics = await self.collect_all_metrics()
                
                # Log summary
                healthy_count = len([m for m in metrics if m.status == 'up'])
                total_count = len(metrics)
                
                logger.info(f"Collected metrics from {healthy_count}/{total_count} services")
                
                # Store metrics (would typically send to monitoring system)
                await self._store_metrics(metrics)
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
                
    async def _store_metrics(self, metrics: List[MediaServerMetrics]):
        """Store metrics (placeholder for actual storage implementation)"""
        # This would typically send metrics to InfluxDB, Prometheus, etc.
        for metric in metrics:
            logger.debug(f"Storing metric for {metric.service_name}: {metric.status}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Media Server Integration")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    async def main():
        config_path = args.config or "config/monitoring.yml"
        integrator = MediaServerIntegrator(config_path)
        
        # Get initial health summary
        health = await integrator.get_service_health_summary()
        
        print("\n" + "="*60)
        print("MEDIA SERVER HEALTH SUMMARY")
        print("="*60)
        print(f"Services: {health['healthy_services']}/{health['total_services']} healthy")
        print(f"Overall Health: {health['overall_health_percent']:.1f}%")
        print(f"Active Streams: {health['total_active_streams']}")
        print(f"Transcoding Sessions: {health['total_transcoding_sessions']}")
        print(f"Total Bandwidth: {health['total_bandwidth_mbps']:.1f} Mbps")
        print(f"Avg Response Time: {health['average_response_time_ms']:.1f} ms")
        
        print("\nService Details:")
        for service in health['service_details']:
            status_icon = "✓" if service['status'] == 'up' else "✗"
            print(f"  {status_icon} {service['service_name']}: {service['response_time']:.1f}ms")
            
        # Start monitoring loop
        print(f"\nStarting monitoring loop (interval: {args.interval}s)")
        await integrator.start_monitoring_loop(args.interval)
        
    asyncio.run(main())