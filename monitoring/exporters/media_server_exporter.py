#!/usr/bin/env python3
"""
Media Server Exporter for Prometheus
Monitors Jellyfin, Sonarr, Radarr, qBittorrent and file system metrics
"""

import os
import time
import json
import requests
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from prometheus_client import start_http_server, Gauge, Counter, Info, CollectorRegistry

# Metrics Registry
REGISTRY = CollectorRegistry(auto_describe=False)

# Media Library Metrics
MEDIA_LIBRARY_SIZE = Gauge('media_library_size_bytes', 'Total size of media library', ['type'], registry=REGISTRY)
MEDIA_LIBRARY_COUNT = Gauge('media_library_count', 'Number of media items', ['type'], registry=REGISTRY)
MEDIA_LIBRARY_QUALITY = Gauge('media_library_quality_distribution', 'Media quality distribution', ['type', 'quality'], registry=REGISTRY)

# Download Metrics
DOWNLOAD_SPEED = Gauge('download_speed_bytes_per_second', 'Current download speed', ['client'], registry=REGISTRY)
DOWNLOAD_QUEUE_SIZE = Gauge('download_queue_size', 'Number of items in download queue', ['client'], registry=REGISTRY)
DOWNLOAD_ACTIVE = Gauge('download_active_count', 'Number of active downloads', ['client'], registry=REGISTRY)

# Streaming Metrics
STREAMING_SESSIONS = Gauge('streaming_active_sessions', 'Number of active streaming sessions', ['server'], registry=REGISTRY)
STREAMING_BANDWIDTH = Gauge('streaming_bandwidth_mbps', 'Streaming bandwidth usage', ['server'], registry=REGISTRY)
STREAMING_TRANSCODING = Gauge('streaming_transcoding_sessions', 'Number of transcoding sessions', ['server'], registry=REGISTRY)

# Service Health Metrics
SERVICE_UP = Gauge('service_up', 'Service availability', ['service'], registry=REGISTRY)
SERVICE_RESPONSE_TIME = Gauge('service_response_time_seconds', 'Service response time', ['service'], registry=REGISTRY)

# User Activity Metrics
USER_ACTIVITY = Gauge('user_activity_count', 'User activity metrics', ['activity_type'], registry=REGISTRY)
USER_SESSIONS = Gauge('user_sessions_total', 'Total user sessions', ['server'], registry=REGISTRY)

# Content Metrics
CONTENT_ADDED = Counter('content_added_total', 'Content added to libraries', ['type'], registry=REGISTRY)
CONTENT_WATCHED = Counter('content_watched_total', 'Content watched', ['type'], registry=REGISTRY)

# System Info
SYSTEM_INFO = Info('media_server_info', 'Media server system information', registry=REGISTRY)

class MediaServerExporter:
    def __init__(self):
        self.jellyfin_url = os.getenv('JELLYFIN_URL', 'http://jellyfin:8096')
        self.sonarr_url = os.getenv('SONARR_URL', 'http://sonarr:8989')
        self.radarr_url = os.getenv('RADARR_URL', 'http://radarr:7878')
        self.qbittorrent_url = os.getenv('QBITTORRENT_URL', 'http://qbittorrent:8080')
        self.media_path = os.getenv('MEDIA_PATH', '/media')
        
        # API Keys (set these in environment)
        self.sonarr_api_key = os.getenv('SONARR_API_KEY', '')
        self.radarr_api_key = os.getenv('RADARR_API_KEY', '')
        self.jellyfin_api_key = os.getenv('JELLYFIN_API_KEY', '')
        
        self.session = requests.Session()
        self.session.timeout = 10

    def get_service_health(self, service_name: str, url: str, endpoint: str = '/health') -> tuple[bool, float]:
        """Check service health and response time"""
        try:
            start_time = time.time()
            response = self.session.get(f"{url}{endpoint}")
            response_time = time.time() - start_time
            
            is_healthy = response.status_code == 200
            SERVICE_UP.labels(service=service_name).set(1 if is_healthy else 0)
            SERVICE_RESPONSE_TIME.labels(service=service_name).set(response_time)
            
            return is_healthy, response_time
        except Exception as e:
            print(f"Health check failed for {service_name}: {e}")
            SERVICE_UP.labels(service=service_name).set(0)
            SERVICE_RESPONSE_TIME.labels(service=service_name).set(0)
            return False, 0

    def collect_jellyfin_metrics(self):
        """Collect Jellyfin streaming and user metrics"""
        try:
            # Check basic health
            self.get_service_health('jellyfin', self.jellyfin_url, '/health')
            
            if not self.jellyfin_api_key:
                print("Jellyfin API key not set, skipping detailed metrics")
                return
                
            headers = {'X-Emby-Token': self.jellyfin_api_key}
            
            # Get active sessions
            sessions_response = self.session.get(f"{self.jellyfin_url}/Sessions", headers=headers)
            if sessions_response.status_code == 200:
                sessions = sessions_response.json()
                active_sessions = len([s for s in sessions if s.get('NowPlayingItem')])
                transcoding_sessions = len([s for s in sessions if s.get('TranscodingInfo')])
                
                STREAMING_SESSIONS.labels(server='jellyfin').set(active_sessions)
                STREAMING_TRANSCODING.labels(server='jellyfin').set(transcoding_sessions)
                
                # Calculate bandwidth
                total_bandwidth = sum(s.get('TranscodingInfo', {}).get('Bitrate', 0) for s in sessions) / 1_000_000
                STREAMING_BANDWIDTH.labels(server='jellyfin').set(total_bandwidth)
            
            # Get library stats
            libraries_response = self.session.get(f"{self.jellyfin_url}/Library/VirtualFolders", headers=headers)
            if libraries_response.status_code == 200:
                libraries = libraries_response.json()
                for library in libraries:
                    lib_type = library.get('CollectionType', 'mixed')
                    # Get item count for this library
                    items_response = self.session.get(
                        f"{self.jellyfin_url}/Items/Counts",
                        params={'ParentId': library.get('ItemId')},
                        headers=headers
                    )
                    if items_response.status_code == 200:
                        counts = items_response.json()
                        MEDIA_LIBRARY_COUNT.labels(type=lib_type).set(counts.get('TotalRecordCount', 0))
                        
        except Exception as e:
            print(f"Error collecting Jellyfin metrics: {e}")

    def collect_sonarr_metrics(self):
        """Collect Sonarr TV show metrics"""
        try:
            self.get_service_health('sonarr', self.sonarr_url, '/api/v3/system/status')
            
            if not self.sonarr_api_key:
                print("Sonarr API key not set, skipping detailed metrics")
                return
                
            headers = {'X-Api-Key': self.sonarr_api_key}
            
            # Get series count
            series_response = self.session.get(f"{self.sonarr_url}/api/v3/series", headers=headers)
            if series_response.status_code == 200:
                series = series_response.json()
                MEDIA_LIBRARY_COUNT.labels(type='tv_series').set(len(series))
                
                # Quality distribution
                quality_counts = {}
                for show in series:
                    profile = show.get('qualityProfileId', 'unknown')
                    quality_counts[profile] = quality_counts.get(profile, 0) + 1
                
                for quality, count in quality_counts.items():
                    MEDIA_LIBRARY_QUALITY.labels(type='tv', quality=str(quality)).set(count)
            
            # Get queue
            queue_response = self.session.get(f"{self.sonarr_url}/api/v3/queue", headers=headers)
            if queue_response.status_code == 200:
                queue = queue_response.json()
                DOWNLOAD_QUEUE_SIZE.labels(client='sonarr').set(len(queue.get('records', [])))
                
        except Exception as e:
            print(f"Error collecting Sonarr metrics: {e}")

    def collect_radarr_metrics(self):
        """Collect Radarr movie metrics"""
        try:
            self.get_service_health('radarr', self.radarr_url, '/api/v3/system/status')
            
            if not self.radarr_api_key:
                print("Radarr API key not set, skipping detailed metrics")
                return
                
            headers = {'X-Api-Key': self.radarr_api_key}
            
            # Get movie count
            movies_response = self.session.get(f"{self.radarr_url}/api/v3/movie", headers=headers)
            if movies_response.status_code == 200:
                movies = movies_response.json()
                MEDIA_LIBRARY_COUNT.labels(type='movies').set(len(movies))
                
                # Quality distribution
                quality_counts = {}
                for movie in movies:
                    profile = movie.get('qualityProfileId', 'unknown')
                    quality_counts[profile] = quality_counts.get(profile, 0) + 1
                
                for quality, count in quality_counts.items():
                    MEDIA_LIBRARY_QUALITY.labels(type='movie', quality=str(quality)).set(count)
            
            # Get queue
            queue_response = self.session.get(f"{self.radarr_url}/api/v3/queue", headers=headers)
            if queue_response.status_code == 200:
                queue = queue_response.json()
                DOWNLOAD_QUEUE_SIZE.labels(client='radarr').set(len(queue.get('records', [])))
                
        except Exception as e:
            print(f"Error collecting Radarr metrics: {e}")

    def collect_qbittorrent_metrics(self):
        """Collect qBittorrent metrics"""
        try:
            self.get_service_health('qbittorrent', self.qbittorrent_url, '/api/v2/app/version')
            
            # Get torrent info
            try:
                # Login (if needed)
                login_response = self.session.post(f"{self.qbittorrent_url}/api/v2/auth/login", 
                                                 data={'username': 'admin', 'password': 'adminadmin'})
                
                # Get active torrents
                torrents_response = self.session.get(f"{self.qbittorrent_url}/api/v2/torrents/info")
                if torrents_response.status_code == 200:
                    torrents = torrents_response.json()
                    
                    active_downloads = len([t for t in torrents if t.get('state') in ['downloading', 'stalledDL']])
                    total_speed = sum(t.get('dlspeed', 0) for t in torrents)
                    
                    DOWNLOAD_ACTIVE.labels(client='qbittorrent').set(active_downloads)
                    DOWNLOAD_SPEED.labels(client='qbittorrent').set(total_speed)
                    DOWNLOAD_QUEUE_SIZE.labels(client='qbittorrent').set(len(torrents))
                    
            except Exception as e:
                print(f"Error getting qBittorrent detailed metrics: {e}")
                
        except Exception as e:
            print(f"Error collecting qBittorrent metrics: {e}")

    def collect_filesystem_metrics(self):
        """Collect media filesystem metrics"""
        try:
            media_path = Path(self.media_path)
            if not media_path.exists():
                print(f"Media path {self.media_path} does not exist")
                return
            
            # Calculate directory sizes
            for category in ['movies', 'tv', 'music', 'books']:
                category_path = media_path / category
                if category_path.exists():
                    total_size = sum(f.stat().st_size for f in category_path.rglob('*') if f.is_file())
                    file_count = len(list(category_path.rglob('*')))
                    
                    MEDIA_LIBRARY_SIZE.labels(type=category).set(total_size)
                    if category not in ['tv_series', 'movies']:  # Avoid duplication with API metrics
                        MEDIA_LIBRARY_COUNT.labels(type=category).set(file_count)
                        
        except Exception as e:
            print(f"Error collecting filesystem metrics: {e}")

    def collect_metrics(self):
        """Collect all metrics"""
        print("Collecting media server metrics...")
        
        # Update system info
        SYSTEM_INFO.info({
            'version': '1.0.0',
            'media_path': self.media_path,
            'services': 'jellyfin,sonarr,radarr,qbittorrent'
        })
        
        # Collect from all services
        self.collect_jellyfin_metrics()
        self.collect_sonarr_metrics()
        self.collect_radarr_metrics()
        self.collect_qbittorrent_metrics()
        self.collect_filesystem_metrics()
        
        print("Metrics collection completed")

def main():
    exporter = MediaServerExporter()
    
    # Start metrics server
    start_http_server(9500, registry=REGISTRY)
    print("Media Server Exporter started on port 9500")
    
    while True:
        try:
            exporter.collect_metrics()
            time.sleep(60)  # Collect metrics every minute
        except KeyboardInterrupt:
            print("Shutting down exporter...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(30)

if __name__ == '__main__':
    main()