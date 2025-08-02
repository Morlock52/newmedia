#!/usr/bin/env python3
"""
Maintenance Automation
Handles backup schedules, update workflows, health monitoring, and log rotation
"""

import os
import sys
import json
import logging
import shutil
import tarfile
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import time
import psutil
import docker
import requests
from typing import Dict, List, Optional, Tuple
import smtplib
from email.mime.text import MIMEText
import sqlite3
import gzip
from concurrent.futures import ThreadPoolExecutor
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/maintenance_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MaintenanceManager:
    def __init__(self, config_path='/config/maintenance.json'):
        self.config = self.load_config(config_path)
        self.docker_client = docker.from_env()
        self.db_path = self.config.get('database_path', '/config/maintenance.db')
        self.init_database()
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        default_config = {
            'backup': {
                'enabled': True,
                'schedule': 'daily',  # daily, weekly, monthly
                'time': '02:00',
                'retention_days': 30,
                'backup_path': '/backups',
                'include_paths': [
                    '/config',
                    '/media/metadata'
                ],
                'exclude_patterns': [
                    '*.tmp',
                    '*.log',
                    'cache/*'
                ],
                'compression': 'gzip',
                'encrypt': False,
                'remote_backup': {
                    'enabled': False,
                    'type': 's3',  # s3, ftp, rsync
                    'destination': ''
                }
            },
            'updates': {
                'enabled': True,
                'auto_update': False,
                'check_schedule': 'daily',
                'maintenance_window': {
                    'start': '03:00',
                    'end': '05:00'
                },
                'update_order': [
                    'prowlarr',
                    'radarr',
                    'sonarr',
                    'bazarr',
                    'overseerr',
                    'jellyfin'
                ],
                'pre_update_backup': True
            },
            'health_monitoring': {
                'enabled': True,
                'check_interval': 5,  # minutes
                'thresholds': {
                    'cpu_percent': 90,
                    'memory_percent': 85,
                    'disk_percent': 90,
                    'response_time_ms': 5000
                },
                'services': [
                    {
                        'name': 'jellyfin',
                        'url': 'http://jellyfin:8096/health',
                        'critical': True
                    },
                    {
                        'name': 'radarr',
                        'url': 'http://radarr:7878/api/v3/health',
                        'critical': True
                    },
                    {
                        'name': 'sonarr',
                        'url': 'http://sonarr:8989/api/v3/health',
                        'critical': True
                    }
                ],
                'alerts': {
                    'email': True,
                    'webhook': True
                }
            },
            'log_rotation': {
                'enabled': True,
                'max_size_mb': 100,
                'max_files': 10,
                'compress': True,
                'paths': [
                    '/var/log/*.log',
                    '/config/*/logs/*.log'
                ]
            },
            'cleanup': {
                'enabled': True,
                'schedule': 'weekly',
                'temp_files_days': 7,
                'orphaned_files': True,
                'empty_directories': True,
                'cache_cleanup': True
            },
            'notifications': {
                'email': {
                    'enabled': True,
                    'smtp_host': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'smtp_user': '',
                    'smtp_password': '',
                    'to_addresses': []
                },
                'webhook': {
                    'enabled': True,
                    'url': ''
                }
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def init_database(self):
        """Initialize SQLite database for tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Backup history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                backup_type TEXT NOT NULL,
                backup_path TEXT NOT NULL,
                size_bytes INTEGER,
                duration_seconds INTEGER,
                status TEXT NOT NULL,
                error_message TEXT
            )
        ''')
        
        # Update history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS update_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                service_name TEXT NOT NULL,
                old_version TEXT,
                new_version TEXT,
                status TEXT NOT NULL,
                error_message TEXT
            )
        ''')
        
        # Health checks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                service_name TEXT NOT NULL,
                status TEXT NOT NULL,
                response_time_ms INTEGER,
                cpu_percent REAL,
                memory_percent REAL,
                disk_percent REAL,
                error_message TEXT
            )
        ''')
        
        # Maintenance tasks
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT NOT NULL,
                scheduled_date TIMESTAMP,
                completed_date TIMESTAMP,
                status TEXT NOT NULL,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def perform_backup(self, backup_type='scheduled'):
        """Perform system backup"""
        start_time = datetime.now()
        backup_name = f"media_server_backup_{start_time.strftime('%Y%m%d_%H%M%S')}"
        backup_path = Path(self.config['backup']['backup_path']) / backup_name
        
        try:
            logger.info(f"Starting {backup_type} backup: {backup_name}")
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup each configured path
            total_size = 0
            for include_path in self.config['backup']['include_paths']:
                if os.path.exists(include_path):
                    path_name = Path(include_path).name
                    tar_path = backup_path / f"{path_name}.tar.gz"
                    
                    # Create tar archive
                    with tarfile.open(tar_path, 'w:gz') as tar:
                        tar.add(include_path, arcname=path_name,
                               filter=self._tar_filter)
                    
                    total_size += tar_path.stat().st_size
                    logger.info(f"Backed up {include_path} to {tar_path}")
            
            # Backup Docker volumes
            self._backup_docker_volumes(backup_path)
            
            # Backup database dumps
            self._backup_databases(backup_path)
            
            # Create backup manifest
            manifest = {
                'backup_date': start_time.isoformat(),
                'backup_type': backup_type,
                'included_paths': self.config['backup']['include_paths'],
                'docker_volumes': self._get_docker_volumes(),
                'total_size': total_size,
                'compression': self.config['backup']['compression']
            }
            
            with open(backup_path / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Compress entire backup if configured
            if self.config['backup']['compression'] == 'gzip':
                archive_path = f"{backup_path}.tar.gz"
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(backup_path, arcname=backup_name)
                
                # Remove uncompressed directory
                shutil.rmtree(backup_path)
                backup_path = archive_path
                total_size = Path(archive_path).stat().st_size
            
            # Upload to remote if configured
            if self.config['backup']['remote_backup']['enabled']:
                self._upload_backup(backup_path)
            
            # Clean old backups
            self._cleanup_old_backups()
            
            # Record in database
            duration = (datetime.now() - start_time).total_seconds()
            self._record_backup(backup_type, str(backup_path), total_size, duration, 'success')
            
            logger.info(f"Backup completed successfully: {backup_path}")
            self._send_notification('Backup Completed', 
                                  f'Backup {backup_name} completed successfully. Size: {total_size/(1024**3):.2f}GB')
            
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            self._record_backup(backup_type, str(backup_path), 0, 0, 'failed', str(e))
            self._send_notification('Backup Failed', f'Backup {backup_name} failed: {str(e)}', 'error')
            return False
    
    def _tar_filter(self, tarinfo):
        """Filter function for tar archive"""
        # Skip excluded patterns
        for pattern in self.config['backup']['exclude_patterns']:
            if pattern.replace('*', '') in tarinfo.name:
                return None
        return tarinfo
    
    def _backup_docker_volumes(self, backup_path: Path):
        """Backup Docker volumes"""
        try:
            volumes_path = backup_path / 'docker_volumes'
            volumes_path.mkdir(exist_ok=True)
            
            for container in self.docker_client.containers.list():
                for mount in container.attrs['Mounts']:
                    if mount['Type'] == 'volume':
                        volume_name = mount['Name']
                        volume_backup = volumes_path / f"{volume_name}.tar"
                        
                        # Use docker to backup volume
                        backup_cmd = f"docker run --rm -v {volume_name}:/data -v {volumes_path}:/backup alpine tar cf /backup/{volume_name}.tar /data"
                        subprocess.run(backup_cmd, shell=True, check=True)
                        
                        logger.info(f"Backed up Docker volume: {volume_name}")
                        
        except Exception as e:
            logger.error(f"Error backing up Docker volumes: {e}")
    
    def _backup_databases(self, backup_path: Path):
        """Backup service databases"""
        db_path = backup_path / 'databases'
        db_path.mkdir(exist_ok=True)
        
        # Backup SQLite databases
        sqlite_paths = [
            '/config/radarr/radarr.db',
            '/config/sonarr/sonarr.db',
            '/config/bazarr/bazarr.db'
        ]
        
        for db_file in sqlite_paths:
            if os.path.exists(db_file):
                shutil.copy2(db_file, db_path / Path(db_file).name)
                logger.info(f"Backed up database: {db_file}")
    
    def _get_docker_volumes(self) -> List[str]:
        """Get list of Docker volumes"""
        volumes = []
        try:
            for volume in self.docker_client.volumes.list():
                volumes.append(volume.name)
        except Exception as e:
            logger.error(f"Error listing Docker volumes: {e}")
        return volumes
    
    def _upload_backup(self, backup_path: Path):
        """Upload backup to remote storage"""
        # Implement based on configured remote type
        remote_type = self.config['backup']['remote_backup']['type']
        
        if remote_type == 's3':
            # Implement S3 upload
            pass
        elif remote_type == 'ftp':
            # Implement FTP upload
            pass
        elif remote_type == 'rsync':
            # Implement rsync
            pass
    
    def _cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        retention_days = self.config['backup']['retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        backup_dir = Path(self.config['backup']['backup_path'])
        if not backup_dir.exists():
            return
        
        for backup in backup_dir.glob('media_server_backup_*'):
            try:
                # Extract date from filename
                date_str = backup.name.split('_')[3] + '_' + backup.name.split('_')[4].split('.')[0]
                backup_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                
                if backup_date < cutoff_date:
                    if backup.is_file():
                        backup.unlink()
                    else:
                        shutil.rmtree(backup)
                    logger.info(f"Removed old backup: {backup}")
                    
            except Exception as e:
                logger.error(f"Error processing backup {backup}: {e}")
    
    def check_updates(self):
        """Check for available updates"""
        updates = []
        
        for service in self.config['updates']['update_order']:
            try:
                container = self.docker_client.containers.get(service)
                current_image = container.image.tags[0] if container.image.tags else 'unknown'
                
                # Pull latest image to check
                logger.info(f"Checking updates for {service}")
                image_name = current_image.split(':')[0]
                
                # Get latest version info
                latest_image = self.docker_client.images.pull(f"{image_name}:latest")
                
                if latest_image.id != container.image.id:
                    updates.append({
                        'service': service,
                        'current_version': current_image,
                        'latest_version': f"{image_name}:latest",
                        'container_id': container.id
                    })
                    
            except Exception as e:
                logger.error(f"Error checking updates for {service}: {e}")
        
        return updates
    
    def perform_updates(self, services: Optional[List[str]] = None):
        """Perform updates for specified services"""
        if not self._in_maintenance_window():
            logger.warning("Not in maintenance window, skipping updates")
            return
        
        # Backup before updates if configured
        if self.config['updates']['pre_update_backup']:
            if not self.perform_backup('pre_update'):
                logger.error("Pre-update backup failed, aborting updates")
                return
        
        services_to_update = services or self.config['updates']['update_order']
        
        for service in services_to_update:
            try:
                self._update_service(service)
            except Exception as e:
                logger.error(f"Failed to update {service}: {e}")
                self._record_update(service, None, None, 'failed', str(e))
    
    def _update_service(self, service_name: str):
        """Update a single service"""
        logger.info(f"Updating {service_name}")
        
        try:
            # Get container
            container = self.docker_client.containers.get(service_name)
            old_image = container.image.tags[0] if container.image.tags else 'unknown'
            
            # Stop container
            container.stop()
            
            # Remove old container
            container.remove()
            
            # Pull latest image
            image_name = old_image.split(':')[0]
            new_image = self.docker_client.images.pull(f"{image_name}:latest")
            
            # Get container config from docker-compose
            compose_config = self._get_compose_config(service_name)
            
            # Create new container
            new_container = self.docker_client.containers.create(
                image=new_image.tags[0],
                name=service_name,
                **compose_config
            )
            
            # Start new container
            new_container.start()
            
            # Wait for health check
            time.sleep(30)
            
            # Verify service is running
            if self._verify_service_health(service_name):
                self._record_update(service_name, old_image, new_image.tags[0], 'success')
                logger.info(f"Successfully updated {service_name}")
            else:
                # Rollback
                self._rollback_service(service_name, old_image)
                self._record_update(service_name, old_image, new_image.tags[0], 'failed', 'Health check failed')
                
        except Exception as e:
            logger.error(f"Error updating {service_name}: {e}")
            raise
    
    def _get_compose_config(self, service_name: str) -> Dict:
        """Get container configuration from docker-compose.yml"""
        # Parse docker-compose.yml
        compose_path = Path('/docker/docker-compose.yml')
        if compose_path.exists():
            with open(compose_path, 'r') as f:
                compose = yaml.safe_load(f)
            
            if service_name in compose.get('services', {}):
                service_config = compose['services'][service_name]
                
                # Convert compose config to container config
                return {
                    'environment': service_config.get('environment', {}),
                    'volumes': service_config.get('volumes', []),
                    'ports': self._parse_ports(service_config.get('ports', [])),
                    'restart_policy': {'Name': service_config.get('restart', 'unless-stopped')},
                    'network': 'media-net'
                }
        
        return {}
    
    def _parse_ports(self, ports: List[str]) -> Dict:
        """Parse port mappings from compose format"""
        port_bindings = {}
        for port in ports:
            if ':' in port:
                host_port, container_port = port.split(':')
                port_bindings[container_port] = host_port
        return port_bindings
    
    def _rollback_service(self, service_name: str, old_image: str):
        """Rollback service to previous version"""
        logger.warning(f"Rolling back {service_name} to {old_image}")
        
        try:
            # Stop current container
            container = self.docker_client.containers.get(service_name)
            container.stop()
            container.remove()
            
            # Create container with old image
            compose_config = self._get_compose_config(service_name)
            old_container = self.docker_client.containers.create(
                image=old_image,
                name=service_name,
                **compose_config
            )
            
            old_container.start()
            logger.info(f"Rolled back {service_name} to {old_image}")
            
        except Exception as e:
            logger.error(f"Error rolling back {service_name}: {e}")
    
    def monitor_health(self):
        """Monitor system and service health"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'system': self._check_system_health(),
            'services': self._check_services_health()
        }
        
        # Check for issues
        issues = []
        
        # System issues
        if health_status['system']['cpu_percent'] > self.config['health_monitoring']['thresholds']['cpu_percent']:
            issues.append(f"High CPU usage: {health_status['system']['cpu_percent']}%")
        
        if health_status['system']['memory_percent'] > self.config['health_monitoring']['thresholds']['memory_percent']:
            issues.append(f"High memory usage: {health_status['system']['memory_percent']}%")
        
        if health_status['system']['disk_percent'] > self.config['health_monitoring']['thresholds']['disk_percent']:
            issues.append(f"High disk usage: {health_status['system']['disk_percent']}%")
        
        # Service issues
        for service in health_status['services']:
            if service['status'] != 'healthy':
                if service['critical']:
                    issues.append(f"Critical service {service['name']} is {service['status']}")
                else:
                    issues.append(f"Service {service['name']} is {service['status']}")
        
        # Send alerts if issues
        if issues:
            self._send_health_alert(issues)
        
        return health_status
    
    def _check_system_health(self) -> Dict:
        """Check system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': os.getloadavg(),
            'uptime_hours': (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600
        }
    
    def _check_services_health(self) -> List[Dict]:
        """Check health of configured services"""
        services_health = []
        
        for service in self.config['health_monitoring']['services']:
            health = {
                'name': service['name'],
                'critical': service.get('critical', False),
                'status': 'unknown',
                'response_time_ms': None,
                'error': None
            }
            
            try:
                # Check if container is running
                container = self.docker_client.containers.get(service['name'])
                if container.status != 'running':
                    health['status'] = 'stopped'
                else:
                    # Check service endpoint
                    start_time = time.time()
                    response = requests.get(
                        service['url'],
                        timeout=5,
                        headers={'X-Api-Key': self.config.get(f'{service["name"]}_api_key', '')}
                    )
                    response_time = (time.time() - start_time) * 1000
                    
                    health['response_time_ms'] = int(response_time)
                    
                    if response.status_code == 200:
                        health['status'] = 'healthy'
                    else:
                        health['status'] = 'unhealthy'
                        health['error'] = f"HTTP {response.status_code}"
                        
            except requests.RequestException as e:
                health['status'] = 'unreachable'
                health['error'] = str(e)
            except docker.errors.NotFound:
                health['status'] = 'not_found'
                health['error'] = 'Container not found'
            except Exception as e:
                health['status'] = 'error'
                health['error'] = str(e)
            
            services_health.append(health)
            
            # Record in database
            self._record_health_check(health)
        
        return services_health
    
    def _verify_service_health(self, service_name: str) -> bool:
        """Verify service is healthy after update"""
        service_config = next((s for s in self.config['health_monitoring']['services'] 
                             if s['name'] == service_name), None)
        
        if not service_config:
            return True  # No health check configured
        
        try:
            response = requests.get(
                service_config['url'],
                timeout=10,
                headers={'X-Api-Key': self.config.get(f'{service_name}_api_key', '')}
            )
            return response.status_code == 200
        except:
            return False
    
    def rotate_logs(self):
        """Rotate log files based on size"""
        for log_pattern in self.config['log_rotation']['paths']:
            for log_file in Path('/').glob(log_pattern.lstrip('/')):
                try:
                    if not log_file.exists():
                        continue
                    
                    file_size_mb = log_file.stat().st_size / (1024 * 1024)
                    
                    if file_size_mb > self.config['log_rotation']['max_size_mb']:
                        self._rotate_log_file(log_file)
                        
                except Exception as e:
                    logger.error(f"Error rotating log {log_file}: {e}")
    
    def _rotate_log_file(self, log_file: Path):
        """Rotate a single log file"""
        logger.info(f"Rotating log file: {log_file}")
        
        # Find next available number
        i = 1
        while (log_file.parent / f"{log_file.name}.{i}").exists():
            i += 1
        
        # Rename current file
        rotated_file = log_file.parent / f"{log_file.name}.{i}"
        log_file.rename(rotated_file)
        
        # Compress if configured
        if self.config['log_rotation']['compress']:
            with open(rotated_file, 'rb') as f_in:
                with gzip.open(f"{rotated_file}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            rotated_file.unlink()
            rotated_file = Path(f"{rotated_file}.gz")
        
        # Create new empty log file
        log_file.touch()
        
        # Clean old logs
        self._cleanup_old_logs(log_file)
    
    def _cleanup_old_logs(self, log_file: Path):
        """Remove old rotated logs"""
        max_files = self.config['log_rotation']['max_files']
        
        # Get all rotated files
        rotated_files = []
        for ext in ['', '.gz']:
            rotated_files.extend(log_file.parent.glob(f"{log_file.name}.*{ext}"))
        
        # Sort by modification time
        rotated_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old files
        for old_file in rotated_files[max_files:]:
            old_file.unlink()
            logger.info(f"Removed old log: {old_file}")
    
    def perform_cleanup(self):
        """Perform system cleanup tasks"""
        logger.info("Starting cleanup tasks")
        
        # Clean temporary files
        if self.config['cleanup']['temp_files_days'] > 0:
            self._cleanup_temp_files()
        
        # Clean orphaned files
        if self.config['cleanup']['orphaned_files']:
            self._cleanup_orphaned_files()
        
        # Clean empty directories
        if self.config['cleanup']['empty_directories']:
            self._cleanup_empty_directories()
        
        # Clean cache
        if self.config['cleanup']['cache_cleanup']:
            self._cleanup_cache()
        
        logger.info("Cleanup tasks completed")
    
    def _cleanup_temp_files(self):
        """Remove old temporary files"""
        temp_dirs = ['/tmp', '/var/tmp', '/media/.temp']
        cutoff_date = datetime.now() - timedelta(days=self.config['cleanup']['temp_files_days'])
        
        for temp_dir in temp_dirs:
            if not os.path.exists(temp_dir):
                continue
            
            for item in Path(temp_dir).iterdir():
                try:
                    if item.stat().st_mtime < cutoff_date.timestamp():
                        if item.is_file():
                            item.unlink()
                        else:
                            shutil.rmtree(item)
                        logger.info(f"Removed old temp file: {item}")
                except Exception as e:
                    logger.error(f"Error removing {item}: {e}")
    
    def _cleanup_orphaned_files(self):
        """Clean orphaned media files"""
        # This would integrate with media server APIs to find orphaned files
        pass
    
    def _cleanup_empty_directories(self):
        """Remove empty directories"""
        media_paths = ['/media/movies', '/media/tv', '/media/music']
        
        for media_path in media_paths:
            if not os.path.exists(media_path):
                continue
            
            for root, dirs, files in os.walk(media_path, topdown=False):
                if not dirs and not files:
                    try:
                        os.rmdir(root)
                        logger.info(f"Removed empty directory: {root}")
                    except Exception as e:
                        logger.error(f"Error removing directory {root}: {e}")
    
    def _cleanup_cache(self):
        """Clean application caches"""
        cache_dirs = [
            '/config/jellyfin/cache',
            '/config/radarr/MediaCover',
            '/config/sonarr/MediaCover'
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    # Keep structure but remove old files
                    cutoff_date = datetime.now() - timedelta(days=30)
                    
                    for item in Path(cache_dir).rglob('*'):
                        if item.is_file() and item.stat().st_mtime < cutoff_date.timestamp():
                            item.unlink()
                            
                except Exception as e:
                    logger.error(f"Error cleaning cache {cache_dir}: {e}")
    
    def _in_maintenance_window(self) -> bool:
        """Check if current time is within maintenance window"""
        now = datetime.now().time()
        start = datetime.strptime(self.config['updates']['maintenance_window']['start'], '%H:%M').time()
        end = datetime.strptime(self.config['updates']['maintenance_window']['end'], '%H:%M').time()
        
        if start <= end:
            return start <= now <= end
        else:
            return now >= start or now <= end
    
    def _record_backup(self, backup_type: str, backup_path: str, size_bytes: int,
                      duration_seconds: int, status: str, error_message: str = None):
        """Record backup in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO backup_history 
            (backup_type, backup_path, size_bytes, duration_seconds, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (backup_type, backup_path, size_bytes, duration_seconds, status, error_message))
        
        conn.commit()
        conn.close()
    
    def _record_update(self, service_name: str, old_version: str, new_version: str,
                      status: str, error_message: str = None):
        """Record update in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO update_history 
            (service_name, old_version, new_version, status, error_message)
            VALUES (?, ?, ?, ?, ?)
        ''', (service_name, old_version, new_version, status, error_message))
        
        conn.commit()
        conn.close()
    
    def _record_health_check(self, health: Dict):
        """Record health check in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        system_health = self._check_system_health()
        
        cursor.execute('''
            INSERT INTO health_checks 
            (service_name, status, response_time_ms, cpu_percent, memory_percent, disk_percent, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            health['name'],
            health['status'],
            health['response_time_ms'],
            system_health['cpu_percent'],
            system_health['memory_percent'],
            system_health['disk_percent'],
            health.get('error')
        ))
        
        conn.commit()
        conn.close()
    
    def _send_notification(self, subject: str, message: str, level: str = 'info'):
        """Send notification via configured channels"""
        if self.config['notifications']['email']['enabled']:
            self._send_email(subject, message, level)
        
        if self.config['notifications']['webhook']['enabled']:
            self._send_webhook(subject, message, level)
    
    def _send_email(self, subject: str, message: str, level: str):
        """Send email notification"""
        try:
            msg = MIMEText(message)
            msg['Subject'] = f"[Media Server] {subject}"
            msg['From'] = self.config['notifications']['email']['smtp_user']
            msg['To'] = ', '.join(self.config['notifications']['email']['to_addresses'])
            
            with smtplib.SMTP(self.config['notifications']['email']['smtp_host'],
                             self.config['notifications']['email']['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.config['notifications']['email']['smtp_user'],
                    self.config['notifications']['email']['smtp_password']
                )
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def _send_webhook(self, subject: str, message: str, level: str):
        """Send webhook notification"""
        try:
            payload = {
                'subject': subject,
                'message': message,
                'level': level,
                'timestamp': datetime.now().isoformat()
            }
            
            requests.post(self.config['notifications']['webhook']['url'], json=payload)
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
    
    def _send_health_alert(self, issues: List[str]):
        """Send health alert notification"""
        message = "Health check detected the following issues:\n\n"
        message += "\n".join(f"- {issue}" for issue in issues)
        
        self._send_notification('Health Alert', message, 'warning')
    
    def schedule_tasks(self):
        """Schedule maintenance tasks"""
        # Backup tasks
        if self.config['backup']['enabled']:
            if self.config['backup']['schedule'] == 'daily':
                schedule.every().day.at(self.config['backup']['time']).do(self.perform_backup)
            elif self.config['backup']['schedule'] == 'weekly':
                schedule.every().week.at(self.config['backup']['time']).do(self.perform_backup)
            elif self.config['backup']['schedule'] == 'monthly':
                schedule.every(30).days.at(self.config['backup']['time']).do(self.perform_backup)
        
        # Update checks
        if self.config['updates']['enabled']:
            if self.config['updates']['check_schedule'] == 'daily':
                schedule.every().day.at("01:00").do(self.check_and_update)
        
        # Health monitoring
        if self.config['health_monitoring']['enabled']:
            schedule.every(self.config['health_monitoring']['check_interval']).minutes.do(self.monitor_health)
        
        # Log rotation
        if self.config['log_rotation']['enabled']:
            schedule.every().day.at("00:00").do(self.rotate_logs)
        
        # Cleanup
        if self.config['cleanup']['enabled']:
            if self.config['cleanup']['schedule'] == 'weekly':
                schedule.every().week.at("04:00").do(self.perform_cleanup)
    
    def check_and_update(self):
        """Check for updates and apply if configured"""
        updates = self.check_updates()
        
        if updates:
            if self.config['updates']['auto_update']:
                self.perform_updates()
            else:
                # Just notify about available updates
                update_list = "\n".join([f"- {u['service']}: {u['current_version']} -> {u['latest_version']}" 
                                        for u in updates])
                self._send_notification('Updates Available', 
                                      f"The following updates are available:\n\n{update_list}")
    
    def run(self):
        """Run maintenance scheduler"""
        self.schedule_tasks()
        logger.info("Maintenance automation started")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Maintenance Automation')
    parser.add_argument('--config', default='/config/maintenance.json', help='Config file path')
    parser.add_argument('--backup', action='store_true', help='Perform backup now')
    parser.add_argument('--update', nargs='*', help='Update services (all if none specified)')
    parser.add_argument('--health', action='store_true', help='Check system health')
    parser.add_argument('--cleanup', action='store_true', help='Perform cleanup now')
    parser.add_argument('--rotate-logs', action='store_true', help='Rotate logs now')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    manager = MaintenanceManager(args.config)
    
    if args.backup:
        manager.perform_backup('manual')
    elif args.update is not None:
        manager.perform_updates(args.update if args.update else None)
    elif args.health:
        health = manager.monitor_health()
        print(json.dumps(health, indent=2))
    elif args.cleanup:
        manager.perform_cleanup()
    elif args.rotate_logs:
        manager.rotate_logs()
    elif args.daemon:
        manager.run()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()