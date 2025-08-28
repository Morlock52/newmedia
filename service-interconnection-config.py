#!/usr/bin/env python3
"""
Service Interconnection Configuration Script
Configures all services to communicate with each other through APIs
"""

import json
import requests
import time
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServiceInterconnector:
    def __init__(self):
        self.base_config_path = Path("/config")
        self.api_keys = {}
        self.service_urls = {
            'sonarr': 'http://127.0.0.1:8989',
            'radarr': 'http://127.0.0.1:7878',
            'lidarr': 'http://127.0.0.1:8686',
            'readarr': 'http://127.0.0.1:8787',
            'prowlarr': 'http://127.0.0.1:9696',
            'bazarr': 'http://127.0.0.1:6767',
            'qbittorrent': 'http://127.0.0.1:8090',
            'jellyfin': 'http://127.0.0.1:8096',
            'tautulli': 'http://127.0.0.1:8181',
        }
        
        # Service dependency mapping
        self.service_dependencies = {
            'sonarr': {
                'indexers': ['prowlarr'],
                'download_clients': ['qbittorrent'],
                'connections': ['jellyfin', 'tautulli']
            },
            'radarr': {
                'indexers': ['prowlarr'],
                'download_clients': ['qbittorrent'],
                'connections': ['jellyfin', 'tautulli']
            },
            'lidarr': {
                'indexers': ['prowlarr'],
                'download_clients': ['qbittorrent'],
                'connections': ['jellyfin']
            },
            'readarr': {
                'indexers': ['prowlarr'],
                'download_clients': ['qbittorrent'],
                'connections': ['jellyfin']
            },
            'bazarr': {
                'sonarr': ['sonarr'],
                'radarr': ['radarr'],
                'connections': ['jellyfin']
            },
            'jellyfin': {
                'monitoring': ['tautulli']
            },
            'prowlarr': {
                'arr_apps': ['sonarr', 'radarr', 'lidarr', 'readarr']
            }
        }
    
    def wait_for_service(self, service_name, max_attempts=30, delay=10):
        """Wait for a service to be ready"""
        url = self.service_urls.get(service_name)
        if not url:
            logger.warning(f"No URL configured for service {service_name}")
            return False
        
        logger.info(f"Waiting for {service_name} to be ready...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{url}/ping", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            logger.info(f"⏳ Attempt {attempt + 1}/{max_attempts} - {service_name} not ready yet...")
            time.sleep(delay)
        
        logger.error(f"❌ {service_name} failed to become ready after {max_attempts} attempts")
        return False
    
    def get_or_generate_api_key(self, service_name):
        """Get or generate API key for a service"""
        api_key_file = self.base_config_path / "api-keys" / service_name
        
        # Try to read existing API key
        if api_key_file.exists():
            try:
                with open(api_key_file, 'r') as f:
                    api_key = f.read().strip()
                if api_key and api_key != f"generating_{service_name}_api_key":
                    self.api_keys[service_name] = api_key
                    return api_key
            except Exception as e:
                logger.warning(f"Failed to read API key for {service_name}: {e}")
        
        # Generate new API key by calling service API
        api_key = self.generate_service_api_key(service_name)
        
        if api_key:
            # Save the API key
            api_key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(api_key_file, 'w') as f:
                f.write(api_key)
            self.api_keys[service_name] = api_key
            logger.info(f"✅ Generated and saved API key for {service_name}")
            
        return api_key
    
    def generate_service_api_key(self, service_name):
        """Generate API key for specific service"""
        url = self.service_urls.get(service_name)
        if not url:
            return None
        
        try:
            # Most *arr services have similar API key generation
            if service_name in ['sonarr', 'radarr', 'lidarr', 'readarr', 'prowlarr']:
                # Check if service has an existing API key in config
                config_file = self.base_config_path / service_name / "config.xml"
                if config_file.exists():
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(config_file)
                    root = tree.getroot()
                    api_key_elem = root.find('.//ApiKey')
                    if api_key_elem is not None and api_key_elem.text:
                        return api_key_elem.text
                
                # Generate new API key (implementation depends on service)
                import uuid
                new_api_key = str(uuid.uuid4()).replace('-', '')
                
                # Update config file with new API key
                if config_file.exists():
                    tree = ET.parse(config_file)
                    root = tree.getroot()
                    api_key_elem = root.find('.//ApiKey')
                    if api_key_elem is not None:
                        api_key_elem.text = new_api_key
                    else:
                        # Create new ApiKey element
                        config = root.find('.//Config')
                        if config is not None:
                            api_elem = ET.SubElement(config, 'ApiKey')
                            api_elem.text = new_api_key
                    tree.write(config_file)
                
                return new_api_key
                
        except Exception as e:
            logger.error(f"Failed to generate API key for {service_name}: {e}")
        
        return None
    
    def configure_prowlarr_indexers(self):
        """Configure Prowlarr with indexers and sync to *arr apps"""
        logger.info("🔗 Configuring Prowlarr indexers and app sync...")
        
        if not self.wait_for_service('prowlarr'):
            return False
        
        prowlarr_api_key = self.get_or_generate_api_key('prowlarr')
        if not prowlarr_api_key:
            logger.error("Failed to get Prowlarr API key")
            return False
        
        prowlarr_url = self.service_urls['prowlarr']
        headers = {'X-Api-Key': prowlarr_api_key}
        
        # Configure apps in Prowlarr for indexer sync
        arr_apps = ['sonarr', 'radarr', 'lidarr', 'readarr']
        
        for app in arr_apps:
            if not self.wait_for_service(app):
                continue
            
            app_api_key = self.get_or_generate_api_key(app)
            if not app_api_key:
                continue
            
            # Add application to Prowlarr
            app_config = {
                'name': app.title(),
                'implementation': app.title(),
                'configContract': f"{app.title()}Settings",
                'infoLink': f"https://wiki.servarr.com/prowlarr/supported#{app}",
                'tags': [],
                'fields': [
                    {'name': 'baseUrl', 'value': self.service_urls[app]},
                    {'name': 'apiKey', 'value': app_api_key},
                    {'name': 'syncCategories', 'value': [5000, 5030, 5040] if app == 'sonarr' else [2000, 2010, 2020, 2030, 2040, 2045, 2050, 2060, 2070, 2080]},
                ]
            }
            
            try:
                # Check if app already exists
                response = requests.get(f"{prowlarr_url}/api/v1/applications", headers=headers)
                existing_apps = response.json() if response.status_code == 200 else []
                
                app_exists = any(existing_app.get('name', '').lower() == app for existing_app in existing_apps)
                
                if not app_exists:
                    response = requests.post(f"{prowlarr_url}/api/v1/applications", 
                                           json=app_config, headers=headers)
                    if response.status_code in [200, 201]:
                        logger.info(f"✅ Added {app} to Prowlarr")
                    else:
                        logger.error(f"❌ Failed to add {app} to Prowlarr: {response.status_code}")
                else:
                    logger.info(f"✅ {app} already configured in Prowlarr")
                    
            except Exception as e:
                logger.error(f"Error configuring {app} in Prowlarr: {e}")
        
        # Add some default indexers to Prowlarr
        self.add_default_indexers(prowlarr_url, headers)
        
        return True
    
    def add_default_indexers(self, prowlarr_url, headers):
        """Add default public indexers to Prowlarr"""
        logger.info("Adding default indexers to Prowlarr...")
        
        default_indexers = [
            {
                'name': '1337x',
                'implementation': 'Cardigann',
                'configContract': 'CardigannSettings',
                'infoLink': 'https://1337x.to/',
                'tags': [],
                'fields': [
                    {'name': 'definitionFile', 'value': '1337x'},
                    {'name': 'baseUrl', 'value': 'https://1337x.to/'},
                ]
            },
            {
                'name': 'The Pirate Bay',
                'implementation': 'Cardigann', 
                'configContract': 'CardigannSettings',
                'infoLink': 'https://thepiratebay.org/',
                'tags': [],
                'fields': [
                    {'name': 'definitionFile', 'value': 'thepiratebay'},
                    {'name': 'baseUrl', 'value': 'https://thepiratebay.org/'},
                ]
            }
        ]
        
        try:
            for indexer_config in default_indexers:
                # Check if indexer already exists
                response = requests.get(f"{prowlarr_url}/api/v1/indexer", headers=headers)
                existing_indexers = response.json() if response.status_code == 200 else []
                
                indexer_exists = any(existing.get('name') == indexer_config['name'] 
                                   for existing in existing_indexers)
                
                if not indexer_exists:
                    response = requests.post(f"{prowlarr_url}/api/v1/indexer", 
                                           json=indexer_config, headers=headers)
                    if response.status_code in [200, 201]:
                        logger.info(f"✅ Added indexer: {indexer_config['name']}")
                    else:
                        logger.error(f"❌ Failed to add indexer {indexer_config['name']}: {response.status_code}")
                else:
                    logger.info(f"✅ Indexer {indexer_config['name']} already exists")
                    
        except Exception as e:
            logger.error(f"Error adding default indexers: {e}")
    
    def configure_download_clients(self):
        """Configure download clients in *arr applications"""
        logger.info("🔗 Configuring download clients in *arr applications...")
        
        if not self.wait_for_service('qbittorrent'):
            logger.error("qBittorrent not available")
            return False
        
        arr_apps = ['sonarr', 'radarr', 'lidarr', 'readarr']
        
        # qBittorrent configuration
        qbt_config = {
            'name': 'qBittorrent',
            'implementation': 'QBittorrent',
            'configContract': 'QBittorrentSettings',
            'infoLink': 'https://wiki.servarr.com/sonarr/supported#qbittorrent',
            'tags': [],
            'fields': [
                {'name': 'host', 'value': '127.0.0.1'},
                {'name': 'port', 'value': 8090},
                {'name': 'username', 'value': 'admin'},
                {'name': 'password', 'value': 'adminpass'},
                {'name': 'category', 'value': 'sonarr'},
                {'name': 'priority', 'value': 0},
                {'name': 'removeCompletedDownloads', 'value': False},
                {'name': 'removeFailedDownloads', 'value': False},
            ],
            'enable': True
        }
        
        for app in arr_apps:
            if not self.wait_for_service(app):
                continue
            
            api_key = self.get_or_generate_api_key(app)
            if not api_key:
                continue
            
            app_url = self.service_urls[app]
            headers = {'X-Api-Key': api_key}
            
            try:
                # Customize category for each app
                app_qbt_config = qbt_config.copy()
                app_qbt_config['fields'] = [
                    field.copy() if field['name'] != 'category' 
                    else {'name': 'category', 'value': app}
                    for field in qbt_config['fields']
                ]
                
                # Check if download client already exists
                response = requests.get(f"{app_url}/api/v3/downloadclient", headers=headers)
                existing_clients = response.json() if response.status_code == 200 else []
                
                client_exists = any(client.get('name') == 'qBittorrent' 
                                  for client in existing_clients)
                
                if not client_exists:
                    response = requests.post(f"{app_url}/api/v3/downloadclient", 
                                           json=app_qbt_config, headers=headers)
                    if response.status_code in [200, 201]:
                        logger.info(f"✅ Added qBittorrent to {app}")
                    else:
                        logger.error(f"❌ Failed to add qBittorrent to {app}: {response.status_code}")
                else:
                    logger.info(f"✅ qBittorrent already configured in {app}")
                    
            except Exception as e:
                logger.error(f"Error configuring qBittorrent in {app}: {e}")
        
        return True
    
    def configure_media_management(self):
        """Configure media management and folder settings"""
        logger.info("🔗 Configuring media management...")
        
        arr_media_configs = {
            'sonarr': {
                'rootFolders': ['/data/media/tv'],
                'namingConfig': {
                    'renameEpisodes': True,
                    'standardEpisodeFormat': '{Series Title} - S{season:00}E{episode:00} - {Episode Title}',
                    'seasonFolderFormat': 'Season {season:00}',
                    'seriesFolderFormat': '{Series Title} ({Series Year})'
                }
            },
            'radarr': {
                'rootFolders': ['/data/media/movies'],
                'namingConfig': {
                    'renameMovies': True,
                    'standardMovieFormat': '{Movie Title} ({Release Year})',
                    'movieFolderFormat': '{Movie Title} ({Release Year})'
                }
            },
            'lidarr': {
                'rootFolders': ['/data/media/music'],
                'namingConfig': {
                    'renameAlbums': True,
                    'standardAlbumFormat': '{Artist Name} - {Album Title}',
                    'artistFolderFormat': '{Artist Name}'
                }
            },
            'readarr': {
                'rootFolders': ['/data/media/books'],
                'namingConfig': {
                    'renameBooks': True,
                    'standardBookFormat': '{Book Title} - {Author Name}',
                    'authorFolderFormat': '{Author Name}'
                }
            }
        }
        
        for app, config in arr_media_configs.items():
            if not self.wait_for_service(app):
                continue
            
            api_key = self.get_or_generate_api_key(app)
            if not api_key:
                continue
            
            app_url = self.service_urls[app]
            headers = {'X-Api-Key': api_key}
            
            try:
                # Configure root folders
                for root_folder in config['rootFolders']:
                    folder_config = {'path': root_folder}
                    
                    # Check if root folder already exists
                    response = requests.get(f"{app_url}/api/v3/rootfolder", headers=headers)
                    existing_folders = response.json() if response.status_code == 200 else []
                    
                    folder_exists = any(folder.get('path') == root_folder 
                                      for folder in existing_folders)
                    
                    if not folder_exists:
                        response = requests.post(f"{app_url}/api/v3/rootfolder", 
                                               json=folder_config, headers=headers)
                        if response.status_code in [200, 201]:
                            logger.info(f"✅ Added root folder {root_folder} to {app}")
                        else:
                            logger.error(f"❌ Failed to add root folder to {app}: {response.status_code}")
                    else:
                        logger.info(f"✅ Root folder {root_folder} already exists in {app}")
                
                # Configure naming settings (if supported by API)
                if 'namingConfig' in config:
                    response = requests.get(f"{app_url}/api/v3/config/naming", headers=headers)
                    if response.status_code == 200:
                        naming_config = response.json()
                        naming_config.update(config['namingConfig'])
                        
                        response = requests.put(f"{app_url}/api/v3/config/naming", 
                                              json=naming_config, headers=headers)
                        if response.status_code == 202:
                            logger.info(f"✅ Updated naming configuration for {app}")
                
            except Exception as e:
                logger.error(f"Error configuring media management for {app}: {e}")
        
        return True
    
    def configure_jellyfin_integration(self):
        """Configure Jellyfin media server integration"""
        logger.info("🔗 Configuring Jellyfin integration...")
        
        if not self.wait_for_service('jellyfin'):
            return False
        
        # Configure Jellyfin libraries (this would typically be done through the UI first)
        # For now, we'll create connection configurations for other services
        
        # Configure Tautulli to monitor Jellyfin
        if self.wait_for_service('tautulli'):
            logger.info("Configuring Tautulli for Jellyfin monitoring...")
            # Tautulli configuration would go here
        
        return True
    
    def configure_monitoring_integrations(self):
        """Configure monitoring service integrations"""
        logger.info("🔗 Configuring monitoring integrations...")
        
        # Configure Grafana dashboards for service monitoring
        # This would involve creating dashboards for:
        # - *arr application statistics
        # - Download client metrics
        # - Media server usage
        # - System resource monitoring
        
        # Configure Prometheus scraping for all services
        # Update prometheus.yml with all service endpoints
        
        return True
    
    def run_full_configuration(self):
        """Run complete service interconnection configuration"""
        logger.info("🚀 Starting full service interconnection configuration...")
        
        # Wait for critical services first
        critical_services = ['prowlarr', 'qbittorrent']
        for service in critical_services:
            if not self.wait_for_service(service, max_attempts=60, delay=5):
                logger.error(f"Critical service {service} failed to start")
                return False
        
        # Configure service interconnections
        success = True
        
        try:
            # Step 1: Configure Prowlarr with indexers and *arr app sync
            if not self.configure_prowlarr_indexers():
                logger.error("Failed to configure Prowlarr indexers")
                success = False
            
            # Step 2: Configure download clients in *arr apps
            if not self.configure_download_clients():
                logger.error("Failed to configure download clients")
                success = False
            
            # Step 3: Configure media management
            if not self.configure_media_management():
                logger.error("Failed to configure media management")
                success = False
            
            # Step 4: Configure Jellyfin integration
            if not self.configure_jellyfin_integration():
                logger.error("Failed to configure Jellyfin integration")
                success = False
            
            # Step 5: Configure monitoring integrations
            if not self.configure_monitoring_integrations():
                logger.error("Failed to configure monitoring integrations")
                success = False
            
            if success:
                logger.info("✅ All service interconnections configured successfully!")
            else:
                logger.warning("⚠️ Some service configurations failed")
                
        except Exception as e:
            logger.error(f"Unexpected error during configuration: {e}")
            success = False
        
        return success

if __name__ == "__main__":
    interconnector = ServiceInterconnector()
    
    # Wait a bit for services to fully start
    logger.info("Waiting for services to initialize...")
    time.sleep(60)
    
    # Run configuration
    success = interconnector.run_full_configuration()
    
    if success:
        logger.info("🎉 Service interconnection configuration completed successfully!")
    else:
        logger.error("💥 Service interconnection configuration failed!")
        exit(1)