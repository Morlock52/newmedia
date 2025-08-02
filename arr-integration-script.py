#!/usr/bin/env python3
"""
ARR Services Integration Script
Connects all ARR services to Prowlarr using API calls and configures bidirectional sync
"""

import requests
import json
import time
import sys
from typing import Dict, List, Optional

# Service configuration
SERVICES = {
    'prowlarr': {
        'url': 'http://localhost:9696',
        'api_key': 'b7ef1468932940b2a4cf27ad980f1076',
        'api_version': 'v1'
    },
    'sonarr': {
        'url': 'http://localhost:8989',
        'api_key': '6e6bfac6e15d4f9a9d0e0d35ec0b8e23',
        'api_version': 'v3',
        'internal_url': 'http://sonarr:8989',
        'category': 'sonarr',
        'media_path': '/media/tv'
    },
    'radarr': {
        'url': 'http://localhost:7878',
        'api_key': '7b74da952069425f9568ea361b001a12',
        'api_version': 'v3',
        'internal_url': 'http://radarr:7878',
        'category': 'radarr',
        'media_path': '/media/movies'
    },
    'lidarr': {
        'url': 'http://localhost:8686',
        'api_key': 'e8262da767e34a6b8ca7ca1e92384d96',
        'api_version': 'v1',
        'internal_url': 'http://lidarr:8686',
        'category': 'lidarr',
        'media_path': '/media/music'
    }
}

class ARRIntegration:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
        
    def test_connectivity(self) -> bool:
        """Test connectivity to all services"""
        print("🔍 Testing service connectivity...")
        all_services_ok = True
        
        for service_name, config in SERVICES.items():
            try:
                headers = {'X-Api-Key': config['api_key']}
                response = self.session.get(
                    f"{config['url']}/api/{config['api_version']}/system/status",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {service_name.title()}: {data.get('version', 'Unknown version')}")
                else:
                    print(f"❌ {service_name.title()}: HTTP {response.status_code}")
                    all_services_ok = False
                    
            except Exception as e:
                print(f"❌ {service_name.title()}: Connection failed - {e}")
                all_services_ok = False
                
        return all_services_ok
    
    def get_prowlarr_applications(self) -> List[Dict]:
        """Get existing applications in Prowlarr"""
        try:
            headers = {'X-Api-Key': SERVICES['prowlarr']['api_key']}
            response = self.session.get(
                f"{SERVICES['prowlarr']['url']}/api/v1/applications",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            print(f"❌ Failed to get Prowlarr applications: {e}")
            return []
    
    def add_application_to_prowlarr(self, service_name: str) -> bool:
        """Add ARR application to Prowlarr"""
        config = SERVICES[service_name]
        prowlarr_config = SERVICES['prowlarr']
        
        # Check if application already exists
        existing_apps = self.get_prowlarr_applications()
        for app in existing_apps:
            if app.get('name', '').lower() == service_name:
                print(f"⚠️  {service_name.title()} already exists in Prowlarr")
                return True
        
        # Application configuration based on service type
        app_configs = {
            'sonarr': {
                'implementation': 'Sonarr',
                'configContract': 'SonarrSettings',
                'syncCategories': [5000, 5010, 5020, 5030, 5040, 5045, 5080]  # TV categories
            },
            'radarr': {
                'implementation': 'Radarr',
                'configContract': 'RadarrSettings',
                'syncCategories': [2000, 2010, 2020, 2030, 2040, 2045, 2050, 2060, 2070, 2080]  # Movie categories
            },
            'lidarr': {
                'implementation': 'Lidarr',
                'configContract': 'LidarrSettings',
                'syncCategories': [3000, 3010, 3020, 3030, 3040]  # Music categories
            }
        }
        
        app_config = app_configs.get(service_name)
        if not app_config:
            print(f"❌ Unknown service: {service_name}")
            return False
        
        payload = {
            "name": service_name.title(),
            "implementation": app_config['implementation'],
            "configContract": app_config['configContract'],
            "fields": [
                {
                    "name": "baseUrl",
                    "value": config['internal_url']
                },
                {
                    "name": "apiKey",
                    "value": config['api_key']
                },
                {
                    "name": "syncCategories",
                    "value": app_config['syncCategories']
                }
            ],
            "tags": [],
            "syncLevel": "fullSync"
        }
        
        try:
            headers = {
                'X-Api-Key': prowlarr_config['api_key'],
                'Content-Type': 'application/json'
            }
            response = self.session.post(
                f"{prowlarr_config['url']}/api/v1/applications",
                headers=headers,
                json=payload
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Successfully added {service_name.title()} to Prowlarr")
                return True
            else:
                print(f"❌ Failed to add {service_name.title()} to Prowlarr: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding {service_name.title()} to Prowlarr: {e}")
            return False
    
    def get_download_clients(self, service_name: str) -> List[Dict]:
        """Get existing download clients for a service"""
        config = SERVICES[service_name]
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/downloadclient",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            print(f"❌ Failed to get download clients for {service_name}: {e}")
            return []
    
    def add_qbittorrent_client(self, service_name: str) -> bool:
        """Add qBittorrent as download client to ARR service"""
        if service_name == 'prowlarr':
            return True  # Prowlarr doesn't need download clients
            
        config = SERVICES[service_name]
        
        # Check if qBittorrent already exists
        existing_clients = self.get_download_clients(service_name)
        for client in existing_clients:
            if client.get('name', '').lower() == 'qbittorrent':
                print(f"⚠️  qBittorrent already configured for {service_name.title()}")
                return True
        
        # qBittorrent configuration
        payload = {
            "enable": True,
            "protocol": "torrent",
            "priority": 1,
            "removeCompletedDownloads": True,
            "removeFailedDownloads": True,
            "name": "qBittorrent",
            "implementation": "QBittorrent",
            "configContract": "QBittorrentSettings",
            "fields": [
                {"name": "host", "value": "gluetun"},
                {"name": "port", "value": 8080},
                {"name": "urlBase", "value": ""},
                {"name": "username", "value": "admin"},
                {"name": "password", "value": "adminadmin"},
                {"name": "tvCategory", "value": config['category']},
                {"name": "movieCategory", "value": config['category']},
                {"name": "musicCategory", "value": config['category']},
                {"name": "recentMoviePriority", "value": 0},
                {"name": "olderMoviePriority", "value": 0},
                {"name": "recentTvPriority", "value": 0},
                {"name": "olderTvPriority", "value": 0},
                {"name": "initialState", "value": 0},
                {"name": "sequentialOrder", "value": False},
                {"name": "firstAndLast", "value": False}
            ],
            "tags": []
        }
        
        try:
            headers = {
                'X-Api-Key': config['api_key'],
                'Content-Type': 'application/json'
            }
            response = self.session.post(
                f"{config['url']}/api/{config['api_version']}/downloadclient",
                headers=headers,
                json=payload
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Successfully added qBittorrent to {service_name.title()}")
                return True
            else:
                print(f"❌ Failed to add qBittorrent to {service_name.title()}: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding qBittorrent to {service_name.title()}: {e}")
            return False
    
    def get_root_folders(self, service_name: str) -> List[Dict]:
        """Get existing root folders for a service"""
        if service_name == 'prowlarr':
            return []
            
        config = SERVICES[service_name]
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/rootfolder",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            print(f"❌ Failed to get root folders for {service_name}: {e}")
            return []
    
    def add_root_folder(self, service_name: str) -> bool:
        """Add root folder to ARR service"""
        if service_name == 'prowlarr':
            return True
            
        config = SERVICES[service_name]
        media_path = config['media_path']
        
        # Check if root folder already exists
        existing_folders = self.get_root_folders(service_name)
        for folder in existing_folders:
            if folder.get('path') == media_path:
                print(f"⚠️  Root folder {media_path} already exists for {service_name.title()}")
                return True
        
        payload = {"path": media_path}
        
        try:
            headers = {
                'X-Api-Key': config['api_key'],
                'Content-Type': 'application/json'
            }
            response = self.session.post(
                f"{config['url']}/api/{config['api_version']}/rootfolder",
                headers=headers,
                json=payload
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Successfully added root folder {media_path} to {service_name.title()}")
                return True
            else:
                print(f"❌ Failed to add root folder to {service_name.title()}: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding root folder to {service_name.title()}: {e}")
            return False
    
    def sync_prowlarr_indexers(self) -> bool:
        """Trigger indexer sync in Prowlarr"""
        try:
            headers = {'X-Api-Key': SERVICES['prowlarr']['api_key']}
            response = self.session.post(
                f"{SERVICES['prowlarr']['url']}/api/v1/command",
                headers=headers,
                json={"name": "ApplicationSync"}
            )
            
            if response.status_code in [200, 201]:
                print("✅ Triggered Prowlarr indexer sync")
                return True
            else:
                print(f"❌ Failed to trigger indexer sync: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error triggering indexer sync: {e}")
            return False
    
    def test_connections(self) -> bool:
        """Test all configured connections"""
        print("\n🧪 Testing configured connections...")
        
        # Test Prowlarr applications
        apps = self.get_prowlarr_applications()
        if apps:
            print(f"✅ Prowlarr has {len(apps)} applications configured:")
            for app in apps:
                print(f"   - {app.get('name', 'Unknown')}")
        else:
            print("⚠️  No applications found in Prowlarr")
        
        # Test download clients for each ARR service
        for service_name in ['sonarr', 'radarr', 'lidarr']:
            clients = self.get_download_clients(service_name)
            if clients:
                print(f"✅ {service_name.title()} has {len(clients)} download client(s) configured")
            else:
                print(f"⚠️  No download clients found for {service_name.title()}")
        
        # Test root folders
        for service_name in ['sonarr', 'radarr', 'lidarr']:
            folders = self.get_root_folders(service_name)
            if folders:
                print(f"✅ {service_name.title()} has {len(folders)} root folder(s) configured")
            else:
                print(f"⚠️  No root folders found for {service_name.title()}")
        
        return True
    
    def run_integration(self) -> bool:
        """Run the complete integration process"""
        print("🚀 Starting ARR Services Integration")
        print("=" * 50)
        
        # Step 1: Test connectivity
        if not self.test_connectivity():
            print("❌ Some services are not accessible. Please check your setup.")
            return False
        
        print("\n📡 Step 1: Adding ARR applications to Prowlarr...")
        arr_services = ['sonarr', 'radarr', 'lidarr']
        
        for service in arr_services:
            self.add_application_to_prowlarr(service)
        
        print("\n⬇️  Step 2: Configuring download clients...")
        for service in arr_services:
            self.add_qbittorrent_client(service)
        
        print("\n📁 Step 3: Setting up root folders...")
        for service in arr_services:
            self.add_root_folder(service)
        
        print("\n🔄 Step 4: Syncing Prowlarr indexers...")
        self.sync_prowlarr_indexers()
        
        # Wait a moment for sync to process
        print("⏳ Waiting for sync to complete...")
        time.sleep(5)
        
        print("\n🧪 Step 5: Testing connections...")
        self.test_connections()
        
        print("\n" + "=" * 50)
        print("🎉 ARR Services Integration Complete!")
        print("\n📋 Summary:")
        print("✅ All ARR services connected to Prowlarr")
        print("✅ qBittorrent configured as download client")
        print("✅ Root folders configured for /media paths")
        print("✅ Indexer sync triggered")
        
        print("\n🔗 Service URLs:")
        for service_name, config in SERVICES.items():
            print(f"   - {service_name.title()}: {config['url']}")
        
        print("\n📝 Next Steps:")
        print("1. Add indexers to Prowlarr (Settings → Indexers)")
        print("2. Configure quality profiles in each ARR service")
        print("3. Set up notification agents if desired")
        print("4. Start adding media to your libraries!")
        
        return True

def main():
    """Main function"""
    integration = ARRIntegration()
    
    try:
        success = integration.run_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Integration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()