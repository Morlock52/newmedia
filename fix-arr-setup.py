#!/usr/bin/env python3
"""
ARR Services Setup Fix Script
Fixes the issues found during integration and prepares the environment
"""

import requests
import json
import time
import sys
from typing import Dict, List, Optional

# Service configuration with corrected settings
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

class ARRSetupFixer:
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
    
    def check_qbittorrent_status(self) -> bool:
        """Check if qBittorrent is accessible through gluetun"""
        print("🔍 Checking qBittorrent accessibility...")
        try:
            # Try to access qBittorrent directly
            response = self.session.get("http://localhost:8080", timeout=5)
            if response.status_code == 200:
                print("✅ qBittorrent is accessible on localhost:8080")
                return True
            else:
                print(f"⚠️  qBittorrent returned HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ qBittorrent is not accessible: {e}")
            return False
    
    def get_quality_profiles(self, service_name: str) -> List[Dict]:
        """Get quality profiles for a service"""
        config = SERVICES[service_name]
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/qualityprofile",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            print(f"❌ Failed to get quality profiles for {service_name}: {e}")
            return []
    
    def get_metadata_profiles(self, service_name: str) -> List[Dict]:
        """Get metadata profiles for Lidarr"""
        if service_name != 'lidarr':
            return []
            
        config = SERVICES[service_name]
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/metadataprofile",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            print(f"❌ Failed to get metadata profiles for {service_name}: {e}")
            return []
    
    def add_application_to_prowlarr_fixed(self, service_name: str) -> bool:
        """Add ARR application to Prowlarr with corrected configuration"""
        config = SERVICES[service_name]
        prowlarr_config = SERVICES['prowlarr']
        
        # Check if application already exists
        try:
            headers = {'X-Api-Key': prowlarr_config['api_key']}
            response = self.session.get(
                f"{prowlarr_config['url']}/api/v1/applications",
                headers=headers
            )
            existing_apps = response.json() if response.status_code == 200 else []
            
            for app in existing_apps:
                if app.get('name', '').lower() == service_name:
                    print(f"⚠️  {service_name.title()} already exists in Prowlarr")
                    return True
        except Exception as e:
            print(f"❌ Error checking existing applications: {e}")
        
        # Application configuration with corrected fields
        app_configs = {
            'sonarr': {
                'implementation': 'Sonarr',
                'configContract': 'SonarrSettings',
                'syncCategories': [5000, 5010, 5020, 5030, 5040, 5045, 5080]
            },
            'radarr': {
                'implementation': 'Radarr',
                'configContract': 'RadarrSettings',
                'syncCategories': [2000, 2010, 2020, 2030, 2040, 2045, 2050, 2060, 2070, 2080]
            },
            'lidarr': {
                'implementation': 'Lidarr',
                'configContract': 'LidarrSettings',
                'syncCategories': [3000, 3010, 3020, 3030, 3040]
            }
        }
        
        app_config = app_configs.get(service_name)
        if not app_config:
            print(f"❌ Unknown service: {service_name}")
            return False
        
        # Corrected payload with prowlarrUrl field
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
                    "name": "prowlarrUrl",
                    "value": "http://prowlarr:9696"
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
    
    def add_qbittorrent_client_fixed(self, service_name: str) -> bool:
        """Add qBittorrent as download client with localhost instead of gluetun"""
        if service_name == 'prowlarr':
            return True
            
        config = SERVICES[service_name]
        
        # Check if qBittorrent already exists
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/downloadclient",
                headers=headers
            )
            existing_clients = response.json() if response.status_code == 200 else []
            
            for client in existing_clients:
                if client.get('name', '').lower() == 'qbittorrent':
                    print(f"⚠️  qBittorrent already configured for {service_name.title()}")
                    return True
        except Exception as e:
            print(f"❌ Error checking existing download clients: {e}")
        
        # qBittorrent configuration using localhost instead of gluetun
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
                {"name": "host", "value": "localhost"},
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
    
    def add_root_folder_fixed(self, service_name: str) -> bool:
        """Add root folder with proper configuration"""
        if service_name == 'prowlarr':
            return True
            
        config = SERVICES[service_name]
        media_path = config['media_path']
        
        # Get quality profiles
        quality_profiles = self.get_quality_profiles(service_name)
        if not quality_profiles:
            print(f"❌ No quality profiles found for {service_name.title()}")
            return False
        
        default_quality_profile_id = quality_profiles[0]['id']
        
        # For Lidarr, also get metadata profiles
        default_metadata_profile_id = None
        if service_name == 'lidarr':
            metadata_profiles = self.get_metadata_profiles(service_name)
            if metadata_profiles:
                default_metadata_profile_id = metadata_profiles[0]['id']
            else:
                print(f"❌ No metadata profiles found for Lidarr")
                return False
        
        # Check if root folder already exists
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/rootfolder",
                headers=headers
            )
            existing_folders = response.json() if response.status_code == 200 else []
            
            for folder in existing_folders:
                if folder.get('path') == media_path:
                    print(f"⚠️  Root folder {media_path} already exists for {service_name.title()}")
                    return True
        except Exception as e:
            print(f"❌ Error checking existing root folders: {e}")
        
        # Build payload based on service type
        if service_name == 'lidarr':
            payload = {
                "path": media_path,
                "name": f"{service_name.title()} Media",
                "defaultMetadataProfileId": default_metadata_profile_id,
                "defaultQualityProfileId": default_quality_profile_id
            }
        else:
            payload = {
                "path": media_path,
                "defaultQualityProfileId": default_quality_profile_id
            }
        
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
    
    def run_fixed_integration(self) -> bool:
        """Run the fixed integration process"""
        print("🔧 Starting ARR Services Setup Fix")
        print("=" * 50)
        
        # Step 1: Test connectivity
        if not self.test_connectivity():
            print("❌ Some services are not accessible. Please check your setup.")
            return False
        
        # Step 2: Check qBittorrent
        qbt_status = self.check_qbittorrent_status()
        if not qbt_status:
            print("⚠️  qBittorrent may not be accessible - will try localhost connection")
        
        print("\n📡 Step 1: Adding ARR applications to Prowlarr (Fixed)...")
        arr_services = ['sonarr', 'radarr', 'lidarr']
        
        for service in arr_services:
            self.add_application_to_prowlarr_fixed(service)
        
        print("\n⬇️  Step 2: Configuring download clients (Fixed)...")
        for service in arr_services:
            self.add_qbittorrent_client_fixed(service)
        
        print("\n📁 Step 3: Setting up root folders (Fixed)...")
        for service in arr_services:
            self.add_root_folder_fixed(service)
        
        print("\n🔄 Step 4: Triggering sync...")
        try:
            headers = {'X-Api-Key': SERVICES['prowlarr']['api_key']}
            response = self.session.post(
                f"{SERVICES['prowlarr']['url']}/api/v1/command",
                headers=headers,
                json={"name": "ApplicationSync"}
            )
            
            if response.status_code in [200, 201]:
                print("✅ Triggered Prowlarr sync")
            else:
                print(f"⚠️  Sync trigger returned HTTP {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not trigger sync: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 ARR Services Setup Fix Complete!")
        
        print("\n🔗 Service URLs:")
        for service_name, config in SERVICES.items():
            print(f"   - {service_name.title()}: {config['url']}")
        
        print("\n📝 Manual Steps Required:")
        print("1. 📁 Create media directories in Docker volumes:")
        print("   - Access containers and create: /media/tv, /media/movies, /media/music")
        print("2. 🔍 Add indexers to Prowlarr (Settings → Indexers)")
        print("3. ⚙️  Configure quality profiles if needed")
        print("4. 🔗 Ensure container networking allows internal communication")
        
        return True

def main():
    """Main function"""
    fixer = ARRSetupFixer()
    
    try:
        success = fixer.run_fixed_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()