#!/usr/bin/env python3
"""
Media Server Library Integration Test
Tests Jellyfin, Plex, and Emby library scanning and ARR integration
"""

import requests
import json
from datetime import datetime

class MediaServerTester:
    def __init__(self, host='localhost'):
        self.host = host
        self.media_paths = {
            'movies': '/media/movies',
            'tv': '/media/tv', 
            'music': '/media/music',
            'books': '/media/books',
            'audiobooks': '/media/audiobooks'
        }
        
    def test_jellyfin(self):
        """Test Jellyfin connectivity and library configuration"""
        print("\n=== Testing Jellyfin ===")
        port = 8096
        base_url = f"http://{self.host}:{port}"
        
        # Test basic connectivity
        try:
            response = requests.get(f"{base_url}/System/Info/Public", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"✓ Jellyfin is accessible (v{info.get('Version', 'Unknown')})")
                print(f"  Server Name: {info.get('ServerName', 'Unknown')}")
                print(f"  Operating System: {info.get('OperatingSystem', 'Unknown')}")
                
                # Configuration guide
                print("\n📚 Jellyfin Library Configuration:")
                print("1. Access Jellyfin at http://localhost:8096")
                print("2. Complete initial setup wizard if needed")
                print("3. Go to Dashboard → Libraries → Add Media Library")
                print("4. Add the following libraries:")
                print("   - Movies: /media/movies")
                print("   - TV Shows: /media/tv")
                print("   - Music: /media/music")
                print("   - Books: /media/books")
                print("5. For each library:")
                print("   - Enable 'Real-time monitoring'")
                print("   - Configure metadata providers")
                print("   - Set preferred language")
                print("6. Go to Dashboard → Scheduled Tasks")
                print("   - Configure 'Scan Media Library' schedule")
                
                print("\n🔗 ARR Integration:")
                print("- Jellyfin will automatically detect new media")
                print("- Use Jellyseerr for request management")
                print("- Configure webhooks in ARR services for notifications")
                
            else:
                print(f"✗ Jellyfin returned status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to connect to Jellyfin: {e}")
            
    def test_plex(self):
        """Test Plex connectivity and library configuration"""
        print("\n=== Testing Plex ===")
        port = 32400
        base_url = f"http://{self.host}:{port}"
        
        # Test basic connectivity
        try:
            response = requests.get(f"{base_url}/identity", timeout=5)
            if response.status_code == 200:
                print("✓ Plex is accessible")
                
                # Try to get more info without token
                try:
                    # Public endpoints
                    prefs_response = requests.get(f"{base_url}/:/prefs", timeout=5)
                    if prefs_response.status_code == 401:
                        print("  Note: Plex requires authentication token for API access")
                except:
                    pass
                
                # Configuration guide
                print("\n📚 Plex Library Configuration:")
                print("1. Access Plex at http://localhost:32400/web")
                print("2. Sign in with your Plex account")
                print("3. Click '+' next to Libraries → Add Library")
                print("4. Add the following libraries:")
                print("   - Movies: Type 'Movies', Path '/media/movies'")
                print("   - TV Shows: Type 'TV Shows', Path '/media/tv'")
                print("   - Music: Type 'Music', Path '/media/music'")
                print("   - Photos: Type 'Photos', Path '/media/photos'")
                print("5. For each library:")
                print("   - Enable 'Scan my library automatically'")
                print("   - Enable 'Run a partial scan when changes detected'")
                print("   - Configure agent (Plex Movie, TheTVDB, etc.)")
                print("6. Go to Settings → Library")
                print("   - Set scan interval")
                print("   - Configure thumbnail generation")
                
                print("\n🔗 ARR Integration:")
                print("- Use Overseerr for request management")
                print("- Configure Plex webhook in ARR services:")
                print("  Sonarr/Radarr → Settings → Connect → Add → Plex")
                print("- Get Plex token from: Settings → Authorized Devices")
                
            else:
                print(f"✗ Plex returned status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to connect to Plex: {e}")
            
    def test_emby(self):
        """Test Emby connectivity and library configuration"""
        print("\n=== Testing Emby ===")
        port = 8097
        base_url = f"http://{self.host}:{port}"
        
        # Test basic connectivity
        try:
            response = requests.get(f"{base_url}/emby/System/Info/Public", timeout=5)
            if response.status_code == 200:
                info = response.json()
                print(f"✓ Emby is accessible (v{info.get('Version', 'Unknown')})")
                print(f"  Server Name: {info.get('ServerName', 'Unknown')}")
                
                # Configuration guide
                print("\n📚 Emby Library Configuration:")
                print("1. Access Emby at http://localhost:8097")
                print("2. Complete initial setup if needed")
                print("3. Go to Settings → Library → Add Media Library")
                print("4. Add the following libraries:")
                print("   - Movies: /media/movies")
                print("   - TV Shows: /media/tv")
                print("   - Music: /media/music")
                print("5. For each library:")
                print("   - Enable real-time monitoring")
                print("   - Configure metadata downloaders")
                print("   - Set content type and display preferences")
                print("6. Go to Settings → Scheduled Tasks")
                print("   - Configure library scan schedule")
                
                print("\n🔗 ARR Integration:")
                print("- Configure Emby webhook in ARR services")
                print("- Use Ombi for request management")
                print("- Get API key from Settings → API Keys")
                
            else:
                print(f"✗ Emby returned status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to connect to Emby: {e}")
            
    def test_arr_integration(self):
        """Provide ARR to Media Server integration guide"""
        print("\n=== ARR to Media Server Integration ===")
        
        print("\n📋 Sonarr/Radarr → Media Server Setup:")
        print("\n1. JELLYFIN Integration:")
        print("   - In Sonarr/Radarr → Settings → Connect → Add → Jellyfin")
        print("   - Host: jellyfin")
        print("   - Port: 8096")
        print("   - API Key: Get from Jellyfin → Dashboard → API Keys")
        print("   - Update Library: ON")
        print("   - Test and Save")
        
        print("\n2. PLEX Integration:")
        print("   - In Sonarr/Radarr → Settings → Connect → Add → Plex Media Server")
        print("   - Host: plex")
        print("   - Port: 32400")
        print("   - Auth Token: Get from plex.tv/devices.xml")
        print("   - Update Library: ON")
        print("   - Test and Save")
        
        print("\n3. EMBY Integration:")
        print("   - In Sonarr/Radarr → Settings → Connect → Add → Emby")
        print("   - Host: emby")
        print("   - Port: 8096")
        print("   - API Key: Get from Emby → Settings → API Keys")
        print("   - Update Library: ON")
        print("   - Test and Save")
        
        print("\n🔄 Media Flow:")
        print("1. User requests media (via Jellyseerr/Overseerr/Ombi)")
        print("2. Request approved → Sonarr/Radarr searches via Prowlarr")
        print("3. Download starts in qBittorrent/SABnzbd")
        print("4. ARR service monitors download progress")
        print("5. On completion: ARR imports to media folder")
        print("6. ARR triggers media server library update")
        print("7. Media appears in Jellyfin/Plex/Emby")
        
    def generate_webhook_config(self):
        """Generate webhook configuration examples"""
        print("\n=== Webhook Configuration Examples ===")
        
        webhook_config = """
# Jellyfin Webhook for Sonarr/Radarr
{
  "name": "Jellyfin",
  "on_grab": false,
  "on_download": true,
  "on_upgrade": true,
  "on_rename": true,
  "on_delete": true,
  "host": "jellyfin",
  "port": 8096,
  "api_key": "YOUR_JELLYFIN_API_KEY",
  "update_library": true,
  "clean_library": true
}

# Plex Webhook URL Format
http://plex:32400/library/sections/{section_id}/refresh?X-Plex-Token={token}

# Custom Webhook for notifications
{
  "webhook_url": "http://your-webhook/notify",
  "webhook_method": "POST",
  "webhook_headers": {
    "Content-Type": "application/json"
  },
  "webhook_body": {
    "event": "{event_type}",
    "series": "{series_title}",
    "episode": "{episode_title}",
    "quality": "{quality}"
  }
}
"""
        
        with open('/Users/morlock/fun/newmedia/TEST_REPORTS/webhook-examples.json', 'w') as f:
            f.write(webhook_config)
        
        print("✓ Webhook examples saved to: webhook-examples.json")
        
    def create_library_structure(self):
        """Create recommended folder structure"""
        print("\n=== Recommended Media Folder Structure ===")
        
        structure = """
/media/
├── movies/
│   ├── Movie Name (Year)/
│   │   ├── Movie Name (Year).mkv
│   │   └── Movie Name (Year)-poster.jpg
├── tv/
│   ├── Show Name/
│   │   ├── Season 01/
│   │   │   ├── Show Name - S01E01 - Episode Title.mkv
│   │   │   └── Show Name - S01E02 - Episode Title.mkv
│   │   └── Season 02/
├── music/
│   ├── Artist Name/
│   │   ├── Album Name/
│   │   │   ├── 01 - Track Name.mp3
│   │   │   └── 02 - Track Name.mp3
├── books/
│   ├── Author Name/
│   │   └── Book Title.epub
└── downloads/
    ├── complete/
    └── incomplete/
"""
        
        print(structure)
        
        # Create setup script
        setup_script = """#!/bin/bash
# Create media folder structure

echo "Creating media folder structure..."

# Create main directories
mkdir -p /media/{movies,tv,music,books,audiobooks,downloads/{complete,incomplete}}

# Set permissions (adjust PUID/PGID as needed)
chown -R 1000:1000 /media
chmod -R 755 /media

echo "✓ Media folder structure created"
echo ""
echo "Folder permissions:"
ls -la /media/
"""
        
        with open('/Users/morlock/fun/newmedia/TEST_REPORTS/create-media-folders.sh', 'w') as f:
            f.write(setup_script)
        
        print("\n✓ Folder creation script saved to: create-media-folders.sh")
        
    def run_all_tests(self):
        """Run all media server tests"""
        print("=== Media Server Integration Tests ===")
        print("Testing media servers for library configuration...")
        
        # Test each media server
        self.test_jellyfin()
        self.test_plex()
        self.test_emby()
        
        # Integration guide
        self.test_arr_integration()
        
        # Generate configuration files
        self.generate_webhook_config()
        self.create_library_structure()
        
        print("\n=== Quick Start Checklist ===")
        print("□ 1. Create folder structure with create-media-folders.sh")
        print("□ 2. Configure libraries in each media server")
        print("□ 3. Add media server connections in ARR services")
        print("□ 4. Configure request services (Jellyseerr/Overseerr)")
        print("□ 5. Test complete workflow: Request → Download → Import → Library")

if __name__ == "__main__":
    tester = MediaServerTester()
    tester.run_all_tests()