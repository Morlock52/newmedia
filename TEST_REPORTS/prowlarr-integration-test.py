#!/usr/bin/env python3
"""
Prowlarr Integration Test
Tests Prowlarr's ability to manage indexers and sync with ARR services
"""

import requests
import json
from datetime import datetime

class ProwlarrIntegrationTest:
    def __init__(self, host='localhost', port=9696):
        self.base_url = f"http://{host}:{port}"
        self.api_key = None  # Will be set if available
        
    def test_connectivity(self):
        """Test basic Prowlarr connectivity"""
        print("Testing Prowlarr connectivity...")
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("✓ Prowlarr web interface is accessible")
                return True
            else:
                print(f"✗ Prowlarr returned status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Failed to connect to Prowlarr: {e}")
            return False
    
    def get_config_instructions(self):
        """Provide configuration instructions"""
        print("\n=== Prowlarr Configuration Guide ===\n")
        
        print("1. ACCESS PROWLARR:")
        print(f"   - Open: {self.base_url}")
        print("   - Complete initial setup if needed")
        print()
        
        print("2. GET API KEY:")
        print("   - Go to: Settings → General → Security")
        print("   - Copy the API Key")
        print("   - Save it in your .env file as PROWLARR_API_KEY")
        print()
        
        print("3. ADD INDEXERS:")
        print("   - Go to: Indexers → Add Indexer")
        print("   - Popular free indexers:")
        print("     • 1337x (Public torrent tracker)")
        print("     • RARBG (Public torrent tracker)")
        print("     • The Pirate Bay (Public torrent tracker)")
        print("   - For Usenet: Add your provider (e.g., NZBgeek, NZBFinder)")
        print()
        
        print("4. CONFIGURE APPLICATIONS (ARR Services):")
        print("   - Go to: Settings → Apps → Add")
        print("   - Add each ARR service:")
        print()
        
        # Sonarr configuration
        print("   SONARR:")
        print("   - Sync Level: Full Sync")
        print("   - Prowlarr Server: http://prowlarr:9696")
        print("   - Sonarr Server: http://sonarr:8989")
        print("   - API Key: [Get from Sonarr → Settings → General]")
        print()
        
        # Radarr configuration
        print("   RADARR:")
        print("   - Sync Level: Full Sync")
        print("   - Prowlarr Server: http://prowlarr:9696")
        print("   - Radarr Server: http://radarr:7878")
        print("   - API Key: [Get from Radarr → Settings → General]")
        print()
        
        # Lidarr configuration
        print("   LIDARR:")
        print("   - Sync Level: Full Sync")
        print("   - Prowlarr Server: http://prowlarr:9696")
        print("   - Lidarr Server: http://lidarr:8686")
        print("   - API Key: [Get from Lidarr → Settings → General]")
        print()
        
        # Readarr configuration
        print("   READARR:")
        print("   - Sync Level: Full Sync")
        print("   - Prowlarr Server: http://prowlarr:9696")
        print("   - Readarr Server: http://readarr:8787")
        print("   - API Key: [Get from Readarr → Settings → General]")
        print()
        
        print("5. TEST SYNC:")
        print("   - After adding apps, click 'Test' for each")
        print("   - Click 'Sync App Indexers' to push indexers to ARR services")
        print("   - Check each ARR service to verify indexers appear")
        print()
    
    def test_with_api_key(self, api_key):
        """Test Prowlarr API functionality with API key"""
        self.api_key = api_key
        headers = {'X-Api-Key': api_key}
        
        print("\n=== Testing Prowlarr API ===\n")
        
        # Test system status
        try:
            response = requests.get(f"{self.base_url}/api/v1/system/status", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("✓ API Key is valid")
                print(f"  - Version: {data.get('version', 'unknown')}")
                print(f"  - Branch: {data.get('branch', 'unknown')}")
            else:
                print(f"✗ API request failed: {response.status_code}")
                return
        except Exception as e:
            print(f"✗ API error: {e}")
            return
        
        # Get indexers
        try:
            response = requests.get(f"{self.base_url}/api/v1/indexer", headers=headers)
            if response.status_code == 200:
                indexers = response.json()
                print(f"\n✓ Found {len(indexers)} configured indexers:")
                for idx in indexers:
                    status = "Enabled" if idx.get('enable', False) else "Disabled"
                    print(f"  - {idx['name']} ({idx['protocol']}) - {status}")
            else:
                print("✗ Failed to get indexers")
        except Exception as e:
            print(f"✗ Error getting indexers: {e}")
        
        # Get applications (connected ARR services)
        try:
            response = requests.get(f"{self.base_url}/api/v1/applications", headers=headers)
            if response.status_code == 200:
                apps = response.json()
                print(f"\n✓ Found {len(apps)} connected applications:")
                for app in apps:
                    sync_level = app.get('syncLevel', 'unknown')
                    print(f"  - {app['name']} (Sync: {sync_level})")
            else:
                print("✗ Failed to get applications")
        except Exception as e:
            print(f"✗ Error getting applications: {e}")
    
    def generate_setup_script(self):
        """Generate a setup script for common indexers"""
        script_content = """#!/bin/bash
# Prowlarr Quick Setup Script
# This script provides commands to quickly configure Prowlarr

PROWLARR_URL="http://localhost:9696"
PROWLARR_API_KEY="YOUR_API_KEY_HERE"  # Replace with your actual API key

echo "=== Prowlarr Quick Setup ==="
echo "Make sure to replace YOUR_API_KEY_HERE with your actual Prowlarr API key"
echo ""

# Function to add an indexer
add_indexer() {
    local name=$1
    local implementation=$2
    local config=$3
    
    echo "Adding indexer: $name"
    curl -X POST "$PROWLARR_URL/api/v1/indexer" \\
        -H "X-Api-Key: $PROWLARR_API_KEY" \\
        -H "Content-Type: application/json" \\
        -d '{
            "name": "'$name'",
            "implementation": "'$implementation'",
            "implementationName": "'$implementation'",
            "configContract": "'$implementation'Config",
            "fields": '$config',
            "enable": true,
            "protocol": "torrent",
            "priority": 25
        }'
}

# Example: Add 1337x indexer
# add_indexer "1337x" "Cardigann" '[{"name":"definitionFile","value":"1337x"}]'

echo ""
echo "To add indexers manually:"
echo "1. Get your API key from Prowlarr Settings"
echo "2. Update PROWLARR_API_KEY in this script"
echo "3. Uncomment the indexer lines you want to add"
echo "4. Run this script"
"""
        
        with open('/Users/morlock/fun/newmedia/TEST_REPORTS/prowlarr-setup.sh', 'w') as f:
            f.write(script_content)
        
        print("\n✓ Setup script created: prowlarr-setup.sh")

if __name__ == "__main__":
    tester = ProwlarrIntegrationTest()
    
    # Test basic connectivity
    if tester.test_connectivity():
        # Provide configuration instructions
        tester.get_config_instructions()
        
        # Generate setup script
        tester.generate_setup_script()
        
        print("\n=== API Testing ===")
        print("To test API functionality, run:")
        print("python3 prowlarr-integration-test.py --api-key YOUR_API_KEY")
        
        # Check if API key was provided as argument
        import sys
        if len(sys.argv) > 2 and sys.argv[1] == '--api-key':
            tester.test_with_api_key(sys.argv[2])
    else:
        print("\n⚠️  Make sure Prowlarr is running:")
        print("docker-compose up -d prowlarr")