#!/usr/bin/env python3
"""
Download Client Integration Test
Tests qBittorrent, Transmission, SABnzbd, and NZBGet connectivity
"""

import requests
import json
import base64
from datetime import datetime

class DownloadClientTester:
    def __init__(self, host='localhost'):
        self.host = host
        self.results = {}
        
    def test_qbittorrent(self):
        """Test qBittorrent connectivity and API"""
        print("\n=== Testing qBittorrent ===")
        port = 8080
        base_url = f"http://{self.host}:{port}"
        
        result = {
            'accessible': False,
            'authenticated': False,
            'version': None,
            'config_guide': []
        }
        
        # Test basic connectivity
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                result['accessible'] = True
                print("✓ qBittorrent web interface is accessible")
            else:
                print(f"✗ qBittorrent returned status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to connect to qBittorrent: {e}")
            print("  Note: qBittorrent runs through Gluetun VPN container")
            
        if result['accessible']:
            # Try default authentication
            try:
                # Login with default credentials
                login_data = {
                    'username': 'admin',
                    'password': 'adminadmin'
                }
                session = requests.Session()
                login_response = session.post(f"{base_url}/api/v2/auth/login", data=login_data)
                
                if login_response.status_code == 200 and login_response.text == 'Ok.':
                    result['authenticated'] = True
                    print("✓ Authenticated with default credentials (admin/adminadmin)")
                    
                    # Get version
                    version_response = session.get(f"{base_url}/api/v2/app/version")
                    if version_response.status_code == 200:
                        result['version'] = version_response.text.strip()
                        print(f"✓ qBittorrent version: {result['version']}")
                    
                    # Get preferences
                    prefs_response = session.get(f"{base_url}/api/v2/app/preferences")
                    if prefs_response.status_code == 200:
                        prefs = prefs_response.json()
                        print(f"✓ Download directory: {prefs.get('save_path', 'Not set')}")
                else:
                    print("✗ Default credentials failed")
                    print("  Change password in qBittorrent settings")
                    
            except Exception as e:
                print(f"✗ Authentication error: {e}")
        
        # Configuration guide
        result['config_guide'] = [
            "\nqBittorrent Configuration for ARR Services:",
            "1. In Sonarr/Radarr → Settings → Download Clients → Add → qBittorrent",
            "2. Host: gluetun (NOT localhost or qbittorrent)",
            "3. Port: 8080",
            "4. Username: admin",
            "5. Password: adminadmin (or your changed password)",
            "6. Category: tv (for Sonarr) or movies (for Radarr)",
            "7. Test and Save",
            "",
            "Important: qBittorrent runs through VPN, so use 'gluetun' as host"
        ]
        
        for line in result['config_guide']:
            print(line)
            
        self.results['qbittorrent'] = result
        
    def test_transmission(self):
        """Test Transmission connectivity and RPC"""
        print("\n=== Testing Transmission ===")
        port = 9091
        base_url = f"http://{self.host}:{port}"
        
        result = {
            'accessible': False,
            'rpc_accessible': False,
            'version': None,
            'config_guide': []
        }
        
        # Test basic connectivity
        try:
            response = requests.get(f"{base_url}/transmission/web/", timeout=5)
            if response.status_code == 200:
                result['accessible'] = True
                print("✓ Transmission web interface is accessible")
            elif response.status_code == 401:
                result['accessible'] = True
                print("✓ Transmission is accessible (authentication required)")
            else:
                print(f"✗ Transmission returned status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to connect to Transmission: {e}")
            print("  Note: Transmission runs through Gluetun VPN container")
            
        if result['accessible']:
            # Test RPC interface
            try:
                # Get session ID
                session_response = requests.get(f"{base_url}/transmission/rpc", timeout=5)
                if 'X-Transmission-Session-Id' in session_response.headers:
                    session_id = session_response.headers['X-Transmission-Session-Id']
                    
                    # Test RPC with session ID
                    headers = {'X-Transmission-Session-Id': session_id}
                    rpc_data = {
                        "method": "session-get",
                        "arguments": {}
                    }
                    
                    rpc_response = requests.post(
                        f"{base_url}/transmission/rpc",
                        json=rpc_data,
                        headers=headers
                    )
                    
                    if rpc_response.status_code == 200:
                        result['rpc_accessible'] = True
                        data = rpc_response.json()
                        if 'arguments' in data:
                            result['version'] = data['arguments'].get('version', 'Unknown')
                            print(f"✓ Transmission RPC accessible (v{result['version']})")
                            print(f"✓ Download dir: {data['arguments'].get('download-dir', 'Not set')}")
                        
            except Exception as e:
                print(f"✗ RPC error: {e}")
        
        # Configuration guide
        result['config_guide'] = [
            "\nTransmission Configuration for ARR Services:",
            "1. In Sonarr/Radarr → Settings → Download Clients → Add → Transmission",
            "2. Host: gluetun (NOT localhost or transmission)",
            "3. Port: 9091",
            "4. Username/Password: Leave blank or set in Transmission",
            "5. Category: tv (for Sonarr) or movies (for Radarr)",
            "6. Test and Save",
            "",
            "Important: Transmission runs through VPN, so use 'gluetun' as host"
        ]
        
        for line in result['config_guide']:
            print(line)
            
        self.results['transmission'] = result
        
    def test_sabnzbd(self):
        """Test SABnzbd connectivity and API"""
        print("\n=== Testing SABnzbd ===")
        port = 8081
        base_url = f"http://{self.host}:{port}"
        
        result = {
            'accessible': False,
            'api_accessible': False,
            'version': None,
            'api_key_needed': True,
            'config_guide': []
        }
        
        # Test basic connectivity
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                result['accessible'] = True
                print("✓ SABnzbd web interface is accessible")
            else:
                print(f"✗ SABnzbd returned status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to connect to SABnzbd: {e}")
            
        if result['accessible']:
            # Test API (without key first)
            try:
                response = requests.get(f"{base_url}/sabnzbd/api?mode=version&output=json")
                if response.status_code == 200:
                    data = response.json()
                    if 'version' in data:
                        result['api_accessible'] = True
                        result['api_key_needed'] = False
                        result['version'] = data['version']
                        print(f"✓ SABnzbd API accessible without key (v{result['version']})")
                    elif 'error' in data:
                        print("! SABnzbd API requires authentication")
                        print("  Get API key from SABnzbd → Config → General → Security")
                        
            except Exception as e:
                print(f"✗ API error: {e}")
        
        # Configuration guide
        result['config_guide'] = [
            "\nSABnzbd Configuration for ARR Services:",
            "1. Get SABnzbd API Key:",
            "   - Open SABnzbd web interface",
            "   - Go to Config → General → Security",
            "   - Copy the API Key",
            "2. In Sonarr/Radarr → Settings → Download Clients → Add → SABnzbd",
            "3. Host: sabnzbd",
            "4. Port: 8080",
            "5. API Key: [paste SABnzbd API key]",
            "6. Category: tv (for Sonarr) or movies (for Radarr)",
            "7. Test and Save"
        ]
        
        for line in result['config_guide']:
            print(line)
            
        self.results['sabnzbd'] = result
        
    def test_nzbget(self):
        """Test NZBGet connectivity and API"""
        print("\n=== Testing NZBGet ===")
        port = 6789
        base_url = f"http://{self.host}:{port}"
        
        result = {
            'accessible': False,
            'api_accessible': False,
            'version': None,
            'config_guide': []
        }
        
        # Test basic connectivity
        try:
            response = requests.get(f"{base_url}/", timeout=5)
            if response.status_code == 200:
                result['accessible'] = True
                print("✓ NZBGet web interface is accessible")
            else:
                print(f"✗ NZBGet returned status code: {response.status_code}")
        except Exception as e:
            print(f"✗ Failed to connect to NZBGet: {e}")
            
        if result['accessible']:
            # Test JSON-RPC API
            try:
                # Default credentials
                auth = base64.b64encode(b'nzbget:tegbzn6789').decode('ascii')
                headers = {
                    'Authorization': f'Basic {auth}',
                    'Content-Type': 'application/json'
                }
                
                rpc_data = {
                    "method": "version",
                    "params": [],
                    "jsonrpc": "2.0",
                    "id": 1
                }
                
                response = requests.post(
                    f"{base_url}/jsonrpc",
                    json=rpc_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'result' in data:
                        result['api_accessible'] = True
                        result['version'] = data['result']
                        print(f"✓ NZBGet JSON-RPC accessible (v{result['version']})")
                        
            except Exception as e:
                print(f"✗ API error: {e}")
        
        # Configuration guide
        result['config_guide'] = [
            "\nNZBGet Configuration for ARR Services:",
            "1. Default credentials: nzbget/tegbzn6789",
            "2. In Sonarr/Radarr → Settings → Download Clients → Add → NZBGet",
            "3. Host: nzbget",
            "4. Port: 6789",
            "5. Username: nzbget",
            "6. Password: tegbzn6789 (or your changed password)",
            "7. Category: tv (for Sonarr) or movies (for Radarr)",
            "8. Test and Save"
        ]
        
        for line in result['config_guide']:
            print(line)
            
        self.results['nzbget'] = result
        
    def generate_arr_config_script(self):
        """Generate script to configure download clients in ARR services"""
        script_content = """#!/bin/bash
# ARR Services Download Client Configuration Helper

echo "=== ARR Services Download Client Configuration ==="
echo ""
echo "This script will help you configure download clients in ARR services"
echo ""

# Function to add download client to Sonarr/Radarr via API
configure_download_client() {
    local arr_name=$1
    local arr_port=$2
    local arr_api_key=$3
    local client_type=$4
    local client_host=$5
    local client_port=$6
    
    echo "Configuring $client_type in $arr_name..."
    
    # This is an example - actual API calls would go here
    echo "Steps:"
    echo "1. Open $arr_name at http://localhost:$arr_port"
    echo "2. Go to Settings → Download Clients → Add → $client_type"
    echo "3. Configure with:"
    echo "   - Host: $client_host"
    echo "   - Port: $client_port"
    echo ""
}

# Example configurations
echo "=== qBittorrent Configuration ==="
echo "For Sonarr:"
configure_download_client "Sonarr" "8989" "YOUR_API_KEY" "qBittorrent" "gluetun" "8080"

echo "For Radarr:"
configure_download_client "Radarr" "7878" "YOUR_API_KEY" "qBittorrent" "gluetun" "8080"

echo "=== SABnzbd Configuration ==="
echo "For Sonarr:"
configure_download_client "Sonarr" "8989" "YOUR_API_KEY" "SABnzbd" "sabnzbd" "8080"

echo "For Radarr:"
configure_download_client "Radarr" "7878" "YOUR_API_KEY" "SABnzbd" "sabnzbd" "8080"

echo ""
echo "Remember:"
echo "- VPN-routed clients (qBittorrent, Transmission) use 'gluetun' as host"
echo "- Direct clients (SABnzbd, NZBGet) use their container name as host"
echo "- Always test the connection before saving"
"""
        
        with open('/Users/morlock/fun/newmedia/TEST_REPORTS/configure-download-clients.sh', 'w') as f:
            f.write(script_content)
        
        print("\n✓ Configuration helper script created: configure-download-clients.sh")
        
    def run_all_tests(self):
        """Run tests for all download clients"""
        print("=== Download Client Integration Tests ===")
        print("Testing all download clients for ARR integration...")
        
        # Test each client
        self.test_qbittorrent()
        self.test_transmission()
        self.test_sabnzbd()
        self.test_nzbget()
        
        # Generate configuration script
        self.generate_arr_config_script()
        
        # Summary
        print("\n=== Summary ===")
        print("\nTorrent Clients (VPN-routed through Gluetun):")
        print("- qBittorrent: Use host 'gluetun' in ARR services")
        print("- Transmission: Use host 'gluetun' in ARR services")
        print("\nUsenet Clients (Direct access):")
        print("- SABnzbd: Use host 'sabnzbd' in ARR services")
        print("- NZBGet: Use host 'nzbget' in ARR services")
        print("\nIMPORTANT: Always use container names, not 'localhost', for inter-container communication")

if __name__ == "__main__":
    tester = DownloadClientTester()
    tester.run_all_tests()