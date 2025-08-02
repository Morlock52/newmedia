#!/usr/bin/env python3
"""
Media Server Integration Test Suite
Tests connectivity between ARR services, download clients, and media servers
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import urllib3

# Disable SSL warnings for local testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Service endpoints and ports
SERVICES = {
    # ARR Services
    'prowlarr': {'port': 9696, 'api_path': '/api/v1'},
    'sonarr': {'port': 8989, 'api_path': '/api/v3'},
    'radarr': {'port': 7878, 'api_path': '/api/v3'},
    'lidarr': {'port': 8686, 'api_path': '/api/v1'},
    'readarr': {'port': 8787, 'api_path': '/api/v1'},
    'bazarr': {'port': 6767, 'api_path': '/api'},
    
    # Download Clients
    'qbittorrent': {'port': 8080, 'api_path': '/api/v2'},
    'transmission': {'port': 9091, 'api_path': '/transmission/rpc'},
    'sabnzbd': {'port': 8081, 'api_path': '/sabnzbd/api'},
    'nzbget': {'port': 6789, 'api_path': '/jsonrpc'},
    
    # Media Servers
    'jellyfin': {'port': 8096, 'api_path': '/'},
    'plex': {'port': 32400, 'api_path': '/'},
    'emby': {'port': 8097, 'api_path': '/'}
}

class IntegrationTester:
    def __init__(self, host='localhost'):
        self.host = host
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'connectivity': {},
            'integrations': {},
            'api_keys': {},
            'errors': []
        }
        
    def log(self, message: str, level: str = 'INFO'):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
        
    def test_service_connectivity(self, service: str, port: int) -> bool:
        """Test if a service is accessible"""
        try:
            url = f"http://{self.host}:{port}"
            response = requests.get(url, timeout=5, verify=False)
            return response.status_code < 500
        except:
            return False
            
    def get_api_key_from_config(self, service: str) -> Optional[str]:
        """Try to get API key from service config (placeholder)"""
        # In a real environment, you would read from config files or environment
        # For testing, we'll return None and document how to obtain keys
        return None
        
    def test_prowlarr_connectivity(self) -> Dict:
        """Test Prowlarr API connectivity and indexer management"""
        self.log("Testing Prowlarr connectivity...")
        result = {
            'accessible': False,
            'api_accessible': False,
            'indexers': [],
            'apps_configured': [],
            'needs_config': []
        }
        
        # Test basic connectivity
        if self.test_service_connectivity('prowlarr', SERVICES['prowlarr']['port']):
            result['accessible'] = True
            self.log("✓ Prowlarr is accessible", "SUCCESS")
            
            # Test API (requires API key)
            api_key = self.get_api_key_from_config('prowlarr')
            if api_key:
                try:
                    headers = {'X-Api-Key': api_key}
                    
                    # Get indexers
                    url = f"http://{self.host}:{SERVICES['prowlarr']['port']}/api/v1/indexer"
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        result['api_accessible'] = True
                        indexers = response.json()
                        result['indexers'] = [{'name': idx['name'], 'enabled': idx['enable']} 
                                            for idx in indexers]
                        self.log(f"✓ Found {len(indexers)} indexers", "SUCCESS")
                    
                    # Get configured apps
                    url = f"http://{self.host}:{SERVICES['prowlarr']['port']}/api/v1/applications"
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        apps = response.json()
                        result['apps_configured'] = [app['name'] for app in apps]
                        self.log(f"✓ Found {len(apps)} configured apps", "SUCCESS")
                        
                except Exception as e:
                    self.log(f"✗ Prowlarr API error: {str(e)}", "ERROR")
            else:
                result['needs_config'].append('API key required - Get from Settings > General > Security')
                self.log("! Prowlarr API key needed for full testing", "WARNING")
        else:
            self.log("✗ Prowlarr is not accessible", "ERROR")
            
        return result
        
    def test_download_client_connectivity(self, client: str) -> Dict:
        """Test download client connectivity"""
        self.log(f"Testing {client} connectivity...")
        result = {
            'accessible': False,
            'api_accessible': False,
            'version': None,
            'needs_config': []
        }
        
        port = SERVICES[client]['port']
        
        if self.test_service_connectivity(client, port):
            result['accessible'] = True
            self.log(f"✓ {client} is accessible", "SUCCESS")
            
            # Test specific client APIs
            try:
                if client == 'qbittorrent':
                    # qBittorrent API test
                    url = f"http://{self.host}:{port}/api/v2/app/version"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        result['api_accessible'] = True
                        result['version'] = response.text.strip()
                        self.log(f"✓ {client} API accessible (v{result['version']})", "SUCCESS")
                    else:
                        # May need login
                        result['needs_config'].append('Default login: admin/adminadmin')
                        
                elif client == 'transmission':
                    # Transmission RPC test
                    url = f"http://{self.host}:{port}/transmission/rpc"
                    session_response = requests.get(url, timeout=5)
                    if 'X-Transmission-Session-Id' in session_response.headers:
                        result['api_accessible'] = True
                        self.log(f"✓ {client} RPC accessible", "SUCCESS")
                    else:
                        result['needs_config'].append('May need authentication')
                        
                elif client == 'sabnzbd':
                    # SABnzbd API test
                    url = f"http://{self.host}:{port}/sabnzbd/api?mode=version"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        result['api_accessible'] = True
                        result['version'] = response.json().get('version', 'unknown')
                        self.log(f"✓ {client} API accessible", "SUCCESS")
                    else:
                        result['needs_config'].append('API key required')
                        
                elif client == 'nzbget':
                    # NZBGet JSON-RPC test
                    url = f"http://{self.host}:{port}/jsonrpc"
                    data = {"method": "version", "params": [], "jsonrpc": "2.0", "id": 1}
                    response = requests.post(url, json=data, timeout=5)
                    if response.status_code == 200:
                        result['api_accessible'] = True
                        result['version'] = response.json().get('result', 'unknown')
                        self.log(f"✓ {client} JSON-RPC accessible", "SUCCESS")
                        
            except Exception as e:
                self.log(f"✗ {client} API error: {str(e)}", "ERROR")
                
        else:
            self.log(f"✗ {client} is not accessible", "ERROR")
            
        return result
        
    def test_arr_to_download_client(self, arr_service: str, download_client: str) -> Dict:
        """Test connection between ARR service and download client"""
        self.log(f"Testing {arr_service} → {download_client} integration...")
        result = {
            'connection_possible': False,
            'configured': False,
            'test_successful': False,
            'needs_config': []
        }
        
        # Check if both services are accessible
        arr_port = SERVICES[arr_service]['port']
        dl_port = SERVICES[download_client]['port']
        
        if self.test_service_connectivity(arr_service, arr_port) and \
           self.test_service_connectivity(download_client, dl_port):
            result['connection_possible'] = True
            self.log(f"✓ Both services are accessible", "SUCCESS")
            
            # Would need API keys to test actual configuration
            api_key = self.get_api_key_from_config(arr_service)
            if api_key:
                # Test would go here with API key
                pass
            else:
                result['needs_config'].append(f'{arr_service} API key needed')
                result['needs_config'].append(f'Configure in {arr_service} > Settings > Download Clients')
                
        else:
            self.log(f"✗ One or both services not accessible", "ERROR")
            
        return result
        
    def test_arr_to_prowlarr(self, arr_service: str) -> Dict:
        """Test connection between ARR service and Prowlarr"""
        self.log(f"Testing {arr_service} → Prowlarr integration...")
        result = {
            'connection_possible': False,
            'configured': False,
            'indexers_synced': False,
            'needs_config': []
        }
        
        arr_port = SERVICES[arr_service]['port']
        prowlarr_port = SERVICES['prowlarr']['port']
        
        if self.test_service_connectivity(arr_service, arr_port) and \
           self.test_service_connectivity('prowlarr', prowlarr_port):
            result['connection_possible'] = True
            self.log(f"✓ Both services are accessible", "SUCCESS")
            
            # Configuration steps
            result['needs_config'].extend([
                f'1. Get Prowlarr API key from Prowlarr > Settings > General',
                f'2. In {arr_service} > Settings > Indexers > Add > Prowlarr',
                f'3. Enter Prowlarr URL: http://prowlarr:{prowlarr_port}',
                f'4. Enter Prowlarr API key',
                f'5. Test and Save'
            ])
            
        else:
            self.log(f"✗ One or both services not accessible", "ERROR")
            
        return result
        
    def test_media_server_library_scan(self, media_server: str) -> Dict:
        """Test media server library scanning capabilities"""
        self.log(f"Testing {media_server} library scanning...")
        result = {
            'accessible': False,
            'api_accessible': False,
            'library_paths': [],
            'scan_capable': False,
            'needs_config': []
        }
        
        port = SERVICES[media_server]['port']
        
        if self.test_service_connectivity(media_server, port):
            result['accessible'] = True
            self.log(f"✓ {media_server} is accessible", "SUCCESS")
            
            if media_server == 'jellyfin':
                # Jellyfin public endpoints
                try:
                    url = f"http://{self.host}:{port}/System/Info/Public"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        result['api_accessible'] = True
                        info = response.json()
                        self.log(f"✓ Jellyfin v{info.get('Version', 'unknown')} accessible", "SUCCESS")
                        result['scan_capable'] = True
                        result['library_paths'] = ['/media/movies', '/media/tv', '/media/music']
                except:
                    pass
                    
            elif media_server == 'plex':
                # Plex requires token
                result['needs_config'].append('Plex token required for API access')
                result['scan_capable'] = True
                result['library_paths'] = ['/media/movies', '/media/tv', '/media/music']
                
            elif media_server == 'emby':
                # Emby public info
                try:
                    url = f"http://{self.host}:{port}/emby/System/Info/Public"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        result['api_accessible'] = True
                        self.log(f"✓ Emby accessible", "SUCCESS")
                        result['scan_capable'] = True
                        result['library_paths'] = ['/media/movies', '/media/tv', '/media/music']
                except:
                    pass
                    
        else:
            self.log(f"✗ {media_server} is not accessible", "ERROR")
            
        return result
        
    def run_all_tests(self):
        """Run all integration tests"""
        self.log("Starting Media Server Integration Tests", "INFO")
        self.log("=" * 60, "INFO")
        
        # Test basic connectivity for all services
        self.log("\n=== Testing Service Connectivity ===", "INFO")
        for service, config in SERVICES.items():
            is_accessible = self.test_service_connectivity(service, config['port'])
            self.results['connectivity'][service] = is_accessible
            status = "✓" if is_accessible else "✗"
            self.log(f"{status} {service:<20} (port {config['port']})", 
                    "SUCCESS" if is_accessible else "ERROR")
        
        # Test Prowlarr
        self.log("\n=== Testing Prowlarr Indexer Management ===", "INFO")
        self.results['integrations']['prowlarr'] = self.test_prowlarr_connectivity()
        
        # Test download clients
        self.log("\n=== Testing Download Clients ===", "INFO")
        for client in ['qbittorrent', 'transmission', 'sabnzbd', 'nzbget']:
            self.results['integrations'][client] = self.test_download_client_connectivity(client)
        
        # Test ARR to Download Client integrations
        self.log("\n=== Testing ARR → Download Client Integrations ===", "INFO")
        arr_services = ['sonarr', 'radarr', 'lidarr', 'readarr']
        download_clients = ['qbittorrent', 'transmission', 'sabnzbd', 'nzbget']
        
        for arr in arr_services:
            self.results['integrations'][f'{arr}_download_clients'] = {}
            for dl in download_clients:
                key = f'{arr}->{dl}'
                self.results['integrations'][f'{arr}_download_clients'][key] = \
                    self.test_arr_to_download_client(arr, dl)
        
        # Test ARR to Prowlarr integrations
        self.log("\n=== Testing ARR → Prowlarr Integrations ===", "INFO")
        for arr in arr_services:
            key = f'{arr}->prowlarr'
            self.results['integrations'][key] = self.test_arr_to_prowlarr(arr)
        
        # Test media server library scanning
        self.log("\n=== Testing Media Server Library Scanning ===", "INFO")
        for server in ['jellyfin', 'plex', 'emby']:
            self.results['integrations'][f'{server}_library'] = \
                self.test_media_server_library_scan(server)
        
        # Generate report
        self.generate_report()
        
    def generate_report(self):
        """Generate detailed test report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'/Users/morlock/fun/newmedia/TEST_REPORTS/integration_test_report_{timestamp}.md'
        
        with open(report_file, 'w') as f:
            f.write("# Media Server Integration Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Service connectivity summary
            f.write("## Service Connectivity Summary\n\n")
            f.write("| Service | Port | Status | Notes |\n")
            f.write("|---------|------|--------|-------|\n")
            
            for service, is_accessible in self.results['connectivity'].items():
                port = SERVICES[service]['port']
                status = "✅ Online" if is_accessible else "❌ Offline"
                notes = ""
                
                # Add specific notes for VPN-routed services
                if service in ['qbittorrent', 'transmission'] and not is_accessible:
                    notes = "Routed through Gluetun VPN"
                    
                f.write(f"| {service} | {port} | {status} | {notes} |\n")
            
            # Prowlarr status
            f.write("\n## Prowlarr Indexer Management\n\n")
            prowlarr = self.results['integrations'].get('prowlarr', {})
            if prowlarr.get('accessible'):
                f.write("✅ **Prowlarr is accessible**\n\n")
                if prowlarr.get('api_accessible'):
                    f.write(f"- Indexers configured: {len(prowlarr.get('indexers', []))}\n")
                    f.write(f"- Apps configured: {', '.join(prowlarr.get('apps_configured', []))}\n")
                else:
                    f.write("⚠️ **API Configuration Required**:\n")
                    for step in prowlarr.get('needs_config', []):
                        f.write(f"- {step}\n")
            else:
                f.write("❌ **Prowlarr is not accessible**\n")
            
            # Download clients
            f.write("\n## Download Client Status\n\n")
            for client in ['qbittorrent', 'transmission', 'sabnzbd', 'nzbget']:
                client_data = self.results['integrations'].get(client, {})
                if client_data.get('accessible'):
                    f.write(f"### {client.title()}\n")
                    f.write(f"✅ **Accessible** ")
                    if client_data.get('version'):
                        f.write(f"(Version: {client_data['version']})")
                    f.write("\n\n")
                    
                    if client_data.get('needs_config'):
                        f.write("**Configuration Notes**:\n")
                        for note in client_data['needs_config']:
                            f.write(f"- {note}\n")
                        f.write("\n")
                else:
                    f.write(f"### {client.title()}\n")
                    f.write("❌ **Not accessible**\n\n")
            
            # Integration matrix
            f.write("\n## Integration Configuration Guide\n\n")
            f.write("### ARR Services → Download Clients\n\n")
            
            arr_services = ['sonarr', 'radarr', 'lidarr', 'readarr']
            for arr in arr_services:
                f.write(f"#### {arr.title()}\n\n")
                arr_dl = self.results['integrations'].get(f'{arr}_download_clients', {})
                
                for dl in ['qbittorrent', 'transmission', 'sabnzbd', 'nzbget']:
                    key = f'{arr}->{dl}'
                    integration = arr_dl.get(key, {})
                    
                    if integration.get('connection_possible'):
                        f.write(f"**{dl.title()}**: ✅ Connection possible\n")
                        f.write("  - Configuration steps:\n")
                        f.write(f"    1. Go to {arr.title()} → Settings → Download Clients\n")
                        f.write(f"    2. Add new {dl.title()} client\n")
                        f.write(f"    3. Host: `{dl}` (or `localhost` if VPN-routed)\n")
                        f.write(f"    4. Port: `{SERVICES[dl]['port']}`\n")
                        
                        if dl == 'qbittorrent':
                            f.write("    5. Username: `admin`, Password: `adminadmin` (default)\n")
                        elif dl == 'transmission':
                            f.write("    5. Check authentication settings if required\n")
                        elif dl in ['sabnzbd', 'nzbget']:
                            f.write("    5. Add API key from the download client\n")
                            
                        f.write("    6. Test connection and save\n\n")
                    else:
                        f.write(f"**{dl.title()}**: ❌ Service not accessible\n\n")
            
            # ARR to Prowlarr
            f.write("\n### ARR Services → Prowlarr\n\n")
            for arr in arr_services:
                key = f'{arr}->prowlarr'
                integration = self.results['integrations'].get(key, {})
                
                if integration.get('connection_possible'):
                    f.write(f"**{arr.title()}**: ✅ Ready for configuration\n")
                    f.write("  - Steps:\n")
                    for step in integration.get('needs_config', []):
                        f.write(f"    - {step}\n")
                    f.write("\n")
                else:
                    f.write(f"**{arr.title()}**: ❌ Connection not possible\n\n")
            
            # Media servers
            f.write("\n## Media Server Library Configuration\n\n")
            for server in ['jellyfin', 'plex', 'emby']:
                server_data = self.results['integrations'].get(f'{server}_library', {})
                
                f.write(f"### {server.title()}\n\n")
                if server_data.get('accessible'):
                    f.write("✅ **Server accessible**\n\n")
                    
                    if server_data.get('scan_capable'):
                        f.write("**Library Configuration**:\n")
                        f.write("1. Add libraries for the following paths:\n")
                        for path in server_data.get('library_paths', []):
                            f.write(f"   - `{path}`\n")
                        f.write("2. Enable automatic library scanning\n")
                        f.write("3. Configure metadata providers\n\n")
                        
                    if server_data.get('needs_config'):
                        f.write("**Additional Configuration**:\n")
                        for note in server_data['needs_config']:
                            f.write(f"- {note}\n")
                        f.write("\n")
                else:
                    f.write("❌ **Server not accessible**\n\n")
            
            # Quick start guide
            f.write("\n## Quick Start Configuration Order\n\n")
            f.write("1. **Start Prowlarr** and add indexers\n")
            f.write("2. **Configure download clients** (qBittorrent/SABnzbd)\n")
            f.write("3. **Link ARR services to Prowlarr** for indexer management\n")
            f.write("4. **Add download clients to ARR services**\n")
            f.write("5. **Configure media servers** and add library paths\n")
            f.write("6. **Test the flow**: Search → Download → Import → Library Update\n")
            
            # Common issues
            f.write("\n## Common Issues and Solutions\n\n")
            f.write("### VPN-routed Download Clients\n")
            f.write("- qBittorrent and Transmission are routed through Gluetun VPN\n")
            f.write("- Access them via `localhost:8080` and `localhost:9091`\n")
            f.write("- In ARR services, use `gluetun` as the host (not `qbittorrent`)\n\n")
            
            f.write("### API Keys\n")
            f.write("- Most services require API keys for integration\n")
            f.write("- Find API keys in each service under Settings → General → Security\n")
            f.write("- Store API keys securely in your `.env` file\n\n")
            
            f.write("### Network Configuration\n")
            f.write("- All services are on the `media-net` Docker network\n")
            f.write("- Services can reach each other using container names\n")
            f.write("- External access uses the mapped ports\n\n")
            
        self.log(f"\n✓ Report generated: {report_file}", "SUCCESS")
        
        # Also save raw results as JSON
        json_file = f'/Users/morlock/fun/newmedia/TEST_REPORTS/integration_test_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log(f"✓ Raw results saved: {json_file}", "SUCCESS")

if __name__ == "__main__":
    tester = IntegrationTester()
    tester.run_all_tests()