#!/usr/bin/env python3
"""
ARR Integration Verification Script
Comprehensive testing and status report of all ARR services integration
"""

import requests
import json
import sys
from typing import Dict, List, Optional
from datetime import datetime

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
        'api_version': 'v3'
    },
    'radarr': {
        'url': 'http://localhost:7878',
        'api_key': '7b74da952069425f9568ea361b001a12',
        'api_version': 'v3'
    },
    'lidarr': {
        'url': 'http://localhost:8686',
        'api_key': 'e8262da767e34a6b8ca7ca1e92384d96',
        'api_version': 'v1'
    }
}

class ARRVerification:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
        self.verification_results = {}
        
    def test_service_status(self, service_name: str) -> Dict:
        """Test service status and get system info"""
        config = SERVICES[service_name]
        result = {
            'accessible': False,
            'version': None,
            'error': None
        }
        
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/system/status",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                result['accessible'] = True
                result['version'] = data.get('version', 'Unknown')
            else:
                result['error'] = f"HTTP {response.status_code}"
                
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def get_prowlarr_applications(self) -> List[Dict]:
        """Get Prowlarr applications"""
        try:
            headers = {'X-Api-Key': SERVICES['prowlarr']['api_key']}
            response = self.session.get(
                f"{SERVICES['prowlarr']['url']}/api/v1/applications",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            return []
    
    def get_prowlarr_indexers(self) -> List[Dict]:
        """Get Prowlarr indexers"""
        try:
            headers = {'X-Api-Key': SERVICES['prowlarr']['api_key']}
            response = self.session.get(
                f"{SERVICES['prowlarr']['url']}/api/v1/indexer",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            return []
    
    def get_download_clients(self, service_name: str) -> List[Dict]:
        """Get download clients for a service"""
        if service_name == 'prowlarr':
            return []
            
        config = SERVICES[service_name]
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/downloadclient",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            return []
    
    def get_root_folders(self, service_name: str) -> List[Dict]:
        """Get root folders for a service"""
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
            return []
    
    def get_indexers_for_service(self, service_name: str) -> List[Dict]:
        """Get indexers configured for a specific ARR service"""
        if service_name == 'prowlarr':
            return []
            
        config = SERVICES[service_name]
        try:
            headers = {'X-Api-Key': config['api_key']}
            response = self.session.get(
                f"{config['url']}/api/{config['api_version']}/indexer",
                headers=headers
            )
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            return []
    
    def run_verification(self) -> Dict:
        """Run comprehensive verification"""
        print("🔍 ARR Services Integration Verification")
        print("=" * 50)
        
        verification_report = {
            'timestamp': datetime.now().isoformat(),
            'services': {},
            'integration_status': {},
            'recommendations': []
        }
        
        # Test all services
        print("\n📡 Testing Service Connectivity...")
        for service_name in SERVICES.keys():
            result = self.test_service_status(service_name)
            verification_report['services'][service_name] = result
            
            if result['accessible']:
                print(f"✅ {service_name.title()}: v{result['version']}")
            else:
                print(f"❌ {service_name.title()}: {result['error']}")
        
        # Check Prowlarr integration
        print("\n🔗 Checking Prowlarr Integration...")
        prowlarr_apps = self.get_prowlarr_applications()
        prowlarr_indexers = self.get_prowlarr_indexers()
        
        verification_report['integration_status']['prowlarr_applications'] = len(prowlarr_apps)
        verification_report['integration_status']['prowlarr_indexers'] = len(prowlarr_indexers)
        
        if prowlarr_apps:
            print(f"✅ Prowlarr Applications: {len(prowlarr_apps)} configured")
            for app in prowlarr_apps:
                app_name = app.get('name', 'Unknown')
                sync_level = app.get('syncLevel', 'Unknown')
                print(f"   - {app_name} (Sync: {sync_level})")
        else:
            print("❌ No applications configured in Prowlarr")
            verification_report['recommendations'].append("Configure ARR applications in Prowlarr")
        
        if prowlarr_indexers:
            print(f"✅ Prowlarr Indexers: {len(prowlarr_indexers)} configured")
            enabled_indexers = [idx for idx in prowlarr_indexers if idx.get('enable', False)]
            print(f"   - {len(enabled_indexers)} enabled, {len(prowlarr_indexers) - len(enabled_indexers)} disabled")
        else:
            print("⚠️  No indexers configured in Prowlarr")
            verification_report['recommendations'].append("Add indexers to Prowlarr for content discovery")
        
        # Check ARR services configuration
        print("\n⬇️  Checking Download Clients...")
        arr_services = ['sonarr', 'radarr', 'lidarr']
        
        for service_name in arr_services:
            if not verification_report['services'][service_name]['accessible']:
                continue
                
            download_clients = self.get_download_clients(service_name)
            verification_report['integration_status'][f'{service_name}_download_clients'] = len(download_clients)
            
            if download_clients:
                print(f"✅ {service_name.title()}: {len(download_clients)} download client(s)")
                for client in download_clients:
                    client_name = client.get('name', 'Unknown')
                    enabled = "✓" if client.get('enable', False) else "✗"
                    print(f"   - {client_name} ({enabled})")
            else:
                print(f"⚠️  {service_name.title()}: No download clients configured")
                verification_report['recommendations'].append(f"Configure download client for {service_name.title()}")
        
        # Check root folders
        print("\n📁 Checking Media Root Folders...")
        for service_name in arr_services:
            if not verification_report['services'][service_name]['accessible']:
                continue
                
            root_folders = self.get_root_folders(service_name)
            verification_report['integration_status'][f'{service_name}_root_folders'] = len(root_folders)
            
            if root_folders:
                print(f"✅ {service_name.title()}: {len(root_folders)} root folder(s)")
                for folder in root_folders:
                    path = folder.get('path', 'Unknown')
                    free_space = folder.get('freeSpace', 0)
                    free_gb = free_space / (1024**3) if free_space else 0
                    print(f"   - {path} ({free_gb:.1f} GB free)")
            else:
                print(f"❌ {service_name.title()}: No root folders configured")
                verification_report['recommendations'].append(f"Configure root folder for {service_name.title()}")
        
        # Check synced indexers
        print("\n🔍 Checking Synced Indexers...")
        for service_name in arr_services:
            if not verification_report['services'][service_name]['accessible']:
                continue
                
            service_indexers = self.get_indexers_for_service(service_name)
            verification_report['integration_status'][f'{service_name}_indexers'] = len(service_indexers)
            
            if service_indexers:
                enabled_count = len([idx for idx in service_indexers if idx.get('enable', False)])
                print(f"✅ {service_name.title()}: {len(service_indexers)} indexers ({enabled_count} enabled)")
            else:
                print(f"⚠️  {service_name.title()}: No indexers synced")
                verification_report['recommendations'].append(f"Trigger indexer sync for {service_name.title()}")
        
        # Overall integration health
        print("\n🏥 Integration Health Check...")
        health_score = 0
        max_score = 0
        
        # Service accessibility (4 points max)
        accessible_services = sum(1 for service in verification_report['services'].values() if service['accessible'])
        health_score += accessible_services * 1
        max_score += 4
        
        # Prowlarr applications (2 points)
        if verification_report['integration_status']['prowlarr_applications'] >= 3:
            health_score += 2
        elif verification_report['integration_status']['prowlarr_applications'] > 0:
            health_score += 1
        max_score += 2
        
        # Prowlarr indexers (2 points)
        if verification_report['integration_status']['prowlarr_indexers'] > 0:
            health_score += 2
        max_score += 2
        
        # Download clients (3 points)
        download_client_services = sum(1 for service in arr_services 
                                     if verification_report['integration_status'].get(f'{service}_download_clients', 0) > 0)
        health_score += download_client_services
        max_score += 3
        
        # Root folders (3 points)
        root_folder_services = sum(1 for service in arr_services 
                                 if verification_report['integration_status'].get(f'{service}_root_folders', 0) > 0)
        health_score += root_folder_services
        max_score += 3
        
        health_percentage = (health_score / max_score) * 100 if max_score > 0 else 0
        verification_report['health_score'] = health_percentage
        
        if health_percentage >= 90:
            print(f"🎉 Excellent: {health_percentage:.1f}% ({health_score}/{max_score})")
        elif health_percentage >= 70:
            print(f"✅ Good: {health_percentage:.1f}% ({health_score}/{max_score})")
        elif health_percentage >= 50:
            print(f"⚠️  Fair: {health_percentage:.1f}% ({health_score}/{max_score})")
        else:
            print(f"❌ Needs Work: {health_percentage:.1f}% ({health_score}/{max_score})")
        
        # Recommendations
        if verification_report['recommendations']:
            print(f"\n📋 Recommendations ({len(verification_report['recommendations'])}):")
            for i, rec in enumerate(verification_report['recommendations'], 1):
                print(f"   {i}. {rec}")
        else:
            print("\n✨ No recommendations - everything looks great!")
        
        print("\n" + "=" * 50)
        print("🎯 Verification Complete!")
        
        # Service URLs
        print(f"\n🔗 Service URLs:")
        for service_name, config in SERVICES.items():
            status = "✅" if verification_report['services'][service_name]['accessible'] else "❌"
            print(f"   {status} {service_name.title()}: {config['url']}")
        print(f"   📦 qBittorrent: http://localhost:8090")
        
        return verification_report

def main():
    """Main function"""
    verifier = ARRVerification()
    
    try:
        report = verifier.run_verification()
        
        # Save report to file
        with open('/Users/morlock/fun/newmedia/arr-integration-report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: arr-integration-report.json")
        
        # Exit with appropriate code based on health score
        health_score = report.get('health_score', 0)
        sys.exit(0 if health_score >= 70 else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()