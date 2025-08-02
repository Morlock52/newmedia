#!/usr/bin/env python3

"""
Configuration Validator for Media Server Stack - 2025
Validates Docker Compose configurations against 2025 best practices
Based on research findings from MEDIA_SERVER_INTEGRATION_RESEARCH_2025.md
"""

import json
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

class ConfigValidator:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = None
        self.results = {
            'health_checks': [],
            'security': [],
            'performance': [],
            'networking': [],
            'best_practices': [],
            'errors': [],
            'warnings': [],
            'score': 0
        }
        self.total_checks = 0
        self.passed_checks = 0

    def load_config(self) -> bool:
        """Load and parse Docker Compose configuration"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f"✅ Loaded configuration from {self.config_path}")
            return True
        except FileNotFoundError:
            self.results['errors'].append(f"Configuration file not found: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            self.results['errors'].append(f"Invalid YAML syntax: {e}")
            return False

    def validate_health_checks(self):
        """Validate health check implementations"""
        print("\n🔍 Validating Health Checks...")
        
        services = self.config.get('services', {})
        critical_services = [
            'jellyfin', 'plex', 'emby', 'sonarr', 'radarr', 'prowlarr',
            'qbittorrent', 'sabnzbd', 'prometheus', 'grafana', 'postgres', 'redis'
        ]

        for service_name, service_config in services.items():
            self.total_checks += 1
            
            if service_name in critical_services:
                if 'healthcheck' in service_config:
                    healthcheck = service_config['healthcheck']
                    
                    # Check for required health check parameters
                    required_params = ['test', 'interval', 'timeout', 'retries']
                    missing_params = [param for param in required_params if param not in healthcheck]
                    
                    if not missing_params:
                        self.passed_checks += 1
                        self.results['health_checks'].append({
                            'service': service_name,
                            'status': 'PASS',
                            'message': 'Complete health check configuration'
                        })
                    else:
                        self.results['health_checks'].append({
                            'service': service_name,
                            'status': 'WARNING',
                            'message': f'Missing health check parameters: {missing_params}'
                        })
                        self.results['warnings'].append(f'{service_name}: Incomplete health check')
                else:
                    self.results['health_checks'].append({
                        'service': service_name,
                        'status': 'FAIL',
                        'message': 'No health check configured'
                    })
                    self.results['errors'].append(f'{service_name}: Missing health check')

    def validate_security_config(self):
        """Validate security configurations"""
        print("🔒 Validating Security Configuration...")
        
        services = self.config.get('services', {})
        
        # Check for Docker socket exposure
        for service_name, service_config in services.items():
            self.total_checks += 1
            volumes = service_config.get('volumes', [])
            
            for volume in volumes:
                if '/var/run/docker.sock' in str(volume):
                    if ':ro' in str(volume):
                        self.passed_checks += 1
                        self.results['security'].append({
                            'service': service_name,
                            'status': 'PASS',
                            'message': 'Docker socket mounted read-only'
                        })
                    else:
                        self.results['security'].append({
                            'service': service_name,
                            'status': 'FAIL',
                            'message': 'Docker socket mounted with write access - security risk'
                        })
                        self.results['errors'].append(f'{service_name}: Insecure Docker socket mount')

        # Check for socket proxy usage
        socket_proxy_found = 'socket-proxy' in services
        if socket_proxy_found:
            self.passed_checks += 1
            self.results['security'].append({
                'service': 'socket-proxy',
                'status': 'PASS',
                'message': 'Socket proxy service found - enhanced security'
            })
        else:
            self.results['security'].append({
                'service': 'general',
                'status': 'WARNING',
                'message': 'No socket proxy found - consider implementing for enhanced security'
            })

        # Check for security options
        for service_name, service_config in services.items():
            self.total_checks += 1
            security_opt = service_config.get('security_opt', [])
            
            if 'no-new-privileges:true' in security_opt:
                self.passed_checks += 1
                self.results['security'].append({
                    'service': service_name,
                    'status': 'PASS',
                    'message': 'no-new-privileges security option enabled'
                })
            else:
                self.results['security'].append({
                    'service': service_name,
                    'status': 'WARNING',
                    'message': 'Consider adding no-new-privileges:true for enhanced security'
                })

    def validate_performance_config(self):
        """Validate performance-related configurations"""
        print("⚡ Validating Performance Configuration...")
        
        services = self.config.get('services', {})
        
        for service_name, service_config in services.items():
            self.total_checks += 1
            
            # Check for resource limits
            deploy_config = service_config.get('deploy', {})
            resources = deploy_config.get('resources', {})
            limits = resources.get('limits', {})
            
            if limits:
                self.passed_checks += 1
                memory_limit = limits.get('memory', 'Not set')
                cpu_limit = limits.get('cpus', 'Not set')
                self.results['performance'].append({
                    'service': service_name,
                    'status': 'PASS',
                    'message': f'Resource limits configured - Memory: {memory_limit}, CPU: {cpu_limit}'
                })
            else:
                self.results['performance'].append({
                    'service': service_name,
                    'status': 'WARNING',
                    'message': 'No resource limits configured - may affect performance under load'
                })

        # Check for restart policies
        for service_name, service_config in services.items():
            self.total_checks += 1
            restart_policy = service_config.get('restart', 'none')
            
            if restart_policy in ['unless-stopped', 'always']:
                self.passed_checks += 1
                self.results['performance'].append({
                    'service': service_name,
                    'status': 'PASS',
                    'message': f'Appropriate restart policy: {restart_policy}'
                })
            else:
                self.results['performance'].append({
                    'service': service_name,
                    'status': 'WARNING',
                    'message': f'Consider using unless-stopped restart policy (current: {restart_policy})'
                })

    def validate_networking_config(self):
        """Validate networking configurations"""
        print("🌐 Validating Network Configuration...")
        
        networks = self.config.get('networks', {})
        services = self.config.get('services', {})
        
        # Check for custom networks
        self.total_checks += 1
        if networks:
            self.passed_checks += 1
            self.results['networking'].append({
                'service': 'general',
                'status': 'PASS',
                'message': f'Custom networks defined: {list(networks.keys())}'
            })
        else:
            self.results['networking'].append({
                'service': 'general',
                'status': 'WARNING',
                'message': 'No custom networks defined - using default bridge'
            })

        # Check for VPN network isolation
        vpn_services = ['gluetun', 'qbittorrent', 'transmission']
        for service_name in vpn_services:
            if service_name in services:
                service_config = services[service_name]
                network_mode = service_config.get('network_mode', '')
                
                self.total_checks += 1
                if 'service:' in network_mode:
                    self.passed_checks += 1
                    self.results['networking'].append({
                        'service': service_name,
                        'status': 'PASS',
                        'message': f'VPN network isolation configured: {network_mode}'
                    })
                else:
                    self.results['networking'].append({
                        'service': service_name,
                        'status': 'WARNING',
                        'message': 'Consider using VPN network isolation for download clients'
                    })

    def validate_2025_best_practices(self):
        """Validate 2025-specific best practices"""
        print("🚀 Validating 2025 Best Practices...")
        
        services = self.config.get('services', {})
        
        # Check for Prowlarr authentication (2025 requirement)
        if 'prowlarr' in services:
            prowlarr_config = services['prowlarr']
            environment = prowlarr_config.get('environment', {})
            
            self.total_checks += 1
            auth_found = False
            
            # Check for authentication environment variables
            if isinstance(environment, dict):
                for key, value in environment.items():
                    if 'AUTHENTICATION' in key.upper():
                        auth_found = True
                        break
            elif isinstance(environment, list):
                for env_var in environment:
                    if 'AUTHENTICATION' in str(env_var).upper():
                        auth_found = True
                        break
            
            if auth_found:
                self.passed_checks += 1
                self.results['best_practices'].append({
                    'service': 'prowlarr',
                    'status': 'PASS',
                    'message': 'Prowlarr authentication configured (2025 requirement)'
                })
            else:
                self.results['best_practices'].append({
                    'service': 'prowlarr',
                    'status': 'FAIL',
                    'message': 'Prowlarr authentication not configured - mandatory in 2025'
                })
                self.results['errors'].append('Prowlarr: Authentication not configured')

        # Check for modern Docker Compose version
        version = self.config.get('version', '')
        self.total_checks += 1
        
        if version.startswith('3.'):
            version_number = float(version.split('.')[1])
            if version_number >= 8:
                self.passed_checks += 1
                self.results['best_practices'].append({
                    'service': 'general',
                    'status': 'PASS',
                    'message': f'Modern Docker Compose version: {version}'
                })
            else:
                self.results['best_practices'].append({
                    'service': 'general',
                    'status': 'WARNING',
                    'message': f'Consider upgrading Docker Compose version (current: {version})'
                })
        else:
            self.results['best_practices'].append({
                'service': 'general',
                'status': 'WARNING',
                'message': f'Unknown Docker Compose version format: {version}'
            })

        # Check for service dependencies with health conditions
        for service_name, service_config in services.items():
            depends_on = service_config.get('depends_on', {})
            
            if isinstance(depends_on, dict):
                self.total_checks += 1
                has_health_condition = any(
                    isinstance(dep, dict) and 'condition' in dep 
                    for dep in depends_on.values()
                )
                
                if has_health_condition:
                    self.passed_checks += 1
                    self.results['best_practices'].append({
                        'service': service_name,
                        'status': 'PASS',
                        'message': 'Service dependencies with health conditions configured'
                    })
                elif depends_on:
                    self.results['best_practices'].append({
                        'service': service_name,
                        'status': 'WARNING',
                        'message': 'Consider adding health conditions to service dependencies'
                    })

    def calculate_score(self) -> int:
        """Calculate overall configuration score"""
        if self.total_checks == 0:
            return 0
        
        base_score = (self.passed_checks / self.total_checks) * 100
        
        # Penalties for critical issues
        critical_errors = len(self.results['errors'])
        penalty = min(critical_errors * 10, 50)  # Max 50% penalty
        
        final_score = max(0, base_score - penalty)
        return int(final_score)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        self.results['score'] = self.calculate_score()
        self.results['summary'] = {
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.total_checks - self.passed_checks,
            'error_count': len(self.results['errors']),
            'warning_count': len(self.results['warnings']),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results

    def print_report(self):
        """Print formatted validation report"""
        print("\n" + "="*80)
        print("📊 CONFIGURATION VALIDATION REPORT - 2025")
        print("="*80)
        
        # Summary
        score = self.results['score']
        score_color = '\033[92m' if score >= 80 else '\033[93m' if score >= 60 else '\033[91m'
        print(f"\n🎯 Overall Score: {score_color}{score}/100\033[0m")
        
        summary = self.results['summary']
        print(f"📈 Checks: {summary['passed_checks']}/{summary['total_checks']} passed")
        print(f"❌ Errors: {summary['error_count']}")
        print(f"⚠️  Warnings: {summary['warning_count']}")
        
        # Detailed results by category
        categories = [
            ('Health Checks', 'health_checks', '🔍'),
            ('Security', 'security', '🔒'),
            ('Performance', 'performance', '⚡'),
            ('Networking', 'networking', '🌐'),
            ('2025 Best Practices', 'best_practices', '🚀')
        ]
        
        for category_name, category_key, icon in categories:
            print(f"\n{icon} {category_name}:")
            results = self.results[category_key]
            
            if not results:
                print("  No checks performed")
                continue
                
            for result in results:
                status_icon = '✅' if result['status'] == 'PASS' else '⚠️ ' if result['status'] == 'WARNING' else '❌'
                print(f"  {status_icon} {result['service']}: {result['message']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if self.results['errors']:
            print("  🚨 Critical Issues to Fix:")
            for error in self.results['errors']:
                print(f"    - {error}")
        
        if self.results['warnings']:
            print("  ⚠️  Improvements to Consider:")
            warning_count = min(5, len(self.results['warnings']))  # Show top 5
            for warning in self.results['warnings'][:warning_count]:
                print(f"    - {warning}")
            
            if len(self.results['warnings']) > 5:
                print(f"    ... and {len(self.results['warnings']) - 5} more warnings")
        
        if score >= 90:
            print("  🎉 Excellent configuration! Your setup follows 2025 best practices.")
        elif score >= 75:
            print("  👍 Good configuration with room for improvement.")
        elif score >= 50:
            print("  🔧 Configuration needs attention to meet 2025 standards.")
        else:
            print("  🚨 Configuration requires significant improvements for production use.")

    def save_report(self, output_path: str):
        """Save validation report to JSON file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as file:
                json.dump(self.results, file, indent=2)
            print(f"\n📁 Report saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")

    def run_validation(self) -> bool:
        """Run complete validation suite"""
        if not self.load_config():
            return False
        
        print("🚀 Starting Docker Compose Configuration Validation - 2025")
        print(f"📋 Analyzing: {self.config_path}")
        
        # Run all validation checks
        self.validate_health_checks()
        self.validate_security_config()
        self.validate_performance_config()
        self.validate_networking_config()
        self.validate_2025_best_practices()
        
        # Generate and display results
        self.generate_report()
        self.print_report()
        
        return len(self.results['errors']) == 0

def main():
    parser = argparse.ArgumentParser(
        description='Validate Docker Compose configuration against 2025 best practices'
    )
    parser.add_argument(
        'config_file',
        help='Path to docker-compose.yml file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file for JSON report'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with non-zero code if any warnings found'
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ConfigValidator(args.config_file)
    
    # Run validation
    success = validator.run_validation()
    
    # Save report if requested
    if args.output:
        validator.save_report(args.output)
    
    # Exit with appropriate code
    if not success:
        sys.exit(1)
    elif args.strict and validator.results['warnings']:
        print("\n⚠️  Exiting with error due to --strict mode and warnings present")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()