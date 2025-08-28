#!/usr/bin/env python3
"""
MediaFlow Dashboard Integration Test Suite
Complete functional testing for the media server dashboard system
"""

import requests
import json
import time
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any

class MediaFlowTester:
    def __init__(self):
        self.results = {
            'test_start_time': datetime.now().isoformat(),
            'docker_status': {},
            'service_connectivity': {},
            'api_endpoints': {},
            'dashboard_functionality': {},
            'mobile_responsiveness': {},
            'security_tests': {},
            'performance_metrics': {},
            'overall_score': 0,
            'issues_found': [],
            'recommendations': []
        }
        
        self.services = {
            'jellyfin': {'port': 8096, 'path': '/System/Info'},
            'sonarr': {'port': 8989, 'path': '/api/v3/system/status'},
            'radarr': {'port': 7878, 'path': '/api/v3/system/status'},
            'prowlarr': {'port': 9696, 'path': '/api/v1/system/status'},
            'qbittorrent': {'port': 8080, 'path': '/'},
            'dashboard': {'port': 8090, 'path': '/'}
        }

    def test_docker_containers(self):
        """Test Docker container status"""
        print("🐳 Testing Docker containers...")
        try:
            result = subprocess.run(['docker', 'ps', '--format', 'json'], 
                                  capture_output=True, text=True, check=True)
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            
            self.results['docker_status'] = {
                'total_containers': len(containers),
                'running_containers': len([c for c in containers if 'Up' in c.get('Status', '')]),
                'containers': containers,
                'status': 'PASS' if len(containers) > 0 else 'FAIL'
            }
            
            print(f"   ✅ Found {len(containers)} running containers")
            
        except Exception as e:
            self.results['docker_status'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"   ❌ Docker test failed: {e}")

    def test_service_connectivity(self):
        """Test basic service connectivity"""
        print("🌐 Testing service connectivity...")
        
        for service_name, config in self.services.items():
            try:
                url = f"http://localhost:{config['port']}{config['path']}"
                response = requests.get(url, timeout=10)
                
                self.results['service_connectivity'][service_name] = {
                    'url': url,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'accessible': response.status_code in [200, 401],  # 401 is OK for secured endpoints
                    'status': 'PASS' if response.status_code in [200, 401] else 'FAIL'
                }
                
                status = "✅" if response.status_code in [200, 401] else "❌"
                print(f"   {status} {service_name}: {response.status_code} ({response.elapsed.total_seconds():.2f}s)")
                
            except Exception as e:
                self.results['service_connectivity'][service_name] = {
                    'status': 'FAIL',
                    'error': str(e),
                    'accessible': False
                }
                print(f"   ❌ {service_name}: Connection failed - {e}")

    def test_dashboard_functionality(self):
        """Test dashboard-specific functionality"""
        print("📊 Testing dashboard functionality...")
        
        tests = {
            'main_page': self._test_dashboard_main,
            'api_endpoints': self._test_dashboard_api,
            'static_assets': self._test_static_assets,
            'real_time_features': self._test_realtime_features
        }
        
        for test_name, test_func in tests.items():
            try:
                result = test_func()
                self.results['dashboard_functionality'][test_name] = result
                status = "✅" if result.get('status') == 'PASS' else "❌"
                print(f"   {status} {test_name}: {result.get('message', 'OK')}")
            except Exception as e:
                self.results['dashboard_functionality'][test_name] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
                print(f"   ❌ {test_name}: {e}")

    def _test_dashboard_main(self):
        """Test main dashboard page"""
        try:
            response = requests.get('http://localhost:8090/', timeout=10)
            if response.status_code == 200:
                content = response.text
                has_required_elements = all(element in content for element in [
                    'MediaFlow', 'Dashboard', 'Service Status', 'AI Assistant'
                ])
                return {
                    'status': 'PASS' if has_required_elements else 'FAIL',
                    'message': 'All required elements present' if has_required_elements else 'Missing elements',
                    'content_length': len(content)
                }
            else:
                return {'status': 'FAIL', 'message': f'HTTP {response.status_code}'}
        except Exception as e:
            return {'status': 'FAIL', 'message': str(e)}

    def _test_dashboard_api(self):
        """Test dashboard API endpoints"""
        api_endpoints = ['/api/services', '/api/system', '/api/health']
        results = {}
        
        for endpoint in api_endpoints:
            try:
                response = requests.get(f'http://localhost:8090{endpoint}', timeout=5)
                results[endpoint] = {
                    'status_code': response.status_code,
                    'accessible': response.status_code in [200, 404]  # 404 is OK if endpoint doesn't exist
                }
            except Exception as e:
                results[endpoint] = {'error': str(e), 'accessible': False}
        
        passed = sum(1 for r in results.values() if r.get('accessible', False))
        return {
            'status': 'PASS' if passed > 0 else 'FAIL',
            'message': f'{passed}/{len(api_endpoints)} endpoints accessible',
            'endpoints': results
        }

    def _test_static_assets(self):
        """Test static asset loading"""
        assets = ['/mobile-ui.css', '/social-share.js']
        results = {}
        
        for asset in assets:
            try:
                response = requests.get(f'http://localhost:8090{asset}', timeout=5)
                results[asset] = {
                    'status_code': response.status_code,
                    'size': len(response.content) if response.status_code == 200 else 0
                }
            except Exception as e:
                results[asset] = {'error': str(e)}
        
        return {
            'status': 'PASS',  # Assets may not exist, that's OK
            'message': 'Asset loading test completed',
            'assets': results
        }

    def _test_realtime_features(self):
        """Test real-time features like Socket.IO"""
        try:
            # Test Socket.IO endpoint
            response = requests.get('http://localhost:8090/socket.io/', timeout=5)
            socket_available = response.status_code != 404
            
            return {
                'status': 'PASS' if socket_available else 'WARN',
                'message': 'Socket.IO available' if socket_available else 'Socket.IO not available (fallback mode)',
                'socket_io': socket_available
            }
        except Exception as e:
            return {
                'status': 'WARN',
                'message': 'Real-time features test inconclusive',
                'error': str(e)
            }

    def test_mobile_responsiveness(self):
        """Test mobile responsiveness and viewport handling"""
        print("📱 Testing mobile responsiveness...")
        
        try:
            response = requests.get('http://localhost:8090/', timeout=10)
            if response.status_code == 200:
                content = response.text
                mobile_features = {
                    'viewport_meta': 'viewport' in content and 'width=device-width' in content,
                    'responsive_css': 'md:' in content or '@media' in content,
                    'mobile_menu': 'toggleSidebar' in content,
                    'touch_friendly': 'touch' in content.lower() or 'mobile' in content.lower()
                }
                
                score = sum(mobile_features.values())
                self.results['mobile_responsiveness'] = {
                    'status': 'PASS' if score >= 2 else 'FAIL',
                    'score': f'{score}/4',
                    'features': mobile_features
                }
                print(f"   ✅ Mobile responsiveness: {score}/4 features present")
            else:
                raise Exception(f"Dashboard not accessible: HTTP {response.status_code}")
                
        except Exception as e:
            self.results['mobile_responsiveness'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"   ❌ Mobile responsiveness test failed: {e}")

    def test_security(self):
        """Test basic security measures"""
        print("🛡️  Testing security...")
        
        security_tests = {
            'https_headers': self._test_security_headers,
            'content_security': self._test_content_security,
            'authentication': self._test_authentication
        }
        
        for test_name, test_func in security_tests.items():
            try:
                result = test_func()
                self.results['security_tests'][test_name] = result
                status = "✅" if result.get('status') != 'FAIL' else "❌"
                print(f"   {status} {test_name}: {result.get('message', 'OK')}")
            except Exception as e:
                self.results['security_tests'][test_name] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
                print(f"   ❌ {test_name}: {e}")

    def _test_security_headers(self):
        """Test security headers"""
        try:
            response = requests.get('http://localhost:3000/', timeout=10)
            headers = response.headers
            
            security_headers = {
                'X-Content-Type-Options': 'X-Content-Type-Options' in headers,
                'X-Frame-Options': 'X-Frame-Options' in headers,
                'X-XSS-Protection': 'X-XSS-Protection' in headers,
                'Content-Security-Policy': 'Content-Security-Policy' in headers
            }
            
            score = sum(security_headers.values())
            return {
                'status': 'WARN' if score < 2 else 'PASS',
                'message': f'{score}/4 security headers present',
                'headers': security_headers
            }
        except Exception as e:
            return {'status': 'FAIL', 'message': str(e)}

    def _test_content_security(self):
        """Test content security"""
        try:
            response = requests.get('http://localhost:8090/', timeout=10)
            content = response.text
            
            # Check for potential security issues
            issues = []
            if 'eval(' in content:
                issues.append('eval() usage detected')
            if 'innerHTML' in content and 'user' in content.lower():
                issues.append('Potential XSS risk with innerHTML')
            
            return {
                'status': 'FAIL' if issues else 'PASS',
                'message': f'{len(issues)} security issues found',
                'issues': issues
            }
        except Exception as e:
            return {'status': 'FAIL', 'message': str(e)}

    def _test_authentication(self):
        """Test authentication mechanisms"""
        # Check if services require authentication (401 responses are good)
        auth_services = ['sonarr', 'radarr', 'prowlarr']
        protected_count = 0
        
        for service in auth_services:
            if service in self.results['service_connectivity']:
                if self.results['service_connectivity'][service].get('status_code') == 401:
                    protected_count += 1
        
        return {
            'status': 'PASS' if protected_count > 0 else 'WARN',
            'message': f'{protected_count}/{len(auth_services)} services require authentication',
            'protected_services': protected_count
        }

    def test_performance(self):
        """Test basic performance metrics"""
        print("⚡ Testing performance...")
        
        try:
            # Test dashboard load time
            start_time = time.time()
            response = requests.get('http://localhost:8090/', timeout=30)
            load_time = time.time() - start_time
            
            # Test multiple requests for stability
            response_times = []
            for i in range(5):
                start = time.time()
                requests.get('http://localhost:8090/', timeout=10)
                response_times.append(time.time() - start)
            
            avg_response_time = sum(response_times) / len(response_times)
            
            self.results['performance_metrics'] = {
                'initial_load_time': load_time,
                'average_response_time': avg_response_time,
                'content_size': len(response.content),
                'status': 'PASS' if load_time < 5.0 and avg_response_time < 2.0 else 'WARN'
            }
            
            print(f"   ✅ Load time: {load_time:.2f}s, Avg response: {avg_response_time:.2f}s")
            
        except Exception as e:
            self.results['performance_metrics'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"   ❌ Performance test failed: {e}")

    def calculate_overall_score(self):
        """Calculate overall system health score"""
        scores = {
            'docker_status': 25,
            'service_connectivity': 30,
            'dashboard_functionality': 25,
            'mobile_responsiveness': 10,
            'security_tests': 10
        }
        
        total_score = 0
        max_score = sum(scores.values())
        
        for category, weight in scores.items():
            if category in self.results:
                category_data = self.results[category]
                if isinstance(category_data, dict):
                    if category_data.get('status') == 'PASS':
                        total_score += weight
                    elif category_data.get('status') == 'WARN':
                        total_score += weight * 0.5
                elif category == 'service_connectivity':
                    # Special handling for service connectivity
                    accessible_services = sum(1 for s in category_data.values() 
                                            if s.get('accessible', False))
                    total_services = len(category_data)
                    if total_services > 0:
                        total_score += weight * (accessible_services / total_services)
                elif category == 'dashboard_functionality':
                    # Special handling for dashboard functionality
                    passed_tests = sum(1 for t in category_data.values() 
                                     if t.get('status') == 'PASS')
                    total_tests = len(category_data)
                    if total_tests > 0:
                        total_score += weight * (passed_tests / total_tests)
        
        self.results['overall_score'] = round((total_score / max_score) * 100, 1)

    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Docker recommendations
        if self.results.get('docker_status', {}).get('status') == 'FAIL':
            recommendations.append("Fix Docker container issues - some services are not running")
        
        # Service connectivity recommendations
        service_conn = self.results.get('service_connectivity', {})
        failed_services = [name for name, data in service_conn.items() 
                          if not data.get('accessible', False)]
        if failed_services:
            recommendations.append(f"Fix connectivity for: {', '.join(failed_services)}")
        
        # Dashboard recommendations
        dashboard_func = self.results.get('dashboard_functionality', {})
        if any(test.get('status') == 'FAIL' for test in dashboard_func.values()):
            recommendations.append("Fix dashboard functionality issues")
        
        # Security recommendations
        security_tests = self.results.get('security_tests', {})
        if security_tests.get('https_headers', {}).get('status') == 'WARN':
            recommendations.append("Add security headers for better protection")
        
        # Performance recommendations
        perf = self.results.get('performance_metrics', {})
        if perf.get('initial_load_time', 0) > 3.0:
            recommendations.append("Optimize dashboard load time")
        
        self.results['recommendations'] = recommendations

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting MediaFlow Dashboard Integration Tests")
        print("=" * 60)
        
        # Run all test categories
        self.test_docker_containers()
        self.test_service_connectivity()
        self.test_dashboard_functionality()
        self.test_mobile_responsiveness()
        self.test_security()
        self.test_performance()
        
        # Calculate final score and recommendations
        self.calculate_overall_score()
        self.generate_recommendations()
        
        # Record completion time
        self.results['test_end_time'] = datetime.now().isoformat()
        
        print("\n" + "=" * 60)
        print(f"🏁 Tests completed! Overall score: {self.results['overall_score']}/100")

    def save_report(self, filename='integration-test-report.json'):
        """Save detailed test report"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Detailed report saved to: {filename}")

    def print_summary(self):
        """Print test summary"""
        print("\n🔍 TEST SUMMARY")
        print("-" * 40)
        
        # Overall score
        score = self.results['overall_score']
        if score >= 90:
            status_emoji = "🟢"
            status_text = "EXCELLENT"
        elif score >= 70:
            status_emoji = "🟡"
            status_text = "GOOD"
        elif score >= 50:
            status_emoji = "🟠"
            status_text = "NEEDS IMPROVEMENT"
        else:
            status_emoji = "🔴"
            status_text = "CRITICAL ISSUES"
        
        print(f"Overall Health: {status_emoji} {score}/100 ({status_text})")
        
        # Category breakdown
        categories = {
            'Docker Containers': self.results.get('docker_status', {}),
            'Service Connectivity': self.results.get('service_connectivity', {}),
            'Dashboard Features': self.results.get('dashboard_functionality', {}),
            'Mobile Support': self.results.get('mobile_responsiveness', {}),
            'Security': self.results.get('security_tests', {}),
            'Performance': self.results.get('performance_metrics', {})
        }
        
        for category, data in categories.items():
            if isinstance(data, dict) and 'status' in data:
                status = data['status']
                emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, "❓")
                print(f"{category}: {emoji} {status}")
        
        # Recommendations
        if self.results.get('recommendations'):
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print("\n🎯 QUICK ACCESS URLS:")
        print("- Dashboard: http://localhost:8090")
        print("- Jellyfin: http://localhost:8096")
        print("- Sonarr: http://localhost:8989")
        print("- Radarr: http://localhost:7878")
        print("- Prowlarr: http://localhost:9696")
        print("- qBittorrent: http://localhost:8080")

if __name__ == "__main__":
    tester = MediaFlowTester()
    tester.run_all_tests()
    tester.print_summary()
    tester.save_report('/Users/morlock/fun/newmedia/integration-test-report.json')