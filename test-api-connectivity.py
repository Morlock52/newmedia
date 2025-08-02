#!/usr/bin/env python3
"""
Ultimate Media Server 2025 - API Connectivity Test Suite
Tests all service APIs, authentication methods, endpoints, and integration points
"""

import asyncio
import aiohttp
import json
import os
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import yaml
import base64
import hashlib
import ssl
import certifi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test-results/api-connectivity-{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for a service API"""
    name: str
    base_url: str
    port: int
    auth_type: str  # none, api_key, basic, bearer, custom
    health_endpoint: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    verify_ssl: bool = True
    timeout: int = 30
    headers: Dict[str, str] = None
    test_endpoints: List[str] = None

@dataclass
class TestResult:
    """Result of an API test"""
    service: str
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    response_data: Optional[Dict] = None

class APITester:
    """Main API testing class"""
    
    def __init__(self, config_file: str = None):
        self.services = {}
        self.results = []
        self.session = None
        self.config_file = config_file or "test-config/api-test-config.yaml"
        
        # Ensure test results directory exists
        os.makedirs("test-results", exist_ok=True)
        
        # Load configuration
        self.load_config()
    
    def load_config(self):
        """Load service configurations"""
        # Default configurations for Ultimate Media Server 2025
        default_configs = {
            "jellyfin": ServiceConfig(
                name="Jellyfin",
                base_url="http://localhost",
                port=8096,
                auth_type="api_key",
                health_endpoint="/health",
                api_key=os.getenv("JELLYFIN_API_KEY"),
                test_endpoints=[
                    "/System/Info",
                    "/Users",
                    "/Items",
                    "/Sessions"
                ]
            ),
            "plex": ServiceConfig(
                name="Plex",
                base_url="http://localhost",
                port=32400,
                auth_type="bearer",
                health_endpoint="/identity",
                api_key=os.getenv("PLEX_TOKEN"),
                test_endpoints=[
                    "/",
                    "/library/sections",
                    "/status/sessions",
                    "/servers"
                ]
            ),
            "emby": ServiceConfig(
                name="Emby",
                base_url="http://localhost",
                port=8097,
                auth_type="api_key",
                health_endpoint="/health",
                api_key=os.getenv("EMBY_API_KEY"),
                test_endpoints=[
                    "/System/Info",
                    "/Users",
                    "/Items"
                ]
            ),
            "sonarr": ServiceConfig(
                name="Sonarr",
                base_url="http://localhost",
                port=8989,
                auth_type="api_key",
                health_endpoint="/api/v3/health",
                api_key=os.getenv("SONARR_API_KEY"),
                test_endpoints=[
                    "/api/v3/system/status",
                    "/api/v3/series",
                    "/api/v3/queue",
                    "/api/v3/calendar"
                ]
            ),
            "radarr": ServiceConfig(
                name="Radarr",
                base_url="http://localhost",
                port=7878,
                auth_type="api_key",
                health_endpoint="/api/v3/health",
                api_key=os.getenv("RADARR_API_KEY"),
                test_endpoints=[
                    "/api/v3/system/status",
                    "/api/v3/movie",
                    "/api/v3/queue",
                    "/api/v3/calendar"
                ]
            ),
            "lidarr": ServiceConfig(
                name="Lidarr",
                base_url="http://localhost",
                port=8686,
                auth_type="api_key",
                health_endpoint="/api/v1/health",
                api_key=os.getenv("LIDARR_API_KEY"),
                test_endpoints=[
                    "/api/v1/system/status",
                    "/api/v1/artist",
                    "/api/v1/queue"
                ]
            ),
            "readarr": ServiceConfig(
                name="Readarr",
                base_url="http://localhost",
                port=8787,
                auth_type="api_key",
                health_endpoint="/api/v1/health",
                api_key=os.getenv("READARR_API_KEY"),
                test_endpoints=[
                    "/api/v1/system/status",
                    "/api/v1/author",
                    "/api/v1/queue"
                ]
            ),
            "bazarr": ServiceConfig(
                name="Bazarr",
                base_url="http://localhost",
                port=6767,
                auth_type="api_key",
                health_endpoint="/api/system/health",
                api_key=os.getenv("BAZARR_API_KEY"),
                test_endpoints=[
                    "/api/system/status",
                    "/api/series",
                    "/api/movies"
                ]
            ),
            "prowlarr": ServiceConfig(
                name="Prowlarr",
                base_url="http://localhost",
                port=9696,
                auth_type="api_key",
                health_endpoint="/api/v1/health",
                api_key=os.getenv("PROWLARR_API_KEY"),
                test_endpoints=[
                    "/api/v1/system/status",
                    "/api/v1/indexer",
                    "/api/v1/indexerstats"
                ]
            ),
            "jellyseerr": ServiceConfig(
                name="Jellyseerr",
                base_url="http://localhost",
                port=5055,
                auth_type="api_key",
                health_endpoint="/api/v1/status",
                api_key=os.getenv("JELLYSEERR_API_KEY"),
                test_endpoints=[
                    "/api/v1/status",
                    "/api/v1/settings",
                    "/api/v1/request"
                ]
            ),
            "overseerr": ServiceConfig(
                name="Overseerr",
                base_url="http://localhost",
                port=5056,
                auth_type="api_key",
                health_endpoint="/api/v1/status",
                api_key=os.getenv("OVERSEERR_API_KEY"),
                test_endpoints=[
                    "/api/v1/status",
                    "/api/v1/settings",
                    "/api/v1/request"
                ]
            ),
            "ombi": ServiceConfig(
                name="Ombi",
                base_url="http://localhost",
                port=3579,
                auth_type="api_key",
                health_endpoint="/api/v1/Status",
                api_key=os.getenv("OMBI_API_KEY"),
                test_endpoints=[
                    "/api/v1/Status",
                    "/api/v1/Settings",
                    "/api/v1/Request/movie"
                ]
            ),
            "qbittorrent": ServiceConfig(
                name="qBittorrent",
                base_url="http://localhost",
                port=8080,
                auth_type="basic",
                health_endpoint="/api/v2/app/version",
                username=os.getenv("QBITTORRENT_USERNAME", "admin"),
                password=os.getenv("QBITTORRENT_PASSWORD", "adminpass"),
                test_endpoints=[
                    "/api/v2/app/version",
                    "/api/v2/torrents/info",
                    "/api/v2/app/preferences"
                ]
            ),
            "sabnzbd": ServiceConfig(
                name="SABnzbd",
                base_url="http://localhost",
                port=8081,
                auth_type="api_key",
                health_endpoint="/sabnzbd/api",
                api_key=os.getenv("SABNZBD_API_KEY"),
                test_endpoints=[
                    "/sabnzbd/api?mode=version",
                    "/sabnzbd/api?mode=queue",
                    "/sabnzbd/api?mode=history"
                ]
            ),
            "nzbget": ServiceConfig(
                name="NZBGet",
                base_url="http://localhost",
                port=6789,
                auth_type="basic",
                health_endpoint="/jsonrpc",
                username=os.getenv("NZBGET_USERNAME", "nzbget"),
                password=os.getenv("NZBGET_PASSWORD", "tegbzn6789"),
                test_endpoints=[
                    "/jsonrpc"
                ]
            ),
            "prometheus": ServiceConfig(
                name="Prometheus",
                base_url="http://localhost",
                port=9090,
                auth_type="none",
                health_endpoint="/-/healthy",
                test_endpoints=[
                    "/-/healthy",
                    "/api/v1/targets",
                    "/api/v1/query?query=up"
                ]
            ),
            "grafana": ServiceConfig(
                name="Grafana",
                base_url="http://localhost",
                port=3000,
                auth_type="basic",
                health_endpoint="/api/health",
                username=os.getenv("GRAFANA_USERNAME", "admin"),
                password=os.getenv("GRAFANA_PASSWORD", "admin"),
                test_endpoints=[
                    "/api/health",
                    "/api/dashboards/home",
                    "/api/datasources"
                ]
            ),
            "loki": ServiceConfig(
                name="Loki",
                base_url="http://localhost",
                port=3100,
                auth_type="none",
                health_endpoint="/ready",
                test_endpoints=[
                    "/ready",
                    "/metrics",
                    "/loki/api/v1/labels"
                ]
            ),
            "uptime_kuma": ServiceConfig(
                name="Uptime Kuma",
                base_url="http://localhost",
                port=3001,
                auth_type="none",
                health_endpoint="/",
                test_endpoints=[
                    "/",
                    "/api/status-page/heartbeat"
                ]
            ),
            "portainer": ServiceConfig(
                name="Portainer",
                base_url="https://localhost",
                port=9443,
                auth_type="bearer",
                health_endpoint="/api/status",
                verify_ssl=False,
                test_endpoints=[
                    "/api/status",
                    "/api/endpoints"
                ]
            ),
            "homepage": ServiceConfig(
                name="Homepage",
                base_url="http://localhost",
                port=3003,
                auth_type="none",
                health_endpoint="/",
                test_endpoints=[
                    "/",
                    "/api/config"
                ]
            ),
            "homarr": ServiceConfig(
                name="Homarr",
                base_url="http://localhost",
                port=7575,
                auth_type="none",
                health_endpoint="/",
                test_endpoints=[
                    "/",
                    "/api/configs"
                ]
            )
        }
        
        # Try to load from config file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    
                for service_name, config_data in file_config.get('services', {}).items():
                    if service_name in default_configs:
                        # Update default config with file config
                        service_config = default_configs[service_name]
                        for key, value in config_data.items():
                            if hasattr(service_config, key):
                                setattr(service_config, key, value)
                    else:
                        # Create new service config
                        default_configs[service_name] = ServiceConfig(**config_data)
                        
            except Exception as e:
                logger.warning(f"Could not load config file {self.config_file}: {e}")
        
        self.services = default_configs
        logger.info(f"Loaded {len(self.services)} service configurations")
    
    async def create_session(self):
        """Create aiohttp session with SSL configuration"""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        timeout = aiohttp.ClientTimeout(total=60, connect=30)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(ssl=ssl_context)
        )
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    def prepare_auth_headers(self, service_config: ServiceConfig) -> Dict[str, str]:
        """Prepare authentication headers based on service configuration"""
        headers = service_config.headers.copy() if service_config.headers else {}
        
        if service_config.auth_type == "api_key":
            if service_config.name in ["Sonarr", "Radarr", "Lidarr", "Readarr", "Prowlarr"]:
                headers["X-Api-Key"] = service_config.api_key
            elif service_config.name == "Jellyfin":
                headers["X-Emby-Authorization"] = f'MediaBrowser Token="{service_config.api_key}"'
            elif service_config.name == "Jellyseerr":
                headers["X-Api-Key"] = service_config.api_key
            elif service_config.name == "Overseerr":
                headers["X-Api-Key"] = service_config.api_key
            elif service_config.name == "Ombi":
                headers["ApiKey"] = service_config.api_key
            elif service_config.name == "SABnzbd":
                # SABnzbd uses API key in URL parameters
                pass
            else:
                headers["Authorization"] = f"Bearer {service_config.api_key}"
                
        elif service_config.auth_type == "bearer":
            if service_config.name == "Plex":
                headers["X-Plex-Token"] = service_config.api_key
            else:
                headers["Authorization"] = f"Bearer {service_config.api_key}"
                
        elif service_config.auth_type == "basic":
            if service_config.username and service_config.password:
                credentials = base64.b64encode(
                    f"{service_config.username}:{service_config.password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        return headers
    
    def prepare_url(self, service_config: ServiceConfig, endpoint: str) -> str:
        """Prepare full URL for endpoint"""
        base_url = f"{service_config.base_url}:{service_config.port}"
        
        # Handle special cases
        if service_config.name == "SABnzbd" and service_config.api_key:
            if "?" in endpoint:
                endpoint += f"&apikey={service_config.api_key}&output=json"
            else:
                endpoint += f"?apikey={service_config.api_key}&output=json"
        
        return f"{base_url}{endpoint}"
    
    async def test_endpoint(self, service_config: ServiceConfig, endpoint: str, method: str = "GET") -> TestResult:
        """Test a specific endpoint"""
        start_time = time.time()
        
        try:
            url = self.prepare_url(service_config, endpoint)
            headers = self.prepare_auth_headers(service_config)
            
            ssl_verify = service_config.verify_ssl
            
            async with self.session.request(
                method,
                url,
                headers=headers,
                ssl=ssl_verify,
                timeout=aiohttp.ClientTimeout(total=service_config.timeout)
            ) as response:
                response_time = time.time() - start_time
                
                # Try to parse JSON response
                try:
                    response_data = await response.json()
                except:
                    response_data = {"text": await response.text()[:500]}
                
                success = 200 <= response.status < 400
                
                result = TestResult(
                    service=service_config.name,
                    endpoint=endpoint,
                    method=method,
                    status_code=response.status,
                    response_time=response_time,
                    success=success,
                    response_data=response_data if success else None,
                    error_message=None if success else f"HTTP {response.status}: {response.reason}"
                )
                
                return result
                
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            return TestResult(
                service=service_config.name,
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service=service_config.name,
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
    
    async def test_service(self, service_name: str, quick_test: bool = False) -> List[TestResult]:
        """Test all endpoints for a service"""
        service_config = self.services.get(service_name)
        if not service_config:
            logger.error(f"Service {service_name} not found in configuration")
            return []
        
        logger.info(f"Testing service: {service_config.name}")
        
        results = []
        
        # Always test health endpoint first
        health_result = await self.test_endpoint(service_config, service_config.health_endpoint)
        results.append(health_result)
        
        if health_result.success:
            logger.info(f"✓ {service_config.name} health check passed")
            
            # Test additional endpoints if not quick test
            if not quick_test and service_config.test_endpoints:
                for endpoint in service_config.test_endpoints:
                    if endpoint != service_config.health_endpoint:  # Avoid duplicate
                        result = await self.test_endpoint(service_config, endpoint)
                        results.append(result)
                        
                        if result.success:
                            logger.info(f"✓ {service_config.name} {endpoint} - {result.response_time:.3f}s")
                        else:
                            logger.error(f"✗ {service_config.name} {endpoint} - {result.error_message}")
        else:
            logger.error(f"✗ {service_config.name} health check failed: {health_result.error_message}")
        
        return results
    
    async def test_integration_endpoints(self) -> List[TestResult]:
        """Test integration between services"""
        logger.info("Testing service integrations")
        
        integration_results = []
        
        # Test ARR service integrations
        arr_services = ["sonarr", "radarr", "lidarr", "readarr"]
        download_clients = ["qbittorrent", "sabnzbd"]
        
        for arr_service in arr_services:
            if arr_service in self.services:
                arr_config = self.services[arr_service]
                
                # Test download client connections
                for client in download_clients:
                    if client in self.services:
                        endpoint = f"/api/v3/downloadclient" if "v3" in arr_config.health_endpoint else f"/api/v1/downloadclient"
                        result = await self.test_endpoint(arr_config, endpoint)
                        integration_results.append(result)
        
        # Test indexer integrations (Prowlarr)
        if "prowlarr" in self.services:
            prowlarr_config = self.services["prowlarr"]
            
            # Test applications (ARR services connected to Prowlarr)
            apps_result = await self.test_endpoint(prowlarr_config, "/api/v1/applications")
            integration_results.append(apps_result)
            
            # Test indexers
            indexers_result = await self.test_endpoint(prowlarr_config, "/api/v1/indexer")
            integration_results.append(indexers_result)
        
        # Test media server integrations
        media_servers = ["jellyfin", "plex", "emby"]
        request_services = ["jellyseerr", "overseerr", "ombi"]
        
        for request_service in request_services:
            if request_service in self.services:
                request_config = self.services[request_service]
                
                # Test media server connections
                settings_result = await self.test_endpoint(request_config, "/api/v1/settings/jellyfin" if "jellyseerr" in request_service else "/api/v1/settings/plex")
                integration_results.append(settings_result)
        
        return integration_results
    
    async def run_tests(self, services: List[str] = None, quick_test: bool = False, include_integrations: bool = True) -> Dict[str, Any]:
        """Run all API tests"""
        logger.info("Starting API connectivity tests")
        
        await self.create_session()
        
        try:
            # Determine which services to test
            test_services = services if services else list(self.services.keys())
            
            # Test each service
            all_results = []
            for service_name in test_services:
                if service_name in self.services:
                    service_results = await self.test_service(service_name, quick_test)
                    all_results.extend(service_results)
                else:
                    logger.warning(f"Service {service_name} not found, skipping")
            
            # Test integrations
            if include_integrations and not quick_test:
                integration_results = await self.test_integration_endpoints()
                all_results.extend(integration_results)
            
            # Analyze results
            self.results = all_results
            report = self.generate_report()
            
            return report
            
        finally:
            await self.close_session()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Group results by service
        service_results = {}
        for result in self.results:
            if result.service not in service_results:
                service_results[result.service] = []
            service_results[result.service].append(result)
        
        # Calculate service statistics
        service_stats = {}
        for service, results in service_results.items():
            service_total = len(results)
            service_passed = sum(1 for r in results if r.success)
            service_stats[service] = {
                "total_tests": service_total,
                "passed_tests": service_passed,
                "failed_tests": service_total - service_passed,
                "success_rate": (service_passed / service_total) * 100 if service_total > 0 else 0,
                "avg_response_time": sum(r.response_time for r in results) / service_total if service_total > 0 else 0,
                "endpoints_tested": [r.endpoint for r in results]
            }
        
        # Identify failures
        failures = [
            {
                "service": r.service,
                "endpoint": r.endpoint,
                "error": r.error_message,
                "status_code": r.status_code
            }
            for r in self.results if not r.success
        ]
        
        # Performance analysis
        response_times = [r.response_time for r in self.results if r.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "test_suite": "API Connectivity Tests",
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": round(success_rate, 2),
                "avg_response_time": round(avg_response_time, 3),
                "max_response_time": round(max_response_time, 3),
                "min_response_time": round(min_response_time, 3)
            },
            "service_statistics": service_stats,
            "failures": failures,
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        # Save report
        report_file = f"test-results/api-connectivity-report-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {report_file}")
        
        return report

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Ultimate Media Server 2025 - API Connectivity Tests")
    parser.add_argument("--services", nargs="+", help="Specific services to test")
    parser.add_argument("--quick", action="store_true", help="Run quick health checks only")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--no-integrations", action="store_true", help="Skip integration tests")
    parser.add_argument("--output", help="Output format (json|text)", default="text")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = APITester(config_file=args.config)
    
    # Run tests
    try:
        report = await tester.run_tests(
            services=args.services,
            quick_test=args.quick,
            include_integrations=not args.no_integrations
        )
        
        # Display results
        if args.output == "json":
            print(json.dumps(report, indent=2))
        else:
            # Text output
            print("\n" + "="*60)
            print("🌐 Ultimate Media Server 2025 - API Connectivity Tests")
            print("="*60)
            
            summary = report["summary"]
            print(f"\nTest Summary:")
            print(f"  Total Tests: {summary['total_tests']}")
            print(f"  ✅ Passed: {summary['passed_tests']}")
            print(f"  ❌ Failed: {summary['failed_tests']}")
            print(f"  Success Rate: {summary['success_rate']}%")
            print(f"  Avg Response Time: {summary['avg_response_time']}s")
            
            # Service breakdown
            print(f"\nService Breakdown:")
            for service, stats in report["service_statistics"].items():
                status = "✅" if stats["success_rate"] == 100 else "⚠️" if stats["success_rate"] > 50 else "❌"
                print(f"  {status} {service}: {stats['passed_tests']}/{stats['total_tests']} ({stats['success_rate']:.1f}%) - {stats['avg_response_time']:.3f}s avg")
            
            # Failures
            if report["failures"]:
                print(f"\nFailures:")
                for failure in report["failures"]:
                    print(f"  ❌ {failure['service']} {failure['endpoint']}: {failure['error']}")
            
            print(f"\nDetailed report saved to test-results/")
        
        # Exit with appropriate code
        exit_code = 0 if report["summary"]["failed_tests"] == 0 else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())