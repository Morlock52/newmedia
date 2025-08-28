#!/usr/bin/env python3
"""
Service Dependency Validator
Analyzes and validates the dependency chain between all media server services
Ensures proper startup order and dependency satisfaction

Features:
- Dependency graph analysis
- Circular dependency detection
- Startup order optimization
- Service health impact analysis
- Real-time dependency monitoring
"""

import asyncio
import aiohttp
import json
import time
import argparse
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    name: str
    port: int
    health_endpoint: Optional[str]
    dependencies: List[str]
    tier: str  # critical, high, medium, optional
    startup_time_seconds: int
    max_startup_time: int
    process_name: str
    expected_memory_mb: int

@dataclass
class ServiceStatus:
    name: str
    is_running: bool
    port_listening: bool
    health_check_status: str
    response_time_ms: int
    last_check: datetime
    dependency_status: Dict[str, bool]
    startup_time_actual: Optional[int] = None

class ServiceDependencyValidator:
    def __init__(self, container_name: Optional[str] = None):
        self.container_name = container_name
        self.services = self._initialize_services()
        self.dependency_graph = nx.DiGraph()
        self.service_status: Dict[str, ServiceStatus] = {}
        self.monitoring = False
        
    def _initialize_services(self) -> Dict[str, ServiceConfig]:
        """Initialize service configurations with dependencies"""
        services = {
            # Database tier - Foundation services (no dependencies)
            'postgres': ServiceConfig(
                name='postgres',
                port=5432,
                health_endpoint=None,
                dependencies=[],
                tier='critical',
                startup_time_seconds=15,
                max_startup_time=30,
                process_name='postgres',
                expected_memory_mb=256
            ),
            'redis': ServiceConfig(
                name='redis',
                port=6379,
                health_endpoint=None,
                dependencies=[],
                tier='critical',
                startup_time_seconds=5,
                max_startup_time=15,
                process_name='redis-server',
                expected_memory_mb=128
            ),
            'mariadb': ServiceConfig(
                name='mariadb',
                port=3306,
                health_endpoint=None,
                dependencies=[],
                tier='critical',
                startup_time_seconds=20,
                max_startup_time=45,
                process_name='mysqld',
                expected_memory_mb=256
            ),
            
            # Media Servers - Core functionality
            'jellyfin': ServiceConfig(
                name='jellyfin',
                port=8096,
                health_endpoint='/health',
                dependencies=[],
                tier='critical',
                startup_time_seconds=30,
                max_startup_time=60,
                process_name='jellyfin',
                expected_memory_mb=512
            ),
            'plex': ServiceConfig(
                name='plex',
                port=32400,
                health_endpoint='/identity',
                dependencies=[],
                tier='critical',
                startup_time_seconds=45,
                max_startup_time=90,
                process_name='plex',
                expected_memory_mb=1024
            ),
            'emby': ServiceConfig(
                name='emby',
                port=8097,
                health_endpoint='/System/Info/Public',
                dependencies=[],
                tier='optional',
                startup_time_seconds=30,
                max_startup_time=60,
                process_name='emby',
                expected_memory_mb=512
            ),
            
            # Indexer - Must start before *ARR services
            'prowlarr': ServiceConfig(
                name='prowlarr',
                port=9696,
                health_endpoint='/ping',
                dependencies=[],
                tier='high',
                startup_time_seconds=20,
                max_startup_time=40,
                process_name='prowlarr',
                expected_memory_mb=256
            ),
            
            # *ARR Stack - Depends on indexer
            'sonarr': ServiceConfig(
                name='sonarr',
                port=8989,
                health_endpoint='/ping',
                dependencies=['prowlarr'],
                tier='high',
                startup_time_seconds=25,
                max_startup_time=50,
                process_name='sonarr',
                expected_memory_mb=256
            ),
            'radarr': ServiceConfig(
                name='radarr',
                port=7878,
                health_endpoint='/ping',
                dependencies=['prowlarr'],
                tier='high',
                startup_time_seconds=25,
                max_startup_time=50,
                process_name='radarr',
                expected_memory_mb=256
            ),
            'lidarr': ServiceConfig(
                name='lidarr',
                port=8686,
                health_endpoint='/ping',
                dependencies=['prowlarr'],
                tier='medium',
                startup_time_seconds=20,
                max_startup_time=40,
                process_name='lidarr',
                expected_memory_mb=200
            ),
            'readarr': ServiceConfig(
                name='readarr',
                port=8787,
                health_endpoint='/ping',
                dependencies=['prowlarr'],
                tier='medium',
                startup_time_seconds=20,
                max_startup_time=40,
                process_name='readarr',
                expected_memory_mb=200
            ),
            'bazarr': ServiceConfig(
                name='bazarr',
                port=6767,
                health_endpoint='/system/status',
                dependencies=['sonarr', 'radarr'],
                tier='medium',
                startup_time_seconds=15,
                max_startup_time=30,
                process_name='bazarr',
                expected_memory_mb=150
            ),
            
            # Download Clients - Independent
            'qbittorrent': ServiceConfig(
                name='qbittorrent',
                port=8080,
                health_endpoint='/api/v2/app/version',
                dependencies=[],
                tier='high',
                startup_time_seconds=15,
                max_startup_time=30,
                process_name='qbittorrent-nox',
                expected_memory_mb=128
            ),
            'transmission': ServiceConfig(
                name='transmission',
                port=9091,
                health_endpoint='/transmission/rpc',
                dependencies=[],
                tier='medium',
                startup_time_seconds=10,
                max_startup_time=20,
                process_name='transmission-daemon',
                expected_memory_mb=64
            ),
            'sabnzbd': ServiceConfig(
                name='sabnzbd',
                port=8081,
                health_endpoint='/api',
                dependencies=[],
                tier='medium',
                startup_time_seconds=15,
                max_startup_time=30,
                process_name='sabnzbd',
                expected_memory_mb=128
            ),
            
            # Request Management - Depends on media servers and *ARR
            'overseerr': ServiceConfig(
                name='overseerr',
                port=5055,
                health_endpoint='/api/v1/status',
                dependencies=['plex', 'sonarr', 'radarr'],
                tier='medium',
                startup_time_seconds=30,
                max_startup_time=60,
                process_name='overseerr',
                expected_memory_mb=256
            ),
            'jellyseerr': ServiceConfig(
                name='jellyseerr',
                port=5056,
                health_endpoint='/api/v1/status',
                dependencies=['jellyfin', 'sonarr', 'radarr'],
                tier='medium',
                startup_time_seconds=30,
                max_startup_time=60,
                process_name='jellyseerr',
                expected_memory_mb=256
            ),
            
            # Content Libraries - Some depend on databases
            'photoprism': ServiceConfig(
                name='photoprism',
                port=2342,
                health_endpoint='/api/v1/status',
                dependencies=['mariadb'],
                tier='optional',
                startup_time_seconds=45,
                max_startup_time=90,
                process_name='photoprism',
                expected_memory_mb=512
            ),
            'nextcloud': ServiceConfig(
                name='nextcloud',
                port=8084,
                health_endpoint='/status.php',
                dependencies=['postgres', 'redis'],
                tier='optional',
                startup_time_seconds=60,
                max_startup_time=120,
                process_name='apache2',
                expected_memory_mb=512
            ),
            
            # Monitoring - Some depend on databases
            'grafana': ServiceConfig(
                name='grafana',
                port=3000,
                health_endpoint='/api/health',
                dependencies=['prometheus'],
                tier='high',
                startup_time_seconds=25,
                max_startup_time=50,
                process_name='grafana-server',
                expected_memory_mb=256
            ),
            'prometheus': ServiceConfig(
                name='prometheus',
                port=9090,
                health_endpoint='/-/healthy',
                dependencies=[],
                tier='critical',
                startup_time_seconds=20,
                max_startup_time=40,
                process_name='prometheus',
                expected_memory_mb=256
            ),
            
            # Management Tools
            'tautulli': ServiceConfig(
                name='tautulli',
                port=8181,
                health_endpoint='/api/v2',
                dependencies=['plex'],
                tier='medium',
                startup_time_seconds=20,
                max_startup_time=40,
                process_name='tautulli',
                expected_memory_mb=128
            )
        }
        
        return services
    
    def build_dependency_graph(self):
        """Build a directed graph representing service dependencies"""
        logger.info("Building dependency graph...")
        
        # Add all services as nodes
        for service_name, config in self.services.items():
            self.dependency_graph.add_node(
                service_name,
                tier=config.tier,
                startup_time=config.startup_time_seconds,
                port=config.port
            )
        
        # Add dependency edges
        for service_name, config in self.services.items():
            for dependency in config.dependencies:
                if dependency in self.services:
                    self.dependency_graph.add_edge(dependency, service_name)
                else:
                    logger.warning(f"Service {service_name} depends on unknown service: {dependency}")
        
        logger.info(f"Dependency graph built: {self.dependency_graph.number_of_nodes()} nodes, "
                   f"{self.dependency_graph.number_of_edges()} edges")
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the service graph"""
        logger.info("Checking for circular dependencies...")
        
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            if cycles:
                logger.error(f"Found {len(cycles)} circular dependencies:")
                for cycle in cycles:
                    logger.error(f"  Cycle: {' -> '.join(cycle + [cycle[0]])}")
            else:
                logger.info("No circular dependencies detected")
            
            return cycles
        except Exception as e:
            logger.error(f"Error detecting circular dependencies: {e}")
            return []
    
    def calculate_optimal_startup_order(self) -> List[str]:
        """Calculate optimal service startup order based on dependencies"""
        logger.info("Calculating optimal startup order...")
        
        try:
            # Topological sort gives us the dependency order
            startup_order = list(nx.topological_sort(self.dependency_graph))
            
            # Group by dependency level for parallel startup
            levels = {}
            for node in startup_order:
                # Calculate the longest path to this node (dependency depth)
                try:
                    paths_to_node = nx.single_source_shortest_path_length(
                        self.dependency_graph.reverse(), node
                    )
                    level = max(paths_to_node.values()) if paths_to_node else 0
                except:
                    level = 0
                
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)
            
            logger.info("Optimal startup order by level:")
            for level, services in sorted(levels.items()):
                logger.info(f"  Level {level}: {', '.join(services)}")
            
            return startup_order, levels
        except nx.NetworkXError as e:
            logger.error(f"Cannot calculate startup order due to cycles: {e}")
            return [], {}
    
    def analyze_dependency_impact(self, failed_service: str) -> Dict[str, List[str]]:
        """Analyze the impact of a failed service on dependent services"""
        if failed_service not in self.dependency_graph:
            return {}
        
        # Find all services that depend on this service (directly or indirectly)
        dependent_services = nx.descendants(self.dependency_graph, failed_service)
        
        # Categorize impact by tier
        impact = defaultdict(list)
        for service in dependent_services:
            tier = self.services[service].tier
            impact[tier].append(service)
        
        logger.info(f"Impact analysis for failed service '{failed_service}':")
        for tier, services in impact.items():
            logger.info(f"  {tier.upper()} tier affected: {', '.join(services)}")
        
        return dict(impact)
    
    async def check_service_health(self, service: ServiceConfig) -> ServiceStatus:
        """Check the health of a single service"""
        start_time = time.time()
        
        # Check if port is listening
        port_listening = await self._check_port(service.port)
        
        # Check process running
        is_running = await self._check_process(service.process_name)
        
        # Check health endpoint if available
        health_status = "unknown"
        response_time_ms = 0
        
        if port_listening and service.health_endpoint:
            health_status, response_time_ms = await self._check_health_endpoint(
                service.port, service.health_endpoint
            )
        elif port_listening:
            health_status = "port_only"
        
        # Check dependencies
        dependency_status = {}
        for dep_name in service.dependencies:
            if dep_name in self.service_status:
                dependency_status[dep_name] = (
                    self.service_status[dep_name].is_running and 
                    self.service_status[dep_name].port_listening
                )
            else:
                dependency_status[dep_name] = False
        
        return ServiceStatus(
            name=service.name,
            is_running=is_running,
            port_listening=port_listening,
            health_check_status=health_status,
            response_time_ms=response_time_ms,
            last_check=datetime.now(),
            dependency_status=dependency_status
        )
    
    async def _check_port(self, port: int) -> bool:
        """Check if a port is listening"""
        try:
            if self.container_name:
                # Check inside container
                result = subprocess.run([
                    'docker', 'exec', self.container_name,
                    'netstat', '-tulpn'
                ], capture_output=True, text=True, timeout=10)
                
                return f":{port} " in result.stdout
            else:
                # Check local port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    result = s.connect_ex(('localhost', port))
                    return result == 0
        except:
            return False
    
    async def _check_process(self, process_name: str) -> bool:
        """Check if a process is running"""
        try:
            if self.container_name:
                result = subprocess.run([
                    'docker', 'exec', self.container_name,
                    'pgrep', '-f', process_name
                ], capture_output=True, timeout=10)
                
                return result.returncode == 0
            else:
                result = subprocess.run([
                    'pgrep', '-f', process_name
                ], capture_output=True, timeout=10)
                
                return result.returncode == 0
        except:
            return False
    
    async def _check_health_endpoint(self, port: int, endpoint: str) -> Tuple[str, int]:
        """Check HTTP health endpoint"""
        url = f"http://localhost:{port}{endpoint}"
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    response_time = int((time.time() - start_time) * 1000)
                    
                    if response.status in [200, 201, 202]:
                        return "healthy", response_time
                    elif response.status == 409 and endpoint == '/transmission/rpc':
                        # Special case for Transmission RPC
                        return "healthy", response_time
                    else:
                        return f"unhealthy_http_{response.status}", response_time
        except asyncio.TimeoutError:
            return "timeout", int((time.time() - start_time) * 1000)
        except Exception as e:
            return f"error_{type(e).__name__}", int((time.time() - start_time) * 1000)
    
    async def validate_all_dependencies(self) -> Dict[str, any]:
        """Validate all service dependencies"""
        logger.info("Validating all service dependencies...")
        
        # Check all services in parallel
        tasks = []
        for service in self.services.values():
            tasks.append(self.check_service_health(service))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update service status
        for result in results:
            if isinstance(result, ServiceStatus):
                self.service_status[result.name] = result
        
        # Analyze results
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_services': len(self.services),
            'running_services': sum(1 for s in self.service_status.values() if s.is_running),
            'healthy_services': sum(1 for s in self.service_status.values() 
                                  if s.health_check_status in ['healthy', 'port_only']),
            'dependency_violations': [],
            'service_details': {},
            'tier_analysis': {},
            'critical_path_analysis': {}
        }
        
        # Check dependency violations
        for service_name, status in self.service_status.items():
            service_config = self.services[service_name]
            
            # Check if all dependencies are satisfied
            unsatisfied_deps = []
            for dep_name, dep_status in status.dependency_status.items():
                if not dep_status:
                    unsatisfied_deps.append(dep_name)
            
            if unsatisfied_deps:
                validation_results['dependency_violations'].append({
                    'service': service_name,
                    'tier': service_config.tier,
                    'unsatisfied_dependencies': unsatisfied_deps
                })
            
            # Store service details
            validation_results['service_details'][service_name] = asdict(status)
        
        # Tier analysis
        tiers = defaultdict(lambda: {'total': 0, 'running': 0, 'healthy': 0})
        for service_name, status in self.service_status.items():
            tier = self.services[service_name].tier
            tiers[tier]['total'] += 1
            if status.is_running:
                tiers[tier]['running'] += 1
            if status.health_check_status in ['healthy', 'port_only']:
                tiers[tier]['healthy'] += 1
        
        validation_results['tier_analysis'] = dict(tiers)
        
        # Critical path analysis
        validation_results['critical_path_analysis'] = self._analyze_critical_paths()
        
        return validation_results
    
    def _analyze_critical_paths(self) -> Dict[str, any]:
        """Analyze critical paths in the dependency graph"""
        critical_paths = {}
        
        # Find services with no dependencies (roots)
        root_services = [node for node in self.dependency_graph.nodes() 
                        if self.dependency_graph.in_degree(node) == 0]
        
        # Find services with no dependents (leaves)
        leaf_services = [node for node in self.dependency_graph.nodes() 
                        if self.dependency_graph.out_degree(node) == 0]
        
        # Calculate longest paths
        longest_paths = []
        for root in root_services:
            for leaf in leaf_services:
                try:
                    path = nx.shortest_path(self.dependency_graph, root, leaf)
                    if len(path) > 1:
                        # Calculate total startup time for this path
                        total_time = sum(self.services[service].startup_time_seconds 
                                       for service in path)
                        longest_paths.append({
                            'path': path,
                            'length': len(path),
                            'total_startup_time': total_time
                        })
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by length and startup time
        longest_paths.sort(key=lambda x: (x['length'], x['total_startup_time']), reverse=True)
        
        return {
            'root_services': root_services,
            'leaf_services': leaf_services,
            'longest_paths': longest_paths[:5],  # Top 5 longest paths
            'max_startup_time': max([p['total_startup_time'] for p in longest_paths]) if longest_paths else 0
        }
    
    def visualize_dependency_graph(self, output_file: str = "dependency_graph.png"):
        """Create a visual representation of the dependency graph"""
        logger.info(f"Generating dependency graph visualization: {output_file}")
        
        plt.figure(figsize=(20, 16))
        
        # Create layout
        pos = nx.spring_layout(self.dependency_graph, k=3, iterations=50)
        
        # Color nodes by tier
        tier_colors = {
            'critical': '#ff4444',
            'high': '#ff8800',
            'medium': '#ffcc00',
            'optional': '#88cc88'
        }
        
        node_colors = []
        for node in self.dependency_graph.nodes():
            tier = self.services[node].tier
            node_colors.append(tier_colors.get(tier, '#cccccc'))
        
        # Draw the graph
        nx.draw(self.dependency_graph, pos, 
                node_color=node_colors,
                node_size=3000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                edge_color='#666666',
                with_labels=True)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=tier.capitalize()) 
                          for tier, color in tier_colors.items()]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.title("Service Dependency Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dependency graph saved to {output_file}")
    
    def generate_startup_script(self, output_file: str = "optimal_startup.sh"):
        """Generate optimized startup script based on dependency analysis"""
        startup_order, levels = self.calculate_optimal_startup_order()
        
        if not levels:
            logger.error("Cannot generate startup script due to dependency issues")
            return
        
        script_content = """#!/bin/bash
# Optimized Service Startup Script
# Generated by Service Dependency Validator
# Starts services in optimal order based on dependency analysis

set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

wait_for_service() {
    local service_name="$1"
    local port="$2"
    local timeout="$3"
    
    log "Waiting for $service_name on port $port..."
    
    for i in $(seq 1 $timeout); do
        if netstat -tulpn 2>/dev/null | grep -q ":$port "; then
            log "$service_name is ready"
            return 0
        fi
        sleep 1
    done
    
    log "WARNING: $service_name did not start within $timeout seconds"
    return 1
}

start_service_level() {
    local level="$1"
    shift
    local services=("$@")
    
    log "Starting service level $level: ${services[*]}"
    
    # Start all services in this level in parallel
    local pids=()
    for service in "${services[@]}"; do
        case "$service" in
"""
        
        # Add service start commands
        for service_name, config in self.services.items():
            script_content += f"""            {service_name})
                # Start {service_name} - {config.tier} tier
                s6-svc -u /run/s6/services/{service_name} &
                pids+=($!)
                ;;
"""
        
        script_content += """        esac
    done
    
    # Wait for all services to start
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    # Verify services are running
    for service in "${services[@]}"; do
        case "$service" in
"""
        
        # Add service verification
        for service_name, config in self.services.items():
            script_content += f"""            {service_name})
                wait_for_service "{service_name}" {config.port} {config.max_startup_time}
                ;;
"""
        
        script_content += """        esac
    done
}

main() {
    log "Starting optimized service startup sequence..."
    
"""
        
        # Add startup levels
        for level, services in sorted(levels.items()):
            services_str = " ".join([f'"{s}"' for s in services])
            script_content += f"""    start_service_level {level} {services_str}
    
"""
        
        script_content += """    log "All services started successfully!"
}

main "$@"
"""
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        subprocess.run(['chmod', '+x', output_file])
        
        logger.info(f"Optimized startup script generated: {output_file}")

async def main():
    parser = argparse.ArgumentParser(description="Service Dependency Validator")
    parser.add_argument('-c', '--container', help='Container name to monitor')
    parser.add_argument('-v', '--visualize', action='store_true', 
                       help='Generate dependency graph visualization')
    parser.add_argument('-s', '--startup-script', action='store_true',
                       help='Generate optimized startup script')
    parser.add_argument('-m', '--monitor', type=int, metavar='SECONDS',
                       help='Continuously monitor dependencies (interval in seconds)')
    parser.add_argument('-o', '--output', default='dependency_analysis.json',
                       help='Output file for analysis results')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ServiceDependencyValidator(args.container)
    
    # Build dependency graph
    validator.build_dependency_graph()
    
    # Check for circular dependencies
    cycles = validator.detect_circular_dependencies()
    if cycles:
        logger.error("Circular dependencies detected! Fix these before proceeding.")
        return 1
    
    # Calculate optimal startup order
    startup_order, levels = validator.calculate_optimal_startup_order()
    
    # Generate visualization if requested
    if args.visualize:
        validator.visualize_dependency_graph()
    
    # Generate startup script if requested
    if args.startup_script:
        validator.generate_startup_script()
    
    # Run dependency validation
    results = await validator.validate_all_dependencies()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Analysis results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SERVICE DEPENDENCY VALIDATION RESULTS")
    print("="*60)
    print(f"Total Services: {results['total_services']}")
    print(f"Running Services: {results['running_services']}")
    print(f"Healthy Services: {results['healthy_services']}")
    print(f"Overall Health: {results['healthy_services']/results['total_services']*100:.1f}%")
    
    if results['dependency_violations']:
        print(f"\n🚨 Dependency Violations: {len(results['dependency_violations'])}")
        for violation in results['dependency_violations']:
            print(f"  - {violation['service']} ({violation['tier']}): "
                  f"Missing {', '.join(violation['unsatisfied_dependencies'])}")
    
    print("\n📊 Service Tier Health:")
    for tier, stats in results['tier_analysis'].items():
        health_pct = stats['healthy']/stats['total']*100 if stats['total'] > 0 else 0
        print(f"  {tier.upper()}: {stats['healthy']}/{stats['total']} ({health_pct:.1f}%)")
    
    # Monitor continuously if requested
    if args.monitor:
        logger.info(f"Starting continuous monitoring (interval: {args.monitor}s)")
        try:
            while True:
                await asyncio.sleep(args.monitor)
                results = await validator.validate_all_dependencies()
                
                # Log critical changes
                violations = len(results['dependency_violations'])
                if violations > 0:
                    logger.warning(f"Dependency violations detected: {violations}")
                
                logger.info(f"Health check: {results['healthy_services']}/{results['total_services']} "
                           f"services healthy ({results['healthy_services']/results['total_services']*100:.1f}%)")
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    # Return exit code based on health
    critical_healthy = results['tier_analysis'].get('critical', {}).get('healthy', 0)
    critical_total = results['tier_analysis'].get('critical', {}).get('total', 1)
    
    if critical_healthy == critical_total and not results['dependency_violations']:
        return 0  # All good
    elif critical_healthy > 0:
        return 1  # Some issues but not critical
    else:
        return 2  # Critical issues

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)