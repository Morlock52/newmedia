#!/usr/bin/env python3
"""
Intelligent Performance Optimization System
==========================================

Advanced performance optimization engine that analyzes system metrics,
identifies bottlenecks, and provides automated optimization recommendations
and implementations.
"""

import asyncio
import json
import logging
import time
import psutil
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import subprocess
import yaml
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationRecommendation:
    """Single optimization recommendation"""
    id: str
    category: str  # cpu, memory, disk, network, application
    severity: str  # low, medium, high, critical
    title: str
    description: str
    impact_estimate: float  # Expected performance improvement (0-100%)
    effort_estimate: str  # low, medium, high
    implementation_complexity: str  # simple, moderate, complex
    automated: bool  # Can be automatically applied
    commands: List[str]  # Commands to execute
    verification_commands: List[str]  # Commands to verify optimization
    rollback_commands: List[str]  # Commands to rollback if needed
    prerequisites: List[str]  # Prerequisites before applying
    risks: List[str]  # Potential risks
    estimated_time_minutes: int  # Time to implement
    cost_benefit_ratio: float  # Higher is better
    dependencies: List[str]  # Other optimizations this depends on

@dataclass
class OptimizationResult:
    """Result of an optimization implementation"""
    recommendation_id: str
    status: str  # success, failed, partial, skipped
    start_time: float
    end_time: float
    duration: float
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_achieved: float  # Actual improvement percentage
    error_message: Optional[str] = None
    warnings: List[str] = None
    rollback_available: bool = True

@dataclass
class SystemAnalysis:
    """Comprehensive system analysis results"""
    timestamp: float
    cpu_analysis: Dict[str, Any]
    memory_analysis: Dict[str, Any]
    disk_analysis: Dict[str, Any]
    network_analysis: Dict[str, Any]
    application_analysis: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    performance_score: float
    recommendations: List[OptimizationRecommendation]

class CPUOptimizer:
    """CPU performance optimization"""
    
    def __init__(self):
        self.name = "CPU Optimizer"
        
    def analyze(self, metrics_history: List[Dict]) -> Tuple[Dict[str, Any], List[OptimizationRecommendation]]:
        """Analyze CPU performance and generate recommendations"""
        if not metrics_history:
            return {}, []
            
        cpu_usages = [m.get('cpu_usage', 0) for m in metrics_history[-20:]]  # Last 20 readings
        avg_cpu = statistics.mean(cpu_usages)
        max_cpu = max(cpu_usages)
        cpu_volatility = statistics.stdev(cpu_usages) if len(cpu_usages) > 1 else 0
        
        analysis = {
            'average_usage': avg_cpu,
            'peak_usage': max_cpu,
            'volatility': cpu_volatility,
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
        
        recommendations = []
        
        # High CPU usage recommendations
        if avg_cpu > 80:
            recommendations.append(OptimizationRecommendation(
                id="cpu_high_usage",
                category="cpu",
                severity="high",
                title="High CPU usage detected",
                description=f"Average CPU usage is {avg_cpu:.1f}%. Consider optimizing CPU-intensive processes or upgrading hardware.",
                impact_estimate=15.0,
                effort_estimate="medium",
                implementation_complexity="moderate",
                automated=True,
                commands=[
                    "nice -n 10 pgrep -f 'cpu-intensive-process'",  # Lower priority of CPU-intensive processes
                    "echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",  # Performance governor
                ],
                verification_commands=["cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],
                rollback_commands=["echo 'ondemand' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"],
                prerequisites=["sudo access required"],
                risks=["May increase power consumption"],
                estimated_time_minutes=5,
                cost_benefit_ratio=3.0,
                dependencies=[]
            ))
            
        # CPU frequency scaling optimization
        if analysis['frequency'].get('current', 0) < analysis['frequency'].get('max', 0) * 0.8:
            recommendations.append(OptimizationRecommendation(
                id="cpu_frequency_scaling",
                category="cpu",
                severity="medium",
                title="CPU not running at optimal frequency",
                description="CPU frequency scaling can be optimized for better performance.",
                impact_estimate=10.0,
                effort_estimate="low",
                implementation_complexity="simple",
                automated=True,
                commands=[
                    "echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
                ],
                verification_commands=["cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],
                rollback_commands=["echo 'ondemand' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"],
                prerequisites=["sudo access required"],
                risks=["Increased power consumption"],
                estimated_time_minutes=2,
                cost_benefit_ratio=5.0,
                dependencies=[]
            ))
            
        # Process optimization
        if cpu_volatility > 20:
            recommendations.append(OptimizationRecommendation(
                id="cpu_process_optimization",
                category="cpu",
                severity="medium",
                title="High CPU usage volatility",
                description="CPU usage is highly variable, indicating inefficient process scheduling.",
                impact_estimate=8.0,
                effort_estimate="medium",
                implementation_complexity="moderate",
                automated=True,
                commands=[
                    "echo 0 | sudo tee /proc/sys/kernel/sched_migration_cost_ns",  # Reduce migration cost
                    "echo 1 | sudo tee /proc/sys/kernel/sched_autogroup_enabled"   # Enable automatic process grouping
                ],
                verification_commands=[
                    "cat /proc/sys/kernel/sched_migration_cost_ns",
                    "cat /proc/sys/kernel/sched_autogroup_enabled"
                ],
                rollback_commands=[
                    "echo 5000000 | sudo tee /proc/sys/kernel/sched_migration_cost_ns",
                    "echo 0 | sudo tee /proc/sys/kernel/sched_autogroup_enabled"
                ],
                prerequisites=["sudo access required"],
                risks=["May affect process scheduling behavior"],
                estimated_time_minutes=3,
                cost_benefit_ratio=2.5,
                dependencies=[]
            ))
            
        return analysis, recommendations

class MemoryOptimizer:
    """Memory performance optimization"""
    
    def __init__(self):
        self.name = "Memory Optimizer"
        
    def analyze(self, metrics_history: List[Dict]) -> Tuple[Dict[str, Any], List[OptimizationRecommendation]]:
        """Analyze memory performance and generate recommendations"""
        if not metrics_history:
            return {}, []
            
        memory_usages = [m.get('memory_usage', 0) for m in metrics_history[-20:]]
        avg_memory = statistics.mean(memory_usages)
        max_memory = max(memory_usages)
        
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        
        analysis = {
            'average_usage': avg_memory,
            'peak_usage': max_memory,
            'total_gb': memory_info.total / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'swap_usage': swap_info.percent,
            'swap_total_gb': swap_info.total / (1024**3),
            'cached_gb': memory_info.cached / (1024**3) if hasattr(memory_info, 'cached') else 0
        }
        
        recommendations = []
        
        # High memory usage
        if avg_memory > 85:
            recommendations.append(OptimizationRecommendation(
                id="memory_high_usage",
                category="memory",
                severity="high",
                title="High memory usage detected",
                description=f"Average memory usage is {avg_memory:.1f}%. Consider optimizing memory usage or adding more RAM.",
                impact_estimate=20.0,
                effort_estimate="medium",
                implementation_complexity="moderate",
                automated=True,
                commands=[
                    "echo 3 | sudo tee /proc/sys/vm/drop_caches",  # Clear caches
                    "echo 1 | sudo tee /proc/sys/vm/compact_memory"  # Compact memory
                ],
                verification_commands=["free -h"],
                rollback_commands=[],  # Cache clearing is safe and automatic
                prerequisites=["sudo access required"],
                risks=["Temporary performance impact while caches rebuild"],
                estimated_time_minutes=2,
                cost_benefit_ratio=4.0,
                dependencies=[]
            ))
            
        # Swap usage optimization
        if swap_info.percent > 10:
            recommendations.append(OptimizationRecommendation(
                id="memory_swap_optimization",
                category="memory",
                severity="high",
                title="High swap usage detected",
                description=f"Swap usage is {swap_info.percent:.1f}%. This can severely impact performance.",
                impact_estimate=25.0,
                effort_estimate="low",
                implementation_complexity="simple",
                automated=True,
                commands=[
                    "echo 10 | sudo tee /proc/sys/vm/swappiness"  # Reduce swap tendency
                ],
                verification_commands=["cat /proc/sys/vm/swappiness"],
                rollback_commands=["echo 60 | sudo tee /proc/sys/vm/swappiness"],
                prerequisites=["sudo access required"],
                risks=["May use more RAM"],
                estimated_time_minutes=1,
                cost_benefit_ratio=8.0,
                dependencies=[]
            ))
            
        # Memory defragmentation
        if analysis['available_gb'] < analysis['total_gb'] * 0.2:  # Less than 20% available
            recommendations.append(OptimizationRecommendation(
                id="memory_defragmentation",
                category="memory",
                severity="medium",
                title="Memory fragmentation optimization",
                description="Memory appears fragmented. Compaction may help.",
                impact_estimate=12.0,
                effort_estimate="low",
                implementation_complexity="simple",
                automated=True,
                commands=[
                    "echo 1 | sudo tee /proc/sys/vm/compact_memory"
                ],
                verification_commands=["cat /proc/meminfo | grep -E '(MemFree|MemAvailable)'"],
                rollback_commands=[],
                prerequisites=["sudo access required"],
                risks=["Brief pause in memory allocations"],
                estimated_time_minutes=1,
                cost_benefit_ratio=6.0,
                dependencies=[]
            ))
            
        return analysis, recommendations

class DiskOptimizer:
    """Disk I/O performance optimization"""
    
    def __init__(self):
        self.name = "Disk Optimizer"
        
    def analyze(self, metrics_history: List[Dict]) -> Tuple[Dict[str, Any], List[OptimizationRecommendation]]:
        """Analyze disk performance and generate recommendations"""
        disk_usage = {}
        io_stats = psutil.disk_io_counters()
        
        # Get disk usage for all partitions
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'total_gb': usage.total / (1024**3),
                    'used_gb': usage.used / (1024**3),
                    'free_gb': usage.free / (1024**3),
                    'percent': (usage.used / usage.total) * 100,
                    'device': partition.device,
                    'fstype': partition.fstype
                }
            except (PermissionError, OSError):
                continue
                
        analysis = {
            'partitions': disk_usage,
            'io_stats': {
                'read_bytes': io_stats.read_bytes,
                'write_bytes': io_stats.write_bytes,
                'read_count': io_stats.read_count,
                'write_count': io_stats.write_count,
                'read_time': io_stats.read_time,
                'write_time': io_stats.write_time
            } if io_stats else {}
        }
        
        recommendations = []
        
        # High disk usage
        for mount, info in disk_usage.items():
            if info['percent'] > 90:
                recommendations.append(OptimizationRecommendation(
                    id=f"disk_cleanup_{mount.replace('/', '_')}",
                    category="disk",
                    severity="critical",
                    title=f"Critical disk space on {mount}",
                    description=f"Disk usage is {info['percent']:.1f}% on {mount}. Immediate cleanup required.",
                    impact_estimate=30.0,
                    effort_estimate="medium",
                    implementation_complexity="moderate",
                    automated=True,
                    commands=[
                        f"find {mount} -name '*.log' -mtime +7 -delete",  # Delete old logs
                        f"find {mount} -name '*.tmp' -delete",  # Delete temp files
                        f"find {mount} -name 'core.*' -delete",  # Delete core dumps
                        "apt-get autoremove -y",  # Remove unused packages (Ubuntu/Debian)
                        "apt-get autoclean"  # Clean package cache
                    ],
                    verification_commands=[f"df -h {mount}"],
                    rollback_commands=[],  # File deletion is not easily reversible
                    prerequisites=["sudo access required", "Backup critical data first"],
                    risks=["Potential data loss if important files are deleted"],
                    estimated_time_minutes=10,
                    cost_benefit_ratio=7.0,
                    dependencies=[]
                ))
                
            elif info['percent'] > 80:
                recommendations.append(OptimizationRecommendation(
                    id=f"disk_cleanup_mild_{mount.replace('/', '_')}",
                    category="disk",
                    severity="medium",
                    title=f"High disk usage on {mount}",
                    description=f"Disk usage is {info['percent']:.1f}% on {mount}. Consider cleanup.",
                    impact_estimate=15.0,
                    effort_estimate="low",
                    implementation_complexity="simple",
                    automated=True,
                    commands=[
                        f"find {mount} -name '*.log' -mtime +30 -delete",
                        f"find {mount} -name '*.tmp' -delete",
                        "journalctl --vacuum-time=7d"  # Clean old journal logs
                    ],
                    verification_commands=[f"df -h {mount}"],
                    rollback_commands=[],
                    prerequisites=["sudo access required"],
                    risks=["Minor risk of deleting needed temporary files"],
                    estimated_time_minutes=5,
                    cost_benefit_ratio=5.0,
                    dependencies=[]
                ))
                
        # I/O scheduler optimization
        recommendations.append(OptimizationRecommendation(
            id="disk_io_scheduler",
            category="disk",
            severity="medium",
            title="Disk I/O scheduler optimization",
            description="Optimize disk I/O scheduler for better performance.",
            impact_estimate=10.0,
            effort_estimate="low",
            implementation_complexity="simple",
            automated=True,
            commands=[
                "echo mq-deadline | sudo tee /sys/block/*/queue/scheduler"  # Use mq-deadline scheduler
            ],
            verification_commands=["cat /sys/block/sda/queue/scheduler"],
            rollback_commands=["echo cfq | sudo tee /sys/block/*/queue/scheduler"],
            prerequisites=["sudo access required"],
            risks=["May affect I/O patterns"],
            estimated_time_minutes=2,
            cost_benefit_ratio=4.0,
            dependencies=[]
        ))
        
        return analysis, recommendations

class NetworkOptimizer:
    """Network performance optimization"""
    
    def __init__(self):
        self.name = "Network Optimizer"
        
    def analyze(self, metrics_history: List[Dict]) -> Tuple[Dict[str, Any], List[OptimizationRecommendation]]:
        """Analyze network performance and generate recommendations"""
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        analysis = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout,
            'dropin': net_io.dropin,
            'dropout': net_io.dropout,
            'active_connections': net_connections
        }
        
        recommendations = []
        
        # Network buffer optimization
        recommendations.append(OptimizationRecommendation(
            id="network_buffer_optimization",
            category="network",
            severity="medium",
            title="Network buffer optimization",
            description="Optimize network buffer sizes for better throughput.",
            impact_estimate=15.0,
            effort_estimate="low",
            implementation_complexity="simple",
            automated=True,
            commands=[
                "echo 'net.core.rmem_max = 33554432' | sudo tee -a /etc/sysctl.conf",
                "echo 'net.core.wmem_max = 33554432' | sudo tee -a /etc/sysctl.conf",
                "echo 'net.ipv4.tcp_rmem = 4096 16384 33554432' | sudo tee -a /etc/sysctl.conf",
                "echo 'net.ipv4.tcp_wmem = 4096 16384 33554432' | sudo tee -a /etc/sysctl.conf",
                "sysctl -p"
            ],
            verification_commands=["sysctl net.core.rmem_max", "sysctl net.core.wmem_max"],
            rollback_commands=[
                "sed -i '/net.core.rmem_max/d' /etc/sysctl.conf",
                "sed -i '/net.core.wmem_max/d' /etc/sysctl.conf",
                "sed -i '/net.ipv4.tcp_rmem/d' /etc/sysctl.conf",
                "sed -i '/net.ipv4.tcp_wmem/d' /etc/sysctl.conf",
                "sysctl -p"
            ],
            prerequisites=["sudo access required"],
            risks=["May increase memory usage"],
            estimated_time_minutes=3,
            cost_benefit_ratio=5.0,
            dependencies=[]
        ))
        
        # TCP congestion control
        recommendations.append(OptimizationRecommendation(
            id="tcp_congestion_control",
            category="network",
            severity="low",
            title="TCP congestion control optimization",
            description="Use BBR congestion control for better network performance.",
            impact_estimate=8.0,
            effort_estimate="low",
            implementation_complexity="simple",
            automated=True,
            commands=[
                "echo 'net.core.default_qdisc = fq' | sudo tee -a /etc/sysctl.conf",
                "echo 'net.ipv4.tcp_congestion_control = bbr' | sudo tee -a /etc/sysctl.conf",
                "sysctl -p"
            ],
            verification_commands=["sysctl net.ipv4.tcp_congestion_control"],
            rollback_commands=[
                "sed -i '/net.core.default_qdisc/d' /etc/sysctl.conf",
                "sed -i '/net.ipv4.tcp_congestion_control/d' /etc/sysctl.conf",
                "sysctl -p"
            ],
            prerequisites=["sudo access required", "BBR module available"],
            risks=["May affect network behavior on some systems"],
            estimated_time_minutes=3,
            cost_benefit_ratio=3.0,
            dependencies=[]
        ))
        
        # High connection count optimization
        if net_connections > 1000:
            recommendations.append(OptimizationRecommendation(
                id="network_connection_optimization",
                category="network",
                severity="medium",
                title="High network connection count",
                description=f"System has {net_connections} active connections. Optimize connection handling.",
                impact_estimate=12.0,
                effort_estimate="medium",
                implementation_complexity="moderate",
                automated=True,
                commands=[
                    "echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf",
                    "echo 'net.ipv4.tcp_max_syn_backlog = 8192' | sudo tee -a /etc/sysctl.conf",
                    "echo 'net.ipv4.tcp_fin_timeout = 30' | sudo tee -a /etc/sysctl.conf",
                    "sysctl -p"
                ],
                verification_commands=["sysctl net.core.somaxconn"],
                rollback_commands=[
                    "sed -i '/net.core.somaxconn/d' /etc/sysctl.conf",
                    "sed -i '/net.ipv4.tcp_max_syn_backlog/d' /etc/sysctl.conf",
                    "sed -i '/net.ipv4.tcp_fin_timeout/d' /etc/sysctl.conf",
                    "sysctl -p"
                ],
                prerequisites=["sudo access required"],
                risks=["May affect connection behavior"],
                estimated_time_minutes=5,
                cost_benefit_ratio=4.0,
                dependencies=[]
            ))
            
        return analysis, recommendations

class ApplicationOptimizer:
    """Application-specific optimization"""
    
    def __init__(self):
        self.name = "Application Optimizer"
        
    def analyze(self, app_metrics: List[Dict]) -> Tuple[Dict[str, Any], List[OptimizationRecommendation]]:
        """Analyze application performance and generate recommendations"""
        if not app_metrics:
            return {}, []
            
        # Analyze each application
        analysis = {}
        recommendations = []
        
        # Docker optimization
        try:
            import docker
            client = docker.from_env()
            containers = client.containers.list()
            
            analysis['docker'] = {
                'container_count': len(containers),
                'running_containers': len([c for c in containers if c.status == 'running'])
            }
            
            # Container resource optimization
            for container in containers:
                if container.status == 'running':
                    stats = container.stats(stream=False)
                    
                    # Check for memory-constrained containers
                    memory_stats = stats.get('memory_stats', {})
                    memory_usage = memory_stats.get('usage', 0)
                    memory_limit = memory_stats.get('limit', 0)
                    
                    if memory_limit > 0 and (memory_usage / memory_limit) > 0.9:
                        recommendations.append(OptimizationRecommendation(
                            id=f"docker_memory_{container.name}",
                            category="application",
                            severity="medium",
                            title=f"Docker container {container.name} memory optimization",
                            description=f"Container {container.name} is using {(memory_usage/memory_limit)*100:.1f}% of allocated memory.",
                            impact_estimate=15.0,
                            effort_estimate="medium",
                            implementation_complexity="moderate",
                            automated=False,  # Requires manual intervention
                            commands=[
                                f"docker update --memory=2g {container.name}",  # Increase memory limit
                                f"docker restart {container.name}"
                            ],
                            verification_commands=[f"docker stats --no-stream {container.name}"],
                            rollback_commands=[f"docker update --memory=1g {container.name}"],
                            prerequisites=["Docker admin access"],
                            risks=["Container restart required"],
                            estimated_time_minutes=5,
                            cost_benefit_ratio=4.0,
                            dependencies=[]
                        ))
                        
        except Exception as e:
            logger.debug(f"Docker analysis skipped: {e}")
            
        # Media server optimizations
        media_services = ['jellyfin', 'plex', 'emby']
        for service in media_services:
            service_metrics = [m for m in app_metrics if m.get('service_name') == service]
            if service_metrics:
                avg_response_time = statistics.mean([m.get('response_time', 0) for m in service_metrics[-10:]])
                
                if avg_response_time > 2000:  # > 2 seconds
                    recommendations.append(OptimizationRecommendation(
                        id=f"media_server_{service}_optimization",
                        category="application",
                        severity="medium",
                        title=f"{service.title()} performance optimization",
                        description=f"{service.title()} average response time is {avg_response_time:.0f}ms. Consider optimization.",
                        impact_estimate=20.0,
                        effort_estimate="medium",
                        implementation_complexity="moderate",
                        automated=False,
                        commands=[
                            f"docker exec {service} rm -rf /config/transcodes/*",  # Clear transcode cache
                            f"docker restart {service}"
                        ],
                        verification_commands=[f"curl -f http://localhost:8096/System/Info"],
                        rollback_commands=[],
                        prerequisites=[f"{service} container running"],
                        risks=["Service restart required"],
                        estimated_time_minutes=3,
                        cost_benefit_ratio=6.0,
                        dependencies=[]
                    ))
                    
        return analysis, recommendations

class PerformanceOptimizer:
    """Main performance optimization orchestrator"""
    
    def __init__(self, config_path: str = "config/monitoring.yml"):
        self.config = self._load_config(config_path)
        self.db_path = "/tmp/optimization_results.db"
        self.init_database()
        
        # Initialize optimizers
        self.cpu_optimizer = CPUOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.disk_optimizer = DiskOptimizer()
        self.network_optimizer = NetworkOptimizer()
        self.app_optimizer = ApplicationOptimizer()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
            
    def init_database(self):
        """Initialize optimization results database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    recommendation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL NOT NULL,
                    before_metrics TEXT,
                    after_metrics TEXT,
                    improvement_achieved REAL,
                    error_message TEXT,
                    warnings TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    analysis_data TEXT NOT NULL,
                    performance_score REAL NOT NULL,
                    recommendations_count INTEGER NOT NULL
                )
            """)
            
    async def run_comprehensive_analysis(self, metrics_history: List[Dict] = None, 
                                       app_metrics: List[Dict] = None) -> SystemAnalysis:
        """Run comprehensive system analysis"""
        logger.info("Starting comprehensive performance analysis")
        
        if metrics_history is None:
            metrics_history = await self._collect_current_metrics()
            
        if app_metrics is None:
            app_metrics = await self._collect_application_metrics()
            
        start_time = time.time()
        
        # Run individual analyses
        cpu_analysis, cpu_recommendations = self.cpu_optimizer.analyze(metrics_history)
        memory_analysis, memory_recommendations = self.memory_optimizer.analyze(metrics_history)
        disk_analysis, disk_recommendations = self.disk_optimizer.analyze(metrics_history)
        network_analysis, network_recommendations = self.network_optimizer.analyze(metrics_history)
        app_analysis, app_recommendations = self.app_optimizer.analyze(app_metrics)
        
        # Combine all recommendations
        all_recommendations = (
            cpu_recommendations + memory_recommendations + 
            disk_recommendations + network_recommendations + app_recommendations
        )
        
        # Sort by cost-benefit ratio and severity
        all_recommendations.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
            x.cost_benefit_ratio
        ), reverse=True)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(
            cpu_analysis, memory_analysis, disk_analysis, 
            network_analysis, app_analysis
        )
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(
            cpu_analysis, memory_analysis, disk_analysis, 
            network_analysis, bottlenecks
        )
        
        analysis = SystemAnalysis(
            timestamp=start_time,
            cpu_analysis=cpu_analysis,
            memory_analysis=memory_analysis,
            disk_analysis=disk_analysis,
            network_analysis=network_analysis,
            application_analysis=app_analysis,
            bottlenecks=bottlenecks,
            performance_score=performance_score,
            recommendations=all_recommendations
        )
        
        # Store analysis results
        await self._store_analysis(analysis)
        
        logger.info(f"Analysis completed. Performance score: {performance_score:.1f}/100")
        logger.info(f"Generated {len(all_recommendations)} optimization recommendations")
        
        return analysis
        
    def _identify_bottlenecks(self, cpu_analysis: Dict, memory_analysis: Dict,
                            disk_analysis: Dict, network_analysis: Dict,
                            app_analysis: Dict) -> List[Dict[str, Any]]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        if cpu_analysis.get('average_usage', 0) > 80:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high' if cpu_analysis['average_usage'] > 90 else 'medium',
                'description': f"CPU usage averaging {cpu_analysis['average_usage']:.1f}%",
                'impact': 'System responsiveness'
            })
            
        # Memory bottleneck
        if memory_analysis.get('average_usage', 0) > 85:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high' if memory_analysis['average_usage'] > 95 else 'medium',
                'description': f"Memory usage averaging {memory_analysis['average_usage']:.1f}%",
                'impact': 'Application performance and stability'
            })
            
        # Swap usage bottleneck
        if memory_analysis.get('swap_usage', 0) > 10:
            bottlenecks.append({
                'type': 'swap',
                'severity': 'critical' if memory_analysis['swap_usage'] > 50 else 'high',
                'description': f"Swap usage at {memory_analysis['swap_usage']:.1f}%",
                'impact': 'Severe performance degradation'
            })
            
        # Disk space bottleneck
        for mount, info in disk_analysis.get('partitions', {}).items():
            if info['percent'] > 90:
                bottlenecks.append({
                    'type': 'disk_space',
                    'severity': 'critical',
                    'description': f"Disk {mount} at {info['percent']:.1f}% capacity",
                    'impact': 'System stability and data integrity'
                })
                
        return bottlenecks
        
    def _calculate_performance_score(self, cpu_analysis: Dict, memory_analysis: Dict,
                                   disk_analysis: Dict, network_analysis: Dict,
                                   bottlenecks: List[Dict]) -> float:
        """Calculate overall performance score (0-100)"""
        base_score = 100.0
        
        # Deduct points for high resource usage
        cpu_usage = cpu_analysis.get('average_usage', 0)
        memory_usage = memory_analysis.get('average_usage', 0)
        swap_usage = memory_analysis.get('swap_usage', 0)
        
        # CPU penalty
        if cpu_usage > 70:
            base_score -= min(30, (cpu_usage - 70) * 1.5)
            
        # Memory penalty
        if memory_usage > 70:
            base_score -= min(25, (memory_usage - 70) * 1.2)
            
        # Swap penalty
        base_score -= min(20, swap_usage * 2)
        
        # Disk space penalty
        for mount, info in disk_analysis.get('partitions', {}).items():
            if info['percent'] > 80:
                base_score -= min(15, (info['percent'] - 80) * 0.75)
                
        # Bottleneck penalties
        for bottleneck in bottlenecks:
            severity_penalty = {
                'low': 2, 'medium': 5, 'high': 10, 'critical': 20
            }.get(bottleneck['severity'], 0)
            base_score -= severity_penalty
            
        return max(0, min(100, base_score))
        
    async def apply_optimization(self, recommendation: OptimizationRecommendation,
                               dry_run: bool = False) -> OptimizationResult:
        """Apply a single optimization recommendation"""
        logger.info(f"Applying optimization: {recommendation.title}")
        
        start_time = time.time()
        
        # Collect before metrics
        before_metrics = await self._collect_current_metrics()
        
        if dry_run:
            logger.info("DRY RUN: Would execute the following commands:")
            for cmd in recommendation.commands:
                logger.info(f"  {cmd}")
                
            return OptimizationResult(
                recommendation_id=recommendation.id,
                status="dry_run",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                improvement_achieved=0.0,
                warnings=["Dry run mode - no changes made"]
            )
            
        # Check prerequisites
        prerequisite_check = await self._check_prerequisites(recommendation.prerequisites)
        if not prerequisite_check['passed']:
            return OptimizationResult(
                recommendation_id=recommendation.id,
                status="failed",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                before_metrics=before_metrics,
                after_metrics=before_metrics,
                improvement_achieved=0.0,
                error_message=f"Prerequisites not met: {prerequisite_check['message']}"
            )
            
        # Execute optimization commands
        warnings = []
        try:
            for cmd in recommendation.commands:
                logger.info(f"Executing: {cmd}")
                
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    error_msg = f"Command failed: {cmd}\nError: {result.stderr}"
                    logger.error(error_msg)
                    
                    # Attempt rollback
                    await self._rollback_optimization(recommendation)
                    
                    return OptimizationResult(
                        recommendation_id=recommendation.id,
                        status="failed",
                        start_time=start_time,
                        end_time=time.time(),
                        duration=time.time() - start_time,
                        before_metrics=before_metrics,
                        after_metrics=await self._collect_current_metrics(),
                        improvement_achieved=0.0,
                        error_message=error_msg,
                        rollback_available=bool(recommendation.rollback_commands)
                    )
                    
                if result.stderr:
                    warnings.append(f"Warning from '{cmd}': {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            error_msg = "Optimization timed out"
            logger.error(error_msg)
            
            await self._rollback_optimization(recommendation)
            
            return OptimizationResult(
                recommendation_id=recommendation.id,
                status="failed",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                before_metrics=before_metrics,
                after_metrics=await self._collect_current_metrics(),
                improvement_achieved=0.0,
                error_message=error_msg,
                rollback_available=bool(recommendation.rollback_commands)
            )
            
        # Wait for changes to take effect
        await asyncio.sleep(5)
        
        # Verify optimization
        verification_passed = await self._verify_optimization(recommendation)
        if not verification_passed:
            warnings.append("Verification checks failed")
            
        # Collect after metrics
        after_metrics = await self._collect_current_metrics()
        
        # Calculate improvement
        improvement = self._calculate_improvement(
            before_metrics, after_metrics, recommendation.category
        )
        
        end_time = time.time()
        result = OptimizationResult(
            recommendation_id=recommendation.id,
            status="success",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_achieved=improvement,
            warnings=warnings
        )
        
        # Store result
        await self._store_optimization_result(result)
        
        logger.info(f"Optimization completed. Improvement: {improvement:.1f}%")
        return result
        
    async def apply_multiple_optimizations(self, recommendations: List[OptimizationRecommendation],
                                         max_concurrent: int = 3, dry_run: bool = False) -> List[OptimizationResult]:
        """Apply multiple optimization recommendations with dependency handling"""
        logger.info(f"Applying {len(recommendations)} optimizations (max concurrent: {max_concurrent})")
        
        # Sort by dependencies and priority
        sorted_recommendations = self._sort_by_dependencies(recommendations)
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def apply_single(rec):
            async with semaphore:
                return await self.apply_optimization(rec, dry_run)
                
        # Apply optimizations in batches, respecting dependencies
        for batch in self._create_dependency_batches(sorted_recommendations):
            batch_tasks = [apply_single(rec) for rec in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Optimization failed with exception: {result}")
                else:
                    results.append(result)
                    
        return results
        
    def _sort_by_dependencies(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Sort recommendations by dependencies and priority"""
        # Simple topological sort for dependencies
        sorted_recs = []
        remaining = recommendations.copy()
        
        while remaining:
            # Find recommendations with no unmet dependencies
            ready = []
            for rec in remaining:
                deps_met = all(
                    any(r.id == dep for r in sorted_recs) or dep not in [r.id for r in recommendations]
                    for dep in rec.dependencies
                )
                if deps_met:
                    ready.append(rec)
                    
            if not ready:
                # Circular dependency or unresolvable - add remaining as-is
                sorted_recs.extend(remaining)
                break
                
            # Sort ready recommendations by priority and cost-benefit
            ready.sort(key=lambda x: (
                {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
                x.cost_benefit_ratio
            ), reverse=True)
            
            sorted_recs.extend(ready)
            for rec in ready:
                remaining.remove(rec)
                
        return sorted_recs
        
    def _create_dependency_batches(self, recommendations: List[OptimizationRecommendation]) -> List[List[OptimizationRecommendation]]:
        """Create batches of recommendations that can be run in parallel"""
        batches = []
        remaining = recommendations.copy()
        completed_ids = set()
        
        while remaining:
            batch = []
            for rec in remaining[:]:
                # Check if all dependencies are completed
                deps_met = all(dep in completed_ids for dep in rec.dependencies)
                if deps_met:
                    batch.append(rec)
                    remaining.remove(rec)
                    
            if batch:
                batches.append(batch)
                completed_ids.update(rec.id for rec in batch)
            else:
                # No progress possible - add remaining as final batch
                batches.append(remaining)
                break
                
        return batches
        
    async def _collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        
        return {
            'timestamp': time.time(),
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'memory_available': memory.available,
            'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
            'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
            'network_bytes_sent': net_io.bytes_sent,
            'network_bytes_recv': net_io.bytes_recv
        }
        
    async def _collect_application_metrics(self) -> List[Dict[str, Any]]:
        """Collect application metrics"""
        # This would typically query your application monitoring system
        # For now, return empty list
        return []
        
    async def _check_prerequisites(self, prerequisites: List[str]) -> Dict[str, Any]:
        """Check if prerequisites are met"""
        for prereq in prerequisites:
            if "sudo" in prereq.lower():
                # Check if sudo is available
                try:
                    result = subprocess.run(
                        ["sudo", "-n", "true"],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode != 0:
                        return {
                            'passed': False,
                            'message': 'Sudo access required but not available'
                        }
                except Exception:
                    return {
                        'passed': False,
                        'message': 'Unable to verify sudo access'
                    }
                    
        return {'passed': True, 'message': 'All prerequisites met'}
        
    async def _verify_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Verify that optimization was applied successfully"""
        for cmd in recommendation.verification_commands:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.warning(f"Verification failed for: {cmd}")
                    return False
            except Exception as e:
                logger.warning(f"Verification error for '{cmd}': {e}")
                return False
                
        return True
        
    async def _rollback_optimization(self, recommendation: OptimizationRecommendation):
        """Rollback an optimization"""
        if not recommendation.rollback_commands:
            logger.warning(f"No rollback commands available for {recommendation.id}")
            return
            
        logger.info(f"Rolling back optimization: {recommendation.id}")
        
        for cmd in recommendation.rollback_commands:
            try:
                subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            except Exception as e:
                logger.error(f"Rollback command failed: {cmd} - {e}")
                
    def _calculate_improvement(self, before_metrics: Dict, after_metrics: Dict, category: str) -> float:
        """Calculate improvement percentage based on metrics"""
        if category == "cpu":
            before_cpu = before_metrics.get('cpu_usage', 0)
            after_cpu = after_metrics.get('cpu_usage', 0)
            if before_cpu > 0:
                return max(0, ((before_cpu - after_cpu) / before_cpu) * 100)
                
        elif category == "memory":
            before_mem = before_metrics.get('memory_usage', 0)
            after_mem = after_metrics.get('memory_usage', 0)
            if before_mem > 0:
                return max(0, ((before_mem - after_mem) / before_mem) * 100)
                
        # Default: no measurable improvement
        return 0.0
        
    async def _store_analysis(self, analysis: SystemAnalysis):
        """Store analysis results in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_analyses 
                (timestamp, analysis_data, performance_score, recommendations_count)
                VALUES (?, ?, ?, ?)
            """, (
                analysis.timestamp,
                json.dumps(asdict(analysis)),
                analysis.performance_score,
                len(analysis.recommendations)
            ))
            
    async def _store_optimization_result(self, result: OptimizationResult):
        """Store optimization result in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_results
                (timestamp, recommendation_id, status, duration, before_metrics,
                 after_metrics, improvement_achieved, error_message, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.start_time,
                result.recommendation_id,
                result.status,
                result.duration,
                json.dumps(result.before_metrics),
                json.dumps(result.after_metrics),
                result.improvement_achieved,
                result.error_message,
                json.dumps(result.warnings) if result.warnings else None
            ))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Optimization System")
    parser.add_argument("--analyze", action="store_true", help="Run system analysis")
    parser.add_argument("--apply", help="Apply optimization by ID")
    parser.add_argument("--apply-all", action="store_true", help="Apply all recommendations")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    async def main():
        config_path = args.config or "config/monitoring.yml"
        optimizer = PerformanceOptimizer(config_path)
        
        if args.analyze:
            analysis = await optimizer.run_comprehensive_analysis()
            
            print("\n" + "="*60)
            print("PERFORMANCE ANALYSIS RESULTS")
            print("="*60)
            print(f"Performance Score: {analysis.performance_score:.1f}/100")
            print(f"Bottlenecks Found: {len(analysis.bottlenecks)}")
            print(f"Recommendations: {len(analysis.recommendations)}")
            
            if analysis.bottlenecks:
                print("\nCritical Bottlenecks:")
                for bottleneck in analysis.bottlenecks:
                    print(f"  • {bottleneck['type'].upper()}: {bottleneck['description']}")
                    
            print("\nTop Recommendations:")
            for i, rec in enumerate(analysis.recommendations[:5]):
                print(f"  {i+1}. {rec.title}")
                print(f"     Impact: {rec.impact_estimate:.1f}% | Effort: {rec.effort_estimate}")
                print(f"     Automated: {'Yes' if rec.automated else 'No'}")
                
        elif args.apply_all:
            analysis = await optimizer.run_comprehensive_analysis()
            results = await optimizer.apply_multiple_optimizations(
                analysis.recommendations, dry_run=args.dry_run
            )
            
            print(f"\nApplied {len(results)} optimizations:")
            for result in results:
                status_icon = "✓" if result.status == "success" else "✗"
                print(f"  {status_icon} {result.recommendation_id}: {result.status}")
                if result.improvement_achieved > 0:
                    print(f"    Improvement: {result.improvement_achieved:.1f}%")
                    
    if args.analyze or args.apply_all:
        asyncio.run(main())
    else:
        print("Please specify --analyze or --apply-all")
        parser.print_help()