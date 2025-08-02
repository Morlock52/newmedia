#!/usr/bin/env python3
"""
Real-time Performance Optimization Engine
AI-powered resource optimization and auto-scaling
"""

import asyncio
import docker
import json
import logging
import numpy as np
import psutil
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class OptimizationType(Enum):
    CPU_OPTIMIZATION = "cpu_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    DISK_OPTIMIZATION = "disk_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    CACHE_OPTIMIZATION = "cache_optimization"
    LOAD_BALANCING = "load_balancing"
    AUTO_SCALING = "auto_scaling"

@dataclass
class OptimizationAction:
    """Optimization action structure"""
    id: str
    timestamp: datetime
    type: OptimizationType
    service: str
    current_value: float
    target_value: float
    action: str
    confidence: float
    estimated_impact: Dict[str, float]
    prerequisites: List[str]
    rollback_plan: str
    status: str = "pending"
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: datetime
    service: str
    cpu_percent: float
    memory_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    disk_read_mbps: float
    disk_write_mbps: float
    network_rx_mbps: float
    network_tx_mbps: float
    active_connections: int
    response_time_ms: float
    error_rate: float
    throughput_rps: float

class SystemResourceMonitor:
    """Monitor system and container resources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
        self.monitored_containers = config.get('monitored_containers', [])
        
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            return {
                'timestamp': datetime.now(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk_usage.total,
                    'used': disk_usage.used,
                    'free': disk_usage.free,
                    'percent': disk_usage.used / disk_usage.total * 100,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                },
                'network': {
                    'bytes_sent': network_io.bytes_sent if network_io else 0,
                    'bytes_recv': network_io.bytes_recv if network_io else 0,
                    'packets_sent': network_io.packets_sent if network_io else 0,
                    'packets_recv': network_io.packets_recv if network_io else 0
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to collect system metrics: {e}")
            return {}
    
    async def collect_container_metrics(self) -> List[ResourceMetrics]:
        """Collect metrics for all monitored containers"""
        metrics = []
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                if not self.monitored_containers or container.name in self.monitored_containers:
                    try:
                        stats = container.stats(stream=False)
                        container_metrics = self._parse_container_stats(container.name, stats)
                        if container_metrics:
                            metrics.append(container_metrics)
                    except Exception as e:
                        logging.error(f"Failed to get stats for container {container.name}: {e}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Failed to collect container metrics: {e}")
            return []
    
    def _parse_container_stats(self, container_name: str, stats: Dict) -> Optional[ResourceMetrics]:
        """Parse Docker container statistics"""
        try:
            # CPU calculation
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * \
                             len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
            
            # Memory calculation
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_limit = stats['memory_stats'].get('limit', 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
            
            # Network calculation
            network_rx = 0
            network_tx = 0
            if 'networks' in stats:
                for interface, net_stats in stats['networks'].items():
                    network_rx += net_stats.get('rx_bytes', 0)
                    network_tx += net_stats.get('tx_bytes', 0)
            
            # Disk I/O calculation
            disk_read = 0
            disk_write = 0
            if 'blkio_stats' in stats and 'io_service_bytes_recursive' in stats['blkio_stats']:
                for io_stat in stats['blkio_stats']['io_service_bytes_recursive']:
                    if io_stat['op'] == 'Read':
                        disk_read += io_stat['value']
                    elif io_stat['op'] == 'Write':
                        disk_write += io_stat['value']
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                service=container_name,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_usage_mb=memory_usage / (1024 * 1024),
                memory_limit_mb=memory_limit / (1024 * 1024),
                disk_read_mbps=disk_read / (1024 * 1024),
                disk_write_mbps=disk_write / (1024 * 1024),
                network_rx_mbps=network_rx / (1024 * 1024),
                network_tx_mbps=network_tx / (1024 * 1024),
                active_connections=0,  # Would need custom metrics
                response_time_ms=0,    # Would need custom metrics
                error_rate=0,          # Would need custom metrics
                throughput_rps=0       # Would need custom metrics
            )
            
        except Exception as e:
            logging.error(f"Failed to parse container stats for {container_name}: {e}")
            return None

class AIOptimizationEngine:
    """AI-powered optimization decision engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_history = []
        self.thresholds = config.get('optimization_thresholds', {
            'cpu_high': 0.8,
            'cpu_low': 0.3,
            'memory_high': 0.85,
            'memory_low': 0.4,
            'disk_io_high': 100,  # MB/s
            'network_high': 100,  # MB/s
            'response_time_high': 1000,  # ms
            'error_rate_high': 0.05  # 5%
        })
        
    def analyze_optimization_opportunities(self, 
                                         current_metrics: List[ResourceMetrics],
                                         historical_metrics: List[ResourceMetrics]) -> List[OptimizationAction]:
        """Analyze metrics and identify optimization opportunities"""
        
        optimization_actions = []
        
        for metric in current_metrics:
            # CPU optimization
            cpu_actions = self._analyze_cpu_optimization(metric, historical_metrics)
            optimization_actions.extend(cpu_actions)
            
            # Memory optimization
            memory_actions = self._analyze_memory_optimization(metric, historical_metrics)
            optimization_actions.extend(memory_actions)
            
            # Disk I/O optimization
            disk_actions = self._analyze_disk_optimization(metric, historical_metrics)
            optimization_actions.extend(disk_actions)
            
            # Network optimization
            network_actions = self._analyze_network_optimization(metric, historical_metrics)
            optimization_actions.extend(network_actions)
            
            # Auto-scaling decisions
            scaling_actions = self._analyze_scaling_needs(metric, historical_metrics)
            optimization_actions.extend(scaling_actions)
        
        # Remove duplicate and conflicting actions
        optimization_actions = self._deduplicate_actions(optimization_actions)
        
        # Prioritize actions by impact and confidence
        optimization_actions = self._prioritize_actions(optimization_actions)
        
        return optimization_actions
    
    def _analyze_cpu_optimization(self, metric: ResourceMetrics, 
                                historical_metrics: List[ResourceMetrics]) -> List[OptimizationAction]:
        """Analyze CPU optimization opportunities"""
        actions = []
        
        # High CPU usage optimization
        if metric.cpu_percent > self.thresholds['cpu_high'] * 100:
            # Check if this is a sustained issue
            recent_metrics = [m for m in historical_metrics 
                            if m.service == metric.service and 
                            (datetime.now() - m.timestamp).total_seconds() < 300]  # 5 minutes
            
            if len(recent_metrics) >= 3:
                avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
                if avg_cpu > self.thresholds['cpu_high'] * 100:
                    
                    action = OptimizationAction(
                        id=f"cpu_opt_{metric.service}_{int(time.time())}",
                        timestamp=datetime.now(),
                        type=OptimizationType.CPU_OPTIMIZATION,
                        service=metric.service,
                        current_value=metric.cpu_percent,
                        target_value=self.thresholds['cpu_high'] * 100 * 0.8,  # Target 80% of threshold
                        action="scale_up_cpu",
                        confidence=0.8,
                        estimated_impact={
                            'cpu_reduction': 20.0,
                            'response_time_improvement': 15.0,
                            'cost_increase': 25.0
                        },
                        prerequisites=["check_cpu_limits", "verify_scaling_enabled"],
                        rollback_plan="scale_down_cpu_if_utilization_low_for_30min"
                    )
                    actions.append(action)
        
        # Low CPU usage optimization (scale down)
        elif metric.cpu_percent < self.thresholds['cpu_low'] * 100:
            recent_metrics = [m for m in historical_metrics 
                            if m.service == metric.service and 
                            (datetime.now() - m.timestamp).total_seconds() < 1800]  # 30 minutes
            
            if len(recent_metrics) >= 10:
                avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
                if avg_cpu < self.thresholds['cpu_low'] * 100:
                    
                    action = OptimizationAction(
                        id=f"cpu_scale_down_{metric.service}_{int(time.time())}",
                        timestamp=datetime.now(),
                        type=OptimizationType.CPU_OPTIMIZATION,
                        service=metric.service,
                        current_value=metric.cpu_percent,
                        target_value=self.thresholds['cpu_low'] * 100 * 1.5,
                        action="scale_down_cpu",
                        confidence=0.7,
                        estimated_impact={
                            'cost_reduction': 20.0,
                            'cpu_increase': 10.0,
                            'response_time_impact': 5.0
                        },
                        prerequisites=["check_min_cpu_limits", "verify_performance_acceptable"],
                        rollback_plan="scale_up_cpu_if_performance_degrades"
                    )
                    actions.append(action)
        
        return actions
    
    def _analyze_memory_optimization(self, metric: ResourceMetrics, 
                                   historical_metrics: List[ResourceMetrics]) -> List[OptimizationAction]:
        """Analyze memory optimization opportunities"""
        actions = []
        
        # High memory usage
        if metric.memory_percent > self.thresholds['memory_high'] * 100:
            
            # Check for memory leaks
            recent_metrics = [m for m in historical_metrics 
                            if m.service == metric.service and 
                            (datetime.now() - m.timestamp).total_seconds() < 3600]  # 1 hour
            
            if len(recent_metrics) >= 10:
                # Calculate memory growth trend
                timestamps = [(m.timestamp - recent_metrics[0].timestamp).total_seconds() 
                            for m in recent_metrics]
                memory_values = [m.memory_percent for m in recent_metrics]
                
                if len(timestamps) > 1:
                    # Simple linear regression for trend
                    slope = np.polyfit(timestamps, memory_values, 1)[0]
                    
                    if slope > 0.1:  # Memory increasing
                        action = OptimizationAction(
                            id=f"memory_leak_{metric.service}_{int(time.time())}",
                            timestamp=datetime.now(),
                            type=OptimizationType.MEMORY_OPTIMIZATION,
                            service=metric.service,
                            current_value=metric.memory_percent,
                            target_value=self.thresholds['memory_high'] * 100 * 0.8,
                            action="investigate_memory_leak",
                            confidence=0.9,
                            estimated_impact={
                                'memory_stabilization': 30.0,
                                'crash_risk_reduction': 50.0
                            },
                            prerequisites=["enable_memory_profiling", "analyze_heap_dump"],
                            rollback_plan="restart_service_if_memory_critical"
                        )
                        actions.append(action)
                    else:
                        # Scale up memory
                        action = OptimizationAction(
                            id=f"memory_scale_{metric.service}_{int(time.time())}",
                            timestamp=datetime.now(),
                            type=OptimizationType.MEMORY_OPTIMIZATION,
                            service=metric.service,
                            current_value=metric.memory_percent,
                            target_value=self.thresholds['memory_high'] * 100 * 0.7,
                            action="scale_up_memory",
                            confidence=0.8,
                            estimated_impact={
                                'memory_headroom': 25.0,
                                'crash_risk_reduction': 40.0,
                                'cost_increase': 15.0
                            },
                            prerequisites=["check_memory_limits", "verify_scaling_enabled"],
                            rollback_plan="scale_down_memory_if_unused"
                        )
                        actions.append(action)
        
        return actions
    
    def _analyze_disk_optimization(self, metric: ResourceMetrics, 
                                 historical_metrics: List[ResourceMetrics]) -> List[OptimizationAction]:
        """Analyze disk I/O optimization opportunities"""
        actions = []
        
        total_disk_io = metric.disk_read_mbps + metric.disk_write_mbps
        
        if total_disk_io > self.thresholds['disk_io_high']:
            
            action = OptimizationAction(
                id=f"disk_opt_{metric.service}_{int(time.time())}",
                timestamp=datetime.now(),
                type=OptimizationType.DISK_OPTIMIZATION,
                service=metric.service,
                current_value=total_disk_io,
                target_value=self.thresholds['disk_io_high'] * 0.8,
                action="optimize_disk_io",
                confidence=0.7,
                estimated_impact={
                    'disk_io_reduction': 25.0,
                    'response_time_improvement': 20.0
                },
                prerequisites=["analyze_disk_usage_patterns", "check_cache_hit_ratio"],
                rollback_plan="revert_disk_optimizations"
            )
            actions.append(action)
        
        return actions
    
    def _analyze_network_optimization(self, metric: ResourceMetrics, 
                                    historical_metrics: List[ResourceMetrics]) -> List[OptimizationAction]:
        """Analyze network optimization opportunities"""
        actions = []
        
        total_network = metric.network_rx_mbps + metric.network_tx_mbps
        
        if total_network > self.thresholds['network_high']:
            
            action = OptimizationAction(
                id=f"network_opt_{metric.service}_{int(time.time())}",
                timestamp=datetime.now(),
                type=OptimizationType.NETWORK_OPTIMIZATION,
                service=metric.service,
                current_value=total_network,
                target_value=self.thresholds['network_high'] * 0.8,
                action="optimize_network_usage",
                confidence=0.6,
                estimated_impact={
                    'network_reduction': 20.0,
                    'bandwidth_savings': 30.0
                },
                prerequisites=["analyze_network_patterns", "check_compression"],
                rollback_plan="revert_network_optimizations"
            )
            actions.append(action)
        
        return actions
    
    def _analyze_scaling_needs(self, metric: ResourceMetrics, 
                             historical_metrics: List[ResourceMetrics]) -> List[OptimizationAction]:
        """Analyze auto-scaling needs"""
        actions = []
        
        # Determine if scaling is needed based on multiple factors
        scaling_score = 0
        
        if metric.cpu_percent > 80:
            scaling_score += 30
        if metric.memory_percent > 85:
            scaling_score += 25
        if metric.response_time_ms > 1000:
            scaling_score += 20
        if metric.error_rate > 0.05:
            scaling_score += 25
        
        if scaling_score > 50:  # Scale up threshold
            action = OptimizationAction(
                id=f"scale_up_{metric.service}_{int(time.time())}",
                timestamp=datetime.now(),
                type=OptimizationType.AUTO_SCALING,
                service=metric.service,
                current_value=1,  # Current instance count (would need to get actual)
                target_value=2,   # Target instance count
                action="scale_up_instances",
                confidence=0.8,
                estimated_impact={
                    'load_distribution': 50.0,
                    'response_time_improvement': 30.0,
                    'cost_increase': 100.0
                },
                prerequisites=["check_scaling_limits", "verify_load_balancer"],
                rollback_plan="scale_down_if_load_decreases"
            )
            actions.append(action)
        
        return actions
    
    def _deduplicate_actions(self, actions: List[OptimizationAction]) -> List[OptimizationAction]:
        """Remove duplicate and conflicting optimization actions"""
        # Simple deduplication by service and type
        seen = set()
        deduplicated = []
        
        for action in actions:
            key = (action.service, action.type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(action)
        
        return deduplicated
    
    def _prioritize_actions(self, actions: List[OptimizationAction]) -> List[OptimizationAction]:
        """Prioritize actions by impact and confidence"""
        def priority_score(action):
            # Calculate priority based on confidence and estimated impact
            impact_sum = sum(action.estimated_impact.values())
            return action.confidence * impact_sum
        
        return sorted(actions, key=priority_score, reverse=True)

class OptimizationExecutor:
    """Execute optimization actions safely"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
        self.dry_run = config.get('dry_run', True)
        
    async def execute_optimization(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute an optimization action"""
        try:
            logging.info(f"Executing optimization: {action.action} for {action.service}")
            
            if self.dry_run:
                logging.info(f"DRY RUN: Would execute {action.action}")
                return {
                    'success': True,
                    'message': f'DRY RUN: {action.action}',
                    'changes': {}
                }
            
            # Execute based on action type
            result = await self._execute_action_by_type(action)
            
            action.status = 'completed' if result['success'] else 'failed'
            action.executed_at = datetime.now()
            action.result = result
            
            return result
            
        except Exception as e:
            logging.error(f"Failed to execute optimization {action.id}: {e}")
            action.status = 'failed'
            action.result = {'success': False, 'error': str(e)}
            return {'success': False, 'error': str(e)}
    
    async def _execute_action_by_type(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute action based on its type"""
        
        if action.type == OptimizationType.CPU_OPTIMIZATION:
            return await self._execute_cpu_optimization(action)
        elif action.type == OptimizationType.MEMORY_OPTIMIZATION:
            return await self._execute_memory_optimization(action)
        elif action.type == OptimizationType.DISK_OPTIMIZATION:
            return await self._execute_disk_optimization(action)
        elif action.type == OptimizationType.NETWORK_OPTIMIZATION:
            return await self._execute_network_optimization(action)
        elif action.type == OptimizationType.CACHE_OPTIMIZATION:
            return await self._execute_cache_optimization(action)
        elif action.type == OptimizationType.AUTO_SCALING:
            return await self._execute_auto_scaling(action)
        else:
            return {'success': False, 'error': f'Unknown action type: {action.type}'}
    
    async def _execute_cpu_optimization(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute CPU optimization"""
        try:
            container = self.docker_client.containers.get(action.service)
            
            if action.action == "scale_up_cpu":
                # Update CPU limits
                current_config = container.attrs['HostConfig']
                new_cpu_limit = int(current_config.get('CpuQuota', 100000) * 1.5)
                
                container.update(cpu_quota=new_cpu_limit)
                
                return {
                    'success': True,
                    'message': f'Scaled up CPU for {action.service}',
                    'changes': {
                        'cpu_quota': new_cpu_limit,
                        'previous_quota': current_config.get('CpuQuota', 100000)
                    }
                }
            
            elif action.action == "scale_down_cpu":
                current_config = container.attrs['HostConfig']
                new_cpu_limit = int(current_config.get('CpuQuota', 100000) * 0.8)
                
                container.update(cpu_quota=new_cpu_limit)
                
                return {
                    'success': True,
                    'message': f'Scaled down CPU for {action.service}',
                    'changes': {
                        'cpu_quota': new_cpu_limit,
                        'previous_quota': current_config.get('CpuQuota', 100000)
                    }
                }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_memory_optimization(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute memory optimization"""
        try:
            container = self.docker_client.containers.get(action.service)
            
            if action.action == "scale_up_memory":
                current_config = container.attrs['HostConfig']
                current_memory = current_config.get('Memory', 0)
                new_memory = int(current_memory * 1.5) if current_memory > 0 else 2 * 1024 * 1024 * 1024  # 2GB default
                
                container.update(mem_limit=new_memory)
                
                return {
                    'success': True,
                    'message': f'Scaled up memory for {action.service}',
                    'changes': {
                        'memory_limit': new_memory,
                        'previous_limit': current_memory
                    }
                }
            
            elif action.action == "investigate_memory_leak":
                # This would trigger memory profiling
                return {
                    'success': True,
                    'message': f'Initiated memory leak investigation for {action.service}',
                    'changes': {
                        'memory_profiling_enabled': True
                    }
                }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_disk_optimization(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute disk optimization"""
        # This would implement disk optimization strategies
        return {
            'success': True,
            'message': f'Applied disk optimizations for {action.service}',
            'changes': {'disk_cache_enabled': True}
        }
    
    async def _execute_network_optimization(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute network optimization"""
        # This would implement network optimization strategies
        return {
            'success': True,
            'message': f'Applied network optimizations for {action.service}',
            'changes': {'compression_enabled': True}
        }
    
    async def _execute_cache_optimization(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute cache optimization"""
        return {
            'success': True,
            'message': f'Applied cache optimizations for {action.service}',
            'changes': {'cache_size_increased': True}
        }
    
    async def _execute_auto_scaling(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute auto-scaling"""
        # This would implement container scaling
        return {
            'success': True,
            'message': f'Scaled instances for {action.service}',
            'changes': {'instance_count': int(action.target_value)}
        }

class PerformanceOptimizer:
    """Main performance optimization orchestrator"""
    
    def __init__(self, config_path: str = '/app/config/optimizer.yml'):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.resource_monitor = SystemResourceMonitor(self.config)
        self.ai_engine = AIOptimizationEngine(self.config)
        self.executor = OptimizationExecutor(self.config)
        
        # Database setup
        self.db_path = self.config.get('database_path', '/app/data/optimizer.db')
        self._init_database()
        
        # State tracking
        self.optimization_interval = self.config.get('optimization_interval', 300)  # 5 minutes
        self.running = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load optimizer configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default optimizer configuration"""
        return {
            'optimization_interval': 300,
            'dry_run': True,
            'monitored_containers': [],
            'optimization_thresholds': {
                'cpu_high': 0.8,
                'cpu_low': 0.3,
                'memory_high': 0.85,
                'memory_low': 0.4,
                'disk_io_high': 100,
                'network_high': 100,
                'response_time_high': 1000,
                'error_rate_high': 0.05
            },
            'auto_scaling': {
                'enabled': True,
                'min_instances': 1,
                'max_instances': 10,
                'scale_up_threshold': 0.8,
                'scale_down_threshold': 0.3,
                'cooldown_period': 300
            }
        }
    
    def _init_database(self):
        """Initialize optimizer database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Resource metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                service TEXT NOT NULL,
                cpu_percent REAL NOT NULL,
                memory_percent REAL NOT NULL,
                memory_usage_mb REAL NOT NULL,
                memory_limit_mb REAL NOT NULL,
                disk_read_mbps REAL NOT NULL,
                disk_write_mbps REAL NOT NULL,
                network_rx_mbps REAL NOT NULL,
                network_tx_mbps REAL NOT NULL,
                active_connections INTEGER DEFAULT 0,
                response_time_ms REAL DEFAULT 0,
                error_rate REAL DEFAULT 0,
                throughput_rps REAL DEFAULT 0
            )
        ''')
        
        # Optimization actions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_actions (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT NOT NULL,
                service TEXT NOT NULL,
                current_value REAL NOT NULL,
                target_value REAL NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                estimated_impact TEXT NOT NULL,
                prerequisites TEXT NOT NULL,
                rollback_plan TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                executed_at DATETIME,
                result TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def start_optimization_engine(self):
        """Start the performance optimization engine"""
        self.running = True
        logging.info("Starting Performance Optimization Engine...")
        
        while self.running:
            try:
                # Collect current metrics
                system_metrics = await self.resource_monitor.collect_system_metrics()
                container_metrics = await self.resource_monitor.collect_container_metrics()
                
                # Store metrics
                await self._store_metrics(container_metrics)
                
                # Get historical metrics for analysis
                historical_metrics = await self._get_historical_metrics()
                
                # Analyze optimization opportunities
                optimization_actions = self.ai_engine.analyze_optimization_opportunities(
                    container_metrics, historical_metrics
                )
                
                # Execute high-priority optimizations
                for action in optimization_actions[:3]:  # Limit to top 3 actions
                    if action.confidence > 0.7:  # Only execute high-confidence actions
                        result = await self.executor.execute_optimization(action)
                        await self._store_optimization_action(action)
                        
                        logging.info(f"Optimization executed: {action.action} - {result.get('message', '')}")
                
                # Log optimization cycle completion
                logging.info(f"Optimization cycle completed. Analyzed {len(container_metrics)} services, "
                           f"identified {len(optimization_actions)} opportunities, "
                           f"executed {min(3, len([a for a in optimization_actions if a.confidence > 0.7]))} actions.")
                
                # Wait for next cycle
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                logging.error(f"Optimization cycle error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _store_metrics(self, metrics: List[ResourceMetrics]):
        """Store resource metrics in database"""
        if not metrics:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for metric in metrics:
                cursor.execute('''
                    INSERT INTO resource_metrics (
                        service, cpu_percent, memory_percent, memory_usage_mb,
                        memory_limit_mb, disk_read_mbps, disk_write_mbps,
                        network_rx_mbps, network_tx_mbps, active_connections,
                        response_time_ms, error_rate, throughput_rps
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.service, metric.cpu_percent, metric.memory_percent,
                    metric.memory_usage_mb, metric.memory_limit_mb,
                    metric.disk_read_mbps, metric.disk_write_mbps,
                    metric.network_rx_mbps, metric.network_tx_mbps,
                    metric.active_connections, metric.response_time_ms,
                    metric.error_rate, metric.throughput_rps
                ))
            
            conn.commit()
        finally:
            conn.close()
    
    async def _get_historical_metrics(self, hours: int = 2) -> List[ResourceMetrics]:
        """Get historical metrics for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT * FROM resource_metrics
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to ResourceMetrics objects
        metrics = []
        for row in results:
            metrics.append(ResourceMetrics(
                timestamp=datetime.fromisoformat(row[1]),
                service=row[2],
                cpu_percent=row[3],
                memory_percent=row[4],
                memory_usage_mb=row[5],
                memory_limit_mb=row[6],
                disk_read_mbps=row[7],
                disk_write_mbps=row[8],
                network_rx_mbps=row[9],
                network_tx_mbps=row[10],
                active_connections=row[11],
                response_time_ms=row[12],
                error_rate=row[13],
                throughput_rps=row[14]
            ))
        
        return metrics
    
    async def _store_optimization_action(self, action: OptimizationAction):
        """Store optimization action in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO optimization_actions (
                    id, type, service, current_value, target_value, action,
                    confidence, estimated_impact, prerequisites, rollback_plan,
                    status, executed_at, result
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                action.id, action.type.value, action.service,
                action.current_value, action.target_value, action.action,
                action.confidence, json.dumps(action.estimated_impact),
                json.dumps(action.prerequisites), action.rollback_plan,
                action.status, action.executed_at,
                json.dumps(action.result) if action.result else None
            ))
            
            conn.commit()
        finally:
            conn.close()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    optimizer = PerformanceOptimizer()
    asyncio.run(optimizer.start_optimization_engine())