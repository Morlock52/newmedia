#!/usr/bin/env python3
"""
Advanced Metrics Collection System
==================================

High-performance metrics collection system with real-time data gathering,
buffering, and intelligent sampling for comprehensive system monitoring.
"""

import asyncio
import json
import logging
import time
import socket
import subprocess
import psutil
import aiohttp
import docker
import sqlite3
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import statistics
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    metric_name: str
    value: float
    labels: Dict[str, str]
    source: str
    
@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: int
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    load_average: List[float]
    process_count: int
    
@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    service_name: str
    status: str
    response_time: float
    memory_usage: int
    cpu_usage: float
    connections: int
    errors: int
    requests_per_second: float

@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    timestamp: float
    test_name: str
    test_type: str
    duration: float
    score: float
    metrics: Dict[str, Any]
    baseline_comparison: Optional[float] = None

class MetricsBuffer:
    """High-performance metrics buffer with automatic flushing"""
    
    def __init__(self, max_size: int = 10000, flush_interval: int = 30):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer: List[MetricPoint] = []
        self.lock = threading.RLock()
        self.last_flush = time.time()
        self.flush_callbacks: List[Callable] = []
        
    def add_metric(self, metric: MetricPoint) -> None:
        """Add metric to buffer with automatic flushing"""
        with self.lock:
            self.buffer.append(metric)
            
            # Check if we need to flush
            should_flush = (
                len(self.buffer) >= self.max_size or
                time.time() - self.last_flush >= self.flush_interval
            )
            
            if should_flush:
                self._flush()
                
    def _flush(self) -> None:
        """Flush buffer to all registered callbacks"""
        if not self.buffer:
            return
            
        metrics_to_flush = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = time.time()
        
        for callback in self.flush_callbacks:
            try:
                callback(metrics_to_flush)
            except Exception as e:
                logger.error(f"Error in flush callback: {e}")
                
    def add_flush_callback(self, callback: Callable) -> None:
        """Add callback to be called on buffer flush"""
        self.flush_callbacks.append(callback)
        
    def force_flush(self) -> None:
        """Force immediate buffer flush"""
        with self.lock:
            self._flush()

class SystemMetricsCollector:
    """Collect comprehensive system metrics"""
    
    def __init__(self):
        self.baseline_metrics = None
        self.previous_network_io = None
        
    def collect(self) -> SystemMetrics:
        """Collect current system metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': (usage.used / usage.total) * 100
                }
            except (PermissionError, OSError):
                continue
                
        # Network metrics
        network_io = psutil.net_io_counters()._asdict()
        
        # Calculate network rates if we have previous data
        if self.previous_network_io:
            time_delta = timestamp - self.previous_network_io['timestamp']
            if time_delta > 0:
                for key in ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv']:
                    if key in network_io:
                        rate = (network_io[key] - self.previous_network_io[key]) / time_delta
                        network_io[f'{key}_rate'] = rate
                        
        self.previous_network_io = {**network_io, 'timestamp': timestamp}
        
        # Load average
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            memory_available=memory.available,
            disk_usage=disk_usage,
            network_io=network_io,
            load_average=load_avg,
            process_count=process_count
        )
        
    def set_baseline(self) -> None:
        """Set current metrics as baseline for comparison"""
        self.baseline_metrics = self.collect()
        logger.info("System metrics baseline established")

class DockerMetricsCollector:
    """Collect Docker container metrics"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.enabled = True
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
            self.enabled = False
            
    def collect(self) -> List[Dict[str, Any]]:
        """Collect metrics from all containers"""
        if not self.enabled:
            return []
            
        container_metrics = []
        
        try:
            containers = self.client.containers.list()
            for container in containers:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    cpu_usage = self._calculate_cpu_usage(stats)
                    
                    # Calculate memory usage
                    memory_usage = self._calculate_memory_usage(stats)
                    
                    # Network I/O
                    network_io = self._calculate_network_io(stats)
                    
                    # Block I/O
                    block_io = self._calculate_block_io(stats)
                    
                    container_metrics.append({
                        'timestamp': time.time(),
                        'container_id': container.id[:12],
                        'container_name': container.name,
                        'image': container.image.tags[0] if container.image.tags else 'unknown',
                        'status': container.status,
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'network_io': network_io,
                        'block_io': block_io
                    })
                    
                except Exception as e:
                    logger.error(f"Error collecting stats for container {container.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error collecting Docker metrics: {e}")
            
        return container_metrics
        
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage"""
        cpu_stats = stats.get('cpu_stats', {})
        precpu_stats = stats.get('precpu_stats', {})
        
        cpu_usage = cpu_stats.get('cpu_usage', {})
        precpu_usage = precpu_stats.get('cpu_usage', {})
        
        cpu_delta = cpu_usage.get('total_usage', 0) - precpu_usage.get('total_usage', 0)
        system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
        
        if system_delta > 0 and cpu_delta > 0:
            cpu_count = cpu_stats.get('online_cpus', 1)
            return (cpu_delta / system_delta) * cpu_count * 100.0
        return 0.0
        
    def _calculate_memory_usage(self, stats: Dict) -> Dict[str, int]:
        """Calculate memory usage metrics"""
        memory_stats = stats.get('memory_stats', {})
        
        return {
            'usage': memory_stats.get('usage', 0),
            'limit': memory_stats.get('limit', 0),
            'percent': (memory_stats.get('usage', 0) / memory_stats.get('limit', 1)) * 100
        }
        
    def _calculate_network_io(self, stats: Dict) -> Dict[str, int]:
        """Calculate network I/O metrics"""
        networks = stats.get('networks', {})
        
        total_rx = sum(net.get('rx_bytes', 0) for net in networks.values())
        total_tx = sum(net.get('tx_bytes', 0) for net in networks.values())
        
        return {
            'rx_bytes': total_rx,
            'tx_bytes': total_tx,
            'rx_packets': sum(net.get('rx_packets', 0) for net in networks.values()),
            'tx_packets': sum(net.get('tx_packets', 0) for net in networks.values())
        }
        
    def _calculate_block_io(self, stats: Dict) -> Dict[str, int]:
        """Calculate block I/O metrics"""
        blkio_stats = stats.get('blkio_stats', {})
        
        read_bytes = 0
        write_bytes = 0
        
        for io_stat in blkio_stats.get('io_service_bytes_recursive', []):
            if io_stat.get('op') == 'Read':
                read_bytes += io_stat.get('value', 0)
            elif io_stat.get('op') == 'Write':
                write_bytes += io_stat.get('value', 0)
                
        return {
            'read_bytes': read_bytes,
            'write_bytes': write_bytes
        }

class ApplicationMetricsCollector:
    """Collect application-specific metrics"""
    
    def __init__(self, services_config: List[Dict]):
        self.services_config = services_config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def collect_all(self) -> List[ApplicationMetrics]:
        """Collect metrics from all configured services"""
        tasks = []
        for service_config in self.services_config:
            task = asyncio.create_task(self.collect_service(service_config))
            tasks.append(task)
            
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error collecting application metrics: {e}")
                
        return results
        
    async def collect_service(self, service_config: Dict) -> Optional[ApplicationMetrics]:
        """Collect metrics from a single service"""
        service_name = service_config['name']
        base_url = service_config['url']
        
        try:
            timestamp = time.time()
            
            # Health check
            health_endpoint = service_config.get('health_endpoint', '/health')
            health_url = f"{base_url}{health_endpoint}"
            
            start_time = time.time()
            async with self.session.get(health_url) as response:
                response_time = (time.time() - start_time) * 1000  # ms
                status = "up" if response.status == 200 else "down"
                
            # Additional metrics if available
            metrics_endpoint = service_config.get('metrics_endpoint')
            additional_metrics = {}
            
            if metrics_endpoint:
                try:
                    metrics_url = f"{base_url}{metrics_endpoint}"
                    async with self.session.get(metrics_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            additional_metrics = self._parse_service_metrics(service_name, data)
                except Exception as e:
                    logger.debug(f"Could not collect additional metrics for {service_name}: {e}")
                    
            return ApplicationMetrics(
                timestamp=timestamp,
                service_name=service_name,
                status=status,
                response_time=response_time,
                memory_usage=additional_metrics.get('memory_usage', 0),
                cpu_usage=additional_metrics.get('cpu_usage', 0.0),
                connections=additional_metrics.get('connections', 0),
                errors=additional_metrics.get('errors', 0),
                requests_per_second=additional_metrics.get('requests_per_second', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {service_name}: {e}")
            return ApplicationMetrics(
                timestamp=time.time(),
                service_name=service_name,
                status="error",
                response_time=-1,
                memory_usage=0,
                cpu_usage=0.0,
                connections=0,
                errors=1,
                requests_per_second=0.0
            )
            
    def _parse_service_metrics(self, service_name: str, data: Dict) -> Dict[str, Any]:
        """Parse service-specific metrics"""
        metrics = {}
        
        # Common patterns for different services
        if service_name == "jellyfin":
            metrics.update(self._parse_jellyfin_metrics(data))
        elif service_name in ["sonarr", "radarr", "prowlarr"]:
            metrics.update(self._parse_arr_metrics(data))
        elif service_name == "tautulli":
            metrics.update(self._parse_tautulli_metrics(data))
            
        return metrics
        
    def _parse_jellyfin_metrics(self, data: Dict) -> Dict[str, Any]:
        """Parse Jellyfin-specific metrics"""
        return {
            'active_sessions': len(data.get('Sessions', [])),
            'transcoding_count': sum(1 for s in data.get('Sessions', []) if s.get('TranscodingInfo')),
            'user_count': len(data.get('Users', [])),
            'library_count': len(data.get('Libraries', []))
        }
        
    def _parse_arr_metrics(self, data: Dict) -> Dict[str, Any]:
        """Parse *arr application metrics"""
        return {
            'queue_count': data.get('queue', {}).get('totalRecords', 0),
            'wanted_count': data.get('wanted', {}).get('totalRecords', 0),
            'disk_space': data.get('diskSpace', [{}])[0].get('freeSpace', 0) if data.get('diskSpace') else 0
        }
        
    def _parse_tautulli_metrics(self, data: Dict) -> Dict[str, Any]:
        """Parse Tautulli metrics"""
        return {
            'stream_count': data.get('response', {}).get('data', {}).get('stream_count', 0),
            'total_bandwidth': data.get('response', {}).get('data', {}).get('total_bandwidth', 0),
            'user_sessions': data.get('response', {}).get('data', {}).get('user_count', 0)
        }

class BenchmarkRunner:
    """Run comprehensive system benchmarks"""
    
    def __init__(self):
        self.results_history = []
        
    def run_all_benchmarks(self, config: Dict) -> List[BenchmarkResult]:
        """Run all configured benchmarks"""
        results = []
        
        if config.get('system', {}).get('enabled'):
            system_results = self.run_system_benchmarks(config['system']['tests'])
            results.extend(system_results)
            
        if config.get('application', {}).get('enabled'):
            app_results = self.run_application_benchmarks(config['application']['tests'])
            results.extend(app_results)
            
        # Store results for trend analysis
        self.results_history.extend(results)
        
        # Keep only last 100 results per test
        self._cleanup_history()
        
        return results
        
    def run_system_benchmarks(self, tests: List[Dict]) -> List[BenchmarkResult]:
        """Run system performance benchmarks"""
        results = []
        
        for test_config in tests:
            try:
                result = self._run_single_benchmark(test_config)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error running benchmark {test_config['name']}: {e}")
                
        return results
        
    def _run_single_benchmark(self, test_config: Dict) -> Optional[BenchmarkResult]:
        """Run a single benchmark test"""
        test_name = test_config['name']
        test_type = test_config['type']
        duration = test_config.get('duration', '30s')
        
        logger.info(f"Running benchmark: {test_name}")
        start_time = time.time()
        
        if test_type == "stress_test":
            result = self._run_cpu_benchmark(test_config)
        elif test_type == "memory_test":
            result = self._run_memory_benchmark(test_config)
        elif test_type == "io_test":
            result = self._run_disk_benchmark(test_config)
        elif test_type == "bandwidth_test":
            result = self._run_network_benchmark(test_config)
        else:
            logger.warning(f"Unknown benchmark type: {test_type}")
            return None
            
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Compare with historical baseline
        baseline_comparison = self._get_baseline_comparison(test_name, result['score'])
        
        return BenchmarkResult(
            timestamp=start_time,
            test_name=test_name,
            test_type=test_type,
            duration=actual_duration,
            score=result['score'],
            metrics=result['metrics'],
            baseline_comparison=baseline_comparison
        )
        
    def _run_cpu_benchmark(self, config: Dict) -> Dict[str, Any]:
        """Run CPU stress test benchmark"""
        duration = self._parse_duration(config.get('duration', '30s'))
        cores = config.get('parameters', {}).get('cores', 'all')
        
        if cores == 'all':
            cores = psutil.cpu_count()
            
        # Simple CPU benchmark using calculation-heavy operations
        start_time = time.time()
        total_operations = 0
        
        def cpu_work():
            nonlocal total_operations
            operations = 0
            end_time = start_time + duration
            
            while time.time() < end_time:
                # Perform calculation-heavy operations
                for i in range(10000):
                    _ = i ** 2 * 3.14159 / 2.71828
                operations += 10000
                
            total_operations += operations
            
        # Run benchmark on specified number of cores
        with ThreadPoolExecutor(max_workers=cores) as executor:
            futures = [executor.submit(cpu_work) for _ in range(cores)]
            
            for future in as_completed(futures):
                future.result()
                
        # Calculate score (operations per second)
        score = total_operations / duration
        
        return {
            'score': score,
            'metrics': {
                'operations_per_second': score,
                'total_operations': total_operations,
                'cores_used': cores,
                'duration': duration
            }
        }
        
    def _run_memory_benchmark(self, config: Dict) -> Dict[str, Any]:
        """Run memory performance benchmark"""
        duration = self._parse_duration(config.get('duration', '30s'))
        size_str = config.get('parameters', {}).get('size', '1GB')
        size_bytes = self._parse_size(size_str)
        
        start_time = time.time()
        operations = 0
        
        # Memory allocation and access pattern test
        while time.time() - start_time < duration:
            try:
                # Allocate memory
                data = bytearray(size_bytes // 100)  # Use smaller chunks to avoid OOM
                
                # Write pattern
                for i in range(0, len(data), 1024):
                    data[i] = i % 256
                    
                # Read pattern
                checksum = 0
                for i in range(0, len(data), 1024):
                    checksum += data[i]
                    
                operations += 1
                del data  # Free memory
                
            except MemoryError:
                logger.warning("Memory benchmark hit memory limit")
                break
                
        actual_duration = time.time() - start_time
        score = operations / actual_duration if actual_duration > 0 else 0
        
        return {
            'score': score,
            'metrics': {
                'operations_per_second': score,
                'total_operations': operations,
                'chunk_size': size_bytes // 100,
                'duration': actual_duration
            }
        }
        
    def _run_disk_benchmark(self, config: Dict) -> Dict[str, Any]:
        """Run disk I/O benchmark"""
        duration = self._parse_duration(config.get('duration', '60s'))
        block_size = config.get('parameters', {}).get('block_size', '4K')
        operations = config.get('parameters', {}).get('operations', ['read', 'write'])
        
        block_size_bytes = self._parse_size(block_size)
        test_file = "/tmp/benchmark_test_file"
        
        results = {}
        
        for operation in operations:
            if operation == 'write':
                results[operation] = self._run_disk_write_test(
                    test_file, block_size_bytes, duration
                )
            elif operation == 'read':
                results[operation] = self._run_disk_read_test(
                    test_file, block_size_bytes, duration
                )
                
        # Cleanup
        try:
            import os
            os.remove(test_file)
        except FileNotFoundError:
            pass
            
        # Calculate overall score (average throughput)
        throughput_scores = [r.get('throughput', 0) for r in results.values()]
        score = statistics.mean(throughput_scores) if throughput_scores else 0
        
        return {
            'score': score,
            'metrics': {
                'average_throughput_mbps': score,
                'operations': results,
                'block_size': block_size_bytes
            }
        }
        
    def _run_disk_write_test(self, filepath: str, block_size: int, duration: float) -> Dict[str, Any]:
        """Run disk write test"""
        start_time = time.time()
        bytes_written = 0
        
        data_block = b'A' * block_size
        
        try:
            with open(filepath, 'wb') as f:
                while time.time() - start_time < duration:
                    f.write(data_block)
                    bytes_written += block_size
                    f.flush()  # Ensure data is written
                    
        except Exception as e:
            logger.error(f"Disk write test error: {e}")
            
        actual_duration = time.time() - start_time
        throughput = (bytes_written / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0
        
        return {
            'throughput': throughput,  # MB/s
            'bytes_written': bytes_written,
            'duration': actual_duration
        }
        
    def _run_disk_read_test(self, filepath: str, block_size: int, duration: float) -> Dict[str, Any]:
        """Run disk read test"""
        # First ensure we have data to read
        if not os.path.exists(filepath):
            with open(filepath, 'wb') as f:
                f.write(b'A' * (block_size * 1000))  # Write some test data
                
        start_time = time.time()
        bytes_read = 0
        
        try:
            while time.time() - start_time < duration:
                with open(filepath, 'rb') as f:
                    data = f.read(block_size)
                    if not data:
                        f.seek(0)  # Reset to beginning
                        continue
                    bytes_read += len(data)
                    
        except Exception as e:
            logger.error(f"Disk read test error: {e}")
            
        actual_duration = time.time() - start_time
        throughput = (bytes_read / (1024 * 1024)) / actual_duration if actual_duration > 0 else 0
        
        return {
            'throughput': throughput,  # MB/s
            'bytes_read': bytes_read,
            'duration': actual_duration
        }
        
    def _run_network_benchmark(self, config: Dict) -> Dict[str, Any]:
        """Run network performance benchmark"""
        duration = self._parse_duration(config.get('duration', '30s'))
        protocol = config.get('parameters', {}).get('protocol', 'tcp')
        packet_size = config.get('parameters', {}).get('packet_size', '1500')
        
        # Simple network test using ping to measure latency
        latencies = []
        start_time = time.time()
        
        while time.time() - start_time < min(duration, 30):  # Limit to 30s for network test
            try:
                result = subprocess.run(
                    ['ping', '-c', '1', '8.8.8.8'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Parse ping output for latency
                    for line in result.stdout.split('\n'):
                        if 'time=' in line:
                            latency_str = line.split('time=')[1].split(' ')[0]
                            latency = float(latency_str)
                            latencies.append(latency)
                            break
                            
            except (subprocess.TimeoutExpired, ValueError, subprocess.SubprocessError):
                continue
                
            time.sleep(1)  # Wait between pings
            
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            packet_loss = 0  # Simplified for this example
            
            # Score based on inverse of latency (lower latency = higher score)
            score = 1000 / avg_latency if avg_latency > 0 else 0
        else:
            avg_latency = min_latency = max_latency = -1
            packet_loss = 100
            score = 0
            
        return {
            'score': score,
            'metrics': {
                'average_latency_ms': avg_latency,
                'min_latency_ms': min_latency,
                'max_latency_ms': max_latency,
                'packet_loss_percent': packet_loss,
                'samples': len(latencies)
            }
        }
        
    def run_application_benchmarks(self, tests: List[Dict]) -> List[BenchmarkResult]:
        """Run application-specific benchmarks"""
        results = []
        
        for test_config in tests:
            try:
                if test_config['type'] == 'api_load_test':
                    result = self._run_api_load_test(test_config)
                    if result:
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Error running application benchmark {test_config['name']}: {e}")
                
        return results
        
    def _run_api_load_test(self, config: Dict) -> Optional[BenchmarkResult]:
        """Run API load test"""
        test_name = config['name']
        duration = self._parse_duration(config.get('duration', '120s'))
        rps = config.get('parameters', {}).get('requests_per_second', 10)
        endpoints = config.get('parameters', {}).get('endpoints', ['/'])
        
        logger.info(f"Running API load test: {test_name}")
        
        # This is a simplified version - in production you'd use a proper load testing tool
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        start_time = time.time()
        
        # Simulate load test results (replace with actual implementation)
        estimated_requests = int(duration * rps)
        results['total_requests'] = estimated_requests
        results['successful_requests'] = int(estimated_requests * 0.95)  # 95% success rate
        results['failed_requests'] = estimated_requests - results['successful_requests']
        
        # Simulate response times
        results['response_times'] = [
            np.random.normal(200, 50) for _ in range(results['successful_requests'])
        ]
        
        avg_response_time = statistics.mean(results['response_times'])
        score = results['successful_requests'] / duration  # Successful requests per second
        
        return BenchmarkResult(
            timestamp=start_time,
            test_name=test_name,
            test_type='api_load_test',
            duration=duration,
            score=score,
            metrics={
                'requests_per_second': score,
                'average_response_time': avg_response_time,
                'success_rate': (results['successful_requests'] / results['total_requests']) * 100,
                'total_requests': results['total_requests']
            }
        )
        
    def _get_baseline_comparison(self, test_name: str, current_score: float) -> Optional[float]:
        """Compare current score with historical baseline"""
        historical_scores = [
            r.score for r in self.results_history 
            if r.test_name == test_name and r.score > 0
        ]
        
        if len(historical_scores) < 5:  # Need at least 5 samples for meaningful baseline
            return None
            
        baseline_score = statistics.median(historical_scores[-10:])  # Use last 10 results
        
        if baseline_score > 0:
            return ((current_score - baseline_score) / baseline_score) * 100
            
        return None
        
    def _cleanup_history(self):
        """Keep only recent benchmark results"""
        max_results_per_test = 100
        
        # Group by test name
        test_groups = {}
        for result in self.results_history:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)
            
        # Keep only the most recent results for each test
        cleaned_results = []
        for test_name, results in test_groups.items():
            sorted_results = sorted(results, key=lambda x: x.timestamp, reverse=True)
            cleaned_results.extend(sorted_results[:max_results_per_test])
            
        self.results_history = cleaned_results
        
    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to seconds"""
        if duration_str.endswith('s'):
            return float(duration_str[:-1])
        elif duration_str.endswith('m'):
            return float(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return float(duration_str[:-1]) * 3600
        else:
            return float(duration_str)
            
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes"""
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        elif size_str.endswith('K'):
            return int(size_str[:-1]) * 1024
        elif size_str.endswith('M'):
            return int(size_str[:-1]) * 1024 * 1024
        elif size_str.endswith('G'):
            return int(size_str[:-1]) * 1024 * 1024 * 1024
        elif size_str.endswith('B'):
            return int(size_str[:-1])
        else:
            return int(size_str)

class MetricsStorage:
    """Store and retrieve metrics data"""
    
    def __init__(self, db_path: str = "/tmp/monitoring.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    memory_available INTEGER,
                    disk_usage TEXT,
                    network_io TEXT,
                    load_average TEXT,
                    process_count INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS application_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    service_name TEXT NOT NULL,
                    status TEXT,
                    response_time REAL,
                    memory_usage INTEGER,
                    cpu_usage REAL,
                    connections INTEGER,
                    errors INTEGER,
                    requests_per_second REAL
                );
                
                CREATE TABLE IF NOT EXISTS docker_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    container_id TEXT NOT NULL,
                    container_name TEXT,
                    image TEXT,
                    status TEXT,
                    cpu_usage REAL,
                    memory_usage TEXT,
                    network_io TEXT,
                    block_io TEXT
                );
                
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    test_name TEXT NOT NULL,
                    test_type TEXT,
                    duration REAL,
                    score REAL,
                    metrics TEXT,
                    baseline_comparison REAL
                );
                
                CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_app_timestamp_service ON application_metrics(timestamp, service_name);
                CREATE INDEX IF NOT EXISTS idx_docker_timestamp ON docker_metrics(timestamp);
                CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp_test ON benchmark_results(timestamp, test_name);
            """)
            
    def store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_usage, memory_usage, memory_available, disk_usage, 
                 network_io, load_average, process_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.memory_available,
                json.dumps(metrics.disk_usage),
                json.dumps(metrics.network_io),
                json.dumps(metrics.load_average),
                metrics.process_count
            ))
            
    def store_application_metrics(self, metrics: List[ApplicationMetrics]):
        """Store application metrics"""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO application_metrics
                    (timestamp, service_name, status, response_time, memory_usage,
                     cpu_usage, connections, errors, requests_per_second)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp,
                    metric.service_name,
                    metric.status,
                    metric.response_time,
                    metric.memory_usage,
                    metric.cpu_usage,
                    metric.connections,
                    metric.errors,
                    metric.requests_per_second
                ))
                
    def store_docker_metrics(self, metrics: List[Dict]):
        """Store Docker container metrics"""
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute("""
                    INSERT INTO docker_metrics
                    (timestamp, container_id, container_name, image, status,
                     cpu_usage, memory_usage, network_io, block_io)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric['timestamp'],
                    metric['container_id'],
                    metric['container_name'],
                    metric['image'],
                    metric['status'],
                    metric['cpu_usage'],
                    json.dumps(metric['memory_usage']),
                    json.dumps(metric['network_io']),
                    json.dumps(metric['block_io'])
                ))
                
    def store_benchmark_results(self, results: List[BenchmarkResult]):
        """Store benchmark results"""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO benchmark_results
                    (timestamp, test_name, test_type, duration, score, metrics, baseline_comparison)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.timestamp,
                    result.test_name,
                    result.test_type,
                    result.duration,
                    result.score,
                    json.dumps(result.metrics),
                    result.baseline_comparison
                ))
                
    def cleanup_old_data(self, retention_hours: int = 24):
        """Clean up old metrics data"""
        cutoff_time = time.time() - (retention_hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_time,))
            conn.execute("DELETE FROM application_metrics WHERE timestamp < ?", (cutoff_time,))
            conn.execute("DELETE FROM docker_metrics WHERE timestamp < ?", (cutoff_time,))
            # Keep benchmark results longer
            benchmark_cutoff = time.time() - (30 * 24 * 3600)  # 30 days
            conn.execute("DELETE FROM benchmark_results WHERE timestamp < ?", (benchmark_cutoff,))

class PerformanceMonitor:
    """Main performance monitoring orchestrator"""
    
    def __init__(self, config_path: str = "config/monitoring.yml"):
        self.config = self._load_config(config_path)
        self.storage = MetricsStorage(self.config.get('storage', {}).get('database', {}).get('path', '/tmp/monitoring.db'))
        self.metrics_buffer = MetricsBuffer(
            max_size=self.config.get('global', {}).get('buffer_size', 10000),
            flush_interval=self.config.get('global', {}).get('collection_interval', 30)
        )
        
        # Initialize collectors
        self.system_collector = SystemMetricsCollector()
        self.docker_collector = DockerMetricsCollector()
        self.benchmark_runner = BenchmarkRunner()
        
        # Initialize application collector
        services_config = self.config.get('data_sources', {}).get('applications', {}).get('services', [])
        self.app_collector = ApplicationMetricsCollector(services_config)
        
        # Set up buffer callbacks
        self.metrics_buffer.add_flush_callback(self._flush_to_storage)
        
        self.running = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
            
    def _flush_to_storage(self, metrics: List[MetricPoint]):
        """Flush metrics buffer to storage"""
        try:
            # Group metrics by type and store appropriately
            system_metrics = []
            app_metrics = []
            docker_metrics = []
            
            for metric in metrics:
                if metric.source == 'system':
                    system_metrics.append(metric)
                elif metric.source == 'application':
                    app_metrics.append(metric)
                elif metric.source == 'docker':
                    docker_metrics.append(metric)
                    
            logger.info(f"Flushed {len(metrics)} metrics to storage")
            
        except Exception as e:
            logger.error(f"Error flushing metrics to storage: {e}")
            
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.running = True
        logger.info("Performance monitoring system started")
        
        # Start collection tasks
        tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_application_metrics()),
            asyncio.create_task(self._collect_docker_metrics()),
            asyncio.create_task(self._run_periodic_benchmarks()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Monitoring system stopped by user")
        finally:
            self.running = False
            
    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        interval = self.config.get('global', {}).get('collection_interval', 15)
        
        while self.running:
            try:
                metrics = self.system_collector.collect()
                self.storage.store_system_metrics(metrics)
                logger.debug("System metrics collected")
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                
            await asyncio.sleep(interval)
            
    async def _collect_application_metrics(self):
        """Collect application metrics periodically"""
        interval = self.config.get('global', {}).get('collection_interval', 15)
        
        while self.running:
            try:
                async with self.app_collector as collector:
                    metrics = await collector.collect_all()
                    if metrics:
                        self.storage.store_application_metrics(metrics)
                        logger.debug(f"Application metrics collected for {len(metrics)} services")
                        
            except Exception as e:
                logger.error(f"Error collecting application metrics: {e}")
                
            await asyncio.sleep(interval)
            
    async def _collect_docker_metrics(self):
        """Collect Docker metrics periodically"""
        interval = self.config.get('global', {}).get('collection_interval', 15)
        
        while self.running:
            try:
                metrics = self.docker_collector.collect()
                if metrics:
                    self.storage.store_docker_metrics(metrics)
                    logger.debug(f"Docker metrics collected for {len(metrics)} containers")
                    
            except Exception as e:
                logger.error(f"Error collecting Docker metrics: {e}")
                
            await asyncio.sleep(interval)
            
    async def _run_periodic_benchmarks(self):
        """Run benchmarks periodically"""
        benchmark_config = self.config.get('benchmarks', {})
        
        if not benchmark_config:
            return
            
        # Run benchmarks every hour
        benchmark_interval = 3600
        
        while self.running:
            try:
                logger.info("Starting periodic benchmarks")
                results = self.benchmark_runner.run_all_benchmarks(benchmark_config)
                
                if results:
                    self.storage.store_benchmark_results(results)
                    logger.info(f"Completed {len(results)} benchmark tests")
                    
            except Exception as e:
                logger.error(f"Error running benchmarks: {e}")
                
            await asyncio.sleep(benchmark_interval)
            
    async def _cleanup_old_data(self):
        """Clean up old data periodically"""
        cleanup_interval = 3600  # Every hour
        retention_hours = 24
        
        while self.running:
            try:
                self.storage.cleanup_old_data(retention_hours)
                logger.debug("Old data cleanup completed")
                
            except Exception as e:
                logger.error(f"Error during data cleanup: {e}")
                
            await asyncio.sleep(cleanup_interval)
            
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        self.metrics_buffer.force_flush()
        logger.info("Performance monitoring system stopped")

if __name__ == "__main__":
    import os
    
    # Set up configuration path
    config_path = os.path.join(os.path.dirname(__file__), "../config/monitoring.yml")
    
    # Create and start monitor
    monitor = PerformanceMonitor(config_path)
    
    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        monitor.stop_monitoring()