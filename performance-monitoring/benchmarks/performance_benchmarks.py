#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite
=========================================

Advanced benchmarking system for testing system performance, application
responsiveness, and overall infrastructure health with detailed analytics.
"""

import asyncio
import json
import logging
import time
import subprocess
import requests
import threading
import statistics
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    name: str
    test_type: str
    duration: int  # seconds
    parameters: Dict[str, Any]
    baseline_comparison: bool = True
    generate_report: bool = True

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    name: str
    test_type: str
    start_time: float
    end_time: float
    duration: float
    score: float
    unit: str
    metrics: Dict[str, Any]
    status: str  # success, failed, timeout
    error_message: Optional[str] = None
    baseline_comparison: Optional[float] = None
    percentile_99: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_50: Optional[float] = None

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    start_time: float
    end_time: float
    total_duration: float
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    system_info: Dict[str, Any]

class CPUBenchmark:
    """CPU performance benchmarking"""
    
    def __init__(self):
        self.name = "CPU Performance Test"
        
    def run_single_core_test(self, duration: int = 30) -> BenchmarkResult:
        """Run single-core CPU intensive test"""
        logger.info("Running single-core CPU benchmark")
        
        start_time = time.time()
        operations = 0
        
        # CPU-intensive calculation
        end_time = start_time + duration
        while time.time() < end_time:
            for i in range(100000):
                _ = i ** 2 * 3.14159 / 2.71828
            operations += 100000
            
        actual_duration = time.time() - start_time
        score = operations / actual_duration  # Operations per second
        
        return BenchmarkResult(
            name="Single Core CPU",
            test_type="cpu_single",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=score,
            unit="ops/sec",
            metrics={
                "total_operations": operations,
                "cpu_utilization": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent
            },
            status="success"
        )
        
    def run_multi_core_test(self, duration: int = 30, cores: Optional[int] = None) -> BenchmarkResult:
        """Run multi-core CPU test"""
        if cores is None:
            cores = psutil.cpu_count()
            
        logger.info(f"Running multi-core CPU benchmark with {cores} cores")
        
        start_time = time.time()
        total_operations = 0
        results = []
        
        def cpu_worker():
            operations = 0
            worker_start = time.time()
            worker_end = worker_start + duration
            
            while time.time() < worker_end:
                for i in range(50000):
                    _ = i ** 2 * 3.14159 / 2.71828
                operations += 50000
                
            return operations
            
        # Run parallel workers
        with ThreadPoolExecutor(max_workers=cores) as executor:
            futures = [executor.submit(cpu_worker) for _ in range(cores)]
            
            for future in as_completed(futures):
                try:
                    operations = future.result()
                    total_operations += operations
                    results.append(operations)
                except Exception as e:
                    logger.error(f"CPU worker failed: {e}")
                    
        actual_duration = time.time() - start_time
        score = total_operations / actual_duration
        
        return BenchmarkResult(
            name="Multi Core CPU",
            test_type="cpu_multi",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=score,
            unit="ops/sec",
            metrics={
                "total_operations": total_operations,
                "cores_used": cores,
                "operations_per_core": total_operations / cores,
                "cpu_utilization": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "individual_results": results
            },
            status="success"
        )
        
    def run_prime_calculation_test(self, duration: int = 30) -> BenchmarkResult:
        """Run prime number calculation benchmark"""
        logger.info("Running prime calculation benchmark")
        
        start_time = time.time()
        primes_found = 0
        current_number = 2
        
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
            
        end_time = start_time + duration
        while time.time() < end_time:
            if is_prime(current_number):
                primes_found += 1
            current_number += 1
            
        actual_duration = time.time() - start_time
        score = primes_found / actual_duration
        
        return BenchmarkResult(
            name="Prime Calculation",
            test_type="cpu_prime",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=score,
            unit="primes/sec",
            metrics={
                "primes_found": primes_found,
                "highest_number_tested": current_number - 1,
                "cpu_utilization": psutil.cpu_percent()
            },
            status="success"
        )

class MemoryBenchmark:
    """Memory performance benchmarking"""
    
    def __init__(self):
        self.name = "Memory Performance Test"
        
    def run_allocation_test(self, duration: int = 30, chunk_size: int = 1024*1024) -> BenchmarkResult:
        """Test memory allocation/deallocation performance"""
        logger.info("Running memory allocation benchmark")
        
        start_time = time.time()
        allocations = 0
        
        end_time = start_time + duration
        while time.time() < end_time:
            try:
                # Allocate memory chunk
                data = bytearray(chunk_size)
                
                # Write some data
                for i in range(0, len(data), 1024):
                    data[i] = i % 256
                    
                # Deallocate
                del data
                allocations += 1
                
            except MemoryError:
                logger.warning("Memory allocation failed - system limit reached")
                break
                
        actual_duration = time.time() - start_time
        score = allocations / actual_duration
        
        memory_info = psutil.virtual_memory()
        
        return BenchmarkResult(
            name="Memory Allocation",
            test_type="memory_allocation",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=score,
            unit="allocs/sec",
            metrics={
                "total_allocations": allocations,
                "chunk_size_mb": chunk_size / (1024*1024),
                "total_memory_allocated_gb": (allocations * chunk_size) / (1024**3),
                "memory_usage_percent": memory_info.percent,
                "available_memory_gb": memory_info.available / (1024**3)
            },
            status="success"
        )
        
    def run_bandwidth_test(self, duration: int = 30, array_size: int = 10*1024*1024) -> BenchmarkResult:
        """Test memory bandwidth"""
        logger.info("Running memory bandwidth benchmark")
        
        # Create test arrays
        array_a = np.random.random(array_size).astype(np.float32)
        array_b = np.random.random(array_size).astype(np.float32)
        
        start_time = time.time()
        operations = 0
        
        end_time = start_time + duration
        while time.time() < end_time:
            # Memory intensive operations
            result = array_a + array_b
            result = result * 1.5
            result = np.sqrt(result)
            operations += 1
            
        actual_duration = time.time() - start_time
        bytes_processed = operations * array_size * 4 * 3  # 3 operations, 4 bytes per float32
        bandwidth_gbps = (bytes_processed / (1024**3)) / actual_duration
        
        return BenchmarkResult(
            name="Memory Bandwidth",
            test_type="memory_bandwidth",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=bandwidth_gbps,
            unit="GB/s",
            metrics={
                "operations": operations,
                "array_size_mb": (array_size * 4) / (1024*1024),
                "total_bytes_processed_gb": bytes_processed / (1024**3),
                "memory_usage_percent": psutil.virtual_memory().percent
            },
            status="success"
        )

class DiskBenchmark:
    """Disk I/O performance benchmarking"""
    
    def __init__(self, test_dir: str = "/tmp"):
        self.test_dir = Path(test_dir)
        self.test_file = self.test_dir / "benchmark_test_file"
        
    def run_sequential_write_test(self, duration: int = 60, block_size: int = 1024*1024) -> BenchmarkResult:
        """Test sequential write performance"""
        logger.info("Running sequential write benchmark")
        
        data_block = b'A' * block_size
        start_time = time.time()
        bytes_written = 0
        
        try:
            with open(self.test_file, 'wb') as f:
                end_time = start_time + duration
                while time.time() < end_time:
                    f.write(data_block)
                    f.flush()
                    bytes_written += block_size
                    
        except Exception as e:
            return BenchmarkResult(
                name="Sequential Write",
                test_type="disk_seq_write",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="MB/s",
                metrics={},
                status="failed",
                error_message=str(e)
            )
        finally:
            # Cleanup
            try:
                self.test_file.unlink(missing_ok=True)
            except:
                pass
                
        actual_duration = time.time() - start_time
        throughput_mbps = (bytes_written / (1024*1024)) / actual_duration
        
        return BenchmarkResult(
            name="Sequential Write",
            test_type="disk_seq_write",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=throughput_mbps,
            unit="MB/s",
            metrics={
                "bytes_written": bytes_written,
                "block_size_kb": block_size / 1024,
                "total_writes": bytes_written // block_size
            },
            status="success"
        )
        
    def run_sequential_read_test(self, duration: int = 60, block_size: int = 1024*1024) -> BenchmarkResult:
        """Test sequential read performance"""
        logger.info("Running sequential read benchmark")
        
        # First, create a test file
        test_file_size = 100 * 1024 * 1024  # 100MB
        data_block = b'A' * block_size
        
        try:
            with open(self.test_file, 'wb') as f:
                bytes_to_write = test_file_size
                while bytes_to_write > 0:
                    write_size = min(block_size, bytes_to_write)
                    f.write(data_block[:write_size])
                    bytes_to_write -= write_size
                    
        except Exception as e:
            return BenchmarkResult(
                name="Sequential Read",
                test_type="disk_seq_read",
                start_time=time.time(),
                end_time=time.time(),
                duration=0,
                score=0,
                unit="MB/s",
                metrics={},
                status="failed",
                error_message=f"Failed to create test file: {e}"
            )
        
        start_time = time.time()
        bytes_read = 0
        
        try:
            end_time = start_time + duration
            while time.time() < end_time:
                with open(self.test_file, 'rb') as f:
                    while True:
                        data = f.read(block_size)
                        if not data:
                            f.seek(0)  # Reset to beginning
                            break
                        bytes_read += len(data)
                        
                        if time.time() >= end_time:
                            break
                            
        except Exception as e:
            return BenchmarkResult(
                name="Sequential Read",
                test_type="disk_seq_read",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="MB/s",
                metrics={},
                status="failed",
                error_message=str(e)
            )
        finally:
            # Cleanup
            try:
                self.test_file.unlink(missing_ok=True)
            except:
                pass
                
        actual_duration = time.time() - start_time
        throughput_mbps = (bytes_read / (1024*1024)) / actual_duration
        
        return BenchmarkResult(
            name="Sequential Read",
            test_type="disk_seq_read",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=throughput_mbps,
            unit="MB/s",
            metrics={
                "bytes_read": bytes_read,
                "block_size_kb": block_size / 1024,
                "total_reads": bytes_read // block_size
            },
            status="success"
        )
        
    def run_random_io_test(self, duration: int = 60, block_size: int = 4096) -> BenchmarkResult:
        """Test random I/O performance"""
        logger.info("Running random I/O benchmark")
        
        # Create test file
        test_file_size = 50 * 1024 * 1024  # 50MB
        
        try:
            with open(self.test_file, 'wb') as f:
                f.write(b'A' * test_file_size)
                
        except Exception as e:
            return BenchmarkResult(
                name="Random I/O",
                test_type="disk_random_io",
                start_time=time.time(),
                end_time=time.time(),
                duration=0,
                score=0,
                unit="IOPS",
                metrics={},
                status="failed",
                error_message=f"Failed to create test file: {e}"
            )
        
        start_time = time.time()
        operations = 0
        read_ops = 0
        write_ops = 0
        
        try:
            with open(self.test_file, 'r+b') as f:
                end_time = start_time + duration
                while time.time() < end_time:
                    # Random position
                    pos = np.random.randint(0, test_file_size - block_size)
                    f.seek(pos)
                    
                    # Random operation (70% read, 30% write)
                    if np.random.random() < 0.7:
                        # Read operation
                        data = f.read(block_size)
                        read_ops += 1
                    else:
                        # Write operation
                        f.write(b'B' * block_size)
                        write_ops += 1
                        
                    operations += 1
                    
        except Exception as e:
            return BenchmarkResult(
                name="Random I/O",
                test_type="disk_random_io",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="IOPS",
                metrics={},
                status="failed",
                error_message=str(e)
            )
        finally:
            # Cleanup
            try:
                self.test_file.unlink(missing_ok=True)
            except:
                pass
                
        actual_duration = time.time() - start_time
        iops = operations / actual_duration
        
        return BenchmarkResult(
            name="Random I/O",
            test_type="disk_random_io",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=iops,
            unit="IOPS",
            metrics={
                "total_operations": operations,
                "read_operations": read_ops,
                "write_operations": write_ops,
                "block_size_kb": block_size / 1024,
                "read_percentage": (read_ops / operations) * 100 if operations > 0 else 0
            },
            status="success"
        )

class NetworkBenchmark:
    """Network performance benchmarking"""
    
    def __init__(self):
        self.name = "Network Performance Test"
        
    def run_latency_test(self, target: str = "8.8.8.8", count: int = 100) -> BenchmarkResult:
        """Test network latency using ping"""
        logger.info(f"Running network latency test to {target}")
        
        start_time = time.time()
        latencies = []
        
        try:
            for _ in range(count):
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', '5000', target],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    # Parse ping output
                    for line in result.stdout.split('\n'):
                        if 'time=' in line:
                            try:
                                latency_str = line.split('time=')[1].split(' ')[0]
                                latency = float(latency_str)
                                latencies.append(latency)
                                break
                            except (IndexError, ValueError):
                                continue
                                
                time.sleep(0.1)  # Small delay between pings
                
        except subprocess.TimeoutExpired:
            logger.warning("Ping command timed out")
        except Exception as e:
            return BenchmarkResult(
                name="Network Latency",
                test_type="network_latency",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="ms",
                metrics={},
                status="failed",
                error_message=str(e)
            )
            
        if not latencies:
            return BenchmarkResult(
                name="Network Latency",
                test_type="network_latency",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="ms",
                metrics={},
                status="failed",
                error_message="No successful pings"
            )
            
        actual_duration = time.time() - start_time
        avg_latency = statistics.mean(latencies)
        
        return BenchmarkResult(
            name="Network Latency",
            test_type="network_latency",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=avg_latency,
            unit="ms",
            metrics={
                "average_latency": avg_latency,
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "median_latency": statistics.median(latencies),
                "std_deviation": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "packet_loss": ((count - len(latencies)) / count) * 100,
                "successful_pings": len(latencies),
                "total_pings": count
            },
            status="success",
            percentile_99=np.percentile(latencies, 99),
            percentile_95=np.percentile(latencies, 95),
            percentile_50=np.percentile(latencies, 50)
        )
        
    def run_dns_resolution_test(self, domains: List[str] = None, iterations: int = 50) -> BenchmarkResult:
        """Test DNS resolution performance"""
        if domains is None:
            domains = ['google.com', 'github.com', 'stackoverflow.com', 'aws.amazon.com']
            
        logger.info("Running DNS resolution benchmark")
        
        start_time = time.time()
        resolution_times = []
        
        try:
            for domain in domains:
                for _ in range(iterations // len(domains)):
                    dns_start = time.time()
                    
                    try:
                        result = subprocess.run(
                            ['nslookup', domain],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        if result.returncode == 0:
                            dns_time = (time.time() - dns_start) * 1000  # Convert to ms
                            resolution_times.append(dns_time)
                            
                    except subprocess.TimeoutExpired:
                        continue
                        
        except Exception as e:
            return BenchmarkResult(
                name="DNS Resolution",
                test_type="network_dns",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="ms",
                metrics={},
                status="failed",
                error_message=str(e)
            )
            
        if not resolution_times:
            return BenchmarkResult(
                name="DNS Resolution",
                test_type="network_dns",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="ms",
                metrics={},
                status="failed",
                error_message="No successful DNS resolutions"
            )
            
        actual_duration = time.time() - start_time
        avg_resolution_time = statistics.mean(resolution_times)
        
        return BenchmarkResult(
            name="DNS Resolution",
            test_type="network_dns",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=avg_resolution_time,
            unit="ms",
            metrics={
                "average_resolution_time": avg_resolution_time,
                "min_resolution_time": min(resolution_times),
                "max_resolution_time": max(resolution_times),
                "median_resolution_time": statistics.median(resolution_times),
                "successful_resolutions": len(resolution_times),
                "domains_tested": domains,
                "iterations_per_domain": iterations // len(domains)
            },
            status="success"
        )

class ApplicationBenchmark:
    """Application-specific benchmarking"""
    
    def __init__(self):
        self.name = "Application Performance Test"
        
    async def run_http_load_test(self, url: str, duration: int = 60, 
                                concurrent_requests: int = 10) -> BenchmarkResult:
        """Run HTTP load test against an endpoint"""
        logger.info(f"Running HTTP load test against {url}")
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        async def make_request(session):
            request_start = time.time()
            try:
                async with session.get(url, timeout=30) as response:
                    await response.text()
                    request_time = (time.time() - request_start) * 1000
                    return True, request_time, response.status
            except Exception as e:
                request_time = (time.time() - request_start) * 1000
                return False, request_time, str(e)
                
        async def worker(session):
            nonlocal successful_requests, failed_requests, response_times
            
            end_time = start_time + duration
            while time.time() < end_time:
                success, response_time, status = await make_request(session)
                
                if success:
                    successful_requests += 1
                    response_times.append(response_time)
                else:
                    failed_requests += 1
                    
                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.1)
                
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                tasks = [worker(session) for _ in range(concurrent_requests)]
                await asyncio.gather(*tasks)
                
        except Exception as e:
            return BenchmarkResult(
                name="HTTP Load Test",
                test_type="app_http_load",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="req/s",
                metrics={},
                status="failed",
                error_message=str(e)
            )
            
        actual_duration = time.time() - start_time
        total_requests = successful_requests + failed_requests
        requests_per_second = total_requests / actual_duration if actual_duration > 0 else 0
        
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        return BenchmarkResult(
            name="HTTP Load Test",
            test_type="app_http_load",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=requests_per_second,
            unit="req/s",
            metrics={
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": (successful_requests / total_requests) * 100 if total_requests > 0 else 0,
                "average_response_time": avg_response_time,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "concurrent_requests": concurrent_requests,
                "url": url
            },
            status="success",
            percentile_99=np.percentile(response_times, 99) if response_times else 0,
            percentile_95=np.percentile(response_times, 95) if response_times else 0,
            percentile_50=np.percentile(response_times, 50) if response_times else 0
        )
        
    def run_media_streaming_test(self, jellyfin_url: str, duration: int = 300) -> BenchmarkResult:
        """Test media streaming performance"""
        logger.info("Running media streaming benchmark")
        
        start_time = time.time()
        
        # Simulate streaming test by making requests to Jellyfin API
        try:
            # Test basic connectivity
            response = requests.get(f"{jellyfin_url}/System/Info", timeout=10)
            
            if response.status_code != 200:
                return BenchmarkResult(
                    name="Media Streaming",
                    test_type="app_media_streaming",
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                    score=0,
                    unit="streams",
                    metrics={},
                    status="failed",
                    error_message=f"Jellyfin not accessible: {response.status_code}"
                )
                
            # Simulate concurrent streaming sessions
            concurrent_streams = 3
            stream_duration = min(duration, 60)  # Limit to 1 minute for testing
            
            successful_streams = 0
            failed_streams = 0
            
            for stream_id in range(concurrent_streams):
                try:
                    # Simulate stream requests
                    for _ in range(stream_duration // 10):  # Request every 10 seconds
                        response = requests.get(f"{jellyfin_url}/System/Info", timeout=5)
                        if response.status_code == 200:
                            successful_streams += 1
                        else:
                            failed_streams += 1
                        time.sleep(1)
                        
                except requests.RequestException:
                    failed_streams += 1
                    
        except Exception as e:
            return BenchmarkResult(
                name="Media Streaming",
                test_type="app_media_streaming",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                score=0,
                unit="streams",
                metrics={},
                status="failed",
                error_message=str(e)
            )
            
        actual_duration = time.time() - start_time
        total_requests = successful_streams + failed_streams
        success_rate = (successful_streams / total_requests) * 100 if total_requests > 0 else 0
        
        return BenchmarkResult(
            name="Media Streaming",
            test_type="app_media_streaming",
            start_time=start_time,
            end_time=time.time(),
            duration=actual_duration,
            score=success_rate,
            unit="% success",
            metrics={
                "concurrent_streams": concurrent_streams,
                "successful_requests": successful_streams,
                "failed_requests": failed_streams,
                "total_requests": total_requests,
                "success_rate": success_rate,
                "jellyfin_url": jellyfin_url
            },
            status="success"
        )

class BenchmarkResultsAnalyzer:
    """Analyze and compare benchmark results"""
    
    def __init__(self, db_path: str = "/tmp/benchmark_results.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize results database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    duration REAL NOT NULL,
                    score REAL NOT NULL,
                    unit TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    baseline_comparison REAL,
                    percentile_99 REAL,
                    percentile_95 REAL,
                    percentile_50 REAL,
                    system_info TEXT
                )
            """)
            
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmark_results(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_type ON benchmark_results(test_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON benchmark_results(name)")
            
    def store_results(self, results: List[BenchmarkResult], system_info: Dict[str, Any]):
        """Store benchmark results in database"""
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                conn.execute("""
                    INSERT INTO benchmark_results 
                    (timestamp, name, test_type, duration, score, unit, metrics, status,
                     error_message, baseline_comparison, percentile_99, percentile_95, 
                     percentile_50, system_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.start_time,
                    result.name,
                    result.test_type,
                    result.duration,
                    result.score,
                    result.unit,
                    json.dumps(result.metrics),
                    result.status,
                    result.error_message,
                    result.baseline_comparison,
                    result.percentile_99,
                    result.percentile_95,
                    result.percentile_50,
                    json.dumps(system_info)
                ))
                
    def get_baseline_score(self, test_type: str, days_back: int = 30) -> Optional[float]:
        """Get baseline score for a test type"""
        cutoff_time = time.time() - (days_back * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT score FROM benchmark_results 
                WHERE test_type = ? AND timestamp > ? AND status = 'success'
                ORDER BY timestamp DESC
                LIMIT 10
            """, (test_type, cutoff_time))
            
            scores = [row[0] for row in cursor.fetchall()]
            
            if len(scores) >= 3:
                return statistics.median(scores)
                
        return None
        
    def calculate_baseline_comparison(self, result: BenchmarkResult) -> Optional[float]:
        """Calculate baseline comparison for a result"""
        baseline = self.get_baseline_score(result.test_type)
        
        if baseline and baseline > 0:
            return ((result.score - baseline) / baseline) * 100
            
        return None
        
    def generate_performance_report(self, suite: BenchmarkSuite) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "suite_name": suite.suite_name,
            "timestamp": suite.start_time,
            "duration": suite.total_duration,
            "system_info": suite.system_info,
            "summary": {
                "total_tests": len(suite.results),
                "successful_tests": len([r for r in suite.results if r.status == "success"]),
                "failed_tests": len([r for r in suite.results if r.status == "failed"]),
                "success_rate": (len([r for r in suite.results if r.status == "success"]) / len(suite.results)) * 100 if suite.results else 0
            },
            "categories": {},
            "recommendations": [],
            "trends": {}
        }
        
        # Group results by category
        categories = {}
        for result in suite.results:
            category = result.test_type.split('_')[0]  # cpu, memory, disk, network, app
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
            
        # Analyze each category
        for category, results in categories.items():
            successful_results = [r for r in results if r.status == "success"]
            
            if successful_results:
                scores = [r.score for r in successful_results]
                report["categories"][category] = {
                    "test_count": len(results),
                    "success_count": len(successful_results),
                    "average_score": statistics.mean(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "tests": [asdict(r) for r in results]
                }
                
                # Generate recommendations based on results
                if category == "cpu" and statistics.mean(scores) < 100000:  # ops/sec
                    report["recommendations"].append({
                        "category": "CPU",
                        "severity": "medium",
                        "message": "CPU performance is below expected baseline. Consider upgrading CPU or optimizing workloads."
                    })
                    
                elif category == "memory" and any(r.test_type == "memory_bandwidth" and r.score < 1.0 for r in successful_results):
                    report["recommendations"].append({
                        "category": "Memory",
                        "severity": "medium", 
                        "message": "Memory bandwidth is low. Consider upgrading RAM or optimizing memory access patterns."
                    })
                    
                elif category == "disk" and any(r.test_type == "disk_seq_write" and r.score < 50 for r in successful_results):
                    report["recommendations"].append({
                        "category": "Disk",
                        "severity": "high",
                        "message": "Disk write performance is critically low. Consider upgrading to SSD or optimizing I/O patterns."
                    })
                    
                elif category == "network" and any(r.test_type == "network_latency" and r.score > 100 for r in successful_results):
                    report["recommendations"].append({
                        "category": "Network",
                        "severity": "medium",
                        "message": "Network latency is high. Check network configuration and internet connection."
                    })
                    
        return report
        
    def generate_trend_analysis(self, test_type: str, days_back: int = 30) -> Dict[str, Any]:
        """Generate trend analysis for a specific test type"""
        cutoff_time = time.time() - (days_back * 24 * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, score, status FROM benchmark_results 
                WHERE test_type = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (test_type, cutoff_time))
            
            data = cursor.fetchall()
            
        if not data:
            return {"error": "No data available for trend analysis"}
            
        timestamps = [row[0] for row in data if row[2] == "success"]
        scores = [row[1] for row in data if row[2] == "success"]
        
        if len(scores) < 2:
            return {"error": "Insufficient data for trend analysis"}
            
        # Calculate trend
        trend_slope = np.polyfit(range(len(scores)), scores, 1)[0]
        
        return {
            "test_type": test_type,
            "data_points": len(scores),
            "time_range_days": days_back,
            "current_score": scores[-1],
            "average_score": statistics.mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "trend_slope": trend_slope,
            "trend_direction": "improving" if trend_slope > 0 else "declining" if trend_slope < 0 else "stable",
            "volatility": statistics.stdev(scores) if len(scores) > 1 else 0,
            "timestamps": timestamps,
            "scores": scores
        }

class ComprehensiveBenchmarkSuite:
    """Run comprehensive benchmark suite"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.analyzer = BenchmarkResultsAnalyzer()
        
        # Initialize benchmarks
        self.cpu_benchmark = CPUBenchmark()
        self.memory_benchmark = MemoryBenchmark()
        self.disk_benchmark = DiskBenchmark()
        self.network_benchmark = NetworkBenchmark()
        self.app_benchmark = ApplicationBenchmark()
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "cpu": {
                "brand": "",  # Would need cpuinfo parsing
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {}
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3)
            },
            "disk": {
                "partitions": [
                    {
                        "device": p.device,
                        "mountpoint": p.mountpoint,
                        "fstype": p.fstype,
                        "total_gb": psutil.disk_usage(p.mountpoint).total / (1024**3) if p.mountpoint else 0
                    }
                    for p in psutil.disk_partitions()
                ]
            },
            "platform": {
                "system": psutil.uname().system,
                "machine": psutil.uname().machine,
                "processor": psutil.uname().processor
            },
            "timestamp": time.time()
        }
        
    async def run_full_suite(self, include_app_tests: bool = False) -> BenchmarkSuite:
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive benchmark suite")
        
        suite_start = time.time()
        results = []
        system_info = self.get_system_info()
        
        try:
            # CPU benchmarks
            logger.info("Running CPU benchmarks...")
            results.append(self.cpu_benchmark.run_single_core_test(30))
            results.append(self.cpu_benchmark.run_multi_core_test(30))
            results.append(self.cpu_benchmark.run_prime_calculation_test(30))
            
            # Memory benchmarks
            logger.info("Running memory benchmarks...")
            results.append(self.memory_benchmark.run_allocation_test(30))
            results.append(self.memory_benchmark.run_bandwidth_test(30))
            
            # Disk benchmarks
            logger.info("Running disk benchmarks...")
            results.append(self.disk_benchmark.run_sequential_write_test(60))
            results.append(self.disk_benchmark.run_sequential_read_test(60))
            results.append(self.disk_benchmark.run_random_io_test(60))
            
            # Network benchmarks
            logger.info("Running network benchmarks...")
            results.append(self.network_benchmark.run_latency_test())
            results.append(self.network_benchmark.run_dns_resolution_test())
            
            # Application benchmarks (if enabled)
            if include_app_tests:
                logger.info("Running application benchmarks...")
                
                # Test Jellyfin if configured
                jellyfin_url = self.config.get('jellyfin_url', 'http://localhost:8096')
                results.append(self.app_benchmark.run_media_streaming_test(jellyfin_url))
                
                # HTTP load test
                test_url = self.config.get('test_url', 'http://localhost:8096/System/Info')
                http_result = await self.app_benchmark.run_http_load_test(test_url, 60, 10)
                results.append(http_result)
                
        except Exception as e:
            logger.error(f"Error during benchmark suite: {e}")
            
        # Calculate baseline comparisons
        for result in results:
            if result.status == "success":
                baseline_comparison = self.analyzer.calculate_baseline_comparison(result)
                result.baseline_comparison = baseline_comparison
                
        suite_end = time.time()
        
        suite = BenchmarkSuite(
            suite_name="Comprehensive Performance Benchmark",
            start_time=suite_start,
            end_time=suite_end,
            total_duration=suite_end - suite_start,
            results=results,
            summary=self.analyzer.generate_performance_report(BenchmarkSuite(
                "temp", suite_start, suite_end, suite_end - suite_start, results, {}, system_info
            )),
            system_info=system_info
        )
        
        # Store results
        self.analyzer.store_results(results, system_info)
        
        logger.info(f"Benchmark suite completed in {suite.total_duration:.2f} seconds")
        return suite
        
    def run_quick_benchmark(self) -> BenchmarkSuite:
        """Run quick benchmark suite (shorter duration tests)"""
        logger.info("Starting quick benchmark suite")
        
        suite_start = time.time()
        results = []
        system_info = self.get_system_info()
        
        try:
            # Quick tests with reduced duration
            results.append(self.cpu_benchmark.run_single_core_test(10))
            results.append(self.memory_benchmark.run_allocation_test(10))
            results.append(self.disk_benchmark.run_sequential_write_test(20))
            results.append(self.network_benchmark.run_latency_test(count=20))
            
        except Exception as e:
            logger.error(f"Error during quick benchmark: {e}")
            
        suite_end = time.time()
        
        suite = BenchmarkSuite(
            suite_name="Quick Performance Benchmark",
            start_time=suite_start,
            end_time=suite_end,
            total_duration=suite_end - suite_start,
            results=results,
            summary={},
            system_info=system_info
        )
        
        # Store results
        self.analyzer.store_results(results, system_info)
        
        logger.info(f"Quick benchmark completed in {suite.total_duration:.2f} seconds")
        return suite

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark suite")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--apps", action="store_true", help="Include application tests")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    suite_runner = ComprehensiveBenchmarkSuite(config)
    
    async def main():
        if args.quick:
            results = suite_runner.run_quick_benchmark()
        else:
            results = await suite_runner.run_full_suite(include_app_tests=args.apps)
            
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        for result in results.results:
            status_icon = "✓" if result.status == "success" else "✗"
            print(f"{status_icon} {result.name}: {result.score:.2f} {result.unit}")
            
            if result.baseline_comparison:
                change = "↑" if result.baseline_comparison > 0 else "↓"
                print(f"   Baseline comparison: {change} {result.baseline_comparison:.1f}%")
                
        print(f"\nTotal duration: {results.total_duration:.2f} seconds")
        print(f"Success rate: {(len([r for r in results.results if r.status == 'success']) / len(results.results)) * 100:.1f}%")
        
    if args.full or args.quick:
        asyncio.run(main())
    else:
        print("Please specify --quick or --full to run benchmarks")
        parser.print_help()