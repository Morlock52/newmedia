#!/usr/bin/env python3
"""
Docker Container Performance Benchmarking Suite
Comprehensive testing for container optimization strategies
"""

import os
import sys
import time
import json
import docker
import psutil
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Any
import subprocess
import threading
import queue
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    test_name: str
    duration: float
    cpu_usage: float
    memory_usage: float
    network_in: float
    network_out: float
    disk_read: float
    disk_write: float
    startup_time: float
    image_size: int
    layer_count: int
    build_time: float
    cache_efficiency: float
    
class DockerBenchmark:
    """Main benchmarking class for Docker containers"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.results = []
        self.test_configurations = []
        
    def benchmark_build_strategies(self) -> List[BenchmarkResult]:
        """Benchmark different build strategies"""
        strategies = [
            {
                "name": "standard_build",
                "dockerfile": "Dockerfile.standard",
                "buildargs": {}
            },
            {
                "name": "multistage_build",
                "dockerfile": "Dockerfile.optimized-nodejs",
                "buildargs": {}
            },
            {
                "name": "buildkit_cache",
                "dockerfile": "Dockerfile.optimized-python",
                "buildargs": {},
                "buildkit": True
            },
            {
                "name": "scratch_build",
                "dockerfile": "Dockerfile.optimized-go",
                "buildargs": {}
            }
        ]
        
        results = []
        for strategy in strategies:
            print(f"Testing build strategy: {strategy['name']}")
            result = self._benchmark_build(strategy)
            results.append(result)
            
        return results
    
    def _benchmark_build(self, strategy: Dict) -> BenchmarkResult:
        """Benchmark a single build strategy"""
        start_time = time.time()
        
        # Enable BuildKit if specified
        env = os.environ.copy()
        if strategy.get('buildkit'):
            env['DOCKER_BUILDKIT'] = '1'
        
        # Build the image
        try:
            image, build_logs = self.client.images.build(
                path=".",
                dockerfile=strategy['dockerfile'],
                buildargs=strategy['buildargs'],
                rm=True,
                forcerm=True
            )
            build_time = time.time() - start_time
            
            # Get image info
            image_info = self.client.api.inspect_image(image.id)
            image_size = image_info['Size']
            layer_count = len(image_info['RootFS']['Layers'])
            
            # Test container startup
            startup_time = self._measure_startup_time(image)
            
            # Run performance tests
            perf_metrics = self._run_performance_tests(image)
            
            # Calculate cache efficiency
            cache_efficiency = self._calculate_cache_efficiency(build_logs)
            
            return BenchmarkResult(
                test_name=strategy['name'],
                duration=perf_metrics['duration'],
                cpu_usage=perf_metrics['cpu_usage'],
                memory_usage=perf_metrics['memory_usage'],
                network_in=perf_metrics['network_in'],
                network_out=perf_metrics['network_out'],
                disk_read=perf_metrics['disk_read'],
                disk_write=perf_metrics['disk_write'],
                startup_time=startup_time,
                image_size=image_size,
                layer_count=layer_count,
                build_time=build_time,
                cache_efficiency=cache_efficiency
            )
            
        except Exception as e:
            print(f"Build failed for {strategy['name']}: {e}")
            return None
    
    def _measure_startup_time(self, image) -> float:
        """Measure container startup time"""
        start_time = time.time()
        
        container = self.client.containers.run(
            image,
            detach=True,
            remove=True,
            command="echo 'Started'"
        )
        
        # Wait for container to be running
        while container.status != 'running':
            container.reload()
            if container.status == 'exited':
                break
            time.sleep(0.01)
        
        startup_time = time.time() - start_time
        
        try:
            container.stop()
            container.remove()
        except:
            pass
            
        return startup_time
    
    def _run_performance_tests(self, image) -> Dict[str, float]:
        """Run performance tests on container"""
        metrics = {
            'duration': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'network_in': 0,
            'network_out': 0,
            'disk_read': 0,
            'disk_write': 0
        }
        
        # Start container with resource monitoring
        container = self.client.containers.run(
            image,
            detach=True,
            remove=True,
            command="sh -c 'while true; do echo test > /tmp/test.txt; sleep 0.1; done'",
            mem_limit='512m',
            cpus=1.0
        )
        
        # Collect metrics for 30 seconds
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < 30:
            try:
                stats = container.stats(stream=False)
                samples.append(self._parse_container_stats(stats))
                time.sleep(1)
            except Exception as e:
                print(f"Error collecting stats: {e}")
                break
        
        # Stop container
        try:
            container.stop()
            container.remove()
        except:
            pass
        
        # Calculate averages
        if samples:
            metrics['duration'] = time.time() - start_time
            metrics['cpu_usage'] = statistics.mean([s['cpu_percent'] for s in samples])
            metrics['memory_usage'] = statistics.mean([s['memory_mb'] for s in samples])
            metrics['network_in'] = sum([s['network_rx'] for s in samples])
            metrics['network_out'] = sum([s['network_tx'] for s in samples])
            metrics['disk_read'] = sum([s['disk_read'] for s in samples])
            metrics['disk_write'] = sum([s['disk_write'] for s in samples])
        
        return metrics
    
    def _parse_container_stats(self, stats: Dict) -> Dict[str, float]:
        """Parse container stats into metrics"""
        parsed = {
            'cpu_percent': 0,
            'memory_mb': 0,
            'network_rx': 0,
            'network_tx': 0,
            'disk_read': 0,
            'disk_write': 0
        }
        
        # CPU usage calculation
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                   stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                      stats['precpu_stats']['system_cpu_usage']
        
        if system_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * 100.0
            parsed['cpu_percent'] = cpu_percent
        
        # Memory usage
        if 'memory_stats' in stats:
            memory_usage = stats['memory_stats'].get('usage', 0)
            parsed['memory_mb'] = memory_usage / 1024 / 1024
        
        # Network I/O
        if 'networks' in stats:
            for interface, network_stats in stats['networks'].items():
                parsed['network_rx'] += network_stats.get('rx_bytes', 0)
                parsed['network_tx'] += network_stats.get('tx_bytes', 0)
        
        # Disk I/O
        if 'blkio_stats' in stats:
            io_stats = stats['blkio_stats'].get('io_service_bytes_recursive', [])
            for io_stat in io_stats:
                if io_stat['op'] == 'Read':
                    parsed['disk_read'] += io_stat['value']
                elif io_stat['op'] == 'Write':
                    parsed['disk_write'] += io_stat['value']
        
        return parsed
    
    def _calculate_cache_efficiency(self, build_logs) -> float:
        """Calculate build cache efficiency"""
        cached_steps = 0
        total_steps = 0
        
        for log in build_logs:
            if 'stream' in log:
                line = log['stream']
                if 'Step' in line:
                    total_steps += 1
                    if 'Using cache' in line:
                        cached_steps += 1
        
        if total_steps > 0:
            return (cached_steps / total_steps) * 100
        return 0
    
    def benchmark_volume_performance(self) -> Dict[str, Any]:
        """Benchmark different volume configurations"""
        volume_configs = [
            {
                "name": "bind_mount",
                "type": "bind",
                "source": "/tmp/bench_bind",
                "target": "/data"
            },
            {
                "name": "named_volume",
                "type": "volume",
                "source": "bench_volume",
                "target": "/data"
            },
            {
                "name": "tmpfs_mount",
                "type": "tmpfs",
                "target": "/data",
                "tmpfs_size": "100m"
            }
        ]
        
        results = {}
        
        for config in volume_configs:
            print(f"Testing volume config: {config['name']}")
            perf = self._benchmark_volume_io(config)
            results[config['name']] = perf
        
        return results
    
    def _benchmark_volume_io(self, volume_config: Dict) -> Dict[str, float]:
        """Benchmark I/O performance for a volume configuration"""
        # Prepare volume mount
        mounts = []
        
        if volume_config['type'] == 'bind':
            os.makedirs(volume_config['source'], exist_ok=True)
            mounts.append(docker.types.Mount(
                target=volume_config['target'],
                source=volume_config['source'],
                type='bind'
            ))
        elif volume_config['type'] == 'volume':
            mounts.append(docker.types.Mount(
                target=volume_config['target'],
                source=volume_config['source'],
                type='volume'
            ))
        elif volume_config['type'] == 'tmpfs':
            mounts.append(docker.types.Mount(
                target=volume_config['target'],
                type='tmpfs',
                tmpfs_size=volume_config.get('tmpfs_size', '100m')
            ))
        
        # Run I/O benchmark
        container = self.client.containers.run(
            'alpine',
            detach=True,
            remove=True,
            mounts=mounts,
            command="sh -c 'apk add --no-cache fio && fio --name=test --filename=/data/test --size=100M --rw=randrw --bs=4k --direct=1 --numjobs=4 --time_based --runtime=30 --group_reporting --output-format=json'"
        )
        
        # Wait for completion
        result = container.wait()
        logs = container.logs().decode('utf-8')
        
        # Parse FIO results
        try:
            fio_json = json.loads(logs.split('\n')[-1])
            read_iops = fio_json['jobs'][0]['read']['iops']
            write_iops = fio_json['jobs'][0]['write']['iops']
            read_bw = fio_json['jobs'][0]['read']['bw']
            write_bw = fio_json['jobs'][0]['write']['bw']
            
            return {
                'read_iops': read_iops,
                'write_iops': write_iops,
                'read_bandwidth_kb': read_bw,
                'write_bandwidth_kb': write_bw
            }
        except Exception as e:
            print(f"Error parsing FIO results: {e}")
            return {
                'read_iops': 0,
                'write_iops': 0,
                'read_bandwidth_kb': 0,
                'write_bandwidth_kb': 0
            }
    
    def benchmark_network_configurations(self) -> Dict[str, Any]:
        """Benchmark different network configurations"""
        network_configs = [
            {
                "name": "bridge_default",
                "driver": "bridge",
                "options": {}
            },
            {
                "name": "bridge_mtu9000",
                "driver": "bridge",
                "options": {
                    "com.docker.network.driver.mtu": "9000"
                }
            },
            {
                "name": "host_network",
                "driver": "host",
                "options": {}
            }
        ]
        
        results = {}
        
        for config in network_configs:
            print(f"Testing network config: {config['name']}")
            if config['driver'] == 'host' and sys.platform == 'darwin':
                print("Skipping host network on macOS")
                continue
                
            perf = self._benchmark_network_performance(config)
            results[config['name']] = perf
        
        return results
    
    def _benchmark_network_performance(self, network_config: Dict) -> Dict[str, float]:
        """Benchmark network performance"""
        # Create network if needed
        network = None
        if network_config['driver'] != 'host':
            network = self.client.networks.create(
                f"bench_{network_config['name']}",
                driver=network_config['driver'],
                options=network_config['options']
            )
        
        try:
            # Start iperf3 server
            server = self.client.containers.run(
                'networkstatic/iperf3',
                detach=True,
                remove=True,
                network=network.name if network else 'host',
                command='-s'
            )
            
            time.sleep(2)  # Wait for server to start
            
            # Get server IP
            server_ip = 'localhost'
            if network:
                server.reload()
                server_ip = server.attrs['NetworkSettings']['Networks'][network.name]['IPAddress']
            
            # Run iperf3 client
            client = self.client.containers.run(
                'networkstatic/iperf3',
                remove=True,
                network=network.name if network else 'host',
                command=f'-c {server_ip} -t 10 -J'
            )
            
            # Parse results
            iperf_results = json.loads(client)
            
            bandwidth_mbps = iperf_results['end']['sum_sent']['bits_per_second'] / 1000000
            retransmits = iperf_results['end']['sum_sent'].get('retransmits', 0)
            
            return {
                'bandwidth_mbps': bandwidth_mbps,
                'retransmits': retransmits
            }
            
        except Exception as e:
            print(f"Network benchmark error: {e}")
            return {
                'bandwidth_mbps': 0,
                'retransmits': 0
            }
            
        finally:
            # Cleanup
            try:
                server.stop()
                server.remove()
            except:
                pass
            
            if network:
                network.remove()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive benchmark report"""
        report = []
        report.append("# Docker Container Performance Benchmark Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Build strategy results
        if 'build_strategies' in results:
            report.append("## Build Strategy Performance")
            report.append("")
            
            df = pd.DataFrame([asdict(r) for r in results['build_strategies'] if r])
            report.append(df.to_string())
            report.append("")
            
            # Best practices based on results
            report.append("### Recommendations:")
            
            smallest = df.nsmallest(1, 'image_size').iloc[0]
            report.append(f"- Smallest image: {smallest['test_name']} ({smallest['image_size'] / 1024 / 1024:.1f} MB)")
            
            fastest_build = df.nsmallest(1, 'build_time').iloc[0]
            report.append(f"- Fastest build: {fastest_build['test_name']} ({fastest_build['build_time']:.1f}s)")
            
            fastest_startup = df.nsmallest(1, 'startup_time').iloc[0]
            report.append(f"- Fastest startup: {fastest_startup['test_name']} ({fastest_startup['startup_time']:.2f}s)")
            
            report.append("")
        
        # Volume performance results
        if 'volume_performance' in results:
            report.append("## Volume Performance")
            report.append("")
            
            for vol_type, metrics in results['volume_performance'].items():
                report.append(f"### {vol_type}")
                report.append(f"- Read IOPS: {metrics['read_iops']:.0f}")
                report.append(f"- Write IOPS: {metrics['write_iops']:.0f}")
                report.append(f"- Read Bandwidth: {metrics['read_bandwidth_kb'] / 1024:.1f} MB/s")
                report.append(f"- Write Bandwidth: {metrics['write_bandwidth_kb'] / 1024:.1f} MB/s")
                report.append("")
        
        # Network performance results
        if 'network_performance' in results:
            report.append("## Network Performance")
            report.append("")
            
            for net_type, metrics in results['network_performance'].items():
                report.append(f"### {net_type}")
                report.append(f"- Bandwidth: {metrics['bandwidth_mbps']:.1f} Mbps")
                report.append(f"- Retransmits: {metrics['retransmits']}")
                report.append("")
        
        return "\n".join(report)
    
    def visualize_results(self, results: Dict[str, Any]):
        """Create visualization of benchmark results"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Docker Container Performance Benchmarks', fontsize=16)
        
        # Build strategies comparison
        if 'build_strategies' in results:
            df = pd.DataFrame([asdict(r) for r in results['build_strategies'] if r])
            
            # Image sizes
            ax = axes[0, 0]
            df.plot(x='test_name', y='image_size', kind='bar', ax=ax, legend=False)
            ax.set_title('Image Sizes')
            ax.set_ylabel('Size (bytes)')
            ax.set_xlabel('Build Strategy')
            
            # Startup times
            ax = axes[0, 1]
            df.plot(x='test_name', y='startup_time', kind='bar', ax=ax, legend=False, color='orange')
            ax.set_title('Container Startup Times')
            ax.set_ylabel('Time (seconds)')
            ax.set_xlabel('Build Strategy')
            
            # Resource usage
            ax = axes[1, 0]
            df[['test_name', 'cpu_usage', 'memory_usage']].set_index('test_name').plot(kind='bar', ax=ax)
            ax.set_title('Resource Usage')
            ax.set_ylabel('Usage')
            ax.set_xlabel('Build Strategy')
            ax.legend(['CPU %', 'Memory MB'])
            
        # Volume performance
        if 'volume_performance' in results:
            ax = axes[1, 1]
            vol_data = results['volume_performance']
            
            vol_names = list(vol_data.keys())
            read_iops = [vol_data[v]['read_iops'] for v in vol_names]
            write_iops = [vol_data[v]['write_iops'] for v in vol_names]
            
            x = range(len(vol_names))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], read_iops, width, label='Read IOPS')
            ax.bar([i + width/2 for i in x], write_iops, width, label='Write IOPS')
            
            ax.set_xlabel('Volume Type')
            ax.set_ylabel('IOPS')
            ax.set_title('Volume I/O Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(vol_names)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('docker_benchmark_results.png', dpi=300)
        plt.show()

def main():
    """Main benchmark execution"""
    print("Starting Docker Container Performance Benchmarks...")
    
    benchmark = DockerBenchmark()
    results = {}
    
    # Run build strategy benchmarks
    print("\n=== Build Strategy Benchmarks ===")
    build_results = benchmark.benchmark_build_strategies()
    results['build_strategies'] = build_results
    
    # Run volume performance benchmarks
    print("\n=== Volume Performance Benchmarks ===")
    volume_results = benchmark.benchmark_volume_performance()
    results['volume_performance'] = volume_results
    
    # Run network performance benchmarks
    print("\n=== Network Performance Benchmarks ===")
    network_results = benchmark.benchmark_network_configurations()
    results['network_performance'] = network_results
    
    # Generate report
    report = benchmark.generate_report(results)
    print("\n" + report)
    
    # Save report
    with open('docker_benchmark_report.md', 'w') as f:
        f.write(report)
    
    # Save raw results
    with open('docker_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualizations
    benchmark.visualize_results(results)
    
    print("\nBenchmark complete! Results saved to:")
    print("- docker_benchmark_report.md")
    print("- docker_benchmark_results.json")
    print("- docker_benchmark_results.png")

if __name__ == "__main__":
    main()