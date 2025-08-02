#!/usr/bin/env python3
"""
Performance Monitor for Media Server
Advanced system performance and container monitoring
"""

import os
import time
import psutil
import docker
import json
from pathlib import Path
from prometheus_client import start_http_server, Gauge, Counter, Histogram, CollectorRegistry

# Metrics Registry
REGISTRY = CollectorRegistry(auto_describe=False)

# Performance Metrics
CPU_USAGE_DETAILED = Gauge('cpu_usage_detailed', 'Detailed CPU usage', ['core', 'type'], registry=REGISTRY)
MEMORY_DETAILED = Gauge('memory_detailed', 'Detailed memory metrics', ['type'], registry=REGISTRY)
DISK_IO_DETAILED = Gauge('disk_io_detailed', 'Detailed disk I/O metrics', ['device', 'type'], registry=REGISTRY)
NETWORK_IO_DETAILED = Gauge('network_io_detailed', 'Detailed network I/O metrics', ['interface', 'type'], registry=REGISTRY)

# Container Performance
CONTAINER_CPU_USAGE = Gauge('container_cpu_usage_percent', 'Container CPU usage', ['container', 'service'], registry=REGISTRY)
CONTAINER_MEMORY_USAGE = Gauge('container_memory_usage_bytes', 'Container memory usage', ['container', 'service'], registry=REGISTRY)
CONTAINER_MEMORY_LIMIT = Gauge('container_memory_limit_bytes', 'Container memory limit', ['container', 'service'], registry=REGISTRY)
CONTAINER_NETWORK_IO = Gauge('container_network_io_bytes', 'Container network I/O', ['container', 'service', 'direction'], registry=REGISTRY)
CONTAINER_DISK_IO = Gauge('container_disk_io_bytes', 'Container disk I/O', ['container', 'service', 'direction'], registry=REGISTRY)

# Application Performance
TRANSCODE_PERFORMANCE = Gauge('transcode_performance', 'Transcoding performance metrics', ['type'], registry=REGISTRY)
DOWNLOAD_PERFORMANCE = Gauge('download_performance', 'Download performance metrics', ['type'], registry=REGISTRY)
STREAMING_PERFORMANCE = Gauge('streaming_performance', 'Streaming performance metrics', ['type'], registry=REGISTRY)

# Resource Utilization
GPU_USAGE = Gauge('gpu_usage_percent', 'GPU usage percentage', ['gpu_id'], registry=REGISTRY)
GPU_MEMORY = Gauge('gpu_memory_usage_bytes', 'GPU memory usage', ['gpu_id'], registry=REGISTRY)
THERMAL_STATUS = Gauge('thermal_status', 'System thermal status', ['sensor'], registry=REGISTRY)

# Performance Histograms
RESPONSE_TIME_HISTOGRAM = Histogram('service_response_time_histogram', 'Service response time distribution', ['service'], registry=REGISTRY)
THROUGHPUT_HISTOGRAM = Histogram('throughput_histogram', 'Throughput distribution', ['type'], registry=REGISTRY)

class PerformanceMonitor:
    def __init__(self):
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"Docker client initialization failed: {e}")
        
        self.host_proc = os.getenv('HOST_PROC', '/host/proc')
        self.host_sys = os.getenv('HOST_SYS', '/host/sys')
        
        # Media server containers to monitor
        self.media_containers = [
            'jellyfin', 'plex', 'emby', 'sonarr', 'radarr', 'lidarr', 
            'bazarr', 'prowlarr', 'qbittorrent', 'sabnzbd', 'overseerr',
            'tautulli', 'homepage'
        ]

    def collect_detailed_cpu_metrics(self):
        """Collect detailed CPU metrics"""
        try:
            # Per-core CPU usage
            cpu_percent_per_core = psutil.cpu_percent(percpu=True)
            for i, usage in enumerate(cpu_percent_per_core):
                CPU_USAGE_DETAILED.labels(core=f'cpu{i}', type='usage_percent').set(usage)
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                CPU_USAGE_DETAILED.labels(core='all', type='freq_current').set(cpu_freq.current)
                CPU_USAGE_DETAILED.labels(core='all', type='freq_min').set(cpu_freq.min)
                CPU_USAGE_DETAILED.labels(core='all', type='freq_max').set(cpu_freq.max)
            
            # CPU times
            cpu_times = psutil.cpu_times()
            CPU_USAGE_DETAILED.labels(core='all', type='user').set(cpu_times.user)
            CPU_USAGE_DETAILED.labels(core='all', type='system').set(cpu_times.system)
            CPU_USAGE_DETAILED.labels(core='all', type='idle').set(cpu_times.idle)
            CPU_USAGE_DETAILED.labels(core='all', type='iowait').set(getattr(cpu_times, 'iowait', 0))
            
        except Exception as e:
            print(f"Error collecting CPU metrics: {e}")

    def collect_detailed_memory_metrics(self):
        """Collect detailed memory metrics"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Virtual memory
            MEMORY_DETAILED.labels(type='total').set(memory.total)
            MEMORY_DETAILED.labels(type='available').set(memory.available)
            MEMORY_DETAILED.labels(type='percent').set(memory.percent)
            MEMORY_DETAILED.labels(type='used').set(memory.used)
            MEMORY_DETAILED.labels(type='free').set(memory.free)
            MEMORY_DETAILED.labels(type='active').set(getattr(memory, 'active', 0))
            MEMORY_DETAILED.labels(type='inactive').set(getattr(memory, 'inactive', 0))
            MEMORY_DETAILED.labels(type='buffers').set(getattr(memory, 'buffers', 0))
            MEMORY_DETAILED.labels(type='cached').set(getattr(memory, 'cached', 0))
            
            # Swap memory
            MEMORY_DETAILED.labels(type='swap_total').set(swap.total)
            MEMORY_DETAILED.labels(type='swap_used').set(swap.used)
            MEMORY_DETAILED.labels(type='swap_free').set(swap.free)
            MEMORY_DETAILED.labels(type='swap_percent').set(swap.percent)
            
        except Exception as e:
            print(f"Error collecting memory metrics: {e}")

    def collect_detailed_disk_metrics(self):
        """Collect detailed disk I/O metrics"""
        try:
            disk_io = psutil.disk_io_counters(perdisk=True)
            
            for device, stats in disk_io.items():
                DISK_IO_DETAILED.labels(device=device, type='read_count').set(stats.read_count)
                DISK_IO_DETAILED.labels(device=device, type='write_count').set(stats.write_count)
                DISK_IO_DETAILED.labels(device=device, type='read_bytes').set(stats.read_bytes)
                DISK_IO_DETAILED.labels(device=device, type='write_bytes').set(stats.write_bytes)
                DISK_IO_DETAILED.labels(device=device, type='read_time').set(stats.read_time)
                DISK_IO_DETAILED.labels(device=device, type='write_time').set(stats.write_time)
                
        except Exception as e:
            print(f"Error collecting disk metrics: {e}")

    def collect_detailed_network_metrics(self):
        """Collect detailed network metrics"""
        try:
            network_io = psutil.net_io_counters(pernic=True)
            
            for interface, stats in network_io.items():
                if interface.startswith('lo'):  # Skip loopback
                    continue
                    
                NETWORK_IO_DETAILED.labels(interface=interface, type='bytes_sent').set(stats.bytes_sent)
                NETWORK_IO_DETAILED.labels(interface=interface, type='bytes_recv').set(stats.bytes_recv)
                NETWORK_IO_DETAILED.labels(interface=interface, type='packets_sent').set(stats.packets_sent)
                NETWORK_IO_DETAILED.labels(interface=interface, type='packets_recv').set(stats.packets_recv)
                NETWORK_IO_DETAILED.labels(interface=interface, type='errin').set(stats.errin)
                NETWORK_IO_DETAILED.labels(interface=interface, type='errout').set(stats.errout)
                NETWORK_IO_DETAILED.labels(interface=interface, type='dropin').set(stats.dropin)
                NETWORK_IO_DETAILED.labels(interface=interface, type='dropout').set(stats.dropout)
                
        except Exception as e:
            print(f"Error collecting network metrics: {e}")

    def collect_container_metrics(self):
        """Collect container performance metrics"""
        if not self.docker_client:
            return
            
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                container_name = container.name
                service_name = container.labels.get('com.docker.compose.service', container_name)
                
                # Skip if not a media container
                if not any(media_service in container_name.lower() for media_service in self.media_containers):
                    continue
                
                try:
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    # CPU usage
                    cpu_usage = self.calculate_cpu_percent(stats)
                    CONTAINER_CPU_USAGE.labels(container=container_name, service=service_name).set(cpu_usage)
                    
                    # Memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    CONTAINER_MEMORY_USAGE.labels(container=container_name, service=service_name).set(memory_usage)
                    CONTAINER_MEMORY_LIMIT.labels(container=container_name, service=service_name).set(memory_limit)
                    
                    # Network I/O
                    networks = stats.get('networks', {})
                    for network_name, network_stats in networks.items():
                        CONTAINER_NETWORK_IO.labels(
                            container=container_name, 
                            service=service_name, 
                            direction='rx'
                        ).set(network_stats.get('rx_bytes', 0))
                        CONTAINER_NETWORK_IO.labels(
                            container=container_name, 
                            service=service_name, 
                            direction='tx'
                        ).set(network_stats.get('tx_bytes', 0))
                    
                    # Disk I/O
                    blkio_stats = stats.get('blkio_stats', {})
                    io_service_bytes_recursive = blkio_stats.get('io_service_bytes_recursive', [])
                    
                    read_bytes = sum(item['value'] for item in io_service_bytes_recursive if item['op'] == 'read')
                    write_bytes = sum(item['value'] for item in io_service_bytes_recursive if item['op'] == 'write')
                    
                    CONTAINER_DISK_IO.labels(
                        container=container_name, 
                        service=service_name, 
                        direction='read'
                    ).set(read_bytes)
                    CONTAINER_DISK_IO.labels(
                        container=container_name, 
                        service=service_name, 
                        direction='write'
                    ).set(write_bytes)
                    
                except Exception as e:
                    print(f"Error collecting stats for container {container_name}: {e}")
                    
        except Exception as e:
            print(f"Error collecting container metrics: {e}")

    def calculate_cpu_percent(self, stats):
        """Calculate CPU usage percentage from Docker stats"""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_usage = cpu_stats['cpu_usage']
            precpu_usage = precpu_stats['cpu_usage']
            
            cpu_delta = cpu_usage['total_usage'] - precpu_usage['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            
            if system_delta > 0.0:
                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100.0
                return cpu_percent
            return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0

    def collect_gpu_metrics(self):
        """Collect GPU metrics if available"""
        try:
            # Try to read nvidia-smi info
            import subprocess
            
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    values = line.split(', ')
                    if len(values) == 4:
                        gpu_util, mem_used, mem_total, temp = values
                        GPU_USAGE.labels(gpu_id=str(i)).set(float(gpu_util))
                        GPU_MEMORY.labels(gpu_id=str(i)).set(float(mem_used) * 1024 * 1024)  # Convert to bytes
                        THERMAL_STATUS.labels(sensor=f'gpu{i}').set(float(temp))
                        
        except Exception as e:
            # GPU monitoring is optional
            pass

    def collect_thermal_metrics(self):
        """Collect thermal sensor data"""
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                for sensor_name, sensors in temps.items():
                    for i, sensor in enumerate(sensors):
                        label = f"{sensor_name}_{sensor.label or i}"
                        THERMAL_STATUS.labels(sensor=label).set(sensor.current)
                        
        except Exception as e:
            print(f"Error collecting thermal metrics: {e}")

    def collect_performance_metrics(self):
        """Collect application-specific performance metrics"""
        try:
            # Transcoding performance (estimated based on CPU usage of media containers)
            if self.docker_client:
                jellyfin_container = None
                plex_container = None
                
                for container in self.docker_client.containers.list():
                    if 'jellyfin' in container.name:
                        jellyfin_container = container
                    elif 'plex' in container.name:
                        plex_container = container
                
                # Estimate transcoding load
                if jellyfin_container:
                    stats = jellyfin_container.stats(stream=False)
                    cpu_usage = self.calculate_cpu_percent(stats)
                    TRANSCODE_PERFORMANCE.labels(type='cpu_usage').set(cpu_usage)
                    
        except Exception as e:
            print(f"Error collecting performance metrics: {e}")

    def collect_metrics(self):
        """Collect all performance metrics"""
        print("Collecting performance metrics...")
        
        self.collect_detailed_cpu_metrics()
        self.collect_detailed_memory_metrics()
        self.collect_detailed_disk_metrics()
        self.collect_detailed_network_metrics()
        self.collect_container_metrics()
        self.collect_gpu_metrics()
        self.collect_thermal_metrics()
        self.collect_performance_metrics()
        
        print("Performance metrics collection completed")

def main():
    monitor = PerformanceMonitor()
    
    # Start metrics server
    start_http_server(9501, registry=REGISTRY)
    print("Performance Monitor started on port 9501")
    
    while True:
        try:
            monitor.collect_metrics()
            time.sleep(30)  # Collect metrics every 30 seconds
        except KeyboardInterrupt:
            print("Shutting down performance monitor...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(30)

if __name__ == '__main__':
    main()