#!/bin/bash

# Media Server Performance Optimization Script 2025
# Tunes system settings for optimal media streaming performance

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi
}

# Optimize system kernel parameters
optimize_kernel() {
    log "Optimizing kernel parameters..."
    
    # Network optimizations
    cat > /etc/sysctl.d/99-media-server-performance.conf << EOF
# Media Server Performance Optimizations 2025

# Network Performance
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_intvl = 30
net.ipv4.tcp_keepalive_probes = 3
net.ipv4.tcp_tw_reuse = 1
net.core.rmem_default = 31457280
net.core.rmem_max = 134217728
net.core.wmem_default = 31457280
net.core.wmem_max = 134217728
net.core.optmem_max = 25165824
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_mtu_probing = 1

# Enable jumbo frames support
net.ipv4.tcp_mtu_probing = 1

# File System
fs.file-max = 2097152
fs.nr_open = 2097152

# Virtual Memory
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.vfs_cache_pressure = 50

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 0
net.ipv6.conf.default.disable_ipv6 = 0

# Security hardening
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_rfc1337 = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
EOF

    # Apply settings
    sysctl -p /etc/sysctl.d/99-media-server-performance.conf
    
    log "Kernel parameters optimized"
}

# Optimize Docker daemon
optimize_docker() {
    log "Optimizing Docker daemon configuration..."
    
    # Create optimized Docker daemon config
    cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "experimental": true,
  "metrics-addr": "0.0.0.0:9323",
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 10,
  "default-runtime": "runc",
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    },
    "nproc": {
      "Name": "nproc",
      "Hard": 65536,
      "Soft": 65536
    }
  },
  "debug": false,
  "live-restore": true,
  "userland-proxy": false,
  "ip-forward": true,
  "iptables": true,
  "ip-masq": true,
  "bridge": "docker0",
  "bip": "172.17.0.1/16",
  "fixed-cidr": "172.17.0.0/16",
  "default-address-pools": [
    {
      "base": "172.20.0.0/14",
      "size": 24
    }
  ]
}
EOF

    # Restart Docker daemon
    systemctl daemon-reload
    systemctl restart docker
    
    log "Docker daemon optimized"
}

# Set up performance monitoring
setup_monitoring() {
    log "Setting up performance monitoring..."
    
    # Create monitoring directories
    mkdir -p /var/lib/prometheus
    mkdir -p /var/lib/grafana
    mkdir -p /etc/prometheus/rules
    
    # Set permissions
    chown -R 65534:65534 /var/lib/prometheus
    chown -R 472:472 /var/lib/grafana
    
    log "Monitoring directories created"
}

# Optimize file system
optimize_filesystem() {
    log "Optimizing file system settings..."
    
    # Increase inotify watchers for media monitoring
    echo "fs.inotify.max_user_watches=524288" >> /etc/sysctl.d/99-media-server-performance.conf
    echo "fs.inotify.max_user_instances=512" >> /etc/sysctl.d/99-media-server-performance.conf
    
    # Apply settings
    sysctl -p /etc/sysctl.d/99-media-server-performance.conf
    
    # Set up hugepages for better memory performance
    echo "vm.nr_hugepages=1024" >> /etc/sysctl.d/99-media-server-performance.conf
    
    log "File system optimized"
}

# Configure system limits
configure_limits() {
    log "Configuring system limits..."
    
    # Update limits.conf
    cat >> /etc/security/limits.conf << EOF

# Media Server Performance Limits
* soft nofile 65536
* hard nofile 65536
* soft nproc 65536
* hard nproc 65536
* soft memlock unlimited
* hard memlock unlimited
EOF

    # Update systemd limits
    mkdir -p /etc/systemd/system.conf.d
    cat > /etc/systemd/system.conf.d/limits.conf << EOF
[Manager]
DefaultLimitNOFILE=65536
DefaultLimitNPROC=65536
DefaultTasksMax=infinity
DefaultLimitMEMLOCK=infinity
EOF

    systemctl daemon-reload
    
    log "System limits configured"
}

# Enable BBR congestion control
enable_bbr() {
    log "Enabling BBR congestion control..."
    
    # Check if BBR is available
    if modprobe -n tcp_bbr &>/dev/null; then
        modprobe tcp_bbr
        echo "tcp_bbr" >> /etc/modules-load.d/modules.conf
        echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.d/99-media-server-performance.conf
        sysctl -w net.ipv4.tcp_congestion_control=bbr
        log "BBR enabled"
    else
        warning "BBR module not available"
    fi
}

# Optimize CPU governor
optimize_cpu() {
    log "Optimizing CPU performance..."
    
    # Set CPU governor to performance
    if command -v cpupower &> /dev/null; then
        cpupower frequency-set -g performance
        log "CPU governor set to performance"
    else
        warning "cpupower not installed, skipping CPU optimization"
    fi
    
    # Disable CPU frequency scaling
    echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
}

# Setup swap for emergency situations
setup_swap() {
    log "Configuring swap settings..."
    
    # Create swap file if it doesn't exist
    if [[ ! -f /swapfile ]]; then
        fallocate -l 4G /swapfile
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
        echo "/swapfile none swap sw 0 0" >> /etc/fstab
        log "4GB swap file created"
    fi
    
    # Configure swappiness
    echo "vm.swappiness=10" >> /etc/sysctl.d/99-media-server-performance.conf
    sysctl -w vm.swappiness=10
}

# Create performance report
generate_report() {
    log "Generating performance optimization report..."
    
    cat > /tmp/performance-optimization-report.txt << EOF
Media Server Performance Optimization Report
Generated: $(date)
============================================

System Information:
- Kernel: $(uname -r)
- CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
- Memory: $(free -h | awk 'NR==2{print $2}')
- Docker: $(docker --version)

Applied Optimizations:
1. Kernel Parameters:
   - Network stack optimized for high throughput
   - BBR congestion control enabled
   - File system limits increased

2. Docker Configuration:
   - Logging optimized
   - Resource limits increased
   - GPU support enabled (if available)

3. System Limits:
   - File descriptors: 65536
   - Process limits: 65536
   - Memory locking: unlimited

4. CPU Performance:
   - Governor: performance
   - Frequency scaling: disabled

5. Memory Management:
   - Swappiness: 10
   - Dirty ratio: 15
   - Cache pressure: 50

Current Performance Metrics:
- Load Average: $(uptime | awk -F'load average:' '{print $2}')
- Memory Usage: $(free -h | awk 'NR==2{printf "%.1f%%", $3/$2*100}')
- Disk I/O: $(iostat -x 1 2 | tail -n +7 | awk '{sum+=$14} END {printf "%.1f%%", sum/NR}')

Recommendations:
1. Monitor performance metrics regularly using Grafana dashboards
2. Adjust cache sizes based on available memory
3. Enable CDN for static content delivery
4. Use SSD storage for media metadata and databases
5. Implement regular cache warming for popular content

EOF

    log "Report generated at /tmp/performance-optimization-report.txt"
}

# Main execution
main() {
    log "Starting Media Server Performance Optimization..."
    
    check_root
    
    # Backup current configurations
    log "Creating configuration backups..."
    mkdir -p /etc/media-server-backups
    cp -r /etc/sysctl.d /etc/media-server-backups/ 2>/dev/null || true
    cp /etc/docker/daemon.json /etc/media-server-backups/ 2>/dev/null || true
    
    # Run optimizations
    optimize_kernel
    optimize_docker
    setup_monitoring
    optimize_filesystem
    configure_limits
    enable_bbr
    optimize_cpu
    setup_swap
    
    # Generate report
    generate_report
    
    log "Performance optimization completed successfully!"
    info "Please reboot the system for all changes to take effect"
    info "View the optimization report: cat /tmp/performance-optimization-report.txt"
}

# Run main function
main "$@"