#!/bin/bash

# Performance Tuning Script for Media Server
# Optimizes system settings for media streaming and processing

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Media Server Performance Tuning${NC}"
echo "=================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Function to backup current settings
backup_settings() {
    echo -e "${YELLOW}Backing up current settings...${NC}"
    
    mkdir -p /etc/sysctl.d/backup
    cp -p /etc/sysctl.conf /etc/sysctl.d/backup/sysctl.conf.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    
    echo -e "${GREEN}✅ Settings backed up${NC}"
}

# Function to apply network optimizations
optimize_network() {
    echo -e "${BLUE}Applying network optimizations...${NC}"
    
    cat > /etc/sysctl.d/99-media-server-network.conf << 'EOF'
# Network optimizations for media streaming

# Increase Linux autotuning TCP buffer limits
net.core.rmem_default = 134217728
net.core.rmem_max = 134217728
net.core.wmem_default = 134217728
net.core.wmem_max = 134217728

# Increase TCP autotuning limits
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# Enable TCP Fast Open
net.ipv4.tcp_fastopen = 3

# Increase the maximum number of connections
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# Increase the maximum number of packets queued
net.core.netdev_max_backlog = 65536

# Enable TCP window scaling
net.ipv4.tcp_window_scaling = 1

# Disable TCP timestamps (can improve CPU usage)
net.ipv4.tcp_timestamps = 0

# Enable TCP selective acknowledgments
net.ipv4.tcp_sack = 1

# Increase the TCP fin timeout
net.ipv4.tcp_fin_timeout = 30

# Reuse TIME_WAIT sockets
net.ipv4.tcp_tw_reuse = 1

# Increase keepalive time
net.ipv4.tcp_keepalive_time = 600
net.ipv4.tcp_keepalive_probes = 3
net.ipv4.tcp_keepalive_intvl = 30

# Congestion control
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr

# Enable MTU probing
net.ipv4.tcp_mtu_probing = 1
EOF
    
    sysctl -p /etc/sysctl.d/99-media-server-network.conf
    echo -e "${GREEN}✅ Network optimizations applied${NC}"
}

# Function to optimize file system
optimize_filesystem() {
    echo -e "${BLUE}Applying filesystem optimizations...${NC}"
    
    cat > /etc/sysctl.d/99-media-server-fs.conf << 'EOF'
# Filesystem optimizations for media files

# Increase inotify limits for file watching
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512

# Increase file descriptor limits
fs.file-max = 2097152

# Decrease swap usage
vm.swappiness = 10

# Improve cache management
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# Increase the maximum number of memory map areas
vm.max_map_count = 262144

# Optimize for SSD if applicable
# vm.dirty_expire_centisecs = 12000
EOF
    
    sysctl -p /etc/sysctl.d/99-media-server-fs.conf
    echo -e "${GREEN}✅ Filesystem optimizations applied${NC}"
}

# Function to optimize Docker
optimize_docker() {
    echo -e "${BLUE}Applying Docker optimizations...${NC}"
    
    # Create Docker daemon configuration
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << 'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  },
  "metrics-addr": "127.0.0.1:9323",
  "experimental": true,
  "features": {
    "buildkit": true
  }
}
EOF
    
    # Restart Docker to apply changes
    systemctl restart docker || true
    echo -e "${GREEN}✅ Docker optimizations applied${NC}"
}

# Function to set up hugepages for databases
setup_hugepages() {
    echo -e "${BLUE}Setting up hugepages for databases...${NC}"
    
    # Calculate hugepages (2MB pages, allocate 2GB)
    local hugepages=1024
    
    cat > /etc/sysctl.d/99-hugepages.conf << EOF
# Hugepages for database performance
vm.nr_hugepages = $hugepages
EOF
    
    sysctl -p /etc/sysctl.d/99-hugepages.conf
    echo -e "${GREEN}✅ Hugepages configured${NC}"
}

# Function to optimize CPU governor
optimize_cpu() {
    echo -e "${BLUE}Optimizing CPU governor...${NC}"
    
    # Check if cpufrequtils is installed
    if command -v cpufreq-set &> /dev/null; then
        # Set performance governor
        for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
            echo performance > $cpu/cpufreq/scaling_governor 2>/dev/null || true
        done
        echo -e "${GREEN}✅ CPU governor set to performance${NC}"
    else
        echo -e "${YELLOW}⚠️  cpufrequtils not installed, skipping CPU optimization${NC}"
    fi
}

# Function to create systemd service for optimizations
create_systemd_service() {
    echo -e "${BLUE}Creating systemd service for persistent settings...${NC}"
    
    cat > /etc/systemd/system/media-server-tuning.service << 'EOF'
[Unit]
Description=Media Server Performance Tuning
After=network.target docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/apply-media-tuning.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    
    # Create the script that will be run by systemd
    cat > /usr/local/bin/apply-media-tuning.sh << 'EOF'
#!/bin/bash

# Apply all sysctl settings
sysctl --system

# Set CPU governor if available
if command -v cpufreq-set &> /dev/null; then
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        echo performance > $cpu/cpufreq/scaling_governor 2>/dev/null || true
    done
fi

# Disable transparent hugepages
echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo never > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true

# Set scheduler for SSDs
for disk in /sys/block/sd*; do
    if [ -f "$disk/queue/rotational" ] && [ "$(cat $disk/queue/rotational)" == "0" ]; then
        echo none > $disk/queue/scheduler 2>/dev/null || true
    fi
done
EOF
    
    chmod +x /usr/local/bin/apply-media-tuning.sh
    
    systemctl daemon-reload
    systemctl enable media-server-tuning.service
    systemctl start media-server-tuning.service
    
    echo -e "${GREEN}✅ Systemd service created and enabled${NC}"
}

# Function to display current limits
show_current_limits() {
    echo -e "${BLUE}Current system limits:${NC}"
    echo "====================="
    
    echo -e "${YELLOW}File descriptors:${NC}"
    ulimit -n
    
    echo -e "${YELLOW}Max user processes:${NC}"
    ulimit -u
    
    echo -e "${YELLOW}Network buffers:${NC}"
    sysctl net.core.rmem_max net.core.wmem_max | column -t
    
    echo -e "${YELLOW}Inotify watches:${NC}"
    sysctl fs.inotify.max_user_watches
    
    echo ""
}

# Function to create monitoring script
create_monitoring_script() {
    echo -e "${BLUE}Creating performance monitoring script...${NC}"
    
    cat > /usr/local/bin/media-server-perfmon.sh << 'EOF'
#!/bin/bash

# Performance monitoring for media server

echo "Media Server Performance Monitor"
echo "==============================="
echo ""

# CPU usage
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print "  Total: " $2 + $4 "%"}'
echo ""

# Memory usage
echo "Memory Usage:"
free -h | grep Mem | awk '{print "  Total: " $2 "  Used: " $3 "  Free: " $4}'
echo ""

# Disk I/O
echo "Disk I/O (last 5 seconds):"
iostat -x 1 2 | grep -A1 avg-cpu | tail -n1
echo ""

# Network connections
echo "Network Connections:"
ss -s | grep -E "TCP:|UDP:" | awk '{print "  " $0}'
echo ""

# Docker stats
echo "Docker Container Stats:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10
echo ""

# Active streams (if Jellyfin is running)
if docker ps | grep -q jellyfin; then
    echo "Jellyfin Active Streams:"
    active_streams=$(docker exec jellyfin curl -s "http://localhost:8096/Sessions" | grep -c "NowPlayingItem" || echo "0")
    echo "  Active streams: $active_streams"
fi
EOF
    
    chmod +x /usr/local/bin/media-server-perfmon.sh
    echo -e "${GREEN}✅ Performance monitoring script created${NC}"
}

# Main execution
main() {
    echo -e "${YELLOW}This script will optimize your system for media server performance.${NC}"
    echo -e "${YELLOW}Some changes require a reboot to take full effect.${NC}"
    echo ""
    
    read -p "Do you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    echo ""
    
    # Show current limits
    show_current_limits
    
    # Backup current settings
    backup_settings
    
    # Apply optimizations
    optimize_network
    optimize_filesystem
    optimize_docker
    setup_hugepages
    optimize_cpu
    create_systemd_service
    create_monitoring_script
    
    echo ""
    echo -e "${GREEN}✨ Performance tuning complete!${NC}"
    echo ""
    echo -e "${YELLOW}To monitor performance, run:${NC}"
    echo -e "${BLUE}  sudo /usr/local/bin/media-server-perfmon.sh${NC}"
    echo ""
    echo -e "${YELLOW}Note: A system reboot is recommended to ensure all optimizations are active.${NC}"
}

# Run main function
main "$@"