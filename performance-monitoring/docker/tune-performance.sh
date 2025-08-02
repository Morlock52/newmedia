#!/bin/bash
# Docker Container Performance Tuning Script
# Implements automated performance optimizations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Enable Docker BuildKit
enable_buildkit() {
    log_info "Enabling Docker BuildKit..."
    
    # Set in daemon.json
    DAEMON_JSON="/etc/docker/daemon.json"
    
    if [[ ! -f "$DAEMON_JSON" ]]; then
        echo '{}' > "$DAEMON_JSON"
    fi
    
    # Update daemon.json with jq
    if command -v jq &> /dev/null; then
        jq '.features.buildkit = true' "$DAEMON_JSON" > "$DAEMON_JSON.tmp" && mv "$DAEMON_JSON.tmp" "$DAEMON_JSON"
        jq '.["max-concurrent-downloads"] = 10' "$DAEMON_JSON" > "$DAEMON_JSON.tmp" && mv "$DAEMON_JSON.tmp" "$DAEMON_JSON"
        jq '.["max-concurrent-uploads"] = 10' "$DAEMON_JSON" > "$DAEMON_JSON.tmp" && mv "$DAEMON_JSON.tmp" "$DAEMON_JSON"
        jq '.["storage-driver"] = "overlay2"' "$DAEMON_JSON" > "$DAEMON_JSON.tmp" && mv "$DAEMON_JSON.tmp" "$DAEMON_JSON"
        jq '.["storage-opts"] = ["overlay2.override_kernel_check=true"]' "$DAEMON_JSON" > "$DAEMON_JSON.tmp" && mv "$DAEMON_JSON.tmp" "$DAEMON_JSON"
        jq '.["log-driver"] = "json-file"' "$DAEMON_JSON" > "$DAEMON_JSON.tmp" && mv "$DAEMON_JSON.tmp" "$DAEMON_JSON"
        jq '.["log-opts"] = {"max-size": "10m", "max-file": "3"}' "$DAEMON_JSON" > "$DAEMON_JSON.tmp" && mv "$DAEMON_JSON.tmp" "$DAEMON_JSON"
    else
        log_warning "jq not found, manually update $DAEMON_JSON"
    fi
    
    # Set environment variable
    echo "export DOCKER_BUILDKIT=1" >> /etc/profile.d/docker.sh
    
    log_success "BuildKit enabled"
}

# Optimize kernel parameters
optimize_kernel() {
    log_info "Optimizing kernel parameters..."
    
    # Backup current sysctl settings
    cp /etc/sysctl.conf /etc/sysctl.conf.backup.$(date +%Y%m%d)
    
    # Network optimizations
    cat >> /etc/sysctl.conf <<EOF

# Docker Performance Optimizations
# Network Performance
net.core.netdev_max_backlog = 5000
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 16777216
net.core.wmem_default = 16777216
net.core.optmem_max = 40960
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.ipv4.tcp_mtu_probing = 1
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
net.core.somaxconn = 65535
net.ipv4.ip_local_port_range = 1024 65535

# File System
fs.file-max = 2097152
fs.nr_open = 1048576
fs.inotify.max_user_watches = 524288
fs.inotify.max_user_instances = 512

# Virtual Memory
vm.max_map_count = 262144
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1

# Container specific
kernel.pid_max = 4194304
kernel.threads-max = 4194304
EOF

    # Apply settings
    sysctl -p
    
    log_success "Kernel parameters optimized"
}

# Configure container runtime
configure_runtime() {
    log_info "Configuring container runtime..."
    
    # Configure containerd
    if [[ -f /etc/containerd/config.toml ]]; then
        cat >> /etc/containerd/config.toml <<EOF

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
    SystemdCgroup = true
    
[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "runc"
  snapshotter = "overlayfs"
EOF
    fi
    
    # Configure Docker daemon limits
    mkdir -p /etc/systemd/system/docker.service.d
    cat > /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
LimitNOFILE=1048576
LimitNPROC=infinity
LimitCORE=infinity
TasksMax=infinity
EOF

    systemctl daemon-reload
    
    log_success "Container runtime configured"
}

# Setup log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."
    
    cat > /etc/logrotate.d/docker <<EOF
/var/lib/docker/containers/*/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 root root
    sharedscripts
    postrotate
        /usr/bin/docker kill -s USR1 \$(docker ps -q) 2>/dev/null || true
    endscript
}
EOF

    log_success "Log rotation configured"
}

# Optimize storage
optimize_storage() {
    log_info "Optimizing storage..."
    
    # Check if using overlay2
    STORAGE_DRIVER=$(docker info --format '{{.Driver}}')
    
    if [[ "$STORAGE_DRIVER" != "overlay2" ]]; then
        log_warning "Not using overlay2 storage driver. Current: $STORAGE_DRIVER"
        log_info "Consider migrating to overlay2 for better performance"
    fi
    
    # Clean up unused resources
    log_info "Cleaning up unused Docker resources..."
    docker system prune -af --volumes || true
    
    # Configure garbage collection
    cat > /usr/local/bin/docker-gc.sh <<'EOF'
#!/bin/bash
# Docker garbage collection script

# Remove stopped containers older than 24 hours
docker container prune -f --filter "until=24h"

# Remove unused images older than 7 days
docker image prune -af --filter "until=168h"

# Remove unused volumes
docker volume prune -f

# Remove unused networks
docker network prune -f

# Clean build cache older than 7 days
docker builder prune -af --filter "until=168h"
EOF

    chmod +x /usr/local/bin/docker-gc.sh
    
    # Add to cron
    echo "0 2 * * * /usr/local/bin/docker-gc.sh >> /var/log/docker-gc.log 2>&1" | crontab -
    
    log_success "Storage optimization configured"
}

# Setup performance monitoring
setup_monitoring() {
    log_info "Setting up performance monitoring..."
    
    # Create monitoring script
    cat > /usr/local/bin/docker-monitor.sh <<'EOF'
#!/bin/bash
# Docker performance monitoring script

OUTPUT_DIR="/var/log/docker-performance"
mkdir -p "$OUTPUT_DIR"

# Collect system stats
echo "=== System Stats $(date) ===" >> "$OUTPUT_DIR/system.log"
top -bn1 | head -20 >> "$OUTPUT_DIR/system.log"
df -h >> "$OUTPUT_DIR/system.log"
free -m >> "$OUTPUT_DIR/system.log"

# Collect Docker stats
echo "=== Docker Stats $(date) ===" >> "$OUTPUT_DIR/docker.log"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" >> "$OUTPUT_DIR/docker.log"

# Collect container resource usage
for container in $(docker ps --format "{{.Names}}"); do
    echo "=== Container: $container ===" >> "$OUTPUT_DIR/containers.log"
    docker exec "$container" ps aux 2>/dev/null | head -10 >> "$OUTPUT_DIR/containers.log" || true
done
EOF

    chmod +x /usr/local/bin/docker-monitor.sh
    
    # Add to cron (every 5 minutes)
    echo "*/5 * * * * /usr/local/bin/docker-monitor.sh" | crontab -
    
    log_success "Performance monitoring setup complete"
}

# Create performance testing script
create_perf_test() {
    log_info "Creating performance testing utilities..."
    
    cat > /usr/local/bin/docker-perf-test.sh <<'EOF'
#!/bin/bash
# Docker performance testing script

echo "Docker Performance Test Suite"
echo "============================"

# Test: Container startup time
echo -e "\n1. Testing container startup time..."
START_TIME=$(date +%s.%N)
docker run --rm alpine echo "Hello" > /dev/null
END_TIME=$(date +%s.%N)
STARTUP_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "Container startup time: ${STARTUP_TIME}s"

# Test: Image pull performance
echo -e "\n2. Testing image pull performance..."
docker rmi alpine:latest 2>/dev/null || true
START_TIME=$(date +%s.%N)
docker pull alpine:latest > /dev/null
END_TIME=$(date +%s.%N)
PULL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "Image pull time: ${PULL_TIME}s"

# Test: Build performance
echo -e "\n3. Testing build performance..."
TMP_DIR=$(mktemp -d)
cat > "$TMP_DIR/Dockerfile" <<DOCKERFILE
FROM alpine:latest
RUN apk add --no-cache curl
COPY test.txt /test.txt
DOCKERFILE
echo "test" > "$TMP_DIR/test.txt"

START_TIME=$(date +%s.%N)
docker build -t perf-test:latest "$TMP_DIR" > /dev/null
END_TIME=$(date +%s.%N)
BUILD_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "Build time: ${BUILD_TIME}s"

# Test: Network performance
echo -e "\n4. Testing network performance..."
docker network create perf-test 2>/dev/null || true
docker run -d --name perf-server --network perf-test alpine sleep 300
docker run --rm --network perf-test alpine ping -c 5 perf-server

# Cleanup
docker stop perf-server 2>/dev/null || true
docker rm perf-server 2>/dev/null || true
docker network rm perf-test 2>/dev/null || true
docker rmi perf-test:latest 2>/dev/null || true
rm -rf "$TMP_DIR"

echo -e "\nPerformance test complete!"
EOF

    chmod +x /usr/local/bin/docker-perf-test.sh
    
    log_success "Performance testing utilities created"
}

# Apply security hardening
apply_security() {
    log_info "Applying security hardening..."
    
    # Enable user namespaces
    echo '{"userns-remap": "default"}' > /etc/docker/daemon-userns.json
    
    # Set secure defaults
    cat >> /etc/docker/daemon.json <<EOF
{
  "icc": false,
  "log-level": "info",
  "iptables": true,
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true
}
EOF

    log_success "Security hardening applied"
}

# Main execution
main() {
    log_info "Starting Docker performance optimization..."
    
    check_root
    
    # Backup current configuration
    log_info "Backing up current configuration..."
    mkdir -p /root/docker-backup-$(date +%Y%m%d)
    cp -r /etc/docker /root/docker-backup-$(date +%Y%m%d)/ 2>/dev/null || true
    
    # Run optimizations
    enable_buildkit
    optimize_kernel
    configure_runtime
    setup_log_rotation
    optimize_storage
    setup_monitoring
    create_perf_test
    apply_security
    
    # Restart Docker
    log_info "Restarting Docker service..."
    systemctl restart docker
    
    # Run performance test
    log_info "Running performance test..."
    sleep 5
    /usr/local/bin/docker-perf-test.sh
    
    log_success "Docker performance optimization complete!"
    log_info "Monitor performance with: docker stats"
    log_info "Run performance tests with: /usr/local/bin/docker-perf-test.sh"
    log_info "View monitoring logs in: /var/log/docker-performance/"
}

# Run main function
main "$@"