#!/bin/bash
# Performance optimization script for Ultimate Media Server 2025

echo "Applying performance optimizations..."

# Network optimizations
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Increase network buffers
    sudo sysctl -w net.core.rmem_max=134217728
    sudo sysctl -w net.core.wmem_max=134217728
    sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
    sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
    
    # Enable BBR congestion control
    sudo modprobe tcp_bbr
    sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
    
    # Increase file descriptors
    sudo sysctl -w fs.file-max=2097152
    
    # Save permanently
    cat << SYSCTL | sudo tee -a /etc/sysctl.conf
# Media Server Optimizations
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728
net.ipv4.tcp_congestion_control=bbr
fs.file-max=2097152
SYSCTL
fi

# Docker optimizations
cat > /tmp/daemon.json << JSON
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
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
JSON

if [ -f /etc/docker/daemon.json ]; then
    echo "Docker daemon.json exists, please merge manually"
else
    sudo mv /tmp/daemon.json /etc/docker/daemon.json
    sudo systemctl restart docker
fi

echo "Performance optimizations applied!"
