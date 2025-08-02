# Performance Monitoring System - Deployment Guide 🚀

Complete deployment guide for the Performance Monitoring & Analytics System with Docker integration and media server compatibility.

## 📋 Table of Contents

- [Quick Deployment](#quick-deployment)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Integration with Existing Media Server](#integration-with-existing-media-server)
- [Verification & Testing](#verification--testing)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## 🚀 Quick Deployment

### Option 1: Standalone Deployment

```bash
# 1. Navigate to the performance monitoring directory
cd /Users/morlock/fun/newmedia/performance-monitoring

# 2. Start the monitoring stack
docker-compose -f docker/docker-compose.monitoring.yml up -d

# 3. Access the dashboard
open http://localhost:8090
```

### Option 2: Integration with Existing Media Server

```bash
# 1. Navigate to the main project directory
cd /Users/morlock/fun/newmedia

# 2. Add monitoring services to existing compose
docker-compose -f docker-compose.yml \
                -f performance-monitoring/docker/docker-compose.monitoring.yml up -d

# 3. Access monitoring dashboard
open http://localhost:8090
```

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 2 cores
- **RAM**: 4GB available
- **Disk**: 20GB free space
- **OS**: macOS 10.15+, Linux, Windows with WSL2

**Recommended Requirements:**
- **CPU**: 4+ cores
- **RAM**: 8GB+ available
- **Disk**: 50GB+ free space (SSD preferred)
- **Network**: Stable internet connection

### Software Requirements

```bash
# Check Docker version (required: 20.10+)
docker --version

# Check Docker Compose version (required: 2.0+)
docker-compose --version

# Verify system resources
free -h  # Linux/WSL
vm_stat  # macOS
```

### Port Requirements

The monitoring system uses these ports:

| Service | Port | Description |
|---------|------|-------------|
| Performance Dashboard | 8090 | Main monitoring dashboard |
| Performance Monitor API | 8766 | REST API endpoint |
| Performance Monitor WS | 8765 | WebSocket for real-time data |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Visualization (optional) |
| InfluxDB | 8086 | Time series database |
| AlertManager | 9093 | Alert management |
| Node Exporter | 9100 | System metrics |
| cAdvisor | 8080 | Container metrics |

## ⚙️ Configuration

### Step 1: Copy Configuration Files

```bash
# Navigate to monitoring directory
cd /Users/morlock/fun/newmedia/performance-monitoring

# Create config directory if it doesn't exist
mkdir -p config

# Copy example configurations
cp config/monitoring.yml.example config/monitoring.yml
cp config/alerting.yml.example config/alerting.yml
```

### Step 2: Configure Monitoring Settings

Edit `config/monitoring.yml`:

```yaml
# Global settings
global:
  collection_interval: 15  # seconds
  timezone: "America/New_York"  # Your timezone
  environment: "production"

# Data sources - configure based on your setup
data_sources:
  # System monitoring
  system:
    enabled: true
    collectors: ["cpu", "memory", "disk", "network", "processes"]
    
  # Docker container monitoring
  docker:
    enabled: true
    socket_path: "/var/run/docker.sock"
    
  # Application monitoring
  applications:
    enabled: true
    services:
      - name: "jellyfin"
        url: "http://jellyfin:8096"
        health_endpoint: "/health"
        api_key: ""  # Add if required
        
      - name: "sonarr"
        url: "http://sonarr:8989"
        health_endpoint: "/ping"
        api_key: "YOUR_SONARR_API_KEY"
        
      - name: "radarr"
        url: "http://radarr:7878"
        health_endpoint: "/ping"
        api_key: "YOUR_RADARR_API_KEY"
        
      - name: "prowlarr"
        url: "http://prowlarr:9696"
        health_endpoint: "/ping"
        api_key: "YOUR_PROWLARR_API_KEY"

# Performance thresholds
thresholds:
  system:
    cpu_usage_warning: 70
    cpu_usage_critical: 85
    memory_usage_warning: 80
    memory_usage_critical: 90
    disk_usage_warning: 80
    disk_usage_critical: 90
```

### Step 3: Configure Alerting (Optional)

Edit `config/alerting.yml`:

```yaml
# Notification channels
notification_channels:
  - id: "email"
    name: "Email Alerts"
    type: "email"
    enabled: true
    config:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      use_tls: true
      username: "your-email@gmail.com"
      password: "your-app-password"
      from_email: "monitoring@yourdomain.com"
      to_emails: ["admin@yourdomain.com"]
      
  # Add Slack, Discord, or webhook notifications as needed

# Alert rules
alert_rules:
  - id: "high_cpu"
    name: "High CPU Usage"
    metric_pattern: "cpu_usage"
    condition: ">"
    threshold: 85.0
    severity: "high"
    duration_seconds: 300
    notification_channels: ["email"]
    
  - id: "high_memory"
    name: "High Memory Usage"
    metric_pattern: "memory_usage"
    condition: ">"
    threshold: 90.0
    severity: "critical"
    duration_seconds: 180
    notification_channels: ["email"]
```

## 🐳 Docker Deployment

### Method 1: Standalone Monitoring Stack

```bash
# Navigate to monitoring directory
cd /Users/morlock/fun/newmedia/performance-monitoring

# Deploy complete monitoring stack
docker-compose -f docker/docker-compose.monitoring.yml up -d

# Check deployment status
docker-compose -f docker/docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker/docker-compose.monitoring.yml logs -f
```

### Method 2: Integrated with Existing Media Server

```bash
# Navigate to main project directory
cd /Users/morlock/fun/newmedia

# Create integrated compose override
cat > docker-compose.monitoring.override.yml << 'EOF'
version: "3.9"

services:
  # Include monitoring services
  performance-monitor:
    extends:
      file: performance-monitoring/docker/docker-compose.monitoring.yml
      service: performance-monitor
    networks:
      - media_network
      
  performance-dashboard:
    extends:
      file: performance-monitoring/docker/docker-compose.monitoring.yml
      service: performance-dashboard
    networks:
      - media_network
      
  prometheus:
    extends:
      file: performance-monitoring/docker/docker-compose.monitoring.yml
      service: prometheus
    networks:
      - media_network

networks:
  media_network:
    external: true
EOF

# Deploy with monitoring
docker-compose -f docker-compose.yml \
                -f docker-compose.monitoring.override.yml up -d
```

### Method 3: Docker Stack Deployment (Docker Swarm)

```bash
# Initialize Docker Swarm (if not already done)
docker swarm init

# Deploy as stack
docker stack deploy -c docker/docker-compose.monitoring.yml monitoring

# Check stack status
docker stack services monitoring
```

## 🔗 Integration with Existing Media Server

### Step 1: Update Main Docker Compose

Add monitoring network to your main `docker-compose.yml`:

```yaml
networks:
  media_network:
    driver: bridge
  monitoring_network:
    driver: bridge
```

### Step 2: Connect Existing Services

Update your existing services to include monitoring network:

```yaml
services:
  jellyfin:
    # ... existing configuration ...
    networks:
      - media_network
      - monitoring_network
    labels:
      - "monitoring.enable=true"
      - "monitoring.port=8096"
      - "monitoring.health=/health"

  sonarr:
    # ... existing configuration ...
    networks:
      - media_network
      - monitoring_network
    labels:
      - "monitoring.enable=true"
      - "monitoring.port=8989"
      - "monitoring.health=/ping"
```

### Step 3: Update Prometheus Configuration

Create `config/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['prometheus:9090']

  # System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Media server services
  - job_name: 'media-services'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        port: 9323
    relabel_configs:
      - source_labels: [__meta_docker_container_label_monitoring_enable]
        action: keep
        regex: true
```

## ✅ Verification & Testing

### Step 1: Check Service Health

```bash
# Check all monitoring services
docker-compose -f docker/docker-compose.monitoring.yml ps

# Test API endpoints
curl -f http://localhost:8766/health
curl -f http://localhost:9090/-/healthy
curl -f http://localhost:3000/api/health

# Check dashboard
curl -f http://localhost:8090/health
```

### Step 2: Verify Data Collection

```bash
# Check if metrics are being collected
curl -s http://localhost:9090/api/v1/query?query=up | jq .

# Verify container metrics
curl -s http://localhost:8080/api/v2.1/stats/container | jq .

# Test WebSocket connection
wscat -c ws://localhost:8765
```

### Step 3: Test Benchmarks

```bash
# Run quick benchmark test
docker exec -it $(docker-compose -f docker/docker-compose.monitoring.yml ps -q benchmark-runner) \
  python -m benchmarks.performance_benchmarks --quick

# Check benchmark results in dashboard
open http://localhost:8090
```

### Step 4: Test Alerting (Optional)

```bash
# Send test alert
docker exec -it $(docker-compose -f docker/docker-compose.monitoring.yml ps -q performance-monitor) \
  python -m alerting.intelligent_alerting --test-alert

# Check alert in dashboard and notifications
```

## 🏃‍♂️ Running Benchmarks

### Quick Performance Test

```bash
# Navigate to monitoring directory
cd /Users/morlock/fun/newmedia/performance-monitoring

# Run quick benchmark (5-10 minutes)
docker-compose -f docker/docker-compose.monitoring.yml exec benchmark-runner \
  python -m benchmarks.performance_benchmarks --quick

# View results in dashboard
open http://localhost:8090
```

### Full Benchmark Suite

```bash
# Run comprehensive benchmarks (30-45 minutes)
docker-compose -f docker/docker-compose.monitoring.yml exec benchmark-runner \
  python -m benchmarks.performance_benchmarks --full

# Include application-specific tests
docker-compose -f docker/docker-compose.monitoring.yml exec benchmark-runner \
  python -m benchmarks.performance_benchmarks --full --apps
```

### Scheduled Benchmarks

Add to crontab for regular performance testing:

```bash
# Edit crontab
crontab -e

# Add daily benchmark at 2 AM
0 2 * * * cd /Users/morlock/fun/newmedia/performance-monitoring && docker-compose -f docker/docker-compose.monitoring.yml exec -T benchmark-runner python -m benchmarks.performance_benchmarks --quick
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Services Not Starting

**Problem**: Containers exit immediately or fail to start

**Solutions**:
```bash
# Check logs for specific service
docker-compose -f docker/docker-compose.monitoring.yml logs performance-monitor

# Check resource availability
docker stats

# Verify configuration syntax
docker-compose -f docker/docker-compose.monitoring.yml config

# Check port conflicts
netstat -tulpn | grep :8090
```

#### 2. Dashboard Not Loading

**Problem**: Cannot access http://localhost:8090

**Solutions**:
```bash
# Check if service is running
docker-compose -f docker/docker-compose.monitoring.yml ps performance-dashboard

# Test direct API access
curl -v http://localhost:8766/health

# Check firewall/network settings
sudo ufw status  # Linux
pfctl -sr | grep 8090  # macOS

# Verify container networking
docker network ls
docker network inspect monitoring_network
```

#### 3. No Metrics Data

**Problem**: Dashboard shows no data or empty charts

**Solutions**:
```bash
# Check metrics collector
docker-compose -f docker/docker-compose.monitoring.yml logs performance-monitor

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Docker socket access
ls -la /var/run/docker.sock
sudo chmod 666 /var/run/docker.sock  # Temporary fix

# Verify configuration
docker-compose -f docker/docker-compose.monitoring.yml exec performance-monitor \
  cat /app/config/monitoring.yml
```

#### 4. Benchmark Failures

**Problem**: Benchmarks fail or produce invalid results

**Solutions**:
```bash
# Check available disk space
df -h

# Verify memory availability
free -h

# Check permissions
docker-compose -f docker/docker-compose.monitoring.yml exec benchmark-runner \
  ls -la /tmp/benchmark-workspace

# Run individual benchmark
docker-compose -f docker/docker-compose.monitoring.yml exec benchmark-runner \
  python -c "from benchmarks.performance_benchmarks import CPUBenchmark; print(CPUBenchmark().run_single_core_test())"
```

#### 5. Alert System Issues

**Problem**: Alerts not being sent or received

**Solutions**:
```bash
# Test SMTP configuration
docker-compose -f docker/docker-compose.monitoring.yml exec performance-monitor \
  python -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your-email@gmail.com', 'password')
print('SMTP connection successful')
server.quit()
"

# Check alert rules
docker-compose -f docker/docker-compose.monitoring.yml logs | grep -i alert

# Test webhook endpoints
curl -X POST -H "Content-Type: application/json" \
  -d '{"test": "alert"}' \
  https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Performance Optimization

#### System Resource Optimization

```bash
# Increase file descriptor limits
echo "fs.file-max = 65536" | sudo tee -a /etc/sysctl.conf

# Optimize Docker settings
echo '{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}' | sudo tee /etc/docker/daemon.json

sudo systemctl restart docker
```

#### Monitoring Stack Optimization

```yaml
# Add to docker-compose.monitoring.yml
services:
  performance-monitor:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
    environment:
      - WORKERS=4
      - BUFFER_SIZE=10000
```

## 🔧 Maintenance

### Regular Maintenance Tasks

#### Daily Tasks

```bash
# Health check script
#!/bin/bash
cd /Users/morlock/fun/newmedia/performance-monitoring

# Check service health
docker-compose -f docker/docker-compose.monitoring.yml ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# Check disk usage
df -h | grep -E "(monitoring|prometheus|grafana)"

# Check container logs for errors
docker-compose -f docker/docker-compose.monitoring.yml logs --tail=100 | grep -i error
```

#### Weekly Tasks

```bash
# Update containers
docker-compose -f docker/docker-compose.monitoring.yml pull
docker-compose -f docker/docker-compose.monitoring.yml up -d

# Clean up old data
docker exec $(docker-compose -f docker/docker-compose.monitoring.yml ps -q prometheus) \
  promtool tsdb create-blocks-from openmetrics /prometheus --mint=now-7d --maxt=now

# Backup configuration
tar -czf monitoring-config-backup-$(date +%Y%m%d).tar.gz config/
```

#### Monthly Tasks

```bash
# Full system benchmark
docker-compose -f docker/docker-compose.monitoring.yml exec benchmark-runner \
  python -m benchmarks.performance_benchmarks --full --apps

# Review and optimize alert rules
# Update thresholds based on historical data
# Clean up resolved alerts older than 30 days
```

### Backup and Recovery

#### Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups/monitoring"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup configuration files
cp -r config/ "$BACKUP_DIR/$DATE/"

# Backup Docker volumes
docker run --rm -v monitoring_prometheus_data:/source:ro -v "$BACKUP_DIR/$DATE":/backup alpine \
  tar -czf /backup/prometheus_data.tar.gz -C /source .

docker run --rm -v monitoring_grafana_data:/source:ro -v "$BACKUP_DIR/$DATE":/backup alpine \
  tar -czf /backup/grafana_data.tar.gz -C /source .

# Backup databases
docker-compose -f docker/docker-compose.monitoring.yml exec -T influxdb \
  influx backup /tmp/backup
docker cp $(docker-compose -f docker/docker-compose.monitoring.yml ps -q influxdb):/tmp/backup \
  "$BACKUP_DIR/$DATE/influxdb_backup"

echo "Backup completed: $BACKUP_DIR/$DATE"
```

#### Recovery Script

```bash
#!/bin/bash
BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

# Stop services
docker-compose -f docker/docker-compose.monitoring.yml down

# Restore configuration
cp -r "$BACKUP_DIR/config"/* config/

# Restore Docker volumes
docker run --rm -v monitoring_prometheus_data:/target -v "$BACKUP_DIR":/backup alpine \
  tar -xzf /backup/prometheus_data.tar.gz -C /target

docker run --rm -v monitoring_grafana_data:/target -v "$BACKUP_DIR":/backup alpine \
  tar -xzf /backup/grafana_data.tar.gz -C /target

# Start services
docker-compose -f docker/docker-compose.monitoring.yml up -d

echo "Recovery completed from: $BACKUP_DIR"
```

### Monitoring System Health

#### Health Check Script

```bash
#!/bin/bash
# monitoring-health-check.sh

SERVICES=("performance-monitor" "prometheus" "grafana" "performance-dashboard")
ENDPOINTS=(
    "http://localhost:8766/health"
    "http://localhost:9090/-/healthy"
    "http://localhost:3000/api/health"
    "http://localhost:8090/health"
)

echo "=== Monitoring System Health Check ==="
echo "Date: $(date)"
echo

for i in "${!SERVICES[@]}"; do
    service="${SERVICES[$i]}"
    endpoint="${ENDPOINTS[$i]}"
    
    echo -n "Checking $service... "
    
    if curl -sf "$endpoint" > /dev/null 2>&1; then
        echo "✓ Healthy"
    else
        echo "✗ Unhealthy"
        # Check if container is running
        if docker ps --format "{{.Names}}" | grep -q "$service"; then
            echo "  Container is running but endpoint is unhealthy"
        else
            echo "  Container is not running"
        fi
    fi
done

echo
echo "=== Resource Usage ==="
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo
echo "=== Recent Errors ==="
docker-compose -f docker/docker-compose.monitoring.yml logs --tail=10 | grep -i error || echo "No recent errors found"
```

### Performance Tuning

#### Optimize Collection Intervals

```yaml
# config/monitoring.yml - Adjust based on your needs
global:
  collection_interval: 30  # Reduce frequency for lower resource usage
  
# Reduce retention for lower disk usage
retention:
  raw_metrics: "12h"      # Instead of 24h
  hourly_aggregates: "14d" # Instead of 30d
```

#### Scale Services Based on Load

```yaml
# docker-compose.monitoring.yml
services:
  performance-monitor:
    deploy:
      replicas: 2  # Scale horizontally for high load
      
  benchmark-runner:
    profiles:
      - benchmarks  # Only start when needed
```

---

## 🎉 Deployment Complete!

Your Performance Monitoring & Analytics System is now deployed and ready to use!

### Next Steps:

1. **Access the dashboard**: http://localhost:8090
2. **Run initial benchmarks** to establish baselines
3. **Configure alerting** for your specific needs
4. **Set up regular maintenance** tasks
5. **Monitor and analyze** your system performance

### Support:

- Check the [README.md](README.md) for detailed usage instructions
- Review [troubleshooting section](#troubleshooting) for common issues
- Create GitHub issues for bugs or feature requests

**Happy Monitoring!** 🚀📊