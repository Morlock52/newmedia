# Performance Monitoring & Analytics System 🚀

A comprehensive real-time performance monitoring and analytics system with advanced benchmarking, intelligent optimization, and ML-powered insights for media server infrastructure.

## 🌟 Features

### Core Monitoring Capabilities
- **Real-time Metrics Collection**: System, application, and Docker container metrics
- **Advanced Benchmarking**: CPU, memory, disk, network, and application performance tests
- **Intelligent Optimization**: Automated performance recommendations and implementations
- **ML-Powered Analytics**: Anomaly detection and predictive maintenance
- **Interactive Dashboard**: Beautiful real-time visualizations and insights

### Media Server Integration
- **Jellyfin Monitoring**: Streaming sessions, transcoding, library stats
- **Arr Suite Integration**: Sonarr, Radarr, Prowlarr queue and health monitoring
- **Tautulli Support**: Plex usage analytics and statistics
- **Docker Container Metrics**: Resource usage and performance tracking

### Intelligent Alerting
- **Smart Correlation**: Reduces alert noise by grouping related issues
- **ML Anomaly Detection**: Automatically learns normal patterns
- **Multi-Channel Notifications**: Email, Slack, Discord, webhooks
- **Escalation Management**: Automated escalation policies
- **Rate Limiting**: Prevents notification spam

### Performance Optimization
- **Automated Analysis**: Identifies bottlenecks and optimization opportunities
- **Smart Recommendations**: Prioritized suggestions with impact estimates
- **Safe Implementation**: Automated optimizations with rollback capabilities
- **Dependency Management**: Handles optimization dependencies intelligently

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Performance Dashboard                        │
│                (Real-time Analytics)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│              Intelligent Alerting                           │
│  ┌─────────────────┐ │ ┌─────────────────┐                 │
│  │ ML Anomaly      │ │ │ Smart           │                 │
│  │ Detection       │ │ │ Correlation     │                 │
│  └─────────────────┘ │ └─────────────────┘                 │
│  ┌─────────────────┐ │ ┌─────────────────┐                 │
│  │ Multi-Channel   │ │ │ Escalation      │                 │
│  │ Notifications   │ │ │ Management      │                 │
│  └─────────────────┘ │ └─────────────────┘                 │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│            Performance Optimization                         │
│  ┌─────────────────┐ │ ┌─────────────────┐                 │
│  │ Bottleneck      │ │ │ Automated       │                 │
│  │ Detection       │ │ │ Optimizations   │                 │
│  └─────────────────┘ │ └─────────────────┘                 │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│              Data Processing Layer                          │
│  ┌─────────────────┐ │ ┌─────────────────┐                 │
│  │ Metrics         │ │ │ Benchmark       │                 │
│  │ Collector       │ │ │ Engine          │                 │
│  └─────────────────┘ │ └─────────────────┘                 │
│  ┌─────────────────┐ │ ┌─────────────────┐                 │
│  │ Media Server    │ │ │ Container       │                 │
│  │ Integration     │ │ │ Monitoring      │                 │
│  └─────────────────┘ │ └─────────────────┘                 │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                Media Server Stack                           │
│          (Jellyfin, Sonarr, Radarr, etc.)                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM (8GB+ recommended)
- 20GB+ disk space
- Python 3.11+ (for development)

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd newmedia/performance-monitoring
   ```

2. **Configure the system**:
   ```bash
   cp config/monitoring.yml.example config/monitoring.yml
   # Edit config/monitoring.yml with your settings
   ```

3. **Deploy the monitoring stack**:
   ```bash
   docker-compose -f docker/docker-compose.monitoring.yml up -d
   ```

4. **Access the dashboard**:
   ```
   http://localhost:8090
   ```

### Manual Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the system**:
   ```bash
   cp config/monitoring.yml.example config/monitoring.yml
   # Edit configuration as needed
   ```

3. **Start the services**:
   ```bash
   # Terminal 1: Start metrics collector
   python -m core.metrics_collector
   
   # Terminal 2: Start benchmark runner
   python -m benchmarks.performance_benchmarks --full
   
   # Terminal 3: Start optimization engine
   python -m optimization.performance_optimizer --analyze
   
   # Terminal 4: Start alerting system
   python -m alerting.intelligent_alerting
   ```

## 📊 Dashboard Access

### Primary Dashboards

- **Performance Dashboard**: http://localhost:8090
  - Real-time system metrics
  - Interactive performance charts
  - Benchmark results and trends
  - Optimization recommendations

- **Grafana** (if enabled): http://localhost:3000
  - Traditional monitoring dashboards
  - Historical data analysis
  - Custom alerting rules
  - Advanced visualizations

- **Prometheus** (if enabled): http://localhost:9090
  - Raw metrics exploration
  - Query interface
  - Target health status

### Mobile Support

The dashboard is fully responsive and optimized for mobile devices with:
- Touch-friendly interface
- Adaptive layouts
- Offline capability (PWA)
- Push notifications

## 🔧 Configuration

### Basic Configuration

Edit `config/monitoring.yml`:

```yaml
# Collection settings
global:
  collection_interval: 15  # seconds
  retention:
    raw_metrics: "24h"
    hourly_aggregates: "30d"
    daily_aggregates: "365d"

# Data sources
data_sources:
  system:
    enabled: true
    collectors: ["cpu", "memory", "disk", "network"]
    
  applications:
    enabled: true
    services:
      - name: "jellyfin"
        url: "http://jellyfin:8096"
        health_endpoint: "/health"
        
  docker:
    enabled: true
    socket_path: "/var/run/docker.sock"

# Performance thresholds
thresholds:
  system:
    cpu_usage_warning: 70
    cpu_usage_critical: 85
    memory_usage_warning: 80
    memory_usage_critical: 90
```

### Media Server Integration

Configure your media services in `config/monitoring.yml`:

```yaml
media_services:
  jellyfin:
    enabled: true
    url: "http://jellyfin:8096"
    api_key: "your-api-key"
    container_name: "jellyfin"
    
  sonarr:
    enabled: true
    url: "http://sonarr:8989"
    api_key: "your-sonarr-api-key"
    container_name: "sonarr"
    
  radarr:
    enabled: true
    url: "http://radarr:7878"
    api_key: "your-radarr-api-key"
    container_name: "radarr"
```

### Alerting Configuration

Configure notifications in `config/alerting.yml`:

```yaml
notification_channels:
  - id: "email"
    name: "Email Notifications"
    type: "email"
    enabled: true
    config:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      username: "your-email@gmail.com"
      password: "your-app-password"
      from_email: "monitoring@yourdomain.com"
      to_emails: ["admin@yourdomain.com"]
      
  - id: "slack"
    name: "Slack Alerts"
    type: "slack"
    enabled: true
    config:
      webhook_url: "https://hooks.slack.com/services/..."
      channel: "#monitoring"
      username: "Performance Bot"

alert_rules:
  - id: "high_cpu"
    name: "High CPU Usage"
    metric_pattern: "cpu_usage"
    condition: ">"
    threshold: 85.0
    severity: "high"
    duration_seconds: 300
    notification_channels: ["email", "slack"]
```

## 🔬 Benchmarking

### Running Benchmarks

```bash
# Quick benchmark (10-15 minutes)
python -m benchmarks.performance_benchmarks --quick

# Full benchmark suite (30-45 minutes)
python -m benchmarks.performance_benchmarks --full

# Include application tests
python -m benchmarks.performance_benchmarks --full --apps

# Custom configuration
python -m benchmarks.performance_benchmarks --config config/benchmark.yml
```

### Benchmark Categories

- **CPU Tests**: Single-core, multi-core, prime calculations
- **Memory Tests**: Allocation performance, bandwidth testing
- **Disk I/O**: Sequential read/write, random I/O, IOPS
- **Network Tests**: Latency, DNS resolution, bandwidth
- **Application Tests**: HTTP load testing, media streaming

### Results Analysis

Benchmarks generate detailed reports including:
- Performance scores and comparisons
- Historical trend analysis
- Baseline comparisons
- Optimization recommendations
- System configuration impact

## ⚡ Performance Optimization

### Automated Analysis

```bash
# Analyze system performance
python -m optimization.performance_optimizer --analyze

# Apply recommended optimizations
python -m optimization.performance_optimizer --apply-all

# Dry run mode (preview changes)
python -m optimization.performance_optimizer --apply-all --dry-run
```

### Optimization Categories

- **CPU Optimizations**: Governor settings, process scheduling
- **Memory Optimizations**: Swap configuration, cache tuning
- **Disk Optimizations**: I/O scheduler, filesystem tuning
- **Network Optimizations**: Buffer sizes, congestion control
- **Application Optimizations**: Container resources, service tuning

### Safety Features

- **Dry Run Mode**: Preview changes before applying
- **Rollback Support**: Automatic rollback on failures
- **Dependency Management**: Handle optimization dependencies
- **Impact Estimation**: Predict performance improvements
- **Risk Assessment**: Identify potential risks

## 🚨 Alerting & Monitoring

### Smart Alerting Features

- **ML Anomaly Detection**: Learns normal behavior patterns
- **Alert Correlation**: Groups related alerts to reduce noise
- **Intelligent Escalation**: Automated escalation policies
- **Multi-Channel Notifications**: Email, Slack, Discord, webhooks
- **Rate Limiting**: Prevents notification spam

### Setting Up Alerts

1. **Configure notification channels** in `config/alerting.yml`
2. **Define alert rules** with conditions and thresholds
3. **Set up escalation policies** for critical alerts
4. **Test notifications** with test alerts

### Alert Management

- **Acknowledge alerts** to stop escalation
- **Suppress alerts** temporarily
- **View alert history** and trends
- **Customize notification preferences**

## 📈 API Reference

### REST API Endpoints

```bash
# System metrics
GET /api/v1/metrics/system
GET /api/v1/metrics/applications
GET /api/v1/metrics/docker

# Benchmarks
GET /api/v1/benchmarks/results
POST /api/v1/benchmarks/run
GET /api/v1/benchmarks/history

# Optimization
GET /api/v1/optimization/analyze
POST /api/v1/optimization/apply
GET /api/v1/optimization/recommendations

# Alerts
GET /api/v1/alerts
POST /api/v1/alerts/{id}/acknowledge
POST /api/v1/alerts/{id}/resolve

# Health check
GET /api/v1/health
```

### WebSocket API

Real-time data streaming:

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'metrics') {
        // Handle real-time metrics
        updateDashboard(data.payload);
    } else if (data.type === 'alert') {
        // Handle new alert
        showAlert(data.payload);
    }
};
```

## 🐳 Docker Deployment

### Production Deployment

```bash
# Deploy full monitoring stack
docker-compose -f docker/docker-compose.monitoring.yml up -d

# Deploy with GPU support
docker-compose -f docker/docker-compose.monitoring.yml \
                -f docker/docker-compose.gpu.yml up -d

# Scale specific services
docker-compose -f docker/docker-compose.monitoring.yml \
                up -d --scale benchmark-runner=2
```

### Service Management

```bash
# View service status
docker-compose -f docker/docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker/docker-compose.monitoring.yml logs -f performance-monitor

# Restart services
docker-compose -f docker/docker-compose.monitoring.yml restart

# Update services
docker-compose -f docker/docker-compose.monitoring.yml pull
docker-compose -f docker/docker-compose.monitoring.yml up -d
```

## 🔐 Security

### Security Features

- **Encrypted Communications**: TLS for all data transfers
- **Access Controls**: Authentication and authorization
- **Audit Logging**: Complete audit trail
- **Data Privacy**: No personal data collection
- **Secure Defaults**: Security-first configuration

### Securing Your Installation

1. **Change default passwords** in configuration files
2. **Enable authentication** for web interfaces
3. **Configure TLS certificates** for production
4. **Set up firewall rules** to restrict access
5. **Regular security updates** for all components

## 📊 Performance Metrics

### Key Performance Indicators

- **System Health Score**: Overall system performance (0-100)
- **Response Time**: Application response times
- **Resource Utilization**: CPU, memory, disk usage
- **Availability**: Service uptime percentage
- **Throughput**: Requests per second, data transfer rates

### Benchmark Results

Typical benchmark scores on modern hardware:

- **CPU Single Core**: 45,000-55,000 ops/sec
- **CPU Multi Core**: 180,000-220,000 ops/sec
- **Memory Bandwidth**: 12-15 GB/s
- **Disk Sequential Write**: 450-550 MB/s (SSD)
- **Network Latency**: 15-25ms (internet)

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd newmedia/performance-monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .
flake8 .
```

### Project Structure

```
performance-monitoring/
├── core/                 # Core monitoring functionality
│   ├── metrics_collector.py
│   └── ...
├── benchmarks/           # Performance benchmarking
│   ├── performance_benchmarks.py
│   └── ...
├── optimization/         # Performance optimization
│   ├── performance_optimizer.py
│   └── ...
├── alerting/            # Intelligent alerting
│   ├── intelligent_alerting.py
│   └── ...
├── integration/         # Media server integration
│   ├── media_server_integration.py
│   └── ...
├── dashboard/           # Web dashboard
│   ├── analytics_dashboard.html
│   └── ...
├── docker/              # Docker configurations
│   ├── docker-compose.monitoring.yml
│   └── ...
└── config/              # Configuration files
    ├── monitoring.yml
    └── ...
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📚 Documentation

### Additional Resources

- [API Documentation](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Troubleshooting Guide](docs/troubleshooting.md)
- [Best Practices](docs/best-practices.md)
- [Deployment Guide](docs/deployment.md)

## 🐛 Troubleshooting

### Common Issues

**Dashboard not loading**:
- Check if services are running: `docker-compose ps`
- Verify network connectivity: `curl http://localhost:8090/health`
- Check logs: `docker-compose logs performance-monitor`

**Metrics not collecting**:
- Verify Docker socket access: `ls -la /var/run/docker.sock`
- Check permissions for system monitoring
- Review configuration file syntax

**Benchmarks failing**:
- Ensure sufficient disk space for temporary files
- Check system resources (CPU, memory availability)
- Verify benchmark workspace permissions

**Alerts not sending**:
- Test notification channels individually
- Check rate limiting configuration
- Verify SMTP/webhook credentials

### Getting Help

- **GitHub Issues**: [Report bugs and request features](https://github.com/your-repo/issues)
- **Documentation**: Check the docs/ directory for detailed guides
- **Logs**: Always check application logs for error details

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization platform
- **scikit-learn**: Machine learning algorithms
- **Chart.js**: Interactive charts
- **Docker**: Containerization platform

---

**Performance Monitoring & Analytics System** - Bringing intelligent monitoring to your media infrastructure. 🚀📊🔍