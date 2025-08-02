# Ultimate Media Server Monitoring Stack

A comprehensive, enterprise-grade monitoring solution for your media server infrastructure. This monitoring stack provides real-time insights, intelligent alerting, and beautiful visualizations for your Jellyfin, Plex, Arr suite, and entire containerized media ecosystem.

## 🚀 Features

### 📊 **Comprehensive Metrics Collection**
- **System Metrics**: CPU, memory, disk, network, temperature monitoring
- **Container Metrics**: Docker container performance and resource usage
- **Media Server Metrics**: Streaming sessions, transcoding load, user activity
- **Download Metrics**: Speed, queue status, completion rates
- **Network Metrics**: Latency, throughput, connection health
- **Custom Metrics**: Media library stats, user engagement, content analytics

### 🎨 **Beautiful Dashboards**
- **Media Server Overview**: Real-time streaming activity, library stats, system health
- **Performance Deep Dive**: Advanced container and system performance analysis
- **User Activity Analytics**: User engagement, content consumption patterns
- **Container Overview**: Docker container monitoring and resource utilization
- **Resource Usage**: Detailed system resource monitoring
- **Security Alerts**: Security monitoring and incident tracking

### 🚨 **Intelligent Alerting**
- **Smart Alert Routing**: Different notification channels for different alert types
- **Alert Escalation**: Automatic escalation based on severity and duration
- **Rich Notifications**: HTML emails, Discord/Slack webhooks, custom integrations
- **Alert Inhibition**: Reduces noise by grouping related alerts
- **Maintenance Windows**: Configurable quiet periods

### 📝 **Advanced Log Management**
- **Centralized Logging**: All container and system logs in one place
- **Log Processing**: Automatic parsing and enrichment with Vector
- **Log Analytics**: Search, filter, and analyze logs with Loki
- **Event Correlation**: Link logs with metrics for better troubleshooting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Grafana Dashboards                          │
│     ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│     │   Media     │ Performance │    User     │  Security   │  │
│     │  Overview   │ Deep Dive   │ Analytics   │   Alerts    │  │
│     └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Prometheus  │  │    Loki     │  │ Alertmanager│            │
│  │  (Metrics)  │  │   (Logs)    │  │  (Alerts)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Metric Collection                            │
│ ┌─────────────┬─────────────┬─────────────┬─────────────┐      │
│ │    Node     │   cAdvisor  │ Blackbox    │   Custom    │      │
│ │  Exporter   │ (Container) │ (Endpoint)  │ Exporters   │      │
│ └─────────────┴─────────────┴─────────────┴─────────────┘      │
│ ┌─────────────┬─────────────┬─────────────┬─────────────┐      │
│ │  Media      │ Performance │ Speed Test  │   Hardware  │      │
│ │  Exporter   │  Monitor    │  Exporter   │  Monitor    │      │
│ └─────────────┴─────────────┴─────────────┴─────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Media Infrastructure                         │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐     │
│  │  Jellyfin   │   Sonarr    │   Radarr    │ qBittorrent │     │
│  │    Plex     │   Lidarr    │   Bazarr    │   SABnzbd   │     │
│  │    Emby     │  Prowlarr   │  Overseerr  │  Tautulli   │     │
│  └─────────────┴─────────────┴─────────────┴─────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Components

### Core Monitoring Stack
- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization and dashboards
- **Alertmanager** - Alert management and routing
- **Loki** - Log aggregation and search
- **Vector** - Advanced log processing and enrichment

### System Monitoring
- **Node Exporter** - Host system metrics
- **cAdvisor** - Container metrics
- **Docker Exporter** - Docker daemon metrics
- **Process Exporter** - Process-level monitoring

### Network & Connectivity
- **Blackbox Exporter** - Endpoint monitoring and health checks
- **Speed Test Exporter** - Internet speed monitoring
- **SNMP Exporter** - Network device monitoring (optional)

### Custom Media Exporters
- **Media Server Exporter** - Custom metrics for Jellyfin, Sonarr, Radarr, qBittorrent
- **Performance Monitor** - Advanced container and system performance
- **Tautulli Exporter** - Plex/Jellyfin usage statistics

### Hardware Monitoring
- **SMART Exporter** - Disk health monitoring
- **Thermal Monitoring** - Temperature sensors
- **GPU Monitoring** - GPU usage and performance (if available)

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM (8GB recommended)
- 20GB free disk space
- Media server stack already running (optional)

### Installation

1. **Clone or navigate to the monitoring directory:**
   ```bash
   cd /Users/morlock/fun/newmedia/monitoring
   ```

2. **Run the deployment script:**
   ```bash
   ./deploy-monitoring.sh
   ```

3. **Follow the interactive prompts** and wait for deployment to complete.

4. **Access the monitoring interfaces:**
   - Grafana: http://localhost:3000 (admin/admin123)
   - Prometheus: http://localhost:9090
   - Alertmanager: http://localhost:9093

### Alternative Deployment Methods

**Deploy only (without interactive prompts):**
```bash
./deploy-monitoring.sh deploy
```

**Check system status:**
```bash
./deploy-monitoring.sh status
```

**View logs:**
```bash
./deploy-monitoring.sh logs
```

## ⚙️ Configuration

### Environment Variables
Edit `.env` file to configure your monitoring setup:

```bash
# Basic Configuration
GRAFANA_USER=admin
GRAFANA_PASSWORD=your_secure_password
TZ=America/New_York

# Email Alerts
ADMIN_EMAIL=admin@yourdomain.com
SMTP_HOST=smtp.gmail.com:587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Webhook Notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# API Keys for Enhanced Monitoring
JELLYFIN_API_KEY=your_jellyfin_api_key
SONARR_API_KEY=your_sonarr_api_key
RADARR_API_KEY=your_radarr_api_key
TAUTULLI_API_KEY=your_tautulli_api_key
```

### Getting API Keys

**Jellyfin:**
1. Go to Dashboard → API Keys
2. Create new API key
3. Copy the key to your `.env` file

**Sonarr/Radarr:**
1. Go to Settings → General
2. Find API Key in the Security section
3. Copy to your `.env` file

**Tautulli:**
1. Go to Settings → Web Interface
2. Find API Key in the HTTP Authentication section
3. Copy to your `.env` file

## 📊 Dashboard Guide

### Media Server Overview
- **Real-time streaming activity** - Active sessions, transcoding load
- **Library statistics** - Content counts, storage usage
- **Download activity** - Speed, queue status, active downloads
- **System health** - Service status, response times
- **Network performance** - Speed tests, latency monitoring

### Performance Deep Dive
- **Container performance** - CPU, memory, disk I/O per container
- **System resources** - Detailed host metrics
- **Thermal monitoring** - Temperature sensors and GPU usage
- **Network interfaces** - Traffic analysis per interface
- **Top resource consumers** - Identify bottlenecks

### User Activity Analytics
- **Live activity** - Current users and sessions
- **Usage patterns** - Peak hours, content preferences
- **Content consumption** - What's being watched most
- **Bandwidth analysis** - Streaming bandwidth usage
- **Engagement metrics** - User login patterns

## 🚨 Alert Configuration

### Alert Severity Levels
- **Critical** - Service down, system failure (immediate notification)
- **Warning** - Performance degradation, capacity warnings (grouped notification)
- **Info** - Informational alerts, trends (daily digest)

### Default Alert Rules
- **Service Down** - Any media service becomes unreachable
- **High CPU Usage** - CPU usage > 80% for 5 minutes
- **High Memory Usage** - Memory usage > 85% for 5 minutes
- **Disk Space Low** - Disk usage > 85%
- **High Transcoding Load** - More than 3 simultaneous transcodes
- **Slow Download Speed** - Download speed < 5 MB/s with active downloads
- **Network Latency** - Network latency > 100ms for 5 minutes
- **SSL Certificate Expiring** - SSL certificates expiring within 30 days

### Notification Channels
- **Email** - HTML formatted alerts with severity styling
- **Discord** - Rich embeds with color coding
- **Slack** - Formatted messages with alert context
- **Webhook** - Custom integrations for other platforms

## 🔧 Customization

### Adding Custom Metrics
Create custom exporters by adding them to `docker-compose.monitoring.yml`:

```yaml
your-custom-exporter:
  image: your/custom-exporter:latest
  ports:
    - "9999:9999"
  networks:
    - monitoring
  labels:
    - "prometheus.io/scrape=true"
    - "prometheus.io/port=9999"
```

### Custom Dashboard
1. Create dashboard in Grafana UI
2. Export JSON
3. Save to `grafana/dashboards/` directory
4. Dashboard will be automatically loaded on restart

### Custom Alert Rules
Add rules to `prometheus/rules/custom-alerts.yml`:

```yaml
groups:
  - name: custom.rules
    rules:
      - alert: CustomAlert
        expr: your_metric > threshold
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Custom alert description"
```

## 🔍 Troubleshooting

### Common Issues

**Grafana won't start:**
```bash
# Check permissions
sudo chown -R 472:472 ./data/grafana
```

**Prometheus can't scrape targets:**
```bash
# Check if services are on the same network
docker network ls
docker network inspect media_network
```

**No data in dashboards:**
```bash
# Check if Prometheus is collecting metrics
curl http://localhost:9090/api/v1/label/__name__/values
```

**Alerts not firing:**
```bash
# Check alert rules syntax
docker exec prometheus promtool check rules /etc/prometheus/rules/*.yml
```

### Logs Analysis
View specific service logs:
```bash
docker-compose -f docker-compose.monitoring.yml logs prometheus
docker-compose -f docker-compose.monitoring.yml logs grafana
docker-compose -f docker-compose.monitoring.yml logs alertmanager
```

### Performance Tuning

**For systems with limited resources:**
1. Reduce scrape intervals in `prometheus.yml`
2. Decrease retention time in Prometheus
3. Limit dashboard refresh rates
4. Disable unnecessary exporters

**For high-traffic systems:**
1. Increase Prometheus storage retention
2. Add Prometheus federation for scaling
3. Use external storage (e.g., Thanos)
4. Implement metric cardinality limits

## 📚 Documentation Links

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Loki Documentation](https://grafana.com/docs/loki/latest/)
- [Docker Monitoring Best Practices](https://docs.docker.com/config/daemon/prometheus/)

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs using `./deploy-monitoring.sh logs`
3. Check service health with `./deploy-monitoring.sh status`
4. Consult the official documentation links

## 📄 License

This monitoring configuration is provided as-is for educational and personal use. Please ensure compliance with the licenses of all included components.

---

**Happy Monitoring!** 🚀📊📈

Your media server infrastructure is now equipped with enterprise-grade monitoring capabilities. Enjoy the insights and stay ahead of any issues!