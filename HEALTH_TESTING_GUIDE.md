# Media Server Health Testing Suite

## Overview

This comprehensive health testing suite provides thorough validation of your media server deployment with 30+ services. It includes service discovery, health checks, dependency validation, load testing, and performance analysis.

## 🚀 Quick Start

### Run Demo (Safe Testing)
```bash
# Quick demo to see what the tools can do
./demo-health-tests.sh
```

### Run Complete Health Check
```bash
# Full comprehensive test (recommended)
./run-all-health-tests.sh

# With custom options
./run-all-health-tests.sh -v --duration 60 --concurrency 20
```

### Run Individual Tests
```bash
# Service discovery only
node comprehensive-service-health-tester.js

# Container monitoring (requires running container)
./container-service-monitor.sh -c your-container-name

# Dependency analysis
python3 service-dependency-validator.py -v --visualize

# Load testing only
node api-load-test-suite.js --duration 30 --concurrency 10
```

## 🧪 Testing Tools

### 1. Comprehensive Service Health Tester
**File:** `comprehensive-service-health-tester.js`

**Features:**
- Tests all 30+ media server services
- Process and port validation
- HTTP health endpoint checks
- Resource usage analysis
- Service dependency verification
- Load testing capabilities
- Comprehensive JSON/HTML reporting

**Usage:**
```bash
node comprehensive-service-health-tester.js
```

### 2. Container Service Monitor  
**File:** `container-service-monitor.sh`

**Features:**
- s6-overlay process supervision validation
- Container-specific health checks
- Resource usage tracking
- Service status with s6-svstat
- Process management validation
- Detailed CSV reporting

**Usage:**
```bash
# Auto-detect container
./container-service-monitor.sh

# Specific container
./container-service-monitor.sh -c ultimate-media-server

# Verbose mode
./container-service-monitor.sh -v -t 60
```

### 3. Service Dependency Validator
**File:** `service-dependency-validator.py`

**Features:**
- Dependency graph analysis
- Circular dependency detection
- Optimal startup order calculation
- Visual dependency graphs
- Service impact analysis
- Startup script generation
- Continuous monitoring mode

**Usage:**
```bash
# Basic validation
python3 service-dependency-validator.py

# With visualization
python3 service-dependency-validator.py --visualize

# Generate startup script
python3 service-dependency-validator.py --startup-script

# Continuous monitoring
python3 service-dependency-validator.py --monitor 30
```

### 4. API Load Test Suite
**File:** `api-load-test-suite.js`

**Features:**
- Concurrent request testing
- Response time analysis (min/max/mean/p95/p99)
- Throughput measurement
- Error rate monitoring
- Performance threshold validation
- Real-time metrics
- HTML reports with charts

**Usage:**
```bash
# Basic load test
node api-load-test-suite.js

# Custom configuration
node api-load-test-suite.js \
  --duration 60 \
  --concurrency 50 \
  --requestRate 100 \
  --outputDir ./load-test-results
```

### 5. Master Test Runner
**File:** `run-all-health-tests.sh`

**Features:**
- Orchestrates all testing tools
- Sequential test execution
- Comprehensive reporting
- Result aggregation
- Performance scoring
- Actionable recommendations

**Usage:**
```bash
# Full test suite
./run-all-health-tests.sh

# Skip load testing (faster)
./run-all-health-tests.sh --skip-load

# Extended load test
./run-all-health-tests.sh --duration 120 --concurrency 50

# Verbose mode
./run-all-health-tests.sh -v
```

## 📊 Service Coverage

The testing suite validates these service categories:

### Critical Tier Services
- **Media Servers:** Jellyfin (8096), Plex (32400), Emby (8097)
- **Databases:** PostgreSQL (5432), Redis (6379), MariaDB (3306)
- **Monitoring:** Prometheus (9090)

### High Priority Services
- ***ARR Stack:** Sonarr (8989), Radarr (7878), Prowlarr (9696)
- **Download Clients:** qBittorrent (8080)
- **Dashboards:** Grafana (3000), Uptime Kuma (3001)

### Medium Priority Services
- **Media Management:** Lidarr (8686), Readarr (8787), Bazarr (6767)
- **Request Management:** Overseerr (5055), Jellyseerr (5056)
- **Download Clients:** Transmission (9091), SABnzbd (8081)
- **Dashboards:** Homarr (7575), Homepage (3003), Tautulli (8181)

### Optional Services
- **Content Libraries:** Calibre-Web (8083), Audiobookshelf (13378), Navidrome (4533)
- **Photo Management:** PhotoPrism (2342), Immich (2283)
- **Utilities:** Vaultwarden (8085), Pi-hole (8053), Syncthing (8384)

## 🎯 Performance Thresholds

Default performance expectations:

- **Average Response Time:** <1000ms
- **95th Percentile Response Time:** <2000ms  
- **99th Percentile Response Time:** <5000ms
- **Error Rate:** <1%
- **Minimum Throughput:** 50 RPS

## 📈 Reporting

### Generated Reports
- **Comprehensive Health Report:** Markdown summary with recommendations
- **Individual Test Results:** JSON/CSV data files
- **Performance Charts:** Visual graphs (PNG/HTML)
- **Load Test Reports:** Detailed HTML reports with metrics
- **Dependency Graphs:** Visual service dependency maps

### Report Structure
```
health-test-results/
├── comprehensive_health_report_TIMESTAMP.md
├── individual-tests/
│   ├── 01_service_discovery_TIMESTAMP.json
│   ├── 02_container_monitoring_TIMESTAMP.csv
│   ├── 03_dependency_validation_TIMESTAMP.json
│   ├── 04_load_testing_TIMESTAMP/
│   └── 05_performance_analysis_TIMESTAMP.json
├── charts/
│   └── dependency_graph.png
├── reports/
│   ├── optimal_startup.sh
│   └── detailed_analysis.html
└── raw-data/
    └── metrics_TIMESTAMP.json
```

## 🔧 Configuration

### Environment Variables
```bash
# Container name (auto-detected if not set)
export CONTAINER_NAME="ultimate-media-server-2025"

# Test timeouts
export HEALTH_CHECK_TIMEOUT=30
export LOAD_TEST_TIMEOUT=10000

# Performance thresholds
export AVG_RESPONSE_TIME_THRESHOLD=1000
export ERROR_RATE_THRESHOLD=1
export THROUGHPUT_THRESHOLD=50
```

### Custom Service Configuration
Edit the service configurations in each tool to add/modify services:

- **JavaScript tools:** Modify `SERVICE_CONFIG` object
- **Shell scripts:** Update `SERVICES` associative array
- **Python tools:** Modify `ServiceConfig` definitions

## 🚨 Troubleshooting

### Common Issues

**"No containers detected"**
- Ensure Docker is running and containers are started
- Check container names with `docker ps`
- Specify container name manually with `-c` option

**"Service not accessible"**
- Verify service ports are open: `netstat -tulpn | grep :8096`
- Check service logs: `docker logs service-name`
- Ensure services have finished starting up

**"Health check timeout"**
- Increase timeout values with `-t` or `--timeout` options
- Check if services are under heavy load
- Verify network connectivity

**"Permission denied"**
- Make scripts executable: `chmod +x *.sh`
- Check Docker permissions: `docker ps` should work without sudo

### Debug Mode
Enable verbose logging:
```bash
# Shell scripts
./script-name.sh -v

# Python scripts  
python3 script-name.py -v

# Set debug environment
export DEBUG=true
```

## 🎯 Best Practices

### Regular Testing
- **Daily:** Quick health checks during active use
- **Weekly:** Comprehensive testing including load tests
- **Monthly:** Full dependency analysis and optimization

### Performance Baselines
1. Run initial comprehensive test after setup
2. Save performance metrics as baseline
3. Compare subsequent tests to identify degradation
4. Set up automated alerting for threshold violations

### Continuous Monitoring
```bash
# Set up continuous dependency monitoring
python3 service-dependency-validator.py --monitor 60 &

# Automated health checks (add to cron)
0 */6 * * * /path/to/run-all-health-tests.sh --skip-load > /var/log/health-check.log
```

## 🛠️ Integration

### CI/CD Pipeline Integration
```yaml
# Example GitHub Actions
- name: Health Check
  run: |
    ./run-all-health-tests.sh --skip-load
    
- name: Load Test
  run: |
    node api-load-test-suite.js --duration 30
```

### Monitoring System Integration
- Export metrics to Prometheus
- Set up Grafana dashboards
- Configure alerting rules
- Integration with Uptime Kuma

### Custom Extensions
The tools are designed to be extensible:
- Add new services to configuration
- Implement custom health checks
- Extend reporting formats
- Add new performance thresholds

## 📚 Technical Details

### Dependencies
- **Node.js 16+** for JavaScript tools
- **Python 3.8+** for dependency validator
- **Bash 4+** for shell scripts
- **Docker** for container testing
- **curl, netstat, ps** for system checks

### Architecture
- **Modular design** with independent tools
- **Concurrent testing** for performance
- **Graceful degradation** when services unavailable
- **Comprehensive error handling** and recovery

### Performance Optimizations  
- **Worker threads** for concurrent load testing
- **Connection pooling** for HTTP requests
- **Efficient metrics collection** with minimal overhead
- **Streaming results** for large datasets

---

## 🎉 Getting Started

1. **Ensure your media server is running:**
   ```bash
   docker ps  # Check containers
   # OR
   systemctl status service-name  # Check host services
   ```

2. **Run the demo to see capabilities:**
   ```bash
   ./demo-health-tests.sh
   ```

3. **Execute comprehensive testing:**
   ```bash
   ./run-all-health-tests.sh -v
   ```

4. **Review the generated reports:**
   ```bash
   ls -la health-test-results/
   cat health-test-results/comprehensive_health_report_*.md
   ```

5. **Set up regular monitoring:**
   ```bash
   # Add to crontab for weekly comprehensive checks
   0 2 * * 0 /path/to/run-all-health-tests.sh
   ```

The testing suite will provide detailed insights into your media server's health, performance bottlenecks, and optimization opportunities. Use the recommendations in the generated reports to improve reliability and performance.

For questions or issues, check the troubleshooting section above or review the detailed logs in the output directory.