# Ultimate Media Server 2025 - Test Configuration

This directory contains all configuration files and scripts for the comprehensive testing system.

## Files Overview

### Test Scripts
- `../test-container-isolation.sh` - Container isolation and resource testing
- `../test-api-connectivity.py` - API endpoint and authentication testing  
- `../test-service-integrations.sh` - Service-to-service integration testing
- `../run-all-tests.sh` - Master test orchestrator

### Configuration Files
- `pytest.ini` - Python test configuration
- `api-test-config.yaml` - API testing configuration (auto-generated)
- `k6-performance-test.js` - Performance testing scenarios
- `docker-security-scan.sh` - Security vulnerability scanning

### Auto-Generated Files
These files are created automatically by the test runner:
- `performance-config.js` - k6 performance test configuration
- `api-test-config.yaml` - Service API configurations

## Quick Start

### 1. Basic Test Run
```bash
# Run all tests in local environment
./run-all-tests.sh

# Run quick smoke tests
./run-all-tests.sh --quick smoke

# Run specific test types
./run-all-tests.sh api integration
```

### 2. Environment-Specific Testing
```bash
# CI/CD pipeline testing
./run-all-tests.sh --environment ci --parallel all

# Staging environment validation
./run-all-tests.sh --environment staging

# Production read-only checks
./run-all-tests.sh --environment production smoke
```

### 3. Performance Testing
```bash
# Run performance tests with k6
./run-all-tests.sh performance

# Custom performance test
k6 run test-config/k6-performance-test.js
```

### 4. Security Scanning
```bash
# Comprehensive security scan
./test-config/docker-security-scan.sh

# Include security in test suite
./run-all-tests.sh security
```

## Test Configuration

### Environment Variables
Set these environment variables for authenticated testing:

```bash
# API Keys
export SONARR_API_KEY="your_sonarr_api_key"
export RADARR_API_KEY="your_radarr_api_key"
export LIDARR_API_KEY="your_lidarr_api_key"
export PROWLARR_API_KEY="your_prowlarr_api_key"
export BAZARR_API_KEY="your_bazarr_api_key"
export JELLYSEERR_API_KEY="your_jellyseerr_api_key"
export OVERSEERR_API_KEY="your_overseerr_api_key"
export OMBI_API_KEY="your_ombi_api_key"

# Media Server Authentication
export JELLYFIN_API_KEY="your_jellyfin_api_key"
export PLEX_TOKEN="your_plex_token"
export EMBY_API_KEY="your_emby_api_key"

# Download Client Authentication
export QBITTORRENT_USERNAME="admin"
export QBITTORRENT_PASSWORD="your_qbt_password"
export SABNZBD_API_KEY="your_sabnzbd_api_key"
export NZBGET_USERNAME="nzbget"
export NZBGET_PASSWORD="your_nzbget_password"

# Monitoring Authentication
export GRAFANA_USERNAME="admin"
export GRAFANA_PASSWORD="your_grafana_password"

# Base URL (if not localhost)
export BASE_URL="http://your-server"
```

### Custom Configuration

#### API Test Configuration (`api-test-config.yaml`)
```yaml
services:
  jellyfin:
    api_key: "${JELLYFIN_API_KEY}"
    timeout: 30
    test_endpoints:
      - "/System/Info"
      - "/Users"
  
  sonarr:
    api_key: "${SONARR_API_KEY}"
    timeout: 30
    test_endpoints:
      - "/api/v3/system/status"
      - "/api/v3/series"

global:
  base_url: "http://localhost"
  verify_ssl: false
  quick_mode: false
```

#### Performance Test Configuration
```javascript
export let options = {
  scenarios: {
    load_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },
        { duration: '5m', target: 10 },
        { duration: '2m', target: 0 },
      ],
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<5000'],
    http_req_failed: ['rate<0.1'],
  },
};
```

## Test Types Explained

### Container Isolation Tests
- ✅ Container health and status
- ✅ Resource limits and usage
- ✅ Network isolation
- ✅ Volume permissions
- ✅ Security configuration
- ✅ Inter-container connectivity
- ✅ Port exposure
- ✅ Restart policies

### API Connectivity Tests
- ✅ Health endpoint validation
- ✅ Authentication methods
- ✅ API response times
- ✅ Error handling
- ✅ SSL/TLS verification
- ✅ Rate limiting compliance
- ✅ API versioning support

### Service Integration Tests
- ✅ ARR suite ↔ Prowlarr synchronization
- ✅ ARR services ↔ Download clients
- ✅ Media servers ↔ Request services
- ✅ Monitoring ↔ All services
- ✅ Database connections
- ✅ VPN network isolation
- ✅ Volume data flow
- ✅ Dashboard integrations

### Performance Tests
- ✅ Load testing (gradual ramp-up)
- ✅ Stress testing (breaking point)
- ✅ Spike testing (sudden traffic)
- ✅ Soak testing (sustained load)
- ✅ Response time analysis
- ✅ Throughput measurement
- ✅ Resource utilization

### Security Tests
- ✅ Container vulnerability scanning
- ✅ Security configuration audit
- ✅ Network security analysis
- ✅ Port exposure assessment
- ✅ Privilege escalation checks
- ✅ Secret management validation

## Test Results

All test results are stored in the `../test-results/` directory:

```
test-results/
├── container-isolation-20250802_143022.log
├── container-isolation-report.json
├── api-connectivity-20250802_143022.log
├── api-connectivity-report-20250802_143022.json
├── service-integrations-20250802_143022.log
├── service-integrations-report.json
├── performance-summary.html
├── performance-summary.json
├── security/
│   ├── trivy-summary.json
│   ├── container-security-audit.json
│   ├── network-security-analysis.json
│   └── security-summary.json
├── final-test-report.json
└── test-report.html
```

### Report Formats

#### JSON Reports
Structured data for CI/CD integration:
```json
{
  "timestamp": "2025-08-02T14:30:22Z",
  "test_suite": "API Connectivity Tests",
  "summary": {
    "total_tests": 45,
    "passed_tests": 42,
    "failed_tests": 3,
    "success_rate": 93.33
  },
  "detailed_results": [...]
}
```

#### HTML Reports
Human-readable reports with:
- Visual charts and graphs
- Color-coded results
- Detailed test breakdowns
- Performance metrics
- Security findings

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Media Server Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup test environment
        run: |
          sudo apt-get update
          sudo apt-get install -y jq curl netcat
          
      - name: Start services
        run: docker-compose up -d
        
      - name: Wait for services
        run: sleep 60
        
      - name: Run smoke tests
        run: ./run-all-tests.sh --environment ci --quick smoke
        
      - name: Run integration tests
        run: ./run-all-tests.sh --environment ci integration
        
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results
          path: test-results/
```

### Jenkins Pipeline Example
```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'docker-compose up -d'
                sh 'sleep 60'
            }
        }
        
        stage('Tests') {
            parallel {
                stage('API Tests') {
                    steps {
                        sh './run-all-tests.sh --environment ci api'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh './run-all-tests.sh --environment ci integration'
                    }
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                sh './run-all-tests.sh --environment ci security'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'test-results/**/*'
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'test-results',
                reportFiles: 'test-report.html',
                reportName: 'Test Report'
            ])
        }
        cleanup {
            sh 'docker-compose down'
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check container status
   docker-compose ps
   
   # View service logs
   docker-compose logs jellyfin
   ```

2. **API authentication failures**
   ```bash
   # Verify API keys are set
   echo $SONARR_API_KEY
   
   # Check service configuration
   docker exec sonarr cat /config/config.xml
   ```

3. **Network connectivity issues**
   ```bash
   # Test container networking
   docker exec jellyfin ping sonarr
   
   # Check port exposure
   netstat -tulpn | grep :8096
   ```

4. **Permission errors**
   ```bash
   # Check volume permissions
   ls -la config/
   
   # Fix permissions if needed
   sudo chown -R 1000:1000 config/
   ```

### Debug Mode

Enable verbose logging:
```bash
export DEBUG=true
export VERBOSE=true
./run-all-tests.sh --environment local all
```

### Test Isolation

Run tests in isolation:
```bash
# Run only container tests
./test-container-isolation.sh

# Run only API tests
python3 test-api-connectivity.py --quick

# Run only integration tests
./test-service-integrations.sh
```

## Best Practices

### Test Development
- Keep tests independent and idempotent
- Use descriptive test names
- Include proper cleanup procedures
- Follow existing patterns
- Document expected behavior

### Performance Testing
- Start with baseline measurements
- Test realistic user scenarios
- Monitor resource usage
- Set appropriate thresholds

### Security Testing
- Scan all container images regularly
- Audit container configurations
- Test network isolation
- Validate secret management

### Continuous Integration
- Run smoke tests on every commit
- Schedule full test suite daily
- Use parallel execution for speed
- Generate artifacts for analysis

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review test logs in `../test-results/`
3. Examine container logs with `docker-compose logs`
4. Open an issue with test output and configuration