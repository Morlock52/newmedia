# NewMedia Docker Testing Suite

A comprehensive testing framework for the NewMedia Docker container ecosystem, covering integration, performance, and security testing.

## Quick Start

```bash
# Run all tests
./run-tests.sh

# Run specific test types
./run-tests.sh integration
./run-tests.sh performance
./run-tests.sh security
./run-tests.sh smoke

# Run tests in different environments
./run-tests.sh all ci
./run-tests.sh integration staging
```

## Test Types

### 🔧 Integration Tests
- **Service connectivity** - Verify all containers can communicate
- **API endpoints** - Test REST APIs and health checks
- **Database connections** - Validate PostgreSQL and Redis connectivity
- **Volume permissions** - Check file system access and permissions
- **Docker compose** - Verify service orchestration

**Location**: `integration/`
**Runtime**: ~10-15 minutes
**Command**: `./run-tests.sh integration`

### ⚡ Performance Tests
Load testing using k6 with multiple scenarios:

- **Load Test** - Gradual ramp-up to 100 concurrent users
- **Stress Test** - Push system to 1000+ users to find breaking point
- **Spike Test** - Sudden traffic spikes (viral scenarios)
- **Soak Test** - Sustained load over 4+ hours for memory leak detection

**Location**: `performance/`
**Runtime**: 30 minutes - 4 hours
**Command**: `./run-tests.sh performance`

### 🛡️ Security Tests
Comprehensive security scanning and validation:

- **Vulnerability Scanning** - Trivy scans for CVEs in all images
- **Network Security** - Port exposure and isolation testing
- **Permission Audit** - Container security configuration review
- **Secrets Detection** - Environment variable security checks

**Location**: `security/`
**Runtime**: ~15-20 minutes
**Command**: `./run-tests.sh security`

### 💨 Smoke Tests
Quick validation tests for CI/CD pipelines:

- **Basic connectivity** - Essential service health checks
- **Port accessibility** - Key service port validation
- **Quick response** - Fast endpoint verification

**Runtime**: ~2-3 minutes
**Command**: `./run-tests.sh smoke`

## Test Architecture

```
tests/
├── integration/           # Integration test suite
│   ├── services.test.js   # Container connectivity tests
│   ├── api.test.js        # API endpoint tests
│   ├── volumes.test.js    # Volume and permission tests
│   ├── setup.js           # Test utilities and helpers
│   └── mocks/             # Mock external services
├── performance/           # k6 performance tests
│   ├── load-test.js       # Standard load testing
│   ├── stress-test.js     # System stress testing
│   ├── spike-test.js      # Traffic spike simulation
│   └── soak-test.js       # Long-term stability testing
├── security/              # Security testing tools
│   ├── trivy-scan.sh      # Vulnerability scanning
│   ├── network-test.sh    # Network security tests
│   └── permission-audit.sh # Container security audit
├── reports/               # Test results and reports
├── docker-compose.test.yml # Test environment setup
└── run-tests.sh           # Main test orchestrator
```

## Configuration

### Environment Variables

```bash
# Service URLs (default: localhost)
BASE_URL=http://localhost

# API Keys for authenticated tests
SONARR_API_KEY=your_sonarr_key
RADARR_API_KEY=your_radarr_key
PROWLARR_API_KEY=your_prowlarr_key

# Grafana credentials
GRAFANA_USER=admin
GRAFANA_PASSWORD=admin

# Test behavior
SKIP_SETUP=false           # Skip test environment setup
CLEANUP_AFTER=true         # Clean up after tests
PARALLEL=false             # Run tests in parallel
```

### Test Environment Setup

The test suite automatically:
1. Creates isolated test networks
2. Starts mock external services
3. Sets up test databases (PostgreSQL, Redis)
4. Configures monitoring (InfluxDB, Grafana)
5. Validates service health before testing

## Performance Benchmarks

### Response Time Targets
- **Simple GET**: <100ms (p95)
- **Complex queries**: <500ms (p95)
- **Write operations**: <1000ms (p95)
- **File uploads**: <5000ms (p95)

### Throughput Targets
- **Read-heavy APIs**: >1000 RPS per instance
- **Write-heavy APIs**: >100 RPS per instance
- **Mixed workload**: >500 RPS per instance

### Error Rate Targets
- **5xx errors**: <0.1%
- **4xx errors**: <5% (excluding 401/403)
- **Timeout errors**: <0.01%

## Security Standards

### Container Security
- ✅ Run as non-root user (PUID/PGID 1000)
- ✅ Read-only root filesystem where possible
- ✅ Dropped unnecessary capabilities
- ✅ No privileged mode unless required
- ✅ Limited device access
- ✅ Proper network segmentation

### Network Security
- ✅ Service isolation between networks
- ✅ VPN routing for download clients
- ✅ No unnecessary port exposure
- ✅ SSL/TLS where applicable

## CI/CD Integration

### GitHub Actions Workflow
The test suite integrates with GitHub Actions for automated testing:

```yaml
# Trigger tests on push/PR
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
```

### Test Matrix
- **Node.js versions**: 18, 20
- **Test environments**: CI, staging, production
- **Parallel execution**: Integration and security tests run in parallel
- **Performance gating**: Only runs on labeled PRs or schedules

### Artifacts
- Integration test reports (HTML)
- Performance metrics (JSON, Grafana dashboards)
- Security scan results (SARIF for GitHub Security)
- Container logs (on failure)

## Reports and Monitoring

### Generated Reports
- `reports/integration-test-report.html` - Integration test results
- `reports/performance/` - k6 performance metrics
- `reports/security/` - Vulnerability and security audit reports
- `reports/test-summary.json` - Overall test summary

### Performance Dashboard
During performance tests, access real-time metrics at:
- **Grafana**: http://localhost:3030
- **InfluxDB**: http://localhost:8086

### Security Reports
- Trivy vulnerability scans (JSON/HTML)
- Network security analysis
- Container permission audit
- SARIF format for GitHub Security integration

## Common Test Scenarios

### Development Workflow
```bash
# Quick smoke test during development
./run-tests.sh smoke

# Full integration test before commit
./run-tests.sh integration

# Performance test before release
./run-tests.sh performance
```

### CI/CD Pipeline
```bash
# Smoke tests (fast feedback)
./run-tests.sh smoke ci

# Parallel integration and security
./run-tests.sh integration ci &
./run-tests.sh security ci &
wait

# Performance tests (on specific triggers)
./run-tests.sh performance ci
```

### Production Validation
```bash
# Read-only smoke tests
./run-tests.sh smoke production

# Security validation
./run-tests.sh security production
```

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check container status
   docker-compose ps
   
   # View logs
   docker-compose logs [service_name]
   ```

2. **Port conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8096
   
   # Stop conflicting services
   docker-compose down
   ```

3. **Permission errors**
   ```bash
   # Check volume permissions
   ls -la ./config/
   
   # Fix permissions if needed
   sudo chown -R 1000:1000 ./config/
   ```

4. **Test timeouts**
   ```bash
   # Increase timeout in run-tests.sh
   INTEGRATION_TIMEOUT=3600  # 1 hour
   
   # Or run specific test types
   ./run-tests.sh integration
   ```

### Debug Mode
Enable verbose logging:
```bash
export DEBUG=true
export VERBOSE=true
./run-tests.sh all local
```

## Contributing

### Adding New Tests

1. **Integration Tests**: Add to `integration/` directory
2. **Performance Tests**: Create k6 scripts in `performance/`
3. **Security Tests**: Add shell scripts to `security/`

### Test Guidelines
- Use descriptive test names
- Include proper error handling
- Document expected behavior
- Add cleanup procedures
- Follow existing patterns

### Example Integration Test
```javascript
describe('New Service Tests', () => {
  test('should respond to health check', async () => {
    const response = await apiRequest('newservice', '/health');
    expect(response.status).toBe(200);
    expect(response.data).toHaveProperty('status', 'healthy');
  });
});
```

### Example Performance Test
```javascript
import http from 'k6/http';
import { check } from 'k6';

export default function () {
  const response = http.get('http://localhost:8080/api/endpoint');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
```

## Best Practices

### Test Organization
- Group related tests in describe blocks
- Use clear, descriptive test names
- Keep tests independent and idempotent
- Clean up resources after tests

### Performance Testing
- Start with baseline measurements
- Test realistic user scenarios
- Monitor resource usage
- Set appropriate thresholds

### Security Testing
- Scan all container images
- Test network isolation
- Audit container permissions
- Validate secret management

### Monitoring
- Track test execution time
- Monitor resource usage during tests
- Set up alerting for test failures
- Review security scan results regularly

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review container logs in `reports/logs/`
3. Examine test reports in `reports/`
4. Open an issue with test output and configuration