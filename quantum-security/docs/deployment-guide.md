# Quantum Security System Deployment Guide

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Node.js 20+ (for local development)
- 4GB RAM minimum, 8GB recommended
- 10GB available disk space

## Quick Start

1. **Clone and setup:**
```bash
cd quantum-security
cp .env.example .env
# Edit .env with your configuration
```

2. **Generate certificates:**
```bash
./scripts/generate-certs.sh
```

3. **Build and start:**
```bash
docker-compose up -d
```

4. **Verify deployment:**
```bash
./scripts/test-quantum-tls.sh
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                 │
│                 Classical TLS + Rate Limiting            │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│              Quantum Security Server                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Post-Quantum TLS (ML-KEM/Hybrid)        │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │    Authentication (ML-DSA/SLH-DSA Signatures)   │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │        Security Monitoring & Alerting           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴──────────┬──────────────┐
        │                    │              │
┌───────▼──────┐   ┌─────────▼────┐  ┌─────▼─────┐
│    Redis     │   │  Prometheus  │  │  Grafana  │
│   Session    │   │   Metrics    │  │ Dashboard │
│    Store     │   │  Collection  │  │   & Viz   │
└──────────────┘   └──────────────┘  └───────────┘
```

## Security Features

### 1. Post-Quantum Algorithms (NIST Standards)

- **ML-KEM (FIPS 203)**: Key encapsulation based on lattice problems
- **ML-DSA (FIPS 204)**: Digital signatures using lattice cryptography
- **SLH-DSA (FIPS 205)**: Hash-based signatures for long-term security

### 2. Hybrid Cryptography

Combines classical and quantum-resistant algorithms:
- `x25519-mlkem768`: X25519 ECDH + ML-KEM-768
- `secp256r1-mlkem768`: P-256 ECDH + ML-KEM-768
- `x448-mlkem1024`: X448 ECDH + ML-KEM-1024

### 3. Security Monitoring

Real-time threat detection:
- Brute force attack detection
- Timing attack analysis
- Anomaly detection
- Automated remediation

## Production Configuration

### 1. Environment Variables

Critical settings for production:

```bash
# Security
NODE_ENV=production
SIGNATURE_ALGORITHM=ml-dsa-87  # Higher security level
KEY_ROTATION_INTERVAL=43200000  # 12 hours

# Performance
WORKER_THREADS=8
MAX_QUANTUM_OPS_PER_SECOND=5000

# Monitoring
ENABLE_SECURITY_MONITORING=true
ENABLE_AUTO_REMEDIATION=true
```

### 2. TLS Configuration

For production, obtain proper certificates:

```bash
# Using Let's Encrypt
docker run -it --rm \
  -v /etc/letsencrypt:/etc/letsencrypt \
  -v /var/lib/letsencrypt:/var/lib/letsencrypt \
  certbot/certbot certonly \
  --webroot -w /var/www/certbot \
  -d your-domain.com
```

### 3. Scaling Considerations

For high-traffic deployments:

```yaml
# docker-compose.override.yml
services:
  quantum-security:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '4'
          memory: 4G
```

### 4. Hardware Security Module (HSM)

For maximum security, integrate with HSM:

```javascript
// config/hsm.js
const { CloudHSM } = require('@aws-sdk/client-cloudhsm-v2');

module.exports = {
  keyStorage: 'hsm',
  hsmConfig: {
    clusterId: process.env.HSM_CLUSTER_ID,
    region: process.env.AWS_REGION
  }
};
```

## Monitoring & Maintenance

### 1. Access Dashboards

- Grafana: http://localhost:3001 (admin/quantumsecure)
- Prometheus: http://localhost:9090
- Security API: https://localhost:8443/api/security/report

### 2. Key Metrics to Monitor

- Quantum operation latency
- Key rotation events
- Authentication failures
- Threat detection alerts

### 3. Backup Procedures

```bash
# Backup quantum keys (encrypted)
docker exec quantum-security-server \
  node scripts/backup-keys.js

# Backup Redis data
docker exec quantum-redis \
  redis-cli --rdb /backup/dump.rdb BGSAVE
```

## Security Best Practices

1. **Key Rotation**: Enable automatic key rotation
2. **Rate Limiting**: Configure appropriate limits
3. **Network Isolation**: Use Docker networks
4. **Audit Logging**: Enable comprehensive logging
5. **Regular Updates**: Keep algorithms current with NIST standards

## Troubleshooting

### Common Issues

1. **High CPU Usage**: Quantum operations are compute-intensive
   - Solution: Scale horizontally or upgrade hardware

2. **Memory Pressure**: Large key sizes require more RAM
   - Solution: Increase container memory limits

3. **Slow Operations**: Expected with higher security levels
   - Solution: Use hybrid modes for balance

### Debug Mode

```bash
# Enable debug logging
docker-compose run -e LOG_LEVEL=debug quantum-security
```

## Migration from Classical Crypto

1. **Enable Hybrid Mode**: Start with hybrid algorithms
2. **Gradual Transition**: Move services incrementally
3. **Compatibility Testing**: Ensure client support
4. **Performance Baseline**: Monitor impact

## Compliance & Standards

This implementation follows:
- NIST Post-Quantum Cryptography standards
- FIPS 203, 204, 205 specifications
- Industry best practices for key management

## Support

For issues or questions:
- Documentation: /docs
- Security advisories: /security
- Performance tuning: /docs/performance.md