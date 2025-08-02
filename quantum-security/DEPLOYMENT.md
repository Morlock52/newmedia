# Quantum Security System - Quick Deployment Guide

## 🚀 Quick Start (5 minutes)

```bash
# 1. Navigate to quantum-security directory
cd quantum-security

# 2. Copy environment configuration
cp .env.example .env

# 3. Generate certificates
./scripts/generate-certs.sh

# 4. Build and start the system
docker-compose up -d

# 5. Verify deployment
./scripts/test-quantum-tls.sh
```

## 📊 Access Points

- **API Server**: https://localhost:8443
- **Health Check**: http://localhost:3000/health
- **Grafana Dashboard**: http://localhost:3001 (admin/quantumsecure)
- **Prometheus**: http://localhost:9090

## 🔒 What You Get

1. **Quantum-Resistant Cryptography**
   - ML-KEM (Kyber) for key exchange
   - ML-DSA (Dilithium) for signatures
   - SLH-DSA (SPHINCS+) for hash-based signatures

2. **Hybrid Security**
   - Combines classical and quantum algorithms
   - Smooth transition from current systems

3. **Real-Time Monitoring**
   - Security threat detection
   - Performance metrics
   - Automated alerts

## 🧪 Test the System

```bash
# Run example client
npm install
node example-client.js

# Or use curl
curl -k https://localhost:8443/health
```

## 🔧 Configuration

Edit `.env` file for:
- Security levels (ML-KEM: 512/768/1024)
- Signature algorithms (ML-DSA/SLH-DSA)
- Performance tuning
- Monitoring settings

## 📚 Documentation

- [API Reference](docs/api-reference.md)
- [Deployment Guide](docs/deployment-guide.md)
- [Security Algorithms](README.md#security-algorithms)

## ⚠️ Production Notes

1. **Replace self-signed certificates** with CA-signed ones
2. **Configure proper domain** instead of localhost
3. **Enable firewall rules** for ports 80, 443, 8443
4. **Set strong passwords** in .env file
5. **Monitor resource usage** - quantum operations are CPU-intensive

## 🆘 Troubleshooting

If containers don't start:
```bash
docker-compose logs -f quantum-security
```

If performance is slow:
- Increase Docker memory allocation
- Use lower security levels for testing
- Enable hybrid mode for better performance

## 🎯 Next Steps

1. **Integrate with your media server** using the API
2. **Configure authentication** for your users
3. **Set up monitoring alerts** for security events
4. **Plan migration** from classical cryptography

---

**Ready to protect against quantum threats!** 🛡️

For detailed information, see the full [deployment guide](docs/deployment-guide.md).