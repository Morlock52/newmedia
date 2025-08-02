# Quantum Security API Reference

## Base URL

```
https://localhost:8443
```

## Authentication

All API endpoints except health checks require quantum-resistant authentication.

### Authentication Flow

1. **Login** → Receive quantum challenge
2. **Sign challenge** → Using quantum private key
3. **Submit signature** → Receive access token
4. **Use token** → In Authorization header

## Endpoints

### Health & Monitoring

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-01-01T00:00:00.000Z",
  "security": "SECURE",
  "uptime": 3600000
}
```

#### GET /api/security/report
Get comprehensive security report.

**Response:**
```json
{
  "status": "SECURE",
  "uptime": 3600000,
  "metrics": {
    "totalOperations": 1234,
    "operationsPerSecond": 0.34,
    "activeThreats": 0,
    "resolvedThreats": 2,
    "errorRate": 0.001
  },
  "activeAlerts": [],
  "recommendations": []
}
```

### Quantum Cryptography

#### POST /api/crypto/keygen/:algorithm
Generate quantum-resistant keypair.

**Parameters:**
- `algorithm`: `ml-kem`, `ml-dsa`, or `slh-dsa`

**Request Body:**
```json
{
  "securityLevel": 768
}
```

**Security Levels:**
- ML-KEM: 512, 768, 1024
- ML-DSA: 44, 65, 87
- SLH-DSA: 128, 192, 256

**Response:**
```json
{
  "success": true,
  "publicKey": "base64-encoded-public-key",
  "privateKey": "base64-encoded-private-key",
  "algorithm": "ML-KEM-768"
}
```

#### POST /api/crypto/encrypt
Encrypt data using ML-KEM.

**Request Body:**
```json
{
  "publicKey": "base64-encoded-public-key",
  "securityLevel": 768
}
```

**Response:**
```json
{
  "success": true,
  "ciphertext": "base64-encoded-ciphertext",
  "sharedSecret": "hex-encoded-shared-secret"
}
```

#### POST /api/crypto/decrypt
Decrypt data using ML-KEM.

**Request Body:**
```json
{
  "privateKey": "base64-encoded-private-key",
  "ciphertext": "base64-encoded-ciphertext",
  "securityLevel": 768
}
```

**Response:**
```json
{
  "success": true,
  "sharedSecret": "hex-encoded-shared-secret"
}
```

#### POST /api/crypto/sign
Create quantum-resistant signature.

**Request Body:**
```json
{
  "privateKey": "base64-encoded-private-key",
  "message": "message-to-sign",
  "algorithm": "ml-dsa",
  "securityLevel": 65
}
```

**Response:**
```json
{
  "success": true,
  "signature": "base64-encoded-signature",
  "algorithm": "ML-DSA-65"
}
```

#### POST /api/crypto/verify
Verify quantum-resistant signature.

**Request Body:**
```json
{
  "publicKey": "base64-encoded-public-key",
  "message": "original-message",
  "signature": "base64-encoded-signature",
  "algorithm": "ml-dsa",
  "securityLevel": 65
}
```

**Response:**
```json
{
  "success": true,
  "isValid": true
}
```

#### POST /api/crypto/hybrid-exchange
Perform hybrid classical-quantum key exchange.

**Request Body:**
```json
{
  "mode": "x25519-mlkem768",
  "isInitiator": true
}
```

**Hybrid Modes:**
- `x25519-mlkem768`: X25519 + ML-KEM-768
- `secp256r1-mlkem768`: P-256 + ML-KEM-768
- `x448-mlkem1024`: X448 + ML-KEM-1024

**Response:**
```json
{
  "success": true,
  "mode": "x25519-mlkem768",
  "classical": {
    "publicKey": "pem-encoded-classical-key",
    "privateKey": "pem-encoded-classical-key"
  },
  "quantum": {
    "publicKey": "base64-encoded-quantum-key",
    "privateKey": "base64-encoded-quantum-key",
    "algorithm": "ML-KEM-768"
  }
}
```

### Authentication

#### POST /api/auth/register
Register new user with quantum credentials.

**Request Body:**
```json
{
  "username": "alice",
  "password": "SecurePassword123!",
  "email": "alice@example.com"
}
```

**Response:**
```json
{
  "success": true,
  "user": {
    "id": "uuid",
    "username": "alice",
    "quantumPublicKey": "base64-encoded-public-key"
  },
  "privateKey": "base64-encoded-private-key"
}
```

#### POST /api/auth/login
Initiate authentication with quantum challenge.

**Request Body:**
```json
{
  "username": "alice",
  "password": "SecurePassword123!"
}
```

**Response:**
```json
{
  "success": true,
  "challengeId": "uuid",
  "challenge": "base64-encoded-challenge",
  "requiresProof": true
}
```

#### POST /api/auth/challenge-response
Complete authentication by signing challenge.

**Request Body:**
```json
{
  "challengeId": "uuid",
  "signature": "base64-encoded-signature",
  "publicKey": "base64-encoded-public-key"
}
```

**Response:**
```json
{
  "success": true,
  "sessionId": "uuid",
  "accessToken": "quantum-jwt-token",
  "refreshToken": "quantum-refresh-token",
  "expiresIn": "24h",
  "tokenType": "QuantumBearer",
  "algorithm": "ml-dsa-65"
}
```

#### POST /api/auth/refresh
Refresh access token.

**Request Body:**
```json
{
  "refreshToken": "quantum-refresh-token"
}
```

**Response:**
```json
{
  "success": true,
  "accessToken": "new-quantum-jwt-token",
  "expiresIn": "24h"
}
```

#### POST /api/auth/logout
Logout and invalidate session.

**Request Body:**
```json
{
  "sessionId": "uuid"
}
```

**Response:**
```json
{
  "success": true
}
```

### Multi-Factor Authentication

#### POST /api/auth/mfa/setup
Setup quantum MFA device.

**Request Body:**
```json
{
  "userId": "user-id",
  "deviceName": "Alice's Phone"
}
```

**Response:**
```json
{
  "success": true,
  "deviceId": "uuid",
  "privateKey": "base64-encoded-device-key",
  "qrCode": "base64-encoded-qr-data"
}
```

#### POST /api/auth/mfa/verify
Verify MFA challenge.

**Request Body:**
```json
{
  "userId": "user-id",
  "deviceId": "device-id",
  "challenge": "mfa-challenge",
  "signature": "base64-encoded-signature"
}
```

**Response:**
```json
{
  "success": true,
  "verified": true
}
```

## Error Responses

All errors follow this format:

```json
{
  "success": false,
  "error": "Error message",
  "stack": "stack trace (development only)"
}
```

### Common Error Codes

- `400`: Bad Request - Invalid parameters
- `401`: Unauthorized - Invalid or missing authentication
- `403`: Forbidden - Insufficient permissions
- `404`: Not Found - Resource not found
- `429`: Too Many Requests - Rate limit exceeded
- `500`: Internal Server Error - Server error

## Rate Limiting

- General API: 100 requests per 15 minutes per IP
- Authentication endpoints: 5 requests per 15 minutes per IP
- Quantum operations: 1000 operations per second globally

## Security Headers

All responses include:

```
X-Frame-Options: SAMEORIGIN
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'
```

## WebSocket Support

Real-time security monitoring via WebSocket:

```javascript
const ws = new WebSocket('wss://localhost:8443/ws/security');

ws.on('message', (data) => {
  const event = JSON.parse(data);
  // Handle security events
});
```

## SDK Examples

### JavaScript/Node.js

```javascript
const QuantumClient = require('@quantum/client');

const client = new QuantumClient({
  baseURL: 'https://localhost:8443',
  algorithm: 'ml-dsa-65'
});

// Generate keys
const keys = await client.generateKeys('ml-kem', 768);

// Sign data
const signature = await client.sign(privateKey, 'message');

// Verify signature
const isValid = await client.verify(publicKey, 'message', signature);
```

### Python

```python
from quantum_client import QuantumClient

client = QuantumClient(
    base_url="https://localhost:8443",
    algorithm="ml-dsa-65"
)

# Generate keys
keys = client.generate_keys("ml-kem", 768)

# Encrypt data
result = client.encrypt(public_key, security_level=768)
```

## Migration Guide

### From RSA/ECDSA to Quantum-Resistant

1. **Enable Hybrid Mode**: Start with hybrid algorithms
2. **Generate Quantum Keys**: Create ML-DSA or SLH-DSA keys
3. **Update Clients**: Ensure quantum algorithm support
4. **Monitor Performance**: Track latency changes
5. **Phase Out Classical**: Gradually remove RSA/ECDSA

## Performance Benchmarks

Typical operation times on standard hardware:

| Operation | Time (ms) | Security Level |
|-----------|-----------|----------------|
| ML-KEM-768 keygen | 15-25 | Medium |
| ML-KEM-768 encapsulate | 8-12 | Medium |
| ML-DSA-65 sign | 20-30 | Medium |
| ML-DSA-65 verify | 10-15 | Medium |
| SLH-DSA-192 sign | 100-150 | High |
| SLH-DSA-192 verify | 50-80 | High |

## Compliance

This API implements:
- NIST FIPS 203 (ML-KEM)
- NIST FIPS 204 (ML-DSA)
- NIST FIPS 205 (SLH-DSA)

For detailed specifications, see: https://csrc.nist.gov/projects/post-quantum-cryptography