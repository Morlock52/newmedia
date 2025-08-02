/**
 * Integration tests for quantum security system
 */

const request = require('supertest');
const QuantumCrypto = require('../../src/crypto/quantum-crypto');

const BASE_URL = process.env.TEST_URL || 'https://localhost:8443';

// Disable certificate validation for testing
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

describe('Quantum Security Integration Tests', () => {
  let api;
  let quantumCrypto;
  
  beforeAll(() => {
    api = request(BASE_URL);
    quantumCrypto = new QuantumCrypto();
  });

  describe('Health Check', () => {
    test('should return healthy status', async () => {
      const response = await api.get('/health');
      
      expect(response.status).toBe(200);
      expect(response.body.status).toBe('ok');
      expect(response.body.security).toBeDefined();
    });
  });

  describe('ML-KEM (Kyber) Operations', () => {
    let publicKey, privateKey;

    test('should generate ML-KEM-768 keypair', async () => {
      const response = await api
        .post('/api/crypto/keygen/ml-kem')
        .send({ securityLevel: 768 });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.publicKey).toBeDefined();
      expect(response.body.privateKey).toBeDefined();
      expect(response.body.algorithm).toBe('ML-KEM-768');
      
      publicKey = response.body.publicKey;
      privateKey = response.body.privateKey;
    });

    test('should encrypt with ML-KEM public key', async () => {
      const response = await api
        .post('/api/crypto/encrypt')
        .send({ publicKey, securityLevel: 768 });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.ciphertext).toBeDefined();
      expect(response.body.sharedSecret).toBeDefined();
    });

    test('should handle invalid ML-KEM parameters', async () => {
      const response = await api
        .post('/api/crypto/keygen/ml-kem')
        .send({ securityLevel: 999 }); // Invalid level
      
      expect(response.status).toBe(500);
      expect(response.body.success).toBe(false);
    });
  });

  describe('ML-DSA (Dilithium) Operations', () => {
    let publicKey, privateKey;
    const testMessage = 'Test message for quantum signatures';

    test('should generate ML-DSA-65 keypair', async () => {
      const response = await api
        .post('/api/crypto/keygen/ml-dsa')
        .send({ securityLevel: 65 });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.algorithm).toBe('ML-DSA-65');
      
      publicKey = response.body.publicKey;
      privateKey = response.body.privateKey;
    });

    test('should sign message with ML-DSA', async () => {
      const response = await api
        .post('/api/crypto/sign')
        .send({
          privateKey,
          message: testMessage,
          algorithm: 'ml-dsa',
          securityLevel: 65
        });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.signature).toBeDefined();
    });

    test('should verify ML-DSA signature', async () => {
      // First sign
      const signResponse = await api
        .post('/api/crypto/sign')
        .send({
          privateKey,
          message: testMessage,
          algorithm: 'ml-dsa',
          securityLevel: 65
        });
      
      const signature = signResponse.body.signature;
      
      // Then verify
      const verifyResponse = await api
        .post('/api/crypto/verify')
        .send({
          publicKey,
          message: testMessage,
          signature,
          algorithm: 'ml-dsa',
          securityLevel: 65
        });
      
      expect(verifyResponse.status).toBe(200);
      expect(verifyResponse.body.success).toBe(true);
      expect(verifyResponse.body.isValid).toBe(true);
    });
  });

  describe('SLH-DSA (SPHINCS+) Operations', () => {
    let publicKey, privateKey;

    test('should generate SLH-DSA-192 keypair', async () => {
      const response = await api
        .post('/api/crypto/keygen/slh-dsa')
        .send({ securityLevel: 192 });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.algorithm).toBe('SLH-DSA-192');
      
      publicKey = response.body.publicKey;
      privateKey = response.body.privateKey;
    });

    test('should sign and verify with SLH-DSA', async () => {
      const message = 'Hash-based signature test';
      
      // Sign
      const signResponse = await api
        .post('/api/crypto/sign')
        .send({
          privateKey,
          message,
          algorithm: 'slh-dsa',
          securityLevel: 192
        });
      
      expect(signResponse.status).toBe(200);
      const signature = signResponse.body.signature;
      
      // Verify
      const verifyResponse = await api
        .post('/api/crypto/verify')
        .send({
          publicKey,
          message,
          signature,
          algorithm: 'slh-dsa',
          securityLevel: 192
        });
      
      expect(verifyResponse.status).toBe(200);
      expect(verifyResponse.body.isValid).toBe(true);
    });
  });

  describe('Hybrid Cryptography', () => {
    test('should perform hybrid key exchange', async () => {
      const response = await api
        .post('/api/crypto/hybrid-exchange')
        .send({
          mode: 'x25519-mlkem768',
          isInitiator: true
        });
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.mode).toBe('x25519-mlkem768');
      expect(response.body.classical).toBeDefined();
      expect(response.body.quantum).toBeDefined();
    });

    test('should support multiple hybrid modes', async () => {
      const modes = ['x25519-mlkem768', 'secp256r1-mlkem768', 'x448-mlkem1024'];
      
      for (const mode of modes) {
        const response = await api
          .post('/api/crypto/hybrid-exchange')
          .send({ mode, isInitiator: true });
        
        expect(response.status).toBe(200);
        expect(response.body.mode).toBe(mode);
      }
    });
  });

  describe('Authentication System', () => {
    const testUser = {
      username: `testuser_${Date.now()}`,
      password: 'SecurePassword123!',
      email: 'test@quantum.local'
    };
    let userPrivateKey;

    test('should register new user with quantum credentials', async () => {
      const response = await api
        .post('/api/auth/register')
        .send(testUser);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.user.id).toBeDefined();
      expect(response.body.user.quantumPublicKey).toBeDefined();
      expect(response.body.privateKey).toBeDefined();
      
      userPrivateKey = response.body.privateKey;
    });

    test('should authenticate with quantum challenge', async () => {
      // Step 1: Initial authentication
      const authResponse = await api
        .post('/api/auth/login')
        .send({
          username: testUser.username,
          password: testUser.password
        });
      
      expect(authResponse.status).toBe(200);
      expect(authResponse.body.challengeId).toBeDefined();
      expect(authResponse.body.challenge).toBeDefined();
      expect(authResponse.body.requiresProof).toBe(true);
      
      // Note: In a real test, we would sign the challenge with the user's private key
      // and submit the signature for verification
    });
  });

  describe('Security Monitoring', () => {
    test('should provide security report', async () => {
      const response = await api.get('/api/security/report');
      
      expect(response.status).toBe(200);
      expect(response.body.status).toBeDefined();
      expect(response.body.metrics).toBeDefined();
      expect(response.body.activeAlerts).toBeDefined();
    });

    test('should track quantum operations', async () => {
      // Perform some operations
      await api.post('/api/crypto/keygen/ml-kem').send({ securityLevel: 512 });
      await api.post('/api/crypto/keygen/ml-dsa').send({ securityLevel: 44 });
      
      // Check report
      const response = await api.get('/api/security/report');
      
      expect(response.body.metrics.totalOperations).toBeGreaterThan(0);
    });
  });

  describe('Performance Tests', () => {
    test('ML-KEM operations should complete within acceptable time', async () => {
      const iterations = 10;
      const startTime = Date.now();
      
      for (let i = 0; i < iterations; i++) {
        await api
          .post('/api/crypto/keygen/ml-kem')
          .send({ securityLevel: 768 });
      }
      
      const duration = Date.now() - startTime;
      const avgTime = duration / iterations;
      
      expect(avgTime).toBeLessThan(1000); // Less than 1 second per operation
    });

    test('ML-DSA signing should be performant', async () => {
      // Generate keypair
      const keygenResponse = await api
        .post('/api/crypto/keygen/ml-dsa')
        .send({ securityLevel: 44 }); // Lowest level for speed
      
      const { privateKey } = keygenResponse.body;
      const message = 'Performance test message';
      
      const startTime = Date.now();
      const iterations = 20;
      
      for (let i = 0; i < iterations; i++) {
        await api
          .post('/api/crypto/sign')
          .send({
            privateKey,
            message,
            algorithm: 'ml-dsa',
            securityLevel: 44
          });
      }
      
      const duration = Date.now() - startTime;
      const avgTime = duration / iterations;
      
      expect(avgTime).toBeLessThan(500); // Less than 500ms per signature
    });
  });

  describe('Error Handling', () => {
    test('should handle invalid algorithm gracefully', async () => {
      const response = await api
        .post('/api/crypto/keygen/invalid-algorithm')
        .send({});
      
      expect(response.status).toBe(500);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });

    test('should enforce rate limiting on auth endpoints', async () => {
      // Make multiple rapid requests
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(
          api.post('/api/auth/login').send({
            username: 'test',
            password: 'test'
          })
        );
      }
      
      const responses = await Promise.all(promises);
      const rateLimited = responses.some(r => r.status === 429);
      
      expect(rateLimited).toBe(true);
    });
  });
});

// Performance benchmarks
describe('Quantum Algorithm Benchmarks', () => {
  const quantumCrypto = new QuantumCrypto();

  test('benchmark ML-KEM key generation', async () => {
    const levels = [512, 768, 1024];
    const results = {};
    
    for (const level of levels) {
      const startTime = Date.now();
      const iterations = 50;
      
      for (let i = 0; i < iterations; i++) {
        await quantumCrypto.mlKemGenerateKeypair(level);
      }
      
      const duration = Date.now() - startTime;
      results[`ML-KEM-${level}`] = duration / iterations;
    }
    
    console.log('ML-KEM Key Generation Benchmarks:', results);
    
    // Verify performance scales appropriately
    expect(results['ML-KEM-512']).toBeLessThan(results['ML-KEM-768']);
    expect(results['ML-KEM-768']).toBeLessThan(results['ML-KEM-1024']);
  });

  test('benchmark signature algorithms', async () => {
    const message = 'Benchmark test message';
    const results = {};
    
    // ML-DSA benchmarks
    const mldsaLevels = [44, 65, 87];
    for (const level of mldsaLevels) {
      const keypair = await quantumCrypto.mlDsaGenerateKeypair(level);
      const startTime = Date.now();
      const iterations = 30;
      
      for (let i = 0; i < iterations; i++) {
        await quantumCrypto.mlDsaSign(keypair.privateKey, message, level);
      }
      
      const duration = Date.now() - startTime;
      results[`ML-DSA-${level}`] = duration / iterations;
    }
    
    // SLH-DSA benchmark (slower, fewer iterations)
    const slhdsaKeypair = await quantumCrypto.slhDsaGenerateKeypair(128);
    const startTime = Date.now();
    const iterations = 10;
    
    for (let i = 0; i < iterations; i++) {
      await quantumCrypto.slhDsaSign(slhdsaKeypair.privateKey, message, 128);
    }
    
    const duration = Date.now() - startTime;
    results['SLH-DSA-128'] = duration / iterations;
    
    console.log('Signature Algorithm Benchmarks:', results);
    
    // ML-DSA should be faster than SLH-DSA
    expect(results['ML-DSA-65']).toBeLessThan(results['SLH-DSA-128']);
  });
});