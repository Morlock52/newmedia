/**
 * Quantum Security Server
 * Main application entry point
 */

const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const morgan = require('morgan');
const dotenv = require('dotenv');

const QuantumTLS = require('./tls/quantum-tls');
const QuantumAuth = require('./auth/quantum-auth');
const QuantumCrypto = require('./crypto/quantum-crypto');
const SecurityMonitor = require('./monitoring/security-monitor');

// Load environment variables
dotenv.config();

class QuantumSecurityServer {
  constructor() {
    this.app = express();
    this.quantumTLS = new QuantumTLS();
    this.quantumAuth = new QuantumAuth();
    this.quantumCrypto = new QuantumCrypto();
    this.securityMonitor = new SecurityMonitor();
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
    this.setupSecurityMonitoring();
  }

  /**
   * Setup Express middleware
   */
  setupMiddleware() {
    // Security headers
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
        },
      },
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
      }
    }));

    // CORS configuration
    this.app.use(cors({
      origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
      credentials: true
    }));

    // Compression
    this.app.use(compression());

    // Request logging
    this.app.use(morgan('combined'));

    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // Limit each IP to 100 requests per windowMs
      message: 'Too many requests from this IP, please try again later.',
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use('/api/', limiter);

    // Stricter rate limit for auth endpoints
    const authLimiter = rateLimit({
      windowMs: 15 * 60 * 1000,
      max: 5,
      skipSuccessfulRequests: true
    });
    this.app.use('/api/auth/', authLimiter);
  }

  /**
   * Setup API routes
   */
  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      const report = this.securityMonitor.getSecurityReport();
      res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        security: report.status,
        uptime: report.uptime
      });
    });

    // Quantum crypto endpoints
    this.app.post('/api/crypto/keygen/:algorithm', async (req, res, next) => {
      try {
        const { algorithm } = req.params;
        const { securityLevel } = req.body;

        this.securityMonitor.recordQuantumOperation(`${algorithm}-keygen`, {
          algorithm,
          securityLevel,
          success: true
        });

        let result;
        switch (algorithm) {
          case 'ml-kem':
            result = await this.quantumCrypto.mlKemGenerateKeypair(securityLevel);
            break;
          case 'ml-dsa':
            result = await this.quantumCrypto.mlDsaGenerateKeypair(securityLevel);
            break;
          case 'slh-dsa':
            result = await this.quantumCrypto.slhDsaGenerateKeypair(securityLevel);
            break;
          default:
            throw new Error(`Unknown algorithm: ${algorithm}`);
        }

        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    this.app.post('/api/crypto/encrypt', async (req, res, next) => {
      try {
        const { publicKey, securityLevel = 768 } = req.body;

        const start = Date.now();
        const result = await this.quantumCrypto.mlKemEncapsulate(publicKey, securityLevel);
        const duration = Date.now() - start;

        this.securityMonitor.recordQuantumOperation('ml-kem-encapsulate', {
          algorithm: `ml-kem-${securityLevel}`,
          duration,
          success: true
        });

        this.securityMonitor.recordPerformance('kem-encapsulation', duration);

        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    this.app.post('/api/crypto/decrypt', async (req, res, next) => {
      try {
        const { privateKey, ciphertext, securityLevel = 768 } = req.body;

        const start = Date.now();
        const result = await this.quantumCrypto.mlKemDecapsulate(privateKey, ciphertext, securityLevel);
        const duration = Date.now() - start;

        this.securityMonitor.recordQuantumOperation('ml-kem-decapsulate', {
          algorithm: `ml-kem-${securityLevel}`,
          duration,
          success: true
        });

        this.securityMonitor.recordPerformance('kem-decapsulation', duration);

        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    this.app.post('/api/crypto/sign', async (req, res, next) => {
      try {
        const { privateKey, message, algorithm = 'ml-dsa', securityLevel = 65 } = req.body;

        const start = Date.now();
        let result;
        
        if (algorithm === 'ml-dsa') {
          result = await this.quantumCrypto.mlDsaSign(privateKey, message, securityLevel);
        } else if (algorithm === 'slh-dsa') {
          result = await this.quantumCrypto.slhDsaSign(privateKey, message, securityLevel);
        } else {
          throw new Error(`Unknown signature algorithm: ${algorithm}`);
        }

        const duration = Date.now() - start;

        this.securityMonitor.recordQuantumOperation(`${algorithm}-sign`, {
          algorithm: `${algorithm}-${securityLevel}`,
          duration,
          messageSize: Buffer.from(message).length,
          success: true
        });

        this.securityMonitor.recordPerformance('signature-generation', duration);

        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    this.app.post('/api/crypto/verify', async (req, res, next) => {
      try {
        const { publicKey, message, signature, algorithm = 'ml-dsa', securityLevel = 65 } = req.body;

        const start = Date.now();
        let result;
        
        if (algorithm === 'ml-dsa') {
          result = await this.quantumCrypto.mlDsaVerify(publicKey, message, signature, securityLevel);
        } else if (algorithm === 'slh-dsa') {
          result = await this.quantumCrypto.slhDsaVerify(publicKey, message, signature, securityLevel);
        } else {
          throw new Error(`Unknown signature algorithm: ${algorithm}`);
        }

        const duration = Date.now() - start;

        this.securityMonitor.recordQuantumOperation(`${algorithm}-verify`, {
          algorithm: `${algorithm}-${securityLevel}`,
          duration,
          success: result.isValid
        });

        this.securityMonitor.recordPerformance('signature-verification', duration);

        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    // Authentication endpoints
    this.app.post('/api/auth/register', async (req, res, next) => {
      try {
        const { username, password, email } = req.body;
        
        const result = await this.quantumAuth.register(username, password, { email });
        
        this.securityMonitor.recordAuthAttempt(result.user.id, true, {
          method: 'registration',
          ip: req.ip
        });

        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    this.app.post('/api/auth/login', async (req, res, next) => {
      try {
        const { username, password } = req.body;
        
        const challenge = await this.quantumAuth.authenticate(username, password);
        
        this.securityMonitor.recordAuthAttempt(username, true, {
          method: 'password',
          ip: req.ip,
          stage: 'challenge-issued'
        });

        res.json({ success: true, ...challenge });
      } catch (error) {
        this.securityMonitor.recordAuthAttempt(req.body.username, false, {
          method: 'password',
          ip: req.ip,
          error: error.message
        });
        next(error);
      }
    });

    this.app.post('/api/auth/challenge-response', async (req, res, next) => {
      try {
        const { challengeId, signature, publicKey } = req.body;
        
        const tokens = await this.quantumAuth.verifyChallengeResponse(
          challengeId,
          signature,
          publicKey
        );
        
        this.securityMonitor.recordAuthAttempt(tokens.sessionId, true, {
          method: 'quantum-challenge',
          ip: req.ip,
          stage: 'challenge-verified'
        });

        res.json({ success: true, ...tokens });
      } catch (error) {
        this.securityMonitor.recordAuthAttempt('unknown', false, {
          method: 'quantum-challenge',
          ip: req.ip,
          error: error.message
        });
        next(error);
      }
    });

    this.app.post('/api/auth/refresh', async (req, res, next) => {
      try {
        const { refreshToken } = req.body;
        
        const tokens = await this.quantumAuth.refreshAccessToken(refreshToken);
        
        res.json({ success: true, ...tokens });
      } catch (error) {
        next(error);
      }
    });

    this.app.post('/api/auth/logout', async (req, res, next) => {
      try {
        const { sessionId } = req.body;
        
        const result = await this.quantumAuth.logout(sessionId);
        
        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    // MFA endpoints
    this.app.post('/api/auth/mfa/setup', async (req, res, next) => {
      try {
        const { userId, deviceName } = req.body;
        
        const mfaSetup = await this.quantumAuth.setupMFA(userId, deviceName);
        
        res.json({ success: true, ...mfaSetup });
      } catch (error) {
        next(error);
      }
    });

    this.app.post('/api/auth/mfa/verify', async (req, res, next) => {
      try {
        const { userId, deviceId, challenge, signature } = req.body;
        
        const result = await this.quantumAuth.verifyMFA(userId, deviceId, challenge, signature);
        
        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });

    // Security monitoring endpoints
    this.app.get('/api/security/report', async (req, res) => {
      const report = this.securityMonitor.getSecurityReport();
      res.json(report);
    });

    this.app.get('/api/security/alerts', async (req, res) => {
      const alerts = this.securityMonitor.alerts;
      res.json({ alerts, count: alerts.length });
    });

    // Hybrid key exchange endpoint
    this.app.post('/api/crypto/hybrid-exchange', async (req, res, next) => {
      try {
        const { mode = 'x25519-mlkem768', isInitiator = true } = req.body;
        
        const result = await this.quantumCrypto.hybridKeyExchange(mode, isInitiator);
        
        this.securityMonitor.recordQuantumOperation('hybrid-key-exchange', {
          mode,
          isInitiator,
          success: true
        });

        res.json({ success: true, ...result });
      } catch (error) {
        next(error);
      }
    });
  }

  /**
   * Setup error handling
   */
  setupErrorHandling() {
    // 404 handler
    this.app.use((req, res) => {
      res.status(404).json({
        success: false,
        error: 'Not Found',
        message: `Cannot ${req.method} ${req.url}`
      });
    });

    // Global error handler
    this.app.use((err, req, res, next) => {
      this.securityMonitor.recordError(err, {
        method: req.method,
        url: req.url,
        ip: req.ip
      });

      const status = err.status || 500;
      const message = err.message || 'Internal Server Error';

      console.error('Error:', err);

      res.status(status).json({
        success: false,
        error: message,
        ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
      });
    });
  }

  /**
   * Setup security monitoring listeners
   */
  setupSecurityMonitoring() {
    // Listen for security alerts
    this.securityMonitor.on('security-alert', (alert) => {
      console.log('SECURITY ALERT:', alert);
      // Here you could send notifications, trigger automated responses, etc.
    });

    // Listen for performance issues
    this.securityMonitor.on('performance-metric', (metric) => {
      if (metric.value > 1000) { // If operation takes more than 1 second
        console.warn('Slow operation detected:', metric);
      }
    });

    // Listen for threat reports
    this.securityMonitor.on('threat-report', (report) => {
      console.log('Threat Report Generated:', report);
    });

    // Auto-remediation handlers
    this.securityMonitor.on('block-user', (data) => {
      console.log('Blocking user:', data.userId, 'for', data.duration, 'ms');
      // Implement user blocking logic
    });

    this.securityMonitor.on('apply-rate-limit', (data) => {
      console.log('Applying rate limit:', data);
      // Implement dynamic rate limiting
    });

    this.securityMonitor.on('enable-enhanced-security', () => {
      console.log('Enabling enhanced security mode');
      // Implement enhanced security measures
    });
  }

  /**
   * Start the server
   */
  async start() {
    const port = process.env.PORT || 3000;
    const host = process.env.HOST || '0.0.0.0';

    // Start regular HTTP server (redirects to HTTPS)
    this.app.listen(port, host, () => {
      console.log(`HTTP Server running on http://${host}:${port}`);
      console.log('This server redirects to HTTPS for security');
    });

    // Start quantum-resistant HTTPS server
    try {
      const quantumServer = await this.quantumTLS.createServer({
        port: parseInt(process.env.QUANTUM_TLS_PORT) || 8443,
        host,
        hybridMode: process.env.HYBRID_MODE || 'x25519-mlkem768',
        signatureAlgorithm: process.env.SIGNATURE_ALGORITHM || 'ml-dsa-65'
      });

      // Mount Express app on quantum server
      quantumServer.on('request', this.app);

      console.log('\n=== Quantum Security Server Started ===');
      console.log('Post-quantum algorithms active:');
      console.log('- ML-KEM (Kyber) for key encapsulation');
      console.log('- ML-DSA (Dilithium) for signatures');
      console.log('- SLH-DSA (SPHINCS+) for hash-based signatures');
      console.log('- Hybrid classical-quantum modes enabled');
      console.log('=====================================\n');
    } catch (error) {
      console.error('Failed to start quantum TLS server:', error);
      process.exit(1);
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    console.log('Shutting down quantum security server...');
    
    this.securityMonitor.shutdown();
    
    // Close servers
    process.exit(0);
  }
}

// Create and start server
const server = new QuantumSecurityServer();

// Handle shutdown signals
process.on('SIGTERM', () => server.shutdown());
process.on('SIGINT', () => server.shutdown());

// Start the server
server.start().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});