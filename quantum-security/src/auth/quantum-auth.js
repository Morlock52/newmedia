/**
 * Quantum-Resistant Authentication System
 * Implements post-quantum authentication using ML-DSA and SLH-DSA
 */

const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const QuantumCrypto = require('../crypto/quantum-crypto');

class QuantumAuth {
  constructor(options = {}) {
    this.quantumCrypto = new QuantumCrypto();
    this.tokenExpiry = options.tokenExpiry || '24h';
    this.refreshTokenExpiry = options.refreshTokenExpiry || '7d';
    this.signatureAlgorithm = options.signatureAlgorithm || 'ml-dsa-65';
    this.keyRotationInterval = options.keyRotationInterval || 24 * 60 * 60 * 1000; // 24 hours
    
    this.keys = new Map();
    this.sessions = new Map();
    this.challenges = new Map();
    
    this.initializeKeys();
  }

  /**
   * Initialize quantum signing keys
   */
  async initializeKeys() {
    await this.rotateKeys();
    
    // Set up key rotation schedule
    setInterval(async () => {
      await this.rotateKeys();
    }, this.keyRotationInterval);
  }

  /**
   * Rotate signing keys
   */
  async rotateKeys() {
    const keyId = crypto.randomUUID();
    
    let keypair;
    if (this.signatureAlgorithm.startsWith('ml-dsa')) {
      const level = parseInt(this.signatureAlgorithm.split('-')[2]);
      keypair = await this.quantumCrypto.mlDsaGenerateKeypair(level);
    } else if (this.signatureAlgorithm.startsWith('slh-dsa')) {
      const level = parseInt(this.signatureAlgorithm.split('-')[2]);
      keypair = await this.quantumCrypto.slhDsaGenerateKeypair(level);
    }

    this.keys.set(keyId, {
      ...keypair,
      createdAt: Date.now(),
      algorithm: this.signatureAlgorithm
    });

    // Keep last 3 keys for verification of older tokens
    const sortedKeys = Array.from(this.keys.entries())
      .sort((a, b) => b[1].createdAt - a[1].createdAt);
    
    if (sortedKeys.length > 3) {
      for (let i = 3; i < sortedKeys.length; i++) {
        this.keys.delete(sortedKeys[i][0]);
      }
    }

    this.currentKeyId = keyId;
    console.log(`Rotated quantum auth keys. New key ID: ${keyId}`);
  }

  /**
   * Register a new user with quantum credentials
   */
  async register(username, password, additionalData = {}) {
    // Generate user-specific quantum keypair
    const userKeypair = await this.quantumCrypto.mlDsaGenerateKeypair(65);
    
    // Hash password with quantum-resistant parameters
    const salt = crypto.randomBytes(32);
    const passwordHash = await this.quantumResistantHash(password, salt);
    
    const user = {
      id: crypto.randomUUID(),
      username,
      passwordHash: passwordHash.toString('hex'),
      salt: salt.toString('hex'),
      quantumPublicKey: userKeypair.publicKey,
      createdAt: Date.now(),
      ...additionalData
    };

    // Sign user data with system key
    const signature = await this.signData(user);
    user.signature = signature;

    return {
      user: {
        id: user.id,
        username: user.username,
        quantumPublicKey: user.quantumPublicKey
      },
      privateKey: userKeypair.privateKey
    };
  }

  /**
   * Authenticate user with quantum-resistant challenge-response
   */
  async authenticate(username, password, clientProof = null) {
    // Step 1: Verify password (simplified for demo)
    const user = await this.verifyPassword(username, password);
    if (!user) {
      throw new Error('Invalid credentials');
    }

    // Step 2: Create quantum challenge
    const challenge = crypto.randomBytes(32);
    const challengeId = crypto.randomUUID();
    
    this.challenges.set(challengeId, {
      challenge,
      userId: user.id,
      createdAt: Date.now(),
      attempts: 0
    });

    // Clean up old challenges
    this.cleanupChallenges();

    return {
      challengeId,
      challenge: challenge.toString('base64'),
      requiresProof: true
    };
  }

  /**
   * Verify quantum challenge response
   */
  async verifyChallengeResponse(challengeId, signature, userPublicKey) {
    const challengeData = this.challenges.get(challengeId);
    if (!challengeData) {
      throw new Error('Invalid or expired challenge');
    }

    // Increment attempts
    challengeData.attempts++;
    if (challengeData.attempts > 3) {
      this.challenges.delete(challengeId);
      throw new Error('Too many attempts');
    }

    // Verify signature
    const isValid = await this.quantumCrypto.mlDsaVerify(
      userPublicKey,
      challengeData.challenge,
      signature,
      65
    );

    if (!isValid.isValid) {
      throw new Error('Invalid challenge response');
    }

    // Challenge verified, create session
    this.challenges.delete(challengeId);
    
    const sessionId = crypto.randomUUID();
    const session = {
      userId: challengeData.userId,
      sessionId,
      createdAt: Date.now(),
      lastActivity: Date.now()
    };

    this.sessions.set(sessionId, session);

    // Generate quantum-signed tokens
    const tokens = await this.generateTokens(session);

    return {
      sessionId,
      ...tokens
    };
  }

  /**
   * Generate quantum-signed JWT tokens
   */
  async generateTokens(session) {
    const currentKey = this.keys.get(this.currentKeyId);
    
    // Create token payload
    const payload = {
      userId: session.userId,
      sessionId: session.sessionId,
      iat: Math.floor(Date.now() / 1000),
      keyId: this.currentKeyId
    };

    // Sign payload with quantum signature
    const signature = await this.signData(payload);
    
    // Create custom quantum JWT
    const quantumToken = this.createQuantumJWT(payload, signature);
    
    // Also create refresh token
    const refreshPayload = {
      ...payload,
      type: 'refresh'
    };
    
    const refreshSignature = await this.signData(refreshPayload);
    const refreshToken = this.createQuantumJWT(refreshPayload, refreshSignature);

    return {
      accessToken: quantumToken,
      refreshToken: refreshToken,
      expiresIn: this.tokenExpiry,
      tokenType: 'QuantumBearer',
      algorithm: this.signatureAlgorithm
    };
  }

  /**
   * Verify quantum-signed token
   */
  async verifyToken(token) {
    try {
      // Parse quantum JWT
      const { header, payload, signature } = this.parseQuantumJWT(token);
      
      // Get the key used to sign
      const signingKey = this.keys.get(payload.keyId);
      if (!signingKey) {
        throw new Error('Unknown signing key');
      }

      // Verify quantum signature
      const dataToVerify = Buffer.concat([
        Buffer.from(JSON.stringify(header)),
        Buffer.from('.'),
        Buffer.from(JSON.stringify(payload))
      ]);

      let isValid;
      if (signingKey.algorithm.startsWith('ml-dsa')) {
        const level = parseInt(signingKey.algorithm.split('-')[2]);
        isValid = await this.quantumCrypto.mlDsaVerify(
          signingKey.publicKey,
          dataToVerify,
          signature,
          level
        );
      } else if (signingKey.algorithm.startsWith('slh-dsa')) {
        const level = parseInt(signingKey.algorithm.split('-')[2]);
        isValid = await this.quantumCrypto.slhDsaVerify(
          signingKey.publicKey,
          dataToVerify,
          signature,
          level
        );
      }

      if (!isValid.isValid) {
        throw new Error('Invalid token signature');
      }

      // Check expiration
      const now = Math.floor(Date.now() / 1000);
      const expiry = payload.iat + this.parseExpiry(this.tokenExpiry);
      
      if (now > expiry) {
        throw new Error('Token expired');
      }

      // Verify session is still active
      const session = this.sessions.get(payload.sessionId);
      if (!session) {
        throw new Error('Session not found');
      }

      // Update last activity
      session.lastActivity = Date.now();

      return {
        valid: true,
        payload,
        session
      };
    } catch (error) {
      return {
        valid: false,
        error: error.message
      };
    }
  }

  /**
   * Refresh access token
   */
  async refreshAccessToken(refreshToken) {
    const verification = await this.verifyToken(refreshToken);
    
    if (!verification.valid) {
      throw new Error('Invalid refresh token');
    }

    if (verification.payload.type !== 'refresh') {
      throw new Error('Not a refresh token');
    }

    // Generate new access token
    const session = verification.session;
    const tokens = await this.generateTokens(session);

    return {
      accessToken: tokens.accessToken,
      expiresIn: tokens.expiresIn
    };
  }

  /**
   * Logout and invalidate session
   */
  async logout(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      this.sessions.delete(sessionId);
      return { success: true };
    }
    return { success: false, error: 'Session not found' };
  }

  /**
   * Multi-factor authentication with quantum signatures
   */
  async setupMFA(userId, deviceName) {
    // Generate device-specific quantum keypair
    const deviceKeypair = await this.quantumCrypto.mlDsaGenerateKeypair(44); // Lower security for faster signing
    
    const device = {
      id: crypto.randomUUID(),
      userId,
      deviceName,
      publicKey: deviceKeypair.publicKey,
      createdAt: Date.now(),
      lastUsed: null
    };

    // Sign device registration
    const signature = await this.signData(device);
    device.signature = signature;

    return {
      deviceId: device.id,
      privateKey: deviceKeypair.privateKey,
      qrCode: await this.generateMFAQRCode(device)
    };
  }

  /**
   * Verify MFA quantum signature
   */
  async verifyMFA(userId, deviceId, challenge, signature) {
    // Lookup device
    const device = await this.getDevice(deviceId);
    
    if (!device || device.userId !== userId) {
      throw new Error('Invalid device');
    }

    // Verify signature
    const isValid = await this.quantumCrypto.mlDsaVerify(
      device.publicKey,
      Buffer.from(challenge),
      signature,
      44
    );

    if (!isValid.isValid) {
      throw new Error('Invalid MFA signature');
    }

    // Update last used
    device.lastUsed = Date.now();

    return { verified: true };
  }

  /**
   * Helper methods
   */

  async quantumResistantHash(password, salt) {
    // Use Argon2 parameters suitable for quantum resistance
    const iterations = 10;
    const memory = 1024 * 1024; // 1GB
    const parallelism = 4;
    const keyLength = 64;

    // Simulate Argon2id (in production, use actual Argon2 library)
    let derived = Buffer.concat([Buffer.from(password), salt]);
    
    for (let i = 0; i < iterations; i++) {
      derived = crypto.createHash('sha512').update(derived).digest();
    }

    return derived;
  }

  async signData(data) {
    const currentKey = this.keys.get(this.currentKeyId);
    const dataBuffer = Buffer.from(JSON.stringify(data));
    
    if (currentKey.algorithm.startsWith('ml-dsa')) {
      const level = parseInt(currentKey.algorithm.split('-')[2]);
      const result = await this.quantumCrypto.mlDsaSign(
        currentKey.privateKey,
        dataBuffer,
        level
      );
      return result.signature;
    } else if (currentKey.algorithm.startsWith('slh-dsa')) {
      const level = parseInt(currentKey.algorithm.split('-')[2]);
      const result = await this.quantumCrypto.slhDsaSign(
        currentKey.privateKey,
        dataBuffer,
        level
      );
      return result.signature;
    }
  }

  createQuantumJWT(payload, signature) {
    const header = {
      alg: this.signatureAlgorithm,
      typ: 'QJWT',
      kid: payload.keyId
    };

    const encodedHeader = Buffer.from(JSON.stringify(header)).toString('base64url');
    const encodedPayload = Buffer.from(JSON.stringify(payload)).toString('base64url');
    const encodedSignature = Buffer.from(signature, 'base64').toString('base64url');

    return `${encodedHeader}.${encodedPayload}.${encodedSignature}`;
  }

  parseQuantumJWT(token) {
    const parts = token.split('.');
    if (parts.length !== 3) {
      throw new Error('Invalid token format');
    }

    const header = JSON.parse(Buffer.from(parts[0], 'base64url').toString());
    const payload = JSON.parse(Buffer.from(parts[1], 'base64url').toString());
    const signature = Buffer.from(parts[2], 'base64url').toString('base64');

    return { header, payload, signature };
  }

  parseExpiry(expiry) {
    const match = expiry.match(/^(\d+)([smhd])$/);
    if (!match) return 3600; // Default 1 hour

    const value = parseInt(match[1]);
    const unit = match[2];

    switch (unit) {
      case 's': return value;
      case 'm': return value * 60;
      case 'h': return value * 3600;
      case 'd': return value * 86400;
      default: return 3600;
    }
  }

  cleanupChallenges() {
    const now = Date.now();
    const timeout = 5 * 60 * 1000; // 5 minutes

    for (const [id, challenge] of this.challenges.entries()) {
      if (now - challenge.createdAt > timeout) {
        this.challenges.delete(id);
      }
    }
  }

  async verifyPassword(username, password) {
    // Simplified for demo - in production, fetch from database
    return {
      id: 'user-' + username,
      username
    };
  }

  async getDevice(deviceId) {
    // Simplified for demo - in production, fetch from database
    return {
      id: deviceId,
      userId: 'demo-user',
      publicKey: 'demo-public-key'
    };
  }

  async generateMFAQRCode(device) {
    // Generate QR code data for MFA setup
    const data = {
      type: 'quantum-mfa',
      deviceId: device.id,
      publicKey: device.publicKey,
      algorithm: 'ml-dsa-44',
      issuer: 'Quantum Security System'
    };

    return Buffer.from(JSON.stringify(data)).toString('base64');
  }
}

module.exports = QuantumAuth;