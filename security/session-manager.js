/**
 * Secure Session Management System
 * Fixes: Weak session management, JWT tokens without proper rotation
 * Author: Security Manager Agent
 * Date: 2025-08-03
 */

const crypto = require('crypto');
const Redis = require('redis');
const jwt = require('jsonwebtoken');

class SecureSessionManager {
  constructor(options = {}) {
    this.redisClient = options.redisClient;
    this.jwtSecret = options.jwtSecret || process.env.JWT_SECRET;
    this.refreshSecret = options.refreshSecret || process.env.REFRESH_SECRET;
    this.sessionTimeout = options.sessionTimeout || 15 * 60 * 1000; // 15 minutes
    this.refreshTimeout = options.refreshTimeout || 7 * 24 * 60 * 60 * 1000; // 7 days
    this.maxSessions = options.maxSessions || 5; // Max concurrent sessions per user
    this.rotationInterval = options.rotationInterval || 5 * 60 * 1000; // 5 minutes
    
    // In-memory fallback if Redis not available
    this.sessionStore = new Map();
    this.userSessions = new Map();
    
    this.initializeCleanup();
  }

  /**
   * Initialize Redis connection
   */
  async initializeRedis(redisUrl) {
    if (!this.redisClient) {
      this.redisClient = Redis.createClient({
        url: redisUrl,
        socket: {
          connectTimeout: 5000,
          lazyConnect: true
        },
        password: process.env.REDIS_PASSWORD
      });

      this.redisClient.on('error', (err) => {
        console.error('Redis connection error:', err);
        // Fall back to in-memory storage
        this.redisClient = null;
      });

      try {
        await this.redisClient.connect();
        console.log('✅ Redis connected for session management');
      } catch (error) {
        console.warn('⚠️ Redis connection failed, using in-memory sessions');
        this.redisClient = null;
      }
    }
  }

  /**
   * Generate cryptographically secure session ID
   */
  generateSessionId() {
    return crypto.randomBytes(32).toString('hex');
  }

  /**
   * Generate fingerprint for device/browser identification
   */
  generateFingerprint(req) {
    const components = [
      req.headers['user-agent'] || '',
      req.headers['accept-language'] || '',
      req.headers['accept-encoding'] || '',
      req.connection.remoteAddress || req.ip || '',
      req.headers['x-forwarded-for'] || ''
    ];
    
    return crypto
      .createHash('sha256')
      .update(components.join('|'))
      .digest('hex');
  }

  /**
   * Create new secure session
   */
  async createSession(userId, userAgent, ipAddress, additionalData = {}) {
    const sessionId = this.generateSessionId();
    const now = Date.now();
    const expiresAt = now + this.sessionTimeout;
    const refreshExpiresAt = now + this.refreshTimeout;

    const sessionData = {
      sessionId,
      userId,
      userAgent: userAgent?.substring(0, 500) || 'unknown', // Limit length
      ipAddress,
      createdAt: now,
      lastActivity: now,
      expiresAt,
      refreshExpiresAt,
      rotationNeeded: false,
      rotationCount: 0,
      fingerprint: this.generateFingerprint({ headers: { 'user-agent': userAgent }, ip: ipAddress }),
      ...additionalData
    };

    // Check and enforce max sessions per user
    await this.enforceMaxSessions(userId);

    // Store session
    await this.storeSession(sessionId, sessionData);
    await this.addUserSession(userId, sessionId);

    // Generate JWT tokens
    const tokens = this.generateTokenPair(sessionId, userId, sessionData);

    return {
      sessionId,
      accessToken: tokens.accessToken,
      refreshToken: tokens.refreshToken,
      expiresAt,
      refreshExpiresAt
    };
  }

  /**
   * Generate JWT token pair
   */
  generateTokenPair(sessionId, userId, sessionData) {
    const accessTokenPayload = {
      sessionId,
      userId,
      type: 'access',
      iat: Math.floor(Date.now() / 1000),
      fingerprint: sessionData.fingerprint
    };

    const refreshTokenPayload = {
      sessionId,
      userId,
      type: 'refresh',
      iat: Math.floor(Date.now() / 1000),
      fingerprint: sessionData.fingerprint
    };

    const accessToken = jwt.sign(accessTokenPayload, this.jwtSecret, {
      expiresIn: '15m',
      issuer: 'media-server',
      audience: 'media-server-api'
    });

    const refreshToken = jwt.sign(refreshTokenPayload, this.refreshSecret, {
      expiresIn: '7d',
      issuer: 'media-server',
      audience: 'media-server-refresh'
    });

    return { accessToken, refreshToken };
  }

  /**
   * Validate and refresh session
   */
  async validateSession(accessToken, req = {}) {
    try {
      // Verify JWT token
      const decoded = jwt.verify(accessToken, this.jwtSecret, {
        issuer: 'media-server',
        audience: 'media-server-api'
      });

      // Get session data
      const sessionData = await this.getSession(decoded.sessionId);
      if (!sessionData) {
        throw new Error('Session not found');
      }

      // Check expiration
      if (Date.now() > sessionData.expiresAt) {
        await this.destroySession(decoded.sessionId);
        throw new Error('Session expired');
      }

      // Validate fingerprint for security
      const currentFingerprint = this.generateFingerprint(req);
      if (sessionData.fingerprint !== currentFingerprint) {
        await this.destroySession(decoded.sessionId);
        throw new Error('Session fingerprint mismatch - possible hijacking');
      }

      // Update last activity
      sessionData.lastActivity = Date.now();
      
      // Check if rotation is needed
      const timeSinceCreation = Date.now() - sessionData.createdAt;
      if (timeSinceCreation > this.rotationInterval) {
        sessionData.rotationNeeded = true;
      }

      await this.storeSession(decoded.sessionId, sessionData);

      return {
        valid: true,
        sessionData,
        needsRotation: sessionData.rotationNeeded
      };
    } catch (error) {
      return {
        valid: false,
        error: error.message
      };
    }
  }

  /**
   * Rotate session tokens
   */
  async rotateSession(refreshToken, req = {}) {
    try {
      // Verify refresh token
      const decoded = jwt.verify(refreshToken, this.refreshSecret, {
        issuer: 'media-server',
        audience: 'media-server-refresh'
      });

      // Get session data
      const sessionData = await this.getSession(decoded.sessionId);
      if (!sessionData) {
        throw new Error('Session not found');
      }

      // Check refresh token expiration
      if (Date.now() > sessionData.refreshExpiresAt) {
        await this.destroySession(decoded.sessionId);
        throw new Error('Refresh token expired');
      }

      // Validate fingerprint
      const currentFingerprint = this.generateFingerprint(req);
      if (sessionData.fingerprint !== currentFingerprint) {
        await this.destroySession(decoded.sessionId);
        throw new Error('Session fingerprint mismatch');
      }

      // Generate new session ID and tokens
      const newSessionId = this.generateSessionId();
      const now = Date.now();
      
      // Update session data
      const updatedSessionData = {
        ...sessionData,
        sessionId: newSessionId,
        lastActivity: now,
        expiresAt: now + this.sessionTimeout,
        rotationNeeded: false,
        rotationCount: sessionData.rotationCount + 1
      };

      // Remove old session
      await this.removeSession(decoded.sessionId);
      
      // Store new session
      await this.storeSession(newSessionId, updatedSessionData);

      // Generate new tokens
      const tokens = this.generateTokenPair(newSessionId, decoded.userId, updatedSessionData);

      return {
        success: true,
        sessionId: newSessionId,
        accessToken: tokens.accessToken,
        refreshToken: tokens.refreshToken,
        expiresAt: updatedSessionData.expiresAt,
        refreshExpiresAt: updatedSessionData.refreshExpiresAt
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Destroy session
   */
  async destroySession(sessionId) {
    const sessionData = await this.getSession(sessionId);
    if (sessionData) {
      await this.removeUserSession(sessionData.userId, sessionId);
    }
    await this.removeSession(sessionId);
  }

  /**
   * Destroy all sessions for a user
   */
  async destroyAllUserSessions(userId) {
    const userSessions = await this.getUserSessions(userId);
    for (const sessionId of userSessions) {
      await this.removeSession(sessionId);
    }
    await this.clearUserSessions(userId);
  }

  /**
   * Enforce maximum sessions per user
   */
  async enforceMaxSessions(userId) {
    const userSessions = await this.getUserSessions(userId);
    
    if (userSessions.length >= this.maxSessions) {
      // Remove oldest sessions
      const sessionsToRemove = userSessions.slice(0, userSessions.length - this.maxSessions + 1);
      
      for (const sessionId of sessionsToRemove) {
        await this.removeSession(sessionId);
        await this.removeUserSession(userId, sessionId);
      }
    }
  }

  /**
   * Store session (Redis or in-memory)
   */
  async storeSession(sessionId, sessionData) {
    if (this.redisClient) {
      try {
        await this.redisClient.setEx(
          `session:${sessionId}`,
          Math.ceil(this.refreshTimeout / 1000),
          JSON.stringify(sessionData)
        );
      } catch (error) {
        console.error('Redis store error:', error);
        this.sessionStore.set(sessionId, sessionData);
      }
    } else {
      this.sessionStore.set(sessionId, sessionData);
    }
  }

  /**
   * Get session (Redis or in-memory)
   */
  async getSession(sessionId) {
    if (this.redisClient) {
      try {
        const data = await this.redisClient.get(`session:${sessionId}`);
        return data ? JSON.parse(data) : null;
      } catch (error) {
        console.error('Redis get error:', error);
        return this.sessionStore.get(sessionId) || null;
      }
    } else {
      return this.sessionStore.get(sessionId) || null;
    }
  }

  /**
   * Remove session (Redis or in-memory)
   */
  async removeSession(sessionId) {
    if (this.redisClient) {
      try {
        await this.redisClient.del(`session:${sessionId}`);
      } catch (error) {
        console.error('Redis delete error:', error);
      }
    }
    this.sessionStore.delete(sessionId);
  }

  /**
   * Add session to user's session list
   */
  async addUserSession(userId, sessionId) {
    if (this.redisClient) {
      try {
        await this.redisClient.sAdd(`user_sessions:${userId}`, sessionId);
        await this.redisClient.expire(`user_sessions:${userId}`, Math.ceil(this.refreshTimeout / 1000));
      } catch (error) {
        console.error('Redis user session add error:', error);
      }
    }
    
    if (!this.userSessions.has(userId)) {
      this.userSessions.set(userId, new Set());
    }
    this.userSessions.get(userId).add(sessionId);
  }

  /**
   * Remove session from user's session list
   */
  async removeUserSession(userId, sessionId) {
    if (this.redisClient) {
      try {
        await this.redisClient.sRem(`user_sessions:${userId}`, sessionId);
      } catch (error) {
        console.error('Redis user session remove error:', error);
      }
    }
    
    if (this.userSessions.has(userId)) {
      this.userSessions.get(userId).delete(sessionId);
    }
  }

  /**
   * Get all sessions for a user
   */
  async getUserSessions(userId) {
    if (this.redisClient) {
      try {
        return await this.redisClient.sMembers(`user_sessions:${userId}`);
      } catch (error) {
        console.error('Redis get user sessions error:', error);
      }
    }
    
    return Array.from(this.userSessions.get(userId) || []);
  }

  /**
   * Clear all sessions for a user
   */
  async clearUserSessions(userId) {
    if (this.redisClient) {
      try {
        await this.redisClient.del(`user_sessions:${userId}`);
      } catch (error) {
        console.error('Redis clear user sessions error:', error);
      }
    }
    
    this.userSessions.delete(userId);
  }

  /**
   * Initialize cleanup intervals
   */
  initializeCleanup() {
    // Clean up expired sessions every 5 minutes
    setInterval(async () => {
      await this.cleanupExpiredSessions();
    }, 5 * 60 * 1000);
  }

  /**
   * Clean up expired sessions
   */
  async cleanupExpiredSessions() {
    const now = Date.now();
    let cleanedCount = 0;

    // Clean in-memory sessions
    for (const [sessionId, sessionData] of this.sessionStore.entries()) {
      if (now > sessionData.refreshExpiresAt) {
        await this.destroySession(sessionId);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      console.log(`🧹 Cleaned up ${cleanedCount} expired sessions`);
    }
  }

  /**
   * Get session statistics
   */
  async getSessionStats() {
    const totalSessions = this.sessionStore.size;
    const activeSessions = Array.from(this.sessionStore.values())
      .filter(session => Date.now() < session.expiresAt).length;
    const totalUsers = this.userSessions.size;

    return {
      totalSessions,
      activeSessions,
      expiredSessions: totalSessions - activeSessions,
      totalUsers,
      averageSessionsPerUser: totalUsers > 0 ? Math.round(totalSessions / totalUsers * 100) / 100 : 0
    };
  }

  /**
   * Express middleware for session validation
   */
  middleware() {
    return async (req, res, next) => {
      const authHeader = req.headers.authorization;
      
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({
          error: 'Authentication required',
          message: 'No valid authorization header found'
        });
      }

      const token = authHeader.substring(7);
      const validation = await this.validateSession(token, req);

      if (!validation.valid) {
        return res.status(401).json({
          error: 'Session validation failed',
          message: validation.error
        });
      }

      // Add session info to request
      req.session = validation.sessionData;
      req.needsRotation = validation.needsRotation;

      // Add rotation hint header
      if (validation.needsRotation) {
        res.setHeader('X-Session-Rotation-Needed', 'true');
      }

      next();
    };
  }
}

module.exports = SecureSessionManager;