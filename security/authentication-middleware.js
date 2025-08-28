/**
 * Comprehensive Authentication Middleware
 * Fixes: Authentication bypass vulnerabilities, weak session management
 * Author: Security Manager Agent
 * Date: 2025-08-03
 */

const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const crypto = require('crypto');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');

class AuthenticationMiddleware {
  constructor(options = {}) {
    this.jwtSecret = options.jwtSecret || process.env.JWT_SECRET;
    this.sessionSecret = options.sessionSecret || process.env.SESSION_SECRET;
    this.saltRounds = options.saltRounds || 12;
    this.tokenExpiry = options.tokenExpiry || '15m';
    this.refreshTokenExpiry = options.refreshTokenExpiry || '7d';
    this.maxLoginAttempts = options.maxLoginAttempts || 5;
    this.lockoutDuration = options.lockoutDuration || 15 * 60 * 1000; // 15 minutes
    
    // Store for tracking login attempts and blacklisted tokens
    this.loginAttempts = new Map();
    this.blacklistedTokens = new Set();
    this.refreshTokens = new Map();
    
    // Session fingerprinting for enhanced security
    this.sessionFingerprints = new Map();
    
    if (!this.jwtSecret || !this.sessionSecret) {
      throw new Error('JWT_SECRET and SESSION_SECRET must be provided');
    }
  }

  /**
   * Apply comprehensive security headers
   */
  securityHeaders() {
    return helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
          fontSrc: ["'self'", "https://fonts.gstatic.com"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'"],
          frameSrc: ["'none'"],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          formAction: ["'self'"],
          frameAncestors: ["'none'"],
          baseUri: ["'self'"],
          upgradeInsecureRequests: [],
        },
      },
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
      },
      frameguard: { action: 'deny' },
      noSniff: true,
      xssFilter: true,
      referrerPolicy: { policy: 'strict-origin-when-cross-origin' }
    });
  }

  /**
   * Rate limiting for authentication endpoints
   */
  authRateLimit() {
    return rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 5, // limit each IP to 5 requests per windowMs
      message: {
        error: 'Too many authentication attempts, please try again later.',
        retryAfter: 15 * 60 * 1000
      },
      standardHeaders: true,
      legacyHeaders: false,
      handler: (req, res) => {
        res.status(429).json({
          error: 'Rate limit exceeded',
          message: 'Too many authentication attempts',
          retryAfter: req.rateLimit.resetTime
        });
      }
    });
  }

  /**
   * General API rate limiting
   */
  apiRateLimit() {
    return rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100, // limit each IP to 100 requests per windowMs
      message: {
        error: 'Too many API requests, please try again later.'
      },
      standardHeaders: true,
      legacyHeaders: false
    });
  }

  /**
   * Hash password with bcrypt
   */
  async hashPassword(password) {
    if (!password || password.length < 8) {
      throw new Error('Password must be at least 8 characters long');
    }
    return await bcrypt.hash(password, this.saltRounds);
  }

  /**
   * Verify password against hash
   */
  async verifyPassword(password, hash) {
    return await bcrypt.compare(password, hash);
  }

  /**
   * Generate JWT token pair (access + refresh)
   */
  generateTokens(payload) {
    const tokenId = crypto.randomUUID();
    
    const accessToken = jwt.sign(
      { 
        ...payload, 
        tokenId,
        type: 'access',
        iat: Math.floor(Date.now() / 1000)
      },
      this.jwtSecret,
      { 
        expiresIn: this.tokenExpiry,
        issuer: 'media-server-auth',
        audience: 'media-server-api'
      }
    );

    const refreshToken = jwt.sign(
      { 
        userId: payload.userId, 
        tokenId,
        type: 'refresh',
        iat: Math.floor(Date.now() / 1000)
      },
      this.sessionSecret,
      { 
        expiresIn: this.refreshTokenExpiry,
        issuer: 'media-server-auth',
        audience: 'media-server-api'
      }
    );

    // Store refresh token for validation
    this.refreshTokens.set(refreshToken, {
      userId: payload.userId,
      tokenId,
      createdAt: Date.now()
    });

    return { accessToken, refreshToken, tokenId };
  }

  /**
   * Verify JWT token
   */
  async verifyToken(token, isRefreshToken = false) {
    try {
      // Check if token is blacklisted
      if (this.blacklistedTokens.has(token)) {
        throw new Error('Token has been revoked');
      }

      const secret = isRefreshToken ? this.sessionSecret : this.jwtSecret;
      const decoded = jwt.verify(token, secret, {
        issuer: 'media-server-auth',
        audience: 'media-server-api'
      });

      // Additional validation for refresh tokens
      if (isRefreshToken) {
        const tokenInfo = this.refreshTokens.get(token);
        if (!tokenInfo || tokenInfo.tokenId !== decoded.tokenId) {
          throw new Error('Invalid refresh token');
        }
      }

      return decoded;
    } catch (error) {
      throw new Error(`Token verification failed: ${error.message}`);
    }
  }

  /**
   * Revoke token (add to blacklist)
   */
  revokeToken(token) {
    this.blacklistedTokens.add(token);
    
    // If it's a refresh token, remove from store
    if (this.refreshTokens.has(token)) {
      this.refreshTokens.delete(token);
    }
  }

  /**
   * Refresh access token
   */
  async refreshAccessToken(refreshToken) {
    const decoded = await this.verifyToken(refreshToken, true);
    
    // Generate new access token with same user data
    const newTokens = this.generateTokens({
      userId: decoded.userId,
      // Add any other user data that should be in the token
    });

    // Revoke old refresh token
    this.revokeToken(refreshToken);

    return newTokens;
  }

  /**
   * Check for brute force attacks
   */
  isAccountLocked(identifier) {
    const attempts = this.loginAttempts.get(identifier);
    if (!attempts) return false;

    if (attempts.count >= this.maxLoginAttempts) {
      const timeSinceLastAttempt = Date.now() - attempts.lastAttempt;
      return timeSinceLastAttempt < this.lockoutDuration;
    }

    return false;
  }

  /**
   * Record failed login attempt
   */
  recordFailedAttempt(identifier) {
    const attempts = this.loginAttempts.get(identifier) || { count: 0, lastAttempt: 0 };
    attempts.count++;
    attempts.lastAttempt = Date.now();
    this.loginAttempts.set(identifier, attempts);
  }

  /**
   * Clear failed login attempts
   */
  clearFailedAttempts(identifier) {
    this.loginAttempts.delete(identifier);
  }

  /**
   * Authentication middleware
   */
  authenticate() {
    return async (req, res, next) => {
      try {
        const authHeader = req.headers.authorization;
        
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
          return res.status(401).json({
            error: 'Authentication required',
            message: 'No valid authorization header found'
          });
        }

        const token = authHeader.substring(7);
        const decoded = await this.verifyToken(token);

        // Add user info to request
        req.user = decoded;
        req.tokenId = decoded.tokenId;

        next();
      } catch (error) {
        return res.status(401).json({
          error: 'Authentication failed',
          message: error.message
        });
      }
    };
  }

  /**
   * Authorization middleware (role-based)
   */
  authorize(requiredRoles = []) {
    return (req, res, next) => {
      if (!req.user) {
        return res.status(401).json({
          error: 'Authentication required'
        });
      }

      if (requiredRoles.length === 0) {
        return next();
      }

      const userRoles = req.user.roles || [];
      const hasRequiredRole = requiredRoles.some(role => userRoles.includes(role));

      if (!hasRequiredRole) {
        return res.status(403).json({
          error: 'Insufficient permissions',
          message: `Required roles: ${requiredRoles.join(', ')}`
        });
      }

      next();
    };
  }

  /**
   * Input validation middleware
   */
  validateInput(schema) {
    return (req, res, next) => {
      const { error, value } = schema.validate(req.body, {
        abortEarly: false,
        stripUnknown: true
      });

      if (error) {
        return res.status(400).json({
          error: 'Validation failed',
          details: error.details.map(detail => ({
            field: detail.path.join('.'),
            message: detail.message
          }))
        });
      }

      req.validatedData = value;
      next();
    };
  }

  /**
   * CSRF protection middleware
   */
  csrfProtection() {
    return (req, res, next) => {
      // Skip CSRF for GET, HEAD, OPTIONS
      if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
        return next();
      }

      const csrfToken = req.headers['x-csrf-token'] || req.body._csrf;
      const sessionToken = req.session?.csrfToken;

      if (!csrfToken || !sessionToken || csrfToken !== sessionToken) {
        return res.status(403).json({
          error: 'CSRF token validation failed'
        });
      }

      next();
    };
  }

  /**
   * Generate CSRF token
   */
  generateCSRFToken(req) {
    const token = crypto.randomBytes(32).toString('hex');
    req.session.csrfToken = token;
    return token;
  }

  /**
   * Session cleanup (remove expired tokens)
   */
  cleanupExpiredTokens() {
    const now = Date.now();
    const refreshTokenExpiry = 7 * 24 * 60 * 60 * 1000; // 7 days in ms

    for (const [token, info] of this.refreshTokens.entries()) {
      if (now - info.createdAt > refreshTokenExpiry) {
        this.refreshTokens.delete(token);
      }
    }

    // Clean up login attempts older than lockout duration
    for (const [identifier, attempts] of this.loginAttempts.entries()) {
      if (now - attempts.lastAttempt > this.lockoutDuration * 2) {
        this.loginAttempts.delete(identifier);
      }
    }
  }

  /**
   * Start cleanup interval
   */
  startCleanup(intervalMs = 60 * 60 * 1000) { // 1 hour
    setInterval(() => {
      this.cleanupExpiredTokens();
    }, intervalMs);
  }
}

module.exports = AuthenticationMiddleware;