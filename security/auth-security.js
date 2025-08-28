const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const speakeasy = require('speakeasy');
const qrcode = require('qrcode');
const winston = require('winston');

// Security logging
const authLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: './logs/auth-error.log', level: 'error' }),
    new winston.transports.File({ filename: './logs/auth-combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

class AuthSecurity {
  constructor(options = {}) {
    this.options = {
      jwtSecret: process.env.JWT_SECRET || 'super-secret-jwt-key-change-in-production',
      jwtExpiry: process.env.JWT_EXPIRY || '1h',
      refreshTokenExpiry: process.env.REFRESH_TOKEN_EXPIRY || '7d',
      bcryptRounds: parseInt(process.env.BCRYPT_ROUNDS) || 12,
      maxLoginAttempts: parseInt(process.env.MAX_LOGIN_ATTEMPTS) || 5,
      lockoutDuration: parseInt(process.env.LOCKOUT_DURATION) || 900000, // 15 minutes
      mfaRequired: process.env.MFA_REQUIRED === 'true',
      ...options
    };
    
    // In-memory store for failed attempts (use Redis in production)
    this.failedAttempts = new Map();
    this.lockedAccounts = new Map();
  }

  // Password security
  async hashPassword(password) {
    try {
      // Validate password strength
      if (!this.isPasswordStrong(password)) {
        throw new Error('Password does not meet security requirements');
      }
      
      const salt = await bcrypt.genSalt(this.options.bcryptRounds);
      return await bcrypt.hash(password, salt);
    } catch (error) {
      authLogger.error('Password hashing failed', { error: error.message });
      throw error;
    }
  }

  async verifyPassword(password, hashedPassword) {
    try {
      return await bcrypt.compare(password, hashedPassword);
    } catch (error) {
      authLogger.error('Password verification failed', { error: error.message });
      throw error;
    }
  }

  isPasswordStrong(password) {
    const minLength = 12;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    const noCommonPatterns = !/123456|password|qwerty|admin|letmein/.test(password.toLowerCase());

    return password.length >= minLength &&
           hasUpperCase &&
           hasLowerCase &&
           hasNumbers &&
           hasSpecialChar &&
           noCommonPatterns;
  }

  // JWT Token Management
  generateTokens(user) {
    try {
      const payload = {
        userId: user.id,
        email: user.email,
        roles: user.roles || ['user'],
        permissions: user.permissions || [],
        iat: Math.floor(Date.now() / 1000)
      };

      const accessToken = jwt.sign(payload, this.options.jwtSecret, {
        expiresIn: this.options.jwtExpiry,
        issuer: 'media-platform',
        audience: 'media-platform-client'
      });

      const refreshToken = jwt.sign(
        { userId: user.id, type: 'refresh' },
        this.options.jwtSecret,
        {
          expiresIn: this.options.refreshTokenExpiry,
          issuer: 'media-platform',
          audience: 'media-platform-client'
        }
      );

      return { accessToken, refreshToken };
    } catch (error) {
      authLogger.error('Token generation failed', { 
        error: error.message,
        userId: user.id 
      });
      throw error;
    }
  }

  verifyToken(token) {
    try {
      return jwt.verify(token, this.options.jwtSecret, {
        issuer: 'media-platform',
        audience: 'media-platform-client'
      });
    } catch (error) {
      authLogger.warn('Token verification failed', { 
        error: error.message,
        token: token.substring(0, 20) + '...' 
      });
      throw error;
    }
  }

  refreshTokens(refreshToken) {
    try {
      const decoded = this.verifyToken(refreshToken);
      
      if (decoded.type !== 'refresh') {
        throw new Error('Invalid refresh token');
      }

      // In a real application, fetch user from database
      const user = { id: decoded.userId }; // Placeholder
      
      return this.generateTokens(user);
    } catch (error) {
      authLogger.error('Token refresh failed', { error: error.message });
      throw error;
    }
  }

  // Multi-Factor Authentication (MFA)
  generateMFASecret(userEmail) {
    try {
      const secret = speakeasy.generateSecret({
        name: `Media Platform (${userEmail})`,
        issuer: 'Media Platform',
        length: 32
      });

      return {
        secret: secret.base32,
        qrCodeUrl: secret.otpauth_url
      };
    } catch (error) {
      authLogger.error('MFA secret generation failed', { 
        error: error.message,
        email: userEmail 
      });
      throw error;
    }
  }

  async generateMFAQRCode(secret) {
    try {
      return await qrcode.toDataURL(secret);
    } catch (error) {
      authLogger.error('MFA QR code generation failed', { error: error.message });
      throw error;
    }
  }

  verifyMFAToken(token, secret) {
    try {
      return speakeasy.totp.verify({
        secret: secret,
        encoding: 'base32',
        token: token,
        window: 2 // Allow 2 time steps of tolerance
      });
    } catch (error) {
      authLogger.error('MFA token verification failed', { error: error.message });
      throw error;
    }
  }

  // Account Security
  recordFailedAttempt(identifier, ip) {
    const key = `${identifier}:${ip}`;
    const attempts = this.failedAttempts.get(key) || 0;
    this.failedAttempts.set(key, attempts + 1);

    authLogger.warn('Failed login attempt', {
      identifier,
      ip,
      attempts: attempts + 1
    });

    if (attempts + 1 >= this.options.maxLoginAttempts) {
      this.lockAccount(identifier, ip);
    }
  }

  lockAccount(identifier, ip) {
    const key = `${identifier}:${ip}`;
    const lockUntil = Date.now() + this.options.lockoutDuration;
    this.lockedAccounts.set(key, lockUntil);

    authLogger.error('Account locked due to failed attempts', {
      identifier,
      ip,
      lockUntil: new Date(lockUntil)
    });
  }

  isAccountLocked(identifier, ip) {
    const key = `${identifier}:${ip}`;
    const lockUntil = this.lockedAccounts.get(key);
    
    if (lockUntil && Date.now() < lockUntil) {
      return true;
    }
    
    if (lockUntil && Date.now() >= lockUntil) {
      // Unlock account
      this.lockedAccounts.delete(key);
      this.failedAttempts.delete(key);
    }
    
    return false;
  }

  clearFailedAttempts(identifier, ip) {
    const key = `${identifier}:${ip}`;
    this.failedAttempts.delete(key);
    this.lockedAccounts.delete(key);
  }

  // Session Security
  generateSecureSessionId() {
    return crypto.randomBytes(32).toString('hex');
  }

  createSessionFingerprint(req) {
    const components = [
      req.get('User-Agent') || '',
      req.get('Accept-Language') || '',
      req.get('Accept-Encoding') || '',
      req.ip || ''
    ];
    
    return crypto
      .createHash('sha256')
      .update(components.join('|'))
      .digest('hex');
  }

  validateSessionFingerprint(req, storedFingerprint) {
    const currentFingerprint = this.createSessionFingerprint(req);
    return currentFingerprint === storedFingerprint;
  }

  // Authorization middleware
  requireAuth(req, res, next) {
    try {
      const authHeader = req.headers.authorization;
      
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({
          error: 'Access token required',
          code: 'UNAUTHORIZED'
        });
      }

      const token = authHeader.substring(7);
      const decoded = this.verifyToken(token);
      
      req.user = decoded;
      next();
    } catch (error) {
      authLogger.warn('Authentication failed', {
        error: error.message,
        ip: req.ip,
        path: req.path
      });
      
      res.status(401).json({
        error: 'Invalid or expired token',
        code: 'UNAUTHORIZED'
      });
    }
  }

  requireRole(roles) {
    return (req, res, next) => {
      if (!req.user) {
        return res.status(401).json({
          error: 'Authentication required',
          code: 'UNAUTHORIZED'
        });
      }

      const userRoles = req.user.roles || [];
      const hasRequiredRole = roles.some(role => userRoles.includes(role));

      if (!hasRequiredRole) {
        authLogger.warn('Insufficient permissions', {
          userId: req.user.userId,
          requiredRoles: roles,
          userRoles: userRoles,
          ip: req.ip,
          path: req.path
        });

        return res.status(403).json({
          error: 'Insufficient permissions',
          code: 'FORBIDDEN'
        });
      }

      next();
    };
  }

  requirePermission(permissions) {
    return (req, res, next) => {
      if (!req.user) {
        return res.status(401).json({
          error: 'Authentication required',
          code: 'UNAUTHORIZED'
        });
      }

      const userPermissions = req.user.permissions || [];
      const hasRequiredPermission = permissions.some(permission => 
        userPermissions.includes(permission)
      );

      if (!hasRequiredPermission) {
        authLogger.warn('Insufficient permissions', {
          userId: req.user.userId,
          requiredPermissions: permissions,
          userPermissions: userPermissions,
          ip: req.ip,
          path: req.path
        });

        return res.status(403).json({
          error: 'Insufficient permissions',
          code: 'FORBIDDEN'
        });
      }

      next();
    };
  }

  // Security audit
  auditSecurityEvent(event, details, req) {
    authLogger.info('Security audit event', {
      event,
      details,
      userId: req.user?.userId,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      timestamp: new Date().toISOString()
    });
  }
}

module.exports = AuthSecurity;