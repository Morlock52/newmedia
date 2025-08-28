/**
 * Secure Logging Configuration
 * Fixes: Logging security issues, sensitive data in logs
 * Author: Security Manager Agent
 * Date: 2025-08-03
 */

const winston = require('winston');
const DailyRotateFile = require('winston-daily-rotate-file');
const crypto = require('crypto');
const path = require('path');

class SecureLogger {
  constructor(options = {}) {
    this.logDir = options.logDir || './security/logs';
    this.maxSize = options.maxSize || '20m';
    this.maxFiles = options.maxFiles || '14d';
    this.level = options.level || 'info';
    this.redactSensitive = options.redactSensitive !== false;
    
    // Sensitive data patterns to redact
    this.sensitivePatterns = [
      /password["\s]*[:=]["\s]*[^"\s,}]+/gi,
      /token["\s]*[:=]["\s]*[^"\s,}]+/gi,
      /secret["\s]*[:=]["\s]*[^"\s,}]+/gi,
      /key["\s]*[:=]["\s]*[^"\s,}]+/gi,
      /authorization:\s*bearer\s+[^\s]+/gi,
      /cookie:\s*[^;]+/gi,
      /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/gi, // Email addresses
      /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/gi, // Credit card numbers
      /\b\d{3}-\d{2}-\d{4}\b/gi, // SSN format
      /"password":\s*"[^"]*"/gi,
      /"token":\s*"[^"]*"/gi,
      /"secret":\s*"[^"]*"/gi,
      /"apikey":\s*"[^"]*"/gi,
      /"api_key":\s*"[^"]*"/gi
    ];
    
    this.logger = this.createLogger();
  }

  /**
   * Create Winston logger with secure configuration
   */
  createLogger() {
    const formats = [
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.json()
    ];

    // Add redaction format if enabled
    if (this.redactSensitive) {
      formats.splice(-1, 0, this.createRedactionFormat());
    }

    const logger = winston.createLogger({
      level: this.level,
      format: winston.format.combine(...formats),
      defaultMeta: {
        service: 'media-server',
        version: process.env.APP_VERSION || '1.0.0',
        environment: process.env.NODE_ENV || 'development'
      },
      transports: [
        // Application logs
        new DailyRotateFile({
          filename: path.join(this.logDir, 'application-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          maxSize: this.maxSize,
          maxFiles: this.maxFiles,
          level: 'info'
        }),
        
        // Error logs
        new DailyRotateFile({
          filename: path.join(this.logDir, 'error-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          maxSize: this.maxSize,
          maxFiles: this.maxFiles,
          level: 'error'
        }),
        
        // Security logs
        new DailyRotateFile({
          filename: path.join(this.logDir, 'security-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          maxSize: this.maxSize,
          maxFiles: this.maxFiles,
          level: 'warn',
          handleExceptions: false,
          handleRejections: false
        }),
        
        // Audit logs (separate from main logs)
        new DailyRotateFile({
          filename: path.join(this.logDir, 'audit-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          maxSize: this.maxSize,
          maxFiles: '30d', // Keep audit logs longer
          level: 'info',
          handleExceptions: false,
          handleRejections: false
        })
      ],
      
      // Handle uncaught exceptions and rejections
      exceptionHandlers: [
        new DailyRotateFile({
          filename: path.join(this.logDir, 'exceptions-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          maxSize: this.maxSize,
          maxFiles: this.maxFiles
        })
      ],
      
      rejectionHandlers: [
        new DailyRotateFile({
          filename: path.join(this.logDir, 'rejections-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          maxSize: this.maxSize,
          maxFiles: this.maxFiles
        })
      ]
    });

    // Add console transport in development
    if (process.env.NODE_ENV !== 'production') {
      logger.add(new winston.transports.Console({
        format: winston.format.combine(
          winston.format.colorize(),
          winston.format.simple(),
          this.redactSensitive ? this.createRedactionFormat() : winston.format.simple()
        )
      }));
    }

    return logger;
  }

  /**
   * Create format for redacting sensitive information
   */
  createRedactionFormat() {
    return winston.format((info) => {
      // Redact sensitive data from message
      if (typeof info.message === 'string') {
        info.message = this.redactSensitiveData(info.message);
      }
      
      // Redact sensitive data from metadata
      if (info.meta && typeof info.meta === 'object') {
        info.meta = this.redactSensitiveObject(info.meta);
      }
      
      // Redact from other properties
      Object.keys(info).forEach(key => {
        if (typeof info[key] === 'string' && key !== 'level' && key !== 'timestamp') {
          info[key] = this.redactSensitiveData(info[key]);
        } else if (typeof info[key] === 'object' && info[key] !== null) {
          info[key] = this.redactSensitiveObject(info[key]);
        }
      });
      
      return info;
    })();
  }

  /**
   * Redact sensitive data from strings
   */
  redactSensitiveData(text) {
    let redacted = text;
    
    this.sensitivePatterns.forEach(pattern => {
      redacted = redacted.replace(pattern, (match) => {
        // Keep the key/field name but redact the value
        const parts = match.split(/[:=]/);
        if (parts.length > 1) {
          return `${parts[0]}:***REDACTED***`;
        }
        return '***REDACTED***';
      });
    });
    
    return redacted;
  }

  /**
   * Redact sensitive data from objects
   */
  redactSensitiveObject(obj, depth = 0) {
    if (depth > 10) return obj; // Prevent infinite recursion
    
    if (Array.isArray(obj)) {
      return obj.map(item => 
        typeof item === 'object' ? this.redactSensitiveObject(item, depth + 1) : 
        typeof item === 'string' ? this.redactSensitiveData(item) : item
      );
    }
    
    if (typeof obj === 'object' && obj !== null) {
      const redacted = {};
      
      Object.keys(obj).forEach(key => {
        const lowerKey = key.toLowerCase();
        
        // Check if key name indicates sensitive data
        if (this.isSensitiveKey(lowerKey)) {
          redacted[key] = '***REDACTED***';
        } else if (typeof obj[key] === 'string') {
          redacted[key] = this.redactSensitiveData(obj[key]);
        } else if (typeof obj[key] === 'object') {
          redacted[key] = this.redactSensitiveObject(obj[key], depth + 1);
        } else {
          redacted[key] = obj[key];
        }
      });
      
      return redacted;
    }
    
    return obj;
  }

  /**
   * Check if a key name indicates sensitive data
   */
  isSensitiveKey(key) {
    const sensitiveKeys = [
      'password', 'passwd', 'pwd', 'secret', 'token', 'jwt', 'auth', 
      'authorization', 'cookie', 'session', 'key', 'apikey', 'api_key',
      'private', 'credential', 'cert', 'certificate', 'signature',
      'hash', 'salt', 'nonce', 'ssn', 'social', 'credit', 'card',
      'account', 'pin', 'otp', 'totp', 'mfa'
    ];
    
    return sensitiveKeys.some(sensitive => key.includes(sensitive));
  }

  /**
   * Log security event
   */
  logSecurity(level, message, metadata = {}) {
    const securityLog = {
      category: 'SECURITY',
      eventId: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      message,
      ...metadata
    };
    
    this.logger.log(level, message, securityLog);
  }

  /**
   * Log audit event
   */
  logAudit(action, userId, resource, metadata = {}) {
    const auditLog = {
      category: 'AUDIT',
      eventId: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      action,
      userId,
      resource,
      ...metadata
    };
    
    this.logger.info('Audit Event', auditLog);
  }

  /**
   * Log authentication event
   */
  logAuth(event, userId, ipAddress, userAgent, success = true, metadata = {}) {
    const authLog = {
      category: 'AUTHENTICATION',
      eventId: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      event,
      userId,
      ipAddress,
      userAgent: userAgent?.substring(0, 200), // Limit length
      success,
      ...metadata
    };
    
    this.logger.info('Authentication Event', authLog);
  }

  /**
   * Log data access event
   */
  logDataAccess(userId, resource, action, ipAddress, metadata = {}) {
    const accessLog = {
      category: 'DATA_ACCESS',
      eventId: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      userId,
      resource,
      action,
      ipAddress,
      ...metadata
    };
    
    this.logger.info('Data Access Event', accessLog);
  }

  /**
   * Log error with context
   */
  logError(error, context = {}) {
    const errorLog = {
      category: 'ERROR',
      eventId: crypto.randomUUID(),
      timestamp: new Date().toISOString(),
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack,
        code: error.code
      },
      ...context
    };
    
    this.logger.error('Application Error', errorLog);
  }

  /**
   * Express middleware for request logging
   */
  requestLogger() {
    return (req, res, next) => {
      const start = Date.now();
      const requestId = crypto.randomUUID();
      
      req.requestId = requestId;
      
      // Log request
      this.logger.info('HTTP Request', {
        category: 'HTTP',
        requestId,
        method: req.method,
        url: req.url,
        userAgent: req.headers['user-agent']?.substring(0, 200),
        ipAddress: req.ip || req.connection.remoteAddress,
        timestamp: new Date().toISOString()
      });
      
      // Override res.end to log response
      const originalEnd = res.end;
      res.end = function(...args) {
        const duration = Date.now() - start;
        
        // Log response
        req.secureLogger.logger.info('HTTP Response', {
          category: 'HTTP',
          requestId,
          statusCode: res.statusCode,
          duration,
          timestamp: new Date().toISOString()
        });
        
        originalEnd.apply(this, args);
      };
      
      req.secureLogger = this;
      next();
    };
  }

  /**
   * Get logger instance for direct use
   */
  getLogger() {
    return this.logger;
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    return new Promise((resolve) => {
      this.logger.end(() => {
        console.log('Logger shutdown complete');
        resolve();
      });
    });
  }
}

module.exports = SecureLogger;