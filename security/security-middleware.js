const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const validator = require('validator');
const xss = require('xss');
const csrf = require('csurf');
const session = require('express-session');
const MongoStore = require('connect-mongo');
const winston = require('winston');

// Security logging
const securityLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: './logs/security-error.log', level: 'error' }),
    new winston.transports.File({ filename: './logs/security-combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

class SecurityMiddleware {
  constructor(options = {}) {
    this.options = {
      rateLimit: true,
      csrf: true,
      xss: true,
      inputValidation: true,
      sessionSecurity: true,
      securityHeaders: true,
      ...options
    };
  }

  // Comprehensive security headers
  getHelmetConfig() {
    return helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: [
            "'self'", 
            "'unsafe-inline'", 
            "'unsafe-eval'", 
            "https://cdn.jsdelivr.net",
            "https://unpkg.com"
          ],
          styleSrc: [
            "'self'", 
            "'unsafe-inline'", 
            "https://fonts.googleapis.com",
            "https://cdn.jsdelivr.net"
          ],
          imgSrc: ["'self'", "data:", "https:", "blob:"],
          fontSrc: [
            "'self'", 
            "https://fonts.gstatic.com",
            "https://cdn.jsdelivr.net"
          ],
          connectSrc: [
            "'self'", 
            "wss:", 
            "ws:", 
            "https://api.openai.com"
          ],
          mediaSrc: ["'self'", "blob:"],
          objectSrc: ["'none'"],
          baseUri: ["'self'"],
          frameAncestors: ["'none'"],
          formAction: ["'self'"],
          upgradeInsecureRequests: [],
          blockAllMixedContent: []
        },
        reportOnly: false
      },
      crossOriginEmbedderPolicy: { policy: "require-corp" },
      crossOriginOpenerPolicy: { policy: "same-origin" },
      crossOriginResourcePolicy: { policy: "same-site" },
      dnsPrefetchControl: { allow: false },
      frameguard: { action: 'deny' },
      hidePoweredBy: true,
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
      },
      ieNoOpen: true,
      noSniff: true,
      originAgentCluster: true,
      permittedCrossDomainPolicies: false,
      referrerPolicy: { policy: "strict-origin-when-cross-origin" },
      xssFilter: true
    });
  }

  // Rate limiting configurations
  getRateLimiters() {
    return {
      general: rateLimit({
        windowMs: 60 * 1000, // 1 minute
        max: 100, // limit each IP to 100 requests per windowMs
        message: {
          error: 'Too many requests from this IP, please try again later.',
          code: 'RATE_LIMIT_EXCEEDED'
        },
        standardHeaders: true,
        legacyHeaders: false,
        handler: (req, res) => {
          securityLogger.warn(`Rate limit exceeded for IP: ${req.ip}`, {
            ip: req.ip,
            userAgent: req.get('User-Agent'),
            path: req.path
          });
          res.status(429).json({
            error: 'Too many requests from this IP, please try again later.',
            code: 'RATE_LIMIT_EXCEEDED'
          });
        }
      }),

      auth: rateLimit({
        windowMs: 60 * 1000, // 1 minute
        max: 5, // limit each IP to 5 authentication attempts per minute
        message: {
          error: 'Too many authentication attempts, please try again later.',
          code: 'AUTH_RATE_LIMIT_EXCEEDED'
        },
        skipSuccessfulRequests: true,
        handler: (req, res) => {
          securityLogger.error(`Authentication rate limit exceeded for IP: ${req.ip}`, {
            ip: req.ip,
            userAgent: req.get('User-Agent'),
            path: req.path
          });
          res.status(429).json({
            error: 'Too many authentication attempts, please try again later.',
            code: 'AUTH_RATE_LIMIT_EXCEEDED'
          });
        }
      }),

      api: rateLimit({
        windowMs: 60 * 1000, // 1 minute
        max: 1000, // limit each IP to 1000 API requests per minute
        message: {
          error: 'API rate limit exceeded, please try again later.',
          code: 'API_RATE_LIMIT_EXCEEDED'
        }
      }),

      upload: rateLimit({
        windowMs: 60 * 1000, // 1 minute
        max: 10, // limit each IP to 10 uploads per minute
        message: {
          error: 'Upload rate limit exceeded, please try again later.',
          code: 'UPLOAD_RATE_LIMIT_EXCEEDED'
        }
      })
    };
  }

  // Input validation and sanitization
  validateAndSanitize(req, res, next) {
    try {
      // Sanitize all string inputs
      const sanitizeObject = (obj) => {
        for (const key in obj) {
          if (typeof obj[key] === 'string') {
            // XSS protection
            obj[key] = xss(obj[key], {
              whiteList: {}, // No HTML allowed
              stripIgnoreTag: true,
              stripIgnoreTagBody: ['script']
            });
            
            // Additional sanitization
            obj[key] = validator.escape(obj[key]);
          } else if (typeof obj[key] === 'object' && obj[key] !== null) {
            sanitizeObject(obj[key]);
          }
        }
      };

      if (req.body) sanitizeObject(req.body);
      if (req.query) sanitizeObject(req.query);
      if (req.params) sanitizeObject(req.params);

      // Validate common patterns
      if (req.body.email && !validator.isEmail(req.body.email)) {
        return res.status(400).json({
          error: 'Invalid email format',
          code: 'VALIDATION_ERROR'
        });
      }

      if (req.body.url && !validator.isURL(req.body.url)) {
        return res.status(400).json({
          error: 'Invalid URL format',
          code: 'VALIDATION_ERROR'
        });
      }

      // SQL injection protection
      const sqlInjectionPattern = /(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE){0,1}|INSERT( +INTO){0,1}|MERGE|SELECT|UPDATE|UNION( +ALL){0,1})\b)/i;
      const checkSqlInjection = (value) => {
        if (typeof value === 'string' && sqlInjectionPattern.test(value)) {
          securityLogger.error('SQL injection attempt detected', {
            ip: req.ip,
            userAgent: req.get('User-Agent'),
            value: value,
            path: req.path
          });
          return true;
        }
        return false;
      };

      const containsSqlInjection = Object.values(req.body || {}).some(checkSqlInjection) ||
                                   Object.values(req.query || {}).some(checkSqlInjection) ||
                                   Object.values(req.params || {}).some(checkSqlInjection);

      if (containsSqlInjection) {
        return res.status(400).json({
          error: 'Invalid input detected',
          code: 'SECURITY_VIOLATION'
        });
      }

      // Path traversal protection
      const pathTraversalPattern = /\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c/i;
      if (pathTraversalPattern.test(req.url)) {
        securityLogger.error('Path traversal attempt detected', {
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          url: req.url
        });
        return res.status(400).json({
          error: 'Invalid path detected',
          code: 'SECURITY_VIOLATION'
        });
      }

      next();
    } catch (error) {
      securityLogger.error('Input validation error', {
        error: error.message,
        ip: req.ip,
        path: req.path
      });
      res.status(500).json({
        error: 'Internal server error',
        code: 'INTERNAL_ERROR'
      });
    }
  }

  // Secure session configuration
  getSessionConfig() {
    return session({
      name: 'sessionId', // Don't use default session name
      secret: process.env.SESSION_SECRET || 'super-secret-key-change-in-production',
      resave: false,
      saveUninitialized: false,
      cookie: {
        secure: process.env.NODE_ENV === 'production', // HTTPS only in production
        httpOnly: true, // Prevent XSS
        maxAge: 3600000, // 1 hour
        sameSite: 'strict' // CSRF protection
      },
      store: process.env.MONGODB_URI ? MongoStore.create({
        mongoUrl: process.env.MONGODB_URI,
        touchAfter: 24 * 3600 // lazy session update
      }) : undefined
    });
  }

  // CSRF protection
  getCsrfProtection() {
    return csrf({
      cookie: {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict'
      }
    });
  }

  // Security monitoring middleware
  securityMonitoring(req, res, next) {
    const startTime = Date.now();
    
    // Log security-relevant events
    securityLogger.info('Request received', {
      ip: req.ip,
      method: req.method,
      path: req.path,
      userAgent: req.get('User-Agent'),
      referer: req.get('Referer'),
      timestamp: new Date().toISOString()
    });

    // Monitor for suspicious patterns
    const suspiciousPatterns = [
      /admin/i,
      /phpMyAdmin/i,
      /wp-login/i,
      /\.env/i,
      /config\.php/i,
      /backup/i
    ];

    if (suspiciousPatterns.some(pattern => pattern.test(req.path))) {
      securityLogger.warn('Suspicious path accessed', {
        ip: req.ip,
        path: req.path,
        userAgent: req.get('User-Agent')
      });
    }

    // Response time monitoring
    res.on('finish', () => {
      const responseTime = Date.now() - startTime;
      if (responseTime > 5000) { // Log slow responses
        securityLogger.warn('Slow response detected', {
          ip: req.ip,
          path: req.path,
          responseTime,
          statusCode: res.statusCode
        });
      }
    });

    next();
  }

  // Error handling middleware
  secureErrorHandler(err, req, res, next) {
    securityLogger.error('Error occurred', {
      error: err.message,
      stack: err.stack,
      ip: req.ip,
      path: req.path,
      timestamp: new Date().toISOString()
    });

    // Don't expose internal errors in production
    if (process.env.NODE_ENV === 'production') {
      res.status(500).json({
        error: 'Internal server error',
        code: 'INTERNAL_ERROR'
      });
    } else {
      res.status(500).json({
        error: err.message,
        stack: err.stack,
        code: 'INTERNAL_ERROR'
      });
    }
  }

  // Apply all security middleware
  applySecurityMiddleware(app) {
    if (this.options.securityHeaders) {
      app.use(this.getHelmetConfig());
    }

    if (this.options.sessionSecurity) {
      app.use(this.getSessionConfig());
    }

    if (this.options.inputValidation) {
      app.use(this.validateAndSanitize);
    }

    app.use(this.securityMonitoring);

    if (this.options.rateLimit) {
      const rateLimiters = this.getRateLimiters();
      app.use('/auth', rateLimiters.auth);
      app.use('/api', rateLimiters.api);
      app.use('/upload', rateLimiters.upload);
      app.use(rateLimiters.general);
    }

    if (this.options.csrf) {
      app.use(this.getCsrfProtection());
    }

    // Error handling (should be last)
    app.use(this.secureErrorHandler);

    return app;
  }
}

module.exports = SecurityMiddleware;