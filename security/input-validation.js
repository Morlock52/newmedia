/**
 * Comprehensive Input Validation and Sanitization
 * Fixes: Unvalidated input processing, XSS, SQL injection
 * Author: Security Manager Agent
 * Date: 2025-08-03
 */

const Joi = require('joi');
const DOMPurify = require('isomorphic-dompurify');
const validator = require('validator');
const xss = require('xss');

class InputValidator {
  constructor() {
    this.schemas = this.defineSchemas();
    this.xssOptions = {
      whiteList: {
        p: [],
        br: [],
        strong: [],
        em: [],
        u: [],
        code: [],
        pre: []
      },
      stripIgnoreTag: true,
      stripIgnoreTagBody: ['script', 'style', 'iframe', 'object', 'embed']
    };
  }

  /**
   * Define validation schemas for different data types
   */
  defineSchemas() {
    return {
      // User authentication schemas
      login: Joi.object({
        username: Joi.string()
          .alphanum()
          .min(3)
          .max(30)
          .required()
          .messages({
            'string.alphanum': 'Username must contain only alphanumeric characters',
            'string.min': 'Username must be at least 3 characters long',
            'string.max': 'Username cannot exceed 30 characters'
          }),
        password: Joi.string()
          .min(8)
          .max(128)
          .pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/)
          .required()
          .messages({
            'string.pattern.base': 'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character',
            'string.min': 'Password must be at least 8 characters long'
          }),
        rememberMe: Joi.boolean().default(false)
      }),

      // User registration schema
      register: Joi.object({
        username: Joi.string()
          .alphanum()
          .min(3)
          .max(30)
          .required(),
        email: Joi.string()
          .email({ tlds: { allow: false } })
          .required(),
        password: Joi.string()
          .min(8)
          .max(128)
          .pattern(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/)
          .required(),
        confirmPassword: Joi.string()
          .valid(Joi.ref('password'))
          .required()
          .messages({
            'any.only': 'Passwords must match'
          }),
        firstName: Joi.string()
          .min(1)
          .max(50)
          .pattern(/^[a-zA-Z\s-']+$/)
          .required(),
        lastName: Joi.string()
          .min(1)
          .max(50)
          .pattern(/^[a-zA-Z\s-']+$/)
          .required()
      }),

      // API key management
      apiKey: Joi.object({
        name: Joi.string()
          .min(3)
          .max(100)
          .pattern(/^[a-zA-Z0-9\s_-]+$/)
          .required(),
        description: Joi.string()
          .max(500)
          .optional(),
        permissions: Joi.array()
          .items(Joi.string().valid('read', 'write', 'admin'))
          .min(1)
          .required(),
        expiresAt: Joi.date()
          .greater('now')
          .optional()
      }),

      // Media server configuration
      mediaConfig: Joi.object({
        serverName: Joi.string()
          .min(3)
          .max(100)
          .pattern(/^[a-zA-Z0-9\s_-]+$/)
          .required(),
        libraryPath: Joi.string()
          .pattern(/^[a-zA-Z0-9\/\\_.-]+$/)
          .required(),
        transcoding: Joi.object({
          enabled: Joi.boolean().required(),
          quality: Joi.string().valid('low', 'medium', 'high', 'original').required(),
          hwAcceleration: Joi.boolean().default(false)
        }).required(),
        networkSettings: Joi.object({
          allowedIPs: Joi.array()
            .items(Joi.string().ip({ version: ['ipv4', 'ipv6'], cidr: 'optional' }))
            .required(),
          port: Joi.number().port().required(),
          ssl: Joi.boolean().default(false)
        }).required()
      }),

      // File upload validation
      fileUpload: Joi.object({
        filename: Joi.string()
          .max(255)
          .pattern(/^[a-zA-Z0-9._-]+$/)
          .required(),
        size: Joi.number()
          .positive()
          .max(100 * 1024 * 1024) // 100MB max
          .required(),
        mimeType: Joi.string()
          .valid(
            'video/mp4', 'video/mkv', 'video/avi', 'video/mov',
            'audio/mp3', 'audio/flac', 'audio/wav',
            'image/jpeg', 'image/png', 'image/gif',
            'application/pdf', 'text/plain'
          )
          .required()
      }),

      // Search and filtering
      search: Joi.object({
        query: Joi.string()
          .min(1)
          .max(200)
          .pattern(/^[a-zA-Z0-9\s.,!?'"()-]+$/)
          .required(),
        category: Joi.string()
          .valid('movies', 'tv', 'music', 'books', 'all')
          .default('all'),
        limit: Joi.number()
          .integer()
          .min(1)
          .max(100)
          .default(20),
        offset: Joi.number()
          .integer()
          .min(0)
          .default(0),
        sortBy: Joi.string()
          .valid('relevance', 'date', 'rating', 'title')
          .default('relevance'),
        sortOrder: Joi.string()
          .valid('asc', 'desc')
          .default('desc')
      }),

      // URL validation for webhooks, etc.
      webhook: Joi.object({
        url: Joi.string()
          .uri({ scheme: ['http', 'https'] })
          .required(),
        events: Joi.array()
          .items(Joi.string().valid('download.complete', 'media.added', 'user.login'))
          .min(1)
          .required(),
        secret: Joi.string()
          .min(16)
          .max(128)
          .optional()
      })
    };
  }

  /**
   * Validate data against a specific schema
   */
  validate(data, schemaName) {
    const schema = this.schemas[schemaName];
    if (!schema) {
      throw new Error(`Unknown validation schema: ${schemaName}`);
    }

    const { error, value } = schema.validate(data, {
      abortEarly: false,
      stripUnknown: true,
      convert: true
    });

    if (error) {
      const validationErrors = error.details.map(detail => ({
        field: detail.path.join('.'),
        message: detail.message,
        value: detail.context?.value
      }));

      return {
        isValid: false,
        errors: validationErrors,
        data: null
      };
    }

    return {
      isValid: true,
      errors: [],
      data: value
    };
  }

  /**
   * Sanitize HTML content to prevent XSS
   */
  sanitizeHtml(html, options = {}) {
    const customOptions = { ...this.xssOptions, ...options };
    return xss(html, customOptions);
  }

  /**
   * Sanitize plain text input
   */
  sanitizeText(text) {
    if (typeof text !== 'string') {
      return '';
    }
    
    // Escape HTML entities to prevent XSS
    return this.escape(text);
  }

  /**
   * Escape HTML entities for safe output
   */
  escape(input) {
    if (typeof input !== 'string') {
      return '';
    }
    
    return input
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#x27;')
      .replace(/\//g, '&#x2F;');
  }

  /**
   * Validate and sanitize email addresses
   */
  sanitizeEmail(email) {
    if (!email || typeof email !== 'string') {
      return null;
    }

    const sanitized = validator.normalizeEmail(email.trim().toLowerCase());
    return validator.isEmail(sanitized) ? sanitized : null;
  }

  /**
   * Validate and sanitize URLs
   */
  sanitizeUrl(url, allowedProtocols = ['http', 'https']) {
    if (!url || typeof url !== 'string') {
      return null;
    }

    const trimmed = url.trim();
    
    if (!validator.isURL(trimmed, { protocols: allowedProtocols })) {
      return null;
    }

    // Additional security checks
    const parsed = new URL(trimmed);
    
    // Block private IP ranges and localhost
    if (this.isPrivateIP(parsed.hostname)) {
      return null;
    }

    return trimmed;
  }

  /**
   * Check if hostname is a private IP or localhost
   */
  isPrivateIP(hostname) {
    const privateRanges = [
      /^127\./, // 127.0.0.0/8
      /^10\./, // 10.0.0.0/8
      /^172\.(1[6-9]|2[0-9]|3[01])\./, // 172.16.0.0/12
      /^192\.168\./, // 192.168.0.0/16
      /^169\.254\./, // 169.254.0.0/16 (link-local)
      /^::1$/, // IPv6 localhost
      /^fe80::/i, // IPv6 link-local
      /^fc00::/i, // IPv6 unique local
      /^fd00::/i // IPv6 unique local
    ];

    if (hostname === 'localhost') {
      return true;
    }

    return privateRanges.some(range => range.test(hostname));
  }

  /**
   * Sanitize file paths to prevent directory traversal
   */
  sanitizeFilePath(path) {
    if (!path || typeof path !== 'string') {
      return null;
    }

    // Remove null bytes and control characters
    let sanitized = path.replace(/\x00/g, '').replace(/[\x01-\x1F\x7F]/g, '');
    
    // Normalize path separators
    sanitized = sanitized.replace(/\\/g, '/');
    
    // Remove directory traversal attempts
    sanitized = sanitized.replace(/\.\./g, '');
    
    // Remove leading/trailing slashes and whitespace
    sanitized = sanitized.replace(/^\/+|\/+$/g, '').trim();
    
    // Only allow safe characters
    if (!/^[a-zA-Z0-9._/-]+$/.test(sanitized)) {
      return null;
    }

    return sanitized;
  }

  /**
   * Validate SQL query parameters to prevent injection
   */
  sanitizeSqlParam(param) {
    if (param === null || param === undefined) {
      return null;
    }

    if (typeof param === 'number') {
      return isFinite(param) ? param : null;
    }

    if (typeof param === 'boolean') {
      return param;
    }

    if (typeof param === 'string') {
      // Remove SQL injection patterns
      const dangerous = /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b|[';\\x00\\n\\r\\\\])/gi;
      
      if (dangerous.test(param)) {
        return null;
      }

      return param.trim().substring(0, 1000); // Limit length
    }

    return null;
  }

  /**
   * Validate JSON input
   */
  sanitizeJson(jsonString, maxDepth = 10, maxKeys = 100) {
    try {
      if (typeof jsonString !== 'string') {
        return null;
      }

      const parsed = JSON.parse(jsonString);
      
      // Check depth and key count to prevent DoS
      if (this.getObjectDepth(parsed) > maxDepth) {
        throw new Error('JSON object too deep');
      }

      if (this.getObjectKeyCount(parsed) > maxKeys) {
        throw new Error('Too many keys in JSON object');
      }

      return parsed;
    } catch (error) {
      return null;
    }
  }

  /**
   * Get object depth recursively
   */
  getObjectDepth(obj, depth = 0) {
    if (depth > 50) return depth; // Prevent stack overflow
    
    if (obj && typeof obj === 'object') {
      return 1 + Math.max(0, ...Object.values(obj).map(v => this.getObjectDepth(v, depth + 1)));
    }
    return 0;
  }

  /**
   * Count total keys in nested object
   */
  getObjectKeyCount(obj, count = 0) {
    if (count > 1000) return count; // Prevent excessive counting
    
    if (obj && typeof obj === 'object') {
      count += Object.keys(obj).length;
      for (const value of Object.values(obj)) {
        if (typeof value === 'object' && value !== null) {
          count = this.getObjectKeyCount(value, count);
        }
      }
    }
    return count;
  }

  /**
   * Express middleware for input validation
   */
  middleware(schemaName) {
    return (req, res, next) => {
      const result = this.validate(req.body, schemaName);
      
      if (!result.isValid) {
        return res.status(400).json({
          error: 'Validation failed',
          details: result.errors
        });
      }

      // Replace req.body with sanitized data
      req.body = result.data;
      req.validationResult = result;
      
      next();
    };
  }

  /**
   * Sanitize all string values in an object recursively
   */
  deepSanitize(obj, maxDepth = 10, currentDepth = 0) {
    if (currentDepth >= maxDepth) {
      return obj;
    }

    if (typeof obj === 'string') {
      return this.sanitizeText(obj);
    }

    if (Array.isArray(obj)) {
      return obj.map(item => this.deepSanitize(item, maxDepth, currentDepth + 1));
    }

    if (obj && typeof obj === 'object') {
      const sanitized = {};
      for (const [key, value] of Object.entries(obj)) {
        const sanitizedKey = this.sanitizeText(key);
        sanitized[sanitizedKey] = this.deepSanitize(value, maxDepth, currentDepth + 1);
      }
      return sanitized;
    }

    return obj;
  }
}

module.exports = InputValidator;