/**
 * API Validator Middleware
 * Request validation and sanitization
 */

const Joi = require('joi');

class APIValidator {
    static validateRequest(req, res, next) {
        // Skip validation for certain endpoints
        const skipValidation = [
            '/health',
            '/api/docs',
            '/api/logs/stream'
        ];

        if (skipValidation.includes(req.path)) {
            return next();
        }

        // Validate API key for sensitive operations
        if (req.method !== 'GET' && req.path.startsWith('/api/')) {
            const apiKey = req.headers['x-api-key'];
            const validApiKey = process.env.API_KEY;

            if (validApiKey && apiKey !== validApiKey) {
                return res.status(401).json({
                    success: false,
                    error: 'Invalid or missing API key',
                    timestamp: new Date().toISOString()
                });
            }
        }

        // Validate request size
        const contentLength = parseInt(req.headers['content-length'] || '0');
        if (contentLength > 10 * 1024 * 1024) { // 10MB limit
            return res.status(413).json({
                success: false,
                error: 'Request entity too large',
                timestamp: new Date().toISOString()
            });
        }

        // Validate content type for POST/PUT requests
        if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
            const contentType = req.headers['content-type'];
            if (!contentType || !contentType.includes('application/json')) {
                return res.status(400).json({
                    success: false,
                    error: 'Content-Type must be application/json',
                    timestamp: new Date().toISOString()
                });
            }
        }

        // Sanitize and validate common parameters
        if (req.query) {
            req.query = APIValidator.sanitizeQuery(req.query);
        }

        next();
    }

    static sanitizeQuery(query) {
        const sanitized = {};

        for (const [key, value] of Object.entries(query)) {
            // Basic XSS prevention
            if (typeof value === 'string') {
                sanitized[key] = value
                    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
                    .replace(/javascript:/gi, '')
                    .replace(/on\w+=/gi, '')
                    .trim();
            } else {
                sanitized[key] = value;
            }
        }

        return sanitized;
    }

    static validateServiceRequest(req, res, next) {
        const schema = Joi.object({
            services: Joi.array().items(Joi.string().alphanum()).optional(),
            profile: Joi.string().valid('minimal', 'media', 'download', 'monitoring', 'full').optional()
        });

        const { error } = schema.validate(req.body);
        if (error) {
            return res.status(400).json({
                success: false,
                error: 'Validation error',
                details: error.details.map(detail => ({
                    field: detail.path.join('.'),
                    message: detail.message
                })),
                timestamp: new Date().toISOString()
            });
        }

        next();
    }

    static validateConfigRequest(req, res, next) {
        const schema = Joi.object({
            general: Joi.object().optional(),
            paths: Joi.object().optional(),
            authentication: Joi.object().optional(),
            apiKeys: Joi.object().optional(),
            cloudflare: Joi.object().optional(),
            monitoring: Joi.object().optional(),
            vpn: Joi.object().optional(),
            smtp: Joi.object().optional()
        });

        const { error } = schema.validate(req.body);
        if (error) {
            return res.status(400).json({
                success: false,
                error: 'Configuration validation error',
                details: error.details.map(detail => ({
                    field: detail.path.join('.'),
                    message: detail.message
                })),
                timestamp: new Date().toISOString()
            });
        }

        next();
    }

    static validateLogRequest(req, res, next) {
        const schema = Joi.object({
            level: Joi.string().valid('error', 'warn', 'info', 'debug', 'trace').optional(),
            service: Joi.string().alphanum().optional(),
            limit: Joi.number().integer().min(1).max(1000).optional(),
            since: Joi.date().iso().optional(),
            until: Joi.date().iso().optional(),
            search: Joi.string().max(100).optional()
        });

        const { error } = schema.validate(req.query);
        if (error) {
            return res.status(400).json({
                success: false,
                error: 'Log request validation error',
                details: error.details.map(detail => ({
                    field: detail.path.join('.'),
                    message: detail.message
                })),
                timestamp: new Date().toISOString()
            });
        }

        next();
    }
}

module.exports = APIValidator;