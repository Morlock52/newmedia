const Joi = require('joi');

class APIValidator {
    static validate(schema) {
        return (req, res, next) => {
            const { error } = schema.validate(req.body);
            if (error) {
                return res.status(400).json({ error: error.details[0].message });
            }
            next();
        };
    }
    
    // Basic request validation middleware
    static validateRequest(req, res, next) {
        // Add basic request validation here if needed
        // For now, just pass through
        next();
    }
}

module.exports = APIValidator;
