const logger = require('../../middleware/logger.js');
class ErrorHandler {
    static handle(err, req, res, next) {
        logger.error(err.stack);
        res.status(err.status || 500).json({
            error: err.message || 'Internal Server Error'
        });
    }
    
    static handleError(err, req, res, next) {
        logger.error('Error:', err);
        
        const statusCode = err.statusCode || err.status || 500;
        const message = err.message || 'Internal Server Error';
        
        res.status(statusCode).json({
            success: false,
            error: message,
            code: err.code || 'INTERNAL_ERROR',
            timestamp: new Date().toISOString(),
            ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
        });
    }
}

module.exports = ErrorHandler;
