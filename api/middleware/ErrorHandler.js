/**
 * Error Handler Middleware
 * Centralized error handling and logging
 */

class ErrorHandler {
    static handleError(error, req, res, next) {
        // Log the error
        console.error('API Error:', {
            message: error.message,
            stack: error.stack,
            url: req.url,
            method: req.method,
            ip: req.ip,
            userAgent: req.get('User-Agent'),
            timestamp: new Date().toISOString()
        });

        // Determine error type and response
        let statusCode = 500;
        let errorResponse = {
            success: false,
            error: 'Internal server error',
            timestamp: new Date().toISOString()
        };

        // Handle specific error types
        if (error.name === 'ValidationError') {
            statusCode = 400;
            errorResponse.error = 'Validation error';
            errorResponse.details = error.details;
        } else if (error.name === 'UnauthorizedError') {
            statusCode = 401;
            errorResponse.error = 'Unauthorized';
        } else if (error.name === 'ForbiddenError') {
            statusCode = 403;
            errorResponse.error = 'Forbidden';
        } else if (error.name === 'NotFoundError') {
            statusCode = 404;
            errorResponse.error = 'Not found';
        } else if (error.name === 'ConflictError') {
            statusCode = 409;
            errorResponse.error = 'Conflict';
        } else if (error.name === 'TooManyRequestsError') {
            statusCode = 429;
            errorResponse.error = 'Too many requests';
        } else if (error.message.includes('ECONNREFUSED')) {
            statusCode = 503;
            errorResponse.error = 'Service unavailable';
            errorResponse.details = 'Unable to connect to required service';
        } else if (error.message.includes('timeout')) {
            statusCode = 504;
            errorResponse.error = 'Gateway timeout';
            errorResponse.details = 'Operation timed out';
        }

        // Add error ID for tracking
        errorResponse.errorId = ErrorHandler.generateErrorId();

        // Include additional details in development
        if (process.env.NODE_ENV === 'development') {
            errorResponse.details = error.message;
            errorResponse.stack = error.stack;
        }

        // Send response
        res.status(statusCode).json(errorResponse);
    }

    static generateErrorId() {
        return `err_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    }

    static asyncHandler(fn) {
        return (req, res, next) => {
            Promise.resolve(fn(req, res, next)).catch(next);
        };
    }

    // Custom error classes
    static ValidationError(message, details = null) {
        const error = new Error(message);
        error.name = 'ValidationError';
        error.details = details;
        return error;
    }

    static UnauthorizedError(message = 'Unauthorized') {
        const error = new Error(message);
        error.name = 'UnauthorizedError';
        return error;
    }

    static ForbiddenError(message = 'Forbidden') {
        const error = new Error(message);
        error.name = 'ForbiddenError';
        return error;
    }

    static NotFoundError(message = 'Not found') {
        const error = new Error(message);
        error.name = 'NotFoundError';
        return error;
    }

    static ConflictError(message = 'Conflict') {
        const error = new Error(message);
        error.name = 'ConflictError';
        return error;
    }

    static TooManyRequestsError(message = 'Too many requests') {
        const error = new Error(message);
        error.name = 'TooManyRequestsError';
        return error;
    }
}

module.exports = ErrorHandler;