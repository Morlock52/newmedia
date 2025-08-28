/**
 * Authentication Middleware
 * JWT-based authentication and authorization
 */

const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const crypto = require('crypto');

class AuthMiddleware {
    constructor() {
        this.jwtSecret = process.env.JWT_SECRET || 'default-secret-change-in-production';
        this.jwtExpiry = process.env.JWT_EXPIRY || '24h';
        this.refreshTokens = new Map(); // In production, use Redis or database
        this.sessions = new Map(); // Active sessions
        this.rateLimitAttempts = new Map(); // Login rate limiting
        
        // Default admin user (in production, use database)
        this.users = new Map([
            ['admin', {
                id: '1',
                username: 'admin',
                email: 'admin@localhost',
                passwordHash: bcrypt.hashSync(process.env.ADMIN_PASSWORD || 'admin123', 12),
                role: 'admin',
                permissions: ['*'],
                createdAt: new Date().toISOString(),
                lastLogin: null,
                isActive: true
            }]
        ]);
        
        this.roles = {
            admin: ['*'], // All permissions
            user: ['read', 'services:view', 'logs:view'],
            operator: ['read', 'write', 'services:*', 'logs:view'],
            readonly: ['read', 'services:view', 'logs:view', 'health:view']
        };
    }

    // Generate JWT token
    generateToken(user, type = 'access') {
        const payload = {
            id: user.id,
            username: user.username,
            email: user.email,
            role: user.role,
            permissions: this.roles[user.role] || [],
            type,
            iat: Math.floor(Date.now() / 1000)
        };

        const options = {
            expiresIn: type === 'refresh' ? '7d' : this.jwtExpiry,
            issuer: 'media-server-api',
            audience: 'media-server-frontend'
        };

        return jwt.sign(payload, this.jwtSecret, options);
    }

    // Generate refresh token
    generateRefreshToken(userId) {
        const refreshToken = crypto.randomBytes(64).toString('hex');
        const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000); // 7 days
        
        this.refreshTokens.set(refreshToken, {
            userId,
            expiresAt,
            createdAt: new Date()
        });

        return refreshToken;
    }

    // Verify JWT token
    verifyToken(token) {
        try {
            const decoded = jwt.verify(token, this.jwtSecret);
            
            // Check if user is still active
            const user = Array.from(this.users.values()).find(u => u.id === decoded.id);
            if (!user || !user.isActive) {
                throw new Error('User not found or inactive');
            }

            return decoded;
        } catch (error) {
            throw new Error('Invalid token: ' + error.message);
        }
    }

    // Authentication middleware
    authenticate(req, res, next) {
        try {
            const authHeader = req.headers.authorization;
            const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

            if (!token) {
                return res.status(401).json({
                    success: false,
                    error: 'Access token required',
                    code: 'TOKEN_REQUIRED'
                });
            }

            const decoded = AuthMiddleware.getInstance().verifyToken(token);
            req.user = decoded;
            req.sessionId = AuthMiddleware.getInstance().createSession(decoded);
            
            next();
        } catch (error) {
            return res.status(401).json({
                success: false,
                error: 'Invalid or expired token',
                code: 'INVALID_TOKEN',
                details: error.message
            });
        }
    }

    // Authorization middleware
    authorize(permissions = []) {
        return (req, res, next) => {
            if (!req.user) {
                return res.status(401).json({
                    success: false,
                    error: 'Authentication required',
                    code: 'AUTHENTICATION_REQUIRED'
                });
            }

            const userPermissions = req.user.permissions || [];
            
            // Admin has all permissions
            if (userPermissions.includes('*')) {
                return next();
            }

            // Check specific permissions
            const hasPermission = permissions.some(permission => 
                userPermissions.includes(permission) ||
                userPermissions.some(userPerm => 
                    userPerm.endsWith('*') && 
                    permission.startsWith(userPerm.slice(0, -1))
                )
            );

            if (!hasPermission) {
                return res.status(403).json({
                    success: false,
                    error: 'Insufficient permissions',
                    code: 'INSUFFICIENT_PERMISSIONS',
                    required: permissions,
                    available: userPermissions
                });
            }

            next();
        };
    }

    // Login endpoint
    async login(req, res) {
        try {
            const { username, password, rememberMe = false } = req.body;

            if (!username || !password) {
                return res.status(400).json({
                    success: false,
                    error: 'Username and password are required',
                    code: 'CREDENTIALS_REQUIRED'
                });
            }

            // Rate limiting
            const clientId = req.ip;
            const attempts = this.rateLimitAttempts.get(clientId) || { count: 0, lastAttempt: 0 };
            
            if (attempts.count >= 5 && Date.now() - attempts.lastAttempt < 15 * 60 * 1000) {
                return res.status(429).json({
                    success: false,
                    error: 'Too many login attempts. Try again in 15 minutes.',
                    code: 'RATE_LIMIT_EXCEEDED'
                });
            }

            // Find user
            const user = this.users.get(username.toLowerCase());
            if (!user) {
                this.recordFailedAttempt(clientId);
                return res.status(401).json({
                    success: false,
                    error: 'Invalid credentials',
                    code: 'INVALID_CREDENTIALS'
                });
            }

            // Verify password
            const isValidPassword = await bcrypt.compare(password, user.passwordHash);
            if (!isValidPassword) {
                this.recordFailedAttempt(clientId);
                return res.status(401).json({
                    success: false,
                    error: 'Invalid credentials',
                    code: 'INVALID_CREDENTIALS'
                });
            }

            // Check if user is active
            if (!user.isActive) {
                return res.status(403).json({
                    success: false,
                    error: 'Account is disabled',
                    code: 'ACCOUNT_DISABLED'
                });
            }

            // Clear failed attempts
            this.rateLimitAttempts.delete(clientId);

            // Generate tokens
            const accessToken = this.generateToken(user);
            const refreshToken = this.generateRefreshToken(user.id);
            
            // Update last login
            user.lastLogin = new Date().toISOString();
            
            // Create session
            const sessionId = this.createSession({
                id: user.id,
                username: user.username,
                role: user.role,
                loginTime: new Date().toISOString(),
                ip: req.ip,
                userAgent: req.get('User-Agent')
            });

            res.json({
                success: true,
                data: {
                    accessToken,
                    refreshToken,
                    sessionId,
                    user: {
                        id: user.id,
                        username: user.username,
                        email: user.email,
                        role: user.role,
                        permissions: this.roles[user.role],
                        lastLogin: user.lastLogin
                    },
                    expiresIn: this.jwtExpiry
                }
            });
        } catch (error) {
            res.status(500).json({
                success: false,
                error: 'Login failed',
                code: 'LOGIN_ERROR',
                details: error.message
            });
        }
    }

    // Refresh token endpoint
    async refreshToken(req, res) {
        try {
            const { refreshToken } = req.body;

            if (!refreshToken) {
                return res.status(400).json({
                    success: false,
                    error: 'Refresh token required',
                    code: 'REFRESH_TOKEN_REQUIRED'
                });
            }

            const tokenData = this.refreshTokens.get(refreshToken);
            if (!tokenData || tokenData.expiresAt < new Date()) {
                this.refreshTokens.delete(refreshToken);
                return res.status(401).json({
                    success: false,
                    error: 'Invalid or expired refresh token',
                    code: 'INVALID_REFRESH_TOKEN'
                });
            }

            // Find user
            const user = Array.from(this.users.values()).find(u => u.id === tokenData.userId);
            if (!user || !user.isActive) {
                return res.status(401).json({
                    success: false,
                    error: 'User not found or inactive',
                    code: 'USER_INACTIVE'
                });
            }

            // Generate new access token
            const newAccessToken = this.generateToken(user);
            
            // Optionally generate new refresh token (rotate refresh tokens)
            const newRefreshToken = this.generateRefreshToken(user.id);
            this.refreshTokens.delete(refreshToken);

            res.json({
                success: true,
                data: {
                    accessToken: newAccessToken,
                    refreshToken: newRefreshToken,
                    expiresIn: this.jwtExpiry
                }
            });
        } catch (error) {
            res.status(500).json({
                success: false,
                error: 'Token refresh failed',
                code: 'REFRESH_ERROR',
                details: error.message
            });
        }
    }

    // Logout endpoint
    logout(req, res) {
        try {
            const { refreshToken } = req.body;
            const sessionId = req.sessionId;

            // Remove refresh token
            if (refreshToken) {
                this.refreshTokens.delete(refreshToken);
            }

            // Remove session
            if (sessionId) {
                this.sessions.delete(sessionId);
            }

            res.json({
                success: true,
                message: 'Logged out successfully'
            });
        } catch (error) {
            res.status(500).json({
                success: false,
                error: 'Logout failed',
                details: error.message
            });
        }
    }

    // Get current user info
    getCurrentUser(req, res) {
        const user = Array.from(this.users.values()).find(u => u.id === req.user.id);
        
        if (!user) {
            return res.status(404).json({
                success: false,
                error: 'User not found',
                code: 'USER_NOT_FOUND'
            });
        }

        res.json({
            success: true,
            data: {
                id: user.id,
                username: user.username,
                email: user.email,
                role: user.role,
                permissions: this.roles[user.role],
                lastLogin: user.lastLogin,
                createdAt: user.createdAt
            }
        });
    }

    // Helper methods
    recordFailedAttempt(clientId) {
        const attempts = this.rateLimitAttempts.get(clientId) || { count: 0, lastAttempt: 0 };
        attempts.count++;
        attempts.lastAttempt = Date.now();
        this.rateLimitAttempts.set(clientId, attempts);
    }

    createSession(sessionData) {
        const sessionId = crypto.randomBytes(32).toString('hex');
        const session = {
            ...sessionData,
            createdAt: new Date().toISOString(),
            lastActivity: new Date().toISOString()
        };
        
        this.sessions.set(sessionId, session);
        
        // Clean up expired sessions
        setTimeout(() => {
            this.cleanupExpiredSessions();
        }, 60000); // Clean up every minute

        return sessionId;
    }

    cleanupExpiredSessions() {
        const now = new Date();
        const expiredTime = 24 * 60 * 60 * 1000; // 24 hours
        
        for (const [sessionId, session] of this.sessions.entries()) {
            if (now - new Date(session.lastActivity) > expiredTime) {
                this.sessions.delete(sessionId);
            }
        }

        // Also cleanup expired refresh tokens
        for (const [token, data] of this.refreshTokens.entries()) {
            if (data.expiresAt < now) {
                this.refreshTokens.delete(token);
            }
        }
    }

    // Singleton pattern
    static getInstance() {
        if (!AuthMiddleware.instance) {
            AuthMiddleware.instance = new AuthMiddleware();
        }
        return AuthMiddleware.instance;
    }
}

// Export both the class and singleton instance
module.exports = AuthMiddleware;
module.exports.auth = AuthMiddleware.getInstance();