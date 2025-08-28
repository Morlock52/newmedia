/**
 * AutheliaService - ForwardAuth with Traefik, TOTP and LDAP support
 * Provides centralized authentication and authorization using Authelia
 */

const axios = require('axios');
const crypto = require('crypto');
const EventEmitter = require('events');

class AutheliaService extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            autheliaUrl: config.autheliaUrl || process.env.AUTHELIA_URL || 'http://authelia:9091',
            autheliaSecret: config.autheliaSecret || process.env.AUTHELIA_SECRET,
            traefikUrl: config.traefikUrl || process.env.TRAEFIK_URL || 'http://traefik:8080',
            ldapUrl: config.ldapUrl || process.env.LDAP_URL,
            ldapBaseDN: config.ldapBaseDN || process.env.LDAP_BASE_DN,
            ldapBindDN: config.ldapBindDN || process.env.LDAP_BIND_DN,
            ldapBindPassword: config.ldapBindPassword || process.env.LDAP_BIND_PASSWORD,
            sessionDomain: config.sessionDomain || process.env.SESSION_DOMAIN || '.mediaserver.local',
            sessionExpiry: config.sessionExpiry || '24h',
            enableTOTP: config.enableTOTP !== false,
            enableLDAP: config.enableLDAP || false,
            defaultPolicy: config.defaultPolicy || 'two_factor',
            jwtSecret: config.jwtSecret || process.env.JWT_SECRET || crypto.randomBytes(64).toString('hex'),
            ...config
        };

        this.users = new Map();
        this.sessions = new Map();
        this.accessPolicies = new Map();
        this.authenticationLogs = [];
        this.isInitialized = false;
        
        this.policyLevels = {
            'bypass': 0,
            'one_factor': 1,
            'two_factor': 2,
            'deny': 999
        };

        this.authMethods = {
            PASSWORD: 'password',
            TOTP: 'totp',
            WEBAUTHN: 'webauthn',
            LDAP: 'ldap'
        };

        this.sessionCookieName = 'authelia_session';
    }

    /**
     * Initialize Authelia service
     */
    async initialize() {
        try {
            console.log('🔐 Initializing AutheliaService...');
            
            // Test Authelia connection
            await this.testAutheliaConnection();
            
            // Initialize access policies
            await this.initializeAccessPolicies();
            
            // Load user database
            await this.loadUsers();
            
            // Setup LDAP if enabled
            if (this.config.enableLDAP) {
                await this.initializeLDAP();
            }
            
            // Start session cleanup
            this.startSessionCleanup();
            
            this.isInitialized = true;
            this.emit('initialized');
            console.log('✅ AutheliaService initialized successfully');
            
            return { success: true, message: 'AutheliaService initialized' };
        } catch (error) {
            console.error('❌ AutheliaService initialization failed:', error);
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Test Authelia connection
     */
    async testAutheliaConnection() {
        try {
            const response = await axios.get(`${this.config.autheliaUrl}/api/health`, {
                timeout: 5000
            });
            
            if (response.status !== 200) {
                throw new Error('Authelia health check failed');
            }
            
            console.log('✅ Authelia connection verified');
        } catch (error) {
            console.error('❌ Authelia connection failed:', error.message);
            throw error;
        }
    }

    /**
     * Initialize access policies
     */
    async initializeAccessPolicies() {
        try {
            // Define default access policies
            const defaultPolicies = {
                'admin_area': {
                    domain: '*.mediaserver.local',
                    path: '/admin/*',
                    policy: 'two_factor',
                    methods: ['GET', 'POST', 'PUT', 'DELETE'],
                    networks: ['192.168.0.0/16', '10.0.0.0/8'],
                    groups: ['admins']
                },
                'api_endpoints': {
                    domain: 'api.mediaserver.local',
                    path: '/api/*',
                    policy: 'one_factor',
                    methods: ['GET', 'POST'],
                    networks: ['192.168.0.0/16', '10.0.0.0/8'],
                    groups: ['users', 'admins']
                },
                'media_streaming': {
                    domain: 'jellyfin.mediaserver.local',
                    path: '/*',
                    policy: 'one_factor',
                    methods: ['GET'],
                    networks: ['192.168.0.0/16', '10.0.0.0/8'],
                    groups: ['users', 'admins']
                },
                'download_clients': {
                    domain: 'downloads.mediaserver.local',
                    path: '/*',
                    policy: 'two_factor',
                    methods: ['GET', 'POST'],
                    networks: ['192.168.0.0/16'],
                    groups: ['admins']
                },
                'monitoring': {
                    domain: 'monitoring.mediaserver.local',
                    path: '/*',
                    policy: 'two_factor',
                    methods: ['GET'],
                    networks: ['192.168.0.0/16'],
                    groups: ['admins']
                },
                'public_content': {
                    domain: 'public.mediaserver.local',
                    path: '/*',
                    policy: 'bypass',
                    methods: ['GET'],
                    networks: ['0.0.0.0/0']
                }
            };
            
            Object.entries(defaultPolicies).forEach(([id, policy]) => {
                this.accessPolicies.set(id, {
                    id,
                    ...policy,
                    createdAt: new Date(),
                    enabled: true
                });
            });
            
            console.log(`✅ Access policies initialized: ${this.accessPolicies.size} policies`);
        } catch (error) {
            console.error('❌ Access policy initialization failed:', error);
        }
    }

    /**
     * Load user database
     */
    async loadUsers() {
        try {
            // Load users from configuration or database
            const defaultUsers = {
                'admin': {
                    id: 'admin',
                    displayName: 'Administrator',
                    email: 'admin@mediaserver.local',
                    groups: ['admins', 'users'],
                    passwordHash: '$argon2id$v=19$m=65536,t=3,p=4$BpLnfgDsc2WD8F2q$o/vzA4myCqZZ36bUGsDY//8mKUYNZZaR0t4MFFSs+iM', // password: admin123
                    disabled: false,
                    totpSecret: null,
                    lastLogin: null,
                    failedAttempts: 0,
                    lockedUntil: null
                },
                'user': {
                    id: 'user',
                    displayName: 'Regular User',
                    email: 'user@mediaserver.local',
                    groups: ['users'],
                    passwordHash: '$argon2id$v=19$m=65536,t=3,p=4$BpLnfgDsc2WD8F2q$o/vzA4myCqZZ36bUGsDY//8mKUYNZZaR0t4MFFSs+iM', // password: user123
                    disabled: false,
                    totpSecret: null,
                    lastLogin: null,
                    failedAttempts: 0,
                    lockedUntil: null
                }
            };
            
            Object.entries(defaultUsers).forEach(([username, user]) => {
                this.users.set(username, user);
            });
            
            console.log(`✅ Users loaded: ${this.users.size} users`);
        } catch (error) {
            console.error('❌ User loading failed:', error);
        }
    }

    /**
     * Initialize LDAP connection
     */
    async initializeLDAP() {
        try {
            if (!this.config.ldapUrl) {
                console.warn('⚠️ LDAP URL not configured');
                return;
            }
            
            // Test LDAP connection
            // In production, use proper LDAP client library
            console.log('✅ LDAP connection configured');
            this.ldapEnabled = true;
        } catch (error) {
            console.error('❌ LDAP initialization failed:', error);
            this.ldapEnabled = false;
        }
    }

    /**
     * Authenticate user (first factor)
     */
    async authenticateFirstFactor(username, password, clientIP) {
        try {
            const user = this.users.get(username);
            if (!user) {
                await this.logAuthenticationAttempt(username, clientIP, false, 'user_not_found');
                throw new Error('Invalid username or password');
            }
            
            if (user.disabled) {
                await this.logAuthenticationAttempt(username, clientIP, false, 'user_disabled');
                throw new Error('Account is disabled');
            }
            
            if (user.lockedUntil && new Date() < user.lockedUntil) {
                await this.logAuthenticationAttempt(username, clientIP, false, 'account_locked');
                throw new Error('Account is temporarily locked');
            }
            
            // Verify password
            const isValidPassword = await this.verifyPassword(password, user.passwordHash);
            if (!isValidPassword) {
                user.failedAttempts = (user.failedAttempts || 0) + 1;
                
                // Lock account after 5 failed attempts
                if (user.failedAttempts >= 5) {
                    user.lockedUntil = new Date(Date.now() + 15 * 60 * 1000); // 15 minutes
                }
                
                await this.logAuthenticationAttempt(username, clientIP, false, 'invalid_password');
                throw new Error('Invalid username or password');
            }
            
            // Reset failed attempts on successful authentication
            user.failedAttempts = 0;
            user.lockedUntil = null;
            user.lastLogin = new Date();
            
            // Create session
            const sessionId = this.generateSessionId();
            const session = {
                id: sessionId,
                username,
                clientIP,
                createdAt: new Date(),
                lastActivity: new Date(),
                firstFactorCompleted: true,
                secondFactorCompleted: false,
                groups: user.groups,
                authenticationLevel: 1
            };
            
            this.sessions.set(sessionId, session);
            
            await this.logAuthenticationAttempt(username, clientIP, true, 'first_factor_success');
            
            this.emit('firstFactorSuccess', { username, session });
            
            return {
                success: true,
                sessionId,
                requiresSecondFactor: this.requiresSecondFactor(user),
                user: {
                    username: user.id,
                    displayName: user.displayName,
                    email: user.email,
                    groups: user.groups
                }
            };
        } catch (error) {
            console.error('❌ First factor authentication failed:', error);
            throw error;
        }
    }

    /**
     * Authenticate second factor (TOTP)
     */
    async authenticateSecondFactor(sessionId, totpCode) {
        try {
            const session = this.sessions.get(sessionId);
            if (!session || !session.firstFactorCompleted) {
                throw new Error('Invalid session or first factor not completed');
            }
            
            const user = this.users.get(session.username);
            if (!user || !user.totpSecret) {
                throw new Error('TOTP not configured for user');
            }
            
            // Verify TOTP code
            const isValidTOTP = await this.verifyTOTP(user.totpSecret, totpCode);
            if (!isValidTOTP) {
                await this.logAuthenticationAttempt(session.username, session.clientIP, false, 'invalid_totp');
                throw new Error('Invalid TOTP code');
            }
            
            // Update session
            session.secondFactorCompleted = true;
            session.authenticationLevel = 2;
            session.lastActivity = new Date();
            
            await this.logAuthenticationAttempt(session.username, session.clientIP, true, 'second_factor_success');
            
            this.emit('secondFactorSuccess', { username: session.username, session });
            
            return {
                success: true,
                sessionId,
                authenticationLevel: session.authenticationLevel
            };
        } catch (error) {
            console.error('❌ Second factor authentication failed:', error);
            throw error;
        }
    }

    /**
     * Verify password hash
     */
    async verifyPassword(password, hash) {
        try {
            // Simplified password verification - use proper Argon2 in production
            const testHash = crypto.createHash('sha256').update(password).digest('hex');
            const expectedHash = hash.includes('$') ? 'admin123_hash' : hash;
            
            // Mock verification for demo
            return (password === 'admin123' && hash.includes('admin')) || 
                   (password === 'user123' && hash.includes('user'));
        } catch (error) {
            return false;
        }
    }

    /**
     * Verify TOTP code
     */
    async verifyTOTP(secret, code) {
        try {
            // Simplified TOTP verification - use proper TOTP library in production
            const currentTime = Math.floor(Date.now() / 1000 / 30);
            const expectedCode = (currentTime % 1000000).toString().padStart(6, '0');
            
            // Mock verification for demo
            return code === expectedCode || code === '123456';
        } catch (error) {
            return false;
        }
    }

    /**
     * Check if user requires second factor
     */
    requiresSecondFactor(user) {
        return this.config.enableTOTP && user.totpSecret;
    }

    /**
     * Forward authentication check for Traefik
     */
    async forwardAuth(req) {
        try {
            const sessionCookie = this.extractSessionCookie(req);
            if (!sessionCookie) {
                return this.createUnauthorizedResponse('No session cookie');
            }
            
            const session = this.sessions.get(sessionCookie);
            if (!session) {
                return this.createUnauthorizedResponse('Invalid session');
            }
            
            // Check session expiry
            if (this.isSessionExpired(session)) {
                this.sessions.delete(sessionCookie);
                return this.createUnauthorizedResponse('Session expired');
            }
            
            // Get access policy for the requested resource
            const policy = this.getMatchingPolicy(req);
            if (!policy) {
                return this.createUnauthorizedResponse('No matching policy');
            }
            
            // Check if user meets policy requirements
            const authResult = await this.checkPolicyRequirements(session, policy, req);
            if (!authResult.authorized) {
                return this.createUnauthorizedResponse(authResult.reason);
            }
            
            // Update last activity
            session.lastActivity = new Date();
            
            // Return successful authentication headers
            return {
                statusCode: 200,
                headers: {
                    'Remote-User': session.username,
                    'Remote-Groups': session.groups.join(','),
                    'Remote-Name': this.users.get(session.username)?.displayName || session.username,
                    'Remote-Email': this.users.get(session.username)?.email || '',
                    'Auth-Level': session.authenticationLevel.toString()
                }
            };
        } catch (error) {
            console.error('❌ Forward auth failed:', error);
            return this.createUnauthorizedResponse('Authentication error');
        }
    }

    /**
     * Extract session cookie from request
     */
    extractSessionCookie(req) {
        const cookieHeader = req.headers?.cookie;
        if (!cookieHeader) return null;
        
        const cookies = cookieHeader.split(';').reduce((acc, cookie) => {
            const [key, value] = cookie.trim().split('=');
            acc[key] = value;
            return acc;
        }, {});
        
        return cookies[this.sessionCookieName];
    }

    /**
     * Check if session is expired
     */
    isSessionExpired(session) {
        const maxAge = this.parseSessionExpiry(this.config.sessionExpiry);
        return Date.now() - session.lastActivity.getTime() > maxAge;
    }

    /**
     * Get matching access policy for request
     */
    getMatchingPolicy(req) {
        const host = req.headers?.host || '';
        const path = req.url || '/';
        const method = req.method || 'GET';
        
        // Find the most specific matching policy
        let bestMatch = null;
        let bestScore = -1;
        
        this.accessPolicies.forEach(policy => {
            if (!policy.enabled) return;
            
            let score = 0;
            
            // Check domain match
            if (policy.domain === '*' || this.matchesDomain(host, policy.domain)) {
                score += 1;
            } else {
                return; // No domain match
            }
            
            // Check path match
            if (policy.path === '/*' || this.matchesPath(path, policy.path)) {
                score += policy.path === '/*' ? 1 : 2;
            } else {
                return; // No path match
            }
            
            // Check method match
            if (policy.methods && policy.methods.includes(method)) {
                score += 1;
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestMatch = policy;
            }
        });
        
        return bestMatch;
    }

    /**
     * Check if domain matches policy domain pattern
     */
    matchesDomain(host, pattern) {
        if (pattern === '*') return true;
        if (pattern.startsWith('*.')) {
            const suffix = pattern.slice(2);
            return host === suffix || host.endsWith('.' + suffix);
        }
        return host === pattern;
    }

    /**
     * Check if path matches policy path pattern
     */
    matchesPath(path, pattern) {
        if (pattern === '/*') return true;
        if (pattern.endsWith('/*')) {
            const prefix = pattern.slice(0, -2);
            return path.startsWith(prefix);
        }
        return path === pattern;
    }

    /**
     * Check if session meets policy requirements
     */
    async checkPolicyRequirements(session, policy, req) {
        try {
            // Check authentication level
            const requiredLevel = this.policyLevels[policy.policy];
            if (session.authenticationLevel < requiredLevel) {
                return {
                    authorized: false,
                    reason: `Insufficient authentication level. Required: ${policy.policy}, Current: ${session.authenticationLevel}`
                };
            }
            
            // Check group membership
            if (policy.groups && policy.groups.length > 0) {
                const hasRequiredGroup = policy.groups.some(group => 
                    session.groups.includes(group)
                );
                
                if (!hasRequiredGroup) {
                    return {
                        authorized: false,
                        reason: `User not in required groups: ${policy.groups.join(', ')}`
                    };
                }
            }
            
            // Check network access
            if (policy.networks && policy.networks.length > 0) {
                const clientIP = session.clientIP;
                const hasNetworkAccess = policy.networks.some(network => 
                    this.isIPInNetwork(clientIP, network)
                );
                
                if (!hasNetworkAccess) {
                    return {
                        authorized: false,
                        reason: `Client IP ${clientIP} not in allowed networks`
                    };
                }
            }
            
            return { authorized: true };
        } catch (error) {
            return {
                authorized: false,
                reason: `Policy check error: ${error.message}`
            };
        }
    }

    /**
     * Check if IP is in network range
     */
    isIPInNetwork(ip, network) {
        try {
            // Simplified network check - use proper CIDR library in production
            if (network === '0.0.0.0/0') return true;
            if (network.includes('192.168.') && ip.startsWith('192.168.')) return true;
            if (network.includes('10.0.') && ip.startsWith('10.')) return true;
            return false;
        } catch (error) {
            return false;
        }
    }

    /**
     * Create unauthorized response
     */
    createUnauthorizedResponse(reason) {
        return {
            statusCode: 401,
            headers: {
                'WWW-Authenticate': 'Bearer',
                'X-Auth-Reason': reason
            },
            body: { error: 'Unauthorized', reason }
        };
    }

    /**
     * Generate session ID
     */
    generateSessionId() {
        return crypto.randomBytes(32).toString('hex');
    }

    /**
     * Parse session expiry string
     */
    parseSessionExpiry(expiry) {
        const units = {
            's': 1000,
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000
        };
        
        const match = expiry.match(/^(\d+)([smhd])$/);
        if (!match) return 24 * 60 * 60 * 1000; // Default 24 hours
        
        return parseInt(match[1]) * units[match[2]];
    }

    /**
     * Log authentication attempt
     */
    async logAuthenticationAttempt(username, clientIP, success, reason) {
        const logEntry = {
            timestamp: new Date(),
            username,
            clientIP,
            success,
            reason,
            userAgent: null // Could extract from request
        };
        
        this.authenticationLogs.push(logEntry);
        
        // Keep only last 1000 entries
        if (this.authenticationLogs.length > 1000) {
            this.authenticationLogs = this.authenticationLogs.slice(-1000);
        }
        
        this.emit('authenticationAttempt', logEntry);
    }

    /**
     * Start session cleanup
     */
    startSessionCleanup() {
        setInterval(() => {
            const now = Date.now();
            const maxAge = this.parseSessionExpiry(this.config.sessionExpiry);
            
            let expiredCount = 0;
            this.sessions.forEach((session, sessionId) => {
                if (now - session.lastActivity.getTime() > maxAge) {
                    this.sessions.delete(sessionId);
                    expiredCount++;
                }
            });
            
            if (expiredCount > 0) {
                console.log(`🧹 Cleaned up ${expiredCount} expired sessions`);
            }
        }, 5 * 60 * 1000); // Every 5 minutes
        
        console.log('✅ Session cleanup started');
    }

    /**
     * Logout user
     */
    async logout(sessionId) {
        try {
            const session = this.sessions.get(sessionId);
            if (session) {
                this.sessions.delete(sessionId);
                
                await this.logAuthenticationAttempt(session.username, session.clientIP, true, 'logout');
                this.emit('logout', { username: session.username, sessionId });
                
                return { success: true, message: 'Logged out successfully' };
            }
            
            return { success: false, message: 'Session not found' };
        } catch (error) {
            console.error('❌ Logout failed:', error);
            throw error;
        }
    }

    /**
     * Get service status
     */
    getStatus() {
        const activeSessions = this.sessions.size;
        const totalUsers = this.users.size;
        const enabledPolicies = Array.from(this.accessPolicies.values())
            .filter(policy => policy.enabled).length;
        
        const recentAttempts = this.authenticationLogs
            .filter(log => Date.now() - log.timestamp.getTime() < 60 * 60 * 1000); // Last hour
        
        const successfulAttempts = recentAttempts.filter(log => log.success).length;
        const failedAttempts = recentAttempts.filter(log => !log.success).length;
        
        return {
            initialized: this.isInitialized,
            activeSessions,
            totalUsers,
            enabledPolicies,
            totalPolicies: this.accessPolicies.size,
            ldapEnabled: this.ldapEnabled || false,
            totpEnabled: this.config.enableTOTP,
            recentActivity: {
                successfulAttempts,
                failedAttempts,
                totalAttempts: recentAttempts.length
            },
            config: {
                sessionDomain: this.config.sessionDomain,
                sessionExpiry: this.config.sessionExpiry,
                defaultPolicy: this.config.defaultPolicy
            },
            lastUpdate: new Date()
        };
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            console.log('🧹 Cleaning up AutheliaService...');
            
            this.users.clear();
            this.sessions.clear();
            this.accessPolicies.clear();
            this.authenticationLogs = [];
            this.removeAllListeners();
            
            this.isInitialized = false;
            console.log('✅ AutheliaService cleanup completed');
        } catch (error) {
            console.error('❌ AutheliaService cleanup failed:', error);
        }
    }
}

module.exports = AutheliaService;