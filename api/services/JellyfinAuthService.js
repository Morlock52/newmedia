const logger = require('../../middleware/logger.js');
/**
 * Jellyfin Authentication Service
 * Handles secure authentication and API management for Jellyfin
 */

const axios = require('axios');
const crypto = require('crypto');
const EventEmitter = require('events');

class JellyfinAuthService extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            baseUrl: config.baseUrl || 'http://jellyfin:8096',
            timeout: config.timeout || 15000,
            maxRetries: config.maxRetries || 3,
            retryDelay: config.retryDelay || 1000,
            tokenRefreshThreshold: config.tokenRefreshThreshold || 300000, // 5 minutes
            ...config
        };
        
        // State management
        this.accessTokens = new Map(); // userId -> { token, expires, refreshToken }
        this.apiKeys = new Map(); // appName -> apiKey
        this.sessionCache = new Map();
        this.healthStatus = 'unknown';
        this.lastHealthCheck = null;
        
        // Initialize
        this.init();
    }

    /**
     * Initialize the service
     */
    async init() {
        logger.info('🔧 Initializing Jellyfin Authentication Service...');
        
        // Create HTTP client
        this.client = axios.create({
            baseURL: this.config.baseUrl,
            timeout: this.config.timeout,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'MediaDashboard/1.0.0'
            }
        });
        
        // Setup request/response interceptors
        this.setupInterceptors();
        
        // Start health monitoring
        this.startHealthMonitoring();
        
        logger.info('✅ Jellyfin Authentication Service initialized');
    }

    /**
     * Setup axios interceptors for request/response handling
     */
    setupInterceptors() {
        // Request interceptor
        this.client.interceptors.request.use(
            (config) => {
                // Add authorization header if token exists
                if (config.authToken) {
                    config.headers['X-Emby-Token'] = config.authToken;
                    delete config.authToken; // Remove from config
                }
                
                // Add device info
                config.headers['X-Emby-Authorization'] = this.generateAuthorizationHeader();
                
                return config;
            },
            (error) => {
                logger.error('❌ Request interceptor error:', error);
                return Promise.reject(error);
            }
        );

        // Response interceptor
        this.client.interceptors.response.use(
            (response) => {
                return response;
            },
            async (error) => {
                const originalRequest = error.config;
                
                // Handle authentication errors
                if (error.response?.status === 401 && !originalRequest._retry) {
                    originalRequest._retry = true;
                    
                    logger.info('🔄 Token expired, attempting refresh...');
                    this.emit('authError', { type: 'tokenExpired', error });
                    
                    // Could implement token refresh logic here if Jellyfin supported it
                    return Promise.reject(error);
                }
                
                // Handle rate limiting
                if (error.response?.status === 429) {
                    const retryAfter = error.response.headers['retry-after'] || 1;
                    logger.info(`⏳ Rate limited, retrying after ${retryAfter}s...`);
                    
                    await new Promise(resolve => setTimeout(resolve, retryAfter * 1000));
                    return this.client.request(originalRequest);
                }
                
                return Promise.reject(error);
            }
        );
    }

    /**
     * Generate authorization header
     */
    generateAuthorizationHeader() {
        const deviceId = crypto.randomUUID();
        const version = '1.0.0';
        
        return `MediaBrowser Client="MediaDashboard", Device="Dashboard", DeviceId="${deviceId}", Version="${version}"`;
    }

    /**
     * Authenticate user with username/password
     */
    async authenticateUser(username, password, rememberMe = false) {
        logger.info(`🔐 Authenticating user: ${username}`);
        
        try {
            const authData = {
                Username: username,
                Pw: password
            };
            
            const response = await this.client.post('/Users/authenticatebyname', authData);
            
            if (response.data && response.data.AccessToken) {
                const userData = response.data;
                const userId = userData.User.Id;
                
                // Store access token
                this.accessTokens.set(userId, {
                    token: userData.AccessToken,
                    user: userData.User,
                    sessionInfo: userData.SessionInfo,
                    serverId: userData.ServerId,
                    created: Date.now()
                });
                
                logger.info(`✅ User authenticated successfully: ${userData.User.Name}`);
                this.emit('userAuthenticated', userData.User);
                
                return {
                    success: true,
                    user: userData.User,
                    token: userData.AccessToken,
                    sessionInfo: userData.SessionInfo
                };
            } else {
                throw new Error('Invalid authentication response');
            }
        } catch (error) {
            logger.error(`❌ Authentication failed for ${username}:`, error.message);
            this.emit('authError', { type: 'authFailed', username, error });
            
            return {
                success: false,
                error: error.message,
                code: error.response?.status
            };
        }
    }

    /**
     * Create API key for long-term access
     */
    async createAPIKey(userId, appName = 'MediaDashboard') {
        logger.info(`🔑 Creating API key for user ${userId}...`);
        
        const userAuth = this.accessTokens.get(userId);
        if (!userAuth) {
            throw new Error('User not authenticated');
        }
        
        try {
            const response = await this.client.post('/Auth/Keys', 
                { App: appName },
                { authToken: userAuth.token }
            );
            
            if (response.data && response.data.AccessToken) {
                const apiKey = response.data.AccessToken;
                
                // Store API key
                this.apiKeys.set(appName, {
                    key: apiKey,
                    userId,
                    created: Date.now(),
                    app: appName
                });
                
                logger.info(`✅ API key created for ${appName}`);
                this.emit('apiKeyCreated', { appName, userId });
                
                return {
                    success: true,
                    apiKey,
                    app: appName
                };
            } else {
                throw new Error('Invalid API key response');
            }
        } catch (error) {
            logger.error(`❌ API key creation failed:`, error.message);
            this.emit('authError', { type: 'apiKeyFailed', userId, error });
            
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Make authenticated request
     */
    async makeAuthenticatedRequest(method, endpoint, data = null, options = {}) {
        const { userId, apiKey, appName } = options;
        let authToken = null;
        
        // Determine authentication method
        if (apiKey) {
            authToken = apiKey;
        } else if (appName && this.apiKeys.has(appName)) {
            authToken = this.apiKeys.get(appName).key;
        } else if (userId && this.accessTokens.has(userId)) {
            authToken = this.accessTokens.get(userId).token;
        }
        
        if (!authToken) {
            throw new Error('No valid authentication token available');
        }
        
        const config = {
            method: method.toLowerCase(),
            url: endpoint,
            authToken,
            ...options
        };
        
        if (data) {
            config.data = data;
        }
        
        let lastError;
        
        // Retry logic
        for (let attempt = 1; attempt <= this.config.maxRetries; attempt++) {
            try {
                const response = await this.client.request(config);
                return response.data;
            } catch (error) {
                lastError = error;
                
                if (attempt < this.config.maxRetries && this.isRetryableError(error)) {
                    logger.info(`⚠️  Request failed (attempt ${attempt}), retrying...`);
                    await new Promise(resolve => setTimeout(resolve, this.config.retryDelay * attempt));
                } else {
                    break;
                }
            }
        }
        
        throw lastError;
    }

    /**
     * Check if error is retryable
     */
    isRetryableError(error) {
        const retryableCodes = [408, 429, 500, 502, 503, 504];
        return error.response && retryableCodes.includes(error.response.status);
    }

    /**
     * Get system information
     */
    async getSystemInfo(options = {}) {
        try {
            const data = await this.makeAuthenticatedRequest('GET', '/System/Info', null, options);
            return { success: true, data };
        } catch (error) {
            logger.error('❌ Failed to get system info:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get public system information (no auth required)
     */
    async getPublicSystemInfo() {
        try {
            const response = await this.client.get('/System/Info/Public');
            return { success: true, data: response.data };
        } catch (error) {
            logger.error('❌ Failed to get public system info:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get library statistics
     */
    async getLibraryStats(options = {}) {
        try {
            const data = await this.makeAuthenticatedRequest('GET', '/Items/Counts', null, options);
            return { success: true, data };
        } catch (error) {
            logger.error('❌ Failed to get library stats:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get active sessions
     */
    async getActiveSessions(options = {}) {
        try {
            const data = await this.makeAuthenticatedRequest('GET', '/Sessions', null, options);
            return { success: true, data: data || [] };
        } catch (error) {
            logger.error('❌ Failed to get active sessions:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get users
     */
    async getUsers(options = {}) {
        try {
            const data = await this.makeAuthenticatedRequest('GET', '/Users', null, options);
            return { success: true, data: data || [] };
        } catch (error) {
            logger.error('❌ Failed to get users:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Search content
     */
    async searchContent(query, limit = 20, options = {}) {
        try {
            const params = new URLSearchParams({
                searchTerm: query,
                limit: limit.toString(),
                recursive: 'true'
            });
            
            const data = await this.makeAuthenticatedRequest('GET', `/Items?${params}`, null, options);
            return { success: true, data: data?.Items || [] };
        } catch (error) {
            logger.error('❌ Failed to search content:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get server configuration
     */
    async getServerConfiguration(options = {}) {
        try {
            const data = await this.makeAuthenticatedRequest('GET', '/System/Configuration', null, options);
            return { success: true, data };
        } catch (error) {
            logger.error('❌ Failed to get server configuration:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const response = await this.client.get('/health', { timeout: 5000 });
            const isHealthy = response.status === 200;
            
            this.healthStatus = isHealthy ? 'healthy' : 'unhealthy';
            this.lastHealthCheck = new Date();
            
            this.emit('healthCheck', {
                status: this.healthStatus,
                timestamp: this.lastHealthCheck,
                response: response.data
            });
            
            return isHealthy;
        } catch (error) {
            this.healthStatus = 'unhealthy';
            this.lastHealthCheck = new Date();
            
            this.emit('healthCheck', {
                status: this.healthStatus,
                timestamp: this.lastHealthCheck,
                error: error.message
            });
            
            return false;
        }
    }

    /**
     * Start health monitoring
     */
    startHealthMonitoring(interval = 30000) {
        if (this.healthInterval) {
            clearInterval(this.healthInterval);
        }
        
        this.healthInterval = setInterval(() => {
            this.healthCheck();
        }, interval);
        
        logger.info(`🏥 Health monitoring started (${interval}ms interval)`);
    }

    /**
     * Stop health monitoring
     */
    stopHealthMonitoring() {
        if (this.healthInterval) {
            clearInterval(this.healthInterval);
            this.healthInterval = null;
            logger.info('🏥 Health monitoring stopped');
        }
    }

    /**
     * Get comprehensive dashboard data
     */
    async getDashboardData(options = {}) {
        logger.info('📊 Fetching comprehensive dashboard data...');
        
        try {
            // Execute requests in parallel
            const [systemInfo, libraryStats, activeSessions, users] = await Promise.allSettled([
                this.getSystemInfo(options),
                this.getLibraryStats(options),
                this.getActiveSessions(options),
                this.getUsers(options)
            ]);
            
            const dashboardData = {
                systemInfo: systemInfo.status === 'fulfilled' && systemInfo.value.success ? systemInfo.value.data : null,
                libraryStats: libraryStats.status === 'fulfilled' && libraryStats.value.success ? libraryStats.value.data : null,
                activeSessions: activeSessions.status === 'fulfilled' && activeSessions.value.success ? activeSessions.value.data : [],
                users: users.status === 'fulfilled' && users.value.success ? users.value.data : [],
                healthStatus: this.healthStatus,
                lastHealthCheck: this.lastHealthCheck,
                timestamp: new Date().toISOString()
            };
            
            logger.info('✅ Dashboard data fetched successfully');
            return { success: true, data: dashboardData };
        } catch (error) {
            logger.error('❌ Failed to fetch dashboard data:', error.message);
            return { success: false, error: error.message };
        }
    }

    /**
     * Logout user
     */
    async logoutUser(userId) {
        logger.info(`🔓 Logging out user: ${userId}`);
        
        const userAuth = this.accessTokens.get(userId);
        if (!userAuth) {
            return { success: true, message: 'User already logged out' };
        }
        
        try {
            // Try to invalidate session on server
            await this.makeAuthenticatedRequest('POST', '/Sessions/Logout', null, { userId });
        } catch (error) {
            logger.info('⚠️  Could not invalidate server session:', error.message);
        }
        
        // Remove from local storage
        this.accessTokens.delete(userId);
        this.emit('userLoggedOut', { userId });
        
        logger.info(`✅ User logged out: ${userId}`);
        return { success: true };
    }

    /**
     * Clean up resources
     */
    async cleanup() {
        logger.info('🧹 Cleaning up Jellyfin Authentication Service...');
        
        this.stopHealthMonitoring();
        this.accessTokens.clear();
        this.apiKeys.clear();
        this.sessionCache.clear();
        this.removeAllListeners();
        
        logger.info('✅ Cleanup completed');
    }

    /**
     * Get service status
     */
    getStatus() {
        return {
            healthStatus: this.healthStatus,
            lastHealthCheck: this.lastHealthCheck,
            activeTokens: this.accessTokens.size,
            apiKeys: this.apiKeys.size,
            baseUrl: this.config.baseUrl
        };
    }
}

module.exports = JellyfinAuthService;