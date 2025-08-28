/**
 * Dashboard Jellyfin Integration Module
 * Handles secure communication between dashboard and Jellyfin
 */

const axios = require('axios');
const fs = require('fs').promises;
const EventEmitter = require('events');

class DashboardJellyfinIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            jellyfinUrl: config.jellyfinUrl || 'http://localhost:8096',
            apiKeyFile: config.apiKeyFile || './jellyfin-api-config.json',
            timeout: config.timeout || 10000,
            retryAttempts: config.retryAttempts || 3,
            retryDelay: config.retryDelay || 1000,
            ...config
        };
        
        this.apiKey = null;
        this.isAuthenticated = false;
        this.lastHealthCheck = null;
        
        // Bind methods
        this.initialize = this.initialize.bind(this);
        this.authenticate = this.authenticate.bind(this);
        this.makeRequest = this.makeRequest.bind(this);
    }

    /**
     * Initialize integration
     */
    async initialize() {
        console.log('🔧 Initializing Dashboard-Jellyfin Integration...');
        
        try {
            // Load API key if available
            await this.loadAPIKey();
            
            // Test connection
            const connected = await this.testConnection();
            if (connected) {
                console.log('✅ Jellyfin integration initialized successfully');
                this.emit('initialized');
                return true;
            } else {
                console.log('❌ Failed to connect to Jellyfin');
                this.emit('error', new Error('Connection failed'));
                return false;
            }
        } catch (error) {
            console.error('❌ Integration initialization failed:', error.message);
            this.emit('error', error);
            return false;
        }
    }

    /**
     * Load API key from file
     */
    async loadAPIKey() {
        try {
            const data = await fs.readFile(this.config.apiKeyFile, 'utf8');
            const config = JSON.parse(data);
            
            if (config.apiKey) {
                this.apiKey = config.apiKey;
                this.isAuthenticated = true;
                console.log('✅ API key loaded from file');
                return true;
            }
        } catch (error) {
            console.log('⚠️  No existing API key file found');
        }
        return false;
    }

    /**
     * Save API key to file
     */
    async saveAPIKey(apiKey, additionalData = {}) {
        try {
            const config = {
                apiKey,
                created: new Date().toISOString(),
                jellyfinUrl: this.config.jellyfinUrl,
                ...additionalData
            };
            
            await fs.writeFile(this.config.apiKeyFile, JSON.stringify(config, null, 2));
            console.log('💾 API key saved to file');
            return true;
        } catch (error) {
            console.error('❌ Failed to save API key:', error.message);
            return false;
        }
    }

    /**
     * Test connection to Jellyfin
     */
    async testConnection() {
        try {
            const response = await this.makeRequest('GET', '/System/Info/Public');
            this.lastHealthCheck = new Date();
            return response !== null;
        } catch (error) {
            console.error('❌ Connection test failed:', error.message);
            return false;
        }
    }

    /**
     * Make authenticated request to Jellyfin
     */
    async makeRequest(method, endpoint, data = null, options = {}) {
        const url = `${this.config.jellyfinUrl}${endpoint}`;
        
        const config = {
            method,
            url,
            timeout: this.config.timeout,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                ...options.headers
            },
            validateStatus: (status) => status < 500
        };
        
        // Add authentication
        if (this.apiKey) {
            config.headers['X-Emby-Token'] = this.apiKey;
        }
        
        if (data) {
            config.data = data;
        }
        
        let lastError;
        
        // Retry logic
        for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
            try {
                const response = await axios(config);
                
                if (response.status >= 200 && response.status < 300) {
                    return response.data;
                } else if (response.status === 401) {
                    this.isAuthenticated = false;
                    throw new Error('Authentication required');
                } else {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
            } catch (error) {
                lastError = error;
                
                if (attempt < this.config.retryAttempts) {
                    console.log(`⚠️  Request failed (attempt ${attempt}), retrying...`);
                    await new Promise(resolve => setTimeout(resolve, this.config.retryDelay));
                }
            }
        }
        
        throw lastError;
    }

    /**
     * Authenticate with username/password and get API key
     */
    async authenticate(username, password) {
        console.log('🔐 Authenticating with Jellyfin...');
        
        try {
            // Clear existing auth
            this.apiKey = null;
            this.isAuthenticated = false;
            
            // Authenticate
            const authData = {
                Username: username,
                Pw: password
            };
            
            const response = await this.makeRequest('POST', '/Users/authenticatebyname', authData);
            
            if (response && response.AccessToken) {
                // Create API key for persistent access
                this.apiKey = response.AccessToken;
                this.isAuthenticated = true;
                
                // Try to create a permanent API key
                try {
                    const apiKeyResponse = await this.makeRequest('POST', '/Auth/Keys', {
                        App: 'MediaDashboard'
                    });
                    
                    if (apiKeyResponse && apiKeyResponse.AccessToken) {
                        this.apiKey = apiKeyResponse.AccessToken;
                        await this.saveAPIKey(this.apiKey, {
                            userId: response.User.Id,
                            username: response.User.Name
                        });
                    }
                } catch (keyError) {
                    console.log('⚠️  Could not create permanent API key, using session token');
                }
                
                console.log('✅ Authentication successful');
                this.emit('authenticated', response.User);
                return response.User;
            } else {
                throw new Error('Invalid response from authentication endpoint');
            }
        } catch (error) {
            console.error('❌ Authentication failed:', error.message);
            this.emit('authError', error);
            throw error;
        }
    }

    /**
     * Get system information
     */
    async getSystemInfo() {
        try {
            return await this.makeRequest('GET', '/System/Info');
        } catch (error) {
            console.error('❌ Failed to get system info:', error.message);
            return null;
        }
    }

    /**
     * Get library statistics
     */
    async getLibraryStats() {
        try {
            const stats = await this.makeRequest('GET', '/Items/Counts');
            return {
                movies: stats.MovieCount || 0,
                series: stats.SeriesCount || 0,
                episodes: stats.EpisodeCount || 0,
                songs: stats.SongCount || 0,
                albums: stats.AlbumCount || 0,
                books: stats.BookCount || 0
            };
        } catch (error) {
            console.error('❌ Failed to get library stats:', error.message);
            return null;
        }
    }

    /**
     * Get active sessions
     */
    async getActiveSessions() {
        try {
            const sessions = await this.makeRequest('GET', '/Sessions');
            return sessions || [];
        } catch (error) {
            console.error('❌ Failed to get active sessions:', error.message);
            return [];
        }
    }

    /**
     * Get recent activities
     */
    async getRecentActivities(limit = 10) {
        try {
            const activities = await this.makeRequest('GET', `/System/ActivityLog/Entries?limit=${limit}`);
            return activities?.Items || [];
        } catch (error) {
            console.error('❌ Failed to get recent activities:', error.message);
            return [];
        }
    }

    /**
     * Get server configuration
     */
    async getServerConfig() {
        try {
            return await this.makeRequest('GET', '/System/Configuration');
        } catch (error) {
            console.error('❌ Failed to get server config:', error.message);
            return null;
        }
    }

    /**
     * Get users
     */
    async getUsers() {
        try {
            return await this.makeRequest('GET', '/Users');
        } catch (error) {
            console.error('❌ Failed to get users:', error.message);
            return [];
        }
    }

    /**
     * Search media items
     */
    async searchMedia(query, limit = 20) {
        try {
            const results = await this.makeRequest('GET', `/Items?searchTerm=${encodeURIComponent(query)}&limit=${limit}`);
            return results?.Items || [];
        } catch (error) {
            console.error('❌ Failed to search media:', error.message);
            return [];
        }
    }

    /**
     * Get dashboard data
     */
    async getDashboardData() {
        try {
            const [systemInfo, libraryStats, activeSessions, recentActivities] = await Promise.allSettled([
                this.getSystemInfo(),
                this.getLibraryStats(),
                this.getActiveSessions(),
                this.getRecentActivities(5)
            ]);

            return {
                systemInfo: systemInfo.status === 'fulfilled' ? systemInfo.value : null,
                libraryStats: libraryStats.status === 'fulfilled' ? libraryStats.value : null,
                activeSessions: activeSessions.status === 'fulfilled' ? activeSessions.value : [],
                recentActivities: recentActivities.status === 'fulfilled' ? recentActivities.value : [],
                lastUpdated: new Date().toISOString()
            };
        } catch (error) {
            console.error('❌ Failed to get dashboard data:', error.message);
            throw error;
        }
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const isHealthy = await this.testConnection();
            
            if (isHealthy) {
                this.emit('healthCheck', { status: 'healthy', timestamp: new Date() });
            } else {
                this.emit('healthCheck', { status: 'unhealthy', timestamp: new Date() });
            }
            
            return isHealthy;
        } catch (error) {
            this.emit('healthCheck', { status: 'error', error: error.message, timestamp: new Date() });
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
        
        console.log(`🏥 Health monitoring started (interval: ${interval}ms)`);
    }

    /**
     * Stop health monitoring
     */
    stopHealthMonitoring() {
        if (this.healthInterval) {
            clearInterval(this.healthInterval);
            this.healthInterval = null;
            console.log('🏥 Health monitoring stopped');
        }
    }

    /**
     * Clean up
     */
    async cleanup() {
        this.stopHealthMonitoring();
        this.removeAllListeners();
        console.log('🧹 Jellyfin integration cleanup completed');
    }
}

module.exports = DashboardJellyfinIntegration;

// Example usage
if (require.main === module) {
    const integration = new DashboardJellyfinIntegration();
    
    integration.on('initialized', () => {
        console.log('🎉 Integration ready!');
        integration.startHealthMonitoring();
    });
    
    integration.on('error', (error) => {
        console.error('🚨 Integration error:', error.message);
    });
    
    integration.on('healthCheck', (status) => {
        console.log(`🏥 Health: ${status.status}`);
    });
    
    // Initialize
    integration.initialize()
        .then(success => {
            if (success) {
                // Test dashboard data retrieval
                return integration.getDashboardData();
            }
        })
        .then(data => {
            if (data) {
                console.log('📊 Dashboard data retrieved successfully');
                console.log(`   Library: ${JSON.stringify(data.libraryStats)}`);
                console.log(`   Sessions: ${data.activeSessions.length} active`);
            }
        })
        .catch(error => {
            console.error('❌ Example failed:', error);
        });
}