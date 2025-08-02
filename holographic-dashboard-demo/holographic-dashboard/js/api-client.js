/**
 * Enhanced API Client for Holographic Media Dashboard
 * Integrates with the new backend services and external media services
 */

class HolographicAPIClient {
    constructor(options = {}) {
        this.baseURL = options.baseURL || 'http://localhost:3000/api/v1';
        this.timeout = options.timeout || 10000;
        this.retries = options.retries || 3;
        
        // Authentication state
        this.authToken = localStorage.getItem('auth_token');
        this.refreshToken = localStorage.getItem('refresh_token');
        
        // Request interceptors
        this.requestInterceptors = [];
        this.responseInterceptors = [];
        
        // Event system for real-time updates
        this.eventHandlers = new Map();
        this.websocket = null;
        
        // Cache for frequently accessed data
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
        
        this.initializeWebSocket();
    }

    // Authentication methods
    async login(credentials) {
        try {
            const response = await this.post('/auth/login', credentials);
            
            if (response.success) {
                this.authToken = response.data.accessToken;
                this.refreshToken = response.data.refreshToken;
                
                localStorage.setItem('auth_token', this.authToken);
                localStorage.setItem('refresh_token', this.refreshToken);
                
                this.emit('auth:login', response.data.user);
            }
            
            return response;
        } catch (error) {
            this.emit('auth:error', error);
            throw error;
        }
    }

    async logout() {
        try {
            await this.post('/auth/logout');
        } catch (error) {
            console.warn('Logout request failed:', error);
        } finally {
            this.authToken = null;
            this.refreshToken = null;
            localStorage.removeItem('auth_token');
            localStorage.removeItem('refresh_token');
            this.cache.clear();
            this.emit('auth:logout');
        }
    }

    async refreshAuthToken() {
        try {
            const response = await this.post('/auth/refresh', {
                refreshToken: this.refreshToken
            });
            
            if (response.success) {
                this.authToken = response.data.accessToken;
                localStorage.setItem('auth_token', this.authToken);
                return true;
            }
            
            return false;
        } catch (error) {
            this.logout();
            return false;
        }
    }

    // Media API methods
    async getMedia(params = {}) {
        const cacheKey = `media:${JSON.stringify(params)}`;
        const cached = this.getFromCache(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const queryString = new URLSearchParams(params).toString();
            const response = await this.get(`/media?${queryString}`);
            
            if (response.success) {
                this.setCache(cacheKey, response);
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'getMedia', error });
            throw error;
        }
    }

    async searchMedia(query, options = {}) {
        try {
            const params = {
                q: query,
                source: options.source || 'all',
                ...options.filters,
                page: options.page || 1,
                limit: options.limit || 20
            };

            const response = await this.get('/media/search', { params });
            
            if (response.success) {
                this.emit('media:search', { query, results: response.data.results });
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'searchMedia', error });
            throw error;
        }
    }

    async getMediaById(id) {
        const cacheKey = `media:${id}`;
        const cached = this.getFromCache(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const response = await this.get(`/media/${id}`);
            
            if (response.success) {
                this.setCache(cacheKey, response);
                this.emit('media:loaded', response.data.media);
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'getMediaById', error });
            throw error;
        }
    }

    async getTrendingMedia(timeframe = '7d', limit = 10) {
        const cacheKey = `trending:${timeframe}:${limit}`;
        const cached = this.getFromCache(cacheKey);
        
        if (cached) {
            return cached;
        }

        try {
            const response = await this.get('/media/trending', {
                params: { timeframe, limit }
            });
            
            if (response.success) {
                this.setCache(cacheKey, response, 5 * 60 * 1000); // Cache for 5 minutes
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'getTrendingMedia', error });
            throw error;
        }
    }

    async getRecommendations(limit = 10) {
        try {
            const response = await this.get('/media/recommendations', {
                params: { limit }
            });
            
            if (response.success) {
                this.emit('media:recommendations', response.data.recommendations);
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'getRecommendations', error });
            throw error;
        }
    }

    async recordView(mediaId, metadata = {}) {
        try {
            const response = await this.post(`/media/${mediaId}/view`, metadata);
            
            if (response.success) {
                this.emit('media:viewed', { mediaId, metadata });
                this.invalidateCache(`media:${mediaId}`);
            }
            
            return response;
        } catch (error) {
            console.warn('Failed to record view:', error);
            // Don't throw - view recording is not critical
        }
    }

    async likeMedia(mediaId) {
        try {
            const response = await this.post(`/media/${mediaId}/like`);
            
            if (response.success) {
                this.emit('media:liked', { mediaId });
                this.invalidateCache(`media:${mediaId}`);
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'likeMedia', error });
            throw error;
        }
    }

    async unlikeMedia(mediaId) {
        try {
            const response = await this.delete(`/media/${mediaId}/like`);
            
            if (response.success) {
                this.emit('media:unliked', { mediaId });
                this.invalidateCache(`media:${mediaId}`);
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'unlikeMedia', error });
            throw error;
        }
    }

    // Service integration methods
    async getServiceHealth() {
        try {
            const response = await this.get('/services/health');
            
            if (response.success) {
                this.emit('services:health', response.data);
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'getServiceHealth', error });
            throw error;
        }
    }

    async startServices(serviceNames = []) {
        try {
            const response = await this.post('/services/start', { services: serviceNames });
            
            if (response.success) {
                this.emit('services:started', { services: serviceNames });
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'startServices', error });
            throw error;
        }
    }

    async stopServices(serviceNames = []) {
        try {
            const response = await this.post('/services/stop', { services: serviceNames });
            
            if (response.success) {
                this.emit('services:stopped', { services: serviceNames });
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'stopServices', error });
            throw error;
        }
    }

    async restartServices(serviceNames = []) {
        try {
            const response = await this.post('/services/restart', { services: serviceNames });
            
            if (response.success) {
                this.emit('services:restarted', { services: serviceNames });
            }
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'restartServices', error });
            throw error;
        }
    }

    // Streaming methods
    async getStreamUrl(mediaId, quality = 'auto') {
        try {
            const response = await this.get(`/stream/${mediaId}`, {
                params: { quality }
            });
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'getStreamUrl', error });
            throw error;
        }
    }

    async getHLSManifest(mediaId, quality = 'auto') {
        try {
            const response = await this.get(`/hls/${mediaId}/playlist.m3u8`, {
                params: { quality }
            });
            
            return response;
        } catch (error) {
            this.emit('api:error', { method: 'getHLSManifest', error });
            throw error;
        }
    }

    // WebSocket methods
    initializeWebSocket() {
        if (this.websocket) {
            this.websocket.close();
        }

        const wsUrl = this.baseURL.replace(/^http/, 'ws').replace('/api/v1', '/ws');
        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.emit('ws:connected');
            
            // Send authentication if available
            if (this.authToken) {
                this.websocket.send(JSON.stringify({
                    type: 'auth',
                    token: this.authToken
                }));
            }
        };

        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.emit('ws:message', data);
                
                // Handle specific message types
                switch (data.type) {
                    case 'media_update':
                        this.invalidateCache(`media:${data.mediaId}`);
                        this.emit('media:updated', data);
                        break;
                    case 'service_status':
                        this.emit('services:status', data);
                        break;
                    case 'notification':
                        this.emit('notification', data);
                        break;
                }
            } catch (error) {
                console.error('WebSocket message parse error:', error);
            }
        };

        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.emit('ws:disconnected');
            
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                this.initializeWebSocket();
            }, 5000);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('ws:error', error);
        };
    }

    // HTTP request methods
    async get(endpoint, options = {}) {
        return this.request('GET', endpoint, null, options);
    }

    async post(endpoint, data = null, options = {}) {
        return this.request('POST', endpoint, data, options);
    }

    async put(endpoint, data = null, options = {}) {
        return this.request('PUT', endpoint, data, options);
    }

    async delete(endpoint, options = {}) {
        return this.request('DELETE', endpoint, null, options);
    }

    async request(method, endpoint, data = null, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add authentication header
        if (this.authToken) {
            config.headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        // Add query parameters
        if (options.params) {
            const queryString = new URLSearchParams(options.params).toString();
            url += (url.includes('?') ? '&' : '?') + queryString;
        }

        // Add request body
        if (data) {
            config.body = JSON.stringify(data);
        }

        let attempt = 0;
        while (attempt < this.retries) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);
                
                config.signal = controller.signal;
                
                const response = await fetch(url, config);
                clearTimeout(timeoutId);

                // Handle authentication errors
                if (response.status === 401 && this.refreshToken) {
                    const refreshed = await this.refreshAuthToken();
                    if (refreshed && attempt === 0) {
                        // Retry with new token
                        config.headers['Authorization'] = `Bearer ${this.authToken}`;
                        attempt++;
                        continue;
                    }
                }

                const result = await response.json();
                
                if (!response.ok) {
                    throw new Error(result.error?.message || `HTTP ${response.status}`);
                }

                return result;

            } catch (error) {
                attempt++;
                
                if (error.name === 'AbortError') {
                    throw new Error('Request timeout');
                }
                
                if (attempt >= this.retries) {
                    throw error;
                }
                
                // Wait before retry
                await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            }
        }
    }

    // Cache methods
    getFromCache(key) {
        const cached = this.cache.get(key);
        if (cached && Date.now() - cached.timestamp < cached.ttl) {
            return cached.data;
        }
        this.cache.delete(key);
        return null;
    }

    setCache(key, data, ttl = this.cacheTimeout) {
        this.cache.set(key, {
            data,
            timestamp: Date.now(),
            ttl
        });
    }

    invalidateCache(pattern) {
        if (typeof pattern === 'string') {
            // Exact match
            this.cache.delete(pattern);
        } else if (pattern instanceof RegExp) {
            // Pattern match
            for (const key of this.cache.keys()) {
                if (pattern.test(key)) {
                    this.cache.delete(key);
                }
            }
        }
    }

    clearCache() {
        this.cache.clear();
    }

    // Event system
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    off(event, handler) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    // Utility methods
    isAuthenticated() {
        return !!this.authToken;
    }

    getAuthToken() {
        return this.authToken;
    }

    setBaseURL(url) {
        this.baseURL = url;
    }

    getBaseURL() {
        return this.baseURL;
    }
}

// Export for browser
if (typeof window !== 'undefined') {
    window.HolographicAPIClient = HolographicAPIClient;
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HolographicAPIClient;
}