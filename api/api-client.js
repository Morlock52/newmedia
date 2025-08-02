/**
 * Modern API Client with Retry Logic, Circuit Breaker, and Fallbacks
 * Implements modern API integration patterns for robust backend communication
 */

class APIClient {
    constructor(options = {}) {
        this.baseURL = options.baseURL || '';
        this.timeout = options.timeout || 10000;
        this.retryAttempts = options.retryAttempts || 3;
        this.retryDelay = options.retryDelay || 1000;
        this.circuitBreakerThreshold = options.circuitBreakerThreshold || 5;
        this.circuitBreakerTimeout = options.circuitBreakerTimeout || 60000;
        
        // Circuit breaker state
        this.circuitBreaker = {
            failures: 0,
            state: 'CLOSED', // CLOSED, OPEN, HALF_OPEN
            nextAttempt: 0
        };

        // Request interceptors
        this.requestInterceptors = [];
        this.responseInterceptors = [];
        
        // Event emitter for monitoring
        this.events = new Map();
    }

    /**
     * Add request interceptor
     */
    addRequestInterceptor(interceptor) {
        this.requestInterceptors.push(interceptor);
    }

    /**
     * Add response interceptor
     */
    addResponseInterceptor(interceptor) {
        this.responseInterceptors.push(interceptor);
    }

    /**
     * Event listener
     */
    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event).push(callback);
    }

    /**
     * Emit event
     */
    emit(event, data) {
        const callbacks = this.events.get(event);
        if (callbacks) {
            callbacks.forEach(callback => callback(data));
        }
    }

    /**
     * Circuit breaker check
     */
    isCircuitOpen() {
        if (this.circuitBreaker.state === 'OPEN') {
            if (Date.now() > this.circuitBreaker.nextAttempt) {
                this.circuitBreaker.state = 'HALF_OPEN';
                return false;
            }
            return true;
        }
        return false;
    }

    /**
     * Update circuit breaker on success
     */
    onSuccess() {
        if (this.circuitBreaker.state === 'HALF_OPEN') {
            this.circuitBreaker.state = 'CLOSED';
        }
        this.circuitBreaker.failures = 0;
        this.emit('circuit-breaker', { state: this.circuitBreaker.state });
    }

    /**
     * Update circuit breaker on failure
     */
    onFailure() {
        this.circuitBreaker.failures++;
        
        if (this.circuitBreaker.failures >= this.circuitBreakerThreshold) {
            this.circuitBreaker.state = 'OPEN';
            this.circuitBreaker.nextAttempt = Date.now() + this.circuitBreakerTimeout;
        }
        
        this.emit('circuit-breaker', { 
            state: this.circuitBreaker.state,
            failures: this.circuitBreaker.failures 
        });
    }

    /**
     * Exponential backoff delay
     */
    getRetryDelay(attempt) {
        return this.retryDelay * Math.pow(2, attempt) + Math.random() * 1000;
    }

    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Create AbortController with timeout
     */
    createAbortController() {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
        }, this.timeout);
        
        // Clear timeout if request completes
        controller.signal.addEventListener('abort', () => {
            clearTimeout(timeoutId);
        });
        
        return controller;
    }

    /**
     * Apply request interceptors
     */
    async applyRequestInterceptors(config) {
        let modifiedConfig = { ...config };
        
        for (const interceptor of this.requestInterceptors) {
            try {
                modifiedConfig = await interceptor(modifiedConfig);
            } catch (error) {
                console.error('Request interceptor error:', error);
            }
        }
        
        return modifiedConfig;
    }

    /**
     * Apply response interceptors
     */
    async applyResponseInterceptors(response) {
        let modifiedResponse = response;
        
        for (const interceptor of this.responseInterceptors) {
            try {
                modifiedResponse = await interceptor(modifiedResponse);
            } catch (error) {
                console.error('Response interceptor error:', error);
            }
        }
        
        return modifiedResponse;
    }

    /**
     * Core request method with retry logic
     */
    async request(config) {
        // Check circuit breaker
        if (this.isCircuitOpen()) {
            const error = new Error('Circuit breaker is OPEN');
            error.code = 'CIRCUIT_BREAKER_OPEN';
            throw error;
        }

        // Apply request interceptors
        const finalConfig = await this.applyRequestInterceptors(config);
        
        let lastError;
        
        for (let attempt = 0; attempt <= this.retryAttempts; attempt++) {
            const controller = this.createAbortController();
            
            try {
                this.emit('request-start', { 
                    attempt: attempt + 1, 
                    config: finalConfig 
                });

                const response = await fetch(this.baseURL + finalConfig.url, {
                    method: finalConfig.method || 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                        ...finalConfig.headers
                    },
                    body: finalConfig.data ? JSON.stringify(finalConfig.data) : undefined,
                    signal: controller.signal
                });

                // Apply response interceptors
                const finalResponse = await this.applyResponseInterceptors(response);

                if (!finalResponse.ok) {
                    throw new Error(`HTTP ${finalResponse.status}: ${finalResponse.statusText}`);
                }

                const data = await finalResponse.json();
                
                this.onSuccess();
                this.emit('request-success', { 
                    attempt: attempt + 1, 
                    data 
                });

                return {
                    data,
                    status: finalResponse.status,
                    statusText: finalResponse.statusText,
                    headers: finalResponse.headers
                };

            } catch (error) {
                lastError = error;
                
                this.emit('request-error', { 
                    attempt: attempt + 1, 
                    error: error.message,
                    code: error.code
                });

                // Don't retry on certain errors
                if (error.name === 'AbortError') {
                    error.code = 'TIMEOUT';
                    break;
                }
                
                if (error.code === 'CIRCUIT_BREAKER_OPEN') {
                    break;
                }

                // Don't retry on 4xx errors (client errors)
                if (error.message.match(/HTTP 4\d\d/)) {
                    break;
                }

                // Wait before retry (except on last attempt)
                if (attempt < this.retryAttempts) {
                    const delay = this.getRetryDelay(attempt);
                    this.emit('request-retry', { 
                        attempt: attempt + 1, 
                        delay,
                        error: error.message 
                    });
                    await this.sleep(delay);
                }
            }
        }

        this.onFailure();
        throw lastError;
    }

    /**
     * GET request
     */
    async get(url, config = {}) {
        return this.request({ 
            ...config, 
            method: 'GET', 
            url 
        });
    }

    /**
     * POST request
     */
    async post(url, data, config = {}) {
        return this.request({ 
            ...config, 
            method: 'POST', 
            url, 
            data 
        });
    }

    /**
     * PUT request
     */
    async put(url, data, config = {}) {
        return this.request({ 
            ...config, 
            method: 'PUT', 
            url, 
            data 
        });
    }

    /**
     * DELETE request
     */
    async delete(url, config = {}) {
        return this.request({ 
            ...config, 
            method: 'DELETE', 
            url 
        });
    }

    /**
     * Health check
     */
    async healthCheck() {
        try {
            const response = await this.get('/api/health');
            return {
                healthy: true,
                ...response.data
            };
        } catch (error) {
            return {
                healthy: false,
                error: error.message,
                code: error.code
            };
        }
    }
}

/**
 * Specific API clients for different services
 */

class ChatbotAPIClient extends APIClient {
    constructor(options = {}) {
        super({
            baseURL: options.baseURL || 'http://localhost:3001',
            ...options
        });

        // Add authentication interceptor
        this.addRequestInterceptor(async (config) => {
            // Add any authentication headers here
            return config;
        });

        // Fallback data for when API is unavailable
        this.fallbackResponses = {
            '/api/chat': {
                response: "I apologize, but I'm currently offline. Please check your connection and try again later. In the meantime, you can:\n\n• Check the system status page\n• Review the troubleshooting guide\n• Contact support if the issue persists",
                usage: { total_tokens: 0 }
            }
        };
    }

    async sendMessage(message, history = [], settings = {}) {
        try {
            return await this.post('/api/chat', {
                message,
                history,
                settings
            });
        } catch (error) {
            console.warn('Chatbot API unavailable, using fallback:', error.message);
            
            // Return fallback response
            return {
                data: this.fallbackResponses['/api/chat'],
                status: 200,
                fallback: true
            };
        }
    }
}

class ConfigServerAPIClient extends APIClient {
    constructor(options = {}) {
        super({
            baseURL: options.baseURL || 'http://localhost:3000',
            ...options
        });

        // Fallback data
        this.fallbackData = {
            services: [
                { name: 'jellyfin', status: 'unknown', message: 'Service status unavailable' },
                { name: 'sonarr', status: 'unknown', message: 'Service status unavailable' },
                { name: 'radarr', status: 'unknown', message: 'Service status unavailable' },
                { name: 'prowlarr', status: 'unknown', message: 'Service status unavailable' }
            ]
        };
    }

    async getServices() {
        try {
            return await this.get('/api/docker/services');
        } catch (error) {
            console.warn('Config server unavailable, using fallback:', error.message);
            
            return {
                data: this.fallbackData,
                status: 200,
                fallback: true
            };
        }
    }

    async getServiceStatus(serviceName) {
        try {
            return await this.get(`/api/docker/services/${serviceName}`);
        } catch (error) {
            console.warn(`Service ${serviceName} status unavailable:`, error.message);
            
            return {
                data: {
                    service: serviceName,
                    status: 'unknown',
                    running: false,
                    message: 'Service status unavailable'
                },
                status: 200,
                fallback: true
            };
        }
    }

    async startServices(services = []) {
        try {
            return await this.post('/api/docker/services/start', { services });
        } catch (error) {
            throw new Error(`Failed to start services: ${error.message}`);
        }
    }

    async stopServices(services = []) {
        try {
            return await this.post('/api/docker/services/stop', { services });
        } catch (error) {
            throw new Error(`Failed to stop services: ${error.message}`);
        }
    }

    async restartServices(services = []) {
        try {
            return await this.post('/api/docker/services/restart', { services });
        } catch (error) {
            throw new Error(`Failed to restart services: ${error.message}`);
        }
    }
}

/**
 * Loading state manager
 */
class LoadingStateManager {
    constructor() {
        this.loadingStates = new Map();
        this.events = new Map();
    }

    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event).push(callback);
    }

    emit(event, data) {
        const callbacks = this.events.get(event);
        if (callbacks) {
            callbacks.forEach(callback => callback(data));
        }
    }

    setLoading(key, isLoading, message = '') {
        this.loadingStates.set(key, { isLoading, message, timestamp: Date.now() });
        this.emit('loading-change', { key, isLoading, message });
    }

    isLoading(key) {
        const state = this.loadingStates.get(key);
        return state ? state.isLoading : false;
    }

    getLoadingMessage(key) {
        const state = this.loadingStates.get(key);
        return state ? state.message : '';
    }

    getAllLoadingStates() {
        return Object.fromEntries(this.loadingStates);
    }
}

/**
 * Global instances
 */
const loadingManager = new LoadingStateManager();
const chatbotAPI = new ChatbotAPIClient();
const configServerAPI = new ConfigServerAPIClient();

// Event logging for debugging
if (typeof window !== 'undefined' && window.CONFIG?.debug?.enabled) {
    [chatbotAPI, configServerAPI].forEach(client => {
        client.on('request-start', (data) => {
            console.log(`API Request Start:`, data);
        });
        
        client.on('request-success', (data) => {
            console.log(`API Request Success:`, data);
        });
        
        client.on('request-error', (data) => {
            console.warn(`API Request Error:`, data);
        });
        
        client.on('request-retry', (data) => {
            console.log(`API Request Retry:`, data);
        });
        
        client.on('circuit-breaker', (data) => {
            console.warn(`Circuit Breaker:`, data);
        });
    });
}

// Export for use in browser
if (typeof window !== 'undefined') {
    window.APIClient = APIClient;
    window.ChatbotAPIClient = ChatbotAPIClient;
    window.ConfigServerAPIClient = ConfigServerAPIClient;
    window.LoadingStateManager = LoadingStateManager;
    window.loadingManager = loadingManager;
    window.chatbotAPI = chatbotAPI;
    window.configServerAPI = configServerAPI;
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        APIClient,
        ChatbotAPIClient,
        ConfigServerAPIClient,
        LoadingStateManager,
        loadingManager,
        chatbotAPI,
        configServerAPI
    };
}