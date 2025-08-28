/**
 * WebSocket Client for Media Dashboard
 * Handles real-time communication with the media server API
 */

class MediaWebSocketClient {
    constructor(url, options = {}) {
        this.url = url;
        this.options = {
            reconnectInterval: 5000,
            maxReconnectAttempts: 10,
            ...options
        };
        
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.listeners = new Map();
        this.messageQueue = [];
        
        this.connect();
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = (event) => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.emit('connected', event);
                
                // Send queued messages
                while (this.messageQueue.length > 0) {
                    const message = this.messageQueue.shift();
                    this.send(message);
                }
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.emit('message', data);
                    
                    // Emit specific event types
                    if (data.type) {
                        this.emit(data.type, data.data || data);
                    }
                } catch (error) {
                    console.error('WebSocket message parse error:', error);
                    this.emit('error', { type: 'parse_error', error });
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason);
                this.isConnected = false;
                this.emit('disconnected', event);
                
                // Attempt reconnection
                if (this.reconnectAttempts < this.options.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.options.maxReconnectAttempts}`);
                    setTimeout(() => this.connect(), this.options.reconnectInterval);
                } else {
                    console.error('Max reconnection attempts reached');
                    this.emit('maxReconnectAttemptsReached');
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.emit('error', error);
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.emit('error', error);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close(1000, 'Client disconnect');
            this.ws = null;
        }
        this.isConnected = false;
    }

    send(message) {
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof message === 'string' ? message : JSON.stringify(message));
        } else {
            // Queue message for later sending
            this.messageQueue.push(message);
        }
    }

    // Event system
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).delete(callback);
        }
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    // Media server specific methods
    subscribeToHealth() {
        this.send({
            action: 'subscribe-health'
        });
    }

    subscribeToLogs(options = {}) {
        this.send({
            action: 'subscribe-logs',
            payload: options
        });
    }

    subscribeToServices() {
        this.send({
            action: 'subscribe-services'
        });
    }

    getStatus() {
        this.send({
            action: 'get-status'
        });
    }

    ping() {
        this.send({
            action: 'ping',
            timestamp: Date.now()
        });
    }

    // Authentication
    authenticate(token) {
        this.send({
            action: 'authenticate',
            payload: { token }
        });
    }

    // Service controls
    controlService(serviceName, action) {
        this.send({
            action: 'control-service',
            payload: {
                service: serviceName,
                action: action
            }
        });
    }

    // Real-time metrics
    requestMetrics(type = 'all') {
        this.send({
            action: 'get-metrics',
            payload: { type }
        });
    }

    // Download manager
    getDownloads() {
        this.send({
            action: 'get-downloads'
        });
    }

    addDownload(url, options = {}) {
        this.send({
            action: 'add-download',
            payload: { url, ...options }
        });
    }

    // Media library
    getMediaLibrary(type = 'all', page = 1, limit = 20) {
        this.send({
            action: 'get-media-library',
            payload: { type, page, limit }
        });
    }

    searchMedia(query, type = 'all') {
        this.send({
            action: 'search-media',
            payload: { query, type }
        });
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MediaWebSocketClient;
} else if (typeof window !== 'undefined') {
    window.MediaWebSocketClient = MediaWebSocketClient;
}