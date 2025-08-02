// WebSocket Client for Real-time Updates

class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.isConnected = false;
        this.eventHandlers = new Map();
        this.reconnectTimer = null;
        
        this.init();
    }

    init() {
        this.connect();
    }

    connect() {
        try {
            console.log(`Connecting to WebSocket: ${this.url}`);
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = this.onOpen.bind(this);
            this.ws.onmessage = this.onMessage.bind(this);
            this.ws.onerror = this.onError.bind(this);
            this.ws.onclose = this.onClose.bind(this);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.scheduleReconnect();
        }
    }

    onOpen(event) {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Clear reconnect timer
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        // Send initial handshake
        this.send({
            type: 'handshake',
            data: {
                clientType: 'holographic-dashboard',
                version: '2.0.0',
                capabilities: ['streaming', 'realtime', '3d-visualization']
            }
        });
        
        // Trigger connected event
        this.emit('connected', { timestamp: Date.now() });
    }

    onMessage(event) {
        try {
            const message = JSON.parse(event.data);
            
            if (CONFIG.debug.logWebSocket) {
                console.log('WebSocket message received:', message);
            }
            
            // Handle different message types
            switch (message.type) {
                case 'stats-update':
                    this.handleStatsUpdate(message.data);
                    break;
                    
                case 'media-update':
                    this.handleMediaUpdate(message.data);
                    break;
                    
                case 'activity':
                    this.handleActivity(message.data);
                    break;
                    
                case 'stream-status':
                    this.handleStreamStatus(message.data);
                    break;
                    
                case 'system-alert':
                    this.handleSystemAlert(message.data);
                    break;
                    
                case 'performance-metrics':
                    this.handlePerformanceMetrics(message.data);
                    break;
                    
                default:
                    // Emit custom event
                    this.emit(message.type, message.data);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }

    onError(error) {
        console.error('WebSocket error:', error);
        this.emit('error', error);
    }

    onClose(event) {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        
        this.emit('disconnected', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
        });
        
        // Schedule reconnection
        if (this.reconnectAttempts < CONFIG.websocket.maxReconnectAttempts) {
            this.scheduleReconnect();
        } else {
            console.error('Max reconnection attempts reached');
            this.emit('reconnect-failed', {
                attempts: this.reconnectAttempts
            });
        }
    }

    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = Math.min(
            CONFIG.websocket.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1),
            30000 // Max 30 seconds
        );
        
        console.log(`Scheduling reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }

    send(data) {
        if (!this.isConnected || this.ws.readyState !== WebSocket.OPEN) {
            console.warn('WebSocket not connected, queuing message');
            // Could implement message queue here
            return false;
        }
        
        try {
            const message = typeof data === 'string' ? data : JSON.stringify(data);
            this.ws.send(message);
            return true;
        } catch (error) {
            console.error('Error sending WebSocket message:', error);
            return false;
        }
    }

    // Event handling
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, new Set());
        }
        this.eventHandlers.get(event).add(handler);
    }

    off(event, handler) {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            handlers.delete(handler);
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

    // Message handlers
    handleStatsUpdate(data) {
        this.emit('stats-update', {
            totalMedia: data.totalMedia || 0,
            storageUsed: data.storageUsed || 0,
            activeUsers: data.activeUsers || 0,
            bandwidth: data.bandwidth || 0,
            activeStreams: data.activeStreams || 0,
            gpuUsage: data.gpuUsage || 0,
            cpuUsage: data.cpuUsage || 0,
            memoryUsage: data.memoryUsage || 0
        });
    }

    handleMediaUpdate(data) {
        this.emit('media-update', {
            action: data.action, // 'added', 'removed', 'updated'
            media: data.media,
            timestamp: data.timestamp
        });
    }

    handleActivity(data) {
        this.emit('activity', {
            icon: data.icon,
            title: data.title,
            description: data.description,
            timestamp: data.timestamp || Date.now(),
            priority: data.priority || 'normal'
        });
    }

    handleStreamStatus(data) {
        this.emit('stream-status', {
            streamId: data.streamId,
            status: data.status, // 'started', 'stopped', 'buffering', 'error'
            quality: data.quality,
            viewers: data.viewers,
            bitrate: data.bitrate
        });
    }

    handleSystemAlert(data) {
        this.emit('system-alert', {
            level: data.level, // 'info', 'warning', 'error', 'critical'
            message: data.message,
            details: data.details,
            timestamp: data.timestamp
        });
    }

    handlePerformanceMetrics(data) {
        this.emit('performance-metrics', {
            fps: data.fps,
            latency: data.latency,
            bufferHealth: data.bufferHealth,
            transcoding: data.transcoding,
            networkQuality: data.networkQuality
        });
    }

    // Public methods
    close() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        this.isConnected = false;
        this.reconnectAttempts = 0;
    }

    // Request methods
    requestStats() {
        return this.send({
            type: 'request-stats',
            timestamp: Date.now()
        });
    }

    requestMediaList(filters = {}) {
        return this.send({
            type: 'request-media-list',
            filters: filters,
            timestamp: Date.now()
        });
    }

    subscribeToStream(streamId) {
        return this.send({
            type: 'subscribe-stream',
            streamId: streamId,
            timestamp: Date.now()
        });
    }

    unsubscribeFromStream(streamId) {
        return this.send({
            type: 'unsubscribe-stream',
            streamId: streamId,
            timestamp: Date.now()
        });
    }

    // Demo mode - simulate WebSocket messages
    startDemoMode() {
        console.log('Starting WebSocket demo mode');
        
        // Simulate connection
        setTimeout(() => {
            this.isConnected = true;
            this.emit('connected', { timestamp: Date.now() });
        }, 1000);
        
        // Simulate periodic stats updates
        setInterval(() => {
            if (!this.isConnected) return;
            
            this.handleStatsUpdate({
                totalMedia: 2847 + Utils.randomInt(-5, 10),
                storageUsed: 47.3 + Utils.randomFloat(-0.5, 0.5),
                activeUsers: 12 + Utils.randomInt(-2, 3),
                bandwidth: 450 + Utils.randomInt(-50, 100),
                activeStreams: 8 + Utils.randomInt(-2, 4),
                gpuUsage: 65 + Utils.randomInt(-10, 10),
                cpuUsage: 45 + Utils.randomInt(-10, 10),
                memoryUsage: 72 + Utils.randomInt(-5, 5)
            });
        }, 3000);
        
        // Simulate random activities
        const activities = [
            { icon: '🎬', title: 'New content added', description: 'Movie: Interstellar 2' },
            { icon: '👤', title: 'User activity', description: 'John started watching The Matrix' },
            { icon: '📡', title: 'Stream update', description: 'Live stream quality upgraded to 4K' },
            { icon: '⚡', title: 'Performance', description: 'Transcoding queue optimized' },
            { icon: '🔄', title: 'System update', description: 'Cache refreshed successfully' }
        ];
        
        setInterval(() => {
            if (!this.isConnected) return;
            
            const activity = activities[Math.floor(Math.random() * activities.length)];
            this.handleActivity(activity);
        }, 8000);
        
        // Simulate performance metrics
        setInterval(() => {
            if (!this.isConnected) return;
            
            this.handlePerformanceMetrics({
                fps: 60 + Utils.randomFloat(-5, 5),
                latency: 15 + Utils.randomInt(-5, 10),
                bufferHealth: 95 + Utils.randomInt(-10, 5),
                transcoding: {
                    queue: Utils.randomInt(0, 10),
                    processing: Utils.randomInt(1, 5),
                    completed: Utils.randomInt(50, 200)
                },
                networkQuality: Utils.randomFloat(0.8, 1.0)
            });
        }, 5000);
    }
}

// Export for use in other modules
window.WebSocketClient = WebSocketClient;