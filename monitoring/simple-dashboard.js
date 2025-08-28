// Simplified Monitoring Dashboard
// Lightweight version for demonstration

const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const os = require('os');

class SimpleMonitoringDashboard {
    constructor(config = {}) {
        this.config = {
            port: config.port || 3005,
            updateInterval: config.updateInterval || 5000,
            ...config
        };

        this.app = express();
        this.server = http.createServer(this.app);
        this.io = socketIo(this.server, {
            cors: { origin: "*", methods: ["GET", "POST"] }
        });
        
        // State management
        this.state = {
            system: {},
            services: {},
            logs: [],
            alerts: [],
            metrics: {
                requests: 0,
                errors: 0,
                avgResponseTime: 0,
                uptime: 0
            }
        };

        this.setupRoutes();
        this.setupSocketHandlers();
        this.startMonitoring();
    }

    setupRoutes() {
        // Middleware
        this.app.use(express.json());
        this.app.use(express.static(path.join(__dirname)));
        
        // API Routes
        this.app.get('/api/status', (req, res) => {
            res.json({
                status: 'healthy',
                uptime: process.uptime(),
                timestamp: new Date().toISOString(),
                state: this.state
            });
        });

        this.app.get('/api/metrics', (req, res) => {
            res.json({
                current: this.state,
                alerts: this.state.alerts
            });
        });

        this.app.get('/api/logs', (req, res) => {
            const limit = parseInt(req.query.limit) || 100;
            res.json(this.state.logs.slice(-limit));
        });

        // Dashboard HTML
        this.app.get('/', (req, res) => {
            res.sendFile(path.join(__dirname, 'simple-dashboard.html'));
        });
    }

    setupSocketHandlers() {
        this.io.on('connection', (socket) => {
            console.log('📱 Client connected:', socket.id);

            // Send initial state
            socket.emit('initial-state', this.state);

            socket.on('refresh', () => {
                this.updateMetrics();
            });

            socket.on('disconnect', () => {
                console.log('📱 Client disconnected:', socket.id);
            });
        });
    }

    async startMonitoring() {
        console.log('🚀 Starting Simple Monitoring Dashboard');
        
        // Initial update
        await this.updateMetrics();
        
        // Regular updates
        setInterval(() => {
            this.updateMetrics();
        }, this.config.updateInterval);
        
        // Generate sample logs
        this.generateSampleLogs();
        
        // Start server
        this.server.listen(this.config.port, () => {
            console.log(`📊 Monitoring dashboard running on http://localhost:${this.config.port}`);
            console.log('🎯 Open your browser to see the live dashboard');
        });
    }

    async updateMetrics() {
        try {
            // Basic system metrics
            const memUsage = process.memoryUsage();
            const cpusInfo = os.cpus();
            const loadAvg = os.loadavg();
            
            this.state.system = {
                cpu: {
                    usage: Math.round(loadAvg[0] * 10), // Approximate CPU usage
                    cores: cpusInfo.length,
                    model: cpusInfo[0].model
                },
                memory: {
                    total: os.totalmem(),
                    used: memUsage.rss,
                    free: os.freemem(),
                    percentage: Math.round((memUsage.rss / os.totalmem()) * 100)
                },
                network: {
                    rx: Math.floor(Math.random() * 1000000), // Simulated
                    tx: Math.floor(Math.random() * 1000000)
                },
                uptime: os.uptime(),
                timestamp: new Date().toISOString()
            };

            // Sample services
            this.state.services = {
                'Jellyfin': {
                    name: 'Jellyfin',
                    healthy: Math.random() > 0.1,
                    responseTime: Math.floor(Math.random() * 200) + 50,
                    lastCheck: new Date().toISOString()
                },
                'Sonarr': {
                    name: 'Sonarr',
                    healthy: Math.random() > 0.05,
                    responseTime: Math.floor(Math.random() * 300) + 100,
                    lastCheck: new Date().toISOString()
                },
                'Radarr': {
                    name: 'Radarr',
                    healthy: Math.random() > 0.05,
                    responseTime: Math.floor(Math.random() * 250) + 80,
                    lastCheck: new Date().toISOString()
                },
                'qBittorrent': {
                    name: 'qBittorrent',
                    healthy: Math.random() > 0.08,
                    responseTime: Math.floor(Math.random() * 150) + 30,
                    lastCheck: new Date().toISOString()
                }
            };

            this.state.metrics.uptime = process.uptime();
            
            // Check for alerts
            this.checkAlerts();
            
            // Emit updates to all connected clients
            this.io.emit('metrics-update', this.state);
            
        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    }

    checkAlerts() {
        // CPU Alert
        if (this.state.system.cpu?.usage > 80) {
            this.createAlert('high-cpu', 'High CPU usage detected', {
                current: this.state.system.cpu.usage,
                threshold: 80
            });
        }

        // Memory Alert
        if (this.state.system.memory?.percentage > 85) {
            this.createAlert('high-memory', 'High memory usage detected', {
                current: this.state.system.memory.percentage,
                threshold: 85
            });
        }

        // Service health alerts
        Object.values(this.state.services).forEach(service => {
            if (!service.healthy) {
                this.createAlert('service-unhealthy', `${service.name} is unhealthy`, {
                    service: service.name,
                    responseTime: service.responseTime
                });
            }
        });
    }

    createAlert(type, message, details = {}) {
        const alertId = `${type}-${Date.now()}`;
        
        // Check if similar alert already exists
        const existingAlert = this.state.alerts.find(a => 
            a.type === type && !a.resolved && 
            (Date.now() - new Date(a.timestamp).getTime()) < 300000 // 5 minutes
        );

        if (existingAlert) {
            existingAlert.count = (existingAlert.count || 1) + 1;
            existingAlert.lastOccurrence = new Date().toISOString();
            return;
        }

        const alert = {
            id: alertId,
            type,
            message,
            details,
            severity: this.getAlertSeverity(type),
            timestamp: new Date().toISOString(),
            acknowledged: false,
            resolved: false
        };

        this.state.alerts.unshift(alert);
        
        // Keep only last 50 alerts
        if (this.state.alerts.length > 50) {
            this.state.alerts = this.state.alerts.slice(0, 50);
        }

        // Emit alert to connected clients
        this.io.emit('alert', alert);
        
        console.log('⚠️  Alert:', alert.message);
    }

    getAlertSeverity(type) {
        const severityMap = {
            'service-unhealthy': 'warning',
            'high-cpu': 'warning',
            'high-memory': 'warning',
            'high-error-rate': 'critical'
        };
        
        return severityMap[type] || 'info';
    }

    generateSampleLogs() {
        const logLevels = ['info', 'warn', 'error', 'debug'];
        const sampleMessages = [
            'User login successful',
            'File processed successfully',
            'Database query completed',
            'Cache miss occurred',
            'Service health check passed',
            'Memory usage normal',
            'Network connection established',
            'Backup completed successfully',
            'Configuration updated',
            'Performance monitoring active'
        ];

        setInterval(() => {
            // Generate 1-3 random logs
            const logCount = Math.floor(Math.random() * 3) + 1;
            
            for (let i = 0; i < logCount; i++) {
                const level = logLevels[Math.floor(Math.random() * logLevels.length)];
                const message = sampleMessages[Math.floor(Math.random() * sampleMessages.length)];
                
                const logEntry = {
                    timestamp: new Date().toISOString(),
                    level,
                    message,
                    service: ['dashboard', 'api', 'worker'][Math.floor(Math.random() * 3)]
                };
                
                this.state.logs.unshift(logEntry);
                
                // Keep only last 200 logs
                if (this.state.logs.length > 200) {
                    this.state.logs = this.state.logs.slice(0, 200);
                }
                
                // Emit to connected clients
                this.io.emit('log', logEntry);
            }
        }, 2000 + Math.random() * 3000);
    }
}

// Start dashboard if run directly
if (require.main === module) {
    const dashboard = new SimpleMonitoringDashboard();
    
    // Graceful shutdown
    process.on('SIGTERM', () => {
        console.log('🛑 Shutting down monitoring dashboard');
        process.exit(0);
    });
    
    process.on('SIGINT', () => {
        console.log('🛑 Shutting down monitoring dashboard');
        process.exit(0);
    });
}

module.exports = SimpleMonitoringDashboard;