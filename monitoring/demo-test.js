// Demo Test for Monitoring System
// Generates sample logs and metrics to demonstrate monitoring capabilities

const { createLogger } = require('./comprehensive-logger');
const axios = require('axios');

// Create logger instance
const logger = createLogger({
    service: 'demo-app',
    environment: 'development',
    enableConsole: true,
    enableFile: true,
    logLevel: 'debug'
});

class MonitoringDemo {
    constructor() {
        this.isRunning = false;
        this.intervalId = null;
    }

    async start() {
        console.log('🎭 Starting Monitoring Demo...');
        this.isRunning = true;

        // Initial startup logs
        logger.info('Demo application starting', {
            version: '1.0.0',
            environment: 'demo'
        });

        // Simulate various application activities
        this.simulateUserActivity();
        this.simulatePerformanceMetrics();
        this.simulateErrors();
        this.simulateSecurityEvents();
        this.simulateBusinessMetrics();

        // Keep running
        this.intervalId = setInterval(() => {
            if (!this.isRunning) {
                clearInterval(this.intervalId);
                return;
            }
            this.generateRandomActivity();
        }, 2000);
    }

    stop() {
        console.log('🛑 Stopping Monitoring Demo...');
        this.isRunning = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
        }
        logger.info('Demo application stopping');
    }

    simulateUserActivity() {
        setInterval(() => {
            if (!this.isRunning) return;

            const activities = [
                { action: 'user-login', user: `user-${Math.floor(Math.random() * 100)}` },
                { action: 'page-view', page: ['dashboard', 'profile', 'settings', 'media'][Math.floor(Math.random() * 4)] },
                { action: 'api-call', endpoint: ['/api/movies', '/api/shows', '/api/music', '/api/status'][Math.floor(Math.random() * 4)] },
                { action: 'file-download', size: Math.floor(Math.random() * 1000000) }
            ];

            const activity = activities[Math.floor(Math.random() * activities.length)];
            
            logger.info(`User activity: ${activity.action}`, {
                type: 'user-activity',
                ...activity,
                timestamp: new Date().toISOString()
            });

            // Log business metrics
            if (activity.action === 'user-login') {
                logger.logBusinessMetric('user-logins', 1);
            }
            if (activity.action === 'file-download') {
                logger.logBusinessMetric('data-transfer', activity.size);
            }

        }, 3000 + Math.random() * 2000);
    }

    simulatePerformanceMetrics() {
        setInterval(() => {
            if (!this.isRunning) return;

            // Simulate database query
            const dbTimer = logger.startTimer('database-query');
            setTimeout(() => {
                logger.endTimer(dbTimer, {
                    query: 'SELECT * FROM media_files WHERE status = ?',
                    rows: Math.floor(Math.random() * 500)
                });
            }, Math.random() * 200);

            // Simulate API response
            const apiTimer = logger.startTimer('api-response');
            setTimeout(() => {
                logger.endTimer(apiTimer, {
                    endpoint: '/api/search',
                    statusCode: Math.random() > 0.9 ? 500 : 200
                });
            }, Math.random() * 100);

            // Log performance metrics
            logger.logBusinessMetric('response-time', Math.random() * 500);
            logger.logBusinessMetric('concurrent-users', Math.floor(Math.random() * 50));

        }, 5000);
    }

    simulateErrors() {
        setInterval(() => {
            if (!this.isRunning) return;

            // Randomly generate different types of errors
            const errorTypes = [
                {
                    type: 'ValidationError',
                    message: 'Invalid file format uploaded',
                    severity: 'medium'
                },
                {
                    type: 'NetworkError',
                    message: 'Connection timeout to external service',
                    severity: 'high'
                },
                {
                    type: 'DatabaseError',
                    message: 'Query execution failed',
                    severity: 'high'
                },
                {
                    type: 'AuthenticationError',
                    message: 'Invalid authentication token',
                    severity: 'medium'
                }
            ];

            // Generate error occasionally (10% chance)
            if (Math.random() < 0.1) {
                const error = errorTypes[Math.floor(Math.random() * errorTypes.length)];
                
                logger.error(error.message, new Error(error.message), {
                    type: error.type,
                    severity: error.severity,
                    userId: `user-${Math.floor(Math.random() * 100)}`,
                    timestamp: new Date().toISOString()
                });
            }

            // Generate warnings occasionally (20% chance)
            if (Math.random() < 0.2) {
                const warnings = [
                    'High memory usage detected',
                    'Slow query performance',
                    'Rate limit approaching',
                    'Disk space running low'
                ];

                const warning = warnings[Math.floor(Math.random() * warnings.length)];
                logger.warn(warning, {
                    type: 'performance-warning',
                    metric: Math.random() * 100
                });
            }

        }, 8000);
    }

    simulateSecurityEvents() {
        setInterval(() => {
            if (!this.isRunning) return;

            // Occasionally generate security events (5% chance)
            if (Math.random() < 0.05) {
                const securityEvents = [
                    {
                        event: 'failed-login-attempt',
                        severity: 'medium',
                        ip: `192.168.1.${Math.floor(Math.random() * 255)}`,
                        attempts: Math.floor(Math.random() * 5) + 1
                    },
                    {
                        event: 'suspicious-activity',
                        severity: 'high',
                        ip: `10.0.0.${Math.floor(Math.random() * 255)}`,
                        activity: 'Multiple rapid requests'
                    },
                    {
                        event: 'unauthorized-access-attempt',
                        severity: 'high',
                        resource: '/admin/config',
                        ip: `172.16.0.${Math.floor(Math.random() * 255)}`
                    }
                ];

                const event = securityEvents[Math.floor(Math.random() * securityEvents.length)];
                
                logger.logSecurityEvent(event.event, event.severity, {
                    ip: event.ip,
                    timestamp: new Date().toISOString(),
                    details: event
                });
            }

        }, 12000);
    }

    simulateBusinessMetrics() {
        setInterval(() => {
            if (!this.isRunning) return;

            // Generate business metrics
            const metrics = [
                { name: 'active-streams', value: Math.floor(Math.random() * 25) },
                { name: 'downloads-completed', value: Math.floor(Math.random() * 10) },
                { name: 'storage-used-gb', value: Math.floor(Math.random() * 1000) },
                { name: 'bandwidth-mbps', value: Math.floor(Math.random() * 100) },
                { name: 'user-satisfaction-score', value: Math.random() * 5 }
            ];

            metrics.forEach(metric => {
                logger.logBusinessMetric(metric.name, metric.value);
            });

        }, 15000);
    }

    generateRandomActivity() {
        const activities = [
            () => logger.info('Media file processed', { 
                file: `movie-${Math.floor(Math.random() * 1000)}.mkv`,
                size: Math.floor(Math.random() * 5000000000),
                quality: ['720p', '1080p', '4K'][Math.floor(Math.random() * 3)]
            }),
            () => logger.debug('Cache hit', {
                key: `media-${Math.floor(Math.random() * 100)}`,
                ttl: Math.floor(Math.random() * 3600)
            }),
            () => logger.info('User session started', {
                sessionId: `session-${Math.floor(Math.random() * 10000)}`,
                duration: Math.floor(Math.random() * 7200)
            }),
            () => {
                // Simulate audit events
                const actions = ['file-upload', 'user-create', 'settings-change', 'permission-grant'];
                const action = actions[Math.floor(Math.random() * actions.length)];
                logger.logAudit(action, `user-${Math.floor(Math.random() * 50)}`, 'system', `resource-${Math.floor(Math.random() * 100)}`);
            }
        ];

        const activity = activities[Math.floor(Math.random() * activities.length)];
        activity();
    }

    async testMonitoringEndpoints() {
        console.log('🧪 Testing monitoring endpoints...');
        
        try {
            // Test monitoring dashboard API
            const response = await axios.get('http://localhost:3005/api/status');
            console.log('✅ Monitoring API is responding:', response.data.status);
            
            // Test metrics endpoint
            const metrics = await axios.get('http://localhost:3005/api/metrics');
            console.log('📊 Metrics endpoint working, found', Object.keys(metrics.data.current).length, 'metric categories');
            
        } catch (error) {
            console.log('⚠️  Monitoring endpoints not yet available:', error.message);
        }
    }
}

// Start demo if run directly
if (require.main === module) {
    const demo = new MonitoringDemo();
    
    // Test endpoints first
    demo.testMonitoringEndpoints();
    
    // Start demo
    demo.start();
    
    // Stop demo on Ctrl+C
    process.on('SIGINT', () => {
        demo.stop();
        process.exit(0);
    });
    
    process.on('SIGTERM', () => {
        demo.stop();
        process.exit(0);
    });
    
    console.log('🎭 Demo running... Press Ctrl+C to stop');
    console.log('📊 Check http://localhost:3005 for live monitoring dashboard');
}

module.exports = MonitoringDemo;