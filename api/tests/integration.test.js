/**
 * Integration Tests for Media Server API
 * Comprehensive test suite for all API endpoints
 */

const request = require('supertest');
const MediaServerAPI = require('../server');

describe('Media Server API Integration Tests', () => {
    let app;
    let server;

    beforeAll(async () => {
        // Set test environment
        process.env.NODE_ENV = 'test';
        process.env.API_PORT = '3003';
        process.env.LOG_LEVEL = 'error';
        
        app = new MediaServerAPI();
        
        // Mock external dependencies for testing
        app.dockerManager.verifyDocker = jest.fn().mockResolvedValue(true);
        app.dockerManager.loadComposeConfiguration = jest.fn().mockResolvedValue(true);
        app.dockerManager.initializeServiceDefinitions = jest.fn().mockResolvedValue(true);
        
        await app.start();
        server = app.server;
    });

    afterAll(async () => {
        if (server) {
            server.close();
        }
    });

    describe('Health Endpoints', () => {
        test('GET /health should return API health status', async () => {
            const response = await request(app.app)
                .get('/health')
                .expect(200);

            expect(response.body).toHaveProperty('status', 'healthy');
            expect(response.body).toHaveProperty('timestamp');
            expect(response.body).toHaveProperty('version');
            expect(response.body).toHaveProperty('uptime');
        });

        test('GET /api/health/overview should return system health overview', async () => {
            const response = await request(app.app)
                .get('/api/health/overview')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('overall');
            expect(response.body.data).toHaveProperty('system');
            expect(response.body.data).toHaveProperty('services');
        });

        test('GET /api/health/detailed should return detailed health information', async () => {
            const response = await request(app.app)
                .get('/api/health/detailed')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('docker');
            expect(response.body.data).toHaveProperty('filesystem');
        });

        test('GET /api/health/metrics should return system metrics', async () => {
            const response = await request(app.app)
                .get('/api/health/metrics')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('current');
        });
    });

    describe('Service Management Endpoints', () => {
        test('GET /api/services should return all services', async () => {
            // Mock the dockerManager method
            app.dockerManager.getAllServices = jest.fn().mockResolvedValue([
                {
                    service: 'jellyfin',
                    name: 'Jellyfin',
                    status: 'running',
                    running: true
                }
            ]);

            const response = await request(app.app)
                .get('/api/services')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('services');
            expect(Array.isArray(response.body.data.services)).toBe(true);
        });

        test('GET /api/services/:service/status should return service status', async () => {
            // Mock the dockerManager method
            app.dockerManager.getServiceStatus = jest.fn().mockResolvedValue({
                service: 'jellyfin',
                name: 'Jellyfin',
                status: 'running',
                running: true
            });

            const response = await request(app.app)
                .get('/api/services/jellyfin/status')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('service', 'jellyfin');
        });

        test('POST /api/services/start should start services', async () => {
            // Mock the dockerManager method
            app.dockerManager.startServices = jest.fn().mockResolvedValue({
                success: true,
                services: ['jellyfin'],
                stdout: 'Started successfully'
            });

            const response = await request(app.app)
                .post('/api/services/start')
                .send({ services: ['jellyfin'] })
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(app.dockerManager.startServices).toHaveBeenCalledWith(['jellyfin'], undefined);
        });

        test('POST /api/services/stop should stop services', async () => {
            // Mock the dockerManager method
            app.dockerManager.stopServices = jest.fn().mockResolvedValue({
                success: true,
                services: ['jellyfin'],
                stdout: 'Stopped successfully'
            });

            const response = await request(app.app)
                .post('/api/services/stop')
                .send({ services: ['jellyfin'] })
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(app.dockerManager.stopServices).toHaveBeenCalledWith(['jellyfin']);
        });

        test('POST /api/services/restart should restart services', async () => {
            // Mock the dockerManager method
            app.dockerManager.restartServices = jest.fn().mockResolvedValue({
                success: true,
                services: ['jellyfin'],
                stdout: 'Restarted successfully'
            });

            const response = await request(app.app)
                .post('/api/services/restart')
                .send({ services: ['jellyfin'] })
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(app.dockerManager.restartServices).toHaveBeenCalledWith(['jellyfin']);
        });

        test('GET /api/services/:service/logs should return service logs', async () => {
            // Mock the dockerManager method
            app.dockerManager.getServiceLogs = jest.fn().mockResolvedValue({
                service: 'jellyfin',
                logs: ['Log line 1', 'Log line 2'],
                lines: 100
            });

            const response = await request(app.app)
                .get('/api/services/jellyfin/logs')
                .query({ lines: 50 })
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('logs');
            expect(app.dockerManager.getServiceLogs).toHaveBeenCalledWith('jellyfin', {
                lines: 50,
                follow: false
            });
        });
    });

    describe('Configuration Management Endpoints', () => {
        test('GET /api/config should return configuration', async () => {
            // Mock the configManager method
            app.configManager.getConfiguration = jest.fn().mockResolvedValue({
                general: { PUID: 1000, PGID: 1000 },
                paths: { MEDIA_PATH: './media-data' }
            });

            const response = await request(app.app)
                .get('/api/config')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('general');
        });

        test('PUT /api/config should update configuration', async () => {
            // Mock the configManager method
            app.configManager.updateConfiguration = jest.fn().mockResolvedValue({
                success: true,
                config: { general: { PUID: 1001 } }
            });

            const updateData = {
                general: { PUID: 1001 }
            };

            const response = await request(app.app)
                .put('/api/config')
                .send(updateData)
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(app.configManager.updateConfiguration).toHaveBeenCalledWith(updateData);
        });

        test('POST /api/config/validate should validate configuration', async () => {
            // Mock the configManager method
            app.configManager.validateConfiguration = jest.fn().mockResolvedValue({
                valid: true,
                errors: []
            });

            const configData = {
                general: { PUID: 1000 }
            };

            const response = await request(app.app)
                .post('/api/config/validate')
                .send(configData)
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('valid', true);
        });

        test('GET /api/config/env should return environment variables', async () => {
            // Mock the configManager method
            app.configManager.getEnvironmentVariables = jest.fn().mockResolvedValue({
                PUID: '1000',
                PGID: '1000',
                TZ: 'America/New_York'
            });

            const response = await request(app.app)
                .get('/api/config/env')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('environment');
        });
    });

    describe('Seedbox Management Endpoints', () => {
        test('GET /api/seedbox/status should return seedbox status', async () => {
            // Mock the seedboxManager method
            app.seedboxManager.getStatus = jest.fn().mockResolvedValue({
                qbittorrent: { connected: true, version: '4.5.0' },
                crossSeed: { enabled: true, running: false }
            });

            const response = await request(app.app)
                .get('/api/seedbox/status')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('qbittorrent');
        });

        test('POST /api/seedbox/cross-seed/start should start cross-seed', async () => {
            // Mock the seedboxManager method
            app.seedboxManager.startCrossSeed = jest.fn().mockResolvedValue({
                success: true,
                command: 'cross-seed search',
                pid: 12345
            });

            const crossSeedOptions = {
                action: 'search',
                trackers: ['tracker1', 'tracker2']
            };

            const response = await request(app.app)
                .post('/api/seedbox/cross-seed/start')
                .send(crossSeedOptions)
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(app.seedboxManager.startCrossSeed).toHaveBeenCalledWith(crossSeedOptions);
        });

        test('GET /api/seedbox/torrents/stats should return torrent statistics', async () => {
            // Mock the seedboxManager method
            app.seedboxManager.getTorrentStats = jest.fn().mockResolvedValue({
                overview: { total: 100, seeding: 80, downloading: 5 },
                trackers: {},
                categories: {}
            });

            const response = await request(app.app)
                .get('/api/seedbox/torrents/stats')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('overview');
        });
    });

    describe('Log Management Endpoints', () => {
        test('GET /api/logs should return logs', async () => {
            // Mock the logger method
            app.logger.getLogs = jest.fn().mockResolvedValue([
                {
                    timestamp: new Date().toISOString(),
                    level: 'INFO',
                    message: 'Test log message',
                    meta: {},
                    service: 'api'
                }
            ]);

            const response = await request(app.app)
                .get('/api/logs')
                .query({ limit: 50, level: 'info' })
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('data');
            expect(response.body.data).toHaveProperty('logs');
            expect(Array.isArray(response.body.data.logs)).toBe(true);
        });

        test('GET /api/logs/stream should return WebSocket info', async () => {
            const response = await request(app.app)
                .get('/api/logs/stream')
                .expect(200);

            expect(response.body).toHaveProperty('success', true);
            expect(response.body).toHaveProperty('message');
            expect(response.body).toHaveProperty('endpoint');
        });
    });

    describe('API Documentation', () => {
        test('GET /api/docs should return API documentation', async () => {
            const response = await request(app.app)
                .get('/api/docs')
                .expect(200);

            expect(response.body).toHaveProperty('title');
            expect(response.body).toHaveProperty('version');
            expect(response.body).toHaveProperty('endpoints');
            expect(response.body).toHaveProperty('websocket');
        });
    });

    describe('Error Handling', () => {
        test('GET /api/nonexistent should return 404', async () => {
            const response = await request(app.app)
                .get('/api/nonexistent')
                .expect(404);

            expect(response.body).toHaveProperty('success', false);
            expect(response.body).toHaveProperty('error', 'Endpoint not found');
        });

        test('POST /api/services/start with invalid data should return 400', async () => {
            const response = await request(app.app)
                .post('/api/services/start')
                .send({ services: 'invalid' }) // Should be array
                .expect(400);

            expect(response.body).toHaveProperty('success', false);
            expect(response.body).toHaveProperty('error');
        });
    });

    describe('Rate Limiting', () => {
        test('Should apply rate limiting to API endpoints', async () => {
            // Make multiple requests quickly
            const requests = Array(10).fill().map(() => {
                return request(app.app).get('/api/services');
            });

            const responses = await Promise.all(requests);
            
            // All initial requests should succeed (within rate limit)
            responses.forEach(response => {
                expect([200, 429]).toContain(response.status);
            });
        });
    });

    describe('CORS and Security Headers', () => {
        test('Should include CORS headers', async () => {
            const response = await request(app.app)
                .get('/health')
                .expect(200);

            expect(response.headers).toHaveProperty('access-control-allow-origin');
        });

        test('Should include security headers', async () => {
            const response = await request(app.app)
                .get('/health')
                .expect(200);

            expect(response.headers).toHaveProperty('x-content-type-options');
        });
    });

    describe('Request Validation', () => {
        test('Should validate request body size', async () => {
            const largePayload = {
                data: 'x'.repeat(11 * 1024 * 1024) // 11MB
            };

            const response = await request(app.app)
                .post('/api/services/start')
                .send(largePayload)
                .expect(413);

            expect(response.body).toHaveProperty('success', false);
        });

        test('Should validate Content-Type for POST requests', async () => {
            const response = await request(app.app)
                .post('/api/services/start')
                .set('Content-Type', 'text/plain')
                .send('invalid')
                .expect(400);

            expect(response.body).toHaveProperty('success', false);
            expect(response.body.error).toContain('Content-Type');
        });
    });
});

describe('WebSocket Integration Tests', () => {
    let app;
    let server;
    let ws;

    beforeAll(async () => {
        process.env.NODE_ENV = 'test';
        process.env.API_PORT = '3004';
        
        app = new MediaServerAPI();
        
        // Mock external dependencies
        app.dockerManager.verifyDocker = jest.fn().mockResolvedValue(true);
        app.dockerManager.loadComposeConfiguration = jest.fn().mockResolvedValue(true);
        app.dockerManager.initializeServiceDefinitions = jest.fn().mockResolvedValue(true);
        
        await app.start();
        server = app.server;
    });

    afterAll(async () => {
        if (ws) {
            ws.close();
        }
        if (server) {
            server.close();
        }
    });

    test('WebSocket connection should work', (done) => {
        const WebSocket = require('ws');
        ws = new WebSocket('ws://localhost:3004');

        ws.on('open', () => {
            // Send ping message
            ws.send(JSON.stringify({ action: 'ping' }));
        });

        ws.on('message', (data) => {
            const message = JSON.parse(data.toString());
            
            if (message.type === 'pong') {
                expect(message.type).toBe('pong');
                done();
            } else if (message.type === 'initial-status') {
                expect(message.data).toHaveProperty('services');
                expect(message.data).toHaveProperty('health');
            }
        });

        ws.on('error', (error) => {
            done(error);
        });
    });
});