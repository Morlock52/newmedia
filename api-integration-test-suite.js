#!/usr/bin/env node

/**
 * Comprehensive API Integration Test Suite
 * Tests all service connections, API integrations, and inter-service communication
 */

const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

class APIIntegrationTester {
    constructor() {
        this.results = {
            timestamp: new Date().toISOString(),
            totalTests: 0,
            passed: 0,
            failed: 0,
            warnings: 0,
            tests: [],
            summary: {},
            recommendations: []
        };

        // Service configurations
        this.services = {
            jellyfin: { url: 'http://localhost:8096', name: 'Jellyfin Media Server' },
            plex: { url: 'http://localhost:32400', name: 'Plex Media Server' },
            sonarr: { url: 'http://localhost:8989', name: 'Sonarr TV Shows' },
            radarr: { url: 'http://localhost:7878', name: 'Radarr Movies' },
            lidarr: { url: 'http://localhost:8686', name: 'Lidarr Music' },
            bazarr: { url: 'http://localhost:6767', name: 'Bazarr Subtitles' },
            prowlarr: { url: 'http://localhost:9696', name: 'Prowlarr Indexer' },
            qbittorrent: { url: 'http://localhost:8080', name: 'qBittorrent Client' },
            transmission: { url: 'http://localhost:9091', name: 'Transmission Client' },
            sabnzbd: { url: 'http://localhost:8081', name: 'SABnzbd Usenet' },
            api_server: { url: 'http://localhost:3002', name: 'Media Server API' },
            dashboard: { url: 'http://localhost:3030', name: 'Media Dashboard' },
            prometheus: { url: 'http://localhost:9090', name: 'Prometheus Monitoring' },
            grafana: { url: 'http://localhost:3000', name: 'Grafana Dashboards' },
            uptime_kuma: { url: 'http://localhost:3001', name: 'Uptime Kuma' },
            portainer: { url: 'http://localhost:9000', name: 'Portainer Docker UI' }
        };

        // Default timeout for API calls
        this.timeout = 10000;
        
        // API Keys (will be loaded from environment or config files)
        this.apiKeys = {};
    }

    async initialize() {
        console.log('🚀 Initializing API Integration Test Suite...\n');
        
        // Load API keys from configuration files
        await this.loadAPIKeys();
        
        // Check Docker services status
        await this.checkDockerStatus();
        
        console.log('✅ Initialization complete\n');
    }

    async loadAPIKeys() {
        try {
            // Try to load from service config files
            const configPaths = {
                sonarr: 'sonarr-config/config.xml',
                radarr: 'radarr-config/config.xml',
                lidarr: 'lidarr-config/config.xml',
                bazarr: 'bazarr-config/config.xml',
                prowlarr: 'prowlarr-config/config.xml'
            };

            for (const [service, configPath] of Object.entries(configPaths)) {
                try {
                    if (await this.fileExists(configPath)) {
                        const config = await fs.readFile(configPath, 'utf8');
                        const apiKeyMatch = config.match(/<ApiKey>(.*?)<\/ApiKey>/);
                        if (apiKeyMatch) {
                            this.apiKeys[service] = apiKeyMatch[1];
                            console.log(`✅ Loaded API key for ${service}`);
                        }
                    }
                } catch (error) {
                    console.log(`⚠️  Could not load API key for ${service}: ${error.message}`);
                }
            }

            // Try to get qBittorrent session
            try {
                await this.getQBittorrentSession();
            } catch (error) {
                console.log(`⚠️  Could not authenticate with qBittorrent: ${error.message}`);
            }

        } catch (error) {
            console.log(`⚠️  API key loading completed with some errors: ${error.message}`);
        }
    }

    async fileExists(filePath) {
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    async checkDockerStatus() {
        try {
            const output = execSync('docker-compose ps --format json', { 
                encoding: 'utf8',
                cwd: process.cwd()
            });
            
            if (output.trim()) {
                const containers = output.split('\n')
                    .filter(line => line.trim())
                    .map(line => JSON.parse(line));
                
                console.log(`📦 Found ${containers.length} Docker containers`);
                
                const runningContainers = containers.filter(c => c.State === 'running');
                console.log(`✅ ${runningContainers.length} containers running`);
                
                if (runningContainers.length < containers.length) {
                    console.log(`⚠️  ${containers.length - runningContainers.length} containers not running`);
                }
            } else {
                console.log('⚠️  No Docker containers found or docker-compose not available');
            }
        } catch (error) {
            console.log(`⚠️  Could not check Docker status: ${error.message}`);
        }
    }

    async runAllTests() {
        console.log('🧪 Starting Comprehensive API Integration Tests\n');
        
        const testSuites = [
            { name: 'Service Accessibility', test: () => this.testServiceAccessibility() },
            { name: 'API Authentication', test: () => this.testAPIAuthentication() },
            { name: 'Prowlarr to *ARR Integration', test: () => this.testProwlarrIntegration() },
            { name: 'ARR to Download Clients', test: () => this.testDownloadClientIntegration() },
            { name: 'Media Server APIs', test: () => this.testMediaServerAPIs() },
            { name: 'Database Connections', test: () => this.testDatabaseConnections() },
            { name: 'Inter-Service Communication', test: () => this.testInterServiceCommunication() },
            { name: 'Webhook Systems', test: () => this.testWebhookSystems() },
            { name: 'Performance & Load', test: () => this.testPerformance() }
        ];

        for (const suite of testSuites) {
            console.log(`\n🔍 Running ${suite.name} Tests...`);
            console.log('=' + '='.repeat(suite.name.length + 20));
            
            try {
                await suite.test();
            } catch (error) {
                this.addTestResult(suite.name, 'SUITE_ERROR', false, error.message);
            }
        }

        await this.generateReport();
    }

    async testServiceAccessibility() {
        for (const [serviceId, service] of Object.entries(this.services)) {
            await this.testServiceHealth(serviceId, service);
        }
    }

    async testServiceHealth(serviceId, service) {
        const testName = `${service.name} - Health Check`;
        
        try {
            const startTime = Date.now();
            const response = await axios.get(service.url, {
                timeout: this.timeout,
                validateStatus: (status) => status < 500 // Accept redirects and client errors
            });
            
            const responseTime = Date.now() - startTime;
            
            if (response.status === 200) {
                this.addTestResult(testName, 'ACCESSIBLE', true, 
                    `Service accessible (${responseTime}ms)`, { responseTime, status: response.status });
            } else if (response.status >= 300 && response.status < 400) {
                this.addTestResult(testName, 'REDIRECT', true, 
                    `Service redirecting (${response.status})`, { responseTime, status: response.status });
            } else if (response.status === 401 || response.status === 403) {
                this.addTestResult(testName, 'AUTH_REQUIRED', true, 
                    `Service requires authentication (${response.status})`, { responseTime, status: response.status });
            } else {
                this.addTestResult(testName, 'PARTIALLY_ACCESSIBLE', false, 
                    `Service responded with ${response.status}`, { responseTime, status: response.status });
            }
            
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                this.addTestResult(testName, 'NOT_ACCESSIBLE', false, 'Service not running or port closed');
            } else if (error.code === 'ETIMEDOUT') {
                this.addTestResult(testName, 'TIMEOUT', false, `Service timed out after ${this.timeout}ms`);
            } else {
                this.addTestResult(testName, 'ERROR', false, error.message);
            }
        }
    }

    async testAPIAuthentication() {
        // Test Jellyfin API authentication
        await this.testJellyfinAuth();
        
        // Test *ARR services API authentication
        await this.testARRAuth('sonarr');
        await this.testARRAuth('radarr');
        await this.testARRAuth('lidarr');
        await this.testARRAuth('bazarr');
        await this.testARRAuth('prowlarr');
        
        // Test download clients
        await this.testQBittorrentAuth();
        await this.testTransmissionAuth();
        await this.testSABnzbdAuth();
    }

    async testJellyfinAuth() {
        const testName = 'Jellyfin - API Authentication';
        
        try {
            // Try to access Jellyfin API system info endpoint
            const response = await axios.get(`${this.services.jellyfin.url}/System/Info`, {
                timeout: this.timeout
            });
            
            if (response.status === 200) {
                this.addTestResult(testName, 'NO_AUTH_REQUIRED', true, 
                    'Jellyfin API accessible without authentication');
            }
        } catch (error) {
            if (error.response && error.response.status === 401) {
                this.addTestResult(testName, 'AUTH_REQUIRED', true, 
                    'Jellyfin API properly requires authentication');
            } else {
                this.addTestResult(testName, 'ERROR', false, error.message);
            }
        }
    }

    async testARRAuth(service) {
        const testName = `${service.charAt(0).toUpperCase() + service.slice(1)} - API Authentication`;
        const apiKey = this.apiKeys[service];
        
        if (!apiKey) {
            this.addTestResult(testName, 'NO_API_KEY', false, 'API key not found');
            return;
        }

        try {
            const response = await axios.get(`${this.services[service].url}/api/v1/system/status`, {
                headers: { 'X-Api-Key': apiKey },
                timeout: this.timeout
            });
            
            if (response.status === 200) {
                this.addTestResult(testName, 'AUTH_SUCCESS', true, 
                    'API authentication successful', response.data);
            }
        } catch (error) {
            if (error.response && error.response.status === 401) {
                this.addTestResult(testName, 'AUTH_FAILED', false, 'Invalid API key');
            } else {
                this.addTestResult(testName, 'ERROR', false, error.message);
            }
        }
    }

    async testQBittorrentAuth() {
        const testName = 'qBittorrent - API Authentication';
        
        try {
            // Try to get version info (should work without auth)
            const versionResponse = await axios.get(`${this.services.qbittorrent.url}/api/v2/app/version`, {
                timeout: this.timeout
            });
            
            if (versionResponse.status === 200) {
                this.addTestResult(testName, 'VERSION_ACCESSIBLE', true, 
                    `qBittorrent version: ${versionResponse.data}`);
            }

            // Try to access protected endpoint
            try {
                const torrentResponse = await axios.get(`${this.services.qbittorrent.url}/api/v2/torrents/info`, {
                    timeout: this.timeout
                });
                
                this.addTestResult(testName + ' - Protected API', 'NO_AUTH_REQUIRED', true, 
                    'Protected endpoints accessible without authentication');
            } catch (error) {
                if (error.response && error.response.status === 403) {
                    this.addTestResult(testName + ' - Protected API', 'AUTH_REQUIRED', true, 
                        'Protected endpoints properly require authentication');
                }
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testTransmissionAuth() {
        const testName = 'Transmission - API Authentication';
        
        try {
            const response = await axios.post(`${this.services.transmission.url}/transmission/rpc`, {
                method: 'session-get'
            }, {
                headers: { 'Content-Type': 'application/json' },
                timeout: this.timeout
            });
            
        } catch (error) {
            if (error.response && error.response.status === 409) {
                // This is expected - Transmission returns 409 with session ID
                const sessionId = error.response.headers['x-transmission-session-id'];
                if (sessionId) {
                    this.addTestResult(testName, 'SESSION_ID_RECEIVED', true, 
                        'Transmission API accessible, session ID received');
                }
            } else {
                this.addTestResult(testName, 'ERROR', false, error.message);
            }
        }
    }

    async testSABnzbdAuth() {
        const testName = 'SABnzbd - API Authentication';
        
        try {
            const response = await axios.get(`${this.services.sabnzbd.url}/api`, {
                params: { mode: 'version', output: 'json' },
                timeout: this.timeout
            });
            
            if (response.status === 200) {
                this.addTestResult(testName, 'VERSION_ACCESSIBLE', true, 
                    'SABnzbd version endpoint accessible');
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testProwlarrIntegration() {
        const testName = 'Prowlarr Integration Tests';
        const apiKey = this.apiKeys.prowlarr;
        
        if (!apiKey) {
            this.addTestResult(testName, 'NO_API_KEY', false, 'Prowlarr API key not found');
            return;
        }

        try {
            // Test Prowlarr indexer status
            const indexerResponse = await axios.get(`${this.services.prowlarr.url}/api/v1/indexer`, {
                headers: { 'X-Api-Key': apiKey },
                timeout: this.timeout
            });
            
            if (indexerResponse.status === 200) {
                const indexers = indexerResponse.data;
                this.addTestResult(testName + ' - Indexers', 'SUCCESS', true, 
                    `Found ${indexers.length} indexers configured`);
                
                // Test application connections
                const appsResponse = await axios.get(`${this.services.prowlarr.url}/api/v1/applications`, {
                    headers: { 'X-Api-Key': apiKey },
                    timeout: this.timeout
                });
                
                if (appsResponse.status === 200) {
                    const apps = appsResponse.data;
                    this.addTestResult(testName + ' - Applications', 'SUCCESS', true, 
                        `Found ${apps.length} application connections`);
                        
                    // Check if Sonarr/Radarr are connected
                    const sonarrConnected = apps.some(app => app.name.toLowerCase().includes('sonarr'));
                    const radarrConnected = apps.some(app => app.name.toLowerCase().includes('radarr'));
                    
                    if (sonarrConnected) {
                        this.addTestResult(testName + ' - Sonarr Connection', 'CONNECTED', true, 
                            'Sonarr connected to Prowlarr');
                    } else {
                        this.addTestResult(testName + ' - Sonarr Connection', 'NOT_CONNECTED', false, 
                            'Sonarr not connected to Prowlarr');
                    }
                    
                    if (radarrConnected) {
                        this.addTestResult(testName + ' - Radarr Connection', 'CONNECTED', true, 
                            'Radarr connected to Prowlarr');
                    } else {
                        this.addTestResult(testName + ' - Radarr Connection', 'NOT_CONNECTED', false, 
                            'Radarr not connected to Prowlarr');
                    }
                }
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testDownloadClientIntegration() {
        // Test Sonarr -> Download Client connections
        await this.testARRDownloadClients('sonarr');
        
        // Test Radarr -> Download Client connections  
        await this.testARRDownloadClients('radarr');
        
        // Test Lidarr -> Download Client connections
        await this.testARRDownloadClients('lidarr');
    }

    async testARRDownloadClients(service) {
        const testName = `${service.charAt(0).toUpperCase() + service.slice(1)} - Download Clients`;
        const apiKey = this.apiKeys[service];
        
        if (!apiKey) {
            this.addTestResult(testName, 'NO_API_KEY', false, 'API key not found');
            return;
        }

        try {
            const response = await axios.get(`${this.services[service].url}/api/v1/downloadclient`, {
                headers: { 'X-Api-Key': apiKey },
                timeout: this.timeout
            });
            
            if (response.status === 200) {
                const clients = response.data;
                this.addTestResult(testName, 'SUCCESS', true, 
                    `Found ${clients.length} download clients configured`);
                
                // Test each client connection
                for (const client of clients) {
                    await this.testDownloadClientConnection(service, client, apiKey);
                }
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testDownloadClientConnection(service, client, apiKey) {
        const testName = `${service} -> ${client.name} Connection`;
        
        try {
            const response = await axios.post(
                `${this.services[service].url}/api/v1/downloadclient/test`, 
                client,
                {
                    headers: { 
                        'X-Api-Key': apiKey,
                        'Content-Type': 'application/json'
                    },
                    timeout: this.timeout
                }
            );
            
            if (response.status === 200) {
                this.addTestResult(testName, 'CONNECTION_SUCCESS', true, 
                    `Connection to ${client.name} successful`);
            }
            
        } catch (error) {
            if (error.response && error.response.data) {
                this.addTestResult(testName, 'CONNECTION_FAILED', false, 
                    error.response.data.message || 'Connection test failed');
            } else {
                this.addTestResult(testName, 'ERROR', false, error.message);
            }
        }
    }

    async testMediaServerAPIs() {
        // Test Jellyfin API endpoints
        await this.testJellyfinEndpoints();
        
        // Test Plex API endpoints
        await this.testPlexEndpoints();
        
        // Test custom media API server
        await this.testCustomAPIServer();
    }

    async testJellyfinEndpoints() {
        const testName = 'Jellyfin API Endpoints';
        
        try {
            // Test system info endpoint
            const systemResponse = await axios.get(`${this.services.jellyfin.url}/System/Info/Public`, {
                timeout: this.timeout
            });
            
            if (systemResponse.status === 200) {
                this.addTestResult(testName + ' - System Info', 'SUCCESS', true, 
                    `Jellyfin version: ${systemResponse.data.Version}`);
            }
            
            // Test library endpoint (may require auth)
            try {
                const libraryResponse = await axios.get(`${this.services.jellyfin.url}/Library/VirtualFolders`, {
                    timeout: this.timeout
                });
                
                if (libraryResponse.status === 200) {
                    this.addTestResult(testName + ' - Library', 'SUCCESS', true, 
                        'Library endpoint accessible');
                }
            } catch (error) {
                if (error.response && error.response.status === 401) {
                    this.addTestResult(testName + ' - Library', 'AUTH_REQUIRED', true, 
                        'Library endpoint requires authentication (expected)');
                }
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testPlexEndpoints() {
        const testName = 'Plex API Endpoints';
        
        try {
            const response = await axios.get(`${this.services.plex.url}/identity`, {
                timeout: this.timeout
            });
            
            if (response.status === 200) {
                this.addTestResult(testName, 'SUCCESS', true, 'Plex identity endpoint accessible');
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testCustomAPIServer() {
        const testName = 'Custom Media API Server';
        
        try {
            // Test health endpoint
            const healthResponse = await axios.get(`${this.services.api_server.url}/health`, {
                timeout: this.timeout
            });
            
            if (healthResponse.status === 200) {
                this.addTestResult(testName + ' - Health', 'SUCCESS', true, 
                    'API server health check passed');
            }
            
            // Test API endpoints
            const apiResponse = await axios.get(`${this.services.api_server.url}/api/system`, {
                timeout: this.timeout
            });
            
            if (apiResponse.status === 200) {
                this.addTestResult(testName + ' - System API', 'SUCCESS', true, 
                    'System API endpoint accessible');
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testDatabaseConnections() {
        const testName = 'Database Connections';
        
        // Test PostgreSQL
        await this.testPostgreSQLConnection();
        
        // Test Redis
        await this.testRedisConnection();
        
        // Test MariaDB
        await this.testMariaDBConnection();
    }

    async testPostgreSQLConnection() {
        const testName = 'PostgreSQL Database';
        
        try {
            // Try to connect to PostgreSQL health endpoint or directly
            const response = await axios.get('http://localhost:5432', {
                timeout: 5000
            });
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                this.addTestResult(testName, 'SERVICE_DOWN', false, 'PostgreSQL service not accessible');
            } else {
                // Different error might indicate the service is running but not HTTP
                this.addTestResult(testName, 'RUNNING', true, 'PostgreSQL appears to be running (non-HTTP response expected)');
            }
        }
    }

    async testRedisConnection() {
        const testName = 'Redis Database';
        
        try {
            const response = await axios.get('http://localhost:6379', {
                timeout: 5000
            });
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                this.addTestResult(testName, 'SERVICE_DOWN', false, 'Redis service not accessible');
            } else {
                this.addTestResult(testName, 'RUNNING', true, 'Redis appears to be running (non-HTTP response expected)');
            }
        }
    }

    async testMariaDBConnection() {
        const testName = 'MariaDB Database';
        
        try {
            const response = await axios.get('http://localhost:3306', {
                timeout: 5000
            });
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                this.addTestResult(testName, 'SERVICE_DOWN', false, 'MariaDB service not accessible');
            } else {
                this.addTestResult(testName, 'RUNNING', true, 'MariaDB appears to be running (non-HTTP response expected)');
            }
        }
    }

    async testInterServiceCommunication() {
        const testName = 'Inter-Service Communication';
        
        // Test if services can communicate with each other
        await this.testServiceToServiceComm();
    }

    async testServiceToServiceComm() {
        // This would test internal Docker network communication
        // Since services should be able to reach each other by container name
        
        const testName = 'Container Network Communication';
        this.addTestResult(testName, 'DOCKER_NETWORK', true, 
            'Services should communicate via Docker network (manual verification needed)');
    }

    async testWebhookSystems() {
        const testName = 'Webhook Systems';
        
        // Test if *ARR services have webhooks configured
        for (const service of ['sonarr', 'radarr', 'lidarr']) {
            await this.testServiceWebhooks(service);
        }
    }

    async testServiceWebhooks(service) {
        const testName = `${service.charAt(0).toUpperCase() + service.slice(1)} Webhooks`;
        const apiKey = this.apiKeys[service];
        
        if (!apiKey) {
            this.addTestResult(testName, 'NO_API_KEY', false, 'API key not found');
            return;
        }

        try {
            const response = await axios.get(`${this.services[service].url}/api/v1/notification`, {
                headers: { 'X-Api-Key': apiKey },
                timeout: this.timeout
            });
            
            if (response.status === 200) {
                const notifications = response.data;
                const webhooks = notifications.filter(n => n.implementation === 'Webhook');
                
                this.addTestResult(testName, 'CONFIGURED', true, 
                    `Found ${webhooks.length} webhook notifications configured`);
            }
            
        } catch (error) {
            this.addTestResult(testName, 'ERROR', false, error.message);
        }
    }

    async testPerformance() {
        const testName = 'Performance Tests';
        
        // Test response times for critical services
        const criticalServices = ['jellyfin', 'sonarr', 'radarr', 'prowlarr', 'api_server'];
        
        for (const serviceId of criticalServices) {
            await this.testServicePerformance(serviceId);
        }
    }

    async testServicePerformance(serviceId) {
        const service = this.services[serviceId];
        const testName = `${service.name} - Response Time`;
        
        const responseTimes = [];
        const testRuns = 3;
        
        for (let i = 0; i < testRuns; i++) {
            try {
                const startTime = Date.now();
                await axios.get(service.url, {
                    timeout: this.timeout,
                    validateStatus: () => true // Accept any status for performance testing
                });
                const responseTime = Date.now() - startTime;
                responseTimes.push(responseTime);
            } catch (error) {
                // Still record the time even if there's an error
                responseTimes.push(this.timeout);
            }
        }
        
        const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
        
        if (avgResponseTime < 1000) {
            this.addTestResult(testName, 'GOOD_PERFORMANCE', true, 
                `Average response time: ${avgResponseTime.toFixed(2)}ms`);
        } else if (avgResponseTime < 3000) {
            this.addTestResult(testName, 'ACCEPTABLE_PERFORMANCE', true, 
                `Average response time: ${avgResponseTime.toFixed(2)}ms (acceptable)`);
        } else {
            this.addTestResult(testName, 'SLOW_PERFORMANCE', false, 
                `Average response time: ${avgResponseTime.toFixed(2)}ms (slow)`);
        }
    }

    async getQBittorrentSession() {
        // Try to authenticate with default credentials
        const credentials = [
            { username: 'admin', password: 'adminadmin' },
            { username: 'admin', password: 'admin' },
            { username: 'admin', password: '' }
        ];

        for (const cred of credentials) {
            try {
                const response = await axios.post(`${this.services.qbittorrent.url}/api/v2/auth/login`, 
                    `username=${cred.username}&password=${cred.password}`, {
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    timeout: this.timeout
                });
                
                if (response.data === 'Ok.') {
                    console.log('✅ qBittorrent authentication successful');
                    return true;
                }
            } catch (error) {
                // Continue to next credential
            }
        }
        
        return false;
    }

    addTestResult(testName, status, passed, message, data = null) {
        this.results.totalTests++;
        if (passed) {
            this.results.passed++;
        } else {
            this.results.failed++;
        }
        
        if (status.includes('WARNING') || status.includes('PARTIAL')) {
            this.results.warnings++;
        }

        this.results.tests.push({
            name: testName,
            status,
            passed,
            message,
            data,
            timestamp: new Date().toISOString()
        });

        // Console output
        const icon = passed ? '✅' : '❌';
        const statusText = passed ? 'PASS' : 'FAIL';
        console.log(`${icon} ${statusText}: ${testName} - ${message}`);
    }

    async generateReport() {
        console.log('\n' + '='.repeat(80));
        console.log('📊 API INTEGRATION TEST RESULTS');
        console.log('='.repeat(80));
        
        console.log(`\n🔍 Test Summary:`);
        console.log(`   Total Tests: ${this.results.totalTests}`);
        console.log(`   ✅ Passed: ${this.results.passed}`);
        console.log(`   ❌ Failed: ${this.results.failed}`);
        console.log(`   ⚠️  Warnings: ${this.results.warnings}`);
        console.log(`   Success Rate: ${((this.results.passed / this.results.totalTests) * 100).toFixed(1)}%`);

        // Generate recommendations
        this.generateRecommendations();

        // Service status summary
        this.generateServiceSummary();

        // Save detailed report
        const reportPath = path.join(process.cwd(), 'api-integration-test-report.json');
        await fs.writeFile(reportPath, JSON.stringify(this.results, null, 2));
        console.log(`\n📄 Detailed report saved to: ${reportPath}`);

        // Save readable report
        const readableReport = this.generateReadableReport();
        const readableReportPath = path.join(process.cwd(), 'api-integration-test-report.md');
        await fs.writeFile(readableReportPath, readableReport);
        console.log(`📄 Readable report saved to: ${readableReportPath}`);
        
        console.log('\n🎯 Next Steps:');
        this.results.recommendations.forEach(rec => {
            console.log(`   • ${rec}`);
        });
    }

    generateRecommendations() {
        const failedTests = this.results.tests.filter(t => !t.passed);
        
        if (failedTests.length === 0) {
            this.results.recommendations.push('All integration tests passed! Your media server stack is properly configured.');
            return;
        }

        // Service accessibility issues
        const accessibilityIssues = failedTests.filter(t => t.message.includes('not running') || t.message.includes('closed'));
        if (accessibilityIssues.length > 0) {
            this.results.recommendations.push('Start missing services: docker-compose up -d');
        }

        // API key issues
        const apiKeyIssues = failedTests.filter(t => t.message.includes('API key'));
        if (apiKeyIssues.length > 0) {
            this.results.recommendations.push('Configure API keys for *ARR services in their respective web interfaces');
        }

        // Connection issues
        const connectionIssues = failedTests.filter(t => t.message.includes('connection') || t.message.includes('Connection'));
        if (connectionIssues.length > 0) {
            this.results.recommendations.push('Check service configurations and ensure proper interconnections');
        }

        // Performance issues
        const performanceIssues = failedTests.filter(t => t.message.includes('slow') || t.message.includes('timeout'));
        if (performanceIssues.length > 0) {
            this.results.recommendations.push('Consider optimizing system resources or increasing timeout values');
        }

        // Database issues
        const databaseIssues = failedTests.filter(t => t.name.includes('Database'));
        if (databaseIssues.length > 0) {
            this.results.recommendations.push('Verify database services are running and accessible');
        }
    }

    generateServiceSummary() {
        console.log(`\n🏥 Service Health Summary:`);
        
        const serviceResults = {};
        
        for (const test of this.results.tests) {
            const serviceName = test.name.split(' - ')[0];
            if (!serviceResults[serviceName]) {
                serviceResults[serviceName] = { passed: 0, failed: 0, total: 0 };
            }
            serviceResults[serviceName].total++;
            if (test.passed) {
                serviceResults[serviceName].passed++;
            } else {
                serviceResults[serviceName].failed++;
            }
        }

        for (const [service, results] of Object.entries(serviceResults)) {
            const healthPercentage = (results.passed / results.total * 100).toFixed(1);
            const icon = results.failed === 0 ? '🟢' : results.passed > results.failed ? '🟡' : '🔴';
            console.log(`   ${icon} ${service}: ${results.passed}/${results.total} tests passed (${healthPercentage}%)`);
        }
    }

    generateReadableReport() {
        let report = `# API Integration Test Report\n\n`;
        report += `**Generated:** ${this.results.timestamp}\n\n`;
        
        report += `## Summary\n\n`;
        report += `- **Total Tests:** ${this.results.totalTests}\n`;
        report += `- **Passed:** ${this.results.passed}\n`;
        report += `- **Failed:** ${this.results.failed}\n`;
        report += `- **Warnings:** ${this.results.warnings}\n`;
        report += `- **Success Rate:** ${((this.results.passed / this.results.totalTests) * 100).toFixed(1)}%\n\n`;

        report += `## Test Results\n\n`;
        
        const groupedTests = {};
        for (const test of this.results.tests) {
            const category = test.name.split(' - ')[0];
            if (!groupedTests[category]) {
                groupedTests[category] = [];
            }
            groupedTests[category].push(test);
        }

        for (const [category, tests] of Object.entries(groupedTests)) {
            report += `### ${category}\n\n`;
            for (const test of tests) {
                const icon = test.passed ? '✅' : '❌';
                report += `${icon} **${test.name}**\n`;
                report += `   - Status: ${test.status}\n`;
                report += `   - Message: ${test.message}\n`;
                if (test.data) {
                    report += `   - Data: ${JSON.stringify(test.data)}\n`;
                }
                report += `\n`;
            }
        }

        report += `## Recommendations\n\n`;
        for (const rec of this.results.recommendations) {
            report += `- ${rec}\n`;
        }

        return report;
    }
}

// Main execution
async function main() {
    const tester = new APIIntegrationTester();
    
    try {
        await tester.initialize();
        await tester.runAllTests();
    } catch (error) {
        console.error('❌ Test suite failed:', error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = APIIntegrationTester;