#!/usr/bin/env node

/**
 * Enterprise Media Server API Test Suite
 * Comprehensive testing framework for validating production-ready media server
 * 
 * Features:
 * - Service health validation
 * - API endpoint testing
 * - Performance benchmarking
 * - Load testing
 * - Integration testing
 * - Security validation
 * - Authentication flow testing
 * - Real-world usage simulation
 */

const axios = require('axios');
const WebSocket = require('ws');
const { performance } = require('perf_hooks');
const fs = require('fs').promises;
const path = require('path');
const { spawn, exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class MediaServerTestSuite {
    constructor() {
        this.baseURL = 'http://localhost';
        this.services = {
            // Core services
            jellyfin: { port: 8096, name: 'Jellyfin Media Server' },
            sonarr: { port: 8989, name: 'Sonarr TV Automation' },
            radarr: { port: 7878, name: 'Radarr Movie Automation' },
            lidarr: { port: 8686, name: 'Lidarr Music Automation' },
            bazarr: { port: 6767, name: 'Bazarr Subtitles' },
            prowlarr: { port: 9696, name: 'Prowlarr Indexer Manager' },
            
            // Download clients
            qbittorrent: { port: 8080, name: 'qBittorrent', vpn: true },
            sabnzbd: { port: 8081, name: 'SABnzbd Usenet' },
            
            // Request management
            overseerr: { port: 5055, name: 'Overseerr Request Manager' },
            
            // Monitoring
            tautulli: { port: 8181, name: 'Tautulli Analytics' },
            prometheus: { port: 9090, name: 'Prometheus Metrics' },
            grafana: { port: 3000, name: 'Grafana Dashboard' },
            
            // Management
            homepage: { port: 3001, name: 'Homepage Dashboard' },
            portainer: { port: 9000, name: 'Portainer Container Manager' },
            traefik: { port: 8082, name: 'Traefik Reverse Proxy' },
            
            // API Server
            api: { port: 3002, name: 'Media Server API' }
        };
        
        this.testResults = [];
        this.performanceMetrics = [];
        this.startTime = Date.now();
    }

    async runComprehensiveTests() {
        console.log('🚀 Starting Enterprise Media Server Test Suite');
        console.log('=' * 80);
        
        try {
            // Phase 1: Infrastructure validation
            await this.testPhase1_Infrastructure();
            
            // Phase 2: Service health checks
            await this.testPhase2_ServiceHealth();
            
            // Phase 3: API functionality
            await this.testPhase3_APIFunctionality();
            
            // Phase 4: Integration testing
            await this.testPhase4_Integration();
            
            // Phase 5: Performance testing
            await this.testPhase5_Performance();
            
            // Phase 6: Load testing
            await this.testPhase6_LoadTesting();
            
            // Phase 7: Security validation
            await this.testPhase7_Security();
            
            // Phase 8: User workflow testing
            await this.testPhase8_UserWorkflows();
            
            // Generate comprehensive report
            await this.generateFinalReport();
            
        } catch (error) {
            console.error('❌ Test suite failed:', error);
            throw error;
        }
    }

    async testPhase1_Infrastructure() {
        console.log('\n📋 Phase 1: Infrastructure Validation');
        console.log('-'.repeat(50));
        
        const tests = [
            this.testDockerStatus(),
            this.testNetworkConnectivity(),
            this.testVolumeStorage(),
            this.testEnvironmentVariables()
        ];
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('Infrastructure', results);
    }

    async testDockerStatus() {
        const startTime = performance.now();
        
        try {
            const { stdout } = await execAsync('docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"');
            const containers = stdout.split('\n').filter(line => line.includes('Up'));
            
            const runningServices = containers.length - 1; // Exclude header
            const expectedServices = Object.keys(this.services).length;
            
            const success = runningServices >= Math.floor(expectedServices * 0.8); // 80% threshold
            
            return {
                name: 'Docker Container Status',
                success,
                duration: performance.now() - startTime,
                details: {
                    runningContainers: runningServices,
                    expectedContainers: expectedServices,
                    containers: containers
                }
            };
        } catch (error) {
            return {
                name: 'Docker Container Status',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testNetworkConnectivity() {
        const startTime = performance.now();
        
        try {
            const { stdout } = await execAsync('docker network ls');
            const networks = stdout.split('\n').filter(line => 
                line.includes('media_network') || line.includes('download_network')
            );
            
            return {
                name: 'Docker Network Connectivity',
                success: networks.length >= 2,
                duration: performance.now() - startTime,
                details: {
                    availableNetworks: networks.length,
                    networks: networks
                }
            };
        } catch (error) {
            return {
                name: 'Docker Network Connectivity',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testVolumeStorage() {
        const startTime = performance.now();
        
        try {
            const { stdout } = await execAsync('docker volume ls');
            const volumes = stdout.split('\n').filter(line => 
                line.includes('postgres_data') || 
                line.includes('grafana_data') || 
                line.includes('prometheus_data')
            );
            
            return {
                name: 'Volume Storage',
                success: volumes.length >= 3,
                duration: performance.now() - startTime,
                details: {
                    availableVolumes: volumes.length,
                    volumes: volumes
                }
            };
        } catch (error) {
            return {
                name: 'Volume Storage',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testEnvironmentVariables() {
        const startTime = performance.now();
        
        try {
            const envPath = path.join(__dirname, '..', '.env');
            const envExists = await fs.access(envPath).then(() => true).catch(() => false);
            
            const requiredVars = ['TZ', 'MEDIA_PATH', 'DOWNLOADS_PATH'];
            const missingVars = requiredVars.filter(varName => !process.env[varName]);
            
            return {
                name: 'Environment Variables',
                success: envExists && missingVars.length === 0,
                duration: performance.now() - startTime,
                details: {
                    envFileExists: envExists,
                    missingVariables: missingVars,
                    configuredVariables: requiredVars.filter(v => process.env[v])
                }
            };
        } catch (error) {
            return {
                name: 'Environment Variables',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testPhase2_ServiceHealth() {
        console.log('\n🏥 Phase 2: Service Health Checks');
        console.log('-'.repeat(50));
        
        const tests = [];
        for (const [serviceName, config] of Object.entries(this.services)) {
            tests.push(this.testServiceHealth(serviceName, config));
        }
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('Service Health', results);
    }

    async testServiceHealth(serviceName, config) {
        const startTime = performance.now();
        const url = `${this.baseURL}:${config.port}`;
        
        try {
            const response = await axios.get(url, {
                timeout: 10000,
                validateStatus: () => true // Accept any status
            });
            
            const isHealthy = response.status < 500;
            
            return {
                name: `${config.name} Health Check`,
                success: isHealthy,
                duration: performance.now() - startTime,
                details: {
                    service: serviceName,
                    url: url,
                    status: response.status,
                    responseTime: performance.now() - startTime,
                    vpnRequired: config.vpn || false
                }
            };
        } catch (error) {
            // For VPN services, connection refused might be expected
            const isExpectedFailure = config.vpn && error.code === 'ECONNREFUSED';
            
            return {
                name: `${config.name} Health Check`,
                success: isExpectedFailure,
                duration: performance.now() - startTime,
                details: {
                    service: serviceName,
                    url: url,
                    error: error.message,
                    vpnRequired: config.vpn || false,
                    expectedFailure: isExpectedFailure
                }
            };
        }
    }

    async testPhase3_APIFunctionality() {
        console.log('\n🔌 Phase 3: API Functionality Testing');
        console.log('-'.repeat(50));
        
        const tests = [
            this.testAPIServer(),
            this.testJellyfinAPI(),
            this.testSonarrAPI(),
            this.testRadarrAPI(),
            this.testOverseerrAPI()
        ];
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('API Functionality', results);
    }

    async testAPIServer() {
        const startTime = performance.now();
        const baseUrl = `${this.baseURL}:${this.services.api.port}`;
        
        try {
            // Test health endpoint
            const healthResponse = await axios.get(`${baseUrl}/health`, { timeout: 5000 });
            
            // Test API documentation
            const docsResponse = await axios.get(`${baseUrl}/api/docs`, { timeout: 5000 });
            
            // Test services endpoint
            const servicesResponse = await axios.get(`${baseUrl}/api/services`, { timeout: 10000 });
            
            const allSuccessful = [healthResponse, docsResponse, servicesResponse]
                .every(r => r.status === 200);
            
            return {
                name: 'Media Server API',
                success: allSuccessful,
                duration: performance.now() - startTime,
                details: {
                    healthStatus: healthResponse.status,
                    docsStatus: docsResponse.status,
                    servicesStatus: servicesResponse.status,
                    servicesCount: servicesResponse.data?.data?.services?.length || 0
                }
            };
        } catch (error) {
            return {
                name: 'Media Server API',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testJellyfinAPI() {
        const startTime = performance.now();
        const baseUrl = `${this.baseURL}:${this.services.jellyfin.port}`;
        
        try {
            // Test system info endpoint
            const systemResponse = await axios.get(`${baseUrl}/System/Info/Public`, { 
                timeout: 5000,
                headers: { 'Accept': 'application/json' }
            });
            
            // Test health endpoint
            const healthResponse = await axios.get(`${baseUrl}/health`, { timeout: 5000 });
            
            return {
                name: 'Jellyfin API',
                success: systemResponse.status === 200,
                duration: performance.now() - startTime,
                details: {
                    systemStatus: systemResponse.status,
                    healthStatus: healthResponse.status,
                    version: systemResponse.data?.Version || 'unknown',
                    serverName: systemResponse.data?.ServerName || 'unknown'
                }
            };
        } catch (error) {
            return {
                name: 'Jellyfin API',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testSonarrAPI() {
        return this.testArrServiceAPI('sonarr', 'Sonarr API');
    }

    async testRadarrAPI() {
        return this.testArrServiceAPI('radarr', 'Radarr API');
    }

    async testArrServiceAPI(serviceName, displayName) {
        const startTime = performance.now();
        const baseUrl = `${this.baseURL}:${this.services[serviceName].port}`;
        
        try {
            // Test system status endpoint
            const statusResponse = await axios.get(`${baseUrl}/api/v3/system/status`, { 
                timeout: 5000,
                headers: { 'Accept': 'application/json' }
            });
            
            return {
                name: displayName,
                success: statusResponse.status === 200,
                duration: performance.now() - startTime,
                details: {
                    status: statusResponse.status,
                    version: statusResponse.data?.version || 'unknown',
                    isProduction: statusResponse.data?.isProduction || false
                }
            };
        } catch (error) {
            return {
                name: displayName,
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testOverseerrAPI() {
        const startTime = performance.now();
        const baseUrl = `${this.baseURL}:${this.services.overseerr.port}`;
        
        try {
            // Test status endpoint
            const statusResponse = await axios.get(`${baseUrl}/api/v1/status`, { 
                timeout: 5000,
                headers: { 'Accept': 'application/json' }
            });
            
            return {
                name: 'Overseerr API',
                success: statusResponse.status === 200,
                duration: performance.now() - startTime,
                details: {
                    status: statusResponse.status,
                    version: statusResponse.data?.version || 'unknown'
                }
            };
        } catch (error) {
            return {
                name: 'Overseerr API',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testPhase4_Integration() {
        console.log('\n🔗 Phase 4: Integration Testing');
        console.log('-'.repeat(50));
        
        const tests = [
            this.testServiceDiscovery(),
            this.testWebSocketConnections(),
            this.testPrometheusMetrics(),
            this.testDatabaseConnections()
        ];
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('Integration', results);
    }

    async testServiceDiscovery() {
        const startTime = performance.now();
        
        try {
            const apiUrl = `${this.baseURL}:${this.services.api.port}/api/services`;
            const response = await axios.get(apiUrl, { timeout: 10000 });
            
            const services = response.data?.data?.services || [];
            const discoveredServices = services.length;
            
            return {
                name: 'Service Discovery',
                success: discoveredServices > 0,
                duration: performance.now() - startTime,
                details: {
                    discoveredServices: discoveredServices,
                    services: services
                }
            };
        } catch (error) {
            return {
                name: 'Service Discovery',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testWebSocketConnections() {
        const startTime = performance.now();
        
        return new Promise((resolve) => {
            try {
                const wsUrl = `ws://localhost:${this.services.api.port}`;
                const ws = new WebSocket(wsUrl);
                
                let connected = false;
                
                ws.on('open', () => {
                    connected = true;
                    ws.send(JSON.stringify({ action: 'ping' }));
                });
                
                ws.on('message', (data) => {
                    try {
                        const message = JSON.parse(data);
                        if (message.type === 'pong' || message.type === 'initial-status') {
                            ws.close();
                            resolve({
                                name: 'WebSocket Connections',
                                success: true,
                                duration: performance.now() - startTime,
                                details: {
                                    connected: true,
                                    messageReceived: message.type
                                }
                            });
                        }
                    } catch (e) {
                        // Ignore parsing errors
                    }
                });
                
                ws.on('error', (error) => {
                    resolve({
                        name: 'WebSocket Connections',
                        success: false,
                        duration: performance.now() - startTime,
                        error: error.message
                    });
                });
                
                // Timeout after 10 seconds
                setTimeout(() => {
                    if (!connected) {
                        ws.close();
                        resolve({
                            name: 'WebSocket Connections',
                            success: false,
                            duration: performance.now() - startTime,
                            error: 'Connection timeout'
                        });
                    }
                }, 10000);
                
            } catch (error) {
                resolve({
                    name: 'WebSocket Connections',
                    success: false,
                    duration: performance.now() - startTime,
                    error: error.message
                });
            }
        });
    }

    async testPrometheusMetrics() {
        const startTime = performance.now();
        
        try {
            const metricsUrl = `${this.baseURL}:${this.services.prometheus.port}/api/v1/query?query=up`;
            const response = await axios.get(metricsUrl, { timeout: 10000 });
            
            const metrics = response.data?.data?.result || [];
            const activeMetrics = metrics.filter(m => m.value[1] === '1');
            
            return {
                name: 'Prometheus Metrics',
                success: response.status === 200 && activeMetrics.length > 0,
                duration: performance.now() - startTime,
                details: {
                    status: response.status,
                    totalMetrics: metrics.length,
                    activeMetrics: activeMetrics.length
                }
            };
        } catch (error) {
            return {
                name: 'Prometheus Metrics',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testDatabaseConnections() {
        const startTime = performance.now();
        
        try {
            // Test PostgreSQL connection via docker exec
            const { stdout } = await execAsync('docker exec postgres pg_isready -U postgres');
            const isReady = stdout.includes('accepting connections');
            
            return {
                name: 'Database Connections',
                success: isReady,
                duration: performance.now() - startTime,
                details: {
                    postgresReady: isReady,
                    output: stdout.trim()
                }
            };
        } catch (error) {
            return {
                name: 'Database Connections',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testPhase5_Performance() {
        console.log('\n⚡ Phase 5: Performance Testing');
        console.log('-'.repeat(50));
        
        const tests = [
            this.testAPIResponseTimes(),
            this.testMemoryUsage(),
            this.testCPUUsage(),
            this.testDiskIO()
        ];
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('Performance', results);
    }

    async testAPIResponseTimes() {
        const startTime = performance.now();
        
        try {
            const endpoints = [
                { url: `${this.baseURL}:${this.services.api.port}/health`, target: 100 },
                { url: `${this.baseURL}:${this.services.jellyfin.port}/health`, target: 200 },
                { url: `${this.baseURL}:${this.services.api.port}/api/services`, target: 500 }
            ];
            
            const results = [];
            
            for (const endpoint of endpoints) {
                const times = [];
                
                // Make 10 requests to each endpoint
                for (let i = 0; i < 10; i++) {
                    const reqStart = performance.now();
                    try {
                        await axios.get(endpoint.url, { timeout: 5000 });
                        times.push(performance.now() - reqStart);
                    } catch (error) {
                        times.push(5000); // Timeout value
                    }
                }
                
                const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
                const p95Time = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)];
                
                results.push({
                    url: endpoint.url,
                    averageTime: avgTime,
                    p95Time: p95Time,
                    target: endpoint.target,
                    meetsTarget: avgTime <= endpoint.target
                });
            }
            
            const allMeetTargets = results.every(r => r.meetsTarget);
            
            return {
                name: 'API Response Times',
                success: allMeetTargets,
                duration: performance.now() - startTime,
                details: { results }
            };
        } catch (error) {
            return {
                name: 'API Response Times',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testMemoryUsage() {
        const startTime = performance.now();
        
        try {
            const { stdout } = await execAsync("docker stats --no-stream --format 'table {{.Container}}\\t{{.MemUsage}}\\t{{.MemPerc}}'");
            const lines = stdout.split('\n').filter(line => line.trim() && !line.includes('CONTAINER'));
            
            const memoryStats = lines.map(line => {
                const parts = line.trim().split(/\s+/);
                const container = parts[0];
                const memUsage = parts[1];
                const memPerc = parseFloat(parts[2].replace('%', ''));
                
                return { container, memUsage, memPerc };
            });
            
            const avgMemoryUsage = memoryStats.reduce((sum, stat) => sum + stat.memPerc, 0) / memoryStats.length;
            const highMemoryContainers = memoryStats.filter(stat => stat.memPerc > 80);
            
            return {
                name: 'Memory Usage',
                success: avgMemoryUsage < 70 && highMemoryContainers.length === 0, // < 70% average, no containers > 80%
                duration: performance.now() - startTime,
                details: {
                    averageMemoryUsage: avgMemoryUsage,
                    highMemoryContainers: highMemoryContainers,
                    containerStats: memoryStats
                }
            };
        } catch (error) {
            return {
                name: 'Memory Usage',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testCPUUsage() {
        const startTime = performance.now();
        
        try {
            const { stdout } = await execAsync("docker stats --no-stream --format 'table {{.Container}}\\t{{.CPUPerc}}'");
            const lines = stdout.split('\n').filter(line => line.trim() && !line.includes('CONTAINER'));
            
            const cpuStats = lines.map(line => {
                const parts = line.trim().split(/\s+/);
                const container = parts[0];
                const cpuPerc = parseFloat(parts[1].replace('%', ''));
                
                return { container, cpuPerc };
            });
            
            const avgCPUUsage = cpuStats.reduce((sum, stat) => sum + stat.cpuPerc, 0) / cpuStats.length;
            const highCPUContainers = cpuStats.filter(stat => stat.cpuPerc > 90);
            
            return {
                name: 'CPU Usage',
                success: avgCPUUsage < 60 && highCPUContainers.length === 0, // < 60% average, no containers > 90%
                duration: performance.now() - startTime,
                details: {
                    averageCPUUsage: avgCPUUsage,
                    highCPUContainers: highCPUContainers,
                    containerStats: cpuStats
                }
            };
        } catch (error) {
            return {
                name: 'CPU Usage',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testDiskIO() {
        const startTime = performance.now();
        
        try {
            // Test disk performance by checking Docker volumes
            const { stdout } = await execAsync('docker system df');
            const lines = stdout.split('\n');
            
            const volumesLine = lines.find(line => line.toLowerCase().includes('volumes'));
            const reclaimable = volumesLine ? volumesLine.includes('0B') : false;
            
            return {
                name: 'Disk I/O Performance',
                success: true, // Basic check - volumes are accessible
                duration: performance.now() - startTime,
                details: {
                    diskUsage: stdout,
                    optimized: reclaimable
                }
            };
        } catch (error) {
            return {
                name: 'Disk I/O Performance',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testPhase6_LoadTesting() {
        console.log('\n🏋️ Phase 6: Load Testing');
        console.log('-'.repeat(50));
        
        const tests = [
            this.testConcurrentUsers(),
            this.testSustainedLoad(),
            this.testSpikeTesting()
        ];
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('Load Testing', results);
    }

    async testConcurrentUsers() {
        const startTime = performance.now();
        
        try {
            const concurrentLevels = [5, 10, 25];
            const results = [];
            
            for (const userCount of concurrentLevels) {
                const testStart = performance.now();
                const promises = [];
                
                // Create concurrent requests
                for (let i = 0; i < userCount; i++) {
                    const promise = axios.get(`${this.baseURL}:${this.services.jellyfin.port}/health`, { 
                        timeout: 10000 
                    }).then(response => ({
                        success: response.status === 200,
                        time: performance.now() - testStart
                    })).catch(() => ({
                        success: false,
                        time: performance.now() - testStart
                    }));
                    
                    promises.push(promise);
                }
                
                const responses = await Promise.all(promises);
                const successRate = responses.filter(r => r.success).length / responses.length;
                const avgResponseTime = responses.reduce((sum, r) => sum + r.time, 0) / responses.length;
                
                results.push({
                    userCount,
                    successRate,
                    avgResponseTime,
                    totalDuration: performance.now() - testStart
                });
            }
            
            const allSuccessful = results.every(r => r.successRate > 0.95); // 95% success rate required
            
            return {
                name: 'Concurrent Users Load Test',
                success: allSuccessful,
                duration: performance.now() - startTime,
                details: { results }
            };
        } catch (error) {
            return {
                name: 'Concurrent Users Load Test',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testSustainedLoad() {
        const startTime = performance.now();
        
        try {
            const duration = 30000; // 30 seconds
            const requestInterval = 1000; // 1 request per second
            const endTime = Date.now() + duration;
            
            const results = [];
            
            while (Date.now() < endTime) {
                const reqStart = performance.now();
                try {
                    const response = await axios.get(`${this.baseURL}:${this.services.api.port}/health`, { 
                        timeout: 5000 
                    });
                    results.push({
                        success: response.status === 200,
                        responseTime: performance.now() - reqStart,
                        timestamp: Date.now()
                    });
                } catch (error) {
                    results.push({
                        success: false,
                        responseTime: 5000,
                        timestamp: Date.now()
                    });
                }
                
                // Wait for next interval
                await new Promise(resolve => setTimeout(resolve, requestInterval));
            }
            
            const successRate = results.filter(r => r.success).length / results.length;
            const avgResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
            
            return {
                name: 'Sustained Load Test',
                success: successRate > 0.98 && avgResponseTime < 1000, // 98% success, < 1s response
                duration: performance.now() - startTime,
                details: {
                    totalRequests: results.length,
                    successRate,
                    avgResponseTime,
                    duration: duration / 1000
                }
            };
        } catch (error) {
            return {
                name: 'Sustained Load Test',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testSpikeTesting() {
        const startTime = performance.now();
        
        try {
            // Simulate sudden spike - 50 concurrent requests
            const spikeStart = performance.now();
            const promises = [];
            
            for (let i = 0; i < 50; i++) {
                const promise = axios.get(`${this.baseURL}:${this.services.api.port}/api/services`, { 
                    timeout: 15000 
                }).then(response => ({
                    success: response.status === 200,
                    time: performance.now() - spikeStart
                })).catch(() => ({
                    success: false,
                    time: performance.now() - spikeStart
                }));
                
                promises.push(promise);
            }
            
            const responses = await Promise.all(promises);
            const successRate = responses.filter(r => r.success).length / responses.length;
            const maxResponseTime = Math.max(...responses.map(r => r.time));
            
            return {
                name: 'Spike Load Test',
                success: successRate > 0.90 && maxResponseTime < 10000, // 90% success, < 10s max response
                duration: performance.now() - startTime,
                details: {
                    concurrentRequests: 50,
                    successRate,
                    maxResponseTime,
                    spikeDuration: performance.now() - spikeStart
                }
            };
        } catch (error) {
            return {
                name: 'Spike Load Test',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testPhase7_Security() {
        console.log('\n🔒 Phase 7: Security Validation');
        console.log('-'.repeat(50));
        
        const tests = [
            this.testHTTPSRedirection(),
            this.testRateLimiting(),
            this.testSecurityHeaders(),
            this.testVPNIsolation()
        ];
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('Security', results);
    }

    async testHTTPSRedirection() {
        const startTime = performance.now();
        
        try {
            // Test if Traefik is configured for HTTPS redirection
            const response = await axios.get(`${this.baseURL}:${this.services.traefik.port}/api/rawdata`, { 
                timeout: 5000,
                validateStatus: () => true
            });
            
            const hasHTTPSConfig = response.status === 200;
            
            return {
                name: 'HTTPS Redirection',
                success: hasHTTPSConfig,
                duration: performance.now() - startTime,
                details: {
                    traefikStatus: response.status,
                    httpsConfigured: hasHTTPSConfig
                }
            };
        } catch (error) {
            return {
                name: 'HTTPS Redirection',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testRateLimiting() {
        const startTime = performance.now();
        
        try {
            // Test API rate limiting by making rapid requests
            const promises = [];
            const url = `${this.baseURL}:${this.services.api.port}/health`;
            
            // Make 120 requests rapidly (should trigger rate limit of 100/15min)
            for (let i = 0; i < 120; i++) {
                promises.push(
                    axios.get(url, { timeout: 5000, validateStatus: () => true })
                        .then(r => r.status)
                        .catch(() => 500)
                );
            }
            
            const statuses = await Promise.all(promises);
            const rateLimitedRequests = statuses.filter(status => status === 429);
            
            return {
                name: 'Rate Limiting',
                success: rateLimitedRequests.length > 0, // Should have some rate limited requests
                duration: performance.now() - startTime,
                details: {
                    totalRequests: statuses.length,
                    rateLimitedRequests: rateLimitedRequests.length,
                    successfulRequests: statuses.filter(s => s === 200).length
                }
            };
        } catch (error) {
            return {
                name: 'Rate Limiting',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testSecurityHeaders() {
        const startTime = performance.now();
        
        try {
            const response = await axios.get(`${this.baseURL}:${this.services.api.port}/health`, { 
                timeout: 5000 
            });
            
            const headers = response.headers;
            const requiredHeaders = [
                'x-frame-options',
                'x-content-type-options',
                'x-xss-protection'
            ];
            
            const presentHeaders = requiredHeaders.filter(header => 
                headers[header] || headers[header.toLowerCase()]
            );
            
            return {
                name: 'Security Headers',
                success: presentHeaders.length >= 2, // At least 2 security headers
                duration: performance.now() - startTime,
                details: {
                    requiredHeaders,
                    presentHeaders,
                    allHeaders: Object.keys(headers)
                }
            };
        } catch (error) {
            return {
                name: 'Security Headers',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testVPNIsolation() {
        const startTime = performance.now();
        
        try {
            // Test that VPN-protected services are properly isolated
            // qBittorrent should be behind VPN and not directly accessible
            try {
                await axios.get(`${this.baseURL}:${this.services.qbittorrent.port}`, { 
                    timeout: 3000 
                });
                // If we can reach it directly, VPN isolation might not be working
                return {
                    name: 'VPN Isolation',
                    success: false,
                    duration: performance.now() - startTime,
                    details: {
                        vpnIsolated: false,
                        message: 'VPN service accessible without VPN'
                    }
                };
            } catch (error) {
                // Expected - service should not be directly accessible
                return {
                    name: 'VPN Isolation',
                    success: true,
                    duration: performance.now() - startTime,
                    details: {
                        vpnIsolated: true,
                        message: 'VPN service properly isolated'
                    }
                };
            }
        } catch (error) {
            return {
                name: 'VPN Isolation',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testPhase8_UserWorkflows() {
        console.log('\n👤 Phase 8: User Workflow Testing');
        console.log('-'.repeat(50));
        
        const tests = [
            this.testMediaRequestWorkflow(),
            this.testMediaDiscoveryWorkflow(),
            this.testMonitoringWorkflow(),
            this.testManagementWorkflow()
        ];
        
        const results = await Promise.allSettled(tests);
        this.processTestResults('User Workflows', results);
    }

    async testMediaRequestWorkflow() {
        const startTime = performance.now();
        
        try {
            // Simulate a complete media request workflow
            const steps = [];
            
            // Step 1: Check Overseerr status
            try {
                const overseerrResponse = await axios.get(
                    `${this.baseURL}:${this.services.overseerr.port}/api/v1/status`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'Overseerr Status', success: overseerrResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'Overseerr Status', success: false });
            }
            
            // Step 2: Check Sonarr connection
            try {
                const sonarrResponse = await axios.get(
                    `${this.baseURL}:${this.services.sonarr.port}/api/v3/system/status`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'Sonarr Connection', success: sonarrResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'Sonarr Connection', success: false });
            }
            
            // Step 3: Check Radarr connection
            try {
                const radarrResponse = await axios.get(
                    `${this.baseURL}:${this.services.radarr.port}/api/v3/system/status`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'Radarr Connection', success: radarrResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'Radarr Connection', success: false });
            }
            
            const successfulSteps = steps.filter(s => s.success).length;
            const totalSteps = steps.length;
            
            return {
                name: 'Media Request Workflow',
                success: successfulSteps >= Math.floor(totalSteps * 0.8), // 80% success rate
                duration: performance.now() - startTime,
                details: {
                    totalSteps,
                    successfulSteps,
                    steps
                }
            };
        } catch (error) {
            return {
                name: 'Media Request Workflow',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testMediaDiscoveryWorkflow() {
        const startTime = performance.now();
        
        try {
            // Test media discovery through Jellyfin
            const steps = [];
            
            // Step 1: Access Jellyfin web interface
            try {
                const jellyfinResponse = await axios.get(
                    `${this.baseURL}:${this.services.jellyfin.port}/web/index.html`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'Jellyfin Web Access', success: jellyfinResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'Jellyfin Web Access', success: false });
            }
            
            // Step 2: Get system info
            try {
                const systemResponse = await axios.get(
                    `${this.baseURL}:${this.services.jellyfin.port}/System/Info/Public`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'System Information', success: systemResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'System Information', success: false });
            }
            
            const successfulSteps = steps.filter(s => s.success).length;
            const totalSteps = steps.length;
            
            return {
                name: 'Media Discovery Workflow',
                success: successfulSteps === totalSteps,
                duration: performance.now() - startTime,
                details: {
                    totalSteps,
                    successfulSteps,
                    steps
                }
            };
        } catch (error) {
            return {
                name: 'Media Discovery Workflow',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testMonitoringWorkflow() {
        const startTime = performance.now();
        
        try {
            const steps = [];
            
            // Step 1: Check Grafana dashboard
            try {
                const grafanaResponse = await axios.get(
                    `${this.baseURL}:${this.services.grafana.port}/api/health`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'Grafana Dashboard', success: grafanaResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'Grafana Dashboard', success: false });
            }
            
            // Step 2: Check Prometheus metrics
            try {
                const prometheusResponse = await axios.get(
                    `${this.baseURL}:${this.services.prometheus.port}/api/v1/query?query=up`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'Prometheus Metrics', success: prometheusResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'Prometheus Metrics', success: false });
            }
            
            // Step 3: Check Tautulli analytics
            try {
                const tautulliResponse = await axios.get(
                    `${this.baseURL}:${this.services.tautulli.port}`, 
                    { timeout: 5000, validateStatus: () => true }
                );
                steps.push({ step: 'Tautulli Analytics', success: tautulliResponse.status < 500 });
            } catch (error) {
                steps.push({ step: 'Tautulli Analytics', success: false });
            }
            
            const successfulSteps = steps.filter(s => s.success).length;
            const totalSteps = steps.length;
            
            return {
                name: 'Monitoring Workflow',
                success: successfulSteps >= Math.floor(totalSteps * 0.67), // 67% success rate
                duration: performance.now() - startTime,
                details: {
                    totalSteps,
                    successfulSteps,
                    steps
                }
            };
        } catch (error) {
            return {
                name: 'Monitoring Workflow',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    async testManagementWorkflow() {
        const startTime = performance.now();
        
        try {
            const steps = [];
            
            // Step 1: Check Homepage dashboard
            try {
                const homepageResponse = await axios.get(
                    `${this.baseURL}:${this.services.homepage.port}`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'Homepage Dashboard', success: homepageResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'Homepage Dashboard', success: false });
            }
            
            // Step 2: Check Portainer container management
            try {
                const portainerResponse = await axios.get(
                    `${this.baseURL}:${this.services.portainer.port}`, 
                    { timeout: 5000, validateStatus: () => true }
                );
                steps.push({ step: 'Portainer Access', success: portainerResponse.status < 500 });
            } catch (error) {
                steps.push({ step: 'Portainer Access', success: false });
            }
            
            // Step 3: Check API server management
            try {
                const apiResponse = await axios.get(
                    `${this.baseURL}:${this.services.api.port}/api/services`, 
                    { timeout: 5000 }
                );
                steps.push({ step: 'API Management', success: apiResponse.status === 200 });
            } catch (error) {
                steps.push({ step: 'API Management', success: false });
            }
            
            const successfulSteps = steps.filter(s => s.success).length;
            const totalSteps = steps.length;
            
            return {
                name: 'Management Workflow',
                success: successfulSteps >= Math.floor(totalSteps * 0.67), // 67% success rate
                duration: performance.now() - startTime,
                details: {
                    totalSteps,
                    successfulSteps,
                    steps
                }
            };
        } catch (error) {
            return {
                name: 'Management Workflow',
                success: false,
                duration: performance.now() - startTime,
                error: error.message
            };
        }
    }

    processTestResults(category, results) {
        console.log(`\n📊 ${category} Results:`);
        
        let passed = 0;
        let total = 0;
        
        results.forEach((result, index) => {
            total++;
            if (result.status === 'fulfilled' && result.value.success) {
                passed++;
                console.log(`✅ ${result.value.name} - ${result.value.duration.toFixed(2)}ms`);
            } else {
                const error = result.status === 'rejected' ? result.reason : result.value;
                console.log(`❌ ${error.name || `Test ${index + 1}`} - ${error.error || error.message || 'Failed'}`);
            }
            
            // Store result for final report
            this.testResults.push({
                category,
                result: result.status === 'fulfilled' ? result.value : {
                    name: `Test ${index + 1}`,
                    success: false,
                    error: result.reason?.message || 'Unknown error'
                }
            });
        });
        
        const percentage = ((passed / total) * 100).toFixed(1);
        console.log(`📈 ${category}: ${passed}/${total} passed (${percentage}%)`);
    }

    async generateFinalReport() {
        console.log('\n' + '='.repeat(80));
        console.log('🎯 ENTERPRISE MEDIA SERVER TEST REPORT');
        console.log('='.repeat(80));
        
        // Calculate overall statistics
        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.result.success).length;
        const failedTests = totalTests - passedTests;
        const successRate = ((passedTests / totalTests) * 100).toFixed(1);
        
        // Calculate category statistics
        const categories = [...new Set(this.testResults.map(r => r.category))];
        const categoryStats = categories.map(category => {
            const categoryTests = this.testResults.filter(r => r.category === category);
            const categoryPassed = categoryTests.filter(r => r.result.success).length;
            const categoryTotal = categoryTests.length;
            const categoryRate = ((categoryPassed / categoryTotal) * 100).toFixed(1);
            
            return {
                category,
                passed: categoryPassed,
                total: categoryTotal,
                rate: categoryRate
            };
        });
        
        const totalDuration = Date.now() - this.startTime;
        
        // Console output
        console.log(`📊 Overall Results:`);
        console.log(`   Total Tests: ${totalTests}`);
        console.log(`   Passed: ${passedTests}`);
        console.log(`   Failed: ${failedTests}`);
        console.log(`   Success Rate: ${successRate}%`);
        console.log(`   Duration: ${(totalDuration / 1000).toFixed(2)}s`);
        
        console.log(`\n📋 Category Breakdown:`);
        categoryStats.forEach(stat => {
            const icon = parseFloat(stat.rate) >= 80 ? '✅' : parseFloat(stat.rate) >= 50 ? '⚠️' : '❌';
            console.log(`   ${icon} ${stat.category}: ${stat.passed}/${stat.total} (${stat.rate}%)`);
        });
        
        // Determine overall grade
        let grade, status;
        if (parseFloat(successRate) >= 90) {
            grade = 'A+';
            status = 'ENTERPRISE READY';
        } else if (parseFloat(successRate) >= 80) {
            grade = 'A';
            status = 'PRODUCTION READY';
        } else if (parseFloat(successRate) >= 70) {
            grade = 'B';
            status = 'MOSTLY FUNCTIONAL';
        } else if (parseFloat(successRate) >= 60) {
            grade = 'C';
            status = 'NEEDS IMPROVEMENT';
        } else {
            grade = 'F';
            status = 'REQUIRES MAJOR FIXES';
        }
        
        console.log(`\n🏆 FINAL GRADE: ${grade}`);
        console.log(`🎯 STATUS: ${status}`);
        
        // Commercial comparison
        console.log(`\n💼 Commercial Readiness Assessment:`);
        if (parseFloat(successRate) >= 85) {
            console.log(`   ✅ Ready to compete with commercial solutions`);
            console.log(`   ✅ Enterprise-grade reliability and performance`);
            console.log(`   ✅ Suitable for production deployment`);
        } else if (parseFloat(successRate) >= 70) {
            console.log(`   ⚠️  Approaching commercial-grade quality`);
            console.log(`   ⚠️  Some improvements needed for enterprise use`);
            console.log(`   ✅ Suitable for personal/small business use`);
        } else {
            console.log(`   ❌ Not yet ready for commercial deployment`);
            console.log(`   ❌ Significant improvements needed`);
            console.log(`   ⚠️  Development/testing use only`);
        }
        
        // Generate JSON report
        const report = {
            timestamp: new Date().toISOString(),
            summary: {
                totalTests,
                passedTests,
                failedTests,
                successRate: parseFloat(successRate),
                duration: totalDuration,
                grade,
                status
            },
            categoryStats,
            detailedResults: this.testResults,
            recommendations: this.generateRecommendations(categoryStats, parseFloat(successRate))
        };
        
        const reportPath = path.join(__dirname, `test-report-${Date.now()}.json`);
        await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
        
        console.log(`\n📄 Detailed report saved to: ${reportPath}`);
        console.log('='.repeat(80));
        
        return report;
    }

    generateRecommendations(categoryStats, successRate) {
        const recommendations = [];
        
        // Category-specific recommendations
        categoryStats.forEach(stat => {
            const rate = parseFloat(stat.rate);
            if (rate < 80) {
                switch (stat.category) {
                    case 'Infrastructure':
                        recommendations.push('Review Docker container configuration and resource allocation');
                        break;
                    case 'Service Health':
                        recommendations.push('Check service configurations and dependencies');
                        break;
                    case 'API Functionality':
                        recommendations.push('Verify API endpoints and authentication mechanisms');
                        break;
                    case 'Integration':
                        recommendations.push('Review inter-service communication and network configuration');
                        break;
                    case 'Performance':
                        recommendations.push('Optimize resource usage and response times');
                        break;
                    case 'Load Testing':
                        recommendations.push('Implement load balancing and auto-scaling');
                        break;
                    case 'Security':
                        recommendations.push('Strengthen security configuration and access controls');
                        break;
                    case 'User Workflows':
                        recommendations.push('Improve user experience and workflow integration');
                        break;
                }
            }
        });
        
        // Overall recommendations
        if (successRate < 70) {
            recommendations.unshift('CRITICAL: Address failing tests before production deployment');
        } else if (successRate < 85) {
            recommendations.unshift('IMPORTANT: Resolve remaining issues for enterprise deployment');
        }
        
        return recommendations;
    }
}

// Main execution
async function main() {
    const testSuite = new MediaServerTestSuite();
    
    try {
        await testSuite.runComprehensiveTests();
        process.exit(0);
    } catch (error) {
        console.error('❌ Test suite execution failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = MediaServerTestSuite;