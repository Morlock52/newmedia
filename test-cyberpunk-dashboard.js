#!/usr/bin/env node

/**
 * Cyberpunk Media Hub Test Suite
 * Tests all features and functionality
 */

const axios = require('axios');
const WebSocket = require('ws');
const colors = require('colors');

// Configuration
const API_URL = 'http://localhost:3738';
const WS_URL = 'ws://localhost:8001';
const DASHBOARD_URL = 'http://localhost:8888/cyberpunk-media-hub.html';

// Test Results
let passed = 0;
let failed = 0;
const results = [];

// Colors
colors.setTheme({
    pass: 'green',
    fail: 'red',
    info: 'cyan',
    warn: 'yellow',
    header: 'magenta'
});

// Test helper
async function test(name, fn) {
    try {
        await fn();
        console.log(`✅ ${name}`.pass);
        passed++;
        results.push({ name, status: 'PASSED' });
    } catch (error) {
        console.log(`❌ ${name}`.fail);
        console.log(`   Error: ${error.message}`.warn);
        failed++;
        results.push({ name, status: 'FAILED', error: error.message });
    }
}

// Tests
async function runTests() {
    console.log('\n' + '═'.repeat(60).header);
    console.log('NEXUS MEDIA HUB - TEST SUITE'.header);
    console.log('═'.repeat(60).header + '\n');

    // 1. Test Dashboard HTML
    await test('Dashboard HTML accessible', async () => {
        const response = await axios.get(DASHBOARD_URL);
        if (!response.data.includes('NEXUS MEDIA CONTROL')) {
            throw new Error('Dashboard HTML not loading correctly');
        }
    });

    // 2. Test API Services Endpoint
    await test('API /services endpoint', async () => {
        const response = await axios.get(`${API_URL}/api/services`);
        if (!Array.isArray(response.data)) {
            throw new Error('Services endpoint not returning array');
        }
        if (response.data.length === 0) {
            throw new Error('No services returned');
        }
    });

    // 3. Test All 30 Services Present
    await test('All 30 services configured', async () => {
        const response = await axios.get(`${API_URL}/api/services`);
        const expectedServices = [
            'Plex', 'Jellyfin', 'Emby', 'Sonarr', 'Radarr', 'Lidarr',
            'Readarr', 'Bazarr', 'Prowlarr', 'qBittorrent', 'SABnzbd',
            'Transmission', 'Overseerr', 'Jellyseerr', 'Tautulli',
            'Organizr', 'Heimdall', 'Homer', 'Portainer', 'Nginx Proxy',
            'Uptime Kuma', 'Grafana', 'Prometheus', 'Watchtower',
            'Duplicati', 'Nextcloud', 'Syncthing', 'FreshRSS',
            'Calibre-Web', 'PhotoPrism'
        ];
        
        const serviceNames = response.data.map(s => s.name);
        const missing = expectedServices.filter(s => !serviceNames.includes(s));
        
        if (missing.length > 0) {
            throw new Error(`Missing services: ${missing.join(', ')}`);
        }
    });

    // 4. Test Stats Endpoint
    await test('API /stats endpoint', async () => {
        const response = await axios.get(`${API_URL}/api/stats`);
        const requiredFields = ['services', 'cpu', 'memory', 'activeStreams', 'downloads', 'bandwidth'];
        
        for (const field of requiredFields) {
            if (response.data[field] === undefined) {
                throw new Error(`Missing field: ${field}`);
            }
        }
    });

    // 5. Test Service Status
    await test('Service status detection', async () => {
        const response = await axios.get(`${API_URL}/api/services`);
        const onlineServices = response.data.filter(s => s.status === 'online');
        
        if (onlineServices.length === 0) {
            throw new Error('No services detected as online');
        }
        
        console.log(`   Found ${onlineServices.length} online services`.info);
    });

    // 6. Test WebSocket Connection
    await test('WebSocket connection', async () => {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(WS_URL);
            
            ws.on('open', () => {
                ws.close();
                resolve();
            });
            
            ws.on('error', (error) => {
                reject(new Error(`WebSocket connection failed: ${error.message}`));
            });
            
            setTimeout(() => {
                ws.close();
                reject(new Error('WebSocket connection timeout'));
            }, 5000);
        });
    });

    // 7. Test WebSocket Data Stream
    await test('WebSocket real-time updates', async () => {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(WS_URL);
            let messageReceived = false;
            
            ws.on('message', (data) => {
                try {
                    const parsed = JSON.parse(data);
                    if (parsed.type === 'stats' && parsed.data) {
                        messageReceived = true;
                        ws.close();
                        resolve();
                    }
                } catch (e) {
                    reject(new Error('Invalid WebSocket message format'));
                }
            });
            
            ws.on('error', reject);
            
            setTimeout(() => {
                ws.close();
                if (!messageReceived) {
                    reject(new Error('No WebSocket messages received'));
                }
            }, 5000);
        });
    });

    // 8. Test AI Chat Endpoint
    await test('AI chat endpoint', async () => {
        const response = await axios.post(`${API_URL}/api/ai/chat`, {
            message: 'status'
        });
        
        if (!response.data.response) {
            throw new Error('AI chat not responding');
        }
    });

    // 9. Test Service Restart Endpoint (without actually restarting)
    await test('Service restart endpoint structure', async () => {
        try {
            // Test with non-existent service to check endpoint structure
            await axios.post(`${API_URL}/api/services/TestService/restart`);
        } catch (error) {
            if (error.response && error.response.status === 404) {
                // Expected behavior for non-existent service
                return;
            }
            throw error;
        }
    });

    // 10. Test Dashboard JavaScript Features
    await test('Dashboard JavaScript components', async () => {
        const response = await axios.get(DASHBOARD_URL);
        const requiredComponents = [
            'initMatrixRain',
            'init3DVisualization',
            'drawNeuralNetwork',
            'generateServiceCards',
            'updateStats'
        ];
        
        for (const component of requiredComponents) {
            if (!response.data.includes(component)) {
                throw new Error(`Missing component: ${component}`);
            }
        }
    });

    // 11. Test Cyberpunk Theme Elements
    await test('Cyberpunk theme implementation', async () => {
        const response = await axios.get(DASHBOARD_URL);
        const themeElements = [
            '--neon-cyan: #00FFFF',
            '--neon-magenta: #FF00FF',
            'glitch',
            'hologram',
            'matrix-bg',
            'Orbitron'
        ];
        
        for (const element of themeElements) {
            if (!response.data.includes(element)) {
                throw new Error(`Missing theme element: ${element}`);
            }
        }
    });

    // 12. Test Service Ports Configuration
    await test('Service ports correctly configured', async () => {
        const response = await axios.get(`${API_URL}/api/services`);
        const servicesWithPorts = response.data.filter(s => s.port !== null);
        
        if (servicesWithPorts.length < 25) {
            throw new Error('Most services should have ports configured');
        }
    });

    // Print Summary
    console.log('\n' + '═'.repeat(60).header);
    console.log('TEST RESULTS SUMMARY'.header);
    console.log('═'.repeat(60).header);
    console.log(`Total Tests: ${passed + failed}`.info);
    console.log(`Passed: ${passed}`.pass);
    console.log(`Failed: ${failed}`.fail);
    console.log('═'.repeat(60).header);

    // Detailed Results
    if (failed > 0) {
        console.log('\nFailed Tests:'.fail);
        results.filter(r => r.status === 'FAILED').forEach(r => {
            console.log(`  • ${r.name}`.fail);
            console.log(`    ${r.error}`.warn);
        });
    }

    // Performance Metrics
    console.log('\n' + 'PERFORMANCE METRICS'.header);
    console.log('─'.repeat(60));
    
    // Test response times
    const startTime = Date.now();
    await axios.get(`${API_URL}/api/services`);
    const apiResponseTime = Date.now() - startTime;
    
    console.log(`API Response Time: ${apiResponseTime}ms`.info);
    console.log(`Test Suite Duration: ${((Date.now() - testStartTime) / 1000).toFixed(2)}s`.info);
    
    // Final Status
    console.log('\n' + '═'.repeat(60).header);
    if (failed === 0) {
        console.log('✅ ALL TESTS PASSED! SYSTEM FULLY OPERATIONAL'.pass);
    } else {
        console.log(`⚠️ ${failed} TESTS FAILED - REVIEW NEEDED`.fail);
    }
    console.log('═'.repeat(60).header + '\n');

    process.exit(failed > 0 ? 1 : 0);
}

// Run tests
const testStartTime = Date.now();
runTests().catch(error => {
    console.error('Test suite error:', error);
    process.exit(1);
});