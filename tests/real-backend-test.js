/**
 * REAL Backend Functionality Test Suite
 * Tests actual functionality, not just file existence
 */

const axios = require('axios');
const WebSocket = require('ws');
const fs = require('fs').promises;

class RealBackendTest {
    constructor() {
        this.baseURL = 'http://localhost:3333';
        this.apiURL = 'http://localhost:3333/api';
        this.failures = [];
        this.results = {
            total: 0,
            passed: 0,
            failed: 0,
            actualErrors: []
        };
    }

    recordResult(test, passed, error = null) {
        this.results.total++;
        if (passed) {
            this.results.passed++;
            console.log(`✅ ${test}`);
        } else {
            this.results.failed++;
            this.failures.push({ test, error: error?.message || error });
            this.results.actualErrors.push({
                test,
                error: error?.message || error,
                stack: error?.stack,
                response: error?.response?.data
            });
            console.log(`❌ ${test}: ${error?.message || error}`);
        }
    }

    // Test 1: Test actual Docker service management
    async testDockerServices() {
        console.log('\n🐳 Testing Docker Service Management...\n');
        
        const services = ['jellyfin', 'plex', 'sonarr', 'radarr', 'prowlarr'];
        
        for (const service of services) {
            // Test start service
            try {
                const startRes = await axios.post(`${this.apiURL}/services/${service}/start`);
                this.recordResult(`Start ${service} service`, startRes.status === 200, startRes.data);
            } catch (error) {
                this.recordResult(`Start ${service} service`, false, error);
            }

            // Test stop service
            try {
                const stopRes = await axios.post(`${this.apiURL}/services/${service}/stop`);
                this.recordResult(`Stop ${service} service`, stopRes.status === 200, stopRes.data);
            } catch (error) {
                this.recordResult(`Stop ${service} service`, false, error);
            }

            // Test service status
            try {
                const statusRes = await axios.get(`${this.apiURL}/services/${service}/status`);
                this.recordResult(`Get ${service} status`, statusRes.status === 200 && statusRes.data, statusRes.data);
            } catch (error) {
                this.recordResult(`Get ${service} status`, false, error);
            }

            // Test service logs
            try {
                const logsRes = await axios.get(`${this.apiURL}/services/${service}/logs`);
                this.recordResult(`Get ${service} logs`, logsRes.status === 200, logsRes.data);
            } catch (error) {
                this.recordResult(`Get ${service} logs`, false, error);
            }
        }
    }

    // Test 2: Test configuration management
    async testConfigManagement() {
        console.log('\n⚙️ Testing Configuration Management...\n');
        
        // Test get config
        try {
            const getConfig = await axios.get(`${this.apiURL}/config`);
            this.recordResult('Get configuration', getConfig.status === 200 && getConfig.data, getConfig.data);
        } catch (error) {
            this.recordResult('Get configuration', false, error);
        }

        // Test update config
        try {
            const updateConfig = await axios.put(`${this.apiURL}/config`, {
                jellyfin_url: 'http://localhost:8096',
                plex_url: 'http://localhost:32400'
            });
            this.recordResult('Update configuration', updateConfig.status === 200, updateConfig.data);
        } catch (error) {
            this.recordResult('Update configuration', false, error);
        }

        // Test validate config
        try {
            const validateConfig = await axios.post(`${this.apiURL}/config/validate`, {
                jellyfin_url: 'http://localhost:8096'
            });
            this.recordResult('Validate configuration', validateConfig.status === 200, validateConfig.data);
        } catch (error) {
            this.recordResult('Validate configuration', false, error);
        }
    }

    // Test 3: Test health monitoring
    async testHealthMonitoring() {
        console.log('\n🏥 Testing Health Monitoring...\n');
        
        // Test overall health
        try {
            const health = await axios.get(`${this.apiURL}/health`);
            this.recordResult('Get system health', health.status === 200 && health.data, health.data);
        } catch (error) {
            this.recordResult('Get system health', false, error);
        }

        // Test service health checks
        const services = ['jellyfin', 'plex', 'sonarr', 'radarr'];
        for (const service of services) {
            try {
                const serviceHealth = await axios.get(`${this.apiURL}/health/${service}`);
                this.recordResult(`Health check ${service}`, serviceHealth.status === 200, serviceHealth.data);
            } catch (error) {
                this.recordResult(`Health check ${service}`, false, error);
            }
        }

        // Test metrics endpoint
        try {
            const metrics = await axios.get(`${this.apiURL}/metrics`);
            this.recordResult('Get system metrics', metrics.status === 200, metrics.data);
        } catch (error) {
            this.recordResult('Get system metrics', false, error);
        }
    }

    // Test 4: Test authentication
    async testAuthentication() {
        console.log('\n🔐 Testing Authentication...\n');
        
        // Test login
        try {
            const login = await axios.post(`${this.apiURL}/auth/login`, {
                username: 'admin',
                password: 'admin123'
            });
            this.recordResult('User login', login.status === 200 && login.data.token, login.data);
            
            if (login.data.token) {
                // Test protected endpoint with token
                try {
                    const profile = await axios.get(`${this.apiURL}/auth/profile`, {
                        headers: { Authorization: `Bearer ${login.data.token}` }
                    });
                    this.recordResult('Get user profile (authenticated)', profile.status === 200, profile.data);
                } catch (error) {
                    this.recordResult('Get user profile (authenticated)', false, error);
                }
            }
        } catch (error) {
            this.recordResult('User login', false, error);
        }

        // Test logout
        try {
            const logout = await axios.post(`${this.apiURL}/auth/logout`);
            this.recordResult('User logout', logout.status === 200, logout.data);
        } catch (error) {
            this.recordResult('User logout', false, error);
        }

        // Test protected endpoint without token (should fail)
        try {
            const profile = await axios.get(`${this.apiURL}/auth/profile`);
            this.recordResult('Protected endpoint blocks unauthenticated', false, 'Should have been blocked');
        } catch (error) {
            this.recordResult('Protected endpoint blocks unauthenticated', error.response?.status === 401, error.response?.status);
        }
    }

    // Test 5: Test WebSocket functionality
    async testWebSocket() {
        console.log('\n🔌 Testing WebSocket Real-time Updates...\n');
        
        return new Promise((resolve) => {
            let ws;
            const timeout = setTimeout(() => {
                this.recordResult('WebSocket connection', false, 'Connection timeout');
                if (ws) ws.close();
                resolve();
            }, 5000);

            try {
                ws = new WebSocket('ws://localhost:3333');
                
                ws.on('open', () => {
                    clearTimeout(timeout);
                    this.recordResult('WebSocket connection', true);
                    
                    // Test sending message
                    ws.send(JSON.stringify({ type: 'subscribe', channel: 'services' }));
                    
                    setTimeout(() => {
                        ws.close();
                        resolve();
                    }, 1000);
                });

                ws.on('message', (data) => {
                    this.recordResult('WebSocket receive message', true, data.toString());
                });

                ws.on('error', (error) => {
                    clearTimeout(timeout);
                    this.recordResult('WebSocket connection', false, error);
                    resolve();
                });
            } catch (error) {
                clearTimeout(timeout);
                this.recordResult('WebSocket connection', false, error);
                resolve();
            }
        });
    }

    // Test 6: Test media library operations
    async testMediaOperations() {
        console.log('\n🎬 Testing Media Operations...\n');
        
        // Test library scan
        try {
            const scan = await axios.post(`${this.apiURL}/media/scan`);
            this.recordResult('Trigger library scan', scan.status === 200, scan.data);
        } catch (error) {
            this.recordResult('Trigger library scan', false, error);
        }

        // Test get libraries
        try {
            const libraries = await axios.get(`${this.apiURL}/media/libraries`);
            this.recordResult('Get media libraries', libraries.status === 200 && Array.isArray(libraries.data), libraries.data);
        } catch (error) {
            this.recordResult('Get media libraries', false, error);
        }

        // Test search
        try {
            const search = await axios.get(`${this.apiURL}/media/search?q=test`);
            this.recordResult('Search media', search.status === 200, search.data);
        } catch (error) {
            this.recordResult('Search media', false, error);
        }

        // Test get recent
        try {
            const recent = await axios.get(`${this.apiURL}/media/recent`);
            this.recordResult('Get recent media', recent.status === 200, recent.data);
        } catch (error) {
            this.recordResult('Get recent media', false, error);
        }
    }

    // Test 7: Test download management
    async testDownloadManagement() {
        console.log('\n📥 Testing Download Management...\n');
        
        // Test get queue
        try {
            const queue = await axios.get(`${this.apiURL}/downloads/queue`);
            this.recordResult('Get download queue', queue.status === 200, queue.data);
        } catch (error) {
            this.recordResult('Get download queue', false, error);
        }

        // Test add download
        try {
            const add = await axios.post(`${this.apiURL}/downloads/add`, {
                type: 'movie',
                title: 'Test Movie',
                url: 'magnet:?xt=test'
            });
            this.recordResult('Add download', add.status === 200, add.data);
        } catch (error) {
            this.recordResult('Add download', false, error);
        }

        // Test pause download
        try {
            const pause = await axios.post(`${this.apiURL}/downloads/pause/1`);
            this.recordResult('Pause download', pause.status === 200, pause.data);
        } catch (error) {
            this.recordResult('Pause download', false, error);
        }

        // Test resume download
        try {
            const resume = await axios.post(`${this.apiURL}/downloads/resume/1`);
            this.recordResult('Resume download', resume.status === 200, resume.data);
        } catch (error) {
            this.recordResult('Resume download', false, error);
        }

        // Test delete download
        try {
            const del = await axios.delete(`${this.apiURL}/downloads/1`);
            this.recordResult('Delete download', del.status === 200, del.data);
        } catch (error) {
            this.recordResult('Delete download', false, error);
        }
    }

    // Test 8: Test service integrations
    async testServiceIntegrations() {
        console.log('\n🔗 Testing Service Integrations...\n');
        
        const integrations = [
            { name: 'Jellyfin', endpoint: '/integrations/jellyfin/test' },
            { name: 'Plex', endpoint: '/integrations/plex/test' },
            { name: 'Sonarr', endpoint: '/integrations/sonarr/test' },
            { name: 'Radarr', endpoint: '/integrations/radarr/test' },
            { name: 'Prowlarr', endpoint: '/integrations/prowlarr/test' },
            { name: 'qBittorrent', endpoint: '/integrations/qbittorrent/test' }
        ];

        for (const integration of integrations) {
            try {
                const test = await axios.get(`${this.apiURL}${integration.endpoint}`);
                this.recordResult(`${integration.name} integration test`, test.status === 200 && test.data.connected, test.data);
            } catch (error) {
                this.recordResult(`${integration.name} integration test`, false, error);
            }
        }
    }

    // Test 9: Test user management
    async testUserManagement() {
        console.log('\n👤 Testing User Management...\n');
        
        // Test create user
        try {
            const create = await axios.post(`${this.apiURL}/users`, {
                username: 'testuser',
                email: 'test@example.com',
                password: 'Test123!'
            });
            this.recordResult('Create user', create.status === 201, create.data);
        } catch (error) {
            this.recordResult('Create user', false, error);
        }

        // Test get users
        try {
            const users = await axios.get(`${this.apiURL}/users`);
            this.recordResult('Get users list', users.status === 200 && Array.isArray(users.data), users.data);
        } catch (error) {
            this.recordResult('Get users list', false, error);
        }

        // Test update user
        try {
            const update = await axios.put(`${this.apiURL}/users/1`, {
                email: 'updated@example.com'
            });
            this.recordResult('Update user', update.status === 200, update.data);
        } catch (error) {
            this.recordResult('Update user', false, error);
        }

        // Test delete user
        try {
            const del = await axios.delete(`${this.apiURL}/users/2`);
            this.recordResult('Delete user', del.status === 200, del.data);
        } catch (error) {
            this.recordResult('Delete user', false, error);
        }
    }

    // Test 10: Test notifications
    async testNotifications() {
        console.log('\n🔔 Testing Notifications...\n');
        
        // Test send notification
        try {
            const send = await axios.post(`${this.apiURL}/notifications/send`, {
                title: 'Test Notification',
                message: 'This is a test',
                type: 'info'
            });
            this.recordResult('Send notification', send.status === 200, send.data);
        } catch (error) {
            this.recordResult('Send notification', false, error);
        }

        // Test get notifications
        try {
            const notifs = await axios.get(`${this.apiURL}/notifications`);
            this.recordResult('Get notifications', notifs.status === 200, notifs.data);
        } catch (error) {
            this.recordResult('Get notifications', false, error);
        }

        // Test mark as read
        try {
            const read = await axios.put(`${this.apiURL}/notifications/1/read`);
            this.recordResult('Mark notification as read', read.status === 200, read.data);
        } catch (error) {
            this.recordResult('Mark notification as read', false, error);
        }
    }

    // Generate detailed failure report
    generateFailureReport() {
        if (this.failures.length === 0) {
            console.log('\n✅ All tests passed!');
            return null;
        }

        console.log('\n' + '='.repeat(80));
        console.log('❌ FAILURE REPORT');
        console.log('='.repeat(80));
        console.log(`\nTotal Tests: ${this.results.total}`);
        console.log(`Passed: ${this.results.passed}`);
        console.log(`Failed: ${this.results.failed}`);
        console.log(`Success Rate: ${((this.results.passed / this.results.total) * 100).toFixed(1)}%`);
        
        console.log('\n🔴 Failed Tests:');
        this.failures.forEach((failure, index) => {
            console.log(`\n${index + 1}. ${failure.test}`);
            console.log(`   Error: ${failure.error}`);
        });

        return {
            summary: {
                total: this.results.total,
                passed: this.results.passed,
                failed: this.results.failed,
                successRate: ((this.results.passed / this.results.total) * 100).toFixed(1)
            },
            failures: this.failures,
            detailedErrors: this.results.actualErrors
        };
    }

    // Save failure report for swarm analysis
    async saveFailureReport() {
        const report = {
            timestamp: new Date(),
            projectId: '3e6fbcc1-60f6-434b-a45b-e811cc9bb891',
            testResults: this.results,
            failures: this.failures,
            detailedErrors: this.results.actualErrors,
            needsSwarmRepair: this.failures.length > 0
        };

        await fs.writeFile(
            '/Users/morlock/fun/newmedia/test-results/backend-failures.json',
            JSON.stringify(report, null, 2)
        );

        console.log('\n📁 Failure report saved to: test-results/backend-failures.json');
        return report;
    }

    // Main test runner
    async runAllTests() {
        console.log('🚀 Starting REAL Backend Functionality Tests...\n');
        console.log('Testing: http://localhost:3333\n');
        console.log('=' .repeat(80));

        await this.testDockerServices();
        await this.testConfigManagement();
        await this.testHealthMonitoring();
        await this.testAuthentication();
        await this.testWebSocket();
        await this.testMediaOperations();
        await this.testDownloadManagement();
        await this.testServiceIntegrations();
        await this.testUserManagement();
        await this.testNotifications();

        const report = this.generateFailureReport();
        await this.saveFailureReport();

        return report;
    }
}

// Run the tests
async function main() {
    const tester = new RealBackendTest();
    
    try {
        const report = await tester.runAllTests();
        
        if (report && report.failures.length > 0) {
            console.log('\n🔧 CRITICAL: Backend has major failures!');
            console.log('📝 Creating detailed Archon tasks for each failure...');
            process.exit(1);
        } else {
            console.log('\n✨ Backend tests completed successfully!');
            process.exit(0);
        }
    } catch (error) {
        console.error('Fatal test error:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main().catch(console.error);
}

module.exports = RealBackendTest;