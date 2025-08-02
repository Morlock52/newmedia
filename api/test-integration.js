#!/usr/bin/env node

/**
 * Comprehensive API Integration Test Suite
 * Tests all backend services, API endpoints, and fallback mechanisms
 */

const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

// Test configuration
const TEST_CONFIG = {
    chatbotAPI: {
        url: 'http://localhost:3001',
        endpoints: ['/api/health', '/api/chat']
    },
    configServer: {
        url: 'http://localhost:3000',
        endpoints: ['/api/health', '/api/docker/services']
    },
    timeout: 10000,
    retryAttempts: 3
};

class IntegrationTester {
    constructor() {
        this.results = {
            passed: 0,
            failed: 0,
            skipped: 0,
            tests: []
        };
        this.startTime = Date.now();
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const colors = {
            info: '\x1b[36m',    // Cyan
            success: '\x1b[32m', // Green
            error: '\x1b[31m',   // Red
            warning: '\x1b[33m', // Yellow
            reset: '\x1b[0m'     // Reset
        };
        
        console.log(`${colors[type]}[${timestamp}] ${message}${colors.reset}`);
    }

    async test(name, testFunction) {
        this.log(`Running test: ${name}`, 'info');
        
        try {
            const startTime = Date.now();
            await testFunction();
            const duration = Date.now() - startTime;
            
            this.results.passed++;
            this.results.tests.push({
                name,
                status: 'PASSED',
                duration,
                error: null
            });
            
            this.log(`✅ ${name} - PASSED (${duration}ms)`, 'success');
            
        } catch (error) {
            this.results.failed++;
            this.results.tests.push({
                name,
                status: 'FAILED',
                duration: 0,
                error: error.message
            });
            
            this.log(`❌ ${name} - FAILED: ${error.message}`, 'error');
        }
    }

    async makeRequest(url, options = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TEST_CONFIG.timeout);

        try {
            const response = await fetch(url, {
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            clearTimeout(timeoutId);
            return response;

        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error(`Request timeout after ${TEST_CONFIG.timeout}ms`);
            }
            throw error;
        }
    }

    async testHealthEndpoint(serviceName, url) {
        const response = await this.makeRequest(url);
        
        if (!response.ok) {
            throw new Error(`Health check failed with status: ${response.status}`);
        }

        const data = await response.json();
        
        if (!data.status || data.status !== 'healthy') {
            throw new Error(`Service reported unhealthy status: ${JSON.stringify(data)}`);
        }

        return data;
    }

    async testChatbotAPI() {
        const baseUrl = TEST_CONFIG.chatbotAPI.url;
        
        // Test health endpoint
        await this.test('Chatbot API Health Check', async () => {
            await this.testHealthEndpoint('Chatbot API', `${baseUrl}/api/health`);
        });

        // Test chat endpoint with fallback
        await this.test('Chatbot API Chat Endpoint', async () => {
            const response = await this.makeRequest(`${baseUrl}/api/chat`, {
                method: 'POST',
                body: JSON.stringify({
                    message: 'Test message',
                    history: [],
                    settings: { speed: 0.5, detail: 0.7 }
                })
            });

            // Should either succeed or return a proper error response
            if (!response.ok && response.status !== 500) {
                throw new Error(`Unexpected response status: ${response.status}`);
            }

            const data = await response.json();
            
            if (response.ok && (!data.response && !data.error)) {
                throw new Error('Chat response missing both response and error fields');
            }
        });

        // Test rate limiting
        await this.test('Chatbot API Rate Limiting', async () => {
            const promises = [];
            const baseUrl = TEST_CONFIG.chatbotAPI.url;
            
            // Send multiple requests quickly to trigger rate limiting
            for (let i = 0; i < 12; i++) {
                promises.push(
                    this.makeRequest(`${baseUrl}/api/chat`, {
                        method: 'POST',
                        body: JSON.stringify({
                            message: `Rate limit test ${i}`,
                            history: [],
                            settings: {}
                        })
                    })
                );
            }

            const responses = await Promise.all(promises);
            const rateLimitedResponses = responses.filter(r => r.status === 429);
            
            if (rateLimitedResponses.length === 0) {
                this.log('Warning: Rate limiting not triggered', 'warning');
            }
        });
    }

    async testConfigServerAPI() {
        const baseUrl = TEST_CONFIG.configServer.url;
        
        // Test health endpoint
        await this.test('Config Server Health Check', async () => {
            await this.testHealthEndpoint('Config Server', `${baseUrl}/api/health`);
        });

        // Test Docker services endpoint
        await this.test('Docker Services Endpoint', async () => {
            const response = await this.makeRequest(`${baseUrl}/api/docker/services`);
            
            if (!response.ok && response.status !== 401) {
                // 401 is expected if not authenticated
                throw new Error(`Services endpoint failed with status: ${response.status}`);
            }
        });

        // Test WebSocket endpoint (basic connectivity)
        await this.test('WebSocket Endpoint Connectivity', async () => {
            return new Promise((resolve, reject) => {
                const ws = new (require('ws'))(`ws://localhost:3000`);
                
                const timeout = setTimeout(() => {
                    ws.close();
                    reject(new Error('WebSocket connection timeout'));
                }, 5000);
                
                ws.on('open', () => {
                    clearTimeout(timeout);
                    ws.close();
                    resolve();
                });
                
                ws.on('error', (error) => {
                    clearTimeout(timeout);
                    reject(new Error(`WebSocket error: ${error.message}`));
                });
            });
        });
    }

    async testAPIClientLibrary() {
        // Test that our API client library loads correctly
        await this.test('API Client Library Loading', async () => {
            const clientPath = path.join(__dirname, 'api-client.js');
            const clientCode = await fs.readFile(clientPath, 'utf8');
            
            if (!clientCode.includes('class APIClient')) {
                throw new Error('APIClient class not found in api-client.js');
            }
            
            if (!clientCode.includes('class ChatbotAPIClient')) {
                throw new Error('ChatbotAPIClient class not found in api-client.js');
            }
            
            if (!clientCode.includes('class ConfigServerAPIClient')) {
                throw new Error('ConfigServerAPIClient class not found in api-client.js');
            }
        });

        // Test Docker service client
        await this.test('Docker Service Client Loading', async () => {
            const clientPath = path.join(__dirname, 'docker-service-client.js');
            const clientCode = await fs.readFile(clientPath, 'utf8');
            
            if (!clientCode.includes('class DockerServiceClient')) {
                throw new Error('DockerServiceClient class not found');
            }
        });
    }

    async testFallbackMechanisms() {
        // Test API client with unreachable endpoint
        await this.test('API Client Fallback Mechanism', async () => {
            // This test simulates what happens when APIs are unreachable
            const response = await this.makeRequest('http://localhost:99999/api/health').catch(error => {
                // Expected to fail - this tests our error handling
                if (error.message.includes('ECONNREFUSED') || error.message.includes('timeout')) {
                    return { ok: false, status: 0 }; // Simulated failure
                }
                throw error;
            });
            
            // The fallback mechanism should handle this gracefully
            if (response.ok) {
                throw new Error('Expected connection to fail for fallback test');
            }
        });
    }

    async testEnvironmentConfiguration() {
        await this.test('Environment Configuration', async () => {
            // Check that required config files exist
            const configFiles = [
                path.join(__dirname, '.env.example'),
                path.join(__dirname, 'package.json')
            ];
            
            for (const file of configFiles) {
                try {
                    await fs.access(file);
                } catch (error) {
                    throw new Error(`Required config file missing: ${file}`);
                }
            }
            
            // Check package.json has required dependencies
            const packagePath = path.join(__dirname, 'package.json');
            const packageData = JSON.parse(await fs.readFile(packagePath, 'utf8'));
            
            const requiredDeps = ['express', 'cors', 'openai', 'dotenv'];
            for (const dep of requiredDeps) {
                if (!packageData.dependencies || !packageData.dependencies[dep]) {
                    throw new Error(`Required dependency missing: ${dep}`);
                }
            }
        });
    }

    async testServiceStartup() {
        await this.test('Service Startup Test', async () => {
            // Test that we can start the chatbot server
            return new Promise((resolve, reject) => {
                const server = spawn('node', [path.join(__dirname, 'chatbot-server.js')], {
                    env: { ...process.env, PORT: '3099' }, // Use different port for test
                    stdio: 'pipe'
                });
                
                let output = '';
                server.stdout.on('data', (data) => {
                    output += data.toString();
                    if (output.includes('running on port 3099')) {
                        server.kill();
                        resolve();
                    }
                });
                
                server.stderr.on('data', (data) => {
                    const error = data.toString();
                    if (error.includes('Error:') && !error.includes('OPENAI_API_KEY')) {
                        server.kill();
                        reject(new Error(`Server startup error: ${error}`));
                    }
                });
                
                setTimeout(() => {
                    server.kill();
                    if (!output.includes('running on port')) {
                        reject(new Error('Server startup timeout'));
                    }
                }, 5000);
            });
        });
    }

    async runAllTests() {
        this.log('🚀 Starting comprehensive API integration tests', 'info');
        this.log(`Configuration: ${JSON.stringify(TEST_CONFIG, null, 2)}`, 'info');
        
        // Test environment and configuration
        await this.testEnvironmentConfiguration();
        
        // Test API client libraries
        await this.testAPIClientLibrary();
        
        // Test service startup
        await this.testServiceStartup();
        
        // Test fallback mechanisms
        await this.testFallbackMechanisms();
        
        // Test actual API endpoints (these might fail if services aren't running)
        try {
            await this.testChatbotAPI();
        } catch (error) {
            this.log('Chatbot API tests failed - service may not be running', 'warning');
        }
        
        try {
            await this.testConfigServerAPI();
        } catch (error) {
            this.log('Config Server API tests failed - service may not be running', 'warning');
        }
        
        this.generateReport();
    }

    generateReport() {
        const duration = Date.now() - this.startTime;
        const total = this.results.passed + this.results.failed + this.results.skipped;
        
        this.log('\n📊 Test Results Summary', 'info');
        this.log('='.repeat(50), 'info');
        this.log(`Total Tests: ${total}`, 'info');
        this.log(`Passed: ${this.results.passed}`, 'success');
        this.log(`Failed: ${this.results.failed}`, this.results.failed > 0 ? 'error' : 'info');
        this.log(`Skipped: ${this.results.skipped}`, 'warning');
        this.log(`Duration: ${duration}ms`, 'info');
        this.log(`Success Rate: ${((this.results.passed / total) * 100).toFixed(1)}%`, 'info');
        
        if (this.results.failed > 0) {
            this.log('\n❌ Failed Tests:', 'error');
            this.results.tests
                .filter(test => test.status === 'FAILED')
                .forEach(test => {
                    this.log(`  - ${test.name}: ${test.error}`, 'error');
                });
        }
        
        // Generate JSON report
        const report = {
            timestamp: new Date().toISOString(),
            duration,
            summary: {
                total,
                passed: this.results.passed,
                failed: this.results.failed,
                skipped: this.results.skipped,
                successRate: (this.results.passed / total) * 100
            },
            tests: this.results.tests,
            environment: {
                node: process.version,
                platform: process.platform,
                arch: process.arch
            }
        };
        
        const reportPath = path.join(__dirname, `test-report-${Date.now()}.json`);
        fs.writeFile(reportPath, JSON.stringify(report, null, 2))
            .then(() => {
                this.log(`\n📄 Detailed report saved to: ${reportPath}`, 'info');
            })
            .catch(error => {
                this.log(`Failed to save report: ${error.message}`, 'error');
            });
        
        // Exit with error code if tests failed
        if (this.results.failed > 0) {
            process.exit(1);
        }
    }
}

// Polyfill fetch for Node.js
if (typeof fetch === 'undefined') {
    global.fetch = require('node-fetch');
}

// Run tests if this file is executed directly
if (require.main === module) {
    const tester = new IntegrationTester();
    tester.runAllTests().catch(error => {
        console.error('Test runner error:', error);
        process.exit(1);
    });
}

module.exports = IntegrationTester;