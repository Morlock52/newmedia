#!/usr/bin/env node

/**
 * Backend Fixes Test Suite
 * Tests the implemented backend improvements
 */

const axios = require('axios');

const API_BASE = 'http://localhost:3002';

class BackendTestSuite {
    constructor() {
        this.results = [];
        this.authToken = null;
    }

    async runAllTests() {
        console.log('🧪 Starting Backend Fixes Test Suite');
        console.log('=====================================\n');

        try {
            // Test basic health endpoint
            await this.testHealthEndpoint();
            
            // Test system endpoint  
            await this.testSystemEndpoint();
            
            // Test authentication endpoints
            await this.testAuthenticationEndpoints();
            
            // Test services endpoint with auth
            await this.testServicesEndpoint();
            
            // Test error handling
            await this.testErrorHandling();
            
            // Test API documentation
            await this.testAPIDocumentation();
            
            // Summary
            this.printSummary();
            
        } catch (error) {
            console.error('❌ Test suite failed:', error.message);
        }
    }

    async testHealthEndpoint() {
        console.log('🩺 Testing Health Endpoint...');
        try {
            const response = await axios.get(`${API_BASE}/health`);
            
            if (response.status === 200 && response.data.status === 'healthy') {
                this.addResult('✅ Health endpoint working correctly');
                console.log('   Status:', response.data.status);
                console.log('   Uptime:', response.data.uptime + 's');
            } else {
                this.addResult('❌ Health endpoint returned unexpected response');
            }
        } catch (error) {
            this.addResult(`❌ Health endpoint failed: ${error.message}`);
        }
        console.log('');
    }

    async testSystemEndpoint() {
        console.log('🖥️  Testing System Endpoint...');
        try {
            const response = await axios.get(`${API_BASE}/api/system`);
            
            if (response.status === 200 && response.data.success && response.data.data.version) {
                this.addResult('✅ System endpoint working correctly');
                console.log('   Version:', response.data.data.version);
                console.log('   Platform:', response.data.data.platform);
                console.log('   Node.js:', response.data.data.nodeVersion);
            } else {
                this.addResult('❌ System endpoint returned unexpected response');
            }
        } catch (error) {
            this.addResult(`❌ System endpoint failed: ${error.message}`);
        }
        console.log('');
    }

    async testAuthenticationEndpoints() {
        console.log('🔐 Testing Authentication Endpoints...');
        
        // Test login endpoint
        try {
            const response = await axios.post(`${API_BASE}/api/auth/login`, {
                username: 'admin',
                password: 'admin123'
            });
            
            if (response.status === 200 && response.data.success && response.data.data.accessToken) {
                this.addResult('✅ Login endpoint working correctly');
                this.authToken = response.data.data.accessToken;
                console.log('   Login successful, token received');
                console.log('   User role:', response.data.data.user.role);
            } else {
                this.addResult('❌ Login endpoint returned unexpected response');
            }
        } catch (error) {
            this.addResult(`❌ Login endpoint failed: ${error.message}`);
        }
        
        // Test user info endpoint
        if (this.authToken) {
            try {
                const response = await axios.get(`${API_BASE}/api/auth/me`, {
                    headers: {
                        'Authorization': `Bearer ${this.authToken}`
                    }
                });
                
                if (response.status === 200 && response.data.success) {
                    this.addResult('✅ User info endpoint working correctly');
                    console.log('   User:', response.data.data.username);
                } else {
                    this.addResult('❌ User info endpoint returned unexpected response');
                }
            } catch (error) {
                this.addResult(`❌ User info endpoint failed: ${error.message}`);
            }
        }
        console.log('');
    }

    async testServicesEndpoint() {
        console.log('🐳 Testing Services Endpoint...');
        try {
            // Test without auth (should fail)
            try {
                await axios.get(`${API_BASE}/api/services`);
                this.addResult('❌ Services endpoint allowed access without auth');
            } catch (error) {
                if (error.response?.status === 401) {
                    this.addResult('✅ Services endpoint correctly requires authentication');
                } else {
                    this.addResult(`❌ Services endpoint failed unexpectedly: ${error.message}`);
                }
            }
            
            // Test with auth
            if (this.authToken) {
                try {
                    const response = await axios.get(`${API_BASE}/api/services`, {
                        headers: {
                            'Authorization': `Bearer ${this.authToken}`
                        }
                    });
                    
                    if (response.status === 200 && response.data.success) {
                        this.addResult('✅ Authenticated services endpoint working correctly');
                        console.log('   Services found:', response.data.data.services.length);
                    } else {
                        this.addResult('❌ Services endpoint returned unexpected response');
                    }
                } catch (error) {
                    // Services might fail if Docker isn't running, but endpoint should work
                    if (error.response?.status === 503) {
                        this.addResult('✅ Services endpoint working (Docker services unavailable)');
                        console.log('   Note: Docker services not available for testing');
                    } else {
                        this.addResult(`❌ Services endpoint failed: ${error.message}`);
                    }
                }
            }
        } catch (error) {
            this.addResult(`❌ Services endpoint test failed: ${error.message}`);
        }
        console.log('');
    }

    async testErrorHandling() {
        console.log('⚠️  Testing Error Handling...');
        try {
            // Test 404 endpoint
            const response = await axios.get(`${API_BASE}/api/nonexistent-endpoint`);
        } catch (error) {
            if (error.response?.status === 404 && error.response?.data?.success === false) {
                this.addResult('✅ 404 error handling working correctly');
                console.log('   404 error format correct');
            } else {
                this.addResult(`❌ 404 error handling failed: ${error.message}`);
            }
        }
        
        // Test validation error
        try {
            await axios.post(`${API_BASE}/api/auth/login`, {
                // Missing required fields
            });
        } catch (error) {
            if (error.response?.status === 400) {
                this.addResult('✅ Validation error handling working correctly');
                console.log('   Validation errors properly formatted');
            } else {
                this.addResult(`❌ Validation error handling failed: ${error.message}`);
            }
        }
        console.log('');
    }

    async testAPIDocumentation() {
        console.log('📚 Testing API Documentation...');
        try {
            const response = await axios.get(`${API_BASE}/api/docs`);
            
            if (response.status === 200 && response.data.title && response.data.endpoints) {
                this.addResult('✅ API documentation working correctly');
                console.log('   Title:', response.data.title);
                console.log('   Endpoints defined:', Object.keys(response.data.endpoints).length);
                
                // Check for new authentication endpoints
                if (response.data.authentication) {
                    console.log('   Authentication endpoints documented');
                }
                if (response.data.socketio) {
                    console.log('   Socket.IO endpoints documented');
                }
            } else {
                this.addResult('❌ API documentation returned unexpected response');
            }
        } catch (error) {
            this.addResult(`❌ API documentation failed: ${error.message}`);
        }
        console.log('');
    }

    addResult(result) {
        this.results.push(result);
    }

    printSummary() {
        console.log('📊 Test Results Summary');
        console.log('======================');
        
        const passed = this.results.filter(r => r.startsWith('✅')).length;
        const failed = this.results.filter(r => r.startsWith('❌')).length;
        
        console.log(`Total tests: ${this.results.length}`);
        console.log(`Passed: ${passed}`);
        console.log(`Failed: ${failed}`);
        console.log('');
        
        console.log('Detailed Results:');
        this.results.forEach(result => {
            console.log(result);
        });
        
        if (failed === 0) {
            console.log('\n🎉 All backend fixes working correctly!');
        } else {
            console.log(`\n⚠️  ${failed} issues found. Check the server is running and try again.`);
        }
    }
}

// Run tests if file is executed directly
if (require.main === module) {
    const tester = new BackendTestSuite();
    tester.runAllTests().catch(console.error);
}

module.exports = BackendTestSuite;