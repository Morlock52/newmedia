/**
 * Jellyfin Authentication Test Suite
 * Comprehensive testing of Jellyfin authentication and API endpoints
 */

const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

class JellyfinAuthTester {
    constructor() {
        this.baseUrl = 'http://localhost:8096';
        this.timeout = 10000;
        this.credentials = {
            username: 'admin',
            password: 'admin123'
        };
        this.accessToken = null;
        this.userId = null;
        this.apiKey = null;
    }

    /**
     * HTTP client with proper headers
     */
    createClient(token = null) {
        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };

        if (token) {
            headers['X-Emby-Authorization'] = `MediaBrowser Client="Dashboard", Device="TestScript", DeviceId="test-device", Version="1.0.0", Token="${token}"`;
        }

        return axios.create({
            baseURL: this.baseUrl,
            timeout: this.timeout,
            headers,
            validateStatus: (status) => status < 500 // Don't throw on 4xx errors
        });
    }

    /**
     * Test basic connectivity
     */
    async testConnectivity() {
        console.log('🔍 Testing basic connectivity...');
        
        try {
            const client = this.createClient();
            const response = await client.get('/health');
            
            if (response.status === 200) {
                console.log('✅ Jellyfin is reachable');
                return true;
            } else {
                console.log(`❌ Connectivity test failed: ${response.status}`);
                return false;
            }
        } catch (error) {
            console.log(`❌ Connectivity test failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Test system info endpoint
     */
    async testSystemInfo() {
        console.log('🔍 Testing system info...');
        
        try {
            const client = this.createClient();
            const response = await client.get('/System/Info/Public');
            
            if (response.status === 200) {
                console.log('✅ System info accessible');
                console.log(`   Server Name: ${response.data.ServerName || 'N/A'}`);
                console.log(`   Version: ${response.data.Version || 'N/A'}`);
                console.log(`   Startup Wizard Completed: ${response.data.StartupWizardCompleted !== false ? 'Yes' : 'No'}`);
                return true;
            } else {
                console.log(`❌ System info failed: ${response.status}`);
                return false;
            }
        } catch (error) {
            console.log(`❌ System info failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Test user authentication
     */
    async testAuthentication() {
        console.log('🔍 Testing user authentication...');
        
        try {
            const client = this.createClient();
            const response = await client.post('/Users/authenticatebyname', {
                Username: this.credentials.username,
                Pw: this.credentials.password
            });
            
            if (response.status === 200 && response.data.AccessToken) {
                this.accessToken = response.data.AccessToken;
                this.userId = response.data.User.Id;
                
                console.log('✅ Authentication successful');
                console.log(`   User ID: ${this.userId}`);
                console.log(`   Access Token: ${this.accessToken.substring(0, 20)}...`);
                return true;
            } else {
                console.log(`❌ Authentication failed: ${response.status}`);
                if (response.data) {
                    console.log(`   Response: ${JSON.stringify(response.data)}`);
                }
                return false;
            }
        } catch (error) {
            console.log(`❌ Authentication failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Test authenticated API calls
     */
    async testAuthenticatedAPIs() {
        console.log('🔍 Testing authenticated APIs...');
        
        if (!this.accessToken) {
            console.log('❌ No access token available');
            return false;
        }
        
        const endpoints = [
            { path: '/Users/Me', name: 'User Profile' },
            { path: '/System/Configuration', name: 'System Configuration' },
            { path: '/Library/VirtualFolders', name: 'Virtual Folders' },
            { path: '/Sessions', name: 'Active Sessions' }
        ];
        
        let successCount = 0;
        const client = this.createClient(this.accessToken);
        
        for (const endpoint of endpoints) {
            try {
                const response = await client.get(endpoint.path);
                
                if (response.status === 200) {
                    console.log(`✅ ${endpoint.name} - OK`);
                    successCount++;
                } else {
                    console.log(`❌ ${endpoint.name} - Failed (${response.status})`);
                }
            } catch (error) {
                console.log(`❌ ${endpoint.name} - Error: ${error.message}`);
            }
        }
        
        console.log(`📊 Authenticated API Tests: ${successCount}/${endpoints.length} passed`);
        return successCount === endpoints.length;
    }

    /**
     * Test API key creation
     */
    async testAPIKeyCreation() {
        console.log('🔍 Testing API key creation...');
        
        if (!this.accessToken) {
            console.log('❌ No access token available for API key creation');
            return false;
        }
        
        try {
            const client = this.createClient(this.accessToken);
            const response = await client.post('/Auth/Keys', {
                App: 'TestDashboard'
            });
            
            if (response.status === 200 && response.data.AccessToken) {
                this.apiKey = response.data.AccessToken;
                console.log('✅ API key created successfully');
                console.log(`   API Key: ${this.apiKey.substring(0, 20)}...`);
                
                // Save API key to file
                await this.saveAPIKey();
                return true;
            } else {
                console.log(`❌ API key creation failed: ${response.status}`);
                return false;
            }
        } catch (error) {
            console.log(`❌ API key creation failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Test API key usage
     */
    async testAPIKeyUsage() {
        console.log('🔍 Testing API key usage...');
        
        if (!this.apiKey) {
            console.log('❌ No API key available');
            return false;
        }
        
        try {
            const headers = {
                'X-Emby-Token': this.apiKey,
                'Content-Type': 'application/json'
            };
            
            const client = axios.create({
                baseURL: this.baseUrl,
                timeout: this.timeout,
                headers
            });
            
            const response = await client.get('/System/Info');
            
            if (response.status === 200) {
                console.log('✅ API key authentication successful');
                return true;
            } else {
                console.log(`❌ API key authentication failed: ${response.status}`);
                return false;
            }
        } catch (error) {
            console.log(`❌ API key authentication failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Save API key to file
     */
    async saveAPIKey() {
        if (this.apiKey) {
            try {
                const keyData = {
                    apiKey: this.apiKey,
                    created: new Date().toISOString(),
                    baseUrl: this.baseUrl,
                    userId: this.userId
                };
                
                await fs.writeFile('jellyfin-api-config.json', JSON.stringify(keyData, null, 2));
                console.log('💾 API configuration saved to jellyfin-api-config.json');
            } catch (error) {
                console.log(`❌ Failed to save API key: ${error.message}`);
            }
        }
    }

    /**
     * Test CORS functionality
     */
    async testCORS() {
        console.log('🔍 Testing CORS functionality...');
        
        try {
            const client = this.createClient();
            
            // Test preflight request
            const preflightResponse = await client.options('/System/Info', {
                headers: {
                    'Origin': 'http://localhost:3000',
                    'Access-Control-Request-Method': 'GET',
                    'Access-Control-Request-Headers': 'Content-Type'
                }
            });
            
            if (preflightResponse.status === 200 || preflightResponse.status === 204) {
                console.log('✅ CORS preflight successful');
                
                // Test actual CORS request
                const corsResponse = await client.get('/System/Info/Public', {
                    headers: {
                        'Origin': 'http://localhost:3000'
                    }
                });
                
                if (corsResponse.status === 200) {
                    console.log('✅ CORS request successful');
                    return true;
                } else {
                    console.log(`❌ CORS request failed: ${corsResponse.status}`);
                    return false;
                }
            } else {
                console.log(`❌ CORS preflight failed: ${preflightResponse.status}`);
                return false;
            }
        } catch (error) {
            console.log(`❌ CORS test failed: ${error.message}`);
            return false;
        }
    }

    /**
     * Generate test report
     */
    async generateReport(results) {
        const report = {
            timestamp: new Date().toISOString(),
            jellyfinUrl: this.baseUrl,
            testResults: results,
            summary: {
                total: Object.keys(results).length,
                passed: Object.values(results).filter(r => r === true).length,
                failed: Object.values(results).filter(r => r === false).length
            }
        };
        
        try {
            await fs.writeFile('jellyfin-auth-test-report.json', JSON.stringify(report, null, 2));
            console.log('📊 Test report saved to jellyfin-auth-test-report.json');
        } catch (error) {
            console.log(`❌ Failed to save test report: ${error.message}`);
        }
        
        return report;
    }

    /**
     * Run all tests
     */
    async runAllTests() {
        console.log('🧪 Starting Jellyfin Authentication Test Suite\n');
        console.log('=' .repeat(60));
        
        const testResults = {};
        
        // Run tests in sequence
        testResults.connectivity = await this.testConnectivity();
        console.log('');
        
        testResults.systemInfo = await this.testSystemInfo();
        console.log('');
        
        testResults.authentication = await this.testAuthentication();
        console.log('');
        
        if (testResults.authentication) {
            testResults.authenticatedAPIs = await this.testAuthenticatedAPIs();
            console.log('');
            
            testResults.apiKeyCreation = await this.testAPIKeyCreation();
            console.log('');
            
            if (testResults.apiKeyCreation) {
                testResults.apiKeyUsage = await this.testAPIKeyUsage();
                console.log('');
            }
        }
        
        testResults.cors = await this.testCORS();
        console.log('');
        
        // Generate report
        const report = await this.generateReport(testResults);
        
        // Display summary
        console.log('=' .repeat(60));
        console.log('📋 TEST SUMMARY');
        console.log('=' .repeat(60));
        console.log(`✅ Passed: ${report.summary.passed}`);
        console.log(`❌ Failed: ${report.summary.failed}`);
        console.log(`📊 Total:  ${report.summary.total}`);
        console.log('');
        
        if (this.accessToken) {
            console.log(`🔑 Access Token: ${this.accessToken.substring(0, 30)}...`);
        }
        
        if (this.apiKey) {
            console.log(`🗝️  API Key: ${this.apiKey.substring(0, 30)}...`);
        }
        
        console.log(`🌐 Jellyfin URL: ${this.baseUrl}`);
        console.log('');
        
        return report.summary.failed === 0;
    }
}

// Export for use in other modules
module.exports = JellyfinAuthTester;

// Run if called directly
if (require.main === module) {
    const tester = new JellyfinAuthTester();
    tester.runAllTests()
        .then(success => {
            process.exit(success ? 0 : 1);
        })
        .catch(error => {
            console.error('❌ Test suite failed:', error);
            process.exit(1);
        });
}