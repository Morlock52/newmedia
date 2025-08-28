#!/usr/bin/env node

/**
 * Focused API Integration Test - Running Services Only
 * Tests API connections for currently running services
 */

const axios = require('axios');
const fs = require('fs').promises;

class FocusedAPITester {
    constructor() {
        this.results = {
            timestamp: new Date().toISOString(),
            tests: [],
            summary: {}
        };

        // Only test currently running services
        this.services = {
            jellyfin: { url: 'http://localhost:8096', name: 'Jellyfin' },
            sonarr: { url: 'http://localhost:8989', name: 'Sonarr' },
            radarr: { url: 'http://localhost:7878', name: 'Radarr' },
            prowlarr: { url: 'http://localhost:9696', name: 'Prowlarr' },
            qbittorrent: { url: 'http://localhost:8080', name: 'qBittorrent' },
            uptime_kuma: { url: 'http://localhost:3001', name: 'Uptime Kuma' },
            portainer: { url: 'http://localhost:9000', name: 'Portainer' }
        };

        this.apiKeys = {};
    }

    async loadAPIKeys() {
        try {
            // Load API keys from config files
            const configPaths = {
                sonarr: './sonarr-config/config.xml',
                radarr: './radarr-config/config.xml', 
                prowlarr: './prowlarr-config/config.xml'
            };

            for (const [service, configPath] of Object.entries(configPaths)) {
                try {
                    const config = await fs.readFile(configPath, 'utf8');
                    const apiKeyMatch = config.match(/<ApiKey>(.*?)<\/ApiKey>/);
                    if (apiKeyMatch && apiKeyMatch[1]) {
                        this.apiKeys[service] = apiKeyMatch[1];
                        console.log(`✅ Loaded API key for ${service}`);
                    }
                } catch (error) {
                    console.log(`⚠️  Could not load API key for ${service}`);
                }
            }
        } catch (error) {
            console.log(`⚠️  Error loading API keys: ${error.message}`);
        }
    }

    async testServiceAPI(serviceId, service) {
        console.log(`\n🔍 Testing ${service.name} API Integration...`);
        
        // Basic connectivity test
        await this.testConnectivity(serviceId, service);
        
        // API-specific tests
        switch(serviceId) {
            case 'jellyfin':
                await this.testJellyfinAPI(service);
                break;
            case 'sonarr':
            case 'radarr':
            case 'prowlarr':
                await this.testARRAPI(serviceId, service);
                break;
            case 'qbittorrent':
                await this.testQBittorrentAPI(service);
                break;
            case 'uptime_kuma':
                await this.testUptimeKumaAPI(service);
                break;
            case 'portainer':
                await this.testPortainerAPI(service);
                break;
        }
    }

    async testConnectivity(serviceId, service) {
        try {
            const response = await axios.get(service.url, {
                timeout: 5000,
                validateStatus: (status) => status < 500
            });
            
            this.addResult(`${service.name} - Connectivity`, true, 
                `Service accessible (HTTP ${response.status})`);
                
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                this.addResult(`${service.name} - Connectivity`, false, 
                    'Service not accessible');
            } else {
                this.addResult(`${service.name} - Connectivity`, false, 
                    error.message);
            }
        }
    }

    async testJellyfinAPI(service) {
        try {
            // Test system info endpoint
            const response = await axios.get(`${service.url}/System/Info/Public`, {
                timeout: 5000
            });
            
            if (response.status === 200) {
                this.addResult('Jellyfin - System API', true, 
                    `Version: ${response.data.Version}, Server: ${response.data.ServerName}`);
            }

            // Test library endpoint
            try {
                const libraryResponse = await axios.get(`${service.url}/Items`, {
                    timeout: 5000
                });
                
                if (libraryResponse.status === 200) {
                    this.addResult('Jellyfin - Library API', true, 
                        'Library API accessible');
                }
            } catch (error) {
                if (error.response && error.response.status === 401) {
                    this.addResult('Jellyfin - Library API', true, 
                        'Authentication required (expected)');
                } else {
                    this.addResult('Jellyfin - Library API', false, error.message);
                }
            }
            
        } catch (error) {
            this.addResult('Jellyfin - API Test', false, error.message);
        }
    }

    async testARRAPI(serviceId, service) {
        const apiKey = this.apiKeys[serviceId];
        
        if (!apiKey) {
            this.addResult(`${service.name} - API Key`, false, 'API key not found');
            return;
        }

        try {
            // Test system status endpoint
            const statusResponse = await axios.get(`${service.url}/api/v1/system/status`, {
                headers: { 'X-Api-Key': apiKey },
                timeout: 5000
            });
            
            if (statusResponse.status === 200) {
                this.addResult(`${service.name} - System API`, true, 
                    `Version: ${statusResponse.data.version}, Start time: ${statusResponse.data.startTime}`);
            }

            // Test health endpoint
            const healthResponse = await axios.get(`${service.url}/api/v1/health`, {
                headers: { 'X-Api-Key': apiKey },
                timeout: 5000
            });
            
            if (healthResponse.status === 200) {
                const unhealthy = healthResponse.data.filter(h => h.type === 'error');
                this.addResult(`${service.name} - Health API`, unhealthy.length === 0, 
                    `Health checks: ${healthResponse.data.length}, Errors: ${unhealthy.length}`);
            }

        } catch (error) {
            if (error.response && error.response.status === 401) {
                this.addResult(`${service.name} - API Auth`, false, 'Invalid API key');
            } else {
                this.addResult(`${service.name} - API Test`, false, error.message);
            }
        }
    }

    async testQBittorrentAPI(service) {
        try {
            // Test version endpoint (usually public)
            const versionResponse = await axios.get(`${service.url}/api/v2/app/version`, {
                timeout: 5000
            });
            
            if (versionResponse.status === 200) {
                this.addResult('qBittorrent - Version API', true, 
                    `Version: ${versionResponse.data}`);
            }

            // Test protected endpoint (should require authentication)
            try {
                const torrentResponse = await axios.get(`${service.url}/api/v2/torrents/info`, {
                    timeout: 5000
                });
                
                if (torrentResponse.status === 200) {
                    this.addResult('qBittorrent - Torrent API', true, 
                        `Found ${torrentResponse.data.length} torrents`);
                } else {
                    this.addResult('qBittorrent - Torrent API', false, 
                        'Unexpected response');
                }
            } catch (error) {
                if (error.response && error.response.status === 403) {
                    this.addResult('qBittorrent - Auth Security', true, 
                        'Protected endpoints require authentication (good)');
                } else {
                    this.addResult('qBittorrent - Torrent API', false, error.message);
                }
            }
            
        } catch (error) {
            this.addResult('qBittorrent - API Test', false, error.message);
        }
    }

    async testUptimeKumaAPI(service) {
        try {
            const response = await axios.get(`${service.url}/metrics`, {
                timeout: 5000
            });
            
            if (response.status === 200) {
                this.addResult('Uptime Kuma - Metrics API', true, 
                    'Metrics endpoint accessible');
            }
            
        } catch (error) {
            if (error.response && error.response.status === 404) {
                this.addResult('Uptime Kuma - Service', true, 
                    'Service running (metrics endpoint may not be enabled)');
            } else {
                this.addResult('Uptime Kuma - API Test', false, error.message);
            }
        }
    }

    async testPortainerAPI(service) {
        try {
            const response = await axios.get(`${service.url}/api/status`, {
                timeout: 5000
            });
            
            if (response.status === 200) {
                this.addResult('Portainer - Status API', true, 
                    'Status API accessible');
            }
            
        } catch (error) {
            if (error.response && error.response.status === 200) {
                this.addResult('Portainer - Service', true, 
                    'Service accessible');
            } else {
                this.addResult('Portainer - API Test', false, error.message);
            }
        }
    }

    async testIntegrations() {
        console.log(`\n🔗 Testing Service Integrations...`);
        
        // Test Prowlarr -> Sonarr/Radarr connections
        await this.testProwlarrConnections();
        
        // Test download client connections
        await this.testDownloadClientConnections();
    }

    async testProwlarrConnections() {
        const prowlarrKey = this.apiKeys.prowlarr;
        
        if (!prowlarrKey) {
            this.addResult('Prowlarr - Integration Setup', false, 'API key required');
            return;
        }

        try {
            const appsResponse = await axios.get(`${this.services.prowlarr.url}/api/v1/applications`, {
                headers: { 'X-Api-Key': prowlarrKey },
                timeout: 5000
            });
            
            if (appsResponse.status === 200) {
                const apps = appsResponse.data;
                const sonarrApp = apps.find(app => app.name.toLowerCase().includes('sonarr'));
                const radarrApp = apps.find(app => app.name.toLowerCase().includes('radarr'));
                
                this.addResult('Prowlarr - Application Connections', true, 
                    `Found ${apps.length} connected applications`);
                
                if (sonarrApp) {
                    this.addResult('Prowlarr -> Sonarr', true, 'Connection configured');
                } else {
                    this.addResult('Prowlarr -> Sonarr', false, 'Connection not found');
                }
                
                if (radarrApp) {
                    this.addResult('Prowlarr -> Radarr', true, 'Connection configured');
                } else {
                    this.addResult('Prowlarr -> Radarr', false, 'Connection not found');
                }
            }
            
        } catch (error) {
            this.addResult('Prowlarr - Integration Test', false, error.message);
        }
    }

    async testDownloadClientConnections() {
        for (const service of ['sonarr', 'radarr']) {
            const apiKey = this.apiKeys[service];
            
            if (!apiKey) {
                this.addResult(`${service} - Download Client Setup`, false, 'API key required');
                continue;
            }

            try {
                const clientsResponse = await axios.get(`${this.services[service].url}/api/v1/downloadclient`, {
                    headers: { 'X-Api-Key': apiKey },
                    timeout: 5000
                });
                
                if (clientsResponse.status === 200) {
                    const clients = clientsResponse.data;
                    this.addResult(`${service} - Download Clients`, true, 
                        `Found ${clients.length} download clients configured`);
                        
                    // Test connection to each client
                    for (const client of clients) {
                        try {
                            const testResponse = await axios.post(
                                `${this.services[service].url}/api/v1/downloadclient/test`,
                                client,
                                {
                                    headers: { 'X-Api-Key': apiKey, 'Content-Type': 'application/json' },
                                    timeout: 5000
                                }
                            );
                            
                            if (testResponse.status === 200) {
                                this.addResult(`${service} -> ${client.name}`, true, 
                                    'Connection test successful');
                            }
                        } catch (testError) {
                            this.addResult(`${service} -> ${client.name}`, false, 
                                testError.response?.data?.message || 'Connection test failed');
                        }
                    }
                }
                
            } catch (error) {
                this.addResult(`${service} - Download Client Test`, false, error.message);
            }
        }
    }

    addResult(testName, passed, message) {
        this.results.tests.push({
            name: testName,
            passed,
            message,
            timestamp: new Date().toISOString()
        });

        const icon = passed ? '✅' : '❌';
        const status = passed ? 'PASS' : 'FAIL';
        console.log(`${icon} ${status}: ${testName} - ${message}`);
    }

    async generateReport() {
        const passed = this.results.tests.filter(t => t.passed).length;
        const failed = this.results.tests.filter(t => !t.passed).length;
        const total = this.results.tests.length;

        this.results.summary = {
            total,
            passed,
            failed,
            successRate: ((passed / total) * 100).toFixed(1)
        };

        console.log('\n' + '='.repeat(60));
        console.log('📊 FOCUSED API INTEGRATION TEST RESULTS');
        console.log('='.repeat(60));
        console.log(`Total Tests: ${total}`);
        console.log(`✅ Passed: ${passed}`);
        console.log(`❌ Failed: ${failed}`);
        console.log(`Success Rate: ${this.results.summary.successRate}%`);

        // Save report
        await fs.writeFile('./focused-api-test-report.json', JSON.stringify(this.results, null, 2));
        console.log('\n📄 Report saved to: focused-api-test-report.json');

        // Generate recommendations
        this.generateRecommendations();
    }

    generateRecommendations() {
        console.log('\n🎯 Recommendations:');
        
        const failedTests = this.results.tests.filter(t => !t.passed);
        
        if (failedTests.length === 0) {
            console.log('   ✅ All tests passed! Your API integrations are working correctly.');
            return;
        }

        const apiKeyIssues = failedTests.filter(t => t.message.includes('API key'));
        const connectionIssues = failedTests.filter(t => t.message.includes('Connection'));
        const authIssues = failedTests.filter(t => t.message.includes('Invalid API key') || t.message.includes('401'));

        if (apiKeyIssues.length > 0) {
            console.log('   🔑 Configure API keys in service web interfaces');
            console.log('       - Go to Settings -> General in each *ARR service');
            console.log('       - Generate and copy API keys');
        }

        if (authIssues.length > 0) {
            console.log('   🔐 Fix API authentication issues');
            console.log('       - Verify API keys are correct');
            console.log('       - Check service configurations');
        }

        if (connectionIssues.length > 0) {
            console.log('   🔗 Set up service integrations');
            console.log('       - Configure Prowlarr -> Sonarr/Radarr connections');
            console.log('       - Set up download client connections in *ARR services');
        }

        console.log('   📖 Access service web interfaces:');
        for (const [serviceId, service] of Object.entries(this.services)) {
            console.log(`       - ${service.name}: ${service.url}`);
        }
    }

    async run() {
        console.log('🚀 Starting Focused API Integration Tests\n');
        
        await this.loadAPIKeys();
        
        // Test individual services
        for (const [serviceId, service] of Object.entries(this.services)) {
            await this.testServiceAPI(serviceId, service);
        }
        
        // Test integrations
        await this.testIntegrations();
        
        // Generate report
        await this.generateReport();
    }
}

// Run the tests
if (require.main === module) {
    const tester = new FocusedAPITester();
    tester.run().catch(console.error);
}

module.exports = FocusedAPITester;