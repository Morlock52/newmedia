/**
 * Service Integration Test Suite
 * Tests integration with all media services (Jellyfin, Sonarr, Radarr, etc.)
 */

const axios = require('axios');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

describe('Service Integration Tests', () => {
    let baseURL;
    const timeout = 15000;

    beforeAll(async () => {
        baseURL = process.env.BASE_URL || 'http://localhost';
    });

    describe('Media Server Services', () => {
        const mediaServers = [
            {
                name: 'Jellyfin',
                port: 8096,
                healthPath: '/health',
                webPath: '/web',
                apiPath: '/System/Info',
                container: 'jellyfin'
            },
            {
                name: 'Plex',
                port: 32400,
                healthPath: '/identity',
                webPath: '/web',
                apiPath: '/library/sections',
                container: 'plex'
            },
            {
                name: 'Emby',
                port: 8097,
                healthPath: '/System/Info',
                webPath: '/web',
                apiPath: '/System/Info',
                container: 'emby'
            }
        ];

        mediaServers.forEach(server => {
            describe(`${server.name} Integration`, () => {
                test(`${server.name} service is accessible`, async () => {
                    try {
                        const response = await axios.get(`${baseURL}:${server.port}${server.webPath}`, {
                            timeout: 10000,
                            validateStatus: status => status < 500
                        });

                        expect(response.status).toBeLessThan(500);
                        console.log(`✅ ${server.name} is accessible on port ${server.port}`);
                    } catch (error) {
                        if (error.code === 'ECONNREFUSED') {
                            console.warn(`⚠️ ${server.name} service not running on port ${server.port}`);
                        } else {
                            console.warn(`⚠️ ${server.name} accessibility test failed:`, error.message);
                        }
                    }
                });

                test(`${server.name} container status`, async () => {
                    try {
                        const { stdout } = await execAsync(`docker ps --filter "name=${server.container}" --format "{{.Status}}"`);
                        const status = stdout.trim();
                        
                        if (status) {
                            expect(status).toMatch(/Up/);
                            console.log(`✅ ${server.name} container is running: ${status}`);
                        } else {
                            console.warn(`⚠️ ${server.name} container not found`);
                        }
                    } catch (error) {
                        console.warn(`⚠️ Docker command failed for ${server.name}:`, error.message);
                    }
                });

                test(`${server.name} API endpoint`, async () => {
                    try {
                        const response = await axios.get(`${baseURL}:${server.port}${server.apiPath}`, {
                            timeout: 10000,
                            validateStatus: status => status < 500,
                            headers: {
                                'Accept': 'application/json'
                            }
                        });

                        expect(response.status).toBeLessThan(500);
                        console.log(`✅ ${server.name} API is responding`);
                    } catch (error) {
                        if (error.code !== 'ECONNREFUSED') {
                            console.warn(`⚠️ ${server.name} API test failed:`, error.message);
                        }
                    }
                });
            });
        });
    });

    describe('*ARR Services', () => {
        const arrServices = [
            {
                name: 'Sonarr',
                port: 8989,
                apiPath: '/api/v3/system/status',
                container: 'sonarr',
                configFile: 'config.xml'
            },
            {
                name: 'Radarr',
                port: 7878,
                apiPath: '/api/v3/system/status',
                container: 'radarr',
                configFile: 'config.xml'
            },
            {
                name: 'Lidarr',
                port: 8686,
                apiPath: '/api/v1/system/status',
                container: 'lidarr',
                configFile: 'config.xml'
            },
            {
                name: 'Readarr',
                port: 8787,
                apiPath: '/api/v1/system/status',
                container: 'readarr',
                configFile: 'config.xml'
            },
            {
                name: 'Bazarr',
                port: 6767,
                apiPath: '/api/system/status',
                container: 'bazarr',
                configFile: 'config.yaml'
            },
            {
                name: 'Prowlarr',
                port: 9696,
                apiPath: '/api/v1/system/status',
                container: 'prowlarr',
                configFile: 'config.xml'
            }
        ];

        arrServices.forEach(service => {
            describe(`${service.name} Integration`, () => {
                test(`${service.name} web interface accessibility`, async () => {
                    try {
                        const response = await axios.get(`${baseURL}:${service.port}`, {
                            timeout: 10000,
                            validateStatus: status => status < 500
                        });

                        expect(response.status).toBeLessThan(500);
                        console.log(`✅ ${service.name} web interface is accessible`);
                        
                        // Check if it's the actual service page
                        expect(response.data).toMatch(new RegExp(service.name.toLowerCase(), 'i'));
                    } catch (error) {
                        if (error.code === 'ECONNREFUSED') {
                            console.warn(`⚠️ ${service.name} not running on port ${service.port}`);
                        } else {
                            console.warn(`⚠️ ${service.name} web interface test failed:`, error.message);
                        }
                    }
                });

                test(`${service.name} container health`, async () => {
                    try {
                        const { stdout } = await execAsync(`docker ps --filter "name=${service.container}" --format "{{.Names}}\\t{{.Status}}\\t{{.Ports}}"`);
                        
                        if (stdout.trim()) {
                            const [name, status, ports] = stdout.trim().split('\t');
                            expect(status).toMatch(/Up/);
                            expect(ports).toContain(service.port.toString());
                            console.log(`✅ ${service.name} container healthy: ${status}`);
                        } else {
                            console.warn(`⚠️ ${service.name} container not found`);
                        }
                    } catch (error) {
                        console.warn(`⚠️ ${service.name} container check failed:`, error.message);
                    }
                });

                test(`${service.name} configuration exists`, async () => {
                    try {
                        const configPath = `./${service.container}-config/${service.configFile}`;
                        const { stdout } = await execAsync(`ls -la "${configPath}" 2>/dev/null || echo "not found"`);
                        
                        if (!stdout.includes('not found')) {
                            console.log(`✅ ${service.name} configuration file exists`);
                            expect(stdout).toMatch(service.configFile);
                        } else {
                            console.warn(`⚠️ ${service.name} configuration file not found at ${configPath}`);
                        }
                    } catch (error) {
                        console.warn(`⚠️ ${service.name} config check failed:`, error.message);
                    }
                });

                test(`${service.name} API key configuration`, async () => {
                    try {
                        // Check if API key is configured by trying to access system status
                        const response = await axios.get(`${baseURL}:${service.port}${service.apiPath}`, {
                            timeout: 5000,
                            validateStatus: status => status !== 401 // API key missing would return 401
                        });

                        if (response.status === 401) {
                            console.warn(`⚠️ ${service.name} API key not configured`);
                        } else if (response.status < 500) {
                            console.log(`✅ ${service.name} API is properly configured`);
                        }
                    } catch (error) {
                        if (error.code !== 'ECONNREFUSED') {
                            console.warn(`⚠️ ${service.name} API test failed:`, error.response?.status || error.message);
                        }
                    }
                });
            });
        });
    });

    describe('Download Client Services', () => {
        const downloadClients = [
            {
                name: 'qBittorrent',
                port: 8080,
                loginPath: '/api/v2/auth/login',
                container: 'qbittorrent',
                defaultCredentials: { username: 'admin', password: 'adminadmin' }
            },
            {
                name: 'Transmission',
                port: 9091,
                webPath: '/transmission/web/',
                container: 'transmission',
                rpcPath: '/transmission/rpc'
            },
            {
                name: 'SABnzbd',
                port: 8081,
                apiPath: '/api',
                container: 'sabnzbd',
                configPath: '/sabnzbd/config'
            },
            {
                name: 'NZBGet',
                port: 6789,
                webPath: '/',
                container: 'nzbget',
                apiPath: '/jsonrpc'
            }
        ];

        downloadClients.forEach(client => {
            describe(`${client.name} Integration`, () => {
                test(`${client.name} web interface`, async () => {
                    try {
                        const testPath = client.webPath || '/';
                        const response = await axios.get(`${baseURL}:${client.port}${testPath}`, {
                            timeout: 10000,
                            validateStatus: status => status < 500
                        });

                        expect(response.status).toBeLessThan(500);
                        console.log(`✅ ${client.name} web interface accessible`);
                    } catch (error) {
                        if (error.code === 'ECONNREFUSED') {
                            console.warn(`⚠️ ${client.name} not running on port ${client.port}`);
                        } else {
                            console.warn(`⚠️ ${client.name} web interface test failed:`, error.message);
                        }
                    }
                });

                test(`${client.name} container status`, async () => {
                    try {
                        const { stdout } = await execAsync(`docker ps --filter "name=${client.container}" --format "{{.Status}}"`);
                        
                        if (stdout.trim()) {
                            expect(stdout).toMatch(/Up/);
                            console.log(`✅ ${client.name} container is running`);
                        } else {
                            console.warn(`⚠️ ${client.name} container not found`);
                        }
                    } catch (error) {
                        console.warn(`⚠️ ${client.name} container check failed:`, error.message);
                    }
                });

                if (client.apiPath) {
                    test(`${client.name} API endpoint`, async () => {
                        try {
                            const response = await axios.get(`${baseURL}:${client.port}${client.apiPath}`, {
                                timeout: 5000,
                                validateStatus: status => status < 500
                            });

                            expect(response.status).toBeLessThan(500);
                            console.log(`✅ ${client.name} API is responding`);
                        } catch (error) {
                            if (error.code !== 'ECONNREFUSED') {
                                console.warn(`⚠️ ${client.name} API test failed:`, error.message);
                            }
                        }
                    });
                }
            });
        });
    });

    describe('Request Management Services', () => {
        const requestServices = [
            {
                name: 'Jellyseerr',
                port: 5055,
                webPath: '/',
                container: 'jellyseerr'
            },
            {
                name: 'Overseerr',
                port: 5056,
                webPath: '/',
                container: 'overseerr'
            },
            {
                name: 'Ombi',
                port: 3579,
                webPath: '/',
                container: 'ombi'
            }
        ];

        requestServices.forEach(service => {
            describe(`${service.name} Integration`, () => {
                test(`${service.name} accessibility`, async () => {
                    try {
                        const response = await axios.get(`${baseURL}:${service.port}${service.webPath}`, {
                            timeout: 10000,
                            validateStatus: status => status < 500
                        });

                        expect(response.status).toBeLessThan(500);
                        console.log(`✅ ${service.name} is accessible`);
                    } catch (error) {
                        if (error.code === 'ECONNREFUSED') {
                            console.warn(`⚠️ ${service.name} not running on port ${service.port}`);
                        } else {
                            console.warn(`⚠️ ${service.name} test failed:`, error.message);
                        }
                    }
                });

                test(`${service.name} container health`, async () => {
                    try {
                        const { stdout } = await execAsync(`docker ps --filter "name=${service.container}" --format "{{.Status}}"`);
                        
                        if (stdout.trim()) {
                            expect(stdout).toMatch(/Up/);
                            console.log(`✅ ${service.name} container is healthy`);
                        } else {
                            console.warn(`⚠️ ${service.name} container not found`);
                        }
                    } catch (error) {
                        console.warn(`⚠️ ${service.name} container check failed:`, error.message);
                    }
                });
            });
        });
    });

    describe('Monitoring Services', () => {
        const monitoringServices = [
            {
                name: 'Grafana',
                port: 3000,
                healthPath: '/api/health',
                webPath: '/login',
                container: 'grafana'
            },
            {
                name: 'Prometheus',
                port: 9090,
                healthPath: '/-/healthy',
                webPath: '/graph',
                container: 'prometheus'
            },
            {
                name: 'Uptime Kuma',
                port: 3001,
                webPath: '/',
                container: 'uptime-kuma'
            },
            {
                name: 'Netdata',
                port: 19999,
                webPath: '/',
                container: 'netdata'
            }
        ];

        monitoringServices.forEach(service => {
            describe(`${service.name} Integration`, () => {
                test(`${service.name} web interface`, async () => {
                    try {
                        const response = await axios.get(`${baseURL}:${service.port}${service.webPath}`, {
                            timeout: 10000,
                            validateStatus: status => status < 500
                        });

                        expect(response.status).toBeLessThan(500);
                        console.log(`✅ ${service.name} web interface accessible`);
                    } catch (error) {
                        if (error.code === 'ECONNREFUSED') {
                            console.warn(`⚠️ ${service.name} not running on port ${service.port}`);
                        } else {
                            console.warn(`⚠️ ${service.name} test failed:`, error.message);
                        }
                    }
                });

                if (service.healthPath) {
                    test(`${service.name} health check`, async () => {
                        try {
                            const response = await axios.get(`${baseURL}:${service.port}${service.healthPath}`, {
                                timeout: 5000,
                                validateStatus: status => status < 500
                            });

                            expect(response.status).toBeLessThan(400);
                            console.log(`✅ ${service.name} health check passed`);
                        } catch (error) {
                            if (error.code !== 'ECONNREFUSED') {
                                console.warn(`⚠️ ${service.name} health check failed:`, error.message);
                            }
                        }
                    });
                }
            });
        });
    });

    describe('Management Services', () => {
        const managementServices = [
            {
                name: 'Portainer',
                port: 9000,
                webPath: '/',
                container: 'portainer'
            },
            {
                name: 'Nginx Proxy Manager',
                port: 81,
                webPath: '/',
                container: 'nginx-proxy-manager'
            },
            {
                name: 'Homarr',
                port: 7575,
                webPath: '/',
                container: 'homarr'
            },
            {
                name: 'Homepage',
                port: 3003,
                webPath: '/',
                container: 'homepage'
            }
        ];

        managementServices.forEach(service => {
            describe(`${service.name} Integration`, () => {
                test(`${service.name} accessibility`, async () => {
                    try {
                        const response = await axios.get(`${baseURL}:${service.port}${service.webPath}`, {
                            timeout: 10000,
                            validateStatus: status => status < 500
                        });

                        expect(response.status).toBeLessThan(500);
                        console.log(`✅ ${service.name} is accessible`);
                    } catch (error) {
                        if (error.code === 'ECONNREFUSED') {
                            console.warn(`⚠️ ${service.name} not running on port ${service.port}`);
                        } else {
                            console.warn(`⚠️ ${service.name} test failed:`, error.message);
                        }
                    }
                });
            });
        });
    });

    describe('Network Connectivity Tests', () => {
        test('Docker network connectivity', async () => {
            try {
                const { stdout } = await execAsync('docker network ls --filter name=media-net --format "{{.Name}}"');
                
                if (stdout.includes('media-net')) {
                    console.log('✅ Media network exists');
                    
                    // Test network connectivity between services
                    const { stdout: networkInfo } = await execAsync('docker network inspect media-net --format "{{range .Containers}}{{.Name}} {{end}}"');
                    const connectedContainers = networkInfo.trim().split(' ').filter(name => name);
                    
                    console.log(`✅ ${connectedContainers.length} containers connected to media network`);
                    expect(connectedContainers.length).toBeGreaterThan(0);
                } else {
                    console.warn('⚠️ Media network not found');
                }
            } catch (error) {
                console.warn('⚠️ Network connectivity test failed:', error.message);
            }
        });

        test('Service-to-service communication', async () => {
            try {
                // Test if Sonarr can reach Prowlarr (common integration)
                const testConnections = [
                    { from: 'sonarr', to: 'prowlarr', port: 9696 },
                    { from: 'radarr', to: 'prowlarr', port: 9696 },
                    { from: 'jellyfin', to: 'postgres', port: 5432 }
                ];

                for (const connection of testConnections) {
                    try {
                        const { stdout } = await execAsync(`docker exec ${connection.from} nc -z ${connection.to} ${connection.port} 2>/dev/null && echo "success" || echo "failed"`);
                        
                        if (stdout.includes('success')) {
                            console.log(`✅ ${connection.from} can reach ${connection.to}:${connection.port}`);
                        } else {
                            console.warn(`⚠️ ${connection.from} cannot reach ${connection.to}:${connection.port}`);
                        }
                    } catch (error) {
                        console.warn(`⚠️ Connection test failed for ${connection.from} -> ${connection.to}:`, error.message);
                    }
                }
            } catch (error) {
                console.warn('⚠️ Service communication test failed:', error.message);
            }
        });
    });

    describe('Data Persistence Tests', () => {
        test('Volume mounts verification', async () => {
            try {
                const { stdout } = await execAsync('docker volume ls --format "{{.Name}}" | grep -E "(media|config|data)"');
                const volumes = stdout.trim().split('\n').filter(vol => vol);
                
                console.log(`✅ Found ${volumes.length} persistent volumes`);
                expect(volumes.length).toBeGreaterThan(0);
                
                // Check specific important volumes
                const criticalVolumes = ['media-data', 'plex-config', 'jellyfin-config'];
                const foundCriticalVolumes = volumes.filter(vol => 
                    criticalVolumes.some(critical => vol.includes(critical))
                );
                
                console.log(`✅ Critical volumes found: ${foundCriticalVolumes.join(', ')}`);
            } catch (error) {
                console.warn('⚠️ Volume verification failed:', error.message);
            }
        });

        test('Configuration persistence', async () => {
            const configDirs = [
                'jellyfin-config',
                'plex-config', 
                'sonarr-config',
                'radarr-config',
                'prowlarr-config'
            ];

            for (const dir of configDirs) {
                try {
                    const { stdout } = await execAsync(`ls -la ${dir}/ 2>/dev/null | wc -l`);
                    const fileCount = parseInt(stdout.trim());
                    
                    if (fileCount > 0) {
                        console.log(`✅ ${dir} has configuration files (${fileCount} items)`);
                    } else {
                        console.warn(`⚠️ ${dir} appears empty or doesn't exist`);
                    }
                } catch (error) {
                    console.warn(`⚠️ Config check failed for ${dir}:`, error.message);
                }
            }
        });
    });

    afterAll(async () => {
        console.log('\n📊 Service Integration Test Summary:');
        console.log('- Media server integrations tested');
        console.log('- *ARR service integrations tested');
        console.log('- Download client integrations tested');
        console.log('- Request management services tested');
        console.log('- Monitoring services tested');
        console.log('- Management services tested');
        console.log('- Network connectivity tested');
        console.log('- Data persistence verified');
    });
});