const logger = require('../../middleware/logger.js');
/**
 * VPNService - Gluetun VPN tunnel management for download protection
 * Provides VPN management and monitoring for secure downloading through Gluetun container
 */

const axios = require('axios');
const { exec } = require('child_process');
const { promisify } = require('util');
const EventEmitter = require('events');

const execAsync = promisify(exec);

class VPNService extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            gluetunUrl: config.gluetunUrl || process.env.GLUETUN_URL || 'http://gluetun:8000',
            gluetunControlPort: config.gluetunControlPort || process.env.GLUETUN_CONTROL_PORT || 8000,
            containerName: config.containerName || 'gluetun',
            vpnProvider: config.vpnProvider || process.env.VPN_PROVIDER || 'nordvpn',
            vpnRegion: config.vpnRegion || process.env.VPN_REGION || 'Switzerland',
            healthCheckInterval: config.healthCheckInterval || 30000, // 30 seconds
            reconnectDelay: config.reconnectDelay || 60000, // 1 minute
            killSwitch: config.killSwitch !== false, // Default enabled
            dnsLeakProtection: config.dnsLeakProtection !== false,
            enablePortForwarding: config.enablePortForwarding || false,
            ...config
        };

        this.vpnStatus = {
            connected: false,
            server: null,
            ip: null,
            location: null,
            protocol: null,
            lastConnected: null,
            uptime: 0
        };

        this.healthCheckTimer = null;
        this.reconnectTimer = null;
        this.isInitialized = false;
        this.connectionHistory = [];
        this.speedTests = [];
        
        this.supportedProviders = {
            nordvpn: { name: 'NordVPN', type: 'openvpn' },
            expressvpn: { name: 'ExpressVPN', type: 'openvpn' },
            surfshark: { name: 'Surfshark', type: 'openvpn' },
            protonvpn: { name: 'ProtonVPN', type: 'openvpn' },
            cyberghost: { name: 'CyberGhost', type: 'openvpn' },
            pia: { name: 'Private Internet Access', type: 'openvpn' },
            windscribe: { name: 'Windscribe', type: 'openvpn' },
            mullvad: { name: 'Mullvad', type: 'wireguard' }
        };
    }

    /**
     * Initialize VPN service
     */
    async initialize() {
        try {
            logger.info('🔐 Initializing VPNService...');
            
            // Check if Gluetun container is running
            await this.checkGluetunContainer();
            
            // Get initial VPN status
            await this.updateVPNStatus();
            
            // Start health monitoring
            this.startHealthMonitoring();
            
            // Load connection history
            await this.loadConnectionHistory();
            
            this.isInitialized = true;
            this.emit('initialized');
            logger.info('✅ VPNService initialized successfully');
            
            return { success: true, message: 'VPNService initialized' };
        } catch (error) {
            logger.error('❌ VPNService initialization failed:', error);
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Check if Gluetun container is running
     */
    async checkGluetunContainer() {
        try {
            const { stdout } = await execAsync(`docker ps --filter name=${this.config.containerName} --format "table {{.Names}}\t{{.Status}}"`);
            
            if (!stdout.includes(this.config.containerName)) {
                throw new Error(`Gluetun container '${this.config.containerName}' is not running`);
            }
            
            logger.info(`✅ Gluetun container '${this.config.containerName}' is running`);
            return true;
        } catch (error) {
            logger.error('❌ Gluetun container check failed:', error);
            throw error;
        }
    }

    /**
     * Get VPN status from Gluetun
     */
    async updateVPNStatus() {
        try {
            // Check VPN connection status
            const statusResponse = await this.makeGluetunRequest('/status');
            
            // Get public IP
            const ipResponse = await this.makeGluetunRequest('/ip');
            
            // Get server info
            const serverResponse = await this.makeGluetunRequest('/server');
            
            this.vpnStatus = {
                connected: statusResponse.status === 'connected',
                server: serverResponse.server || null,
                ip: ipResponse.ip || null,
                location: serverResponse.location || null,
                protocol: serverResponse.protocol || null,
                lastConnected: this.vpnStatus.connected ? (this.vpnStatus.lastConnected || new Date()) : null,
                uptime: this.vpnStatus.connected ? Date.now() - (this.vpnStatus.lastConnected || Date.now()) : 0
            };
            
            this.emit('statusUpdated', this.vpnStatus);
            return this.vpnStatus;
        } catch (error) {
            logger.warn('⚠️ VPN status update failed:', error.message);
            
            // Fallback: assume disconnected if can't get status
            this.vpnStatus.connected = false;
            return this.vpnStatus;
        }
    }

    /**
     * Make request to Gluetun API
     */
    async makeGluetunRequest(endpoint, method = 'GET', data = null) {
        try {
            const response = await axios({
                method,
                url: `${this.config.gluetunUrl}${endpoint}`,
                data,
                timeout: 10000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            return response.data;
        } catch (error) {
            if (error.code === 'ECONNREFUSED') {
                throw new Error('Cannot connect to Gluetun API');
            }
            throw error;
        }
    }

    /**
     * Connect to VPN
     */
    async connect(options = {}) {
        try {
            logger.info('🔌 Connecting to VPN...');
            
            const connectOptions = {
                provider: options.provider || this.config.vpnProvider,
                region: options.region || this.config.vpnRegion,
                protocol: options.protocol || 'openvpn'
            };
            
            // Send connect request to Gluetun
            await this.makeGluetunRequest('/connect', 'POST', connectOptions);
            
            // Wait for connection to establish
            await this.waitForConnection();
            
            // Update status
            await this.updateVPNStatus();
            
            // Log connection
            await this.logConnection('connected', connectOptions);
            
            this.emit('connected', this.vpnStatus);
            logger.info(`✅ VPN connected: ${this.vpnStatus.server} (${this.vpnStatus.location})`);
            
            return {
                success: true,
                message: 'VPN connected successfully',
                status: this.vpnStatus
            };
        } catch (error) {
            logger.error('❌ VPN connection failed:', error);
            await this.logConnection('failed', { error: error.message });
            throw error;
        }
    }

    /**
     * Disconnect from VPN
     */
    async disconnect() {
        try {
            logger.info('🔌 Disconnecting from VPN...');
            
            // Send disconnect request to Gluetun
            await this.makeGluetunRequest('/disconnect', 'POST');
            
            // Wait for disconnection
            await this.waitForDisconnection();
            
            // Update status
            await this.updateVPNStatus();
            
            // Log disconnection
            await this.logConnection('disconnected');
            
            this.emit('disconnected');
            logger.info('✅ VPN disconnected');
            
            return {
                success: true,
                message: 'VPN disconnected successfully'
            };
        } catch (error) {
            logger.error('❌ VPN disconnection failed:', error);
            throw error;
        }
    }

    /**
     * Reconnect VPN
     */
    async reconnect(options = {}) {
        try {
            logger.info('🔁 Reconnecting VPN...');
            
            // Disconnect first if connected
            if (this.vpnStatus.connected) {
                await this.disconnect();
            }
            
            // Wait before reconnecting
            await new Promise(resolve => setTimeout(resolve, this.config.reconnectDelay));
            
            // Connect with new options
            return await this.connect(options);
        } catch (error) {
            logger.error('❌ VPN reconnection failed:', error);
            throw error;
        }
    }

    /**
     * Wait for VPN connection to establish
     */
    async waitForConnection(maxWait = 30000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait) {
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            try {
                await this.updateVPNStatus();
                if (this.vpnStatus.connected) {
                    return true;
                }
            } catch (error) {
                // Continue waiting
            }
        }
        
        throw new Error('VPN connection timeout');
    }

    /**
     * Wait for VPN disconnection
     */
    async waitForDisconnection(maxWait = 15000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            try {
                await this.updateVPNStatus();
                if (!this.vpnStatus.connected) {
                    return true;
                }
            } catch (error) {
                // Assume disconnected if can't get status
                return true;
            }
        }
        
        return true; // Assume disconnected after timeout
    }

    /**
     * Get available VPN servers
     */
    async getAvailableServers(provider = null) {
        try {
            const targetProvider = provider || this.config.vpnProvider;
            
            // Get servers from Gluetun
            const servers = await this.makeGluetunRequest(`/servers/${targetProvider}`);
            
            return {
                success: true,
                provider: targetProvider,
                servers: servers.servers || [],
                count: servers.servers?.length || 0
            };
        } catch (error) {
            logger.error('❌ Failed to get available servers:', error);
            
            // Return mock servers if API fails
            return {
                success: false,
                provider: targetProvider,
                servers: this.getMockServers(targetProvider),
                count: 0,
                error: error.message
            };
        }
    }

    /**
     * Get mock servers for fallback
     */
    getMockServers(provider) {
        const mockServers = {
            nordvpn: [
                { name: 'Switzerland #1', location: 'Switzerland', ip: '185.246.209.105' },
                { name: 'Netherlands #1', location: 'Netherlands', ip: '146.70.122.81' },
                { name: 'Germany #1', location: 'Germany', ip: '77.68.21.75' }
            ],
            expressvpn: [
                { name: 'Switzerland', location: 'Switzerland', ip: '185.159.157.71' },
                { name: 'Netherlands', location: 'Netherlands', ip: '185.159.158.236' }
            ]
        };
        
        return mockServers[provider] || [];
    }

    /**
     * Test VPN connection speed
     */
    async testSpeed() {
        try {
            logger.info('📊 Testing VPN speed...');
            
            if (!this.vpnStatus.connected) {
                throw new Error('VPN is not connected');
            }
            
            const startTime = Date.now();
            
            // Test download speed using a speed test service
            const response = await axios.get('https://speed.cloudflare.com/__down?bytes=10000000', {
                timeout: 30000,
                responseType: 'stream'
            });
            
            let downloadedBytes = 0;
            
            await new Promise((resolve, reject) => {
                response.data.on('data', (chunk) => {
                    downloadedBytes += chunk.length;
                });
                
                response.data.on('end', resolve);
                response.data.on('error', reject);
            });
            
            const endTime = Date.now();
            const duration = (endTime - startTime) / 1000; // seconds
            const speedMbps = (downloadedBytes * 8) / (1024 * 1024 * duration);
            
            const speedTest = {
                timestamp: new Date(),
                server: this.vpnStatus.server,
                location: this.vpnStatus.location,
                downloadSpeed: parseFloat(speedMbps.toFixed(2)),
                downloadedBytes,
                duration
            };
            
            this.speedTests.push(speedTest);
            
            // Keep only last 10 speed tests
            if (this.speedTests.length > 10) {
                this.speedTests = this.speedTests.slice(-10);
            }
            
            this.emit('speedTest', speedTest);
            logger.info(`✅ Speed test completed: ${speedTest.downloadSpeed} Mbps`);
            
            return {
                success: true,
                speedTest
            };
        } catch (error) {
            logger.error('❌ Speed test failed:', error);
            throw error;
        }
    }

    /**
     * Check for DNS leaks
     */
    async checkDNSLeak() {
        try {
            logger.info('🔍 Checking for DNS leaks...');
            
            if (!this.vpnStatus.connected) {
                throw new Error('VPN is not connected');
            }
            
            // Check DNS servers
            const dnsResponse = await axios.get('https://1.1.1.1/cdn-cgi/trace', {
                timeout: 10000
            });
            
            const traces = dnsResponse.data.split('\n').reduce((acc, line) => {
                const [key, value] = line.split('=');
                if (key && value) acc[key] = value;
                return acc;
            }, {});
            
            const leak = {
                timestamp: new Date(),
                ip: traces.ip,
                country: traces.loc,
                asn: traces.asn,
                isLeak: traces.loc !== this.vpnStatus.location
            };
            
            this.emit('dnsLeakCheck', leak);
            
            if (leak.isLeak) {
                logger.warn('⚠️ DNS leak detected!');
            } else {
                logger.info('✅ No DNS leak detected');
            }
            
            return {
                success: true,
                leak
            };
        } catch (error) {
            logger.error('❌ DNS leak check failed:', error);
            throw error;
        }
    }

    /**
     * Start health monitoring
     */
    startHealthMonitoring() {
        if (this.healthCheckTimer) {
            clearInterval(this.healthCheckTimer);
        }
        
        this.healthCheckTimer = setInterval(async () => {
            try {
                const previousStatus = this.vpnStatus.connected;
                await this.updateVPNStatus();
                
                // Check for connection changes
                if (previousStatus && !this.vpnStatus.connected) {
                    logger.warn('⚠️ VPN connection lost!');
                    this.emit('connectionLost');
                    
                    // Auto-reconnect if enabled
                    if (this.config.autoReconnect) {
                        setTimeout(() => this.reconnect(), this.config.reconnectDelay);
                    }
                } else if (!previousStatus && this.vpnStatus.connected) {
                    logger.info('✅ VPN connection restored');
                    this.emit('connectionRestored');
                }
            } catch (error) {
                logger.warn('⚠️ Health check failed:', error.message);
            }
        }, this.config.healthCheckInterval);
        
        logger.info('✅ VPN health monitoring started');
    }

    /**
     * Log connection event
     */
    async logConnection(event, data = {}) {
        try {
            const entry = {
                timestamp: new Date(),
                event,
                server: this.vpnStatus.server,
                location: this.vpnStatus.location,
                ip: this.vpnStatus.ip,
                ...data
            };
            
            this.connectionHistory.push(entry);
            
            // Keep only last 100 entries
            if (this.connectionHistory.length > 100) {
                this.connectionHistory = this.connectionHistory.slice(-100);
            }
            
            this.emit('connectionLog', entry);
        } catch (error) {
            logger.error('❌ Connection logging failed:', error);
        }
    }

    /**
     * Load connection history
     */
    async loadConnectionHistory() {
        try {
            // In production, load from persistent storage
            logger.info('📚 Loading connection history...');
            
            // For now, start with empty history
            this.connectionHistory = [];
            
            logger.info('✅ Connection history loaded');
        } catch (error) {
            logger.warn('⚠️ Connection history loading failed:', error.message);
        }
    }

    /**
     * Get kill switch status
     */
    async getKillSwitchStatus() {
        try {
            // Check iptables rules for kill switch
            const { stdout } = await execAsync('docker exec gluetun iptables -L');
            
            const hasKillSwitch = stdout.includes('DROP') && stdout.includes('!tun');
            
            return {
                enabled: hasKillSwitch,
                active: hasKillSwitch && this.vpnStatus.connected
            };
        } catch (error) {
            logger.warn('⚠️ Kill switch status check failed:', error.message);
            return { enabled: false, active: false };
        }
    }

    /**
     * Get service status
     */
    getStatus() {
        return {
            initialized: this.isInitialized,
            vpnStatus: this.vpnStatus,
            provider: this.config.vpnProvider,
            region: this.config.vpnRegion,
            killSwitch: this.config.killSwitch,
            dnsLeakProtection: this.config.dnsLeakProtection,
            connectionHistory: this.connectionHistory.length,
            speedTests: this.speedTests.length,
            healthMonitoring: !!this.healthCheckTimer,
            lastSpeedTest: this.speedTests[this.speedTests.length - 1] || null,
            uptime: this.vpnStatus.connected ? Date.now() - (this.vpnStatus.lastConnected || Date.now()) : 0,
            lastUpdate: new Date()
        };
    }

    /**
     * Get connection statistics
     */
    getConnectionStats() {
        const totalConnections = this.connectionHistory.filter(entry => entry.event === 'connected').length;
        const totalDisconnections = this.connectionHistory.filter(entry => entry.event === 'disconnected').length;
        const failedConnections = this.connectionHistory.filter(entry => entry.event === 'failed').length;
        
        const avgSpeed = this.speedTests.length > 0 
            ? this.speedTests.reduce((sum, test) => sum + test.downloadSpeed, 0) / this.speedTests.length
            : 0;
        
        return {
            totalConnections,
            totalDisconnections,
            failedConnections,
            successRate: totalConnections > 0 ? ((totalConnections / (totalConnections + failedConnections)) * 100).toFixed(2) : 0,
            averageSpeed: parseFloat(avgSpeed.toFixed(2)),
            speedTests: this.speedTests.length,
            currentUptime: this.vpnStatus.connected ? Date.now() - (this.vpnStatus.lastConnected || Date.now()) : 0
        };
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger.info('🧹 Cleaning up VPNService...');
            
            if (this.healthCheckTimer) {
                clearInterval(this.healthCheckTimer);
                this.healthCheckTimer = null;
            }
            
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }
            
            this.connectionHistory = [];
            this.speedTests = [];
            this.removeAllListeners();
            
            this.isInitialized = false;
            logger.info('✅ VPNService cleanup completed');
        } catch (error) {
            logger.error('❌ VPNService cleanup failed:', error);
        }
    }
}

module.exports = VPNService;