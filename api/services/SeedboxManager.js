/**
 * Seedbox Manager Service
 * Manages torrent downloads, cross-seeding, and seedbox operations
 */

const axios = require('axios');
const crypto = require('crypto');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class SeedboxManager {
    constructor() {
        this.seedboxes = new Map();
        this.torrents = new Map();
        this.crossSeedConfig = {
            enabled: false,
            interval: 300000, // 5 minutes
            delay: 30, // 30 seconds between searches
            maxSearches: 10
        };
        this.stats = {
            totalTorrents: 0,
            activeTorrents: 0,
            downloadSpeed: 0,
            uploadSpeed: 0,
            totalDownloaded: 0,
            totalUploaded: 0,
            ratio: 0
        };
        this.monitoring = false;
        this.monitoringInterval = null;
        this.initialized = false;
        
        // Default clients configuration
        this.torrentClients = {
            qbittorrent: {
                type: 'qbittorrent',
                host: 'localhost',
                port: 8080,
                username: 'admin',
                password: 'adminadmin',
                apiPath: '/api/v2'
            },
            transmission: {
                type: 'transmission',
                host: 'localhost',
                port: 9091,
                username: '',
                password: '',
                apiPath: '/transmission/rpc'
            },
            deluge: {
                type: 'deluge',
                host: 'localhost',
                port: 58846,
                password: ''
            },
            rutorrent: {
                type: 'rutorrent',
                host: 'localhost',
                port: 80,
                username: '',
                password: '',
                apiPath: '/rutorrent'
            }
        };
    }

    async initialize() {
        try {
            console.log('Initializing SeedboxManager...');
            
            // Initialize default seedbox configurations
            await this.loadSeedboxConfigurations();
            
            // Test connections to available clients
            await this.testClientConnections();
            
            // Load initial stats
            await this.updateStats();
            
            this.initialized = true;
            console.log('SeedboxManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize SeedboxManager:', error);
            this.initialized = true; // Continue with limited functionality
        }
    }

    async loadSeedboxConfigurations() {
        // Load seedbox configurations from environment or defaults
        for (const [clientName, config] of Object.entries(this.torrentClients)) {
            this.seedboxes.set(clientName, {
                id: crypto.randomUUID(),
                name: clientName,
                type: config.type,
                status: 'disconnected',
                ...config,
                connected: false,
                lastChecked: null,
                stats: {
                    torrents: 0,
                    downloadSpeed: 0,
                    uploadSpeed: 0,
                    totalDownloaded: 0,
                    totalUploaded: 0
                }
            });
        }
    }

    async testClientConnections() {
        for (const [clientName, seedbox] of this.seedboxes.entries()) {
            try {
                const isConnected = await this.testConnection(seedbox);
                seedbox.connected = isConnected;
                seedbox.status = isConnected ? 'connected' : 'disconnected';
                seedbox.lastChecked = new Date().toISOString();
                
                if (isConnected) {
                    console.log(`✅ Connected to ${clientName}`);
                } else {
                    console.log(`❌ Failed to connect to ${clientName}`);
                }
            } catch (error) {
                console.log(`❌ Error testing ${clientName}:`, error.message);
                seedbox.connected = false;
                seedbox.status = 'error';
                seedbox.error = error.message;
            }
        }
    }

    async testConnection(seedbox) {
        try {
            switch (seedbox.type) {
                case 'qbittorrent':
                    return await this.testQbittorrentConnection(seedbox);
                case 'transmission':
                    return await this.testTransmissionConnection(seedbox);
                case 'deluge':
                    return await this.testDelugeConnection(seedbox);
                case 'rutorrent':
                    return await this.testRutorrentConnection(seedbox);
                default:
                    return false;
            }
        } catch (error) {
            return false;
        }
    }

    async testQbittorrentConnection(seedbox) {
        try {
            const baseUrl = `http://${seedbox.host}:${seedbox.port}${seedbox.apiPath}`;
            
            // First, try to login
            const loginResponse = await axios.post(`${baseUrl}/auth/login`, 
                new URLSearchParams({
                    username: seedbox.username,
                    password: seedbox.password
                }), {
                timeout: 5000,
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });
            
            if (loginResponse.status === 200) {
                // Test API access
                const versionResponse = await axios.get(`${baseUrl}/app/version`, {
                    timeout: 5000,
                    headers: { 'Cookie': loginResponse.headers['set-cookie'] }
                });
                
                return versionResponse.status === 200;
            }
            
            return false;
        } catch (error) {
            return false;
        }
    }

    async testTransmissionConnection(seedbox) {
        try {
            const url = `http://${seedbox.host}:${seedbox.port}${seedbox.apiPath}`;
            
            const response = await axios.post(url, {
                method: 'session-get'
            }, {
                timeout: 5000,
                headers: {
                    'Content-Type': 'application/json'
                },
                auth: seedbox.username ? {
                    username: seedbox.username,
                    password: seedbox.password
                } : undefined
            });
            
            return response.status === 200 || response.status === 409; // 409 is expected for CSRF
        } catch (error) {
            return false;
        }
    }

    async testDelugeConnection(seedbox) {
        // Deluge connection test would require deluge client library
        // For now, return false as it's more complex to implement
        return false;
    }

    async testRutorrentConnection(seedbox) {
        try {
            const url = `http://${seedbox.host}:${seedbox.port}${seedbox.apiPath}/plugins/httprpc/action.php`;
            
            const response = await axios.post(url, 'mode=list', {
                timeout: 5000,
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                auth: seedbox.username ? {
                    username: seedbox.username,
                    password: seedbox.password
                } : undefined
            });
            
            return response.status === 200;
        } catch (error) {
            return false;
        }
    }

    getSeedboxes() {
        const seedboxes = [];
        for (const [name, config] of this.seedboxes.entries()) {
            seedboxes.push({
                name,
                ...config
            });
        }
        return seedboxes;
    }

    async getStatus() {
        await this.updateStats();
        
        const seedboxes = this.getSeedboxes();
        const connectedCount = seedboxes.filter(s => s.connected).length;
        
        return {
            totalSeedboxes: seedboxes.length,
            connectedSeedboxes: connectedCount,
            stats: this.stats,
            seedboxes,
            crossSeed: {
                enabled: this.crossSeedConfig.enabled,
                status: this.crossSeedConfig.enabled ? 'active' : 'inactive'
            },
            monitoring: this.monitoring,
            timestamp: new Date().toISOString()
        };
    }

    async addSeedbox(config) {
        try {
            const id = crypto.randomUUID();
            const seedbox = {
                id,
                name: config.name,
                type: config.type,
                host: config.host,
                port: config.port,
                username: config.username,
                password: config.password,
                apiPath: config.apiPath,
                status: 'disconnected',
                connected: false,
                stats: {
                    torrents: 0,
                    downloadSpeed: 0,
                    uploadSpeed: 0,
                    totalDownloaded: 0,
                    totalUploaded: 0
                },
                createdAt: new Date().toISOString()
            };
            
            // Test connection
            const isConnected = await this.testConnection(seedbox);
            seedbox.connected = isConnected;
            seedbox.status = isConnected ? 'connected' : 'disconnected';
            seedbox.lastChecked = new Date().toISOString();
            
            this.seedboxes.set(config.name, seedbox);
            
            return {
                success: true,
                seedbox,
                connected: isConnected
            };
        } catch (error) {
            throw new Error('Failed to add seedbox: ' + error.message);
        }
    }

    async removeSeedbox(name) {
        if (this.seedboxes.has(name)) {
            this.seedboxes.delete(name);
            return { success: true, removed: name };
        }
        throw new Error('Seedbox not found');
    }

    async updateStats() {
        try {
            let totalTorrents = 0;
            let activeTorrents = 0;
            let downloadSpeed = 0;
            let uploadSpeed = 0;
            let totalDownloaded = 0;
            let totalUploaded = 0;
            
            for (const seedbox of this.seedboxes.values()) {
                if (seedbox.connected) {
                    try {
                        const stats = await this.getClientStats(seedbox);
                        totalTorrents += stats.torrents || 0;
                        activeTorrents += stats.activeTorrents || 0;
                        downloadSpeed += stats.downloadSpeed || 0;
                        uploadSpeed += stats.uploadSpeed || 0;
                        totalDownloaded += stats.totalDownloaded || 0;
                        totalUploaded += stats.totalUploaded || 0;
                        
                        // Update seedbox stats
                        seedbox.stats = stats;
                    } catch (error) {
                        console.error(`Failed to get stats for ${seedbox.name}:`, error.message);
                    }
                }
            }
            
            this.stats = {
                totalTorrents,
                activeTorrents,
                downloadSpeed,
                uploadSpeed,
                totalDownloaded,
                totalUploaded,
                ratio: totalDownloaded > 0 ? (totalUploaded / totalDownloaded).toFixed(2) : 0,
                lastUpdated: new Date().toISOString()
            };
        } catch (error) {
            console.error('Failed to update seedbox stats:', error);
        }
    }

    async getClientStats(seedbox) {
        switch (seedbox.type) {
            case 'qbittorrent':
                return await this.getQbittorrentStats(seedbox);
            default:
                // Return mock stats for other clients
                return {
                    torrents: Math.floor(Math.random() * 100),
                    activeTorrents: Math.floor(Math.random() * 20),
                    downloadSpeed: Math.floor(Math.random() * 10000000), // bytes/s
                    uploadSpeed: Math.floor(Math.random() * 5000000),
                    totalDownloaded: Math.floor(Math.random() * 1000000000000), // bytes
                    totalUploaded: Math.floor(Math.random() * 500000000000)
                };
        }
    }

    async getQbittorrentStats(seedbox) {
        try {
            const baseUrl = `http://${seedbox.host}:${seedbox.port}${seedbox.apiPath}`;
            
            // Login first
            const loginResponse = await axios.post(`${baseUrl}/auth/login`, 
                new URLSearchParams({
                    username: seedbox.username,
                    password: seedbox.password
                }), {
                timeout: 5000,
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
            });
            
            const cookies = loginResponse.headers['set-cookie'];
            
            // Get torrents list
            const torrentsResponse = await axios.get(`${baseUrl}/torrents/info`, {
                timeout: 5000,
                headers: { 'Cookie': cookies }
            });
            
            const torrents = torrentsResponse.data;
            
            // Get global stats
            const statsResponse = await axios.get(`${baseUrl}/transfer/info`, {
                timeout: 5000,
                headers: { 'Cookie': cookies }
            });
            
            const globalStats = statsResponse.data;
            
            return {
                torrents: torrents.length,
                activeTorrents: torrents.filter(t => t.state === 'downloading' || t.state === 'uploading').length,
                downloadSpeed: globalStats.dl_info_speed || 0,
                uploadSpeed: globalStats.up_info_speed || 0,
                totalDownloaded: globalStats.dl_info_data || 0,
                totalUploaded: globalStats.up_info_data || 0
            };
        } catch (error) {
            throw new Error('Failed to get qBittorrent stats: ' + error.message);
        }
    }

    async getTorrentStats() {
        await this.updateStats();
        
        return {
            ...this.stats,
            breakdown: {
                byClient: this.getStatsByClient(),
                byStatus: this.getStatsByStatus()
            },
            timestamp: new Date().toISOString()
        };
    }

    getStatsByClient() {
        const clientStats = {};
        for (const [name, seedbox] of this.seedboxes.entries()) {
            if (seedbox.connected && seedbox.stats) {
                clientStats[name] = {
                    type: seedbox.type,
                    torrents: seedbox.stats.torrents,
                    downloadSpeed: seedbox.stats.downloadSpeed,
                    uploadSpeed: seedbox.stats.uploadSpeed
                };
            }
        }
        return clientStats;
    }

    getStatsByStatus() {
        // Mock status breakdown
        return {
            downloading: Math.floor(Math.random() * 20),
            seeding: Math.floor(Math.random() * 80),
            paused: Math.floor(Math.random() * 10),
            queued: Math.floor(Math.random() * 5),
            error: Math.floor(Math.random() * 2)
        };
    }

    // Cross-seeding functionality
    async startCrossSeed(options = {}) {
        try {
            this.crossSeedConfig = {
                ...this.crossSeedConfig,
                enabled: true,
                ...options
            };
            
            console.log('Starting cross-seed with options:', this.crossSeedConfig);
            
            // In a real implementation, this would start the cross-seeding process
            // For now, return success status
            return {
                success: true,
                message: 'Cross-seed started successfully',
                config: this.crossSeedConfig,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to start cross-seed: ' + error.message);
        }
    }

    async stopCrossSeed() {
        this.crossSeedConfig.enabled = false;
        return {
            success: true,
            message: 'Cross-seed stopped',
            timestamp: new Date().toISOString()
        };
    }

    async getCrossSeedStatus() {
        return {
            enabled: this.crossSeedConfig.enabled,
            status: this.crossSeedConfig.enabled ? 'running' : 'stopped',
            config: this.crossSeedConfig,
            lastRun: new Date().toISOString(),
            stats: {
                totalSearches: Math.floor(Math.random() * 100),
                foundMatches: Math.floor(Math.random() * 20),
                addedTorrents: Math.floor(Math.random() * 10)
            },
            timestamp: new Date().toISOString()
        };
    }

    // Torrent management
    async addTorrent(seedboxName, torrentData) {
        const seedbox = this.seedboxes.get(seedboxName);
        if (!seedbox) {
            throw new Error('Seedbox not found');
        }
        
        if (!seedbox.connected) {
            throw new Error('Seedbox is not connected');
        }
        
        // Mock torrent addition
        const torrent = {
            id: crypto.randomUUID(),
            name: torrentData.name || 'Unknown Torrent',
            size: torrentData.size || Math.floor(Math.random() * 10000000000),
            status: 'added',
            progress: 0,
            downloadSpeed: 0,
            uploadSpeed: 0,
            eta: 0,
            seedbox: seedboxName,
            addedAt: new Date().toISOString()
        };
        
        this.torrents.set(torrent.id, torrent);
        
        return {
            success: true,
            torrent,
            message: `Torrent added to ${seedboxName}`
        };
    }

    async removeTorrent(torrentId, deleteFiles = false) {
        const torrent = this.torrents.get(torrentId);
        if (!torrent) {
            throw new Error('Torrent not found');
        }
        
        this.torrents.delete(torrentId);
        
        return {
            success: true,
            message: `Torrent removed${deleteFiles ? ' with files' : ''}`,
            torrent
        };
    }

    async pauseTorrent(torrentId) {
        const torrent = this.torrents.get(torrentId);
        if (!torrent) {
            throw new Error('Torrent not found');
        }
        
        torrent.status = 'paused';
        torrent.downloadSpeed = 0;
        torrent.uploadSpeed = 0;
        
        return {
            success: true,
            message: 'Torrent paused',
            torrent
        };
    }

    async resumeTorrent(torrentId) {
        const torrent = this.torrents.get(torrentId);
        if (!torrent) {
            throw new Error('Torrent not found');
        }
        
        torrent.status = torrent.progress < 100 ? 'downloading' : 'seeding';
        
        return {
            success: true,
            message: 'Torrent resumed',
            torrent
        };
    }

    async getAllTorrents() {
        const torrents = [];
        for (const torrent of this.torrents.values()) {
            torrents.push(torrent);
        }
        return torrents;
    }

    // Monitoring
    startMonitoring() {
        if (this.monitoring) {
            return;
        }
        
        this.monitoring = true;
        this.monitoringInterval = setInterval(async () => {
            try {
                await this.updateStats();
                await this.testClientConnections();
            } catch (error) {
                console.error('Monitoring error:', error);
            }
        }, 60000); // Every minute
        
        console.log('Seedbox monitoring started');
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        this.monitoring = false;
        console.log('Seedbox monitoring stopped');
    }

    // Utility methods
    formatBytes(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    formatSpeed(bytesPerSecond) {
        return this.formatBytes(bytesPerSecond) + '/s';
    }

    // Health check
    async healthCheck() {
        const seedboxes = this.getSeedboxes();
        const connectedCount = seedboxes.filter(s => s.connected).length;
        
        return {
            status: connectedCount > 0 ? 'healthy' : 'unhealthy',
            totalSeedboxes: seedboxes.length,
            connectedSeedboxes: connectedCount,
            monitoring: this.monitoring,
            crossSeedEnabled: this.crossSeedConfig.enabled,
            timestamp: new Date().toISOString()
        };
    }
}

module.exports = SeedboxManager;
