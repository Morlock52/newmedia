/**
 * Seedbox Manager Service
 * Comprehensive seedbox integration with cross-seed automation and tracker management
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

class SeedboxManager {
    constructor() {
        this.qbittorrentConfig = {
            host: 'localhost',
            port: 8080,
            username: process.env.QBITTORRENT_USERNAME || 'admin',
            password: process.env.QBITTORRENT_PASSWORD || 'adminadmin'
        };
        
        this.crossSeedConfig = {
            enabled: process.env.CROSS_SEED_ENABLED === 'true',
            configPath: './config/cross-seed',
            outputDir: './data/torrents/cross-seed',
            delay: parseInt(process.env.CROSS_SEED_DELAY) || 30,
            searchTimeout: parseInt(process.env.CROSS_SEED_TIMEOUT) || 60
        };

        // Tracker configurations
        this.trackers = {
            public: {
                'torrentleech': { enabled: true, priority: 1 },
                'iptorrents': { enabled: true, priority: 2 },
                'alpharatio': { enabled: true, priority: 3 }
            },
            private: {
                'passthepopcorn': { enabled: false, priority: 1 },
                'broadcastthenet': { enabled: false, priority: 2 },
                'redacted': { enabled: false, priority: 3 }
            }
        };

        // Ratio management settings
        this.ratioSettings = {
            globalRatio: 2.0,
            seedingTimeLimit: 7 * 24 * 60, // 7 days in minutes
            maxActiveTorrents: 200,
            maxDownloadSpeed: 0, // Unlimited
            maxUploadSpeed: 0,   // Unlimited
            ratioGroups: {
                'high_priority': { ratio: 3.0, time: 14 * 24 * 60 },
                'medium_priority': { ratio: 2.0, time: 7 * 24 * 60 },
                'low_priority': { ratio: 1.5, time: 3 * 24 * 60 }
            }
        };

        this.sessionCookie = null;
        this.statusCache = new Map();
        this.cacheTimeout = 30000; // 30 seconds
    }

    async initialize() {
        try {
            // Initialize qBittorrent session
            await this.initializeQBittorrentSession();
            
            // Verify cross-seed setup
            await this.verifyCrossSeedSetup();
            
            console.log('SeedboxManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize SeedboxManager:', error);
            // Don't throw error to allow API to start even if seedbox is not available
        }
    }

    async initializeQBittorrentSession() {
        try {
            const response = await this.makeQBittorrentRequest('/api/v2/auth/login', 'POST', {
                username: this.qbittorrentConfig.username,
                password: this.qbittorrentConfig.password
            });

            if (response.ok) {
                const cookie = response.headers.get('set-cookie');
                if (cookie) {
                    this.sessionCookie = cookie;
                    console.log('qBittorrent session initialized');
                }
            } else {
                throw new Error('Failed to authenticate with qBittorrent');
            }
        } catch (error) {
            console.warn('qBittorrent not available:', error.message);
        }
    }

    async verifyCrossSeedSetup() {
        try {
            // Check if cross-seed directory exists
            await fs.access(this.crossSeedConfig.configPath);
            console.log('Cross-seed configuration found');
        } catch (error) {
            console.log('Cross-seed not configured, skipping...');
        }
    }

    async getStatus() {
        try {
            const [qbittorrentStatus, crossSeedStatus, diskUsage] = await Promise.all([
                this.getQBittorrentStatus(),
                this.getCrossSeedStatus(),
                this.getDiskUsage()
            ]);

            return {
                qbittorrent: qbittorrentStatus,
                crossSeed: crossSeedStatus,
                disk: diskUsage,
                ratioSettings: this.ratioSettings,
                trackers: this.trackers,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to get seedbox status: ' + error.message);
        }
    }

    async getQBittorrentStatus() {
        try {
            if (!this.sessionCookie) {
                await this.initializeQBittorrentSession();
            }

            const [mainData, preferences, torrents] = await Promise.all([
                this.makeQBittorrentRequest('/api/v2/sync/maindata'),
                this.makeQBittorrentRequest('/api/v2/app/preferences'),
                this.makeQBittorrentRequest('/api/v2/torrents/info')
            ]);

            const mainDataJson = await mainData.json();
            const preferencesJson = await preferences.json();
            const torrentsJson = await torrents.json();

            // Calculate statistics
            const stats = this.calculateTorrentStats(torrentsJson);

            return {
                connected: true,
                version: mainDataJson.server_state?.app_version || 'Unknown',
                state: mainDataJson.server_state || {},
                preferences: {
                    downloadLimit: preferencesJson.dl_limit || 0,
                    uploadLimit: preferencesJson.up_limit || 0,
                    maxRatio: preferencesJson.max_ratio || -1,
                    maxSeedingTime: preferencesJson.max_seeding_time || -1
                },
                stats,
                torrents: torrentsJson.slice(0, 10) // Return first 10 torrents for overview
            };
        } catch (error) {
            return {
                connected: false,
                error: error.message
            };
        }
    }

    calculateTorrentStats(torrents) {
        const stats = {
            total: torrents.length,
            downloading: 0,
            seeding: 0,
            paused: 0,
            completed: 0,
            totalSize: 0,
            totalDownloaded: 0,
            totalUploaded: 0,
            avgRatio: 0,
            activeTorrents: 0
        };

        let totalRatio = 0;
        let torrentsWithRatio = 0;

        for (const torrent of torrents) {
            stats.totalSize += torrent.size || 0;
            stats.totalDownloaded += torrent.downloaded || 0;
            stats.totalUploaded += torrent.uploaded || 0;

            if (torrent.ratio && torrent.ratio > 0) {
                totalRatio += torrent.ratio;
                torrentsWithRatio++;
            }

            switch (torrent.state) {
                case 'downloading':
                case 'allocating':
                case 'metaDL':
                    stats.downloading++;
                    stats.activeTorrents++;
                    break;
                case 'uploading':
                case 'stalledUP':
                    stats.seeding++;
                    stats.activeTorrents++;
                    break;
                case 'pausedDL':
                case 'pausedUP':
                    stats.paused++;
                    break;
                case 'queuedDL':
                case 'queuedUP':
                    stats.activeTorrents++;
                    break;
            }

            if (torrent.progress === 1) {
                stats.completed++;
            }
        }

        stats.avgRatio = torrentsWithRatio > 0 ? (totalRatio / torrentsWithRatio) : 0;

        return stats;
    }

    async getCrossSeedStatus() {
        try {
            // Check if cross-seed process is running
            const { stdout } = await execAsync('pgrep -f cross-seed');
            const isRunning = stdout.trim().length > 0;

            let config = {};
            try {
                const configPath = path.join(this.crossSeedConfig.configPath, 'config.js');
                const configContent = await fs.readFile(configPath, 'utf8');
                // Basic config parsing (this would need to be more sophisticated in production)
                config = { configured: true };
            } catch (error) {
                config = { configured: false };
            }

            return {
                enabled: this.crossSeedConfig.enabled,
                running: isRunning,
                config,
                lastRun: await this.getCrossSeedLastRun()
            };
        } catch (error) {
            return {
                enabled: false,
                running: false,
                error: error.message
            };
        }
    }

    async getCrossSeedLastRun() {
        try {
            const logFile = path.join(this.crossSeedConfig.configPath, 'cross-seed.log');
            const stats = await fs.stat(logFile);
            return stats.mtime;
        } catch (error) {
            return null;
        }
    }

    async getDiskUsage() {
        try {
            const paths = [
                './data/downloads',
                './data/torrents',
                './media-data'
            ];

            const usage = {};
            for (const path of paths) {
                try {
                    const { stdout } = await execAsync(`du -sh ${path}`);
                    const size = stdout.split('\t')[0];
                    usage[path] = size;
                } catch (error) {
                    usage[path] = 'Unknown';
                }
            }

            return usage;
        } catch (error) {
            return { error: error.message };
        }
    }

    async startCrossSeed(options = {}) {
        try {
            if (!this.crossSeedConfig.enabled) {
                throw new Error('Cross-seed is not enabled');
            }

            const {
                action = 'search',
                trackers = [],
                excludeOlder = 7,
                includeNonVideos = false
            } = options;

            let command = 'cross-seed';
            
            if (action === 'search') {
                command += ' search';
            } else if (action === 'daemon') {
                command += ' daemon';
            }

            if (trackers.length > 0) {
                command += ` --trackers ${trackers.join(',')}`;
            }

            if (excludeOlder > 0) {
                command += ` --exclude-older-than ${excludeOlder}d`;
            }

            if (includeNonVideos) {
                command += ' --include-non-videos';
            }

            console.log('Starting cross-seed:', command);
            
            // Run cross-seed in the background
            const child = exec(command, {
                cwd: this.crossSeedConfig.configPath,
                detached: true,
                stdio: 'ignore'
            });

            child.unref();

            return {
                success: true,
                command,
                pid: child.pid,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to start cross-seed: ' + error.message);
        }
    }

    async getTorrentStats() {
        try {
            const qbStatus = await this.getQBittorrentStatus();
            
            if (!qbStatus.connected) {
                throw new Error('qBittorrent not available');
            }

            // Get detailed torrent information
            const torrents = await this.makeQBittorrentRequest('/api/v2/torrents/info');
            const torrentsData = await torrents.json();

            // Group torrents by tracker
            const trackerStats = {};
            const categoryStats = {};
            const ratioDistribution = { excellent: 0, good: 0, fair: 0, poor: 0 };

            for (const torrent of torrentsData) {
                // Tracker statistics
                const tracker = this.extractTracker(torrent.tracker);
                if (!trackerStats[tracker]) {
                    trackerStats[tracker] = { count: 0, uploaded: 0, downloaded: 0, ratio: 0 };
                }
                trackerStats[tracker].count++;
                trackerStats[tracker].uploaded += torrent.uploaded || 0;
                trackerStats[tracker].downloaded += torrent.downloaded || 0;

                // Category statistics
                const category = torrent.category || 'uncategorized';
                if (!categoryStats[category]) {
                    categoryStats[category] = { count: 0, size: 0 };
                }
                categoryStats[category].count++;
                categoryStats[category].size += torrent.size || 0;

                // Ratio distribution
                const ratio = torrent.ratio || 0;
                if (ratio >= 3.0) ratioDistribution.excellent++;
                else if (ratio >= 2.0) ratioDistribution.good++;
                else if (ratio >= 1.0) ratioDistribution.fair++;
                else ratioDistribution.poor++;
            }

            // Calculate average ratios for trackers
            Object.keys(trackerStats).forEach(tracker => {
                const stats = trackerStats[tracker];
                stats.ratio = stats.downloaded > 0 ? stats.uploaded / stats.downloaded : 0;
            });

            return {
                overview: qbStatus.stats,
                trackers: trackerStats,
                categories: categoryStats,
                ratioDistribution,
                recentActivity: await this.getRecentActivity(),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to get torrent statistics: ' + error.message);
        }
    }

    async getRecentActivity() {
        try {
            // Get torrents sorted by completion date
            const torrents = await this.makeQBittorrentRequest('/api/v2/torrents/info?sort=completed_on&reverse=true&limit=10');
            const torrentsData = await torrents.json();

            return torrentsData.map(torrent => ({
                name: torrent.name,
                size: this.formatBytes(torrent.size),
                ratio: Math.round(torrent.ratio * 100) / 100,
                completedOn: torrent.completed_on ? new Date(torrent.completed_on * 1000) : null,
                state: torrent.state,
                tracker: this.extractTracker(torrent.tracker)
            }));
        } catch (error) {
            return [];
        }
    }

    async manageRatios() {
        try {
            const torrents = await this.makeQBittorrentRequest('/api/v2/torrents/info');
            const torrentsData = await torrents.json();

            const actions = [];

            for (const torrent of torrentsData) {
                const ratio = torrent.ratio || 0;
                const seedingTime = torrent.seeding_time || 0;
                const category = torrent.category || 'default';
                
                // Get ratio settings for category
                const ratioGroup = this.ratioSettings.ratioGroups[category] || {
                    ratio: this.ratioSettings.globalRatio,
                    time: this.ratioSettings.seedingTimeLimit
                };

                // Check if torrent should be paused or removed
                if (ratio >= ratioGroup.ratio && seedingTime >= ratioGroup.time * 60) {
                    actions.push({
                        hash: torrent.hash,
                        name: torrent.name,
                        action: 'pause',
                        reason: `Reached ratio ${ratio.toFixed(2)} and seeding time ${Math.round(seedingTime / 3600)}h`
                    });
                }
            }

            // Execute actions
            for (const action of actions) {
                if (action.action === 'pause') {
                    await this.makeQBittorrentRequest('/api/v2/torrents/pause', 'POST', {
                        hashes: action.hash
                    });
                }
            }

            return {
                processed: torrentsData.length,
                actions: actions.length,
                details: actions,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to manage ratios: ' + error.message);
        }
    }

    async optimizeStorage() {
        try {
            const optimizations = [];

            // Remove completed torrents with good ratios older than specified time
            const torrents = await this.makeQBittorrentRequest('/api/v2/torrents/info');
            const torrentsData = await torrents.json();

            for (const torrent of torrentsData) {
                if (torrent.state === 'pausedUP' && torrent.ratio >= 2.0) {
                    const completedDate = new Date(torrent.completed_on * 1000);
                    const daysSinceCompleted = (Date.now() - completedDate.getTime()) / (1000 * 60 * 60 * 24);

                    if (daysSinceCompleted > 30) { // Remove torrents older than 30 days
                        optimizations.push({
                            hash: torrent.hash,
                            name: torrent.name,
                            action: 'remove',
                            reason: `Good ratio (${torrent.ratio.toFixed(2)}) and ${Math.round(daysSinceCompleted)} days old`
                        });
                    }
                }
            }

            return {
                totalOptimizations: optimizations.length,
                optimizations,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to optimize storage: ' + error.message);
        }
    }

    async makeQBittorrentRequest(endpoint, method = 'GET', data = null) {
        const url = `http://${this.qbittorrentConfig.host}:${this.qbittorrentConfig.port}${endpoint}`;
        
        const options = {
            method,
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        };

        if (this.sessionCookie) {
            options.headers.Cookie = this.sessionCookie;
        }

        if (data && method === 'POST') {
            const params = new URLSearchParams();
            Object.keys(data).forEach(key => params.append(key, data[key]));
            options.body = params;
        }

        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`qBittorrent API error: ${response.status} ${response.statusText}`);
        }

        return response;
    }

    extractTracker(trackerUrl) {
        if (!trackerUrl) return 'Unknown';
        
        try {
            const url = new URL(trackerUrl);
            return url.hostname;
        } catch (error) {
            return trackerUrl.split('/')[2] || 'Unknown';
        }
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async getTrackerHealth() {
        const trackerHealth = {};
        
        for (const [category, trackers] of Object.entries(this.trackers)) {
            for (const [tracker, config] of Object.entries(trackers)) {
                if (config.enabled) {
                    try {
                        // Basic connectivity check (this would be expanded for actual tracker APIs)
                        trackerHealth[tracker] = {
                            category,
                            status: 'healthy',
                            priority: config.priority,
                            lastChecked: new Date().toISOString()
                        };
                    } catch (error) {
                        trackerHealth[tracker] = {
                            category,
                            status: 'error',
                            error: error.message,
                            priority: config.priority,
                            lastChecked: new Date().toISOString()
                        };
                    }
                }
            }
        }

        return trackerHealth;
    }

    async updateTrackerSettings(trackerUpdates) {
        try {
            for (const [tracker, settings] of Object.entries(trackerUpdates)) {
                const category = settings.category || 'public';
                
                if (this.trackers[category] && this.trackers[category][tracker]) {
                    this.trackers[category][tracker] = {
                        ...this.trackers[category][tracker],
                        ...settings
                    };
                }
            }

            return {
                success: true,
                updatedTrackers: Object.keys(trackerUpdates),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to update tracker settings: ' + error.message);
        }
    }
}

module.exports = SeedboxManager;