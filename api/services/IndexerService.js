const logger = require('../../middleware/logger.js');
/**
 * IndexerService - Prowlarr master indexer with 500+ trackers
 * Provides comprehensive indexer management and search capabilities
 */

const axios = require('axios');
const EventEmitter = require('events');

class IndexerService extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            prowlarrUrl: config.prowlarrUrl || process.env.PROWLARR_URL || 'http://prowlarr:9696',
            prowlarrApiKey: config.prowlarrApiKey || process.env.PROWLARR_API_KEY,
            enableAutoConfig: config.enableAutoConfig !== false,
            syncInterval: config.syncInterval || 5 * 60 * 1000, // 5 minutes
            maxRetries: config.maxRetries || 3,
            retryDelay: config.retryDelay || 5000,
            searchTimeout: config.searchTimeout || 30000,
            enableTorrent: config.enableTorrent !== false,
            enableUsenet: config.enableUsenet !== false,
            minSeeders: config.minSeeders || 1,
            maxResults: config.maxResults || 100,
            ...config
        };

        this.indexers = new Map();
        this.searchHistory = [];
        this.syncApps = new Map();
        this.statistics = {
            totalIndexers: 0,
            activeIndexers: 0,
            torrentIndexers: 0,
            usenetIndexers: 0,
            totalSearches: 0,
            successfulSearches: 0,
            failedSearches: 0
        };
        this.isInitialized = false;
        this.syncTimer = null;
        
        this.indexerCategories = {
            1000: 'Console',
            2000: 'Movies',
            3000: 'Audio',
            4000: 'PC',
            5000: 'TV',
            6000: 'XXX',
            7000: 'Books',
            8000: 'Other'
        };

        this.searchTypes = {
            SEARCH: 'search',
            TV_SEARCH: 'tvsearch',
            MOVIE_SEARCH: 'movie',
            MUSIC_SEARCH: 'music',
            BOOK_SEARCH: 'book'
        };

        this.indexerTypes = {
            TORRENT: 'torrent',
            USENET: 'usenet',
            AGGREGATE: 'aggregate'
        };
    }

    /**
     * Initialize Indexer service
     */
    async initialize() {
        try {
            logger.info('🔍 Initializing IndexerService...');
            
            // Test Prowlarr connection
            await this.testProwlarrConnection();
            
            // Load indexers from Prowlarr
            await this.loadIndexers();
            
            // Configure sync apps
            await this.configureSyncApps();
            
            // Start automatic sync
            this.startSyncTimer();
            
            this.isInitialized = true;
            this.emit('initialized');
            logger.info('✅ IndexerService initialized successfully');
            
            return { success: true, message: 'IndexerService initialized' };
        } catch (error) {
            logger.error('❌ IndexerService initialization failed:', error);
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Test Prowlarr connection
     */
    async testProwlarrConnection() {
        try {
            const response = await axios.get(`${this.config.prowlarrUrl}/api/v1/system/status`, {
                headers: this.getApiHeaders(),
                timeout: 10000
            });
            
            if (!response.data || !response.data.version) {
                throw new Error('Invalid Prowlarr response');
            }
            
            logger.info(`✅ Prowlarr connection verified (v${response.data.version})`);
        } catch (error) {
            logger.error('❌ Prowlarr connection failed:', error.message);
            throw error;
        }
    }

    /**
     * Get API headers for Prowlarr requests
     */
    getApiHeaders() {
        const headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'MediaServer-IndexerService/1.0'
        };
        
        if (this.config.prowlarrApiKey) {
            headers['X-Api-Key'] = this.config.prowlarrApiKey;
        }
        
        return headers;
    }

    /**
     * Load indexers from Prowlarr
     */
    async loadIndexers() {
        try {
            const response = await axios.get(`${this.config.prowlarrUrl}/api/v1/indexer`, {
                headers: this.getApiHeaders(),
                timeout: 15000
            });
            
            const indexers = response.data || [];
            
            // Clear existing indexers
            this.indexers.clear();
            
            // Process and store indexers
            indexers.forEach(indexer => {
                const processedIndexer = this.processIndexer(indexer);
                this.indexers.set(indexer.id, processedIndexer);
            });
            
            // Update statistics
            this.updateStatistics();
            
            logger.info(`✅ Loaded ${this.indexers.size} indexers from Prowlarr`);
            this.emit('indexersLoaded', { count: this.indexers.size });
        } catch (error) {
            logger.error('❌ Failed to load indexers:', error);
            throw error;
        }
    }

    /**
     * Process raw indexer data from Prowlarr
     */
    processIndexer(rawIndexer) {
        return {
            id: rawIndexer.id,
            name: rawIndexer.name,
            description: rawIndexer.description || '',
            language: rawIndexer.language || 'en-US',
            encoding: rawIndexer.encoding || 'UTF-8',
            enable: rawIndexer.enable || false,
            redirect: rawIndexer.redirect || false,
            supportsRss: rawIndexer.supportsRss || false,
            supportsSearch: rawIndexer.supportsSearch || false,
            protocol: rawIndexer.protocol || 'torrent',
            privacy: rawIndexer.privacy || 'public',
            priority: rawIndexer.priority || 25,
            added: new Date(rawIndexer.added || Date.now()),
            capabilities: rawIndexer.capabilities || {},
            categories: rawIndexer.categories || [],
            tags: rawIndexer.tags || [],
            fields: rawIndexer.fields || [],
            status: this.getIndexerStatus(rawIndexer),
            lastTest: null,
            responseTime: 0,
            successRate: 0,
            errorCount: 0
        };
    }

    /**
     * Get indexer status
     */
    getIndexerStatus(indexer) {
        if (!indexer.enable) return 'disabled';
        if (indexer.redirect) return 'redirect';
        return 'active';
    }

    /**
     * Configure sync applications (Sonarr, Radarr, etc.)
     */
    async configureSyncApps() {
        try {
            if (!this.config.enableAutoConfig) {
                logger.info('⚠️ Auto-configuration disabled, skipping sync apps setup');
                return;
            }
            
            // Get configured applications from Prowlarr
            const response = await axios.get(`${this.config.prowlarrUrl}/api/v1/applications`, {
                headers: this.getApiHeaders(),
                timeout: 10000
            });
            
            const apps = response.data || [];
            
            // Store sync apps
            apps.forEach(app => {
                this.syncApps.set(app.id, {
                    id: app.id,
                    name: app.name,
                    implementation: app.implementation,
                    baseUrl: app.fields?.find(f => f.name === 'baseUrl')?.value || '',
                    apiKey: app.fields?.find(f => f.name === 'apiKey')?.value || '',
                    syncCategories: app.syncCategories || [],
                    tags: app.tags || [],
                    enable: app.enable || false
                });
            });
            
            logger.info(`✅ Configured ${this.syncApps.size} sync applications`);
        } catch (error) {
            logger.error('❌ Sync apps configuration failed:', error.message);
        }
    }

    /**
     * Search across all active indexers
     */
    async search(query, options = {}) {
        try {
            const searchId = `search_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            
            const searchOptions = {
                type: options.type || this.searchTypes.SEARCH,
                categories: options.categories || [],
                limit: options.limit || this.config.maxResults,
                offset: options.offset || 0,
                minSeeders: options.minSeeders || this.config.minSeeders,
                indexerIds: options.indexerIds || [], // Empty = all active indexers
                ...options
            };
            
            logger.info(`🔍 Starting search: "${query}" (${searchId})`);
            
            // Build search URL
            const searchUrl = this.buildSearchUrl(query, searchOptions);
            
            // Execute search
            const startTime = Date.now();
            const response = await axios.get(searchUrl, {
                headers: this.getApiHeaders(),
                timeout: this.config.searchTimeout
            });
            
            const results = response.data || [];
            const duration = Date.now() - startTime;
            
            // Process and filter results
            const processedResults = this.processSearchResults(results, searchOptions);
            
            // Log search
            const searchLog = {
                id: searchId,
                query,
                options: searchOptions,
                resultsCount: processedResults.length,
                duration,
                timestamp: new Date(),
                success: true
            };
            
            this.logSearch(searchLog);
            this.updateSearchStatistics(true);
            
            this.emit('searchCompleted', searchLog);
            
            logger.info(`✅ Search completed: ${processedResults.length} results in ${duration}ms`);
            
            return {
                success: true,
                searchId,
                query,
                results: processedResults,
                totalResults: processedResults.length,
                duration,
                indexersSearched: this.getActiveIndexerCount(searchOptions.indexerIds)
            };
        } catch (error) {
            logger.error('❌ Search failed:', error);
            
            this.updateSearchStatistics(false);
            
            const searchLog = {
                id: `search_${Date.now()}`,
                query,
                options,
                resultsCount: 0,
                duration: 0,
                timestamp: new Date(),
                success: false,
                error: error.message
            };
            
            this.logSearch(searchLog);
            
            throw error;
        }
    }

    /**
     * Build search URL for Prowlarr API
     */
    buildSearchUrl(query, options) {
        const baseUrl = `${this.config.prowlarrUrl}/api/v1/search`;
        const params = new URLSearchParams();
        
        params.append('query', query);
        params.append('type', options.type);
        
        if (options.categories && options.categories.length > 0) {
            params.append('categories', options.categories.join(','));
        }
        
        if (options.indexerIds && options.indexerIds.length > 0) {
            params.append('indexerIds', options.indexerIds.join(','));
        }
        
        if (options.limit) {
            params.append('limit', options.limit.toString());
        }
        
        if (options.offset) {
            params.append('offset', options.offset.toString());
        }
        
        return `${baseUrl}?${params.toString()}`;
    }

    /**
     * Process and filter search results
     */
    processSearchResults(results, options) {
        return results
            .filter(result => this.filterResult(result, options))
            .map(result => this.processResult(result))
            .sort((a, b) => {
                // Sort by seeders (descending) then by size (descending)
                if (b.seeders !== a.seeders) {
                    return b.seeders - a.seeders;
                }
                return b.size - a.size;
            })
            .slice(0, options.limit);
    }

    /**
     * Filter individual search result
     */
    filterResult(result, options) {
        // Filter by minimum seeders for torrents
        if (result.protocol === 'torrent') {
            const seeders = result.seeders || 0;
            if (seeders < options.minSeeders) {
                return false;
            }
        }
        
        // Filter by file size (optional)
        if (options.minSize && result.size < options.minSize) {
            return false;
        }
        
        if (options.maxSize && result.size > options.maxSize) {
            return false;
        }
        
        return true;
    }

    /**
     * Process individual search result
     */
    processResult(result) {
        return {
            guid: result.guid,
            title: result.title,
            size: result.size || 0,
            link: result.link || result.downloadUrl,
            magnetUrl: result.magnetUrl,
            indexer: result.indexer,
            indexerId: result.indexerId,
            category: result.category,
            categoryDesc: this.indexerCategories[result.category] || 'Other',
            protocol: result.protocol || 'torrent',
            seeders: result.seeders || 0,
            leechers: result.leechers || 0,
            downloadVolumeFactor: result.downloadVolumeFactor || 1,
            uploadVolumeFactor: result.uploadVolumeFactor || 1,
            publishDate: new Date(result.publishDate || Date.now()),
            age: result.age || 0,
            ageHours: result.ageHours || 0,
            ageDays: result.ageDays || 0,
            imdbId: result.imdbId,
            tmdbId: result.tmdbId,
            tvdbId: result.tvdbId,
            files: result.files || [],
            languages: result.languages || [],
            resolution: result.resolution,
            videoCodec: result.videoCodec,
            audioCodec: result.audioCodec,
            audioChannels: result.audioChannels,
            quality: result.quality,
            qualityWeight: result.qualityWeight || 0,
            preferredWords: result.preferredWords || 0,
            special: result.special || false,
            score: this.calculateResultScore(result)
        };
    }

    /**
     * Calculate result score for ranking
     */
    calculateResultScore(result) {
        let score = 0;
        
        // Seeders weight (0-50 points)
        if (result.seeders) {
            score += Math.min(result.seeders, 50);
        }
        
        // Size weight (prefer reasonable sizes)
        if (result.size) {
            const sizeGB = result.size / (1024 * 1024 * 1024);
            if (sizeGB >= 0.5 && sizeGB <= 50) {
                score += 20;
            } else if (sizeGB > 50) {
                score += Math.max(0, 20 - (sizeGB - 50));
            }
        }
        
        // Age weight (newer is better)
        if (result.ageHours !== undefined) {
            if (result.ageHours <= 24) {
                score += 15;
            } else if (result.ageHours <= 168) { // 1 week
                score += 10;
            } else if (result.ageHours <= 720) { // 1 month
                score += 5;
            }
        }
        
        // Quality weight
        if (result.qualityWeight) {
            score += Math.min(result.qualityWeight, 15);
        }
        
        return Math.round(score);
    }

    /**
     * Test specific indexer
     */
    async testIndexer(indexerId) {
        try {
            const indexer = this.indexers.get(indexerId);
            if (!indexer) {
                throw new Error('Indexer not found');
            }
            
            const startTime = Date.now();
            
            const response = await axios.post(
                `${this.config.prowlarrUrl}/api/v1/indexer/test`,
                { id: indexerId },
                {
                    headers: this.getApiHeaders(),
                    timeout: 15000
                }
            );
            
            const duration = Date.now() - startTime;
            const success = response.data && !response.data.error;
            
            // Update indexer status
            indexer.lastTest = new Date();
            indexer.responseTime = duration;
            
            if (success) {
                indexer.errorCount = 0;
                indexer.successRate = Math.min(indexer.successRate + 10, 100);
            } else {
                indexer.errorCount++;
                indexer.successRate = Math.max(indexer.successRate - 5, 0);
            }
            
            this.emit('indexerTested', { indexer, success, duration });
            
            return {
                success,
                indexer: indexer.name,
                duration,
                error: success ? null : response.data?.error
            };
        } catch (error) {
            logger.error(`❌ Indexer test failed for ${indexerId}:`, error.message);
            
            const indexer = this.indexers.get(indexerId);
            if (indexer) {
                indexer.errorCount++;
                indexer.successRate = Math.max(indexer.successRate - 10, 0);
            }
            
            throw error;
        }
    }

    /**
     * Test all active indexers
     */
    async testAllIndexers() {
        try {
            logger.info('🧪 Testing all active indexers...');
            
            const activeIndexers = Array.from(this.indexers.values())
                .filter(indexer => indexer.enable && indexer.status === 'active');
            
            const results = [];
            
            // Test indexers in batches to avoid overwhelming
            const batchSize = 5;
            for (let i = 0; i < activeIndexers.length; i += batchSize) {
                const batch = activeIndexers.slice(i, i + batchSize);
                
                const batchPromises = batch.map(async (indexer) => {
                    try {
                        const result = await this.testIndexer(indexer.id);
                        return { indexerId: indexer.id, ...result };
                    } catch (error) {
                        return {
                            indexerId: indexer.id,
                            success: false,
                            error: error.message
                        };
                    }
                });
                
                const batchResults = await Promise.allSettled(batchPromises);
                results.push(...batchResults.map(r => r.value || r.reason));
                
                // Wait between batches
                if (i + batchSize < activeIndexers.length) {
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
            }
            
            const successful = results.filter(r => r.success).length;
            const failed = results.filter(r => !r.success).length;
            
            logger.info(`✅ Indexer testing completed: ${successful} successful, ${failed} failed`);
            
            return {
                success: true,
                results,
                summary: {
                    total: results.length,
                    successful,
                    failed,
                    successRate: Math.round((successful / results.length) * 100)
                }
            };
        } catch (error) {
            logger.error('❌ Indexer testing failed:', error);
            throw error;
        }
    }

    /**
     * Get active indexer count
     */
    getActiveIndexerCount(indexerIds = []) {
        if (indexerIds.length > 0) {
            return indexerIds.length;
        }
        
        return Array.from(this.indexers.values())
            .filter(indexer => indexer.enable && indexer.status === 'active').length;
    }

    /**
     * Log search activity
     */
    logSearch(searchLog) {
        this.searchHistory.push(searchLog);
        
        // Keep only last 1000 searches
        if (this.searchHistory.length > 1000) {
            this.searchHistory = this.searchHistory.slice(-1000);
        }
    }

    /**
     * Update search statistics
     */
    updateSearchStatistics(success) {
        this.statistics.totalSearches++;
        
        if (success) {
            this.statistics.successfulSearches++;
        } else {
            this.statistics.failedSearches++;
        }
    }

    /**
     * Update indexer statistics
     */
    updateStatistics() {
        const indexers = Array.from(this.indexers.values());
        
        this.statistics.totalIndexers = indexers.length;
        this.statistics.activeIndexers = indexers.filter(i => i.enable && i.status === 'active').length;
        this.statistics.torrentIndexers = indexers.filter(i => i.protocol === 'torrent').length;
        this.statistics.usenetIndexers = indexers.filter(i => i.protocol === 'usenet').length;
    }

    /**
     * Start sync timer
     */
    startSyncTimer() {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
        }
        
        this.syncTimer = setInterval(async () => {
            try {
                await this.loadIndexers();
                logger.info('🔄 Indexers synchronized with Prowlarr');
            } catch (error) {
                logger.warn('⚠️ Indexer sync failed:', error.message);
            }
        }, this.config.syncInterval);
        
        logger.info('✅ Indexer sync timer started');
    }

    /**
     * Get indexer statistics
     */
    getIndexerStats() {
        const indexers = Array.from(this.indexers.values());
        
        const statsByProtocol = {
            torrent: { total: 0, active: 0, disabled: 0 },
            usenet: { total: 0, active: 0, disabled: 0 }
        };
        
        const statsByPrivacy = {
            public: 0,
            semi_private: 0,
            private: 0
        };
        
        indexers.forEach(indexer => {
            // Protocol stats
            const protocol = indexer.protocol || 'torrent';
            if (statsByProtocol[protocol]) {
                statsByProtocol[protocol].total++;
                if (indexer.enable && indexer.status === 'active') {
                    statsByProtocol[protocol].active++;
                } else {
                    statsByProtocol[protocol].disabled++;
                }
            }
            
            // Privacy stats
            const privacy = indexer.privacy || 'public';
            statsByPrivacy[privacy] = (statsByPrivacy[privacy] || 0) + 1;
        });
        
        return {
            total: this.statistics.totalIndexers,
            active: this.statistics.activeIndexers,
            byProtocol: statsByProtocol,
            byPrivacy: statsByPrivacy,
            searchStats: {
                totalSearches: this.statistics.totalSearches,
                successfulSearches: this.statistics.successfulSearches,
                failedSearches: this.statistics.failedSearches,
                successRate: this.statistics.totalSearches > 0 ? 
                    Math.round((this.statistics.successfulSearches / this.statistics.totalSearches) * 100) : 0
            }
        };
    }

    /**
     * Get service status
     */
    getStatus() {
        return {
            initialized: this.isInitialized,
            prowlarrConnected: this.isInitialized,
            totalIndexers: this.statistics.totalIndexers,
            activeIndexers: this.statistics.activeIndexers,
            torrentIndexers: this.statistics.torrentIndexers,
            usenetIndexers: this.statistics.usenetIndexers,
            syncApps: this.syncApps.size,
            searchHistory: this.searchHistory.length,
            autoSyncEnabled: !!this.syncTimer,
            statistics: this.statistics,
            config: {
                prowlarrUrl: this.config.prowlarrUrl,
                enableTorrent: this.config.enableTorrent,
                enableUsenet: this.config.enableUsenet,
                minSeeders: this.config.minSeeders,
                maxResults: this.config.maxResults,
                syncInterval: this.config.syncInterval
            },
            lastUpdate: new Date()
        };
    }

    /**
     * Get recent search history
     */
    getSearchHistory(limit = 50) {
        return this.searchHistory
            .slice(-limit)
            .reverse()
            .map(search => ({
                id: search.id,
                query: search.query,
                type: search.options.type,
                resultsCount: search.resultsCount,
                duration: search.duration,
                timestamp: search.timestamp,
                success: search.success,
                error: search.error
            }));
    }

    /**
     * Get indexer list
     */
    getIndexers(includeDisabled = false) {
        return Array.from(this.indexers.values())
            .filter(indexer => includeDisabled || indexer.enable)
            .map(indexer => ({
                id: indexer.id,
                name: indexer.name,
                protocol: indexer.protocol,
                privacy: indexer.privacy,
                enable: indexer.enable,
                status: indexer.status,
                categories: indexer.categories.length,
                supportsSearch: indexer.supportsSearch,
                lastTest: indexer.lastTest,
                responseTime: indexer.responseTime,
                successRate: indexer.successRate
            }));
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger.info('🧹 Cleaning up IndexerService...');
            
            if (this.syncTimer) {
                clearInterval(this.syncTimer);
                this.syncTimer = null;
            }
            
            this.indexers.clear();
            this.syncApps.clear();
            this.searchHistory = [];
            this.removeAllListeners();
            
            this.isInitialized = false;
            logger.info('✅ IndexerService cleanup completed');
        } catch (error) {
            logger.error('❌ IndexerService cleanup failed:', error);
        }
    }
}

module.exports = IndexerService;