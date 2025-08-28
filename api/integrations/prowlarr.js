/**
 * Prowlarr Integration Wrapper
 * Simplified interface for Prowlarr indexer management
 */

const ProwlarrIntegration = require('./ProwlarrIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a Prowlarr integration instance
 * @param {Object} config - Configuration options
 * @returns {ProwlarrIntegration} Configured Prowlarr integration instance
 */
function createProwlarrIntegration(config = {}) {
    return new ProwlarrIntegration(config);
}

/**
 * Default configuration for Prowlarr integration
 */
const defaultConfig = {
    baseURL: process.env.PROWLARR_URL || 'http://localhost:9696',
    apiKey: process.env.PROWLARR_API_KEY,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true,
    version: 'v1'
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<ProwlarrIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const prowlarr = new ProwlarrIntegration(config);
    
    try {
        // Test connection
        const connectionResult = await prowlarr.testConnection();
        if (connectionResult.success) {
            console.log('✅ Prowlarr integration setup successfully');
        } else {
            console.warn('⚠️ Prowlarr connection test failed:', connectionResult.error);
        }
        
        return prowlarr;
    } catch (error) {
        console.error('❌ Prowlarr quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common Prowlarr operations
 */
const utils = {
    /**
     * Format indexer info for display
     * @param {Object} indexer - Indexer object from Prowlarr
     * @returns {Object} Formatted indexer info
     */
    formatIndexerInfo(indexer) {
        return {
            id: indexer.id,
            name: indexer.name,
            definitionName: indexer.definitionName,
            description: indexer.description,
            language: indexer.language,
            protocol: indexer.protocol,
            privacy: indexer.privacy,
            supportsRss: indexer.supportsRss,
            supportsSearch: indexer.supportsSearch,
            supportsRedirect: indexer.supportsRedirect,
            enable: indexer.enable,
            redirect: indexer.redirect,
            priority: indexer.priority,
            downloadClientId: indexer.downloadClientId,
            configContract: indexer.configContract,
            implementationName: indexer.implementationName,
            implementation: indexer.implementation,
            tags: indexer.tags || [],
            fields: indexer.fields || []
        };
    },

    /**
     * Format search result for display
     * @param {Object} result - Search result from Prowlarr
     * @returns {Object} Formatted search result
     */
    formatSearchResult(result) {
        return {
            guid: result.guid,
            title: result.title,
            size: result.size,
            indexer: result.indexer,
            indexerId: result.indexerId,
            publishDate: result.publishDate,
            commentUrl: result.commentUrl,
            downloadUrl: result.downloadUrl,
            infoUrl: result.infoUrl,
            posterUrl: result.posterUrl,
            categories: result.categories || [],
            seeders: result.seeders,
            leechers: result.leechers,
            language: result.language,
            year: result.year,
            author: result.author,
            bookTitle: result.bookTitle,
            imdb: result.imdb,
            tmdb: result.tmdb,
            tvdb: result.tvdb,
            tvmaze: result.tvmaze,
            doubanId: result.doubanId,
            magnetUrl: result.magnetUrl,
            infoHash: result.infoHash
        };
    },

    /**
     * Parse Prowlarr webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        return {
            event: payload.eventType,
            timestamp: new Date(payload.dateTime || Date.now()),
            indexer: payload.indexer ? {
                id: payload.indexer.id,
                name: payload.indexer.name,
                type: payload.indexer.type
            } : null,
            release: payload.release ? {
                title: payload.release.title,
                size: payload.release.size,
                indexer: payload.release.indexer,
                categories: payload.release.categories,
                publishDate: payload.release.publishDate,
                downloadUrl: payload.release.downloadUrl,
                infoUrl: payload.release.infoUrl
            } : null,
            query: payload.query ? {
                searchType: payload.query.searchType,
                searchTerm: payload.query.searchTerm,
                season: payload.query.season,
                episode: payload.query.episode,
                year: payload.query.year,
                categories: payload.query.categories
            } : null
        };
    },

    /**
     * Format file size
     * @param {number} bytes - Size in bytes
     * @returns {string} Formatted size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Get category name from ID
     * @param {number} categoryId - Category ID
     * @returns {string} Category name
     */
    getCategoryName(categoryId) {
        const categories = {
            1000: 'Console',
            2000: 'Movies',
            3000: 'Audio',
            4000: 'PC',
            5000: 'TV',
            6000: 'XXX',
            7000: 'Books',
            8000: 'Other'
        };
        
        // Find the base category
        const baseCategory = Math.floor(categoryId / 1000) * 1000;
        return categories[baseCategory] || 'Unknown';
    },

    /**
     * Calculate health score for search results
     * @param {Array} results - Array of search results
     * @returns {Object} Health analysis
     */
    analyzeSearchHealth(results) {
        if (!results || results.length === 0) {
            return {
                score: 0,
                issues: ['No results found'],
                recommendations: ['Check indexer availability', 'Verify search terms']
            };
        }
        
        const totalResults = results.length;
        const withSeeders = results.filter(r => r.seeders && r.seeders > 0).length;
        const highQuality = results.filter(r => r.size > 1024 * 1024 * 500).length; // > 500MB
        const recent = results.filter(r => {
            const publishDate = new Date(r.publishDate);
            const daysDiff = (Date.now() - publishDate.getTime()) / (1000 * 60 * 60 * 24);
            return daysDiff < 30;
        }).length;
        
        let score = 0;
        const issues = [];
        const recommendations = [];
        
        // Score based on seeders
        const seederRatio = withSeeders / totalResults;
        if (seederRatio > 0.8) score += 30;
        else if (seederRatio > 0.5) score += 20;
        else if (seederRatio > 0.2) score += 10;
        else issues.push('Low number of results with seeders');
        
        // Score based on file sizes (quality indicator)
        const qualityRatio = highQuality / totalResults;
        if (qualityRatio > 0.5) score += 30;
        else if (qualityRatio > 0.3) score += 20;
        else if (qualityRatio > 0.1) score += 10;
        else issues.push('Most results appear to be low quality');
        
        // Score based on recency
        const recentRatio = recent / totalResults;
        if (recentRatio > 0.7) score += 30;
        else if (recentRatio > 0.4) score += 20;
        else if (recentRatio > 0.2) score += 10;
        else issues.push('Most results are old');
        
        // Base score for having results
        score += 10;
        
        if (score < 50) {
            recommendations.push('Enable more indexers');
            recommendations.push('Check indexer status');
        }
        if (issues.includes('Low number of results with seeders')) {
            recommendations.push('Try different search terms');
        }
        
        return {
            score: Math.min(100, score),
            totalResults,
            withSeeders,
            highQuality,
            recent,
            issues,
            recommendations
        };
    }
};

/**
 * Health check function
 * @param {Object} config - Prowlarr configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const prowlarr = createProwlarrIntegration(config);
        const result = await prowlarr.testConnection();
        
        return {
            service: 'prowlarr',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            version: result.version,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'prowlarr',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

/**
 * Indexer management utilities
 */
const indexerManager = {
    /**
     * Get indexer statistics
     * @param {ProwlarrIntegration} prowlarr - Prowlarr integration instance
     * @returns {Promise<Object>} Indexer statistics
     */
    async getIndexerStats(prowlarr) {
        const indexers = await prowlarr.getIndexers();
        const totalIndexers = indexers.length;
        const enabledIndexers = indexers.filter(i => i.enable).length;
        const publicIndexers = indexers.filter(i => i.privacy === 'public').length;
        const privateIndexers = indexers.filter(i => i.privacy === 'private').length;
        const semiPrivateIndexers = indexers.filter(i => i.privacy === 'semiPrivate').length;
        
        // Group by protocol
        const protocolBreakdown = indexers.reduce((acc, indexer) => {
            acc[indexer.protocol] = (acc[indexer.protocol] || 0) + 1;
            return acc;
        }, {});
        
        // Group by language
        const languageBreakdown = indexers.reduce((acc, indexer) => {
            const lang = indexer.language || 'unknown';
            acc[lang] = (acc[lang] || 0) + 1;
            return acc;
        }, {});
        
        return {
            totalIndexers,
            enabledIndexers,
            disabledIndexers: totalIndexers - enabledIndexers,
            publicIndexers,
            privateIndexers,
            semiPrivateIndexers,
            protocolBreakdown,
            languageBreakdown,
            enabledPercentage: totalIndexers > 0 ? Math.round((enabledIndexers / totalIndexers) * 100) : 0
        };
    },

    /**
     * Test all indexers
     * @param {ProwlarrIntegration} prowlarr - Prowlarr integration instance
     * @param {Object} options - Test options
     * @returns {Promise<Array>} Test results
     */
    async testAllIndexers(prowlarr, options = {}) {
        const indexers = await prowlarr.getIndexers();
        const indexersToTest = options.enabledOnly ? 
            indexers.filter(i => i.enable) : indexers;
        
        const testPromises = indexersToTest.map(async (indexer) => {
            try {
                const result = await prowlarr.testIndexer(indexer.id);
                return {
                    id: indexer.id,
                    name: indexer.name,
                    success: true,
                    response: result
                };
            } catch (error) {
                return {
                    id: indexer.id,
                    name: indexer.name,
                    success: false,
                    error: error.message
                };
            }
        });
        
        const results = await Promise.allSettled(testPromises);
        return results.map(r => r.value || r.reason);
    },

    /**
     * Bulk enable/disable indexers
     * @param {ProwlarrIntegration} prowlarr - Prowlarr integration instance
     * @param {Array} indexerIds - Array of indexer IDs
     * @param {boolean} enabled - Enable status
     * @returns {Promise<Array>} Update results
     */
    async bulkToggleIndexers(prowlarr, indexerIds, enabled = true) {
        const updatePromises = indexerIds.map(async (id) => {
            try {
                const indexer = await prowlarr.getIndexer(id);
                const updated = await prowlarr.updateIndexer({
                    ...indexer,
                    enable: enabled
                });
                return {
                    id,
                    name: indexer.name,
                    success: true,
                    enabled: updated.enable
                };
            } catch (error) {
                return {
                    id,
                    success: false,
                    error: error.message
                };
            }
        });
        
        return Promise.allSettled(updatePromises);
    },

    /**
     * Search across multiple indexers and aggregate results
     * @param {ProwlarrIntegration} prowlarr - Prowlarr integration instance
     * @param {string} query - Search query
     * @param {Object} options - Search options
     * @returns {Promise<Object>} Aggregated search results
     */
    async searchMultipleIndexers(prowlarr, query, options = {}) {
        try {
            const results = await prowlarr.search(query, options);
            
            // Group results by indexer
            const resultsByIndexer = results.reduce((acc, result) => {
                if (!acc[result.indexer]) {
                    acc[result.indexer] = [];
                }
                acc[result.indexer].push(utils.formatSearchResult(result));
                return acc;
            }, {});
            
            // Calculate health metrics
            const healthAnalysis = utils.analyzeSearchHealth(results);
            
            // Sort results by seeders and size
            const sortedResults = results
                .map(r => utils.formatSearchResult(r))
                .sort((a, b) => {
                    if (a.seeders !== b.seeders) {
                        return (b.seeders || 0) - (a.seeders || 0);
                    }
                    return (b.size || 0) - (a.size || 0);
                });
            
            return {
                query,
                totalResults: results.length,
                indexerCount: Object.keys(resultsByIndexer).length,
                results: sortedResults,
                resultsByIndexer,
                healthAnalysis,
                searchTime: Date.now()
            };
        } catch (error) {
            return {
                query,
                error: error.message,
                totalResults: 0,
                results: []
            };
        }
    }
};

module.exports = {
    ProwlarrIntegration,
    createProwlarrIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    indexerManager,
    
    // Aliases for convenience
    create: createProwlarrIntegration,
    setup: quickSetup,
    Integration: ProwlarrIntegration
};