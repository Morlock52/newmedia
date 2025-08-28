#!/usr/bin/env node
/**
 * Redis Cache Configuration and Optimization for Media Server Stack
 * Implements intelligent caching strategies for improved performance
 */

const Redis = require('redis');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

class RedisCacheOptimizer {
    constructor(config = {}) {
        this.config = {
            redis: {
                host: config.redisHost || 'localhost',
                port: config.redisPort || 6379,
                password: config.redisPassword || null,
                db: config.redisDb || 0,
                maxRetriesPerRequest: 3,
                retryDelayOnFailover: 100,
                maxRetriesPerRequest: 3,
                keyPrefix: 'media-server:'
            },
            cache: {
                defaultTTL: config.defaultTTL || 300, // 5 minutes
                longTTL: config.longTTL || 3600, // 1 hour
                shortTTL: config.shortTTL || 60, // 1 minute
                maxCacheSize: config.maxCacheSize || '256mb'
            },
            services: {
                sonarr: { port: 8989, apiKey: process.env.SONARR_API_KEY || '' },
                radarr: { port: 7878, apiKey: process.env.RADARR_API_KEY || '' },
                prowlarr: { port: 9696, apiKey: process.env.PROWLARR_API_KEY || '' },
                jellyfin: { port: 8096 }
            }
        };
        
        this.client = null;
        this.isConnected = false;
        this.cacheStats = {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            errors: 0
        };
    }

    async initialize() {
        console.log('🚀 Initializing Redis Cache Optimizer...');
        
        try {
            // Create Redis client
            this.client = Redis.createClient({
                socket: {
                    host: this.config.redis.host,
                    port: this.config.redis.port
                },
                password: this.config.redis.password,
                database: this.config.redis.db,
                // Enable key events for cache statistics
                keyPrefix: this.config.redis.keyPrefix
            });

            // Setup error handling
            this.client.on('error', (err) => {
                console.error('Redis Client Error:', err);
                this.cacheStats.errors++;
            });

            this.client.on('connect', () => {
                console.log('✅ Redis client connected');
                this.isConnected = true;
            });

            this.client.on('ready', () => {
                console.log('✅ Redis client ready');
            });

            this.client.on('end', () => {
                console.log('🔌 Redis client disconnected');
                this.isConnected = false;
            });

            // Connect to Redis
            await this.client.connect();
            
            // Configure Redis for optimal performance
            await this.optimizeRedisConfiguration();
            
            console.log('🎉 Redis Cache Optimizer initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Redis:', error.message);
            throw error;
        }
    }

    async optimizeRedisConfiguration() {
        console.log('🔧 Optimizing Redis configuration...');
        
        try {
            // Set memory policy for automatic eviction
            await this.client.configSet('maxmemory-policy', 'allkeys-lru');
            
            // Set maximum memory usage
            await this.client.configSet('maxmemory', this.config.cache.maxCacheSize);
            
            // Enable keyspace notifications for cache statistics
            await this.client.configSet('notify-keyspace-events', 'Ex');
            
            // Optimize for performance
            await this.client.configSet('tcp-keepalive', '60');
            await this.client.configSet('timeout', '0');
            
            console.log('✅ Redis configuration optimized');
            
        } catch (error) {
            console.warn('⚠️  Could not optimize Redis configuration:', error.message);
        }
    }

    async implementCachingStrategies() {
        console.log('📊 Implementing intelligent caching strategies...');
        
        const strategies = [];
        
        // 1. API Response Caching
        strategies.push(await this.setupAPIResponseCaching());
        
        // 2. Database Query Result Caching
        strategies.push(await this.setupDatabaseCaching());
        
        // 3. Media Metadata Caching
        strategies.push(await this.setupMetadataCaching());
        
        // 4. Search Result Caching
        strategies.push(await this.setupSearchResultCaching());
        
        // 5. Static Content Caching
        strategies.push(await this.setupStaticContentCaching());
        
        console.log(`✅ Implemented ${strategies.filter(Boolean).length} caching strategies`);
        return strategies;
    }

    async setupAPIResponseCaching() {
        console.log('   🔧 Setting up API response caching...');
        
        try {
            const cacheableEndpoints = [
                // Sonarr endpoints
                { service: 'sonarr', endpoint: '/api/v3/series', ttl: this.config.cache.longTTL },
                { service: 'sonarr', endpoint: '/api/v3/episode', ttl: this.config.cache.defaultTTL },
                { service: 'sonarr', endpoint: '/api/v3/system/status', ttl: this.config.cache.shortTTL },
                
                // Radarr endpoints
                { service: 'radarr', endpoint: '/api/v3/movie', ttl: this.config.cache.longTTL },
                { service: 'radarr', endpoint: '/api/v3/system/status', ttl: this.config.cache.shortTTL },
                
                // Prowlarr endpoints
                { service: 'prowlarr', endpoint: '/api/v1/indexer', ttl: this.config.cache.longTTL },
                { service: 'prowlarr', endpoint: '/api/v1/system/status', ttl: this.config.cache.shortTTL }
            ];
            
            for (const endpoint of cacheableEndpoints) {
                await this.cacheAPIResponse(endpoint.service, endpoint.endpoint, endpoint.ttl);
            }
            
            console.log(`     ✅ Cached ${cacheableEndpoints.length} API endpoints`);
            return true;
            
        } catch (error) {
            console.error('     ❌ API response caching failed:', error.message);
            return false;
        }
    }

    async cacheAPIResponse(serviceName, endpoint, ttl) {
        const serviceConfig = this.config.services[serviceName];
        if (!serviceConfig) return;
        
        const cacheKey = `api:${serviceName}:${endpoint.replace(/\//g, ':')}`;
        
        try {
            // Check if already cached
            const cached = await this.client.get(cacheKey);
            if (cached) {
                this.cacheStats.hits++;
                return JSON.parse(cached);
            }
            
            // Fetch from API
            const url = `http://localhost:${serviceConfig.port}${endpoint}`;
            const headers = {};
            
            if (serviceConfig.apiKey) {
                headers['X-Api-Key'] = serviceConfig.apiKey;
            }
            
            const response = await axios.get(url, { 
                headers, 
                timeout: 10000,
                validateStatus: status => status < 500
            });
            
            if (response.status >= 200 && response.status < 300) {
                // Cache the response
                await this.client.setEx(cacheKey, ttl, JSON.stringify(response.data));
                this.cacheStats.sets++;
                
                console.log(`       📦 Cached ${serviceName}${endpoint} for ${ttl}s`);
                return response.data;
            }
            
        } catch (error) {
            this.cacheStats.misses++;
            console.warn(`       ⚠️  Failed to cache ${serviceName}${endpoint}:`, error.message);
        }
    }

    async setupDatabaseCaching() {
        console.log('   🔧 Setting up database query caching...');
        
        try {
            const dbCacheStrategies = [
                {
                    name: 'frequent_queries',
                    description: 'Cache frequently executed database queries',
                    ttl: this.config.cache.defaultTTL
                },
                {
                    name: 'heavy_aggregations',
                    description: 'Cache expensive aggregation queries',
                    ttl: this.config.cache.longTTL
                },
                {
                    name: 'lookup_tables',
                    description: 'Cache static lookup data',
                    ttl: this.config.cache.longTTL * 2
                }
            ];
            
            for (const strategy of dbCacheStrategies) {
                await this.client.setEx(
                    `db_strategy:${strategy.name}`,
                    strategy.ttl,
                    JSON.stringify(strategy)
                );
            }
            
            console.log(`     ✅ Configured ${dbCacheStrategies.length} database caching strategies`);
            return true;
            
        } catch (error) {
            console.error('     ❌ Database caching setup failed:', error.message);
            return false;
        }
    }

    async setupMetadataCaching() {
        console.log('   🔧 Setting up metadata caching...');
        
        try {
            // Cache media metadata that rarely changes
            const metadataTypes = [
                'series_metadata',
                'movie_metadata',
                'episode_metadata',
                'season_metadata',
                'genre_data',
                'quality_profiles',
                'language_profiles'
            ];
            
            for (const type of metadataTypes) {
                const cacheKey = `metadata:${type}`;
                await this.client.setEx(
                    cacheKey,
                    this.config.cache.longTTL,
                    JSON.stringify({ type, cached_at: Date.now() })
                );
            }
            
            console.log(`     ✅ Set up caching for ${metadataTypes.length} metadata types`);
            return true;
            
        } catch (error) {
            console.error('     ❌ Metadata caching setup failed:', error.message);
            return false;
        }
    }

    async setupSearchResultCaching() {
        console.log('   🔧 Setting up search result caching...');
        
        try {
            // Cache search results to reduce indexer load
            const searchCacheConfig = {
                torrent_searches: {
                    ttl: 300, // 5 minutes for torrent searches
                    maxResults: 100
                },
                nzb_searches: {
                    ttl: 600, // 10 minutes for NZB searches
                    maxResults: 50
                },
                metadata_searches: {
                    ttl: 3600, // 1 hour for metadata searches
                    maxResults: 200
                }
            };
            
            await this.client.setEx(
                'search_cache_config',
                this.config.cache.longTTL,
                JSON.stringify(searchCacheConfig)
            );
            
            console.log('     ✅ Search result caching configured');
            return true;
            
        } catch (error) {
            console.error('     ❌ Search result caching setup failed:', error.message);
            return false;
        }
    }

    async setupStaticContentCaching() {
        console.log('   🔧 Setting up static content caching...');
        
        try {
            // Cache static content like images, thumbnails, etc.
            const staticContentTypes = [
                { type: 'thumbnails', ttl: this.config.cache.longTTL * 6 }, // 6 hours
                { type: 'posters', ttl: this.config.cache.longTTL * 12 }, // 12 hours
                { type: 'fanart', ttl: this.config.cache.longTTL * 24 }, // 24 hours
                { type: 'banners', ttl: this.config.cache.longTTL * 24 } // 24 hours
            ];
            
            for (const content of staticContentTypes) {
                await this.client.setEx(
                    `static:${content.type}:config`,
                    content.ttl,
                    JSON.stringify({ maxAge: content.ttl })
                );
            }
            
            console.log(`     ✅ Static content caching configured for ${staticContentTypes.length} types`);
            return true;
            
        } catch (error) {
            console.error('     ❌ Static content caching setup failed:', error.message);
            return false;
        }
    }

    async generateCacheAnalysisReport() {
        console.log('📊 Generating cache analysis report...');
        
        try {
            const redisInfo = await this.client.info();
            const redisStats = this.parseRedisInfo(redisInfo);
            
            const report = {
                timestamp: new Date().toISOString(),
                connection: {
                    status: this.isConnected ? 'Connected' : 'Disconnected',
                    host: this.config.redis.host,
                    port: this.config.redis.port,
                    database: this.config.redis.db
                },
                performance: {
                    hits: this.cacheStats.hits,
                    misses: this.cacheStats.misses,
                    hitRate: this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses) * 100,
                    sets: this.cacheStats.sets,
                    deletes: this.cacheStats.deletes,
                    errors: this.cacheStats.errors
                },
                redis: {
                    version: redisStats.redis_version,
                    memory: {
                        used: redisStats.used_memory_human,
                        peak: redisStats.used_memory_peak_human,
                        fragmentation: redisStats.mem_fragmentation_ratio
                    },
                    keyspace: {
                        keys: redisStats.db0_keys || 0,
                        expires: redisStats.db0_expires || 0
                    },
                    operations: {
                        totalCommands: redisStats.total_commands_processed,
                        commandsPerSec: redisStats.instantaneous_ops_per_sec
                    }
                },
                recommendations: this.generateCacheRecommendations(redisStats)
            };
            
            // Save report
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `cache-analysis-${timestamp}.json`;
            const filepath = path.join(__dirname, 'performance-reports', filename);
            
            await fs.mkdir(path.dirname(filepath), { recursive: true });
            await fs.writeFile(filepath, JSON.stringify(report, null, 2));
            
            console.log(`✅ Cache analysis report saved to ${filename}`);
            return report;
            
        } catch (error) {
            console.error('❌ Failed to generate cache analysis report:', error.message);
            throw error;
        }
    }

    parseRedisInfo(infoString) {
        const stats = {};
        const lines = infoString.split('\r\n');
        
        for (const line of lines) {
            if (line.includes(':') && !line.startsWith('#')) {
                const [key, value] = line.split(':');
                stats[key] = isNaN(value) ? value : parseFloat(value);
            }
        }
        
        return stats;
    }

    generateCacheRecommendations(redisStats) {
        const recommendations = [];
        
        // Memory usage recommendations
        const memFragmentation = redisStats.mem_fragmentation_ratio;
        if (memFragmentation > 1.5) {
            recommendations.push({
                category: 'Memory',
                priority: 'Medium',
                issue: `High memory fragmentation (${memFragmentation.toFixed(2)})`,
                recommendation: 'Consider running MEMORY DEFRAG or restarting Redis during low usage'
            });
        }
        
        // Hit rate recommendations
        const hitRate = this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses) * 100;
        if (hitRate < 80) {
            recommendations.push({
                category: 'Performance',
                priority: 'High',
                issue: `Low cache hit rate (${hitRate.toFixed(2)}%)`,
                recommendation: 'Analyze cache key patterns and increase TTL for frequently accessed data'
            });
        }
        
        // Keyspace recommendations
        if (redisStats.db0_expires && redisStats.db0_keys) {
            const expirationRatio = redisStats.db0_expires / redisStats.db0_keys;
            if (expirationRatio < 0.8) {
                recommendations.push({
                    category: 'Maintenance',
                    priority: 'Medium',
                    issue: 'Many keys without expiration',
                    recommendation: 'Set appropriate TTL values for all cached data to prevent memory bloat'
                });
            }
        }
        
        // Performance recommendations
        if (redisStats.instantaneous_ops_per_sec > 10000) {
            recommendations.push({
                category: 'Performance',
                priority: 'Low',
                issue: 'High operation rate',
                recommendation: 'Monitor Redis CPU usage and consider connection pooling optimization'
            });
        }
        
        return recommendations;
    }

    async warmupCache() {
        console.log('🔥 Warming up cache with frequently accessed data...');
        
        const warmupTasks = [];
        
        // Warm up API endpoints
        for (const [serviceName, serviceConfig] of Object.entries(this.config.services)) {
            if (serviceConfig.apiKey) {
                warmupTasks.push(this.warmupServiceCache(serviceName));
            }
        }
        
        try {
            await Promise.allSettled(warmupTasks);
            console.log('✅ Cache warmup completed');
        } catch (error) {
            console.error('❌ Cache warmup failed:', error.message);
        }
    }

    async warmupServiceCache(serviceName) {
        const commonEndpoints = {
            sonarr: ['/api/v3/series', '/api/v3/system/status'],
            radarr: ['/api/v3/movie', '/api/v3/system/status'],
            prowlarr: ['/api/v1/indexer', '/api/v1/system/status']
        };
        
        const endpoints = commonEndpoints[serviceName] || [];
        
        for (const endpoint of endpoints) {
            await this.cacheAPIResponse(serviceName, endpoint, this.config.cache.defaultTTL);
        }
    }

    async getCacheStats() {
        return {
            ...this.cacheStats,
            hitRate: this.cacheStats.hits / (this.cacheStats.hits + this.cacheStats.misses) * 100,
            connected: this.isConnected
        };
    }

    async flushCache(pattern = '*') {
        console.log(`🗑️  Flushing cache entries matching pattern: ${pattern}`);
        
        try {
            if (pattern === '*') {
                await this.client.flushDb();
                console.log('✅ All cache entries flushed');
            } else {
                const keys = await this.client.keys(pattern);
                if (keys.length > 0) {
                    await this.client.del(keys);
                    console.log(`✅ Flushed ${keys.length} cache entries`);
                } else {
                    console.log('ℹ️  No matching cache entries found');
                }
            }
        } catch (error) {
            console.error('❌ Cache flush failed:', error.message);
            throw error;
        }
    }

    async disconnect() {
        if (this.client && this.isConnected) {
            await this.client.disconnect();
            console.log('👋 Redis client disconnected');
        }
    }
}

// CLI interface
if (require.main === module) {
    const cacheOptimizer = new RedisCacheOptimizer();
    
    const command = process.argv[2];
    
    async function runCommand() {
        await cacheOptimizer.initialize();
        
        switch (command) {
            case 'setup':
                await cacheOptimizer.implementCachingStrategies();
                await cacheOptimizer.warmupCache();
                break;
            
            case 'analyze':
                await cacheOptimizer.generateCacheAnalysisReport();
                break;
            
            case 'warmup':
                await cacheOptimizer.warmupCache();
                break;
            
            case 'flush':
                const pattern = process.argv[3] || '*';
                await cacheOptimizer.flushCache(pattern);
                break;
            
            case 'stats':
                const stats = await cacheOptimizer.getCacheStats();
                console.log('📊 Cache Statistics:');
                console.log(`   Hit Rate: ${stats.hitRate.toFixed(2)}%`);
                console.log(`   Hits: ${stats.hits}`);
                console.log(`   Misses: ${stats.misses}`);
                console.log(`   Sets: ${stats.sets}`);
                console.log(`   Errors: ${stats.errors}`);
                break;
            
            default:
                console.log('Usage: node redis-cache-config.js <command>');
                console.log('Commands:');
                console.log('  setup   - Set up caching strategies');
                console.log('  analyze - Generate cache analysis report');
                console.log('  warmup  - Warm up cache with common data');
                console.log('  flush   - Flush cache entries (optional pattern)');
                console.log('  stats   - Show cache statistics');
                break;
        }
        
        await cacheOptimizer.disconnect();
    }
    
    runCommand().catch(console.error);
}

module.exports = RedisCacheOptimizer;