// Database Performance Optimizer
// Optimizes database queries, indexing strategies, and connection pooling

const fs = require('fs').promises;
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const { promisify } = require('util');

class DatabasePerformanceOptimizer {
    constructor(options = {}) {
        this.options = {
            connectionPoolSize: 10,
            queryTimeout: 30000,
            cacheTTL: 300000, // 5 minutes
            enableWAL: true,
            enableForeignKeys: true,
            enableQueryPlan: true,
            analyzeFrequency: 3600000, // 1 hour
            ...options
        };
        
        this.connections = new Map();
        this.queryCache = new Map();
        this.performanceMetrics = new Map();
        this.slowQueries = [];
        this.optimizationHistory = [];
    }

    async optimizeDatabase(dbPath, serviceName) {
        console.log(`🔧 Optimizing database performance for ${serviceName}...`);
        
        try {
            const db = await this.getOptimizedConnection(dbPath);
            
            // Database configuration optimization
            await this.optimizeConfiguration(db);
            
            // Index optimization
            await this.optimizeIndexes(db, serviceName);
            
            // Query optimization
            await this.analyzeQueries(db, serviceName);
            
            // Maintenance optimization
            await this.performMaintenance(db);
            
            // Connection pooling setup
            this.setupConnectionPooling(dbPath, serviceName);
            
            const metrics = await this.generateOptimizationReport(db, serviceName);
            
            console.log(`✅ Database optimization completed for ${serviceName}`);
            return metrics;
            
        } catch (error) {
            console.error(`❌ Database optimization failed for ${serviceName}:`, error);
            throw error;
        }
    }

    async getOptimizedConnection(dbPath) {
        if (this.connections.has(dbPath)) {
            return this.connections.get(dbPath);
        }
        
        const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READWRITE);
        const dbAsync = {
            run: promisify(db.run.bind(db)),
            get: promisify(db.get.bind(db)),
            all: promisify(db.all.bind(db)),
            exec: promisify(db.exec.bind(db)),
            close: promisify(db.close.bind(db))
        };
        
        this.connections.set(dbPath, dbAsync);
        return dbAsync;
    }

    async optimizeConfiguration(db) {
        console.log('⚙️  Applying database configuration optimizations...');
        
        const optimizations = [
            // Enable WAL mode for better concurrency
            'PRAGMA journal_mode = WAL;',
            
            // Optimize synchronous mode
            'PRAGMA synchronous = NORMAL;',
            
            // Increase cache size (negative value = KB)
            'PRAGMA cache_size = -64000;', // 64MB cache
            
            // Enable foreign key constraints
            'PRAGMA foreign_keys = ON;',
            
            // Optimize page size
            'PRAGMA page_size = 4096;',
            
            // Set temporary store to memory
            'PRAGMA temp_store = MEMORY;',
            
            // Optimize memory-mapped I/O
            'PRAGMA mmap_size = 268435456;', // 256MB
            
            // Enable recursive triggers
            'PRAGMA recursive_triggers = ON;',
            
            // Optimize busy timeout
            'PRAGMA busy_timeout = 30000;',
            
            // Enable query planner optimization
            'PRAGMA optimize;'
        ];
        
        for (const pragma of optimizations) {
            try {
                await db.run(pragma);
                console.log(`✅ Applied: ${pragma}`);
            } catch (error) {
                console.warn(`⚠️  Failed to apply: ${pragma} - ${error.message}`);
            }
        }
    }

    async optimizeIndexes(db, serviceName) {
        console.log('📊 Analyzing and optimizing database indexes...');
        
        // Get all tables
        const tables = await db.all(`
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        `);
        
        const indexOptimizations = [];
        
        for (const table of tables) {
            const tableName = table.name;
            
            // Analyze table structure
            const columns = await db.all(`PRAGMA table_info(${tableName})`);
            const indexes = await db.all(`PRAGMA index_list(${tableName})`);
            const stats = await this.getTableStats(db, tableName);
            
            // Service-specific optimizations
            const recommendations = this.generateIndexRecommendations(
                serviceName, tableName, columns, indexes, stats
            );
            
            for (const rec of recommendations) {
                try {
                    await db.run(rec.sql);
                    indexOptimizations.push({
                        table: tableName,
                        type: rec.type,
                        sql: rec.sql,
                        impact: rec.impact
                    });
                    console.log(`✅ Created ${rec.type} for ${tableName}: ${rec.description}`);
                } catch (error) {
                    console.warn(`⚠️  Failed to create index: ${error.message}`);
                }
            }
        }
        
        return indexOptimizations;
    }

    generateIndexRecommendations(serviceName, tableName, columns, existingIndexes, stats) {
        const recommendations = [];
        const existingIndexNames = existingIndexes.map(idx => idx.name);
        
        // Service-specific index recommendations
        switch (serviceName) {
            case 'sonarr':
                if (tableName === 'Series') {
                    recommendations.push(...this.getSonarrSeriesIndexes(existingIndexNames));
                } else if (tableName === 'Episodes') {
                    recommendations.push(...this.getSonarrEpisodeIndexes(existingIndexNames));
                }
                break;
                
            case 'radarr':
                if (tableName === 'Movies') {
                    recommendations.push(...this.getRadarrMovieIndexes(existingIndexNames));
                } else if (tableName === 'MovieFiles') {
                    recommendations.push(...this.getRadarrFileIndexes(existingIndexNames));
                }
                break;
                
            case 'prowlarr':
                if (tableName === 'IndexerStatus') {
                    recommendations.push(...this.getProwlarrStatusIndexes(existingIndexNames));
                }
                break;
                
            case 'jellyfin':
                if (tableName === 'MediaItems') {
                    recommendations.push(...this.getJellyfinMediaIndexes(existingIndexNames));
                }
                break;
        }
        
        // Generic optimizations based on table analysis
        recommendations.push(...this.getGenericIndexRecommendations(
            tableName, columns, existingIndexNames, stats
        ));
        
        return recommendations.filter(rec => !existingIndexNames.includes(rec.name));
    }

    getSonarrSeriesIndexes(existingNames) {
        return [
            {
                type: 'performance_index',
                name: 'IX_Series_TvdbId',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Series_TvdbId ON Series (TvdbId)',
                description: 'TVDB ID lookup optimization',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_Series_Title_SortTitle',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Series_Title_SortTitle ON Series (Title, SortTitle)',
                description: 'Series search optimization',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_Series_Monitored_Status',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Series_Monitored_Status ON Series (Monitored, Status)',
                description: 'Monitoring status queries',
                impact: 'medium'
            }
        ];
    }

    getSonarrEpisodeIndexes(existingNames) {
        return [
            {
                type: 'performance_index',
                name: 'IX_Episodes_SeriesId_SeasonNumber',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Episodes_SeriesId_SeasonNumber ON Episodes (SeriesId, SeasonNumber)',
                description: 'Season episode lookups',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_Episodes_AirDateUtc',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Episodes_AirDateUtc ON Episodes (AirDateUtc)',
                description: 'Air date sorting optimization',
                impact: 'medium'
            },
            {
                type: 'performance_index',
                name: 'IX_Episodes_Monitored_HasFile',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Episodes_Monitored_HasFile ON Episodes (Monitored, HasFile)',
                description: 'Missing episode queries',
                impact: 'high'
            }
        ];
    }

    getRadarrMovieIndexes(existingNames) {
        return [
            {
                type: 'performance_index',
                name: 'IX_Movies_TmdbId',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Movies_TmdbId ON Movies (TmdbId)',
                description: 'TMDB ID lookup optimization',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_Movies_Title_SortTitle',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Movies_Title_SortTitle ON Movies (Title, SortTitle)',
                description: 'Movie search optimization',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_Movies_Monitored_Status',
                sql: 'CREATE INDEX IF NOT EXISTS IX_Movies_Monitored_Status ON Movies (Monitored, Status)',
                description: 'Monitoring queries',
                impact: 'medium'
            }
        ];
    }

    getRadarrFileIndexes(existingNames) {
        return [
            {
                type: 'performance_index',
                name: 'IX_MovieFiles_MovieId',
                sql: 'CREATE INDEX IF NOT EXISTS IX_MovieFiles_MovieId ON MovieFiles (MovieId)',
                description: 'Movie file associations',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_MovieFiles_Path',
                sql: 'CREATE INDEX IF NOT EXISTS IX_MovieFiles_Path ON MovieFiles (Path)',
                description: 'File path lookups',
                impact: 'medium'
            }
        ];
    }

    getProwlarrStatusIndexes(existingNames) {
        return [
            {
                type: 'performance_index',
                name: 'IX_IndexerStatus_IndexerId',
                sql: 'CREATE INDEX IF NOT EXISTS IX_IndexerStatus_IndexerId ON IndexerStatus (IndexerId)',
                description: 'Indexer status lookups',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_IndexerStatus_LastRssSyncReleaseInfo',
                sql: 'CREATE INDEX IF NOT EXISTS IX_IndexerStatus_LastRssSyncReleaseInfo ON IndexerStatus (LastRssSyncReleaseInfo)',
                description: 'RSS sync optimization',
                impact: 'medium'
            }
        ];
    }

    getJellyfinMediaIndexes(existingNames) {
        return [
            {
                type: 'performance_index',
                name: 'IX_MediaItems_Path',
                sql: 'CREATE INDEX IF NOT EXISTS IX_MediaItems_Path ON MediaItems (Path)',
                description: 'Media path lookups',
                impact: 'high'
            },
            {
                type: 'performance_index',
                name: 'IX_MediaItems_Type_DateCreated',
                sql: 'CREATE INDEX IF NOT EXISTS IX_MediaItems_Type_DateCreated ON MediaItems (Type, DateCreated)',
                description: 'Media type and date queries',
                impact: 'medium'
            }
        ];
    }

    getGenericIndexRecommendations(tableName, columns, existingIndexes, stats) {
        const recommendations = [];
        
        // Look for foreign key columns without indexes
        const foreignKeyColumns = columns.filter(col => 
            col.name.toLowerCase().includes('id') && 
            col.name.toLowerCase() !== 'id' &&
            col.pk === 0
        );
        
        foreignKeyColumns.forEach(col => {
            const indexName = `IX_${tableName}_${col.name}`;
            if (!existingIndexes.includes(indexName)) {
                recommendations.push({
                    type: 'foreign_key_index',
                    name: indexName,
                    sql: `CREATE INDEX IF NOT EXISTS ${indexName} ON ${tableName} (${col.name})`,
                    description: `Foreign key optimization for ${col.name}`,
                    impact: 'medium'
                });
            }
        });
        
        // Look for frequently queried text columns
        const textColumns = columns.filter(col => 
            col.type.toLowerCase().includes('text') || 
            col.type.toLowerCase().includes('varchar')
        );
        
        textColumns.forEach(col => {
            if (['name', 'title', 'path', 'filename'].some(keyword => 
                col.name.toLowerCase().includes(keyword))) {
                const indexName = `IX_${tableName}_${col.name}`;
                if (!existingIndexes.includes(indexName)) {
                    recommendations.push({
                        type: 'text_search_index',
                        name: indexName,
                        sql: `CREATE INDEX IF NOT EXISTS ${indexName} ON ${tableName} (${col.name})`,
                        description: `Text search optimization for ${col.name}`,
                        impact: 'low'
                    });
                }
            }
        });
        
        return recommendations;
    }

    async getTableStats(db, tableName) {
        try {
            const stats = await db.get(`
                SELECT 
                    COUNT(*) as row_count,
                    MAX(rowid) as max_rowid
                FROM ${tableName}
            `);
            
            return stats;
        } catch (error) {
            console.warn(`Failed to get stats for ${tableName}:`, error.message);
            return { row_count: 0, max_rowid: 0 };
        }
    }

    async analyzeQueries(db, serviceName) {
        console.log('🔍 Analyzing query performance...');
        
        // Enable query plan analysis
        await db.run('PRAGMA query_only = ON');
        
        const commonQueries = this.getCommonQueries(serviceName);
        const queryAnalysis = [];
        
        for (const query of commonQueries) {
            try {
                const plan = await db.all(`EXPLAIN QUERY PLAN ${query.sql}`);
                const complexity = this.analyzeQueryComplexity(plan);
                
                queryAnalysis.push({
                    query: query.name,
                    sql: query.sql,
                    plan: plan,
                    complexity: complexity,
                    recommendations: this.getQueryOptimizationRecommendations(plan, complexity)
                });
                
                if (complexity.score > 100) {
                    this.slowQueries.push({
                        service: serviceName,
                        query: query.name,
                        complexity: complexity.score,
                        recommendations: this.getQueryOptimizationRecommendations(plan, complexity)
                    });
                }
                
            } catch (error) {
                console.warn(`Failed to analyze query ${query.name}:`, error.message);
            }
        }
        
        await db.run('PRAGMA query_only = OFF');
        return queryAnalysis;
    }

    getCommonQueries(serviceName) {
        const queries = {
            sonarr: [
                {
                    name: 'GetSeriesByTitle',
                    sql: 'SELECT * FROM Series WHERE Title LIKE "%test%" ORDER BY SortTitle'
                },
                {
                    name: 'GetEpisodesBySeries',
                    sql: 'SELECT * FROM Episodes WHERE SeriesId = 1 ORDER BY SeasonNumber, EpisodeNumber'
                },
                {
                    name: 'GetUnmonitoredEpisodes',
                    sql: 'SELECT * FROM Episodes WHERE Monitored = 0 AND HasFile = 0'
                }
            ],
            radarr: [
                {
                    name: 'GetMoviesByTitle',
                    sql: 'SELECT * FROM Movies WHERE Title LIKE "%test%" ORDER BY SortTitle'
                },
                {
                    name: 'GetMonitoredMovies',
                    sql: 'SELECT * FROM Movies WHERE Monitored = 1 AND Status != "deleted"'
                }
            ],
            prowlarr: [
                {
                    name: 'GetIndexerStatus',
                    sql: 'SELECT * FROM IndexerStatus ORDER BY LastRssSyncReleaseInfo DESC'
                }
            ]
        };
        
        return queries[serviceName] || [];
    }

    analyzeQueryComplexity(plan) {
        let score = 0;
        let issues = [];
        
        plan.forEach(step => {
            const detail = step.detail.toLowerCase();
            
            // Scan type analysis
            if (detail.includes('scan table')) {
                if (detail.includes('using index')) {
                    score += 1; // Good - using index
                } else {
                    score += 50; // Bad - full table scan
                    issues.push('Full table scan detected');
                }
            }
            
            if (detail.includes('temp b-tree')) {
                score += 20; // Expensive temporary sorting
                issues.push('Temporary B-tree for sorting');
            }
            
            if (detail.includes('compound subqueries')) {
                score += 30; // Complex subqueries
                issues.push('Complex subqueries detected');
            }
        });
        
        return {
            score,
            issues,
            complexity: score < 10 ? 'low' : score < 50 ? 'medium' : 'high'
        };
    }

    getQueryOptimizationRecommendations(plan, complexity) {
        const recommendations = [];
        
        complexity.issues.forEach(issue => {
            switch (issue) {
                case 'Full table scan detected':
                    recommendations.push({
                        type: 'indexing',
                        priority: 'high',
                        suggestion: 'Add appropriate indexes for WHERE clauses',
                        impact: 'Query performance will improve significantly'
                    });
                    break;
                    
                case 'Temporary B-tree for sorting':
                    recommendations.push({
                        type: 'indexing',
                        priority: 'medium',
                        suggestion: 'Add index on ORDER BY columns',
                        impact: 'Eliminates temporary sorting operations'
                    });
                    break;
                    
                case 'Complex subqueries detected':
                    recommendations.push({
                        type: 'query_rewrite',
                        priority: 'medium',
                        suggestion: 'Consider rewriting with JOINs or EXISTS clauses',
                        impact: 'Reduces query complexity and improves performance'
                    });
                    break;
            }
        });
        
        return recommendations;
    }

    async performMaintenance(db) {
        console.log('🧹 Performing database maintenance...');
        
        const maintenanceTasks = [
            {
                name: 'Analyze Statistics',
                sql: 'ANALYZE;',
                description: 'Update query planner statistics'
            },
            {
                name: 'Vacuum Incremental',
                sql: 'PRAGMA incremental_vacuum;',
                description: 'Reclaim unused space incrementally'
            },
            {
                name: 'Optimize',
                sql: 'PRAGMA optimize;',
                description: 'Run query planner optimizations'
            },
            {
                name: 'WAL Checkpoint',
                sql: 'PRAGMA wal_checkpoint(TRUNCATE);',
                description: 'Checkpoint WAL file to main database'
            }
        ];
        
        const results = [];
        
        for (const task of maintenanceTasks) {
            try {
                const startTime = Date.now();
                await db.run(task.sql);
                const duration = Date.now() - startTime;
                
                results.push({
                    task: task.name,
                    duration,
                    success: true
                });
                
                console.log(`✅ ${task.name} completed in ${duration}ms`);
            } catch (error) {
                results.push({
                    task: task.name,
                    success: false,
                    error: error.message
                });
                
                console.warn(`⚠️  ${task.name} failed: ${error.message}`);
            }
        }
        
        return results;
    }

    setupConnectionPooling(dbPath, serviceName) {
        console.log(`🔗 Setting up connection pooling for ${serviceName}...`);
        
        // Implement connection pooling strategy
        const pool = {
            connections: [],
            maxConnections: this.options.connectionPoolSize,
            currentConnections: 0,
            
            async getConnection() {
                if (this.connections.length > 0) {
                    return this.connections.pop();
                }
                
                if (this.currentConnections < this.maxConnections) {
                    this.currentConnections++;
                    return await this.createConnection(dbPath);
                }
                
                // Wait for available connection
                return new Promise((resolve) => {
                    const checkConnection = () => {
                        if (this.connections.length > 0) {
                            resolve(this.connections.pop());
                        } else {
                            setTimeout(checkConnection, 10);
                        }
                    };
                    checkConnection();
                });
            },
            
            releaseConnection(connection) {
                this.connections.push(connection);
            },
            
            async createConnection(dbPath) {
                const db = new sqlite3.Database(dbPath, sqlite3.OPEN_READWRITE);
                await this.optimizeConfiguration(db);
                return db;
            }
        };
        
        this.connectionPools = this.connectionPools || new Map();
        this.connectionPools.set(serviceName, pool);
        
        console.log(`✅ Connection pool created for ${serviceName} (max: ${this.options.connectionPoolSize})`);
    }

    async generateOptimizationReport(db, serviceName) {
        const report = {
            service: serviceName,
            timestamp: new Date().toISOString(),
            optimizations: [],
            performance: {},
            recommendations: []
        };
        
        // Collect database statistics
        const stats = await db.all(`
            SELECT 
                name,
                tbl_name,
                sql
            FROM sqlite_master 
            WHERE type = 'index'
        `);
        
        report.performance.indexCount = stats.length;
        report.performance.slowQueries = this.slowQueries.filter(q => q.service === serviceName);
        
        // Database size analysis
        const sizeInfo = await db.get('PRAGMA page_count');
        const pageSize = await db.get('PRAGMA page_size');
        
        report.performance.databaseSize = sizeInfo.page_count * pageSize.page_size;
        report.performance.optimization_score = this.calculateOptimizationScore(report);
        
        return report;
    }

    calculateOptimizationScore(report) {
        let score = 100;
        
        // Deduct points for slow queries
        score -= report.performance.slowQueries.length * 10;
        
        // Deduct points for large database without adequate indexes
        if (report.performance.databaseSize > 100000000) { // 100MB
            if (report.performance.indexCount < 10) {
                score -= 20;
            }
        }
        
        // Bonus points for having indexes
        score += Math.min(report.performance.indexCount * 2, 20);
        
        return Math.max(0, Math.min(100, score));
    }

    async optimizeAllServices() {
        console.log('🚀 Starting comprehensive database optimization...');
        
        const services = [
            { name: 'sonarr', dbPath: './sonarr-config/sonarr.db' },
            { name: 'radarr', dbPath: './radarr-config/radarr.db' },
            { name: 'lidarr', dbPath: './lidarr-config/lidarr.db' },
            { name: 'prowlarr', dbPath: './prowlarr-config/prowlarr.db' },
            { name: 'bazarr', dbPath: './bazarr-config/db/bazarr.db' }
        ];
        
        const results = [];
        
        for (const service of services) {
            try {
                const serviceExists = await fs.access(service.dbPath).then(() => true).catch(() => false);
                
                if (serviceExists) {
                    const result = await this.optimizeDatabase(service.dbPath, service.name);
                    results.push({
                        service: service.name,
                        success: true,
                        metrics: result
                    });
                } else {
                    console.log(`⚠️  Database not found for ${service.name}: ${service.dbPath}`);
                    results.push({
                        service: service.name,
                        success: false,
                        error: 'Database file not found'
                    });
                }
            } catch (error) {
                console.error(`❌ Optimization failed for ${service.name}:`, error.message);
                results.push({
                    service: service.name,
                    success: false,
                    error: error.message
                });
            }
        }
        
        // Generate summary report
        const summary = this.generateSummaryReport(results);
        await this.saveSummaryReport(summary);
        
        console.log('✅ Database optimization completed for all services');
        return summary;
    }

    generateSummaryReport(results) {
        const summary = {
            timestamp: new Date().toISOString(),
            totalServices: results.length,
            successfulOptimizations: results.filter(r => r.success).length,
            failedOptimizations: results.filter(r => !r.success).length,
            overallScore: 0,
            services: results,
            recommendations: []
        };
        
        // Calculate overall score
        const successfulResults = results.filter(r => r.success);
        if (successfulResults.length > 0) {
            summary.overallScore = successfulResults.reduce((sum, r) => 
                sum + (r.metrics?.performance?.optimization_score || 0), 0
            ) / successfulResults.length;
        }
        
        // Generate global recommendations
        if (summary.overallScore < 80) {
            summary.recommendations.push({
                type: 'performance',
                priority: 'high',
                message: 'Multiple databases require optimization',
                actions: [
                    'Implement regular maintenance schedules',
                    'Monitor query performance',
                    'Consider database cleanup procedures'
                ]
            });
        }
        
        return summary;
    }

    async saveSummaryReport(summary) {
        const reportPath = './performance-reports/database-optimization-summary.json';
        
        try {
            await fs.mkdir(path.dirname(reportPath), { recursive: true });
            await fs.writeFile(reportPath, JSON.stringify(summary, null, 2));
            console.log(`📊 Summary report saved to ${reportPath}`);
        } catch (error) {
            console.warn('Failed to save summary report:', error.message);
        }
    }

    async cleanup() {
        // Close all database connections
        for (const [path, db] of this.connections) {
            try {
                await db.close();
            } catch (error) {
                console.warn(`Failed to close connection for ${path}:`, error.message);
            }
        }
        
        this.connections.clear();
        console.log('🧹 Database connections cleaned up');
    }
}

// CLI usage
if (require.main === module) {
    (async () => {
        const optimizer = new DatabasePerformanceOptimizer();
        
        try {
            await optimizer.optimizeAllServices();
        } catch (error) {
            console.error('Database optimization failed:', error);
            process.exit(1);
        } finally {
            await optimizer.cleanup();
        }
    })();
}

module.exports = DatabasePerformanceOptimizer;