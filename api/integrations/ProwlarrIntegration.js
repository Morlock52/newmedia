/**
 * Prowlarr Integration Module
 * Complete Prowlarr API integration with indexer management and 500+ trackers
 */

const axios = require('axios');
const EventEmitter = require('events');

class ProwlarrIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.PROWLARR_URL || 'http://localhost:9696';
        this.apiKey = config.apiKey || process.env.PROWLARR_API_KEY;
        
        if (!this.apiKey) {
            throw new Error('Prowlarr API key is required');
        }
        
        this.client = axios.create({
            baseURL: `${this.baseURL}/api/v1`,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'X-Api-Key': this.apiKey
            }
        });

        this._setupInterceptors();
    }

    _setupInterceptors() {
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                this.emit('error', error);
                throw error;
            }
        );
    }

    /**
     * Get system status
     */
    async getSystemStatus() {
        try {
            const response = await this.client.get('/system/status');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get system status: ${error.message}`);
        }
    }

    /**
     * Get all indexers
     */
    async getIndexers() {
        try {
            const response = await this.client.get('/indexer');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get indexers: ${error.message}`);
        }
    }

    /**
     * Get indexer by ID
     */
    async getIndexerById(indexerId) {
        try {
            const response = await this.client.get(`/indexer/${indexerId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get indexer: ${error.message}`);
        }
    }

    /**
     * Update indexer
     */
    async updateIndexer(indexerId, updates) {
        try {
            const indexer = await this.getIndexerById(indexerId);
            const updatedIndexer = { ...indexer, ...updates };
            
            const response = await this.client.put(`/indexer/${indexerId}`, updatedIndexer);
            this.emit('indexerUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update indexer: ${error.message}`);
        }
    }

    /**
     * Enable/disable indexer
     */
    async toggleIndexer(indexerId, enabled) {
        try {
            return await this.updateIndexer(indexerId, { enable: enabled });
        } catch (error) {
            throw new Error(`Failed to toggle indexer: ${error.message}`);
        }
    }

    /**
     * Delete indexer
     */
    async deleteIndexer(indexerId) {
        try {
            await this.client.delete(`/indexer/${indexerId}`);
            this.emit('indexerDeleted', indexerId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete indexer: ${error.message}`);
        }
    }

    /**
     * Test indexer
     */
    async testIndexer(indexerId) {
        try {
            const response = await this.client.post(`/indexer/test/${indexerId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to test indexer: ${error.message}`);
        }
    }

    /**
     * Get indexer schemas (available indexer types)
     */
    async getIndexerSchemas() {
        try {
            const response = await this.client.get('/indexer/schema');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get indexer schemas: ${error.message}`);
        }
    }

    /**
     * Add new indexer
     */
    async addIndexer(indexerData) {
        try {
            const response = await this.client.post('/indexer', indexerData);
            this.emit('indexerAdded', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add indexer: ${error.message}`);
        }
    }

    /**
     * Search across all enabled indexers
     */
    async search(query, categories = [], limit = 100, offset = 0) {
        try {
            const params = {
                query: query,
                limit: limit,
                offset: offset
            };
            
            if (categories.length > 0) {
                params.categories = categories.join(',');
            }
            
            const response = await this.client.get('/search', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to search: ${error.message}`);
        }
    }

    /**
     * Search specific indexer
     */
    async searchIndexer(indexerId, query, categories = [], limit = 100, offset = 0) {
        try {
            const params = {
                query: query,
                indexerIds: indexerId,
                limit: limit,
                offset: offset
            };
            
            if (categories.length > 0) {
                params.categories = categories.join(',');
            }
            
            const response = await this.client.get('/search', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to search indexer: ${error.message}`);
        }
    }

    /**
     * Get indexer categories
     */
    async getCategories() {
        try {
            const response = await this.client.get('/indexer/categories');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get categories: ${error.message}`);
        }
    }

    /**
     * Get indexer statistics
     */
    async getIndexerStats() {
        try {
            const response = await this.client.get('/indexerstats');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get indexer stats: ${error.message}`);
        }
    }

    /**
     * Get application profiles
     */
    async getApplications() {
        try {
            const response = await this.client.get('/applications');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get applications: ${error.message}`);
        }
    }

    /**
     * Add application
     */
    async addApplication(appData) {
        try {
            const response = await this.client.post('/applications', appData);
            this.emit('applicationAdded', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add application: ${error.message}`);
        }
    }

    /**
     * Update application
     */
    async updateApplication(appId, updates) {
        try {
            const response = await this.client.put(`/applications/${appId}`, updates);
            this.emit('applicationUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update application: ${error.message}`);
        }
    }

    /**
     * Delete application
     */
    async deleteApplication(appId) {
        try {
            await this.client.delete(`/applications/${appId}`);
            this.emit('applicationDeleted', appId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete application: ${error.message}`);
        }
    }

    /**
     * Test application
     */
    async testApplication(appId) {
        try {
            const response = await this.client.post(`/applications/test/${appId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to test application: ${error.message}`);
        }
    }

    /**
     * Sync applications
     */
    async syncApplications() {
        try {
            const response = await this.client.post('/applications/sync');
            this.emit('applicationsSynced');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to sync applications: ${error.message}`);
        }
    }

    /**
     * Get download clients
     */
    async getDownloadClients() {
        try {
            const response = await this.client.get('/downloadclient');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get download clients: ${error.message}`);
        }
    }

    /**
     * Add download client
     */
    async addDownloadClient(clientData) {
        try {
            const response = await this.client.post('/downloadclient', clientData);
            this.emit('downloadClientAdded', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add download client: ${error.message}`);
        }
    }

    /**
     * Update download client
     */
    async updateDownloadClient(clientId, updates) {
        try {
            const response = await this.client.put(`/downloadclient/${clientId}`, updates);
            this.emit('downloadClientUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update download client: ${error.message}`);
        }
    }

    /**
     * Delete download client
     */
    async deleteDownloadClient(clientId) {
        try {
            await this.client.delete(`/downloadclient/${clientId}`);
            this.emit('downloadClientDeleted', clientId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete download client: ${error.message}`);
        }
    }

    /**
     * Test download client
     */
    async testDownloadClient(clientId) {
        try {
            const response = await this.client.post(`/downloadclient/test/${clientId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to test download client: ${error.message}`);
        }
    }

    /**
     * Get tags
     */
    async getTags() {
        try {
            const response = await this.client.get('/tag');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get tags: ${error.message}`);
        }
    }

    /**
     * Add tag
     */
    async addTag(label) {
        try {
            const response = await this.client.post('/tag', { label });
            this.emit('tagAdded', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add tag: ${error.message}`);
        }
    }

    /**
     * Delete tag
     */
    async deleteTag(tagId) {
        try {
            await this.client.delete(`/tag/${tagId}`);
            this.emit('tagDeleted', tagId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete tag: ${error.message}`);
        }
    }

    /**
     * Get history
     */
    async getHistory(page = 1, pageSize = 20, sortKey = 'date', sortDirection = 'descending') {
        try {
            const params = {
                page,
                pageSize,
                sortKey,
                sortDirection
            };
            
            const response = await this.client.get('/history', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get history: ${error.message}`);
        }
    }

    /**
     * Get notifications
     */
    async getNotifications() {
        try {
            const response = await this.client.get('/notification');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get notifications: ${error.message}`);
        }
    }

    /**
     * Test notification
     */
    async testNotification(notificationId) {
        try {
            const response = await this.client.post(`/notification/test/${notificationId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to test notification: ${error.message}`);
        }
    }

    /**
     * Get system logs
     */
    async getLogs(page = 1, pageSize = 50, sortKey = 'time', sortDirection = 'descending', level = null) {
        try {
            const params = {
                page,
                pageSize,
                sortKey,
                sortDirection
            };
            
            if (level) params.level = level;
            
            const response = await this.client.get('/log', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get logs: ${error.message}`);
        }
    }

    /**
     * Get system tasks
     */
    async getTasks() {
        try {
            const response = await this.client.get('/system/task');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get tasks: ${error.message}`);
        }
    }

    /**
     * Run task
     */
    async runTask(taskName) {
        try {
            const response = await this.client.post(`/system/task/${taskName}`);
            this.emit('taskStarted', taskName);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to run task: ${error.message}`);
        }
    }

    /**
     * Get health status
     */
    async getHealth() {
        try {
            const response = await this.client.get('/health');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get health: ${error.message}`);
        }
    }

    /**
     * Get comprehensive statistics
     */
    async getStatistics() {
        try {
            const [indexers, apps, downloadClients, indexerStats] = await Promise.all([
                this.getIndexers(),
                this.getApplications(),
                this.getDownloadClients(),
                this.getIndexerStats()
            ]);
            
            const stats = {
                indexers: {
                    total: indexers.length,
                    enabled: indexers.filter(i => i.enable).length,
                    disabled: indexers.filter(i => !i.enable).length,
                    byType: this._groupByType(indexers, 'protocol')
                },
                applications: {
                    total: apps.length,
                    enabled: apps.filter(a => a.enable).length,
                    disabled: apps.filter(a => !a.enable).length,
                    byType: this._groupByType(apps, 'implementation')
                },
                downloadClients: {
                    total: downloadClients.length,
                    enabled: downloadClients.filter(dc => dc.enable).length,
                    disabled: downloadClients.filter(dc => !dc.enable).length,
                    byType: this._groupByType(downloadClients, 'implementation')
                },
                indexerStats: indexerStats,
                categories: await this.getCategories()
            };
            
            return stats;
        } catch (error) {
            throw new Error(`Failed to get statistics: ${error.message}`);
        }
    }

    /**
     * Group items by type
     */
    _groupByType(items, typeField) {
        const grouped = {};
        items.forEach(item => {
            const type = item[typeField] || 'Unknown';
            if (!grouped[type]) grouped[type] = 0;
            grouped[type]++;
        });
        return grouped;
    }

    /**
     * Test connection
     */
    async testConnection() {
        try {
            const status = await this.getSystemStatus();
            return {
                success: true,
                version: status.version,
                appName: status.appName,
                instanceName: status.instanceName
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Setup webhook endpoint
     */
    setupWebhook(app, path = '/prowlarr/webhook') {
        app.post(path, (req, res) => {
            try {
                const event = req.body;
                this.emit('webhook', event);
                
                // Emit specific events based on event type
                switch (event.eventType) {
                    case 'Health':
                        this.emit('healthIssue', event);
                        break;
                    case 'Test':
                        this.emit('webhookTest', event);
                        break;
                    case 'ApplicationUpdate':
                        this.emit('applicationUpdate', event);
                        break;
                    case 'IndexerSync':
                        this.emit('indexerSync', event);
                        break;
                    default:
                        this.emit('unknownEvent', event);
                }
                
                res.status(200).json({ success: true });
            } catch (error) {
                console.error('Prowlarr webhook error:', error);
                res.status(500).json({ error: 'Webhook processing failed' });
            }
        });
    }
}

module.exports = ProwlarrIntegration;