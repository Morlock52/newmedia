const logger = require('../../middleware/logger.js');
/**
 * Sonarr Integration Module
 * Complete Sonarr API v3 integration with calendar, queue, history
 */

const axios = require('axios');
const EventEmitter = require('events');

class SonarrIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.SONARR_URL || 'http://localhost:8989';
        this.apiKey = config.apiKey || process.env.SONARR_API_KEY;
        
        if (!this.apiKey) {
            throw new Error('Sonarr API key is required');
        }
        
        this.client = axios.create({
            baseURL: `${this.baseURL}/api/v3`,
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
     * Get all series
     */
    async getSeries(includeSeasonImages = false) {
        try {
            const params = includeSeasonImages ? { includeSeasonImages: true } : {};
            const response = await this.client.get('/series', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get series: ${error.message}`);
        }
    }

    /**
     * Get series by ID
     */
    async getSeriesById(seriesId) {
        try {
            const response = await this.client.get(`/series/${seriesId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get series: ${error.message}`);
        }
    }

    /**
     * Add new series
     */
    async addSeries(seriesData) {
        try {
            const requiredFields = {
                title: seriesData.title,
                titleSlug: seriesData.titleSlug || seriesData.title.toLowerCase().replace(/\s+/g, '-'),
                tvdbId: seriesData.tvdbId,
                qualityProfileId: seriesData.qualityProfileId || 1,
                languageProfileId: seriesData.languageProfileId || 1,
                rootFolderPath: seriesData.rootFolderPath,
                monitored: seriesData.monitored !== false,
                seasonFolder: seriesData.seasonFolder !== false,
                seriesType: seriesData.seriesType || 'standard',
                images: seriesData.images || [],
                seasons: seriesData.seasons || []
            };

            const response = await this.client.post('/series', requiredFields);
            this.emit('seriesAdded', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add series: ${error.message}`);
        }
    }

    /**
     * Update series
     */
    async updateSeries(seriesId, updates) {
        try {
            const series = await this.getSeriesById(seriesId);
            const updatedSeries = { ...series, ...updates };
            
            const response = await this.client.put(`/series/${seriesId}`, updatedSeries);
            this.emit('seriesUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update series: ${error.message}`);
        }
    }

    /**
     * Delete series
     */
    async deleteSeries(seriesId, deleteFiles = false, addImportListExclusion = false) {
        try {
            const params = {
                deleteFiles,
                addImportListExclusion
            };
            
            await this.client.delete(`/series/${seriesId}`, { params });
            this.emit('seriesDeleted', { seriesId, deleteFiles, addImportListExclusion });
            return true;
        } catch (error) {
            throw new Error(`Failed to delete series: ${error.message}`);
        }
    }

    /**
     * Search for series
     */
    async searchSeries(term) {
        try {
            const response = await this.client.get(`/series/lookup`, {
                params: { term }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to search series: ${error.message}`);
        }
    }

    /**
     * Get episodes for series
     */
    async getEpisodes(seriesId, seasonNumber = null) {
        try {
            const params = { seriesId };
            if (seasonNumber !== null) {
                params.seasonNumber = seasonNumber;
            }
            
            const response = await this.client.get('/episode', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get episodes: ${error.message}`);
        }
    }

    /**
     * Get episode by ID
     */
    async getEpisodeById(episodeId) {
        try {
            const response = await this.client.get(`/episode/${episodeId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get episode: ${error.message}`);
        }
    }

    /**
     * Update episode
     */
    async updateEpisode(episodeId, updates) {
        try {
            const episode = await this.getEpisodeById(episodeId);
            const updatedEpisode = { ...episode, ...updates };
            
            const response = await this.client.put(`/episode/${episodeId}`, updatedEpisode);
            this.emit('episodeUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update episode: ${error.message}`);
        }
    }

    /**
     * Get calendar
     */
    async getCalendar(startDate = null, endDate = null, unmonitored = false) {
        try {
            const params = { unmonitored };
            
            if (startDate) params.start = startDate;
            if (endDate) params.end = endDate;
            
            const response = await this.client.get('/calendar', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get calendar: ${error.message}`);
        }
    }

    /**
     * Get upcoming episodes
     */
    async getUpcoming(days = 7) {
        try {
            const startDate = new Date().toISOString();
            const endDate = new Date(Date.now() + (days * 24 * 60 * 60 * 1000)).toISOString();
            
            return await this.getCalendar(startDate, endDate);
        } catch (error) {
            throw new Error(`Failed to get upcoming episodes: ${error.message}`);
        }
    }

    /**
     * Get queue
     */
    async getQueue(page = 1, pageSize = 20, sortKey = 'timeleft', sortDirection = 'ascending') {
        try {
            const params = {
                page,
                pageSize,
                sortKey,
                sortDirection,
                includeUnknownSeriesItems: true
            };
            
            const response = await this.client.get('/queue', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get queue: ${error.message}`);
        }
    }

    /**
     * Remove item from queue
     */
    async removeFromQueue(queueId, removeFromClient = true, blocklist = false) {
        try {
            const params = { removeFromClient, blocklist };
            await this.client.delete(`/queue/${queueId}`, { params });
            
            this.emit('queueItemRemoved', { queueId, removeFromClient, blocklist });
            return true;
        } catch (error) {
            throw new Error(`Failed to remove from queue: ${error.message}`);
        }
    }

    /**
     * Get history
     */
    async getHistory(page = 1, pageSize = 20, sortKey = 'date', sortDirection = 'descending', episodeId = null, eventType = null) {
        try {
            const params = {
                page,
                pageSize,
                sortKey,
                sortDirection
            };
            
            if (episodeId) params.episodeId = episodeId;
            if (eventType) params.eventType = eventType;
            
            const response = await this.client.get('/history', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get history: ${error.message}`);
        }
    }

    /**
     * Get activity
     */
    async getActivity() {
        try {
            const response = await this.client.get('/system/task');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get activity: ${error.message}`);
        }
    }

    /**
     * Trigger search for series
     */
    async searchSeries(seriesId) {
        try {
            const command = {
                name: 'SeriesSearch',
                seriesId: seriesId
            };
            
            const response = await this.client.post('/command', command);
            this.emit('searchTriggered', { type: 'series', seriesId, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to trigger series search: ${error.message}`);
        }
    }

    /**
     * Trigger search for season
     */
    async searchSeason(seriesId, seasonNumber) {
        try {
            const command = {
                name: 'SeasonSearch',
                seriesId: seriesId,
                seasonNumber: seasonNumber
            };
            
            const response = await this.client.post('/command', command);
            this.emit('searchTriggered', { type: 'season', seriesId, seasonNumber, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to trigger season search: ${error.message}`);
        }
    }

    /**
     * Trigger search for episode
     */
    async searchEpisode(episodeIds) {
        try {
            const command = {
                name: 'EpisodeSearch',
                episodeIds: Array.isArray(episodeIds) ? episodeIds : [episodeIds]
            };
            
            const response = await this.client.post('/command', command);
            this.emit('searchTriggered', { type: 'episode', episodeIds, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to trigger episode search: ${error.message}`);
        }
    }

    /**
     * Refresh series
     */
    async refreshSeries(seriesId = null) {
        try {
            const command = seriesId ? 
                { name: 'RefreshSeries', seriesId: seriesId } :
                { name: 'RefreshSeries' };
            
            const response = await this.client.post('/command', command);
            this.emit('refreshTriggered', { seriesId, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to refresh series: ${error.message}`);
        }
    }

    /**
     * Rescan series
     */
    async rescanSeries(seriesId = null) {
        try {
            const command = seriesId ?
                { name: 'RescanSeries', seriesId: seriesId } :
                { name: 'RescanSeries' };
            
            const response = await this.client.post('/command', command);
            this.emit('rescanTriggered', { seriesId, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to rescan series: ${error.message}`);
        }
    }

    /**
     * Get quality profiles
     */
    async getQualityProfiles() {
        try {
            const response = await this.client.get('/qualityprofile');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get quality profiles: ${error.message}`);
        }
    }

    /**
     * Get language profiles
     */
    async getLanguageProfiles() {
        try {
            const response = await this.client.get('/languageprofile');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get language profiles: ${error.message}`);
        }
    }

    /**
     * Get root folders
     */
    async getRootFolders() {
        try {
            const response = await this.client.get('/rootfolder');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get root folders: ${error.message}`);
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
     * Get indexers
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
     * Get series statistics
     */
    async getStatistics() {
        try {
            const series = await this.getSeries();
            const queue = await this.getQueue();
            
            const stats = {
                totalSeries: series.length,
                monitoredSeries: series.filter(s => s.monitored).length,
                unmonitoredSeries: series.filter(s => !s.monitored).length,
                continuingSeries: series.filter(s => s.status === 'continuing').length,
                endedSeries: series.filter(s => s.status === 'ended').length,
                totalEpisodes: series.reduce((total, s) => total + (s.statistics?.totalEpisodeCount || 0), 0),
                downloadedEpisodes: series.reduce((total, s) => total + (s.statistics?.episodeFileCount || 0), 0),
                queuedDownloads: queue.totalRecords || 0,
                diskSpace: await this._getDiskSpace()
            };
            
            return stats;
        } catch (error) {
            throw new Error(`Failed to get statistics: ${error.message}`);
        }
    }

    /**
     * Get disk space information
     */
    async _getDiskSpace() {
        try {
            const response = await this.client.get('/diskspace');
            return response.data;
        } catch (error) {
            return [];
        }
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
    setupWebhook(app, path = '/sonarr/webhook') {
        app.post(path, (req, res) => {
            try {
                const event = req.body;
                this.emit('webhook', event);
                
                // Emit specific events based on event type
                switch (event.eventType) {
                    case 'Grab':
                        this.emit('episodeGrabbed', event);
                        break;
                    case 'Download':
                        this.emit('episodeDownloaded', event);
                        break;
                    case 'Rename':
                        this.emit('episodeRenamed', event);
                        break;
                    case 'EpisodeFileDelete':
                        this.emit('episodeFileDeleted', event);
                        break;
                    case 'SeriesDelete':
                        this.emit('seriesDeleted', event);
                        break;
                    case 'Health':
                        this.emit('healthIssue', event);
                        break;
                    case 'Test':
                        this.emit('webhookTest', event);
                        break;
                    default:
                        this.emit('unknownEvent', event);
                }
                
                res.status(200).json({ success: true });
            } catch (error) {
                logger.error('Sonarr webhook error:', error);
                res.status(500).json({ error: 'Webhook processing failed' });
            }
        });
    }
}

module.exports = SonarrIntegration;