/**
 * Radarr Integration Module
 * Complete Radarr API v3 integration with movie management
 */

const axios = require('axios');
const EventEmitter = require('events');

class RadarrIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.RADARR_URL || 'http://localhost:7878';
        this.apiKey = config.apiKey || process.env.RADARR_API_KEY;
        
        if (!this.apiKey) {
            throw new Error('Radarr API key is required');
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
     * Get all movies
     */
    async getMovies() {
        try {
            const response = await this.client.get('/movie');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get movies: ${error.message}`);
        }
    }

    /**
     * Get movie by ID
     */
    async getMovieById(movieId) {
        try {
            const response = await this.client.get(`/movie/${movieId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get movie: ${error.message}`);
        }
    }

    /**
     * Add new movie
     */
    async addMovie(movieData) {
        try {
            const requiredFields = {
                title: movieData.title,
                titleSlug: movieData.titleSlug || movieData.title.toLowerCase().replace(/\s+/g, '-'),
                tmdbId: movieData.tmdbId,
                imdbId: movieData.imdbId,
                year: movieData.year,
                qualityProfileId: movieData.qualityProfileId || 1,
                rootFolderPath: movieData.rootFolderPath,
                monitored: movieData.monitored !== false,
                minimumAvailability: movieData.minimumAvailability || 'announced',
                images: movieData.images || [],
                tags: movieData.tags || []
            };

            const response = await this.client.post('/movie', requiredFields);
            this.emit('movieAdded', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add movie: ${error.message}`);
        }
    }

    /**
     * Update movie
     */
    async updateMovie(movieId, updates) {
        try {
            const movie = await this.getMovieById(movieId);
            const updatedMovie = { ...movie, ...updates };
            
            const response = await this.client.put(`/movie/${movieId}`, updatedMovie);
            this.emit('movieUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update movie: ${error.message}`);
        }
    }

    /**
     * Delete movie
     */
    async deleteMovie(movieId, deleteFiles = false, addImportListExclusion = false) {
        try {
            const params = {
                deleteFiles,
                addImportListExclusion
            };
            
            await this.client.delete(`/movie/${movieId}`, { params });
            this.emit('movieDeleted', { movieId, deleteFiles, addImportListExclusion });
            return true;
        } catch (error) {
            throw new Error(`Failed to delete movie: ${error.message}`);
        }
    }

    /**
     * Search for movies
     */
    async searchMovies(term) {
        try {
            const response = await this.client.get(`/movie/lookup`, {
                params: { term }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to search movies: ${error.message}`);
        }
    }

    /**
     * Get movie files
     */
    async getMovieFiles(movieId = null) {
        try {
            const params = movieId ? { movieId } : {};
            const response = await this.client.get('/moviefile', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get movie files: ${error.message}`);
        }
    }

    /**
     * Get movie file by ID
     */
    async getMovieFileById(fileId) {
        try {
            const response = await this.client.get(`/moviefile/${fileId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get movie file: ${error.message}`);
        }
    }

    /**
     * Delete movie file
     */
    async deleteMovieFile(fileId) {
        try {
            await this.client.delete(`/moviefile/${fileId}`);
            this.emit('movieFileDeleted', fileId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete movie file: ${error.message}`);
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
     * Get upcoming movies
     */
    async getUpcoming(days = 30) {
        try {
            const startDate = new Date().toISOString();
            const endDate = new Date(Date.now() + (days * 24 * 60 * 60 * 1000)).toISOString();
            
            return await this.getCalendar(startDate, endDate);
        } catch (error) {
            throw new Error(`Failed to get upcoming movies: ${error.message}`);
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
                includeUnknownMovieItems: true
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
    async getHistory(page = 1, pageSize = 20, sortKey = 'date', sortDirection = 'descending', movieId = null, eventType = null) {
        try {
            const params = {
                page,
                pageSize,
                sortKey,
                sortDirection
            };
            
            if (movieId) params.movieId = movieId;
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
     * Trigger search for movie
     */
    async searchMovie(movieId) {
        try {
            const command = {
                name: 'MoviesSearch',
                movieIds: [movieId]
            };
            
            const response = await this.client.post('/command', command);
            this.emit('searchTriggered', { type: 'movie', movieId, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to trigger movie search: ${error.message}`);
        }
    }

    /**
     * Refresh movie
     */
    async refreshMovie(movieId = null) {
        try {
            const command = movieId ? 
                { name: 'RefreshMovie', movieIds: [movieId] } :
                { name: 'RefreshMovie' };
            
            const response = await this.client.post('/command', command);
            this.emit('refreshTriggered', { movieId, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to refresh movie: ${error.message}`);
        }
    }

    /**
     * Rescan movie
     */
    async rescanMovie(movieId = null) {
        try {
            const command = movieId ?
                { name: 'RescanMovie', movieIds: [movieId] } :
                { name: 'RescanMovie' };
            
            const response = await this.client.post('/command', command);
            this.emit('rescanTriggered', { movieId, commandId: response.data.id });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to rescan movie: ${error.message}`);
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
     * Get import lists
     */
    async getImportLists() {
        try {
            const response = await this.client.get('/importlist');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get import lists: ${error.message}`);
        }
    }

    /**
     * Get exclusions
     */
    async getExclusions() {
        try {
            const response = await this.client.get('/exclusions');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get exclusions: ${error.message}`);
        }
    }

    /**
     * Add exclusion
     */
    async addExclusion(tmdbId, title, year) {
        try {
            const exclusion = {
                tmdbId: tmdbId,
                movieTitle: title,
                movieYear: year
            };
            
            const response = await this.client.post('/exclusions', exclusion);
            this.emit('exclusionAdded', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add exclusion: ${error.message}`);
        }
    }

    /**
     * Delete exclusion
     */
    async deleteExclusion(exclusionId) {
        try {
            await this.client.delete(`/exclusions/${exclusionId}`);
            this.emit('exclusionDeleted', exclusionId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete exclusion: ${error.message}`);
        }
    }

    /**
     * Get movie credits
     */
    async getMovieCredits(movieId) {
        try {
            const response = await this.client.get(`/movie/${movieId}/credits`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get movie credits: ${error.message}`);
        }
    }

    /**
     * Get movie statistics
     */
    async getStatistics() {
        try {
            const movies = await this.getMovies();
            const queue = await this.getQueue();
            
            const stats = {
                totalMovies: movies.length,
                monitoredMovies: movies.filter(m => m.monitored).length,
                unmonitoredMovies: movies.filter(m => !m.monitored).length,
                downloadedMovies: movies.filter(m => m.hasFile).length,
                missingMovies: movies.filter(m => m.monitored && !m.hasFile).length,
                queuedDownloads: queue.totalRecords || 0,
                diskSpace: await this._getDiskSpace(),
                moviesByYear: this._groupMoviesByYear(movies),
                moviesByStudio: this._groupMoviesByStudio(movies)
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
     * Group movies by year
     */
    _groupMoviesByYear(movies) {
        const grouped = {};
        movies.forEach(movie => {
            const year = movie.year || 'Unknown';
            if (!grouped[year]) grouped[year] = 0;
            grouped[year]++;
        });
        return grouped;
    }

    /**
     * Group movies by studio
     */
    _groupMoviesByStudio(movies) {
        const grouped = {};
        movies.forEach(movie => {
            const studio = movie.studio || 'Unknown';
            if (!grouped[studio]) grouped[studio] = 0;
            grouped[studio]++;
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
    setupWebhook(app, path = '/radarr/webhook') {
        app.post(path, (req, res) => {
            try {
                const event = req.body;
                this.emit('webhook', event);
                
                // Emit specific events based on event type
                switch (event.eventType) {
                    case 'Grab':
                        this.emit('movieGrabbed', event);
                        break;
                    case 'Download':
                        this.emit('movieDownloaded', event);
                        break;
                    case 'Rename':
                        this.emit('movieRenamed', event);
                        break;
                    case 'MovieFileDelete':
                        this.emit('movieFileDeleted', event);
                        break;
                    case 'MovieDelete':
                        this.emit('movieDeleted', event);
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
                console.error('Radarr webhook error:', error);
                res.status(500).json({ error: 'Webhook processing failed' });
            }
        });
    }
}

module.exports = RadarrIntegration;