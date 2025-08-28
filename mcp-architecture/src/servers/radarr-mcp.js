/**
 * Radarr MCP Server
 * Provides MCP interface for Radarr movie management API
 */

const { McpServer } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { ListResourcesRequestSchema, ReadResourceRequestSchema, ListToolsRequestSchema, CallToolRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const axios = require('axios');

class RadarrMCP {
  constructor(options = {}) {
    this.port = options.port || 3003;
    this.radarrUrl = options.radarrUrl || process.env.RADARR_URL || 'http://localhost:7878';
    this.apiKey = options.apiKey || process.env.RADARR_API_KEY;
    this.io = options.io; // Socket.io instance for real-time updates
    
    this.server = new McpServer(
      {
        name: 'radarr-mcp',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupTools();
    this.setupResources();
    this.setupRequestHandlers();
  }

  setupTools() {
    // Movie Search
    this.server.tool('search_movies', {
      description: 'Search for movies in Radarr',
      parameters: {
        type: 'object',
        properties: {
          term: {
            type: 'string',
            description: 'Search term for movies'
          }
        },
        required: ['term']
      }
    }, this.searchMovies.bind(this));

    // Get Movie Info
    this.server.tool('get_movie', {
      description: 'Get detailed information about a specific movie',
      parameters: {
        type: 'object',
        properties: {
          movieId: {
            type: 'number',
            description: 'Radarr movie ID'
          }
        },
        required: ['movieId']
      }
    }, this.getMovie.bind(this));

    // Monitor Movie
    this.server.tool('monitor_movie', {
      description: 'Set monitoring status for a movie',
      parameters: {
        type: 'object',
        properties: {
          movieId: {
            type: 'number',
            description: 'Radarr movie ID'
          },
          monitored: {
            type: 'boolean',
            description: 'Whether to monitor the movie'
          }
        },
        required: ['movieId', 'monitored']
      }
    }, this.monitorMovie.bind(this));

    // Add Movie
    this.server.tool('add_movie', {
      description: 'Add a new movie to Radarr',
      parameters: {
        type: 'object',
        properties: {
          tmdbId: {
            type: 'number',
            description: 'TMDB ID of the movie'
          },
          qualityProfileId: {
            type: 'number',
            description: 'Quality profile ID'
          },
          rootFolderPath: {
            type: 'string',
            description: 'Root folder path for the movie'
          },
          monitored: {
            type: 'boolean',
            description: 'Whether to monitor the movie (default: true)'
          },
          searchForMovie: {
            type: 'boolean',
            description: 'Whether to search for the movie immediately (default: true)'
          }
        },
        required: ['tmdbId', 'qualityProfileId', 'rootFolderPath']
      }
    }, this.addMovie.bind(this));

    // Get Queue
    this.server.tool('get_queue', {
      description: 'Get current download queue',
      parameters: {
        type: 'object',
        properties: {
          page: {
            type: 'number',
            description: 'Page number (default: 1)'
          },
          pageSize: {
            type: 'number',
            description: 'Items per page (default: 20)'
          }
        }
      }
    }, this.getQueue.bind(this));

    // Get Calendar
    this.server.tool('get_calendar', {
      description: 'Get movies releasing in specified date range',
      parameters: {
        type: 'object',
        properties: {
          start: {
            type: 'string',
            description: 'Start date (YYYY-MM-DD format)'
          },
          end: {
            type: 'string',
            description: 'End date (YYYY-MM-DD format)'
          }
        }
      }
    }, this.getCalendar.bind(this));

    // Search Movie
    this.server.tool('search_movie', {
      description: 'Trigger a search for a specific movie',
      parameters: {
        type: 'object',
        properties: {
          movieId: {
            type: 'number',
            description: 'Radarr movie ID'
          }
        },
        required: ['movieId']
      }
    }, this.searchMovie.bind(this));

    // Get System Status
    this.server.tool('get_system_status', {
      description: 'Get Radarr system status and information',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getSystemStatus.bind(this));

    // Get Missing Movies
    this.server.tool('get_missing_movies', {
      description: 'Get list of missing/wanted movies',
      parameters: {
        type: 'object',
        properties: {
          page: {
            type: 'number',
            description: 'Page number (default: 1)'
          },
          pageSize: {
            type: 'number',
            description: 'Items per page (default: 20)'
          }
        }
      }
    }, this.getMissingMovies.bind(this));

    // Get Quality Profiles
    this.server.tool('get_quality_profiles', {
      description: 'Get available quality profiles',
      parameters: {
        type: 'object',
        properties: {}
      }
    }, this.getQualityProfiles.bind(this));
  }

  setupResources() {
    this.server.resource('radarr://movies', {
      description: 'All movies in Radarr',
      mimeType: 'application/json'
    });

    this.server.resource('radarr://queue', {
      description: 'Current download queue',
      mimeType: 'application/json'
    });

    this.server.resource('radarr://calendar', {
      description: 'Movie release calendar',
      mimeType: 'application/json'
    });

    this.server.resource('radarr://system', {
      description: 'System status and information',
      mimeType: 'application/json'
    });

    this.server.resource('radarr://missing', {
      description: 'Missing/wanted movies',
      mimeType: 'application/json'
    });
  }

  setupRequestHandlers() {
    this.server.request(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'radarr://movies',
            name: 'Movie Library',
            description: 'All movies managed by Radarr',
            mimeType: 'application/json'
          },
          {
            uri: 'radarr://queue',
            name: 'Download Queue',
            description: 'Current download queue status',
            mimeType: 'application/json'
          },
          {
            uri: 'radarr://calendar',
            name: 'Release Calendar',
            description: 'Movie release dates and calendar',
            mimeType: 'application/json'
          },
          {
            uri: 'radarr://system',
            name: 'System Status',
            description: 'Radarr system information and status',
            mimeType: 'application/json'
          },
          {
            uri: 'radarr://missing',
            name: 'Missing Movies',
            description: 'Movies that are wanted but not downloaded',
            mimeType: 'application/json'
          }
        ]
      };
    });

    this.server.request(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      try {
        switch (uri) {
          case 'radarr://movies':
            const movies = await this.apiRequest('/api/v3/movie');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(movies, null, 2)
              }]
            };

          case 'radarr://queue':
            const queue = await this.apiRequest('/api/v3/queue');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(queue, null, 2)
              }]
            };

          case 'radarr://calendar':
            const today = new Date().toISOString().split('T')[0];
            const nextMonth = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            const calendar = await this.apiRequest(`/api/v3/calendar?start=${today}&end=${nextMonth}`);
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(calendar, null, 2)
              }]
            };

          case 'radarr://system':
            const status = await this.apiRequest('/api/v3/system/status');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(status, null, 2)
              }]
            };

          case 'radarr://missing':
            const missing = await this.apiRequest('/api/v3/wanted/missing');
            return {
              contents: [{
                uri,
                mimeType: 'application/json',
                text: JSON.stringify(missing, null, 2)
              }]
            };

          default:
            throw new Error(`Unknown resource: ${uri}`);
        }
      } catch (error) {
        throw new Error(`Failed to read resource ${uri}: ${error.message}`);
      }
    });
  }

  async apiRequest(endpoint, method = 'GET', data = null) {
    try {
      const config = {
        method,
        url: `${this.radarrUrl}${endpoint}`,
        headers: {
          'X-Api-Key': this.apiKey,
          'Content-Type': 'application/json'
        }
      };

      if (data) {
        config.data = data;
      }

      const response = await axios(config);
      return response.data;
    } catch (error) {
      console.error(`Radarr API error (${method} ${endpoint}):`, error.message);
      throw new Error(`API request failed: ${error.response?.data?.message || error.message}`);
    }
  }

  // Tool implementations
  async searchMovies({ term }) {
    try {
      const results = await this.apiRequest(`/api/v3/movie/lookup?term=${encodeURIComponent(term)}`);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('radarr-search', { term, results: results.length });
      }

      return {
        content: [{
          type: 'text',
          text: `Found ${results.length} movies matching "${term}":
${results.slice(0, 10).map(movie => 
  `• ${movie.title} (${movie.year})
    Runtime: ${movie.runtime || 'Unknown'} min | Rating: ${movie.ratings?.value ? `${movie.ratings.value}/10` : 'Not rated'}
    Genres: ${movie.genres?.join(', ') || 'Unknown'}
    TMDB ID: ${movie.tmdbId}`
).join('\n')}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error searching movies: ${error.message}`
        }]
      };
    }
  }

  async getMovie({ movieId }) {
    try {
      const movie = await this.apiRequest(`/api/v3/movie/${movieId}`);
      
      return {
        content: [{
          type: 'text',
          text: `🎬 ${movie.title} (${movie.year})

Rating: ${movie.ratings?.value ? `${movie.ratings.value}/10` : 'Not rated'}
Runtime: ${movie.runtime || 'Unknown'} minutes
Genres: ${movie.genres?.join(', ') || 'Unknown'}
Studio: ${movie.studio || 'Unknown'}

📊 Status:
• Monitored: ${movie.monitored ? 'Yes' : 'No'}
• Downloaded: ${movie.hasFile ? 'Yes' : 'No'}
• Available: ${movie.isAvailable ? 'Yes' : 'No'}
• Quality Profile: ${movie.qualityProfileId}

📅 Dates:
• Release Date: ${movie.physicalRelease || movie.digitalRelease || movie.inCinemas || 'Unknown'}
• Added: ${new Date(movie.added).toLocaleDateString()}

📍 Path: ${movie.path}
🔗 TMDB ID: ${movie.tmdbId}
🔗 IMDB ID: ${movie.imdbId || 'Not available'}

📝 Overview:
${movie.overview || 'No overview available'}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting movie information: ${error.message}`
        }]
      };
    }
  }

  async monitorMovie({ movieId, monitored }) {
    try {
      const movie = await this.apiRequest(`/api/v3/movie/${movieId}`);
      movie.monitored = monitored;
      
      await this.apiRequest(`/api/v3/movie/${movieId}`, 'PUT', movie);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('radarr-monitor-change', { movieId, monitored, title: movie.title });
      }

      return {
        content: [{
          type: 'text',
          text: `✅ Successfully ${monitored ? 'enabled' : 'disabled'} monitoring for "${movie.title}"`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error updating monitoring status: ${error.message}`
        }]
      };
    }
  }

  async addMovie({ tmdbId, qualityProfileId, rootFolderPath, monitored = true, searchForMovie = true }) {
    try {
      // First, lookup the movie details
      const lookupResults = await this.apiRequest(`/api/v3/movie/lookup/tmdb?tmdbId=${tmdbId}`);
      
      if (!lookupResults || lookupResults.length === 0) {
        throw new Error(`Movie with TMDB ID ${tmdbId} not found`);
      }

      const movieToAdd = lookupResults[0];
      movieToAdd.qualityProfileId = qualityProfileId;
      movieToAdd.rootFolderPath = rootFolderPath;
      movieToAdd.monitored = monitored;
      movieToAdd.searchForMovie = searchForMovie;
      movieToAdd.addOptions = {
        searchForMovie: searchForMovie
      };

      const addedMovie = await this.apiRequest('/api/v3/movie', 'POST', movieToAdd);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('radarr-movie-added', { 
          movieId: addedMovie.id, 
          title: addedMovie.title,
          monitored,
          searchForMovie
        });
      }

      return {
        content: [{
          type: 'text',
          text: `✅ Successfully added "${addedMovie.title}" (${addedMovie.year}) to Radarr
${monitored ? '👀 Monitoring enabled' : '😴 Monitoring disabled'}
${searchForMovie ? '🔍 Search triggered automatically' : '⏸️ No automatic search'}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error adding movie: ${error.message}`
        }]
      };
    }
  }

  async getQueue({ page = 1, pageSize = 20 }) {
    try {
      const queue = await this.apiRequest(`/api/v3/queue?page=${page}&pageSize=${pageSize}`);
      
      if (!queue.records || queue.records.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '📭 Download queue is empty'
          }]
        };
      }

      const queueText = queue.records.map(item => {
        const progress = item.size > 0 ? ((item.size - item.sizeleft) / item.size * 100).toFixed(1) : 0;
        return `• ${item.movie?.title} (${item.movie?.year})
  Status: ${item.status} | Progress: ${progress}%
  Quality: ${item.quality?.quality?.name} | Size: ${this.formatBytes(item.size)}
  ETA: ${item.timeleft || 'Unknown'}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `📥 Download Queue (Page ${page}/${Math.ceil(queue.totalRecords / pageSize)}):

${queueText}

Total Items: ${queue.totalRecords}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting queue: ${error.message}`
        }]
      };
    }
  }

  async getCalendar({ start, end }) {
    try {
      const today = start || new Date().toISOString().split('T')[0];
      const nextMonth = end || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
      
      const calendar = await this.apiRequest(`/api/v3/calendar?start=${today}&end=${nextMonth}`);
      
      if (calendar.length === 0) {
        return {
          content: [{
            type: 'text',
            text: `📅 No movies releasing between ${today} and ${nextMonth}`
          }]
        };
      }

      const groupedByDate = calendar.reduce((acc, movie) => {
        const releaseDate = movie.physicalRelease || movie.digitalRelease || movie.inCinemas;
        if (releaseDate) {
          const date = releaseDate.split('T')[0];
          if (!acc[date]) acc[date] = [];
          acc[date].push(movie);
        }
        return acc;
      }, {});

      const calendarText = Object.entries(groupedByDate)
        .sort(([a], [b]) => new Date(a) - new Date(b))
        .map(([date, movies]) => {
          const formattedDate = new Date(date).toLocaleDateString();
          const moviesList = movies.map(movie => 
            `  • ${movie.title} (${movie.year})
    ${movie.hasFile ? '✅ Downloaded' : '⏳ Not Downloaded'} | Rating: ${movie.ratings?.value ? `${movie.ratings.value}/10` : 'Not rated'}`
          ).join('\n');
          
          return `📅 ${formattedDate}:\n${moviesList}`;
        }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `🎬 Movie Releases (${today} to ${nextMonth}):

${calendarText}

Total Movies: ${calendar.length}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting calendar: ${error.message}`
        }]
      };
    }
  }

  async searchMovie({ movieId }) {
    try {
      await this.apiRequest('/api/v3/command', 'POST', {
        name: 'MoviesSearch',
        movieIds: [movieId]
      });

      const movie = await this.apiRequest(`/api/v3/movie/${movieId}`);
      
      // Emit real-time update
      if (this.io) {
        this.io.emit('radarr-search-movie', { movieId, title: movie.title });
      }

      return {
        content: [{
          type: 'text',
          text: `🔍 Movie search triggered for "${movie.title}" (${movie.year}). Check the queue for progress.`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error triggering movie search: ${error.message}`
        }]
      };
    }
  }

  async getSystemStatus() {
    try {
      const status = await this.apiRequest('/api/v3/system/status');
      const diskSpace = await this.apiRequest('/api/v3/diskspace');
      
      return {
        content: [{
          type: 'text',
          text: `🖥️ Radarr System Status:

Version: ${status.version}
Build Date: ${new Date(status.buildTime).toLocaleDateString()}
Runtime Version: ${status.runtimeVersion}
Database Version: ${status.databaseVersion}

🔧 Configuration:
• Start Time: ${new Date(status.startTime).toLocaleString()}
• App Data: ${status.appData}
• OS Name: ${status.osName}
• Is Production: ${status.isProduction ? 'Yes' : 'No'}

💾 Disk Space:
${diskSpace.map(disk => 
  `• ${disk.label}: ${this.formatBytes(disk.freeSpace)} free / ${this.formatBytes(disk.totalSpace)} total (${((disk.totalSpace - disk.freeSpace) / disk.totalSpace * 100).toFixed(1)}% used)`
).join('\n')}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting system status: ${error.message}`
        }]
      };
    }
  }

  async getMissingMovies({ page = 1, pageSize = 20 }) {
    try {
      const missing = await this.apiRequest(`/api/v3/wanted/missing?page=${page}&pageSize=${pageSize}&sortKey=releaseDate&sortDirection=descending`);
      
      if (!missing.records || missing.records.length === 0) {
        return {
          content: [{
            type: 'text',
            text: '🎉 No missing movies! All monitored movies are downloaded.'
          }]
        };
      }

      const missingText = missing.records.map(movie => {
        const releaseDate = movie.physicalRelease || movie.digitalRelease || movie.inCinemas;
        return `• ${movie.title} (${movie.year})
  Release: ${releaseDate ? new Date(releaseDate).toLocaleDateString() : 'Unknown'}
  Rating: ${movie.ratings?.value ? `${movie.ratings.value}/10` : 'Not rated'}
  Quality Profile: ${movie.qualityProfile?.name || 'Unknown'}`;
      }).join('\n\n');

      return {
        content: [{
          type: 'text',
          text: `🎬 Missing Movies (Page ${page}/${Math.ceil(missing.totalRecords / pageSize)}):

${missingText}

Total Missing: ${missing.totalRecords}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting missing movies: ${error.message}`
        }]
      };
    }
  }

  async getQualityProfiles() {
    try {
      const profiles = await this.apiRequest('/api/v3/qualityprofile');
      
      return {
        content: [{
          type: 'text',
          text: `🎭 Available Quality Profiles:

${profiles.map(profile => 
  `• ID: ${profile.id} - ${profile.name}
    Cutoff: ${profile.cutoff?.name || 'Not set'}
    Qualities: ${profile.items?.map(item => item.quality?.name).filter(Boolean).join(', ') || 'Not configured'}`
).join('\n\n')}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error getting quality profiles: ${error.message}`
        }]
      };
    }
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.log(`Radarr MCP Server running on port ${this.port}`);
  }
}

module.exports = RadarrMCP;