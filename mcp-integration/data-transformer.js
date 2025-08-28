#!/usr/bin/env node

/**
 * MCP Data Transformer
 * Handles data transformation between different service formats
 */

class MCPDataTransformer {
  constructor() {
    this.transformers = new Map();
    this.validators = new Map();
    this.initializeTransformers();
  }

  initializeTransformers() {
    console.log('🔄 Initializing MCP Data Transformers...');

    // Sonarr transformers
    this.transformers.set('sonarr_series', {
      input: this.validateSonarrSeries.bind(this),
      output: this.transformSonarrSeries.bind(this)
    });

    // Jellyfin transformers  
    this.transformers.set('jellyfin_media', {
      input: this.validateJellyfinMedia.bind(this),
      output: this.transformJellyfinMedia.bind(this)
    });

    // Generic API response transformer
    this.transformers.set('api_response', {
      input: this.validateAPIResponse.bind(this),
      output: this.transformAPIResponse.bind(this)
    });

    console.log(`✅ Initialized ${this.transformers.size} data transformers`);
  }

  async transform(type, data, options = {}) {
    const transformer = this.transformers.get(type);
    if (!transformer) {
      throw new Error(`No transformer found for type: ${type}`);
    }

    try {
      // Validate input data
      const validatedData = await transformer.input(data);
      
      // Transform data
      const transformedData = await transformer.output(validatedData, options);
      
      console.log(`✅ Transformed ${type} data successfully`);
      return transformedData;
    } catch (error) {
      console.error(`❌ Transformation failed for ${type}:`, error.message);
      throw error;
    }
  }

  // Sonarr Series Validation
  async validateSonarrSeries(data) {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid Sonarr series data: must be an object');
    }

    const required = ['title', 'year', 'tvdbId'];
    for (const field of required) {
      if (!data[field]) {
        throw new Error(`Missing required field: ${field}`);
      }
    }

    return data;
  }

  // Sonarr Series Transformation
  async transformSonarrSeries(data, options) {
    return {
      id: data.id || null,
      title: data.title,
      originalTitle: data.originalTitle || data.title,
      year: data.year,
      status: data.status || 'continuing',
      network: data.network || 'Unknown',
      genres: data.genres || [],
      rating: data.rating || 0,
      seasons: data.seasons ? this.transformSeasons(data.seasons) : [],
      images: data.images ? this.transformImages(data.images) : [],
      externalIds: {
        tvdb: data.tvdbId,
        imdb: data.imdbId,
        tmdb: data.tmdbId
      },
      metadata: {
        source: 'sonarr',
        transformed: Date.now(),
        options
      }
    };
  }

  // Jellyfin Media Validation
  async validateJellyfinMedia(data) {
    if (!data || typeof data !== 'object') {
      throw new Error('Invalid Jellyfin media data: must be an object');
    }

    if (!data.Name && !data.title) {
      throw new Error('Missing required field: Name or title');
    }

    return data;
  }

  // Jellyfin Media Transformation
  async transformJellyfinMedia(data, options) {
    return {
      id: data.Id || data.id,
      name: data.Name || data.title,
      type: data.Type || 'Unknown',
      year: data.ProductionYear || data.year,
      overview: data.Overview || data.description,
      rating: data.CommunityRating || data.rating || 0,
      runtime: data.RunTimeTicks ? Math.round(data.RunTimeTicks / 600000000) : null,
      genres: data.Genres || data.genres || [],
      studios: data.Studios || [],
      people: data.People || [],
      images: data.ImageTags ? this.transformJellyfinImages(data.ImageTags, data.Id) : [],
      externalIds: {
        imdb: data.ProviderIds?.Imdb,
        tmdb: data.ProviderIds?.Tmdb,
        tvdb: data.ProviderIds?.Tvdb
      },
      metadata: {
        source: 'jellyfin',
        transformed: Date.now(),
        options
      }
    };
  }

  // Generic API Response Validation
  async validateAPIResponse(data) {
    if (!data) {
      throw new Error('API response is null or undefined');
    }

    // Check for common error indicators
    if (data.error || data.Error) {
      throw new Error(`API Error: ${data.error || data.Error}`);
    }

    return data;
  }

  // Generic API Response Transformation
  async transformAPIResponse(data, options) {
    const baseResponse = {
      success: true,
      data: data,
      metadata: {
        transformed: Date.now(),
        source: options.source || 'unknown',
        options
      }
    };

    // Handle paginated responses
    if (data.totalRecords || data.total || data.count) {
      baseResponse.pagination = {
        total: data.totalRecords || data.total || data.count,
        page: data.page || 1,
        pageSize: data.pageSize || data.records?.length || 0
      };
    }

    // Handle array responses
    if (Array.isArray(data)) {
      baseResponse.data = data;
      baseResponse.count = data.length;
    }

    return baseResponse;
  }

  // Helper: Transform Seasons
  transformSeasons(seasons) {
    return seasons.map(season => ({
      number: season.seasonNumber,
      episodeCount: season.statistics?.episodeCount || 0,
      availableEpisodes: season.statistics?.episodeFileCount || 0,
      monitored: season.monitored || false
    }));
  }

  // Helper: Transform Images
  transformImages(images) {
    return images.map(image => ({
      type: image.coverType || 'unknown',
      url: image.url || image.remoteUrl,
      height: image.height,
      width: image.width
    }));
  }

  // Helper: Transform Jellyfin Images
  transformJellyfinImages(imageTags, itemId) {
    const images = [];
    for (const [type, tag] of Object.entries(imageTags)) {
      images.push({
        type: type.toLowerCase(),
        url: `/Items/${itemId}/Images/${type}?tag=${tag}`,
        tag
      });
    }
    return images;
  }

  // Batch transformation
  async batchTransform(type, dataArray, options = {}) {
    if (!Array.isArray(dataArray)) {
      throw new Error('Batch transform requires an array of data');
    }

    const results = [];
    const errors = [];

    for (let i = 0; i < dataArray.length; i++) {
      try {
        const transformed = await this.transform(type, dataArray[i], options);
        results.push(transformed);
      } catch (error) {
        errors.push({ index: i, error: error.message });
      }
    }

    return {
      success: results.length,
      errors: errors.length,
      total: dataArray.length,
      results,
      errors
    };
  }

  // Get available transformers
  getAvailableTransformers() {
    return Array.from(this.transformers.keys());
  }
}

// Export for use in other modules
module.exports = MCPDataTransformer;

// CLI usage
if (require.main === module) {
  const transformer = new MCPDataTransformer();
  console.log('🔄 MCP Data Transformer ready!');
  console.log('📋 Available transformers:', transformer.getAvailableTransformers());
}