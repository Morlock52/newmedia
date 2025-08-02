/**
 * Media Library Service with Natural Language Interface
 * Provides conversational access to media content with AI-powered search and management
 */

import fs from 'fs/promises';
import path from 'path';
import { EventEmitter } from 'events';
import crypto from 'crypto';

export class MediaLibraryService extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      // Storage configuration
      mediaDirectory: config.mediaDirectory || './media',
      metadataDirectory: config.metadataDirectory || './data/metadata',
      thumbnailDirectory: config.thumbnailDirectory || './data/thumbnails',
      
      // Supported formats
      supportedVideoFormats: config.supportedVideoFormats || ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'],
      supportedAudioFormats: config.supportedAudioFormats || ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
      supportedImageFormats: config.supportedImageFormats || ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'],
      supportedDocumentFormats: config.supportedDocumentFormats || ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf'],
      
      // Indexing and search
      enableFullTextSearch: config.enableFullTextSearch !== false,
      enableMetadataExtraction: config.enableMetadataExtraction !== false,
      enableThumbnailGeneration: config.enableThumbnailGeneration !== false,
      indexingBatchSize: config.indexingBatchSize || 100,
      
      // AI-powered features
      enableAIDescriptions: config.enableAIDescriptions !== false,
      enableAutoTagging: config.enableAutoTagging !== false,
      enableContentAnalysis: config.enableContentAnalysis !== false,
      enableContentSummarization: config.enableContentSummarization !== false,
      
      // Search and recommendation settings
      searchResultLimit: config.searchResultLimit || 50,
      enableFuzzySearch: config.enableFuzzySearch !== false,
      enableSemanticSearch: config.enableSemanticSearch !== false,
      
      ...config
    };

    // Media index and metadata
    this.mediaIndex = new Map();
    this.searchIndex = new Map();
    this.tagIndex = new Map();
    this.metadataCache = new Map();
    
    // Conversation context for natural language interface
    this.conversationContexts = new Map();
    this.userPreferences = new Map();
    
    // File system watchers
    this.watchers = new Map();
    
    this.isInitialized = false;
  }

  /**
   * Initialize the media library service
   */
  async initialize() {
    try {
      console.log('Initializing Media Library Service...');
      
      // Create required directories
      await this.ensureDirectories();
      
      // Load existing metadata
      await this.loadMetadataCache();
      
      // Build media index
      await this.buildMediaIndex();
      
      // Setup file system watchers
      await this.setupFileWatchers();
      
      // Initialize search indices
      await this.buildSearchIndices();
      
      this.isInitialized = true;
      console.log(`Media Library Service initialized with ${this.mediaIndex.size} media files`);
      
      return true;
    } catch (error) {
      console.error('Failed to initialize Media Library Service:', error);
      throw error;
    }
  }

  /**
   * Ensure required directories exist
   */
  async ensureDirectories() {
    const directories = [
      this.config.mediaDirectory,
      this.config.metadataDirectory,
      this.config.thumbnailDirectory
    ];

    for (const dir of directories) {
      try {
        await fs.access(dir);
      } catch {
        await fs.mkdir(dir, { recursive: true });
      }
    }
  }

  /**
   * Load existing metadata cache
   */
  async loadMetadataCache() {
    try {
      const cacheFile = path.join(this.config.metadataDirectory, 'metadata_cache.json');
      const cacheData = await fs.readFile(cacheFile, 'utf8');
      const cache = JSON.parse(cacheData);
      
      Object.entries(cache).forEach(([key, value]) => {
        this.metadataCache.set(key, value);
      });
      
      console.log(`Loaded ${this.metadataCache.size} metadata entries from cache`);
    } catch (error) {
      console.log('No existing metadata cache found, starting fresh');
    }
  }

  /**
   * Build comprehensive media index
   */
  async buildMediaIndex() {
    const mediaFiles = await this.scanMediaDirectory();
    
    for (const filePath of mediaFiles) {
      try {
        await this.indexMediaFile(filePath);
      } catch (error) {
        console.warn(`Failed to index ${filePath}:`, error.message);
      }
    }
    
    // Save updated metadata cache
    await this.saveMetadataCache();
  }

  /**
   * Scan media directory for supported files
   */
  async scanMediaDirectory(directory = this.config.mediaDirectory) {
    const mediaFiles = [];
    
    try {
      const entries = await fs.readdir(directory, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(directory, entry.name);
        
        if (entry.isDirectory()) {
          // Recursively scan subdirectories
          const subFiles = await this.scanMediaDirectory(fullPath);
          mediaFiles.push(...subFiles);
        } else if (entry.isFile()) {
          const ext = path.extname(entry.name).toLowerCase();
          const allFormats = [
            ...this.config.supportedVideoFormats,
            ...this.config.supportedAudioFormats,
            ...this.config.supportedImageFormats,
            ...this.config.supportedDocumentFormats
          ];
          
          if (allFormats.includes(ext)) {
            mediaFiles.push(fullPath);
          }
        }
      }
    } catch (error) {
      console.error(`Error scanning directory ${directory}:`, error.message);
    }
    
    return mediaFiles;
  }

  /**
   * Index a single media file
   */
  async indexMediaFile(filePath) {
    const stats = await fs.stat(filePath);
    const ext = path.extname(filePath).toLowerCase();
    const basename = path.basename(filePath, ext);
    const mediaId = this.generateMediaId(filePath);
    
    // Determine media type
    const mediaType = this.determineMediaType(ext);
    
    // Get or extract metadata
    let metadata = this.metadataCache.get(filePath);
    if (!metadata || metadata.modifiedTime !== stats.mtime.getTime()) {
      metadata = await this.extractMetadata(filePath, mediaType);
      metadata.modifiedTime = stats.mtime.getTime();
      this.metadataCache.set(filePath, metadata);
    }

    // Create media entry
    const mediaEntry = {
      id: mediaId,
      filePath,
      fileName: path.basename(filePath),
      baseName: basename,
      extension: ext,
      mediaType,
      size: stats.size,
      createdAt: stats.birthtime,
      modifiedAt: stats.mtime,
      indexedAt: new Date(),
      metadata,
      tags: metadata.tags || [],
      description: metadata.description || '',
      thumbnail: metadata.thumbnail || null
    };

    // Add to main index
    this.mediaIndex.set(mediaId, mediaEntry);
    
    // Update search indices
    this.updateSearchIndices(mediaEntry);
    
    this.emit('mediaIndexed', mediaEntry);
    
    return mediaEntry;
  }

  /**
   * Determine media type from file extension
   */
  determineMediaType(extension) {
    if (this.config.supportedVideoFormats.includes(extension)) return 'video';
    if (this.config.supportedAudioFormats.includes(extension)) return 'audio';
    if (this.config.supportedImageFormats.includes(extension)) return 'image';
    if (this.config.supportedDocumentFormats.includes(extension)) return 'document';
    return 'unknown';
  }

  /**
   * Extract metadata from media file
   */
  async extractMetadata(filePath, mediaType) {
    const metadata = {
      extracted: true,
      extractedAt: new Date().toISOString()
    };

    try {
      // Basic file information
      const stats = await fs.stat(filePath);
      metadata.fileSize = stats.size;
      metadata.createdDate = stats.birthtime.toISOString();
      metadata.modifiedDate = stats.mtime.toISOString();

      // Type-specific metadata extraction
      switch (mediaType) {
        case 'video':
          metadata = { ...metadata, ...(await this.extractVideoMetadata(filePath)) };
          break;
        case 'audio':
          metadata = { ...metadata, ...(await this.extractAudioMetadata(filePath)) };
          break;
        case 'image':
          metadata = { ...metadata, ...(await this.extractImageMetadata(filePath)) };
          break;
        case 'document':
          metadata = { ...metadata, ...(await this.extractDocumentMetadata(filePath)) };
          break;
      }

      // Generate AI description if enabled
      if (this.config.enableAIDescriptions) {
        metadata.aiDescription = await this.generateAIDescription(filePath, mediaType, metadata);
      }

      // Auto-generate tags if enabled
      if (this.config.enableAutoTagging) {
        metadata.autoTags = await this.generateAutoTags(filePath, mediaType, metadata);
        metadata.tags = [...(metadata.tags || []), ...(metadata.autoTags || [])];
      }

      // Generate thumbnail if enabled
      if (this.config.enableThumbnailGeneration && ['video', 'image'].includes(mediaType)) {
        metadata.thumbnail = await this.generateThumbnail(filePath, mediaType);
      }

    } catch (error) {
      console.warn(`Metadata extraction failed for ${filePath}:`, error.message);
      metadata.extractionError = error.message;
    }

    return metadata;
  }

  /**
   * Extract video metadata (simplified - in production use ffprobe)
   */
  async extractVideoMetadata(filePath) {
    // In production, use ffprobe or similar tool
    // For now, return basic metadata structure
    return {
      duration: null,
      width: null,
      height: null,
      frameRate: null,
      bitRate: null,
      codec: null,
      audioTracks: [],
      subtitleTracks: []
    };
  }

  /**
   * Extract audio metadata (simplified - in production use music metadata libraries)
   */
  async extractAudioMetadata(filePath) {
    // In production, use node-id3, music-metadata, or similar
    return {
      duration: null,
      artist: null,
      album: null,
      title: null,
      genre: null,
      year: null,
      track: null,
      bitRate: null,
      sampleRate: null
    };
  }

  /**
   * Extract image metadata (simplified - in production use exif libraries)
   */
  async extractImageMetadata(filePath) {
    // In production, use exif-parser, piexifjs, or similar
    return {
      width: null,
      height: null,
      colorSpace: null,
      camera: null,
      lens: null,
      exposureTime: null,
      fNumber: null,
      iso: null,
      gpsLocation: null
    };
  }

  /**
   * Extract document metadata (simplified)
   */
  async extractDocumentMetadata(filePath) {
    try {
      // For text files, extract first few lines as preview
      const ext = path.extname(filePath).toLowerCase();
      if (['.txt', '.md'].includes(ext)) {
        const content = await fs.readFile(filePath, 'utf8');
        const lines = content.split('\n').slice(0, 5);
        return {
          preview: lines.join('\n'),
          wordCount: content.split(/\s+/).length,
          lineCount: content.split('\n').length,
          characterCount: content.length
        };
      }
    } catch (error) {
      console.warn(`Document metadata extraction failed for ${filePath}:`, error.message);
    }

    return {
      pageCount: null,
      author: null,
      title: null,
      creationDate: null,
      modificationDate: null
    };
  }

  /**
   * Generate AI description for media (placeholder)
   */
  async generateAIDescription(filePath, mediaType, metadata) {
    // In production, integrate with vision/audio analysis APIs
    const basename = path.basename(filePath, path.extname(filePath));
    
    // Generate basic description based on filename and metadata
    let description = `${mediaType} file`;
    
    if (basename.includes('_') || basename.includes('-')) {
      const parts = basename.split(/[_-]/).filter(p => p.length > 2);
      if (parts.length > 0) {
        description += ` related to ${parts.join(', ')}`;
      }
    }
    
    if (metadata.duration) {
      const minutes = Math.floor(metadata.duration / 60);
      description += `, duration: ${minutes} minutes`;
    }
    
    if (metadata.width && metadata.height) {
      description += `, resolution: ${metadata.width}x${metadata.height}`;
    }
    
    return description;
  }

  /**
   * Generate auto tags (placeholder)
   */
  async generateAutoTags(filePath, mediaType, metadata) {
    const tags = [mediaType];
    const basename = path.basename(filePath, path.extname(filePath)).toLowerCase();
    
    // Extract potential tags from filename
    const words = basename.split(/[_\-\s]+/).filter(word => word.length > 2);
    tags.push(...words);
    
    // Add metadata-based tags
    if (metadata.artist) tags.push(metadata.artist.toLowerCase());
    if (metadata.album) tags.push(metadata.album.toLowerCase());
    if (metadata.genre) tags.push(metadata.genre.toLowerCase());
    
    // Add date-based tags
    if (metadata.year) tags.push(metadata.year.toString());
    
    // Remove duplicates and return
    return [...new Set(tags)];
  }

  /**
   * Generate thumbnail (placeholder)
   */
  async generateThumbnail(filePath, mediaType) {
    // In production, use ffmpeg for video thumbnails and image processing libraries for images
    const mediaId = this.generateMediaId(filePath);
    const thumbnailPath = path.join(this.config.thumbnailDirectory, `${mediaId}.jpg`);
    
    // For now, just return the expected thumbnail path
    return thumbnailPath;
  }

  /**
   * Update search indices
   */
  updateSearchIndices(mediaEntry) {
    const { id, fileName, baseName, tags, description, metadata } = mediaEntry;
    
    // Full-text search index
    if (this.config.enableFullTextSearch) {
      const searchableText = [
        fileName,
        baseName,
        description,
        ...(tags || []),
        metadata.title,
        metadata.artist,
        metadata.album,
        metadata.author
      ].filter(Boolean).join(' ').toLowerCase();
      
      // Simple word-based indexing
      const words = searchableText.split(/\s+/).filter(word => word.length > 2);
      words.forEach(word => {
        if (!this.searchIndex.has(word)) {
          this.searchIndex.set(word, new Set());
        }
        this.searchIndex.get(word).add(id);
      });
    }
    
    // Tag index
    (tags || []).forEach(tag => {
      if (!this.tagIndex.has(tag)) {
        this.tagIndex.set(tag, new Set());
      }
      this.tagIndex.get(tag).add(id);
    });
  }

  /**
   * Build search indices from existing media
   */
  async buildSearchIndices() {
    this.searchIndex.clear();
    this.tagIndex.clear();
    
    for (const mediaEntry of this.mediaIndex.values()) {
      this.updateSearchIndices(mediaEntry);
    }
    
    console.log(`Built search indices: ${this.searchIndex.size} terms, ${this.tagIndex.size} tags`);
  }

  /**
   * Natural language search interface
   */
  async search(options) {
    const {
      query,
      type = null,
      language = 'en',
      userId = null,
      contextId = null,
      limit = this.config.searchResultLimit
    } = options;

    if (!query) {
      throw new Error('Search query is required');
    }

    try {
      // Parse natural language query
      const parsedQuery = await this.parseSearchQuery(query, { type, language, userId });
      
      // Execute search based on parsed criteria
      const results = await this.executeSearch(parsedQuery, limit);
      
      // Sort and rank results
      const rankedResults = this.rankSearchResults(results, parsedQuery, userId);
      
      // Update conversation context
      if (contextId) {
        this.updateConversationContext(contextId, {
          query,
          parsedQuery,
          results: rankedResults.slice(0, 10), // Store top 10 for context
          timestamp: Date.now()
        });
      }
      
      this.emit('searchCompleted', {
        query,
        resultCount: rankedResults.length,
        userId,
        contextId
      });
      
      return {
        query,
        parsedQuery,
        results: rankedResults.slice(0, limit),
        totalResults: rankedResults.length,
        searchTime: Date.now()
      };

    } catch (error) {
      console.error('Search failed:', error);
      this.emit('searchError', { query, error, userId });
      throw error;
    }
  }

  /**
   * Parse natural language search query
   */
  async parseSearchQuery(query, options = {}) {
    const lowerQuery = query.toLowerCase();
    const parsed = {
      originalQuery: query,
      searchTerms: [],
      mediaTypes: [],
      timeRange: null,
      tags: [],
      metadata: {},
      sortBy: 'relevance',
      filters: {}
    };

    // Extract media type specifications
    const typeKeywords = {
      video: ['video', 'movie', 'film', 'clip'],
      audio: ['audio', 'music', 'song', 'sound', 'track'],
      image: ['image', 'photo', 'picture', 'pic'],
      document: ['document', 'doc', 'file', 'text', 'pdf']
    };

    Object.entries(typeKeywords).forEach(([type, keywords]) => {
      if (keywords.some(keyword => lowerQuery.includes(keyword))) {
        parsed.mediaTypes.push(type);
      }
    });

    // If specific type requested in options, use it
    if (options.type && !parsed.mediaTypes.includes(options.type)) {
      parsed.mediaTypes.push(options.type);
    }

    // Extract time-related queries
    const timePatterns = {
      today: /today|recent/,
      thisWeek: /this week|past week/,
      thisMonth: /this month|past month/,
      thisYear: /this year|past year/,
      lastYear: /last year/
    };

    Object.entries(timePatterns).forEach(([period, pattern]) => {
      if (pattern.test(lowerQuery)) {
        parsed.timeRange = period;
      }
    });

    // Extract sort preferences
    const sortPatterns = {
      newest: /newest|recent|latest/,
      oldest: /oldest|first/,
      largest: /largest|biggest/,
      smallest: /smallest/,
      name: /alphabetical|name/,
      duration: /longest|shortest|duration/
    };

    Object.entries(sortPatterns).forEach(([sort, pattern]) => {
      if (pattern.test(lowerQuery)) {
        parsed.sortBy = sort;
      }
    });

    // Extract metadata-specific queries
    if (lowerQuery.includes('by ')) {
      const artistMatch = lowerQuery.match(/by ([a-zA-Z\s]+)/);
      if (artistMatch) {
        parsed.metadata.artist = artistMatch[1].trim();
      }
    }

    // Extract basic search terms (remove command words)
    const commandWords = [
      'find', 'search', 'show', 'get', 'play', 'open', 'display',
      'video', 'audio', 'image', 'document', 'file',
      'by', 'from', 'in', 'with', 'of', 'the', 'a', 'an'
    ];

    const words = lowerQuery.split(/\s+/)
      .filter(word => word.length > 2 && !commandWords.includes(word));

    parsed.searchTerms = words;

    return parsed;
  }

  /**
   * Execute structured search
   */
  async executeSearch(parsedQuery, limit) {
    let results = new Set();
    
    // Search by terms
    if (parsedQuery.searchTerms.length > 0) {
      const termResults = this.searchByTerms(parsedQuery.searchTerms);
      termResults.forEach(id => results.add(id));
    } else {
      // If no specific terms, include all media
      this.mediaIndex.forEach((_, id) => results.add(id));
    }

    // Filter by media type
    if (parsedQuery.mediaTypes.length > 0) {
      results = new Set([...results].filter(id => {
        const media = this.mediaIndex.get(id);
        return media && parsedQuery.mediaTypes.includes(media.mediaType);
      }));
    }

    // Filter by time range
    if (parsedQuery.timeRange) {
      results = new Set([...results].filter(id => {
        const media = this.mediaIndex.get(id);
        return media && this.matchesTimeRange(media, parsedQuery.timeRange);
      }));
    }

    // Filter by metadata
    if (Object.keys(parsedQuery.metadata).length > 0) {
      results = new Set([...results].filter(id => {
        const media = this.mediaIndex.get(id);
        return media && this.matchesMetadata(media, parsedQuery.metadata);
      }));
    }

    // Convert to media objects
    return [...results].map(id => this.mediaIndex.get(id)).filter(Boolean);
  }

  /**
   * Search by terms using index
   */
  searchByTerms(terms) {
    const results = new Set();
    
    terms.forEach(term => {
      // Exact match
      if (this.searchIndex.has(term)) {
        this.searchIndex.get(term).forEach(id => results.add(id));
      }
      
      // Fuzzy search if enabled
      if (this.config.enableFuzzySearch) {
        this.searchIndex.forEach((ids, indexedTerm) => {
          if (this.isFuzzyMatch(term, indexedTerm)) {
            ids.forEach(id => results.add(id));
          }
        });
      }
    });
    
    return results;
  }

  /**
   * Check if term matches using fuzzy logic
   */
  isFuzzyMatch(term, indexedTerm) {
    if (term === indexedTerm) return true;
    if (indexedTerm.includes(term) || term.includes(indexedTerm)) return true;
    
    // Simple Levenshtein distance check
    const distance = this.levenshteinDistance(term, indexedTerm);
    const maxDistance = Math.floor(Math.max(term.length, indexedTerm.length) * 0.3);
    
    return distance <= maxDistance;
  }

  /**
   * Calculate Levenshtein distance
   */
  levenshteinDistance(str1, str2) {
    const matrix = [];
    
    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  /**
   * Check if media matches time range
   */
  matchesTimeRange(media, timeRange) {
    const now = new Date();
    const mediaDate = new Date(media.modifiedAt);
    
    switch (timeRange) {
      case 'today':
        return mediaDate.toDateString() === now.toDateString();
      case 'thisWeek':
        const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        return mediaDate >= weekAgo;
      case 'thisMonth':
        return mediaDate.getMonth() === now.getMonth() && mediaDate.getFullYear() === now.getFullYear();
      case 'thisYear':
        return mediaDate.getFullYear() === now.getFullYear();
      case 'lastYear':
        return mediaDate.getFullYear() === now.getFullYear() - 1;
      default:
        return true;
    }
  }

  /**
   * Check if media matches metadata criteria
   */
  matchesMetadata(media, metadataCriteria) {
    return Object.entries(metadataCriteria).every(([key, value]) => {
      const mediaValue = media.metadata[key];
      if (!mediaValue) return false;
      
      return mediaValue.toLowerCase().includes(value.toLowerCase());
    });
  }

  /**
   * Rank search results by relevance
   */
  rankSearchResults(results, parsedQuery, userId) {
    return results.map(media => {
      let relevanceScore = 0;
      
      // Term relevance
      const searchableText = [
        media.fileName,
        media.baseName,
        media.description,
        ...(media.tags || [])
      ].join(' ').toLowerCase();
      
      parsedQuery.searchTerms.forEach(term => {
        const termCount = (searchableText.match(new RegExp(term, 'g')) || []).length;
        relevanceScore += termCount * 10;
        
        // Boost for exact filename matches
        if (media.fileName.toLowerCase().includes(term)) {
          relevanceScore += 50;
        }
      });
      
      // Metadata relevance
      if (parsedQuery.metadata.artist && media.metadata.artist) {
        if (media.metadata.artist.toLowerCase().includes(parsedQuery.metadata.artist.toLowerCase())) {
          relevanceScore += 100;
        }
      }
      
      // User preference boost (if available)
      if (userId) {
        const preferences = this.userPreferences.get(userId);
        if (preferences) {
          // Boost based on preferred media types
          if (preferences.preferredTypes && preferences.preferredTypes.includes(media.mediaType)) {
            relevanceScore += 20;
          }
          
          // Boost based on interaction history
          if (preferences.frequentTags) {
            media.tags?.forEach(tag => {
              if (preferences.frequentTags.includes(tag)) {
                relevanceScore += 10;
              }
            });
          }
        }
      }
      
      // Recency boost (newer files get slight boost)
      const daysSinceModified = (Date.now() - media.modifiedAt.getTime()) / (1000 * 60 * 60 * 24);
      relevanceScore += Math.max(0, 10 - daysSinceModified * 0.1);
      
      return {
        ...media,
        relevanceScore
      };
    }).sort((a, b) => {
      // Sort by relevance first, then by sort preference
      if (parsedQuery.sortBy === 'newest') {
        return b.modifiedAt - a.modifiedAt;
      } else if (parsedQuery.sortBy === 'oldest') {
        return a.modifiedAt - b.modifiedAt;
      } else if (parsedQuery.sortBy === 'largest') {
        return b.size - a.size;
      } else if (parsedQuery.sortBy === 'smallest') {
        return a.size - b.size;
      } else if (parsedQuery.sortBy === 'name') {
        return a.fileName.localeCompare(b.fileName);
      } else {
        // Default: sort by relevance
        return b.relevanceScore - a.relevanceScore;
      }
    });
  }

  /**
   * Conversational search interface
   */
  async conversationalSearch(options) {
    const { message, userId, contextId, language = 'en' } = options;
    
    try {
      // Get conversation context
      const context = this.getConversationContext(contextId);
      
      // Parse conversational intent
      const intent = await this.parseConversationalIntent(message, context);
      
      let response;
      
      switch (intent.type) {
        case 'search':
          response = await this.handleSearchIntent(intent, userId, contextId);
          break;
        case 'filter':
          response = await this.handleFilterIntent(intent, context, userId);
          break;
        case 'play':
          response = await this.handlePlayIntent(intent, context, userId);
          break;
        case 'info':
          response = await this.handleInfoIntent(intent, context, userId);
          break;
        case 'help':
          response = this.handleHelpIntent(intent, language);
          break;
        default:
          response = this.handleUnknownIntent(message, language);
      }
      
      // Update conversation context
      this.updateConversationContext(contextId, {
        userMessage: message,
        intent,
        response,
        timestamp: Date.now()
      });
      
      return response;
      
    } catch (error) {
      console.error('Conversational search failed:', error);
      return {
        type: 'error',
        message: 'I encountered an error while processing your request. Please try again.',
        error: error.message
      };
    }
  }

  /**
   * Parse conversational intent from message
   */
  async parseConversationalIntent(message, context) {
    const lowerMessage = message.toLowerCase();
    
    // Intent patterns
    const patterns = {
      search: /find|search|show|get|look for|where is/,
      filter: /only|just|filter|exclude|without/,
      play: /play|start|open|watch|listen/,
      info: /tell me about|what is|information|details/,
      help: /help|how to|what can|commands/
    };
    
    // Determine intent type
    let intentType = 'search'; // default
    for (const [type, pattern] of Object.entries(patterns)) {
      if (pattern.test(lowerMessage)) {
        intentType = type;
        break;
      }
    }
    
    return {
      type: intentType,
      originalMessage: message,
      searchQuery: this.extractSearchQuery(message, intentType),
      mediaType: this.extractMediaType(message),
      action: this.extractAction(message),
      context: context?.lastSearch || null
    };
  }

  /**
   * Extract search query from conversational message
   */
  extractSearchQuery(message, intentType) {
    let query = message;
    
    // Remove common command words
    const commandWords = [
      'find', 'search', 'show', 'get', 'look for', 'where is',
      'play', 'start', 'open', 'watch', 'listen',
      'tell me about', 'what is', 'information', 'details'
    ];
    
    commandWords.forEach(command => {
      const regex = new RegExp(`\\b${command}\\b`, 'gi');
      query = query.replace(regex, '').trim();
    });
    
    return query;
  }

  /**
   * Extract media type from message
   */
  extractMediaType(message) {
    const lowerMessage = message.toLowerCase();
    
    if (/video|movie|film|clip/.test(lowerMessage)) return 'video';
    if (/audio|music|song|sound|track/.test(lowerMessage)) return 'audio';
    if (/image|photo|picture|pic/.test(lowerMessage)) return 'image';
    if (/document|doc|file|text|pdf/.test(lowerMessage)) return 'document';
    
    return null;
  }

  /**
   * Extract action from message
   */
  extractAction(message) {
    const lowerMessage = message.toLowerCase();
    
    if (/play|start|watch|listen/.test(lowerMessage)) return 'play';
    if (/info|details|about/.test(lowerMessage)) return 'info';
    if (/delete|remove/.test(lowerMessage)) return 'delete';
    if (/edit|modify/.test(lowerMessage)) return 'edit';
    
    return 'display';
  }

  /**
   * Handle search intent
   */
  async handleSearchIntent(intent, userId, contextId) {
    const searchResults = await this.search({
      query: intent.searchQuery,
      type: intent.mediaType,
      userId,
      contextId
    });
    
    const count = searchResults.results.length;
    let message;
    
    if (count === 0) {
      message = `I couldn't find any media matching "${intent.searchQuery}". Try using different keywords or check if the files are in the media directory.`;
    } else if (count === 1) {
      message = `I found 1 item matching "${intent.searchQuery}".`;
    } else {
      message = `I found ${count} items matching "${intent.searchQuery}".`;
    }
    
    return {
      type: 'search_results',
      message,
      query: intent.searchQuery,
      results: searchResults.results.slice(0, 10), // Return top 10
      totalCount: count,
      hasMore: count > 10
    };
  }

  /**
   * Handle filter intent
   */
  async handleFilterIntent(intent, context, userId) {
    if (!context?.lastSearch) {
      return {
        type: 'error',
        message: 'I need a previous search to apply filters. Please search for something first.'
      };
    }
    
    // Apply additional filters to previous search
    const filteredResults = context.lastSearch.results.filter(media => {
      // Apply intent-based filtering
      if (intent.mediaType && media.mediaType !== intent.mediaType) {
        return false;
      }
      
      // Additional filtering logic based on intent
      return true;
    });
    
    return {
      type: 'filtered_results',
      message: `Filtered to ${filteredResults.length} items.`,
      results: filteredResults.slice(0, 10),
      totalCount: filteredResults.length
    };
  }

  /**
   * Handle play intent
   */
  async handlePlayIntent(intent, context, userId) {
    // If specific media mentioned, find and play it
    if (intent.searchQuery) {
      const searchResults = await this.search({
        query: intent.searchQuery,
        userId,
        limit: 1
      });
      
      if (searchResults.results.length > 0) {
        const media = searchResults.results[0];
        return {
          type: 'play_media',
          message: `Playing "${media.fileName}".`,
          media,
          action: 'play'
        };
      } else {
        return {
          type: 'error',
          message: `I couldn't find "${intent.searchQuery}" to play.`
        };
      }
    }
    
    // If no specific media, play from last search results
    if (context?.lastSearch?.results?.length > 0) {
      const media = context.lastSearch.results[0];
      return {
        type: 'play_media',
        message: `Playing "${media.fileName}" from your last search.`,
        media,
        action: 'play'
      };
    }
    
    return {
      type: 'error',
      message: 'Please specify what you want to play or search for media first.'
    };
  }

  /**
   * Handle info intent
   */
  async handleInfoIntent(intent, context, userId) {
    if (intent.searchQuery) {
      const searchResults = await this.search({
        query: intent.searchQuery,
        userId,
        limit: 1
      });
      
      if (searchResults.results.length > 0) {
        const media = searchResults.results[0];
        return {
          type: 'media_info',
          message: this.formatMediaInfo(media),
          media
        };
      }
    }
    
    return {
      type: 'error',
      message: 'Please specify which media you want information about.'
    };
  }

  /**
   * Handle help intent
   */
  handleHelpIntent(intent, language) {
    const helpMessage = `I can help you with your media library! Here's what you can ask me:

📍 Search: "Find my vacation videos" or "Show me jazz music"
🎵 Play: "Play the latest song" or "Watch the video about cats"  
📊 Info: "Tell me about this file" or "What's in my music collection"
🔍 Filter: "Only show videos from last month" or "Just audio files"

I understand natural language, so feel free to ask in your own words!`;

    return {
      type: 'help',
      message: helpMessage
    };
  }

  /**
   * Handle unknown intent
   */
  handleUnknownIntent(message, language) {
    return {
      type: 'clarification',
      message: `I'm not sure what you want to do with "${message}". Try asking me to search, play, or get information about your media files.`
    };
  }

  /**
   * Format media information for display
   */
  formatMediaInfo(media) {
    let info = `📁 **${media.fileName}**\n`;
    info += `📂 Type: ${media.mediaType}\n`;
    info += `📏 Size: ${this.formatFileSize(media.size)}\n`;
    info += `📅 Modified: ${new Date(media.modifiedAt).toLocaleDateString()}\n`;
    
    if (media.metadata.duration) {
      info += `⏱️ Duration: ${this.formatDuration(media.metadata.duration)}\n`;
    }
    
    if (media.metadata.width && media.metadata.height) {
      info += `📐 Resolution: ${media.metadata.width}x${media.metadata.height}\n`;
    }
    
    if (media.tags && media.tags.length > 0) {
      info += `🏷️ Tags: ${media.tags.join(', ')}\n`;
    }
    
    if (media.description) {
      info += `📝 Description: ${media.description}`;
    }
    
    return info;
  }

  /**
   * Format file size for display
   */
  formatFileSize(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  /**
   * Format duration for display
   */
  formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
  }

  /**
   * Generate AI-powered content summary for media
   */
  async summarizeMediaContent(mediaId, options = {}) {
    if (!this.config.enableContentSummarization) {
      throw new Error('Content summarization is disabled');
    }

    const media = this.mediaIndex.get(mediaId);
    if (!media) {
      throw new Error('Media not found');
    }

    try {
      // Get content for summarization
      const content = await this.extractContentForSummary(media);
      
      if (!content) {
        throw new Error('No content available for summarization');
      }

      // Use LLM processor for summarization (injected dependency)
      if (!this.llmProcessor) {
        throw new Error('LLM processor not available for summarization');
      }

      const summaryOptions = {
        content,
        type: this.mapMediaTypeToSummaryType(media.mediaType),
        length: options.length || 'medium',
        style: options.style || 'informative',
        language: options.language || 'en',
        context: {
          mediaInfo: {
            title: media.fileName,
            type: media.mediaType,
            duration: media.metadata.duration,
            size: media.size
          },
          ...options.context
        }
      };

      const summary = await this.llmProcessor.summarizeContent(summaryOptions);

      // Cache the summary
      await this.cacheSummary(mediaId, summary, options);

      // Update media metadata with summary
      media.summary = {
        text: summary.summary,
        insights: summary.insights,
        confidence: summary.confidence,
        generatedAt: Date.now(),
        options: summaryOptions
      };

      this.emit('contentSummarized', {
        mediaId,
        summary,
        media: media.fileName
      });

      return {
        mediaId,
        fileName: media.fileName,
        summary: summary.summary,
        insights: summary.insights,
        confidence: summary.confidence,
        compressionRatio: summary.compressionRatio,
        processingTime: summary.processingTime,
        generatedAt: Date.now()
      };

    } catch (error) {
      console.error('Media content summarization failed:', error);
      throw error;
    }
  }

  /**
   * Extract content for summarization based on media type
   */
  async extractContentForSummary(media) {
    switch (media.mediaType) {
      case 'video':
        return await this.extractVideoContent(media);
      case 'audio':
        return await this.extractAudioContent(media);
      case 'document':
        return await this.extractDocumentContent(media);
      case 'image':
        return await this.extractImageContent(media);
      default:
        return null;
    }
  }

  /**
   * Extract video content for summarization
   */
  async extractVideoContent(media) {
    // In production, extract subtitles, transcripts, or scene descriptions
    let content = '';

    // Use metadata if available
    if (media.metadata.title) content += `Title: ${media.metadata.title}\n`;
    if (media.metadata.description) content += `Description: ${media.metadata.description}\n`;
    if (media.metadata.tags) content += `Tags: ${media.metadata.tags.join(', ')}\n`;
    
    // Mock transcript extraction (in production, use speech-to-text)
    if (media.metadata.duration) {
      content += `This is a ${Math.floor(media.metadata.duration / 60)} minute video `;
      if (media.metadata.genre) content += `in the ${media.metadata.genre} category `;
      content += `titled "${media.fileName}".`;
    }

    // Add file-based content hints
    const baseName = media.baseName.toLowerCase();
    if (baseName.includes('tutorial')) content += ' This appears to be a tutorial video.';
    if (baseName.includes('meeting')) content += ' This appears to be a meeting recording.';
    if (baseName.includes('presentation')) content += ' This appears to be a presentation.';

    return content || `Video file: ${media.fileName}`;
  }

  /**
   * Extract audio content for summarization
   */
  async extractAudioContent(media) {
    let content = '';

    // Use metadata
    if (media.metadata.title) content += `Title: ${media.metadata.title}\n`;
    if (media.metadata.artist) content += `Artist: ${media.metadata.artist}\n`;
    if (media.metadata.album) content += `Album: ${media.metadata.album}\n`;
    if (media.metadata.genre) content += `Genre: ${media.metadata.genre}\n`;
    if (media.metadata.year) content += `Year: ${media.metadata.year}\n`;

    // Mock transcript for voice recordings (in production, use speech-to-text)
    const baseName = media.baseName.toLowerCase();
    if (baseName.includes('podcast')) {
      content += 'This appears to be a podcast episode.';
    } else if (baseName.includes('interview')) {
      content += 'This appears to be an interview recording.';
    } else if (baseName.includes('meeting')) {
      content += 'This appears to be a meeting audio recording.';
    } else if (media.metadata.artist && media.metadata.title) {
      content += `This is a music track by ${media.metadata.artist}.`;
    }

    return content || `Audio file: ${media.fileName}`;
  }

  /**
   * Extract document content for summarization
   */
  async extractDocumentContent(media) {
    try {
      // In production, extract text from PDF, DOC, etc.
      const ext = path.extname(media.filePath).toLowerCase();
      
      if (ext === '.txt' || ext === '.md') {
        // Read text file content
        const content = await fs.readFile(media.filePath, 'utf-8');
        return content.substring(0, 10000); // Limit to 10k chars for summarization
      }
      
      // Mock content extraction for other document types
      return `Document: ${media.fileName}. This is a ${ext} file that contains textual content.`;
      
    } catch (error) {
      console.warn('Failed to extract document content:', error);
      return `Document file: ${media.fileName}`;
    }
  }

  /**
   * Extract image content for summarization
   */
  async extractImageContent(media) {
    // In production, use image recognition/OCR for content extraction
    let content = `Image: ${media.fileName}`;
    
    if (media.metadata.width && media.metadata.height) {
      content += ` (${media.metadata.width}x${media.metadata.height})`;
    }

    // Add context from filename
    const baseName = media.baseName.toLowerCase();
    if (baseName.includes('screenshot')) content += '. This appears to be a screenshot.';
    if (baseName.includes('photo')) content += '. This appears to be a photograph.';
    if (baseName.includes('diagram')) content += '. This appears to be a diagram or chart.';

    return content;
  }

  /**
   * Map media type to summary type
   */
  mapMediaTypeToSummaryType(mediaType) {
    const mapping = {
      'video': 'video',
      'audio': 'audio',
      'document': 'document',
      'image': 'article'
    };
    return mapping[mediaType] || 'auto';
  }

  /**
   * Cache summary for future use
   */
  async cacheSummary(mediaId, summary, options) {
    try {
      const cacheDir = path.join(this.config.metadataDirectory, 'summaries');
      await fs.mkdir(cacheDir, { recursive: true });
      
      const cacheFile = path.join(cacheDir, `${mediaId}.json`);
      const cacheData = {
        mediaId,
        summary,
        options,
        cachedAt: Date.now()
      };
      
      await fs.writeFile(cacheFile, JSON.stringify(cacheData, null, 2));
    } catch (error) {
      console.warn('Failed to cache summary:', error);
    }
  }

  /**
   * Get cached summary if available
   */
  async getCachedSummary(mediaId, options = {}) {
    try {
      const cacheFile = path.join(this.config.metadataDirectory, 'summaries', `${mediaId}.json`);
      const cacheData = JSON.parse(await fs.readFile(cacheFile, 'utf-8'));
      
      // Check if cache is still valid (24 hours)
      const maxAge = 24 * 60 * 60 * 1000;
      if (Date.now() - cacheData.cachedAt < maxAge) {
        return cacheData.summary;
      }
    } catch (error) {
      // Cache miss or error - will regenerate
    }
    return null;
  }

  /**
   * Batch summarize multiple media files
   */
  async batchSummarizeMedia(mediaIds, options = {}) {
    const results = [];
    const batchSize = options.batchSize || 3;
    
    for (let i = 0; i < mediaIds.length; i += batchSize) {
      const batch = mediaIds.slice(i, i + batchSize);
      
      const batchPromises = batch.map(async (mediaId) => {
        try {
          // Check cache first
          const cached = await this.getCachedSummary(mediaId, options);
          if (cached && !options.forceRegenerate) {
            return {
              mediaId,
              cached: true,
              summary: cached
            };
          }
          
          const summary = await this.summarizeMediaContent(mediaId, options);
          return {
            mediaId,
            cached: false,
            summary
          };
        } catch (error) {
          return {
            mediaId,
            error: error.message
          };
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
      
      // Add delay between batches
      if (i + batchSize < mediaIds.length) {
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }

    return {
      summaries: results,
      totalProcessed: mediaIds.length,
      successCount: results.filter(r => !r.error).length,
      errorCount: results.filter(r => r.error).length,
      cachedCount: results.filter(r => r.cached).length
    };
  }

  /**
   * Generate highlights for media content
   */
  async generateMediaHighlights(mediaId, maxHighlights = 5) {
    const media = this.mediaIndex.get(mediaId);
    if (!media) {
      throw new Error('Media not found');
    }

    const content = await this.extractContentForSummary(media);
    if (!content) {
      throw new Error('No content available for highlight generation');
    }

    if (!this.llmProcessor) {
      throw new Error('LLM processor not available');
    }

    const highlights = await this.llmProcessor.generateHighlights(content, maxHighlights);
    
    // Cache highlights
    media.highlights = {
      ...highlights,
      generatedAt: Date.now()
    };

    return {
      mediaId,
      fileName: media.fileName,
      highlights: highlights.highlights,
      confidence: highlights.confidence,
      totalHighlights: highlights.totalHighlights,
      generatedAt: Date.now()
    };
  }

  /**
   * Set LLM processor for summarization (dependency injection)
   */
  setLLMProcessor(llmProcessor) {
    this.llmProcessor = llmProcessor;
  }

  /**
   * Execute voice editing command
   */
  async executeEditCommand(parsedCommand) {
    const { parsedCommands, mediaId } = parsedCommand;
    
    if (!mediaId) {
      throw new Error('Media ID is required for editing commands');
    }
    
    const media = this.mediaIndex.get(mediaId);
    if (!media) {
      throw new Error('Media not found');
    }
    
    const results = [];
    
    for (const command of parsedCommands) {
      try {
        const result = await this.executeEditOperation(media, command);
        results.push(result);
      } catch (error) {
        results.push({
          command: command.type,
          success: false,
          error: error.message
        });
      }
    }
    
    return {
      mediaId,
      originalFile: media.filePath,
      operations: results,
      timestamp: Date.now()
    };
  }

  /**
   * Execute individual edit operation (placeholder)
   */
  async executeEditOperation(media, command) {
    // In production, integrate with ffmpeg or similar tools
    console.log(`Executing ${command.type} on ${media.fileName} with parameters:`, command.parameters);
    
    return {
      command: command.type,
      success: true,
      parameters: command.parameters,
      message: `${command.type} operation completed successfully`
    };
  }

  /**
   * Get or create conversation context
   */
  getConversationContext(contextId) {
    if (!contextId) return null;
    
    return this.conversationContexts.get(contextId) || null;
  }

  /**
   * Update conversation context
   */
  updateConversationContext(contextId, update) {
    if (!contextId) return;
    
    let context = this.conversationContexts.get(contextId);
    if (!context) {
      context = {
        id: contextId,
        createdAt: Date.now(),
        interactions: []
      };
    }
    
    // Update with new information
    Object.assign(context, update);
    context.lastActivity = Date.now();
    
    // Store search results for context
    if (update.results) {
      context.lastSearch = {
        query: update.query,
        results: update.results,
        timestamp: Date.now()
      };
    }
    
    this.conversationContexts.set(contextId, context);
  }

  /**
   * Generate unique media ID
   */
  generateMediaId(filePath) {
    return crypto.createHash('md5').update(filePath).digest('hex');
  }

  /**
   * Setup file system watchers
   */
  async setupFileWatchers() {
    // In production, use fs.watch or chokidar for file system monitoring
    console.log('File system watchers would be set up here');
  }

  /**
   * Save metadata cache
   */
  async saveMetadataCache() {
    try {
      const cacheFile = path.join(this.config.metadataDirectory, 'metadata_cache.json');
      const cacheData = Object.fromEntries(this.metadataCache);
      await fs.writeFile(cacheFile, JSON.stringify(cacheData, null, 2));
    } catch (error) {
      console.warn('Failed to save metadata cache:', error.message);
    }
  }

  /**
   * Get media library statistics
   */
  getLibraryStats() {
    const stats = {
      totalFiles: this.mediaIndex.size,
      byType: {},
      totalSize: 0,
      searchTerms: this.searchIndex.size,
      tags: this.tagIndex.size
    };
    
    for (const media of this.mediaIndex.values()) {
      stats.byType[media.mediaType] = (stats.byType[media.mediaType] || 0) + 1;
      stats.totalSize += media.size || 0;
    }
    
    return stats;
  }

  /**
   * Check if service is ready
   */
  isReady() {
    return this.isInitialized;
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      mediaFiles: this.mediaIndex.size,
      searchTerms: this.searchIndex.size,
      tags: this.tagIndex.size,
      activeContexts: this.conversationContexts.size,
      libraryStats: this.getLibraryStats()
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    console.log('Cleaning up Media Library Service...');
    
    // Save metadata cache before cleanup
    await this.saveMetadataCache();
    
    // Close file watchers
    this.watchers.forEach(watcher => {
      if (watcher.close) watcher.close();
    });
    
    // Clear data structures
    this.mediaIndex.clear();
    this.searchIndex.clear();
    this.tagIndex.clear();
    this.metadataCache.clear();
    this.conversationContexts.clear();
    this.userPreferences.clear();
    
    this.isInitialized = false;
    this.removeAllListeners();
  }
}

export default MediaLibraryService;