/**
 * Translation Service
 * Handles real-time translation with multiple providers and advanced features
 */

import axios from 'axios';
import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';

export class TranslationService extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      // Provider configuration
      primaryProvider: config.primaryProvider || 'google',
      fallbackProviders: config.fallbackProviders || ['azure', 'deepl'],
      
      // API keys
      googleApiKey: config.googleApiKey || process.env.GOOGLE_TRANSLATE_API_KEY,
      azureKey: config.azureKey || process.env.AZURE_TRANSLATOR_KEY,
      azureRegion: config.azureRegion || process.env.AZURE_REGION,
      deeplApiKey: config.deeplApiKey || process.env.DEEPL_API_KEY,
      awsAccessKey: config.awsAccessKey || process.env.AWS_ACCESS_KEY,
      awsSecretKey: config.awsSecretKey || process.env.AWS_SECRET_KEY,
      
      // Translation settings
      maxTextLength: config.maxTextLength || 5000,
      batchSize: config.batchSize || 100,
      requestTimeout: config.requestTimeout || 10000,
      
      // Caching
      enableCaching: config.enableCaching !== false,
      cacheDirectory: config.cacheDirectory || './data/translation_cache',
      cacheTimeout: config.cacheTimeout || 86400000, // 24 hours
      maxCacheSize: config.maxCacheSize || 10000,
      
      // Quality and detection
      confidenceThreshold: config.confidenceThreshold || 0.7,
      enableLanguageDetection: config.enableLanguageDetection !== false,
      enableQualityScoring: config.enableQualityScoring !== false,
      
      // Real-time features
      enableRealTimeTranslation: config.enableRealTimeTranslation !== false,
      realTimeDebounceMs: config.realTimeDebounceMs || 500,
      maxConcurrentRequests: config.maxConcurrentRequests || 10,
      
      ...config
    };

    // Provider status
    this.providerStatus = {
      google: false,
      azure: false,
      deepl: false,
      aws: false,
      local: true
    };

    // Translation cache
    this.translationCache = new Map();
    this.qualityCache = new Map();
    
    // Language detection cache
    this.languageDetectionCache = new Map();
    
    // Real-time translation sessions
    this.realTimeSessions = new Map();
    
    // Request queue for rate limiting
    this.requestQueue = [];
    this.activeRequests = 0;
    
    // Supported language pairs by provider
    this.supportedLanguages = new Map();
    
    this.isInitialized = false;
  }

  /**
   * Initialize the translation service
   */
  async initialize() {
    try {
      console.log('Initializing Translation Service...');
      
      // Create cache directory
      await this.ensureCacheDirectory();
      
      // Test provider connectivity
      await this.testProviderConnectivity();
      
      // Load supported languages
      await this.loadSupportedLanguages();
      
      // Load translation cache
      await this.loadTranslationCache();
      
      // Setup cleanup intervals
      this.setupCleanupIntervals();
      
      // Initialize request processing
      this.startRequestProcessor();
      
      this.isInitialized = true;
      console.log('Translation Service initialized successfully');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize Translation Service:', error);
      throw error;
    }
  }

  /**
   * Ensure cache directory exists
   */
  async ensureCacheDirectory() {
    try {
      await fs.access(this.config.cacheDirectory);
    } catch {
      await fs.mkdir(this.config.cacheDirectory, { recursive: true });
    }
  }

  /**
   * Test connectivity to translation providers
   */
  async testProviderConnectivity() {
    const tests = [];

    // Test Google Translate
    if (this.config.googleApiKey) {
      tests.push(this.testGoogleConnectivity());
    }

    // Test Azure Translator
    if (this.config.azureKey) {
      tests.push(this.testAzureConnectivity());
    }

    // Test DeepL
    if (this.config.deeplApiKey) {
      tests.push(this.testDeepLConnectivity());
    }

    const results = await Promise.allSettled(tests);
    
    results.forEach((result, index) => {
      const providers = ['google', 'azure', 'deepl'];
      if (result.status === 'fulfilled') {
        this.providerStatus[providers[index]] = true;
        console.log(`✅ ${providers[index]} translation provider connected`);
      } else {
        console.warn(`⚠️  ${providers[index]} translation provider not available:`, result.reason.message);
      }
    });
  }

  /**
   * Test Google Translate connectivity
   */
  async testGoogleConnectivity() {
    try {
      const response = await axios.get(`https://translation.googleapis.com/language/translate/v2/languages?key=${this.config.googleApiKey}`, {
        timeout: 5000
      });
      return response.status === 200;
    } catch (error) {
      throw new Error(`Google Translate connection failed: ${error.message}`);
    }
  }

  /**
   * Test Azure Translator connectivity
   */
  async testAzureConnectivity() {
    try {
      const response = await axios.get('https://api.cognitive.microsofttranslator.com/languages?api-version=3.0', {
        headers: {
          'Ocp-Apim-Subscription-Key': this.config.azureKey,
          'Ocp-Apim-Subscription-Region': this.config.azureRegion
        },
        timeout: 5000
      });
      return response.status === 200;
    } catch (error) {
      throw new Error(`Azure Translator connection failed: ${error.message}`);
    }
  }

  /**
   * Test DeepL connectivity
   */
  async testDeepLConnectivity() {
    try {
      const response = await axios.get(`https://api-free.deepl.com/v2/languages?auth_key=${this.config.deeplApiKey}`, {
        timeout: 5000
      });
      return response.status === 200;
    } catch (error) {
      throw new Error(`DeepL connection failed: ${error.message}`);
    }
  }

  /**
   * Load supported languages from providers
   */
  async loadSupportedLanguages() {
    // Load Google languages
    if (this.providerStatus.google) {
      await this.loadGoogleLanguages();
    }

    // Load Azure languages
    if (this.providerStatus.azure) {
      await this.loadAzureLanguages();
    }

    // Load DeepL languages
    if (this.providerStatus.deepl) {
      await this.loadDeepLLanguages();
    }

    // Add default language support
    this.initializeDefaultLanguageSupport();
  }

  /**
   * Load Google Translate supported languages
   */
  async loadGoogleLanguages() {
    try {
      const response = await axios.get(`https://translation.googleapis.com/language/translate/v2/languages?key=${this.config.googleApiKey}`);
      
      const languages = response.data.data.languages.map(lang => ({
        code: lang.language,
        name: lang.name || lang.language
      }));
      
      this.supportedLanguages.set('google', languages);
      console.log(`Loaded ${languages.length} Google Translate languages`);
    } catch (error) {
      console.warn('Failed to load Google languages:', error.message);
    }
  }

  /**
   * Load Azure Translator supported languages
   */
  async loadAzureLanguages() {
    try {
      const response = await axios.get('https://api.cognitive.microsofttranslator.com/languages?api-version=3.0&scope=translation', {
        headers: {
          'Ocp-Apim-Subscription-Key': this.config.azureKey
        }
      });
      
      const languages = Object.entries(response.data.translation).map(([code, info]) => ({
        code,
        name: info.name,
        nativeName: info.nativeName,
        dir: info.dir
      }));
      
      this.supportedLanguages.set('azure', languages);
      console.log(`Loaded ${languages.length} Azure Translator languages`);
    } catch (error) {
      console.warn('Failed to load Azure languages:', error.message);
    }
  }

  /**
   * Load DeepL supported languages
   */
  async loadDeepLLanguages() {
    try {
      const response = await axios.get(`https://api-free.deepl.com/v2/languages?auth_key=${this.config.deeplApiKey}`);
      
      const languages = response.data.map(lang => ({
        code: lang.language.toLowerCase(),
        name: lang.name
      }));
      
      this.supportedLanguages.set('deepl', languages);
      console.log(`Loaded ${languages.length} DeepL languages`);
    } catch (error) {
      console.warn('Failed to load DeepL languages:', error.message);
    }
  }

  /**
   * Initialize default language support
   */
  initializeDefaultLanguageSupport() {
    const defaultLanguages = [
      { code: 'en', name: 'English' },
      { code: 'es', name: 'Spanish' },
      { code: 'fr', name: 'French' },
      { code: 'de', name: 'German' },
      { code: 'it', name: 'Italian' },
      { code: 'pt', name: 'Portuguese' },
      { code: 'ru', name: 'Russian' },
      { code: 'zh', name: 'Chinese' },
      { code: 'ja', name: 'Japanese' },
      { code: 'ko', name: 'Korean' },
      { code: 'ar', name: 'Arabic' },
      { code: 'hi', name: 'Hindi' },
      { code: 'th', name: 'Thai' },
      { code: 'vi', name: 'Vietnamese' },
      { code: 'tr', name: 'Turkish' },
      { code: 'pl', name: 'Polish' },
      { code: 'nl', name: 'Dutch' },
      { code: 'sv', name: 'Swedish' },
      { code: 'da', name: 'Danish' },
      { code: 'no', name: 'Norwegian' },
      { code: 'fi', name: 'Finnish' }
    ];

    this.supportedLanguages.set('default', defaultLanguages);
  }

  /**
   * Load translation cache from disk
   */
  async loadTranslationCache() {
    try {
      const cacheFile = path.join(this.config.cacheDirectory, 'translation_cache.json');
      const cacheData = await fs.readFile(cacheFile, 'utf8');
      const cache = JSON.parse(cacheData);
      
      Object.entries(cache).forEach(([key, value]) => {
        this.translationCache.set(key, value);
      });
      
      console.log(`Loaded ${this.translationCache.size} translation cache entries`);
    } catch (error) {
      console.log('No existing translation cache found');
    }
  }

  /**
   * Setup cleanup intervals
   */
  setupCleanupIntervals() {
    // Clean expired cache entries every hour
    setInterval(() => {
      this.cleanupExpiredCache();
    }, 3600000);

    // Clean up old real-time sessions every 30 minutes
    setInterval(() => {
      this.cleanupOldSessions();
    }, 1800000);

    // Save cache every 10 minutes
    setInterval(() => {
      this.saveTranslationCache();
    }, 600000);
  }

  /**
   * Start request processor for rate limiting
   */
  startRequestProcessor() {
    setInterval(() => {
      this.processRequestQueue();
    }, 100); // Process queue every 100ms
  }

  /**
   * Translate text with automatic provider selection
   */
  async translate(options) {
    const {
      text,
      targetLanguage,
      sourceLanguage = null,
      provider = this.config.primaryProvider,
      quality = 'balanced', // 'fast', 'balanced', 'high'
      context = null
    } = options;

    if (!text || !text.trim()) {
      throw new Error('Text is required for translation');
    }

    if (!targetLanguage) {
      throw new Error('Target language is required');
    }

    if (text.length > this.config.maxTextLength) {
      throw new Error(`Text too long. Maximum length: ${this.config.maxTextLength} characters`);
    }

    try {
      // Auto-detect source language if not provided
      let sourceLang = sourceLanguage;
      if (!sourceLang && this.config.enableLanguageDetection) {
        sourceLang = await this.detectLanguage(text);
      }

      // Check if translation is needed
      if (sourceLang === targetLanguage) {
        return {
          translatedText: text,
          sourceLanguage: sourceLang,
          targetLanguage,
          confidence: 1.0,
          provider: 'none',
          cached: false
        };
      }

      // Check cache first
      const cacheKey = this.generateCacheKey(text, sourceLang, targetLanguage, provider);
      if (this.config.enableCaching && this.translationCache.has(cacheKey)) {
        const cached = this.translationCache.get(cacheKey);
        if (Date.now() - cached.timestamp < this.config.cacheTimeout) {
          return {
            ...cached.result,
            cached: true
          };
        }
      }

      // Determine optimal provider based on quality setting
      const optimalProvider = this.selectOptimalProvider(sourceLang, targetLanguage, quality);

      // Perform translation
      const result = await this.translateWithProvider(text, sourceLang, targetLanguage, optimalProvider, context);

      // Assess translation quality if enabled
      if (this.config.enableQualityScoring) {
        result.qualityScore = await this.assessTranslationQuality(text, result.translatedText, sourceLang, targetLanguage);
      }

      // Cache result
      if (this.config.enableCaching) {
        this.cacheTranslation(cacheKey, result);
      }

      this.emit('translationCompleted', {
        sourceText: text,
        result,
        provider: optimalProvider
      });

      return {
        ...result,
        cached: false
      };

    } catch (error) {
      console.error('Translation failed:', error);
      this.emit('translationError', { text, targetLanguage, error });
      throw error;
    }
  }

  /**
   * Select optimal provider based on language pair and quality requirements
   */
  selectOptimalProvider(sourceLanguage, targetLanguage, quality) {
    // Provider quality rankings for different scenarios
    const providerRankings = {
      'high': ['deepl', 'google', 'azure'],
      'balanced': ['google', 'azure', 'deepl'],
      'fast': ['azure', 'google', 'deepl']
    };

    const preferredOrder = providerRankings[quality] || providerRankings['balanced'];
    
    // Find first available provider in preferred order
    for (const provider of preferredOrder) {
      if (this.providerStatus[provider] && this.supportsLanguagePair(provider, sourceLanguage, targetLanguage)) {
        return provider;
      }
    }

    // Fallback to any available provider
    for (const provider of Object.keys(this.providerStatus)) {
      if (this.providerStatus[provider] && provider !== 'local') {
        return provider;
      }
    }

    return 'local';
  }

  /**
   * Check if provider supports language pair
   */
  supportsLanguagePair(provider, sourceLanguage, targetLanguage) {
    const languages = this.supportedLanguages.get(provider);
    if (!languages) return true; // Assume support if we don't have the list
    
    const codes = languages.map(l => l.code);
    return codes.includes(sourceLanguage) && codes.includes(targetLanguage);
  }

  /**
   * Translate with specific provider
   */
  async translateWithProvider(text, sourceLanguage, targetLanguage, provider, context) {
    return new Promise((resolve, reject) => {
      const request = {
        text,
        sourceLanguage,
        targetLanguage,
        provider,
        context,
        resolve,
        reject,
        timestamp: Date.now()
      };

      this.requestQueue.push(request);
    });
  }

  /**
   * Process request queue for rate limiting
   */
  async processRequestQueue() {
    while (this.requestQueue.length > 0 && this.activeRequests < this.config.maxConcurrentRequests) {
      const request = this.requestQueue.shift();
      this.activeRequests++;

      this.executeTranslationRequest(request)
        .then(result => request.resolve(result))
        .catch(error => request.reject(error))
        .finally(() => {
          this.activeRequests--;
        });
    }
  }

  /**
   * Execute translation request with specific provider
   */
  async executeTranslationRequest(request) {
    const { text, sourceLanguage, targetLanguage, provider, context } = request;

    try {
      switch (provider) {
        case 'google':
          return await this.translateWithGoogle(text, sourceLanguage, targetLanguage, context);
        case 'azure':
          return await this.translateWithAzure(text, sourceLanguage, targetLanguage, context);
        case 'deepl':
          return await this.translateWithDeepL(text, sourceLanguage, targetLanguage, context);
        case 'aws':
          return await this.translateWithAWS(text, sourceLanguage, targetLanguage, context);
        default:
          return this.translateLocally(text, sourceLanguage, targetLanguage);
      }
    } catch (error) {
      // Try fallback providers
      for (const fallbackProvider of this.config.fallbackProviders) {
        if (fallbackProvider !== provider && this.providerStatus[fallbackProvider]) {
          try {
            console.warn(`Falling back to ${fallbackProvider} after ${provider} failed`);
            return await this.executeTranslationRequest({
              ...request,
              provider: fallbackProvider
            });
          } catch (fallbackError) {
            console.warn(`Fallback provider ${fallbackProvider} also failed:`, fallbackError.message);
          }
        }
      }
      throw error;
    }
  }

  /**
   * Translate with Google Translate
   */
  async translateWithGoogle(text, sourceLanguage, targetLanguage, context) {
    try {
      const response = await axios.post(`https://translation.googleapis.com/language/translate/v2?key=${this.config.googleApiKey}`, {
        q: text,
        source: sourceLanguage,
        target: targetLanguage,
        format: 'text'
      }, {
        timeout: this.config.requestTimeout
      });

      const translation = response.data.data.translations[0];
      
      return {
        translatedText: translation.translatedText,
        sourceLanguage: translation.detectedSourceLanguage || sourceLanguage,
        targetLanguage,
        confidence: 0.9, // Google doesn't provide confidence scores
        provider: 'google'
      };
    } catch (error) {
      throw new Error(`Google Translate failed: ${error.message}`);
    }
  }

  /**
   * Translate with Azure Translator
   */
  async translateWithAzure(text, sourceLanguage, targetLanguage, context) {
    try {
      const url = `https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&from=${sourceLanguage}&to=${targetLanguage}`;
      
      const response = await axios.post(url, [{ text }], {
        headers: {
          'Ocp-Apim-Subscription-Key': this.config.azureKey,
          'Ocp-Apim-Subscription-Region': this.config.azureRegion,
          'Content-Type': 'application/json'
        },
        timeout: this.config.requestTimeout
      });

      const translation = response.data[0];
      const result = translation.translations[0];
      
      return {
        translatedText: result.text,
        sourceLanguage: translation.detectedLanguage?.language || sourceLanguage,
        targetLanguage,
        confidence: translation.detectedLanguage?.score || 0.8,
        provider: 'azure',
        alternatives: translation.translations.slice(1).map(alt => ({
          text: alt.text,
          confidence: 0.7 // Azure doesn't provide alternative confidence scores
        }))
      };
    } catch (error) {
      throw new Error(`Azure Translator failed: ${error.message}`);
    }
  }

  /**
   * Translate with DeepL
   */
  async translateWithDeepL(text, sourceLanguage, targetLanguage, context) {
    try {
      const response = await axios.post('https://api-free.deepl.com/v2/translate', null, {
        params: {
          auth_key: this.config.deeplApiKey,
          text,
          source_lang: sourceLanguage.toUpperCase(),
          target_lang: targetLanguage.toUpperCase(),
          preserve_formatting: '1',
          tag_handling: 'xml'
        },
        timeout: this.config.requestTimeout
      });

      const translation = response.data.translations[0];
      
      return {
        translatedText: translation.text,
        sourceLanguage: translation.detected_source_language?.toLowerCase() || sourceLanguage,
        targetLanguage,
        confidence: 0.95, // DeepL is generally high quality
        provider: 'deepl'
      };
    } catch (error) {
      throw new Error(`DeepL translation failed: ${error.message}`);
    }
  }

  /**
   * Translate with AWS Translate (placeholder)
   */
  async translateWithAWS(text, sourceLanguage, targetLanguage, context) {
    // AWS SDK implementation would go here
    throw new Error('AWS Translate not implemented in this demo');
  }

  /**
   * Local translation fallback
   */
  translateLocally(text, sourceLanguage, targetLanguage) {
    // Simple placeholder - in production could use offline translation models
    return {
      translatedText: `[Translation unavailable: ${text}]`,
      sourceLanguage,
      targetLanguage,
      confidence: 0.1,
      provider: 'local'
    };
  }

  /**
   * Detect language of text
   */
  async detectLanguage(text) {
    if (!text || !text.trim()) {
      return 'unknown';
    }

    // Check cache first
    const cacheKey = `detect_${text.slice(0, 100)}`;
    if (this.languageDetectionCache.has(cacheKey)) {
      const cached = this.languageDetectionCache.get(cacheKey);
      if (Date.now() - cached.timestamp < this.config.cacheTimeout) {
        return cached.language;
      }
    }

    try {
      let detectedLanguage = 'unknown';

      // Try Google Translate detection
      if (this.providerStatus.google) {
        detectedLanguage = await this.detectLanguageWithGoogle(text);
      } else if (this.providerStatus.azure) {
        detectedLanguage = await this.detectLanguageWithAzure(text);
      }

      // Cache result
      this.languageDetectionCache.set(cacheKey, {
        language: detectedLanguage,
        timestamp: Date.now()
      });

      return detectedLanguage;
    } catch (error) {
      console.warn('Language detection failed:', error.message);
      return 'auto';
    }
  }

  /**
   * Detect language with Google Translate
   */
  async detectLanguageWithGoogle(text) {
    try {
      const response = await axios.post(`https://translation.googleapis.com/language/translate/v2/detect?key=${this.config.googleApiKey}`, {
        q: text
      });

      const detection = response.data.data.detections[0][0];
      return detection.language;
    } catch (error) {
      throw new Error(`Google language detection failed: ${error.message}`);
    }
  }

  /**
   * Detect language with Azure Translator
   */
  async detectLanguageWithAzure(text) {
    try {
      const response = await axios.post('https://api.cognitive.microsofttranslator.com/detect?api-version=3.0', [{ text }], {
        headers: {
          'Ocp-Apim-Subscription-Key': this.config.azureKey,
          'Ocp-Apim-Subscription-Region': this.config.azureRegion,
          'Content-Type': 'application/json'
        }
      });

      const detection = response.data[0];
      return detection.language;
    } catch (error) {
      throw new Error(`Azure language detection failed: ${error.message}`);
    }
  }

  /**
   * Batch translate multiple texts
   */
  async batchTranslate(texts, targetLanguage, sourceLanguage = null, options = {}) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error('Texts array is required');
    }

    const batches = [];
    for (let i = 0; i < texts.length; i += this.config.batchSize) {
      batches.push(texts.slice(i, i + this.config.batchSize));
    }

    const results = [];
    for (const batch of batches) {
      const batchResults = await Promise.all(
        batch.map(text => this.translate({
          text,
          targetLanguage,
          sourceLanguage,
          ...options
        }))
      );
      results.push(...batchResults);
    }

    return results;
  }

  /**
   * Real-time translation for live conversations
   */
  async startRealTimeTranslation(options) {
    const {
      sessionId,
      sourceLanguage,
      targetLanguage,
      onTranslation,
      debounceMs = this.config.realTimeDebounceMs
    } = options;

    if (!sessionId) {
      throw new Error('Session ID is required for real-time translation');
    }

    const session = {
      id: sessionId,
      sourceLanguage,
      targetLanguage,
      onTranslation,
      debounceMs,
      buffer: '',
      debounceTimer: null,
      translations: [],
      createdAt: Date.now(),
      lastActivity: Date.now()
    };

    this.realTimeSessions.set(sessionId, session);

    this.emit('realTimeSessionStarted', { sessionId, sourceLanguage, targetLanguage });

    return session;
  }

  /**
   * Add text to real-time translation session
   */
  async addToRealTimeTranslation(sessionId, text, isFinal = false) {
    const session = this.realTimeSessions.get(sessionId);
    if (!session) {
      throw new Error('Real-time translation session not found');
    }

    session.buffer += text;
    session.lastActivity = Date.now();

    // Clear existing debounce timer
    if (session.debounceTimer) {
      clearTimeout(session.debounceTimer);
    }

    // If final or buffer is getting long, translate immediately
    if (isFinal || session.buffer.length > 200) {
      await this.processRealTimeBuffer(session);
    } else {
      // Set debounce timer
      session.debounceTimer = setTimeout(() => {
        this.processRealTimeBuffer(session);
      }, session.debounceMs);
    }
  }

  /**
   * Process real-time translation buffer
   */
  async processRealTimeBuffer(session) {
    if (!session.buffer.trim()) {
      return;
    }

    try {
      const result = await this.translate({
        text: session.buffer,
        sourceLanguage: session.sourceLanguage,
        targetLanguage: session.targetLanguage,
        quality: 'fast' // Use fast translation for real-time
      });

      session.translations.push({
        original: session.buffer,
        translated: result.translatedText,
        timestamp: Date.now(),
        confidence: result.confidence
      });

      // Call callback if provided
      if (session.onTranslation) {
        session.onTranslation(result);
      }

      this.emit('realTimeTranslation', {
        sessionId: session.id,
        result
      });

      // Clear buffer
      session.buffer = '';
    } catch (error) {
      console.error('Real-time translation failed:', error);
      this.emit('realTimeTranslationError', {
        sessionId: session.id,
        error
      });
    }
  }

  /**
   * End real-time translation session
   */
  endRealTimeTranslation(sessionId) {
    const session = this.realTimeSessions.get(sessionId);
    if (!session) {
      return { ended: false, reason: 'Session not found' };
    }

    // Clear any pending timer
    if (session.debounceTimer) {
      clearTimeout(session.debounceTimer);
    }

    // Process any remaining buffer
    if (session.buffer.trim()) {
      this.processRealTimeBuffer(session);
    }

    // Remove session
    this.realTimeSessions.delete(sessionId);

    this.emit('realTimeSessionEnded', {
      sessionId,
      translations: session.translations
    });

    return {
      ended: true,
      translations: session.translations,
      duration: Date.now() - session.createdAt
    };
  }

  /**
   * Assess translation quality using various metrics
   */
  async assessTranslationQuality(originalText, translatedText, sourceLanguage, targetLanguage) {
    try {
      // Check quality cache
      const qualityKey = this.generateCacheKey(originalText, sourceLanguage, targetLanguage, 'quality');
      if (this.qualityCache.has(qualityKey)) {
        const cached = this.qualityCache.get(qualityKey);
        if (Date.now() - cached.timestamp < this.config.cacheTimeout) {
          return cached.score;
        }
      }

      let qualityScore = 0.5; // Base score

      // Length-based assessment
      const lengthRatio = translatedText.length / originalText.length;
      if (lengthRatio > 0.3 && lengthRatio < 3.0) {
        qualityScore += 0.2;
      }

      // Character diversity assessment
      const originalChars = new Set(originalText.toLowerCase());
      const translatedChars = new Set(translatedText.toLowerCase());
      const charDiversityRatio = translatedChars.size / originalChars.size;
      if (charDiversityRatio > 0.5) {
        qualityScore += 0.1;
      }

      // Word count assessment
      const originalWords = originalText.split(/\s+/).length;
      const translatedWords = translatedText.split(/\s+/).length;
      const wordRatio = translatedWords / originalWords;
      if (wordRatio > 0.4 && wordRatio < 2.5) {
        qualityScore += 0.1;
      }

      // Structure preservation (punctuation, capitalization)
      const originalPunct = (originalText.match(/[.!?;:,]/g) || []).length;
      const translatedPunct = (translatedText.match(/[.!?;:,]/g) || []).length;
      if (originalPunct > 0 && Math.abs(originalPunct - translatedPunct) <= 2) {
        qualityScore += 0.1;
      }

      qualityScore = Math.max(0, Math.min(1, qualityScore));

      // Cache quality score
      this.qualityCache.set(qualityKey, {
        score: qualityScore,
        timestamp: Date.now()
      });

      return qualityScore;
    } catch (error) {
      console.warn('Quality assessment failed:', error.message);
      return 0.5; // Default score
    }
  }

  /**
   * Generate cache key for translations
   */
  generateCacheKey(text, sourceLanguage, targetLanguage, suffix = '') {
    const crypto = require('crypto');
    const data = `${text}|${sourceLanguage}|${targetLanguage}${suffix ? '|' + suffix : ''}`;
    return crypto.createHash('md5').update(data).digest('hex');
  }

  /**
   * Cache translation result
   */
  cacheTranslation(cacheKey, result) {
    // Check cache size limit
    if (this.translationCache.size >= this.config.maxCacheSize) {
      // Remove oldest entries
      const entries = Array.from(this.translationCache.entries());
      entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      for (let i = 0; i < Math.floor(this.config.maxCacheSize * 0.1); i++) {
        this.translationCache.delete(entries[i][0]);
      }
    }

    this.translationCache.set(cacheKey, {
      result,
      timestamp: Date.now()
    });
  }

  /**
   * Save translation cache to disk
   */
  async saveTranslationCache() {
    try {
      const cacheFile = path.join(this.config.cacheDirectory, 'translation_cache.json');
      const cacheData = Object.fromEntries(this.translationCache);
      await fs.writeFile(cacheFile, JSON.stringify(cacheData, null, 2));
    } catch (error) {
      console.warn('Failed to save translation cache:', error.message);
    }
  }

  /**
   * Clean up expired cache entries
   */
  cleanupExpiredCache() {
    const now = Date.now();
    
    // Clean translation cache
    for (const [key, entry] of this.translationCache.entries()) {
      if (now - entry.timestamp > this.config.cacheTimeout) {
        this.translationCache.delete(key);
      }
    }
    
    // Clean language detection cache
    for (const [key, entry] of this.languageDetectionCache.entries()) {
      if (now - entry.timestamp > this.config.cacheTimeout) {
        this.languageDetectionCache.delete(key);
      }
    }
    
    // Clean quality cache
    for (const [key, entry] of this.qualityCache.entries()) {
      if (now - entry.timestamp > this.config.cacheTimeout) {
        this.qualityCache.delete(key);
      }
    }
  }

  /**
   * Clean up old real-time sessions
   */
  cleanupOldSessions() {
    const now = Date.now();
    const maxAge = 60 * 60 * 1000; // 1 hour
    
    for (const [sessionId, session] of this.realTimeSessions.entries()) {
      if (now - session.lastActivity > maxAge) {
        this.endRealTimeTranslation(sessionId);
      }
    }
  }

  /**
   * Get supported languages for a provider
   */
  getSupportedLanguages(provider = null) {
    if (provider) {
      return this.supportedLanguages.get(provider) || [];
    }
    
    // Return combined list from all providers
    const allLanguages = new Map();
    
    for (const languages of this.supportedLanguages.values()) {
      languages.forEach(lang => {
        allLanguages.set(lang.code, lang);
      });
    }
    
    return Array.from(allLanguages.values());
  }

  /**
   * Get translation statistics
   */
  getTranslationStats() {
    return {
      cacheSize: this.translationCache.size,
      qualityCacheSize: this.qualityCache.size,
      languageDetectionCacheSize: this.languageDetectionCache.size,
      activeRealTimeSessions: this.realTimeSessions.size,
      pendingRequests: this.requestQueue.length,
      activeRequests: this.activeRequests
    };
  }

  /**
   * Check if service is ready
   */
  isReady() {
    return this.isInitialized && Object.values(this.providerStatus).some(status => status);
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      providers: this.providerStatus,
      supportedLanguagesCount: this.getSupportedLanguages().length,
      stats: this.getTranslationStats(),
      features: {
        caching: this.config.enableCaching,
        languageDetection: this.config.enableLanguageDetection,
        qualityScoring: this.config.enableQualityScoring,
        realTimeTranslation: this.config.enableRealTimeTranslation
      }
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    console.log('Cleaning up Translation Service...');
    
    // Save cache
    await this.saveTranslationCache();
    
    // End all real-time sessions
    for (const sessionId of this.realTimeSessions.keys()) {
      this.endRealTimeTranslation(sessionId);
    }
    
    // Clear data structures
    this.translationCache.clear();
    this.qualityCache.clear();
    this.languageDetectionCache.clear();
    this.realTimeSessions.clear();
    this.requestQueue.length = 0;
    this.supportedLanguages.clear();
    
    this.isInitialized = false;
    this.removeAllListeners();
  }
}

export default TranslationService;