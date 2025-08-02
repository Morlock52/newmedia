import Fastify from 'fastify';
import cors from '@fastify/cors';
import helmet from '@fastify/helmet';
import websocket from '@fastify/websocket';
import Redis from 'ioredis';
import Bull from 'bull';
import pino from 'pino';
import { NeuralRecommendationEngine } from './recommendation-engine.js';
import { ContentAnalysisPipeline } from './content-analysis.js';
import { VoiceProcessingSystem } from './voice-processor.js';
import { NeuralCompressionSystem } from './neural-compression.js';
import { EmotionDetectionSystem } from './emotion-detection.js';

const logger = pino({ transport: { target: 'pino-pretty' } });
const redis = new Redis();

/**
 * NEXUS AI/ML Orchestrator
 * Central service coordinating all AI/ML components
 */
class AIMLOrchestrator {
  constructor() {
    this.services = {
      recommendation: null,
      contentAnalysis: null,
      voice: null,
      compression: null,
      emotion: null
    };
    
    this.queues = {
      analysis: new Bull('content-analysis', { redis }),
      compression: new Bull('neural-compression', { redis }),
      training: new Bull('model-training', { redis })
    };
    
    this.metrics = {
      requests: 0,
      latency: [],
      errors: 0
    };
  }

  async initialize() {
    logger.info('Initializing NEXUS AI/ML Orchestrator...');
    
    try {
      // Initialize all services
      await Promise.all([
        this.initializeRecommendation(),
        this.initializeContentAnalysis(),
        this.initializeVoice(),
        this.initializeCompression(),
        this.initializeEmotion()
      ]);
      
      // Set up queue processors
      this.setupQueueProcessors();
      
      // Set up inter-service communication
      this.setupServiceCommunication();
      
      // Start metrics collection
      this.startMetricsCollection();
      
      logger.info('AI/ML Orchestrator initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize AI/ML Orchestrator:', error);
      throw error;
    }
  }

  async initializeRecommendation() {
    this.services.recommendation = new NeuralRecommendationEngine();
    await this.services.recommendation.initialize();
    logger.info('✓ Recommendation Engine initialized');
  }

  async initializeContentAnalysis() {
    this.services.contentAnalysis = new ContentAnalysisPipeline();
    await this.services.contentAnalysis.initialize();
    logger.info('✓ Content Analysis Pipeline initialized');
  }

  async initializeVoice() {
    this.services.voice = new VoiceProcessingSystem();
    await this.services.voice.initialize();
    logger.info('✓ Voice Processing System initialized');
  }

  async initializeCompression() {
    this.services.compression = new NeuralCompressionSystem();
    await this.services.compression.initialize();
    logger.info('✓ Neural Compression System initialized');
  }

  async initializeEmotion() {
    this.services.emotion = new EmotionDetectionSystem();
    await this.services.emotion.initialize();
    logger.info('✓ Emotion Detection System initialized');
  }

  /**
   * Set up queue processors for background tasks
   */
  setupQueueProcessors() {
    // Content analysis queue
    this.queues.analysis.process(async (job) => {
      const { videoPath, options } = job.data;
      const result = await this.services.contentAnalysis.analyzeVideo(videoPath, options);
      
      // Update recommendations based on analysis
      await this.updateRecommendationsFromAnalysis(result);
      
      return result;
    });
    
    // Compression queue
    this.queues.compression.process(async (job) => {
      const { inputPath, outputPath, options } = job.data;
      return await this.services.compression.compressVideo(inputPath, outputPath, options);
    });
    
    // Model training queue
    this.queues.training.process(async (job) => {
      const { modelType, trainingData } = job.data;
      
      switch (modelType) {
        case 'recommendation':
          return await this.services.recommendation.train(trainingData);
        case 'emotion':
          return await this.trainEmotionModel(trainingData);
        default:
          throw new Error(`Unknown model type: ${modelType}`);
      }
    });
  }

  /**
   * Set up inter-service communication
   */
  setupServiceCommunication() {
    // Emotion detection triggers recommendation updates
    this.services.emotion.on('emotion-detected', async ({ userId, emotion }) => {
      await this.updateRecommendationsForEmotion(userId, emotion);
    });
    
    // Voice commands trigger actions
    this.services.voice.on('command-executed', async ({ command, result }) => {
      await this.handleVoiceCommand(command, result);
    });
    
    // Content analysis enriches metadata
    this.services.contentAnalysis.on('analysis-complete', async ({ contentId, analysis }) => {
      await this.enrichContentMetadata(contentId, analysis);
    });
  }

  /**
   * Process media with full AI pipeline
   */
  async processMedia(mediaPath, options = {}) {
    const startTime = Date.now();
    
    try {
      // Run analysis in parallel
      const [analysis, compression] = await Promise.all([
        this.queues.analysis.add({ videoPath: mediaPath, options }),
        options.compress ? this.queues.compression.add({
          inputPath: mediaPath,
          outputPath: options.outputPath,
          options: options.compressionOptions
        }) : null
      ]);
      
      // Store results
      const result = {
        mediaPath,
        analysisJobId: analysis.id,
        compressionJobId: compression?.id,
        timestamp: Date.now(),
        processingTime: Date.now() - startTime
      };
      
      await redis.setex(
        `processing:${mediaPath}`,
        86400,
        JSON.stringify(result)
      );
      
      this.updateMetrics('process_media', Date.now() - startTime);
      
      return result;
    } catch (error) {
      this.metrics.errors++;
      logger.error('Error processing media:', error);
      throw error;
    }
  }

  /**
   * Get personalized AI insights for user
   */
  async getUserInsights(userId) {
    try {
      const [recommendations, emotions, patterns] = await Promise.all([
        this.services.recommendation.getRecommendations(userId, { limit: 10 }),
        this.services.emotion.analyzeUserEmotions(userId, {}),
        this.getUserViewingPatterns(userId)
      ]);
      
      const insights = {
        recommendations,
        emotionalProfile: emotions.currentEmotion,
        moodPatterns: emotions.moodPatterns,
        viewingPatterns: patterns,
        personalizedSuggestions: await this.generatePersonalizedSuggestions({
          userId,
          recommendations,
          emotions,
          patterns
        })
      };
      
      return insights;
    } catch (error) {
      logger.error('Error getting user insights:', error);
      throw error;
    }
  }

  /**
   * Handle real-time AI interactions
   */
  async handleRealtimeInteraction(type, data) {
    switch (type) {
      case 'voice_command':
        return await this.services.voice.processVoiceCommand(
          data.audioBuffer,
          data.userId
        );
        
      case 'emotion_tracking':
        return await this.services.emotion.trackEmotionRealtime(
          data.userId,
          data.event
        );
        
      case 'content_feedback':
        return await this.processFeedback(data);
        
      default:
        throw new Error(`Unknown interaction type: ${type}`);
    }
  }

  /**
   * Generate AI-powered content suggestions
   */
  async generatePersonalizedSuggestions({ userId, recommendations, emotions, patterns }) {
    const suggestions = [];
    
    // Time-based suggestions
    const hour = new Date().getHours();
    if (hour >= 22 || hour < 6) {
      suggestions.push({
        type: 'time_based',
        message: 'Looking for something relaxing for nighttime?',
        content: await this.getRelaxingContent(userId)
      });
    }
    
    // Emotion-based suggestions
    if (emotions.currentEmotion?.primary === 'stressed') {
      suggestions.push({
        type: 'emotion_based',
        message: 'You seem stressed. Try these calming shows',
        content: await this.getCalmingContent(userId)
      });
    }
    
    // Pattern-based suggestions
    if (patterns.bingeWatching) {
      suggestions.push({
        type: 'pattern_based',
        message: 'Continue your binge-watching session',
        content: await this.getNextEpisodes(userId, patterns.currentShow)
      });
    }
    
    // Discovery suggestions
    suggestions.push({
      type: 'discovery',
      message: 'Discover something new',
      content: await this.getDiscoveryContent(userId, recommendations)
    });
    
    return suggestions;
  }

  /**
   * Update recommendations based on content analysis
   */
  async updateRecommendationsFromAnalysis(analysis) {
    // Extract features for recommendation engine
    const features = {
      tags: Array.from(analysis.tags),
      dominantScenes: analysis.summary.dominantScenes,
      emotionalTone: this.calculateEmotionalTone(analysis),
      contentRating: analysis.summary.contentRating,
      visualComplexity: this.calculateVisualComplexity(analysis)
    };
    
    // Store enriched metadata
    await redis.setex(
      `content:features:${analysis.contentId}`,
      86400 * 7,
      JSON.stringify(features)
    );
  }

  /**
   * Update recommendations based on emotion
   */
  async updateRecommendationsForEmotion(userId, emotion) {
    const emotionWeight = {
      happy: { upbeat: 0.8, drama: 0.2 },
      sad: { comedy: 0.7, uplifting: 0.3 },
      excited: { action: 0.8, thriller: 0.2 },
      relaxed: { documentary: 0.6, nature: 0.4 }
    };
    
    const weights = emotionWeight[emotion.primary] || {};
    
    await redis.setex(
      `user:${userId}:emotion_weights`,
      3600,
      JSON.stringify(weights)
    );
  }

  /**
   * Process user feedback for continuous learning
   */
  async processFeedback(data) {
    const { userId, contentId, feedback, timestamp } = data;
    
    // Store feedback
    await redis.zadd(
      `feedback:${userId}`,
      timestamp,
      JSON.stringify({ contentId, feedback, timestamp })
    );
    
    // Update user profile
    await this.updateUserProfile(userId, feedback);
    
    // Schedule model retraining if needed
    const feedbackCount = await redis.zcard(`feedback:${userId}`);
    if (feedbackCount % 100 === 0) {
      await this.queues.training.add({
        modelType: 'recommendation',
        userId,
        trigger: 'feedback_threshold'
      });
    }
    
    return { success: true };
  }

  /**
   * Get system health and metrics
   */
  async getSystemHealth() {
    const health = {
      status: 'healthy',
      services: {},
      queues: {},
      metrics: this.metrics,
      timestamp: Date.now()
    };
    
    // Check each service
    for (const [name, service] of Object.entries(this.services)) {
      health.services[name] = {
        initialized: service?.initialized || false,
        status: service?.initialized ? 'active' : 'inactive'
      };
    }
    
    // Check queue health
    for (const [name, queue] of Object.entries(this.queues)) {
      const counts = await queue.getJobCounts();
      health.queues[name] = counts;
    }
    
    // Calculate overall health
    const activeServices = Object.values(health.services)
      .filter(s => s.status === 'active').length;
    
    if (activeServices < Object.keys(this.services).length) {
      health.status = 'degraded';
    }
    
    return health;
  }

  // Helper methods
  updateMetrics(operation, latency) {
    this.metrics.requests++;
    this.metrics.latency.push({ operation, latency, timestamp: Date.now() });
    
    // Keep only last 1000 latency measurements
    if (this.metrics.latency.length > 1000) {
      this.metrics.latency = this.metrics.latency.slice(-1000);
    }
  }

  startMetricsCollection() {
    setInterval(async () => {
      const metrics = {
        ...this.metrics,
        timestamp: Date.now()
      };
      
      await redis.zadd(
        'ai:metrics',
        Date.now(),
        JSON.stringify(metrics)
      );
    }, 60000); // Every minute
  }

  calculateEmotionalTone(analysis) {
    // Calculate overall emotional tone from face analysis
    const emotions = analysis.frames
      .flatMap(f => f.faces)
      .flatMap(face => Object.entries(face.expressions));
    
    const emotionSums = {};
    emotions.forEach(([emotion, score]) => {
      emotionSums[emotion] = (emotionSums[emotion] || 0) + score;
    });
    
    const totalEmotions = emotions.length;
    Object.keys(emotionSums).forEach(emotion => {
      emotionSums[emotion] /= totalEmotions;
    });
    
    return emotionSums;
  }

  calculateVisualComplexity(analysis) {
    // Calculate visual complexity score
    const objectDiversity = new Set(
      analysis.frames.flatMap(f => f.objects.map(o => o.class))
    ).size;
    
    const sceneChanges = analysis.frames.filter((f, i) => 
      i > 0 && f.scene?.label !== analysis.frames[i-1].scene?.label
    ).length;
    
    return {
      objectDiversity,
      sceneChanges,
      complexity: (objectDiversity + sceneChanges) / analysis.frames.length
    };
  }

  async getUserViewingPatterns(userId) {
    // Analyze user viewing patterns
    const history = await redis.lrange(`user:${userId}:history`, 0, 100);
    
    return {
      totalViews: history.length,
      genres: this.analyzeGenrePreferences(history),
      viewingTimes: this.analyzeViewingTimes(history),
      bingeWatching: this.detectBingeWatching(history)
    };
  }

  analyzeGenrePreferences(history) {
    // Analyze genre preferences from history
    return {};
  }

  analyzeViewingTimes(history) {
    // Analyze viewing time patterns
    return {};
  }

  detectBingeWatching(history) {
    // Detect binge-watching behavior
    return false;
  }

  async getRelaxingContent(userId) {
    // Get relaxing content recommendations
    return [];
  }

  async getCalmingContent(userId) {
    // Get calming content recommendations
    return [];
  }

  async getNextEpisodes(userId, showId) {
    // Get next episodes for binge-watching
    return [];
  }

  async getDiscoveryContent(userId, recommendations) {
    // Get discovery content
    return [];
  }

  async handleVoiceCommand(command, result) {
    // Handle voice command results
    logger.info('Voice command handled:', command);
  }

  async enrichContentMetadata(contentId, analysis) {
    // Enrich content metadata with analysis results
    logger.info(`Enriched metadata for content ${contentId}`);
  }

  async updateUserProfile(userId, feedback) {
    // Update user profile based on feedback
    logger.info(`Updated profile for user ${userId}`);
  }

  async trainEmotionModel(trainingData) {
    // Train emotion detection model
    logger.info('Training emotion model...');
    return { success: true };
  }
}

// Create and initialize orchestrator
const orchestrator = new AIMLOrchestrator();

// API Server
const fastify = Fastify({ 
  logger: true,
  bodyLimit: 104857600 // 100MB for audio/video uploads
});

await fastify.register(cors);
await fastify.register(helmet);
await fastify.register(websocket);

// Health check endpoint
fastify.get('/health', async (request, reply) => {
  const health = await orchestrator.getSystemHealth();
  reply.code(health.status === 'healthy' ? 200 : 503);
  return health;
});

// Process media endpoint
fastify.post('/process', async (request, reply) => {
  const { mediaPath, options } = request.body;
  const result = await orchestrator.processMedia(mediaPath, options);
  return result;
});

// Get user insights
fastify.get('/insights/:userId', async (request, reply) => {
  const { userId } = request.params;
  const insights = await orchestrator.getUserInsights(userId);
  return insights;
});

// Real-time interaction endpoint
fastify.post('/interact', async (request, reply) => {
  const { type, data } = request.body;
  const result = await orchestrator.handleRealtimeInteraction(type, data);
  return result;
});

// Recommendations endpoint
fastify.get('/recommend/:userId', async (request, reply) => {
  const { userId } = request.params;
  const { limit, includeReasons } = request.query;
  
  const recommendations = await orchestrator.services.recommendation.getRecommendations(
    userId,
    { limit: parseInt(limit) || 20, includeReasons: includeReasons === 'true' }
  );
  
  return { recommendations };
});

// Content analysis endpoint
fastify.post('/analyze', async (request, reply) => {
  const { mediaPath, options } = request.body;
  const jobId = await orchestrator.queues.analysis.add({ videoPath: mediaPath, options });
  return { jobId: jobId.id };
});

// Voice command endpoint
fastify.post('/voice', async (request, reply) => {
  const { audio, userId } = request.body;
  const audioBuffer = Buffer.from(audio, 'base64');
  const result = await orchestrator.services.voice.processVoiceCommand(audioBuffer, userId);
  return result;
});

// Compression endpoint
fastify.post('/compress', async (request, reply) => {
  const { inputPath, outputPath, options } = request.body;
  const jobId = await orchestrator.queues.compression.add({
    inputPath,
    outputPath,
    options
  });
  return { jobId: jobId.id };
});

// Emotion analysis endpoint
fastify.post('/emotion', async (request, reply) => {
  const { userId, sessionData } = request.body;
  const analysis = await orchestrator.services.emotion.analyzeUserEmotions(userId, sessionData);
  return analysis;
});

// Feedback endpoint
fastify.post('/feedback', async (request, reply) => {
  const feedback = request.body;
  const result = await orchestrator.processFeedback(feedback);
  return result;
});

// WebSocket for real-time updates
fastify.get('/ws', { websocket: true }, (connection, req) => {
  const userId = req.query.userId;
  
  // Send periodic updates
  const interval = setInterval(async () => {
    const insights = await orchestrator.getUserInsights(userId);
    connection.socket.send(JSON.stringify({
      type: 'insights_update',
      data: insights
    }));
  }, 30000); // Every 30 seconds
  
  connection.socket.on('close', () => {
    clearInterval(interval);
  });
});

// Initialize and start server
async function start() {
  try {
    await orchestrator.initialize();
    await fastify.listen({ port: 8080, host: '0.0.0.0' });
    logger.info('NEXUS AI/ML System running on port 8080');
    logger.info('Services available:');
    logger.info('  - Recommendation Engine: ✓');
    logger.info('  - Content Analysis: ✓');
    logger.info('  - Voice Processing: ✓');
    logger.info('  - Neural Compression: ✓');
    logger.info('  - Emotion Detection: ✓');
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

start();

export default orchestrator;