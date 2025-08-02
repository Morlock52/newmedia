import * as tf from '@tensorflow/tfjs-node-gpu';
import { pipeline } from '@xenova/transformers';
import Redis from 'ioredis';
import pino from 'pino';

const logger = pino({ transport: { target: 'pino-pretty' } });
const redis = new Redis();

/**
 * Neural Recommendation Engine using Collaborative Filtering and Transformers
 * Implements matrix factorization, deep learning, and transformer-based embeddings
 */
export class NeuralRecommendationEngine {
  constructor() {
    this.model = null;
    this.userEmbeddings = new Map();
    this.itemEmbeddings = new Map();
    this.sentenceEncoder = null;
    this.initialized = false;
  }

  async initialize() {
    logger.info('Initializing Neural Recommendation Engine...');
    
    // Initialize transformer for content understanding
    this.sentenceEncoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    
    // Build neural collaborative filtering model
    this.model = await this.buildModel();
    
    // Load cached embeddings
    await this.loadEmbeddings();
    
    this.initialized = true;
    logger.info('Neural Recommendation Engine initialized successfully');
  }

  /**
   * Build deep neural network for collaborative filtering
   */
  async buildModel() {
    const userInput = tf.input({ shape: [1], name: 'user_id' });
    const itemInput = tf.input({ shape: [1], name: 'item_id' });
    
    // User embedding layer
    const userEmbedding = tf.layers.embedding({
      inputDim: 10000,
      outputDim: 128,
      name: 'user_embedding'
    }).apply(userInput);
    
    // Item embedding layer
    const itemEmbedding = tf.layers.embedding({
      inputDim: 100000,
      outputDim: 128,
      name: 'item_embedding'
    }).apply(itemInput);
    
    // Flatten embeddings
    const userVector = tf.layers.flatten().apply(userEmbedding);
    const itemVector = tf.layers.flatten().apply(itemEmbedding);
    
    // Concatenate user and item vectors
    const concat = tf.layers.concatenate().apply([userVector, itemVector]);
    
    // Deep neural network layers
    let dense = tf.layers.dense({
      units: 512,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }).apply(concat);
    
    dense = tf.layers.dropout({ rate: 0.3 }).apply(dense);
    
    dense = tf.layers.dense({
      units: 256,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }).apply(dense);
    
    dense = tf.layers.dropout({ rate: 0.2 }).apply(dense);
    
    dense = tf.layers.dense({
      units: 128,
      activation: 'relu'
    }).apply(dense);
    
    // Output layer with sigmoid for rating prediction
    const output = tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
      name: 'rating'
    }).apply(dense);
    
    const model = tf.model({
      inputs: [userInput, itemInput],
      outputs: output
    });
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy', 'mse']
    });
    
    return model;
  }

  /**
   * Generate content embeddings using transformers
   */
  async generateContentEmbedding(content) {
    const { title, description, genres, tags } = content;
    const text = `${title} ${description} ${genres.join(' ')} ${tags.join(' ')}`;
    
    const output = await this.sentenceEncoder(text);
    const embedding = Array.from(output.data);
    
    return embedding;
  }

  /**
   * Get personalized recommendations using hybrid approach
   */
  async getRecommendations(userId, options = {}) {
    const {
      limit = 20,
      includeReasons = true,
      diversityFactor = 0.3,
      excludeWatched = true
    } = options;
    
    try {
      // Get user profile and history
      const userProfile = await this.getUserProfile(userId);
      const watchHistory = await this.getWatchHistory(userId);
      
      // Generate collaborative filtering recommendations
      const cfRecommendations = await this.collaborativeFiltering(userId, userProfile);
      
      // Generate content-based recommendations
      const cbRecommendations = await this.contentBasedFiltering(userProfile, watchHistory);
      
      // Generate trending recommendations
      const trendingItems = await this.getTrendingItems();
      
      // Hybrid recommendation fusion
      const recommendations = await this.fuseRecommendations({
        collaborative: cfRecommendations,
        contentBased: cbRecommendations,
        trending: trendingItems,
        weights: { collaborative: 0.5, contentBased: 0.3, trending: 0.2 },
        diversityFactor
      });
      
      // Apply business rules and filters
      const filtered = this.applyFilters(recommendations, {
        excludeWatched,
        userProfile,
        watchHistory
      });
      
      // Add explanation for recommendations
      if (includeReasons) {
        return this.addRecommendationReasons(filtered.slice(0, limit), userProfile);
      }
      
      return filtered.slice(0, limit);
    } catch (error) {
      logger.error('Error generating recommendations:', error);
      throw error;
    }
  }

  /**
   * Collaborative filtering using neural network
   */
  async collaborativeFiltering(userId, userProfile) {
    const allItems = await this.getAllItems();
    const predictions = [];
    
    // Batch prediction for efficiency
    const batchSize = 1000;
    for (let i = 0; i < allItems.length; i += batchSize) {
      const batch = allItems.slice(i, i + batchSize);
      const userIds = new Array(batch.length).fill(userId);
      const itemIds = batch.map(item => item.id);
      
      const userTensor = tf.tensor2d(userIds, [batch.length, 1]);
      const itemTensor = tf.tensor2d(itemIds, [batch.length, 1]);
      
      const scores = await this.model.predict([userTensor, itemTensor]).array();
      
      batch.forEach((item, idx) => {
        predictions.push({
          ...item,
          score: scores[idx][0],
          method: 'collaborative'
        });
      });
      
      userTensor.dispose();
      itemTensor.dispose();
    }
    
    return predictions.sort((a, b) => b.score - a.score);
  }

  /**
   * Content-based filtering using embeddings
   */
  async contentBasedFiltering(userProfile, watchHistory) {
    // Calculate user preference embedding
    const userEmbedding = await this.calculateUserPreferenceEmbedding(watchHistory);
    
    // Get all items with embeddings
    const allItems = await this.getAllItemsWithEmbeddings();
    
    // Calculate cosine similarity
    const recommendations = allItems.map(item => {
      const similarity = this.cosineSimilarity(userEmbedding, item.embedding);
      return {
        ...item,
        score: similarity,
        method: 'content'
      };
    });
    
    return recommendations.sort((a, b) => b.score - a.score);
  }

  /**
   * Calculate user preference embedding from watch history
   */
  async calculateUserPreferenceEmbedding(watchHistory) {
    const embeddings = await Promise.all(
      watchHistory.map(async item => {
        const cached = await redis.get(`embedding:${item.id}`);
        if (cached) return JSON.parse(cached);
        
        const embedding = await this.generateContentEmbedding(item);
        await redis.setex(`embedding:${item.id}`, 3600, JSON.stringify(embedding));
        return embedding;
      })
    );
    
    // Weight recent items more heavily
    const weights = watchHistory.map((_, idx) => 
      Math.exp(-idx * 0.1) // Exponential decay
    );
    
    // Calculate weighted average
    const avgEmbedding = new Array(embeddings[0].length).fill(0);
    embeddings.forEach((emb, idx) => {
      emb.forEach((val, j) => {
        avgEmbedding[j] += val * weights[idx];
      });
    });
    
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    return avgEmbedding.map(val => val / totalWeight);
  }

  /**
   * Cosine similarity calculation
   */
  cosineSimilarity(vec1, vec2) {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
  }

  /**
   * Get trending items using time-decay algorithm
   */
  async getTrendingItems() {
    const recentViews = await redis.zrevrange('trending:views', 0, 100, 'WITHSCORES');
    const items = [];
    
    for (let i = 0; i < recentViews.length; i += 2) {
      const itemId = recentViews[i];
      const score = parseFloat(recentViews[i + 1]);
      const item = await this.getItem(itemId);
      
      if (item) {
        items.push({
          ...item,
          score: score,
          method: 'trending'
        });
      }
    }
    
    return items;
  }

  /**
   * Fuse multiple recommendation sources
   */
  async fuseRecommendations({ collaborative, contentBased, trending, weights, diversityFactor }) {
    const scoreMap = new Map();
    
    // Normalize and combine scores
    const normalize = (items, weight) => {
      const maxScore = Math.max(...items.map(item => item.score));
      items.forEach(item => {
        const normalizedScore = (item.score / maxScore) * weight;
        const current = scoreMap.get(item.id) || { item, score: 0, methods: [] };
        current.score += normalizedScore;
        current.methods.push(item.method);
        scoreMap.set(item.id, current);
      });
    };
    
    normalize(collaborative, weights.collaborative);
    normalize(contentBased, weights.contentBased);
    normalize(trending, weights.trending);
    
    // Apply diversity boost
    const recommendations = Array.from(scoreMap.values());
    if (diversityFactor > 0) {
      this.applyDiversityBoost(recommendations, diversityFactor);
    }
    
    return recommendations.sort((a, b) => b.score - a.score);
  }

  /**
   * Apply diversity boost to recommendations
   */
  applyDiversityBoost(recommendations, factor) {
    const genreCount = new Map();
    
    recommendations.forEach((rec, idx) => {
      const genres = rec.item.genres || [];
      let diversityPenalty = 0;
      
      genres.forEach(genre => {
        const count = genreCount.get(genre) || 0;
        diversityPenalty += count * factor;
        genreCount.set(genre, count + 1);
      });
      
      rec.score *= Math.exp(-diversityPenalty);
    });
  }

  /**
   * Add explanations for recommendations
   */
  async addRecommendationReasons(recommendations, userProfile) {
    return recommendations.map(rec => {
      const reasons = [];
      
      if (rec.methods.includes('collaborative')) {
        reasons.push('Users with similar taste also enjoyed this');
      }
      
      if (rec.methods.includes('content')) {
        reasons.push('Based on your viewing history');
      }
      
      if (rec.methods.includes('trending')) {
        reasons.push('Trending now');
      }
      
      // Add specific genre/actor/director matches
      const matchedGenres = rec.item.genres.filter(g => 
        userProfile.preferredGenres.includes(g)
      );
      
      if (matchedGenres.length > 0) {
        reasons.push(`Matches your interest in ${matchedGenres.join(', ')}`);
      }
      
      return {
        ...rec,
        reasons
      };
    });
  }

  /**
   * Train the model with new interaction data
   */
  async train(interactions) {
    logger.info(`Training model with ${interactions.length} interactions`);
    
    const userIds = interactions.map(i => i.userId);
    const itemIds = interactions.map(i => i.itemId);
    const ratings = interactions.map(i => i.rating);
    
    const userTensor = tf.tensor2d(userIds, [interactions.length, 1]);
    const itemTensor = tf.tensor2d(itemIds, [interactions.length, 1]);
    const ratingTensor = tf.tensor2d(ratings, [interactions.length, 1]);
    
    const history = await this.model.fit(
      [userTensor, itemTensor],
      ratingTensor,
      {
        epochs: 10,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            logger.info(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`);
          }
        }
      }
    );
    
    userTensor.dispose();
    itemTensor.dispose();
    ratingTensor.dispose();
    
    // Save model
    await this.model.save('file://./models/recommendation-model');
    
    return history;
  }

  // Helper methods
  async getUserProfile(userId) {
    const cached = await redis.get(`user:${userId}:profile`);
    return cached ? JSON.parse(cached) : this.buildUserProfile(userId);
  }

  async getWatchHistory(userId) {
    const history = await redis.lrange(`user:${userId}:history`, 0, 100);
    return history.map(item => JSON.parse(item));
  }

  async getAllItems() {
    // This would connect to your media database
    return [];
  }

  async getAllItemsWithEmbeddings() {
    const items = await this.getAllItems();
    return Promise.all(items.map(async item => ({
      ...item,
      embedding: await this.generateContentEmbedding(item)
    })));
  }

  async getItem(itemId) {
    const cached = await redis.get(`item:${itemId}`);
    return cached ? JSON.parse(cached) : null;
  }

  async loadEmbeddings() {
    // Load pre-computed embeddings from storage
    logger.info('Loading pre-computed embeddings...');
  }

  applyFilters(recommendations, filters) {
    return recommendations.filter(rec => {
      if (filters.excludeWatched) {
        const watched = filters.watchHistory.some(h => h.id === rec.item.id);
        if (watched) return false;
      }
      return true;
    });
  }

  async buildUserProfile(userId) {
    // Build user profile from historical data
    return {
      userId,
      preferredGenres: [],
      preferredActors: [],
      averageRating: 0,
      viewingPatterns: {}
    };
  }
}

// API Server
import Fastify from 'fastify';
import cors from '@fastify/cors';

const fastify = Fastify({ logger: true });
await fastify.register(cors);

const engine = new NeuralRecommendationEngine();

fastify.get('/recommendations/:userId', async (request, reply) => {
  const { userId } = request.params;
  const { limit, includeReasons } = request.query;
  
  const recommendations = await engine.getRecommendations(userId, {
    limit: parseInt(limit) || 20,
    includeReasons: includeReasons === 'true'
  });
  
  return { recommendations };
});

fastify.post('/train', async (request, reply) => {
  const { interactions } = request.body;
  const history = await engine.train(interactions);
  return { success: true, history };
});

fastify.post('/feedback', async (request, reply) => {
  const { userId, itemId, action, rating } = request.body;
  
  // Store feedback for future training
  await redis.zadd(`user:${userId}:feedback`, Date.now(), JSON.stringify({
    itemId, action, rating, timestamp: Date.now()
  }));
  
  return { success: true };
});

// Initialize and start server
async function start() {
  await engine.initialize();
  await fastify.listen({ port: 8081, host: '0.0.0.0' });
  logger.info('Recommendation Engine running on port 8081');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  start().catch(console.error);
}

export default engine;