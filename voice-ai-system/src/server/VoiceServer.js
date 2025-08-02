/**
 * Advanced Voice AI Server with LLM Integration
 * Handles real-time voice processing, emotion detection, and multi-language support
 */

import express from 'express';
import { createServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import WebSocket, { WebSocketServer } from 'ws';
import cors from 'cors';
import dotenv from 'dotenv';
import { EmotionDetector } from './services/EmotionDetector.js';
import { LanguageProcessor } from './services/LanguageProcessor.js';
import { BiometricAuth } from './services/BiometricAuth.js';
import { LLMProcessor } from './services/LLMProcessor.js';
import { MediaLibraryService } from './services/MediaLibraryService.js';
import { SpeechService } from './services/SpeechService.js';
import { TranslationService } from './services/TranslationService.js';

dotenv.config();

export class VoiceServer {
  constructor(config = {}) {
    this.config = {
      port: process.env.PORT || 8080,
      corsOrigin: process.env.CORS_ORIGIN || "*",
      maxConnections: process.env.MAX_CONNECTIONS || 1000,
      rateLimit: process.env.RATE_LIMIT || 100,
      ...config
    };

    // Initialize Express app
    this.app = express();
    this.server = createServer(this.app);
    
    // Initialize Socket.IO
    this.io = new SocketIOServer(this.server, {
      cors: {
        origin: this.config.corsOrigin,
        methods: ["GET", "POST"]
      }
    });

    // Initialize WebSocket server for voice streaming
    this.wss = new WebSocketServer({ 
      server: this.server,
      path: '/voice'
    });

    // Initialize services
    this.emotionDetector = new EmotionDetector();
    this.languageProcessor = new LanguageProcessor();
    this.biometricAuth = new BiometricAuth();
    this.llmProcessor = new LLMProcessor();
    this.mediaLibrary = new MediaLibraryService();
    this.speechService = new SpeechService();
    this.translationService = new TranslationService();

    // Active sessions
    this.activeSessions = new Map();
    this.connectionCount = 0;

    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocketHandlers();
    this.setupSocketIOHandlers();
  }

  /**
   * Setup Express middleware
   */
  setupMiddleware() {
    this.app.use(cors({
      origin: this.config.corsOrigin
    }));
    
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));
    
    // Rate limiting middleware
    this.app.use((req, res, next) => {
      // Simple rate limiting implementation
      const clientId = req.ip;
      const now = Date.now();
      
      if (!this.rateLimitMap) {
        this.rateLimitMap = new Map();
      }
      
      const clientRequests = this.rateLimitMap.get(clientId) || [];
      const recentRequests = clientRequests.filter(time => now - time < 60000);
      
      if (recentRequests.length >= this.config.rateLimit) {
        return res.status(429).json({ error: 'Rate limit exceeded' });
      }
      
      recentRequests.push(now);
      this.rateLimitMap.set(clientId, recentRequests);
      
      next();
    });

    // Logging middleware
    this.app.use((req, res, next) => {
      console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
      next();
    });
  }

  /**
   * Setup REST API routes
   */
  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        connections: this.connectionCount,
        activeSessions: this.activeSessions.size,
        services: {
          emotionDetector: this.emotionDetector.isReady(),
          languageProcessor: this.languageProcessor.isReady(),
          biometricAuth: this.biometricAuth.isReady(),
          llmProcessor: this.llmProcessor.isReady(),
          mediaLibrary: this.mediaLibrary.isReady()
        }
      });
    });

    // Get supported languages
    this.app.get('/api/languages', (req, res) => {
      res.json({
        languages: this.languageProcessor.getSupportedLanguages(),
        count: this.languageProcessor.getSupportedLanguages().length
      });
    });

    // Text-to-speech endpoint
    this.app.post('/api/tts', async (req, res) => {
      try {
        const { text, language = 'en-US', voice, emotion } = req.body;
        
        if (!text) {
          return res.status(400).json({ error: 'Text is required' });
        }

        const audioBuffer = await this.speechService.synthesize({
          text,
          language,
          voice,
          emotion
        });

        res.set({
          'Content-Type': 'audio/wav',
          'Content-Length': audioBuffer.length
        });

        res.send(audioBuffer);
      } catch (error) {
        console.error('TTS error:', error);
        res.status(500).json({ error: 'Text-to-speech failed' });
      }
    });

    // Translation endpoint
    this.app.post('/api/translate', async (req, res) => {
      try {
        const { text, targetLanguage, sourceLanguage } = req.body;
        
        if (!text || !targetLanguage) {
          return res.status(400).json({ error: 'Text and target language are required' });
        }

        const result = await this.translationService.translate({
          text,
          targetLanguage,
          sourceLanguage
        });

        res.json(result);
      } catch (error) {
        console.error('Translation error:', error);
        res.status(500).json({ error: 'Translation failed' });
      }
    });

    // Media library search
    this.app.get('/api/media/search', async (req, res) => {
      try {
        const { query, type, language } = req.query;
        
        const results = await this.mediaLibrary.search({
          query,
          type,
          language
        });

        res.json(results);
      } catch (error) {
        console.error('Media search error:', error);
        res.status(500).json({ error: 'Media search failed' });
      }
    });

    // Voice biometric enrollment
    this.app.post('/api/biometric/enroll', async (req, res) => {
      try {
        const { speakerId, audioData } = req.body;
        
        const result = await this.biometricAuth.enroll({
          speakerId,
          audioData
        });

        res.json(result);
      } catch (error) {
        console.error('Biometric enrollment error:', error);
        res.status(500).json({ error: 'Biometric enrollment failed' });
      }
    });

    // Content summarization endpoints
    this.app.post('/api/media/summarize', async (req, res) => {
      try {
        const { mediaId, length = 'medium', style = 'informative', language = 'en' } = req.body;
        
        if (!mediaId) {
          return res.status(400).json({ error: 'Media ID is required' });
        }

        const summary = await this.mediaLibrary.summarizeMediaContent(mediaId, {
          length,
          style,
          language
        });

        res.json(summary);
      } catch (error) {
        console.error('Content summarization error:', error);
        res.status(500).json({ error: 'Content summarization failed' });
      }
    });

    // Batch summarization
    this.app.post('/api/media/batch-summarize', async (req, res) => {
      try {
        const { mediaIds, options = {} } = req.body;
        
        if (!mediaIds || !Array.isArray(mediaIds)) {
          return res.status(400).json({ error: 'Media IDs array is required' });
        }

        const results = await this.mediaLibrary.batchSummarizeMedia(mediaIds, options);
        res.json(results);
      } catch (error) {
        console.error('Batch summarization error:', error);
        res.status(500).json({ error: 'Batch summarization failed' });
      }
    });

    // Generate content highlights
    this.app.post('/api/media/highlights', async (req, res) => {
      try {
        const { mediaId, maxHighlights = 5 } = req.body;
        
        if (!mediaId) {
          return res.status(400).json({ error: 'Media ID is required' });
        }

        const highlights = await this.mediaLibrary.generateMediaHighlights(mediaId, maxHighlights);
        res.json(highlights);
      } catch (error) {
        console.error('Highlight generation error:', error);
        res.status(500).json({ error: 'Highlight generation failed' });
      }
    });

    // Direct LLM content summarization
    this.app.post('/api/llm/summarize', async (req, res) => {
      try {
        const { content, type = 'auto', length = 'medium', style = 'informative', language = 'en' } = req.body;
        
        if (!content) {
          return res.status(400).json({ error: 'Content is required' });
        }

        const summary = await this.llmProcessor.summarizeContent({
          content,
          type,
          length,
          style,
          language
        });

        res.json(summary);
      } catch (error) {
        console.error('LLM summarization error:', error);
        res.status(500).json({ error: 'LLM summarization failed' });
      }
    });

    // Serve static files
    this.app.use(express.static('public'));
  }

  /**
   * Setup WebSocket handlers for real-time voice processing
   */
  setupWebSocketHandlers() {
    this.wss.on('connection', (ws, request) => {
      this.connectionCount++;
      console.log(`WebSocket connection established. Total connections: ${this.connectionCount}`);

      let sessionData = {
        id: null,
        language: 'en-US',
        isListening: false,
        emotionDetection: true,
        biometrics: false,
        audioBuffer: [],
        lastActivity: Date.now()
      };

      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());
          await this.handleWebSocketMessage(ws, message, sessionData);
        } catch (error) {
          console.error('WebSocket message error:', error);
          ws.send(JSON.stringify({
            type: 'error',
            error: error.message
          }));
        }
      });

      ws.on('close', () => {
        this.connectionCount--;
        if (sessionData.id) {
          this.activeSessions.delete(sessionData.id);
        }
        console.log(`WebSocket connection closed. Total connections: ${this.connectionCount}`);
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });

      // Send welcome message
      ws.send(JSON.stringify({
        type: 'connection_established',
        timestamp: new Date().toISOString()
      }));
    });
  }

  /**
   * Handle WebSocket messages
   */
  async handleWebSocketMessage(ws, message, sessionData) {
    sessionData.lastActivity = Date.now();

    switch (message.type) {
      case 'session_start':
        sessionData.id = message.sessionId;
        sessionData.language = message.config.language || 'en-US';
        sessionData.emotionDetection = message.config.emotionDetection;
        sessionData.biometrics = message.config.biometrics;
        
        this.activeSessions.set(sessionData.id, sessionData);
        
        ws.send(JSON.stringify({
          type: 'session_started',
          sessionId: sessionData.id
        }));
        break;

      case 'audio_chunk':
        await this.processAudioChunk(ws, message, sessionData);
        break;

      case 'start_listening':
        sessionData.isListening = true;
        sessionData.audioBuffer = [];
        break;

      case 'stop_listening':
        sessionData.isListening = false;
        if (sessionData.audioBuffer.length > 0) {
          await this.finalizeAudioProcessing(ws, sessionData);
        }
        break;

      case 'update_language':
        sessionData.language = message.language;
        break;

      case 'translate_request':
        await this.handleTranslationRequest(ws, message, sessionData);
        break;

      case 'llm_process':
        await this.handleLLMRequest(ws, message, sessionData);
        break;

      case 'enroll_biometric':
        await this.handleBiometricEnrollment(ws, message, sessionData);
        break;

      default:
        ws.send(JSON.stringify({
          type: 'error',
          error: `Unknown message type: ${message.type}`
        }));
    }
  }

  /**
   * Process audio chunk for real-time analysis
   */
  async processAudioChunk(ws, message, sessionData) {
    if (!sessionData.isListening) return;

    const audioData = new Int16Array(message.data);
    sessionData.audioBuffer.push(...audioData);

    // Perform real-time transcription
    const transcriptionResult = await this.speechService.transcribeRealTime({
      audioData,
      language: sessionData.language,
      sessionId: sessionData.id
    });

    if (transcriptionResult.partial) {
      ws.send(JSON.stringify({
        type: 'transcription_partial',
        text: transcriptionResult.text,
        confidence: transcriptionResult.confidence,
        language: transcriptionResult.language
      }));
    }

    if (transcriptionResult.final) {
      // Process with additional services
      const results = await Promise.all([
        // Emotion detection
        sessionData.emotionDetection ? 
          this.emotionDetector.analyze(audioData, transcriptionResult.text) : 
          Promise.resolve(null),
        
        // Language detection
        this.languageProcessor.detectLanguage(transcriptionResult.text),
        
        // Speaker verification (if enabled)
        sessionData.biometrics ? 
          this.biometricAuth.verify(audioData, sessionData.id) : 
          Promise.resolve(null)
      ]);

      const [emotionResult, languageResult, biometricResult] = results;

      ws.send(JSON.stringify({
        type: 'transcription_final',
        text: transcriptionResult.text,
        confidence: transcriptionResult.confidence,
        language: languageResult?.language || sessionData.language,
        timestamp: Date.now(),
        emotion: emotionResult?.emotion,
        sentiment: emotionResult?.sentiment,
        speaker: biometricResult?.speakerId
      }));

      // Send individual results if available
      if (emotionResult) {
        ws.send(JSON.stringify({
          type: 'emotion_detected',
          emotion: emotionResult.emotion,
          confidence: emotionResult.confidence,
          sentiment: emotionResult.sentiment,
          timestamp: Date.now()
        }));
      }

      if (languageResult && languageResult.language !== sessionData.language) {
        ws.send(JSON.stringify({
          type: 'language_detected',
          language: languageResult.language,
          confidence: languageResult.confidence
        }));
      }

      if (biometricResult) {
        ws.send(JSON.stringify({
          type: 'speaker_verified',
          verified: biometricResult.verified,
          confidence: biometricResult.confidence,
          speakerId: biometricResult.speakerId
        }));
      }
    }
  }

  /**
   * Handle translation request
   */
  async handleTranslationRequest(ws, message, sessionData) {
    try {
      const result = await this.translationService.translate({
        text: message.text,
        targetLanguage: message.targetLanguage,
        sourceLanguage: message.sourceLanguage
      });

      ws.send(JSON.stringify({
        type: 'translation_result',
        originalText: message.text,
        translatedText: result.translatedText,
        sourceLanguage: result.sourceLanguage,
        targetLanguage: message.targetLanguage,
        confidence: result.confidence
      }));
    } catch (error) {
      ws.send(JSON.stringify({
        type: 'error',
        error: `Translation failed: ${error.message}`
      }));
    }
  }

  /**
   * Handle LLM processing request
   */
  async handleLLMRequest(ws, message, sessionData) {
    try {
      const response = await this.llmProcessor.process({
        text: message.text,
        context: message.context,
        sessionId: sessionData.id
      });

      ws.send(JSON.stringify({
        type: 'llm_response',
        query: message.text,
        response: response.text,
        model: response.model,
        confidence: response.confidence,
        contextUsed: response.contextUsed
      }));
    } catch (error) {
      ws.send(JSON.stringify({
        type: 'error',
        error: `LLM processing failed: ${error.message}`
      }));
    }
  }

  /**
   * Handle biometric enrollment
   */
  async handleBiometricEnrollment(ws, message, sessionData) {
    try {
      const result = await this.biometricAuth.enroll({
        speakerId: message.speakerId,
        audioData: sessionData.audioBuffer,
        sessionId: sessionData.id
      });

      ws.send(JSON.stringify({
        type: 'biometric_enrolled',
        success: result.success,
        speakerId: message.speakerId,
        confidence: result.confidence
      }));
    } catch (error) {
      ws.send(JSON.stringify({
        type: 'error',
        error: `Biometric enrollment failed: ${error.message}`
      }));
    }
  }

  /**
   * Setup Socket.IO handlers for additional features
   */
  setupSocketIOHandlers() {
    this.io.on('connection', (socket) => {
      console.log('Socket.IO client connected:', socket.id);

      // Media library conversation
      socket.on('media_conversation', async (data) => {
        try {
          const response = await this.mediaLibrary.conversationalSearch(data);
          socket.emit('media_response', response);
        } catch (error) {
          socket.emit('error', { message: error.message });
        }
      });

      // Voice-controlled media editing
      socket.on('voice_edit_command', async (data) => {
        try {
          const result = await this.processVoiceEditCommand(data);
          socket.emit('edit_result', result);
        } catch (error) {
          socket.emit('error', { message: error.message });
        }
      });

      // Real-time content summarization
      socket.on('summarize_content', async (data) => {
        try {
          const { content, options = {} } = data;
          
          if (!content) {
            socket.emit('error', { message: 'Content is required for summarization' });
            return;
          }

          // Emit progress update
          socket.emit('summarization_progress', { status: 'analyzing', progress: 25 });

          const summary = await this.llmProcessor.summarizeContent({
            content,
            type: options.type || 'auto',
            length: options.length || 'medium',
            style: options.style || 'informative',
            language: options.language || 'en'
          });

          socket.emit('summarization_progress', { status: 'complete', progress: 100 });
          socket.emit('content_summarized', summary);
        } catch (error) {
          socket.emit('error', { message: error.message });
        }
      });

      // Media content summarization
      socket.on('summarize_media', async (data) => {
        try {
          const { mediaId, options = {} } = data;
          
          if (!mediaId) {
            socket.emit('error', { message: 'Media ID is required' });
            return;
          }

          socket.emit('summarization_progress', { status: 'extracting', progress: 20 });
          
          const summary = await this.mediaLibrary.summarizeMediaContent(mediaId, options);
          
          socket.emit('summarization_progress', { status: 'complete', progress: 100 });
          socket.emit('media_summarized', summary);
        } catch (error) {
          socket.emit('error', { message: error.message });
        }
      });

      // Batch media summarization with progress
      socket.on('batch_summarize_media', async (data) => {
        try {
          const { mediaIds, options = {} } = data;
          
          if (!mediaIds || !Array.isArray(mediaIds)) {
            socket.emit('error', { message: 'Media IDs array is required' });
            return;
          }

          const totalItems = mediaIds.length;
          let processedItems = 0;

          // Override batch processing to emit progress
          const results = [];
          const batchSize = options.batchSize || 3;
          
          for (let i = 0; i < mediaIds.length; i += batchSize) {
            const batch = mediaIds.slice(i, i + batchSize);
            
            const batchPromises = batch.map(async (mediaId) => {
              try {
                const summary = await this.mediaLibrary.summarizeMediaContent(mediaId, options);
                processedItems++;
                
                const progress = Math.round((processedItems / totalItems) * 100);
                socket.emit('batch_summarization_progress', { 
                  processed: processedItems, 
                  total: totalItems, 
                  progress 
                });
                
                return { mediaId, summary };
              } catch (error) {
                processedItems++;
                return { mediaId, error: error.message };
              }
            });

            const batchResults = await Promise.all(batchPromises);
            results.push(...batchResults);
            
            // Add delay between batches
            if (i + batchSize < mediaIds.length) {
              await new Promise(resolve => setTimeout(resolve, 1000));
            }
          }

          socket.emit('batch_summarization_complete', {
            summaries: results,
            totalProcessed: mediaIds.length,
            successCount: results.filter(r => !r.error).length,
            errorCount: results.filter(r => r.error).length
          });
        } catch (error) {
          socket.emit('error', { message: error.message });
        }
      });

      // Generate content highlights
      socket.on('generate_highlights', async (data) => {
        try {
          const { mediaId, maxHighlights = 5 } = data;
          
          if (!mediaId) {
            socket.emit('error', { message: 'Media ID is required' });
            return;
          }

          const highlights = await this.mediaLibrary.generateMediaHighlights(mediaId, maxHighlights);
          socket.emit('highlights_generated', highlights);
        } catch (error) {
          socket.emit('error', { message: error.message });
        }
      });

      socket.on('disconnect', () => {
        console.log('Socket.IO client disconnected:', socket.id);
      });
    });
  }

  /**
   * Process voice editing commands
   */
  async processVoiceEditCommand(data) {
    const { command, mediaId, parameters } = data;
    
    // Use LLM to parse and understand the voice command
    const parsedCommand = await this.llmProcessor.parseEditCommand({
      command,
      context: { mediaId, parameters }
    });

    // Execute the parsed command
    return await this.mediaLibrary.executeEditCommand(parsedCommand);
  }

  /**
   * Start the server
   */
  async start() {
    try {
      // Initialize all services
      await Promise.all([
        this.emotionDetector.initialize(),
        this.languageProcessor.initialize(),
        this.biometricAuth.initialize(),
        this.llmProcessor.initialize(),
        this.mediaLibrary.initialize(),
        this.speechService.initialize(),
        this.translationService.initialize()
      ]);

      // Wire up service dependencies for advanced features
      this.mediaLibrary.setLLMProcessor(this.llmProcessor);
      
      console.log('✅ All services initialized and wired successfully');

      // Start server
      this.server.listen(this.config.port, () => {
        console.log(`🎙️  Voice AI Server running on port ${this.config.port}`);
        console.log(`🌐 WebSocket endpoint: ws://localhost:${this.config.port}/voice`);
        console.log(`🔌 Socket.IO endpoint: http://localhost:${this.config.port}/socket.io/`);
        console.log(`📚 API docs: http://localhost:${this.config.port}/health`);
      });

      // Setup cleanup on exit
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());

    } catch (error) {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    console.log('Shutting down voice AI server...');
    
    // Close WebSocket connections
    this.wss.clients.forEach(ws => {
      ws.close(1000, 'Server shutdown');
    });

    // Close Socket.IO connections
    this.io.close();

    // Close HTTP server
    this.server.close();

    // Cleanup services
    await Promise.all([
      this.emotionDetector.cleanup(),
      this.languageProcessor.cleanup(),
      this.biometricAuth.cleanup(),
      this.llmProcessor.cleanup(),
      this.mediaLibrary.cleanup(),
      this.speechService.cleanup(),
      this.translationService.cleanup()
    ]);

    console.log('Server shutdown complete');
    process.exit(0);
  }
}

export default VoiceServer;