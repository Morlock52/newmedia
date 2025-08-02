/**
 * Advanced Voice Client
 * Handles client-side integration with the voice AI system
 */

import { VoiceEngine } from '../core/VoiceEngine.js';

export class VoiceClient {
  constructor(config = {}) {
    this.config = {
      serverUrl: config.serverUrl || 'ws://localhost:8080',
      voiceWsPath: config.voiceWsPath || '/voice',
      socketIOPath: config.socketIOPath || '/socket.io/',
      
      // Connection settings
      reconnectAttempts: config.reconnectAttempts || 5,
      reconnectDelay: config.reconnectDelay || 2000,
      
      // Voice settings
      sampleRate: config.sampleRate || 16000,
      channels: config.channels || 1,
      language: config.language || 'en-US',
      
      // Feature flags
      enableEmotionDetection: config.enableEmotionDetection !== false,
      enableBiometrics: config.enableBiometrics !== false,
      enableRealTimeTranslation: config.enableRealTimeTranslation !== false,
      enableLLMProcessing: config.enableLLMProcessing !== false,
      
      ...config
    };

    // Connection states
    this.isConnected = false;
    this.reconnectCount = 0;
    
    // WebSocket connections
    this.voiceWebSocket = null;
    this.socketIO = null;
    
    // Voice engine
    this.voiceEngine = null;
    
    // Session state
    this.sessionId = null;
    this.isListening = false;
    this.currentLanguage = this.config.language;
    
    // Event callbacks
    this.callbacks = {
      onConnectionChange: null,
      onTranscription: null,
      onEmotionDetected: null,
      onTranslation: null,
      onMediaResponse: null,
      onBiometricResult: null,
      onError: null
    };

    // Initialize
    this.init();
  }

  /**
   * Initialize the voice client
   */
  async init() {
    try {
      // Create voice engine
      this.voiceEngine = new VoiceEngine({
        sampleRate: this.config.sampleRate,
        channels: this.config.channels,
        llmProcessing: this.config.enableLLMProcessing,
        emotionDetection: this.config.enableEmotionDetection,
        realtimeTranscription: true
      });

      // Setup voice engine event handlers
      this.setupVoiceEngineHandlers();

      console.log('Voice client initialized');
    } catch (error) {
      console.error('Failed to initialize voice client:', error);
      this.handleError('initialization', error);
    }
  }

  /**
   * Connect to voice services
   */
  async connect() {
    try {
      console.log('Connecting to voice services...');
      
      // Initialize voice engine
      const engineInit = await this.voiceEngine.initialize();
      if (!engineInit) {
        throw new Error('Failed to initialize voice engine');
      }

      // Connect WebSocket for voice streaming
      await this.connectVoiceWebSocket();
      
      // Connect Socket.IO for additional features
      await this.connectSocketIO();
      
      this.isConnected = true;
      this.reconnectCount = 0;
      
      console.log('Connected to voice services');
      this.notifyConnectionChange(true);
      
      return true;
    } catch (error) {
      console.error('Failed to connect to voice services:', error);
      this.handleError('connection', error);
      
      // Attempt reconnection
      if (this.reconnectCount < this.config.reconnectAttempts) {
        setTimeout(() => this.reconnect(), this.config.reconnectDelay);
      }
      
      return false;
    }
  }

  /**
   * Connect voice WebSocket
   */
  async connectVoiceWebSocket() {
    return new Promise((resolve, reject) => {
      const wsUrl = `${this.config.serverUrl}${this.config.voiceWsPath}`;
      
      try {
        this.voiceWebSocket = new WebSocket(wsUrl);
        
        this.voiceWebSocket.onopen = () => {
          console.log('Voice WebSocket connected');
          
          // Initialize session
          this.initializeVoiceSession();
          resolve();
        };
        
        this.voiceWebSocket.onmessage = (event) => {
          this.handleVoiceWebSocketMessage(JSON.parse(event.data));
        };
        
        this.voiceWebSocket.onerror = (error) => {
          console.error('Voice WebSocket error:', error);
          reject(error);
        };
        
        this.voiceWebSocket.onclose = (event) => {
          console.log('Voice WebSocket closed:', event.code, event.reason);
          this.handleVoiceWebSocketClose();
        };
        
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Connect Socket.IO for additional features
   */
  async connectSocketIO() {
    return new Promise((resolve, reject) => {
      try {
        // Use dynamic import for Socket.IO client
        import('socket.io-client').then(({ io }) => {
          const socketUrl = this.config.serverUrl.replace('ws://', 'http://').replace('wss://', 'https://');
          
          this.socketIO = io(socketUrl, {
            path: this.config.socketIOPath,
            transports: ['websocket', 'polling']
          });
          
          this.socketIO.on('connect', () => {
            console.log('Socket.IO connected');
            resolve();
          });
          
          this.socketIO.on('disconnect', () => {
            console.log('Socket.IO disconnected');
          });
          
          this.socketIO.on('error', (error) => {
            console.error('Socket.IO error:', error);
            reject(error);
          });
          
          // Setup Socket.IO event handlers
          this.setupSocketIOHandlers();
          
        }).catch(error => {
          console.warn('Socket.IO not available, using WebSocket only');
          resolve(); // Continue without Socket.IO
        });
        
      } catch (error) {
        console.warn('Socket.IO connection failed, continuing without it');
        resolve(); // Continue without Socket.IO
      }
    });
  }

  /**
   * Setup voice engine event handlers
   */
  setupVoiceEngineHandlers() {
    this.voiceEngine.on('partialTranscription', (data) => {
      this.handlePartialTranscription(data);
    });

    this.voiceEngine.on('finalTranscription', (data) => {
      this.handleFinalTranscription(data);
    });

    this.voiceEngine.on('emotionDetected', (data) => {
      this.handleEmotionDetected(data);
    });

    this.voiceEngine.on('languageChanged', (data) => {
      this.handleLanguageChanged(data);
    });

    this.voiceEngine.on('speakerVerified', (data) => {
      this.handleSpeakerVerified(data);
    });

    this.voiceEngine.on('translationComplete', (data) => {
      this.handleTranslationComplete(data);
    });

    this.voiceEngine.on('llmResponse', (data) => {
      this.handleLLMResponse(data);
    });

    this.voiceEngine.on('audioLevel', (data) => {
      this.handleAudioLevel(data);
    });

    this.voiceEngine.on('error', (error) => {
      this.handleError('voice_engine', error);
    });
  }

  /**
   * Setup Socket.IO event handlers
   */
  setupSocketIOHandlers() {
    if (!this.socketIO) return;

    this.socketIO.on('media_response', (data) => {
      this.handleMediaResponse(data);
    });

    this.socketIO.on('edit_result', (data) => {
      this.handleEditResult(data);
    });

    this.socketIO.on('error', (data) => {
      this.handleError('socket_io', data);
    });
  }

  /**
   * Initialize voice session
   */
  initializeVoiceSession() {
    this.sessionId = this.generateSessionId();
    
    const sessionData = {
      type: 'session_start',
      sessionId: this.sessionId,
      config: {
        language: this.currentLanguage,
        emotionDetection: this.config.enableEmotionDetection,
        biometrics: this.config.enableBiometrics
      }
    };

    this.sendVoiceMessage(sessionData);
  }

  /**
   * Start voice listening
   */
  async startListening() {
    if (!this.isConnected || this.isListening) {
      console.warn('Cannot start listening: not connected or already listening');
      return false;
    }

    try {
      // Start voice engine
      await this.voiceEngine.startListening();
      
      // Notify server
      this.sendVoiceMessage({
        type: 'start_listening',
        sessionId: this.sessionId
      });
      
      this.isListening = true;
      console.log('Started voice listening');
      
      return true;
    } catch (error) {
      console.error('Failed to start listening:', error);
      this.handleError('start_listening', error);
      return false;
    }
  }

  /**
   * Stop voice listening
   */
  stopListening() {
    if (!this.isListening) {
      return;
    }

    try {
      // Stop voice engine
      this.voiceEngine.stopListening();
      
      // Notify server
      this.sendVoiceMessage({
        type: 'stop_listening',
        sessionId: this.sessionId
      });
      
      this.isListening = false;
      console.log('Stopped voice listening');
      
    } catch (error) {
      console.error('Failed to stop listening:', error);
      this.handleError('stop_listening', error);
    }
  }

  /**
   * Change language
   */
  setLanguage(languageCode) {
    if (this.currentLanguage === languageCode) {
      return;
    }

    this.currentLanguage = languageCode;
    
    // Update voice engine
    try {
      this.voiceEngine.setLanguage(languageCode);
    } catch (error) {
      console.warn('Voice engine language change failed:', error.message);
    }
    
    // Notify server
    if (this.isConnected) {
      this.sendVoiceMessage({
        type: 'update_language',
        sessionId: this.sessionId,
        language: languageCode
      });
    }
  }

  /**
   * Request translation
   */
  translateText(text, targetLanguage, sourceLanguage = null) {
    if (!this.isConnected) {
      console.warn('Cannot translate: not connected');
      return;
    }

    this.sendVoiceMessage({
      type: 'translate_request',
      sessionId: this.sessionId,
      text,
      targetLanguage,
      sourceLanguage: sourceLanguage || this.currentLanguage
    });
  }

  /**
   * Send media query
   */
  queryMedia(query) {
    if (!this.socketIO) {
      console.warn('Cannot query media: Socket.IO not connected');
      return;
    }

    this.socketIO.emit('media_conversation', {
      message: query,
      userId: this.sessionId,
      contextId: this.sessionId,
      language: this.currentLanguage
    });
  }

  /**
   * Send voice edit command
   */
  sendEditCommand(command, mediaId) {
    if (!this.socketIO) {
      console.warn('Cannot send edit command: Socket.IO not connected');
      return;
    }

    this.socketIO.emit('voice_edit_command', {
      command,
      mediaId,
      parameters: {}
    });
  }

  /**
   * Start biometric enrollment
   */
  startBiometricEnrollment(speakerId) {
    if (!this.isConnected) {
      console.warn('Cannot start biometric enrollment: not connected');
      return;
    }

    this.sendVoiceMessage({
      type: 'enroll_biometric',
      sessionId: this.sessionId,
      speakerId
    });
  }

  /**
   * Handle voice WebSocket messages
   */
  handleVoiceWebSocketMessage(message) {
    switch (message.type) {
      case 'session_started':
        console.log('Voice session started:', message.sessionId);
        break;
        
      case 'transcription_partial':
        this.handlePartialTranscription(message);
        break;
        
      case 'transcription_final':
        this.handleFinalTranscription(message);
        break;
        
      case 'emotion_detected':
        this.handleEmotionDetected(message);
        break;
        
      case 'language_detected':
        this.handleLanguageChanged(message);
        break;
        
      case 'speaker_verified':
        this.handleSpeakerVerified(message);
        break;
        
      case 'translation_result':
        this.handleTranslationComplete(message);
        break;
        
      case 'llm_response':
        this.handleLLMResponse(message);
        break;
        
      case 'biometric_enrolled':
        this.handleBiometricResult(message);
        break;
        
      case 'error':
        this.handleError('server', message.error);
        break;
        
      default:
        console.warn('Unknown voice message type:', message.type);
    }
  }

  /**
   * Handle partial transcription
   */
  handlePartialTranscription(data) {
    if (this.callbacks.onTranscription) {
      this.callbacks.onTranscription({
        text: data.text,
        confidence: data.confidence,
        language: data.language,
        isPartial: true
      });
    }
  }

  /**
   * Handle final transcription
   */
  handleFinalTranscription(data) {
    if (this.callbacks.onTranscription) {
      this.callbacks.onTranscription({
        text: data.text,
        confidence: data.confidence,
        language: data.language,
        isPartial: false,
        emotion: data.emotion,
        sentiment: data.sentiment,
        speaker: data.speaker
      });
    }
  }

  /**
   * Handle emotion detection
   */
  handleEmotionDetected(data) {
    if (this.callbacks.onEmotionDetected) {
      this.callbacks.onEmotionDetected({
        emotion: data.emotion,
        confidence: data.confidence,
        sentiment: data.sentiment,
        history: data.history
      });
    }
  }

  /**
   * Handle language change
   */
  handleLanguageChanged(data) {
    this.currentLanguage = data.newLanguage;
    console.log('Language changed to:', data.newLanguage);
  }

  /**
   * Handle speaker verification
   */
  handleSpeakerVerified(data) {
    if (this.callbacks.onBiometricResult) {
      this.callbacks.onBiometricResult({
        type: 'verification',
        verified: data.verified,
        confidence: data.confidence,
        speakerId: data.speakerId
      });
    }
  }

  /**
   * Handle translation completion
   */
  handleTranslationComplete(data) {
    if (this.callbacks.onTranslation) {
      this.callbacks.onTranslation({
        originalText: data.originalText,
        translatedText: data.translatedText,
        sourceLanguage: data.sourceLanguage,
        targetLanguage: data.targetLanguage,
        confidence: data.confidence
      });
    }
  }

  /**
   * Handle LLM response
   */
  handleLLMResponse(data) {
    console.log('LLM Response:', data.response);
    // Could be used for intelligent voice interactions
  }

  /**
   * Handle audio level updates
   */
  handleAudioLevel(data) {
    // Forward to UI for visualization
    if (this.callbacks.onAudioLevel) {
      this.callbacks.onAudioLevel(data);
    }
  }

  /**
   * Handle media response
   */
  handleMediaResponse(data) {
    if (this.callbacks.onMediaResponse) {
      this.callbacks.onMediaResponse(data);
    }
  }

  /**
   * Handle edit result
   */
  handleEditResult(data) {
    console.log('Edit result:', data);
    // Could be used to show edit operation feedback
  }

  /**
   * Handle biometric result
   */
  handleBiometricResult(data) {
    if (this.callbacks.onBiometricResult) {
      this.callbacks.onBiometricResult({
        type: 'enrollment',
        success: data.success,
        speakerId: data.speakerId,
        confidence: data.confidence
      });
    }
  }

  /**
   * Handle WebSocket close
   */
  handleVoiceWebSocketClose() {
    this.isConnected = false;
    this.isListening = false;
    this.notifyConnectionChange(false);
    
    // Attempt reconnection
    if (this.reconnectCount < this.config.reconnectAttempts) {
      setTimeout(() => this.reconnect(), this.config.reconnectDelay);
    }
  }

  /**
   * Attempt reconnection
   */
  async reconnect() {
    this.reconnectCount++;
    console.log(`Attempting reconnection ${this.reconnectCount}/${this.config.reconnectAttempts}...`);
    
    await this.connect();
  }

  /**
   * Send message to voice WebSocket
   */
  sendVoiceMessage(message) {
    if (this.voiceWebSocket && this.voiceWebSocket.readyState === WebSocket.OPEN) {
      this.voiceWebSocket.send(JSON.stringify(message));
    } else {
      console.warn('Voice WebSocket not ready, message not sent:', message);
    }
  }

  /**
   * Notify connection change
   */
  notifyConnectionChange(connected) {
    if (this.callbacks.onConnectionChange) {
      this.callbacks.onConnectionChange(connected);
    }
  }

  /**
   * Handle errors
   */
  handleError(type, error) {
    console.error(`Voice client error (${type}):`, error);
    
    if (this.callbacks.onError) {
      this.callbacks.onError({ type, error });
    }
  }

  /**
   * Generate session ID
   */
  generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  }

  /**
   * Get connection status
   */
  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      isListening: this.isListening,
      sessionId: this.sessionId,
      currentLanguage: this.currentLanguage,
      reconnectCount: this.reconnectCount
    };
  }

  /**
   * Get supported languages
   */
  getSupportedLanguages() {
    return this.voiceEngine ? this.voiceEngine.getSupportedLanguages() : [];
  }

  /**
   * Set event callbacks
   */
  setCallbacks(callbacks) {
    Object.assign(this.callbacks, callbacks);
  }

  /**
   * Disconnect and cleanup
   */
  async disconnect() {
    console.log('Disconnecting voice client...');
    
    // Stop listening
    if (this.isListening) {
      this.stopListening();
    }
    
    // Close WebSocket
    if (this.voiceWebSocket) {
      this.voiceWebSocket.close();
      this.voiceWebSocket = null;
    }
    
    // Close Socket.IO
    if (this.socketIO) {
      this.socketIO.disconnect();
      this.socketIO = null;
    }
    
    // Cleanup voice engine
    if (this.voiceEngine) {
      this.voiceEngine.destroy();
      this.voiceEngine = null;
    }
    
    this.isConnected = false;
    this.isListening = false;
    this.sessionId = null;
    
    this.notifyConnectionChange(false);
    
    console.log('Voice client disconnected');
  }

  /**
   * Destroy the client
   */
  destroy() {
    this.disconnect();
    
    // Clear callbacks
    this.callbacks = {};
    
    console.log('Voice client destroyed');
  }
}

export default VoiceClient;