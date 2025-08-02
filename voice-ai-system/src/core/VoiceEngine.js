/**
 * Advanced Voice Engine with LLM Integration and Emotion Detection
 * Supports WebRTC, real-time processing, and multi-language capabilities
 */

import { EventEmitter } from 'events';
import WebSocket from 'ws';
import { v4 as uuidv4 } from 'uuid';

export class VoiceEngine extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      sampleRate: 16000,
      channels: 1,
      bitDepth: 16,
      bufferSize: 4096,
      vadThreshold: 0.3,
      emotionDetection: true,
      realtimeTranscription: true,
      languageDetection: true,
      maxSilenceDuration: 2000,
      ...config
    };

    this.isListening = false;
    this.audioContext = null;
    this.mediaStream = null;
    this.processor = null;
    this.webSocket = null;
    this.sessionId = uuidv4();
    
    // LLM Integration
    this.llmProvider = config.llmProvider || 'openai';
    this.apiKey = config.apiKey;
    
    // Emotion and sentiment tracking
    this.emotionHistory = [];
    this.currentEmotion = null;
    this.sentimentScore = 0;
    
    // Language and translation
    this.currentLanguage = 'en-US';
    this.supportedLanguages = new Set();
    this.translationQueue = [];
    
    // Voice biometrics
    this.voicePrint = null;
    this.speakerVerification = config.enableBiometrics || false;
    
    this.initializeLanguageSupport();
  }

  /**
   * Initialize comprehensive language support (100+ languages)
   */
  initializeLanguageSupport() {
    // Major world languages with their codes
    const languages = [
      'en-US', 'en-GB', 'en-AU', 'en-CA', 'en-IN', 'en-ZA',
      'es-ES', 'es-MX', 'es-AR', 'es-CO', 'es-CL', 'es-PE',
      'fr-FR', 'fr-CA', 'fr-BE', 'fr-CH',
      'de-DE', 'de-AT', 'de-CH', 
      'it-IT', 'pt-BR', 'pt-PT',
      'ru-RU', 'zh-CN', 'zh-TW', 'zh-HK',
      'ja-JP', 'ko-KR', 'ar-SA', 'ar-EG',
      'hi-IN', 'bn-IN', 'ta-IN', 'te-IN', 'mr-IN', 'gu-IN',
      'th-TH', 'vi-VN', 'id-ID', 'ms-MY',
      'tr-TR', 'pl-PL', 'nl-NL', 'sv-SE', 'da-DK', 'no-NO', 'fi-FI',
      'he-IL', 'fa-IR', 'ur-PK', 'sw-KE', 'am-ET',
      'uk-UA', 'bg-BG', 'hr-HR', 'cs-CZ', 'sk-SK', 'sl-SI',
      'et-EE', 'lv-LV', 'lt-LT', 'mt-MT',
      'hu-HU', 'ro-RO', 'sq-AL', 'mk-MK', 'sr-RS', 'bs-BA',
      'is-IS', 'ga-IE', 'cy-GB', 'eu-ES', 'ca-ES', 'gl-ES',
      'af-ZA', 'zu-ZA', 'xh-ZA', 'st-ZA', 'tn-ZA', 've-ZA',
      'yo-NG', 'ig-NG', 'ha-NG', 'ff-SN', 'wo-SN',
      'mn-MN', 'km-KH', 'lo-LA', 'my-MM', 
      'ne-NP', 'si-LK', 'dv-MV', 'ps-AF', 'tg-TJ', 'ky-KG', 'kk-KZ', 'uz-UZ',
      'az-AZ', 'ka-GE', 'hy-AM', 'be-BY', 'mk-MK',
      'mi-NZ', 'sm-WS', 'to-TO', 'fj-FJ', 'haw-US'
    ];
    
    languages.forEach(lang => this.supportedLanguages.add(lang));
    console.log(`Initialized support for ${this.supportedLanguages.size} languages`);
  }

  /**
   * Initialize voice engine with WebRTC support
   */
  async initialize() {
    try {
      // Initialize Web Audio API
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this.config.sampleRate
      });

      // Request microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // Create audio processing chain
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.processor = this.audioContext.createScriptProcessor(
        this.config.bufferSize, 
        this.config.channels, 
        this.config.channels
      );

      // Connect audio processing
      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      // Set up real-time processing
      this.processor.onaudioprocess = (event) => {
        if (this.isListening) {
          this.processAudioBuffer(event.inputBuffer);
        }
      };

      // Initialize WebSocket connection for real-time processing
      await this.initializeWebSocketConnection();

      console.log('Voice engine initialized successfully');
      this.emit('initialized');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize voice engine:', error);
      this.emit('error', { type: 'initialization', error });
      return false;
    }
  }

  /**
   * Initialize WebSocket connection for real-time processing
   */
  async initializeWebSocketConnection() {
    const wsUrl = process.env.VOICE_WS_URL || 'ws://localhost:8080/voice';
    
    this.webSocket = new WebSocket(wsUrl);
    
    this.webSocket.onopen = () => {
      console.log('WebSocket connection established');
      this.webSocket.send(JSON.stringify({
        type: 'session_start',
        sessionId: this.sessionId,
        config: {
          language: this.currentLanguage,
          emotionDetection: this.config.emotionDetection,
          biometrics: this.speakerVerification
        }
      }));
    };

    this.webSocket.onmessage = (event) => {
      this.handleWebSocketMessage(JSON.parse(event.data));
    };

    this.webSocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', { type: 'websocket', error });
    };

    this.webSocket.onclose = () => {
      console.log('WebSocket connection closed');
      // Implement reconnection logic
      setTimeout(() => this.initializeWebSocketConnection(), 5000);
    };
  }

  /**
   * Process audio buffer for real-time analysis
   */
  processAudioBuffer(inputBuffer) {
    const audioData = inputBuffer.getChannelData(0);
    
    // Voice Activity Detection (VAD)
    const vadResult = this.detectVoiceActivity(audioData);
    
    if (vadResult.hasVoice) {
      // Convert to appropriate format and send for processing
      const audioChunk = this.convertAudioData(audioData);
      
      if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
        this.webSocket.send(JSON.stringify({
          type: 'audio_chunk',
          sessionId: this.sessionId,
          data: Array.from(audioChunk),
          timestamp: Date.now(),
          vadScore: vadResult.confidence
        }));
      }
    }

    // Emit audio level for UI feedback
    this.emit('audioLevel', {
      level: this.calculateAudioLevel(audioData),
      hasVoice: vadResult.hasVoice,
      vadConfidence: vadResult.confidence
    });
  }

  /**
   * Voice Activity Detection using energy-based approach
   */
  detectVoiceActivity(audioData) {
    const energy = audioData.reduce((sum, sample) => sum + Math.abs(sample), 0) / audioData.length;
    const hasVoice = energy > this.config.vadThreshold;
    
    return {
      hasVoice,
      confidence: Math.min(energy / this.config.vadThreshold, 1.0),
      energy
    };
  }

  /**
   * Convert audio data to appropriate format
   */
  convertAudioData(float32Array) {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      int16Array[i] = Math.max(-32768, Math.min(32767, float32Array[i] * 32768));
    }
    return int16Array;
  }

  /**
   * Calculate audio level for visualization
   */
  calculateAudioLevel(audioData) {
    const sum = audioData.reduce((acc, val) => acc + Math.abs(val), 0);
    return Math.sqrt(sum / audioData.length);
  }

  /**
   * Handle WebSocket messages from server
   */
  handleWebSocketMessage(message) {
    switch (message.type) {
      case 'transcription_partial':
        this.emit('partialTranscription', {
          text: message.text,
          confidence: message.confidence,
          language: message.language
        });
        break;

      case 'transcription_final':
        this.handleFinalTranscription(message);
        break;

      case 'emotion_detected':
        this.handleEmotionDetection(message);
        break;

      case 'language_detected':
        this.handleLanguageDetection(message);
        break;

      case 'speaker_verified':
        this.handleSpeakerVerification(message);
        break;

      case 'translation_result':
        this.handleTranslationResult(message);
        break;

      case 'llm_response':
        this.handleLLMResponse(message);
        break;

      case 'error':
        this.emit('error', { type: 'server', error: message.error });
        break;

      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  /**
   * Handle final transcription with emotion and language data
   */
  handleFinalTranscription(message) {
    const transcription = {
      text: message.text,
      confidence: message.confidence,
      language: message.language,
      timestamp: message.timestamp,
      speaker: message.speaker,
      emotion: message.emotion,
      sentiment: message.sentiment
    };

    this.emit('finalTranscription', transcription);
    
    // Trigger LLM processing if enabled
    if (this.config.llmProcessing) {
      this.processWithLLM(transcription);
    }
  }

  /**
   * Handle emotion detection results
   */
  handleEmotionDetection(message) {
    this.currentEmotion = message.emotion;
    this.sentimentScore = message.sentiment;
    
    this.emotionHistory.push({
      emotion: message.emotion,
      confidence: message.confidence,
      sentiment: message.sentiment,
      timestamp: message.timestamp
    });

    // Keep only last 100 emotion readings
    if (this.emotionHistory.length > 100) {
      this.emotionHistory.shift();
    }

    this.emit('emotionDetected', {
      emotion: message.emotion,
      confidence: message.confidence,
      sentiment: message.sentiment,
      history: this.emotionHistory.slice(-10) // Last 10 readings
    });
  }

  /**
   * Handle language detection
   */
  handleLanguageDetection(message) {
    if (message.language !== this.currentLanguage) {
      this.currentLanguage = message.language;
      
      this.emit('languageChanged', {
        previousLanguage: this.currentLanguage,
        newLanguage: message.language,
        confidence: message.confidence
      });

      // Update voice recognition settings
      this.webSocket.send(JSON.stringify({
        type: 'update_language',
        sessionId: this.sessionId,
        language: message.language
      }));
    }
  }

  /**
   * Handle speaker verification results
   */
  handleSpeakerVerification(message) {
    this.emit('speakerVerified', {
      verified: message.verified,
      confidence: message.confidence,
      speakerId: message.speakerId,
      voicePrintMatch: message.voicePrintMatch
    });
  }

  /**
   * Handle translation results
   */
  handleTranslationResult(message) {
    this.emit('translationComplete', {
      originalText: message.originalText,
      translatedText: message.translatedText,
      sourceLanguage: message.sourceLanguage,
      targetLanguage: message.targetLanguage,
      confidence: message.confidence
    });
  }

  /**
   * Handle LLM responses
   */
  handleLLMResponse(message) {
    this.emit('llmResponse', {
      query: message.query,
      response: message.response,
      model: message.model,
      confidence: message.confidence,
      contextUsed: message.contextUsed
    });
  }

  /**
   * Process transcription with LLM for intelligent responses
   */
  async processWithLLM(transcription) {
    if (!this.webSocket || this.webSocket.readyState !== WebSocket.OPEN) {
      return;
    }

    this.webSocket.send(JSON.stringify({
      type: 'llm_process',
      sessionId: this.sessionId,
      text: transcription.text,
      context: {
        emotion: this.currentEmotion,
        sentiment: this.sentimentScore,
        language: this.currentLanguage,
        speaker: transcription.speaker
      }
    }));
  }

  /**
   * Start voice listening
   */
  async startListening() {
    if (!this.audioContext || !this.processor) {
      throw new Error('Voice engine not initialized');
    }

    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    this.isListening = true;
    this.emit('listeningStarted');
    
    // Notify server
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify({
        type: 'start_listening',
        sessionId: this.sessionId
      }));
    }
  }

  /**
   * Stop voice listening
   */
  stopListening() {
    this.isListening = false;
    this.emit('listeningStopped');
    
    // Notify server
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify({
        type: 'stop_listening',
        sessionId: this.sessionId
      }));
    }
  }

  /**
   * Request translation to target language
   */
  translateText(text, targetLanguage, sourceLanguage = null) {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify({
        type: 'translate_request',
        sessionId: this.sessionId,
        text,
        targetLanguage,
        sourceLanguage: sourceLanguage || this.currentLanguage
      }));
    }
  }

  /**
   * Set up voice biometric enrollment
   */
  enrollVoiceBiometric(speakerId) {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify({
        type: 'enroll_biometric',
        sessionId: this.sessionId,
        speakerId
      }));
    }
  }

  /**
   * Get current emotion analysis
   */
  getCurrentEmotion() {
    return {
      emotion: this.currentEmotion,
      sentiment: this.sentimentScore,
      history: this.emotionHistory.slice(-10)
    };
  }

  /**
   * Get supported languages
   */
  getSupportedLanguages() {
    return Array.from(this.supportedLanguages);
  }

  /**
   * Change current language
   */
  setLanguage(languageCode) {
    if (!this.supportedLanguages.has(languageCode)) {
      throw new Error(`Language ${languageCode} is not supported`);
    }

    this.currentLanguage = languageCode;
    
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(JSON.stringify({
        type: 'update_language',
        sessionId: this.sessionId,
        language: languageCode
      }));
    }
  }

  /**
   * Cleanup resources
   */
  destroy() {
    this.stopListening();

    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.webSocket) {
      this.webSocket.close();
      this.webSocket = null;
    }

    this.emit('destroyed');
  }
}

export default VoiceEngine;