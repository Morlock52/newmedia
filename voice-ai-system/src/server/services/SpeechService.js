/**
 * Speech Service
 * Handles speech-to-text and text-to-speech operations with multiple providers
 */

import axios from 'axios';
import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';

export class SpeechService extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      // Provider configuration
      primarySTTProvider: config.primarySTTProvider || 'azure',
      primaryTTSProvider: config.primaryTTSProvider || 'azure',
      
      // API keys
      azureKey: config.azureKey || process.env.AZURE_SPEECH_KEY,
      azureRegion: config.azureRegion || process.env.AZURE_REGION,
      googleApiKey: config.googleApiKey || process.env.GOOGLE_SPEECH_API_KEY,
      awsAccessKey: config.awsAccessKey || process.env.AWS_ACCESS_KEY,
      awsSecretKey: config.awsSecretKey || process.env.AWS_SECRET_KEY,
      elevenLabsApiKey: config.elevenLabsApiKey || process.env.ELEVENLABS_API_KEY,
      
      // Speech recognition settings
      sampleRate: config.sampleRate || 16000,
      channels: config.channels || 1,
      encoding: config.encoding || 'LINEAR16',
      languageCode: config.languageCode || 'en-US',
      
      // Real-time processing
      enableRealTimeTranscription: config.enableRealTimeTranscription !== false,
      chunkSize: config.chunkSize || 1024,
      silenceTimeout: config.silenceTimeout || 2000,
      
      // Text-to-speech settings
      defaultVoice: config.defaultVoice || 'en-US-AriaNeural',
      speechRate: config.speechRate || 1.0,
      speechPitch: config.speechPitch || 1.0,
      audioFormat: config.audioFormat || 'audio-24khz-48kbitrate-mono-mp3',
      
      // Voice enhancement
      enableVoiceEffects: config.enableVoiceEffects !== false,
      enableEmotionalTTS: config.enableEmotionalTTS !== false,
      enableSSML: config.enableSSML !== false,
      
      // Caching
      enableCaching: config.enableCaching !== false,
      cacheDirectory: config.cacheDirectory || './data/speech_cache',
      cacheTimeout: config.cacheTimeout || 86400000, // 24 hours
      
      ...config
    };

    // Provider status
    this.providerStatus = {
      azure: false,
      google: false,
      aws: false,
      elevenlabs: false,
      local: true
    };

    // Real-time transcription sessions
    this.transcriptionSessions = new Map();
    
    // TTS cache
    this.ttsCache = new Map();
    
    // Supported voices and languages
    this.supportedVoices = new Map();
    this.supportedLanguages = new Set();
    
    this.isInitialized = false;
  }

  /**
   * Initialize the speech service
   */
  async initialize() {
    try {
      console.log('Initializing Speech Service...');
      
      // Create cache directory
      await this.ensureCacheDirectory();
      
      // Test provider connectivity
      await this.testProviderConnectivity();
      
      // Load supported voices and languages
      await this.loadSupportedVoicesAndLanguages();
      
      // Load TTS cache
      await this.loadTTSCache();
      
      // Setup cleanup intervals
      this.setupCleanupIntervals();
      
      this.isInitialized = true;
      console.log('Speech Service initialized successfully');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize Speech Service:', error);
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
   * Test connectivity to speech providers
   */
  async testProviderConnectivity() {
    const tests = [];

    // Test Azure Speech Services
    if (this.config.azureKey) {
      tests.push(this.testAzureConnectivity());
    }

    // Test Google Cloud Speech
    if (this.config.googleApiKey) {
      tests.push(this.testGoogleConnectivity());
    }

    // Test ElevenLabs
    if (this.config.elevenLabsApiKey) {
      tests.push(this.testElevenLabsConnectivity());
    }

    const results = await Promise.allSettled(tests);
    
    results.forEach((result, index) => {
      const providers = ['azure', 'google', 'elevenlabs'];
      if (result.status === 'fulfilled') {
        this.providerStatus[providers[index]] = true;
        console.log(`✅ ${providers[index]} speech provider connected`);
      } else {
        console.warn(`⚠️  ${providers[index]} speech provider not available:`, result.reason.message);
      }
    });
  }

  /**
   * Test Azure Speech Services connectivity
   */
  async testAzureConnectivity() {
    try {
      const response = await axios.get(`https://${this.config.azureRegion}.tts.speech.microsoft.com/cognitiveservices/voices/list`, {
        headers: {
          'Ocp-Apim-Subscription-Key': this.config.azureKey
        },
        timeout: 5000
      });
      return response.status === 200;
    } catch (error) {
      throw new Error(`Azure Speech Services connection failed: ${error.message}`);
    }
  }

  /**
   * Test Google Cloud Speech connectivity
   */
  async testGoogleConnectivity() {
    try {
      // Test with a simple recognition request
      const response = await axios.post(`https://speech.googleapis.com/v1/speech:recognize?key=${this.config.googleApiKey}`, {
        config: {
          encoding: 'LINEAR16',
          sampleRateHertz: 16000,
          languageCode: 'en-US'
        },
        audio: {
          content: Buffer.from([]).toString('base64')
        }
      }, {
        timeout: 5000
      });
      // Even with empty audio, successful API call indicates connectivity
      return true;
    } catch (error) {
      // Google might return 400 for empty audio, but that means the service is available
      if (error.response && error.response.status === 400) {
        return true;
      }
      throw new Error(`Google Cloud Speech connection failed: ${error.message}`);
    }
  }

  /**
   * Test ElevenLabs connectivity
   */
  async testElevenLabsConnectivity() {
    try {
      const response = await axios.get('https://api.elevenlabs.io/v1/voices', {
        headers: {
          'xi-api-key': this.config.elevenLabsApiKey
        },
        timeout: 5000
      });
      return response.status === 200;
    } catch (error) {
      throw new Error(`ElevenLabs connection failed: ${error.message}`);
    }
  }

  /**
   * Load supported voices and languages
   */
  async loadSupportedVoicesAndLanguages() {
    // Load Azure voices if available
    if (this.providerStatus.azure) {
      await this.loadAzureVoices();
    }

    // Load ElevenLabs voices if available
    if (this.providerStatus.elevenlabs) {
      await this.loadElevenLabsVoices();
    }

    // Add default language support
    this.initializeDefaultLanguageSupport();
  }

  /**
   * Load Azure voices
   */
  async loadAzureVoices() {
    try {
      const response = await axios.get(`https://${this.config.azureRegion}.tts.speech.microsoft.com/cognitiveservices/voices/list`, {
        headers: {
          'Ocp-Apim-Subscription-Key': this.config.azureKey
        }
      });

      const voices = response.data;
      voices.forEach(voice => {
        this.supportedVoices.set(voice.ShortName, {
          name: voice.DisplayName,
          gender: voice.Gender,
          locale: voice.Locale,
          provider: 'azure',
          neural: voice.VoiceType === 'Neural',
          styles: voice.StyleList || []
        });
        
        this.supportedLanguages.add(voice.Locale);
      });

      console.log(`Loaded ${voices.length} Azure voices`);
    } catch (error) {
      console.warn('Failed to load Azure voices:', error.message);
    }
  }

  /**
   * Load ElevenLabs voices
   */
  async loadElevenLabsVoices() {
    try {
      const response = await axios.get('https://api.elevenlabs.io/v1/voices', {
        headers: {
          'xi-api-key': this.config.elevenLabsApiKey
        }
      });

      const voices = response.data.voices;
      voices.forEach(voice => {
        this.supportedVoices.set(voice.voice_id, {
          name: voice.name,
          category: voice.category,
          provider: 'elevenlabs',
          previewUrl: voice.preview_url,
          settings: voice.settings
        });
      });

      console.log(`Loaded ${voices.length} ElevenLabs voices`);
    } catch (error) {
      console.warn('Failed to load ElevenLabs voices:', error.message);
    }
  }

  /**
   * Initialize default language support
   */
  initializeDefaultLanguageSupport() {
    const defaultLanguages = [
      'en-US', 'en-GB', 'en-AU', 'en-CA', 'en-IN',
      'es-ES', 'es-MX', 'es-AR', 'fr-FR', 'fr-CA',
      'de-DE', 'it-IT', 'pt-BR', 'pt-PT', 'ru-RU',
      'zh-CN', 'zh-TW', 'ja-JP', 'ko-KR', 'ar-SA',
      'hi-IN', 'th-TH', 'vi-VN', 'tr-TR', 'pl-PL',
      'nl-NL', 'sv-SE', 'da-DK', 'no-NO', 'fi-FI'
    ];

    defaultLanguages.forEach(lang => this.supportedLanguages.add(lang));
  }

  /**
   * Load TTS cache from disk
   */
  async loadTTSCache() {
    try {
      const cacheFile = path.join(this.config.cacheDirectory, 'tts_cache.json');
      const cacheData = await fs.readFile(cacheFile, 'utf8');
      const cache = JSON.parse(cacheData);
      
      Object.entries(cache).forEach(([key, value]) => {
        this.ttsCache.set(key, value);
      });
      
      console.log(`Loaded ${this.ttsCache.size} TTS cache entries`);
    } catch (error) {
      console.log('No existing TTS cache found');
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

    // Clean up old transcription sessions every 30 minutes
    setInterval(() => {
      this.cleanupOldSessions();
    }, 1800000);
  }

  /**
   * Transcribe audio to text
   */
  async transcribe(audioData, options = {}) {
    const {
      language = this.config.languageCode,
      provider = this.config.primarySTTProvider,
      enableWordTimestamps = false,
      enablePunctuationAndCapitalization = true,
      enableProfanityFilter = false
    } = options;

    if (!audioData || audioData.length === 0) {
      throw new Error('Audio data is required');
    }

    try {
      // Try primary provider first
      if (this.providerStatus[provider]) {
        return await this.transcribeWithProvider(audioData, provider, {
          language,
          enableWordTimestamps,
          enablePunctuationAndCapitalization,
          enableProfanityFilter
        });
      }

      // Fallback to available providers
      const availableProviders = Object.keys(this.providerStatus)
        .filter(p => this.providerStatus[p] && p !== 'local');
      
      for (const fallbackProvider of availableProviders) {
        try {
          return await this.transcribeWithProvider(audioData, fallbackProvider, {
            language,
            enableWordTimestamps,
            enablePunctuationAndCapitalization,
            enableProfanityFilter
          });
        } catch (error) {
          console.warn(`Transcription with ${fallbackProvider} failed:`, error.message);
        }
      }

      // Use local fallback
      return this.transcribeLocally(audioData, options);

    } catch (error) {
      console.error('Transcription failed:', error);
      throw error;
    }
  }

  /**
   * Transcribe with specific provider
   */
  async transcribeWithProvider(audioData, provider, options) {
    switch (provider) {
      case 'azure':
        return await this.transcribeWithAzure(audioData, options);
      case 'google':
        return await this.transcribeWithGoogle(audioData, options);
      case 'aws':
        return await this.transcribeWithAWS(audioData, options);
      default:
        throw new Error(`Unknown transcription provider: ${provider}`);
    }
  }

  /**
   * Transcribe with Azure Speech Services
   */
  async transcribeWithAzure(audioData, options) {
    try {
      // Convert audio data to WAV format for Azure
      const wavBuffer = this.convertToWav(audioData);
      
      const response = await axios.post(
        `https://${this.config.azureRegion}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1`,
        wavBuffer,
        {
          headers: {
            'Ocp-Apim-Subscription-Key': this.config.azureKey,
            'Content-Type': 'audio/wav',
            'Accept': 'application/json'
          },
          params: {
            language: options.language,
            format: 'detailed',
            profanity: options.enableProfanityFilter ? 'masked' : 'raw'
          }
        }
      );

      const result = response.data;
      
      return {
        text: result.DisplayText || result.RecognitionStatus === 'Success' ? result.DisplayText : '',
        confidence: result.Confidence || 0.8,
        language: options.language,
        provider: 'azure',
        words: options.enableWordTimestamps ? this.extractAzureWordTimestamps(result) : null,
        alternatives: result.NBest ? result.NBest.slice(1).map(alt => ({
          text: alt.Display,
          confidence: alt.Confidence
        })) : []
      };
    } catch (error) {
      throw new Error(`Azure transcription failed: ${error.message}`);
    }
  }

  /**
   * Transcribe with Google Cloud Speech
   */
  async transcribeWithGoogle(audioData, options) {
    try {
      const audioContent = Buffer.from(audioData.buffer).toString('base64');
      
      const response = await axios.post(
        `https://speech.googleapis.com/v1/speech:recognize?key=${this.config.googleApiKey}`,
        {
          config: {
            encoding: this.config.encoding,
            sampleRateHertz: this.config.sampleRate,
            languageCode: options.language,
            enableWordTimeOffsets: options.enableWordTimestamps,
            enableAutomaticPunctuation: options.enablePunctuationAndCapitalization,
            profanityFilter: options.enableProfanityFilter,
            alternativeLanguageCodes: ['en-US'], // Fallback language
            maxAlternatives: 3
          },
          audio: {
            content: audioContent
          }
        }
      );

      const result = response.data.results?.[0];
      if (!result) {
        return {
          text: '',
          confidence: 0,
          language: options.language,
          provider: 'google'
        };
      }

      const topAlternative = result.alternatives[0];
      
      return {
        text: topAlternative.transcript || '',
        confidence: topAlternative.confidence || 0.8,
        language: options.language,
        provider: 'google',
        words: options.enableWordTimestamps ? topAlternative.words : null,
        alternatives: result.alternatives.slice(1).map(alt => ({
          text: alt.transcript,
          confidence: alt.confidence
        }))
      };
    } catch (error) {
      throw new Error(`Google transcription failed: ${error.message}`);
    }
  }

  /**
   * Transcribe with AWS (placeholder)
   */
  async transcribeWithAWS(audioData, options) {
    // AWS Transcribe implementation would go here
    throw new Error('AWS transcription not implemented in this demo');
  }

  /**
   * Local transcription fallback
   */
  transcribeLocally(audioData, options) {
    // Simple placeholder for local transcription
    return {
      text: '[Audio transcription unavailable - no providers connected]',
      confidence: 0.1,
      language: options.language,
      provider: 'local',
      words: null,
      alternatives: []
    };
  }

  /**
   * Real-time transcription
   */
  async transcribeRealTime(options) {
    const {
      audioData,
      language = this.config.languageCode,
      sessionId,
      isPartial = true
    } = options;

    if (!sessionId) {
      throw new Error('Session ID is required for real-time transcription');
    }

    let session = this.transcriptionSessions.get(sessionId);
    if (!session) {
      session = {
        id: sessionId,
        language,
        audioBuffer: [],
        partialResults: [],
        finalResults: [],
        lastActivity: Date.now(),
        totalAudioTime: 0
      };
      this.transcriptionSessions.set(sessionId, session);
    }

    // Add audio data to buffer
    session.audioBuffer.push(...audioData);
    session.lastActivity = Date.now();
    session.totalAudioTime += audioData.length / this.config.sampleRate;

    // Process if we have enough audio or if this is marked as final
    const shouldProcess = session.audioBuffer.length >= this.config.chunkSize || !isPartial;
    
    if (shouldProcess && session.audioBuffer.length > 0) {
      try {
        const audioChunk = new Int16Array(session.audioBuffer);
        const result = await this.transcribe(audioChunk, { language });
        
        if (result.text && result.text.trim()) {
          if (isPartial) {
            session.partialResults.push(result);
            return {
              partial: true,
              final: false,
              text: result.text,
              confidence: result.confidence,
              language: result.language,
              sessionId
            };
          } else {
            session.finalResults.push(result);
            // Clear audio buffer after final processing
            session.audioBuffer = [];
            
            return {
              partial: false,
              final: true,
              text: result.text,
              confidence: result.confidence,
              language: result.language,
              sessionId,
              words: result.words,
              alternatives: result.alternatives
            };
          }
        }
      } catch (error) {
        console.error('Real-time transcription error:', error);
      }
    }

    return {
      partial: true,
      final: false,
      text: '',
      confidence: 0,
      language,
      sessionId
    };
  }

  /**
   * Synthesize text to speech
   */
  async synthesize(options) {
    const {
      text,
      language = 'en-US',
      voice = this.config.defaultVoice,
      emotion = null,
      speed = this.config.speechRate,
      pitch = this.config.speechPitch,
      provider = this.config.primaryTTSProvider
    } = options;

    if (!text) {
      throw new Error('Text is required for synthesis');
    }

    // Check cache first
    const cacheKey = this.generateTTSCacheKey(text, language, voice, emotion, speed, pitch);
    if (this.config.enableCaching && this.ttsCache.has(cacheKey)) {
      const cached = this.ttsCache.get(cacheKey);
      if (Date.now() - cached.timestamp < this.config.cacheTimeout) {
        return cached.audioBuffer;
      }
    }

    try {
      let audioBuffer;

      // Try primary provider first
      if (this.providerStatus[provider]) {
        audioBuffer = await this.synthesizeWithProvider(text, provider, {
          language,
          voice,
          emotion,
          speed,
          pitch
        });
      } else {
        // Fallback to available providers
        const availableProviders = Object.keys(this.providerStatus)
          .filter(p => this.providerStatus[p] && p !== 'local');
        
        for (const fallbackProvider of availableProviders) {
          try {
            audioBuffer = await this.synthesizeWithProvider(text, fallbackProvider, {
              language,
              voice,
              emotion,
              speed,
              pitch
            });
            break;
          } catch (error) {
            console.warn(`TTS with ${fallbackProvider} failed:`, error.message);
          }
        }
      }

      if (!audioBuffer) {
        audioBuffer = this.synthesizeLocally(text, options);
      }

      // Cache the result
      if (this.config.enableCaching) {
        this.ttsCache.set(cacheKey, {
          audioBuffer,
          timestamp: Date.now()
        });
        await this.saveTTSCache();
      }

      return audioBuffer;

    } catch (error) {
      console.error('TTS synthesis failed:', error);
      throw error;
    }
  }

  /**
   * Synthesize with specific provider
   */
  async synthesizeWithProvider(text, provider, options) {
    switch (provider) {
      case 'azure':
        return await this.synthesizeWithAzure(text, options);
      case 'elevenlabs':
        return await this.synthesizeWithElevenLabs(text, options);
      case 'google':
        return await this.synthesizeWithGoogle(text, options);
      default:
        throw new Error(`Unknown TTS provider: ${provider}`);
    }
  }

  /**
   * Synthesize with Azure Speech Services
   */
  async synthesizeWithAzure(text, options) {
    try {
      // Build SSML if emotion or advanced features are requested
      let ssmlText = text;
      if (this.config.enableSSML && (options.emotion || options.speed !== 1.0)) {
        ssmlText = this.buildAzureSSML(text, options);
      }

      const response = await axios.post(
        `https://${this.config.azureRegion}.tts.speech.microsoft.com/cognitiveservices/v1`,
        ssmlText,
        {
          headers: {
            'Ocp-Apim-Subscription-Key': this.config.azureKey,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': this.config.audioFormat,
            'User-Agent': 'VoiceAI-System'
          },
          responseType: 'arraybuffer'
        }
      );

      return Buffer.from(response.data);
    } catch (error) {
      throw new Error(`Azure TTS failed: ${error.message}`);
    }
  }

  /**
   * Synthesize with ElevenLabs
   */
  async synthesizeWithElevenLabs(text, options) {
    try {
      const voiceId = this.findElevenLabsVoice(options.voice) || 'EXAVITQu4vr4xnSDxMaL'; // Default voice

      const response = await axios.post(
        `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
        {
          text,
          model_id: 'eleven_monolingual_v1',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.5,
            style: options.emotion ? this.mapEmotionToElevenLabsStyle(options.emotion) : 0,
            use_speaker_boost: true
          }
        },
        {
          headers: {
            'xi-api-key': this.config.elevenLabsApiKey,
            'Content-Type': 'application/json',
            'Accept': 'audio/mpeg'
          },
          responseType: 'arraybuffer'
        }
      );

      return Buffer.from(response.data);
    } catch (error) {
      throw new Error(`ElevenLabs TTS failed: ${error.message}`);
    }
  }

  /**
   * Synthesize with Google Cloud Text-to-Speech
   */
  async synthesizeWithGoogle(text, options) {
    try {
      const response = await axios.post(
        `https://texttospeech.googleapis.com/v1/text:synthesize?key=${this.config.googleApiKey}`,
        {
          input: { text },
          voice: {
            languageCode: options.language,
            name: options.voice,
            ssmlGender: 'NEUTRAL'
          },
          audioConfig: {
            audioEncoding: 'MP3',
            speakingRate: options.speed,
            pitch: options.pitch * 4 - 20 // Convert to semitone range
          }
        }
      );

      const audioContent = response.data.audioContent;
      return Buffer.from(audioContent, 'base64');
    } catch (error) {
      throw new Error(`Google TTS failed: ${error.message}`);
    }
  }

  /**
   * Local TTS synthesis fallback
   */
  synthesizeLocally(text, options) {
    // Return a simple audio file indication
    // In production, this could use espeak or similar local TTS
    const message = `Text-to-speech unavailable: ${text}`;
    return Buffer.from(message, 'utf8');
  }

  /**
   * Build Azure SSML for advanced speech features
   */
  buildAzureSSML(text, options) {
    const { voice, emotion, speed, pitch } = options;
    
    let ssml = `<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">`;
    ssml += `<voice name="${voice || this.config.defaultVoice}">`;
    
    if (emotion && this.config.enableEmotionalTTS) {
      ssml += `<mstts:express-as style="${emotion}">`;
    }
    
    if (speed !== 1.0 || pitch !== 1.0) {
      const ratePercent = Math.round((speed - 1) * 100);
      const pitchPercent = Math.round((pitch - 1) * 100);
      ssml += `<prosody rate="${ratePercent >= 0 ? '+' : ''}${ratePercent}%" pitch="${pitchPercent >= 0 ? '+' : ''}${pitchPercent}%">`;
    }
    
    ssml += text;
    
    if (speed !== 1.0 || pitch !== 1.0) {
      ssml += `</prosody>`;
    }
    
    if (emotion && this.config.enableEmotionalTTS) {
      ssml += `</mstts:express-as>`;
    }
    
    ssml += `</voice></speak>`;
    
    return ssml;
  }

  /**
   * Find ElevenLabs voice ID
   */
  findElevenLabsVoice(voiceName) {
    for (const [id, voice] of this.supportedVoices.entries()) {
      if (voice.provider === 'elevenlabs' && voice.name === voiceName) {
        return id;
      }
    }
    return null;
  }

  /**
   * Map emotion to ElevenLabs style
   */
  mapEmotionToElevenLabsStyle(emotion) {
    const emotionMap = {
      happy: 0.8,
      sad: 0.2,
      angry: 0.9,
      excited: 0.9,
      calm: 0.1,
      neutral: 0.5
    };
    
    return emotionMap[emotion.toLowerCase()] || 0.5;
  }

  /**
   * Convert audio data to WAV format
   */
  convertToWav(audioData) {
    // Simple WAV header creation for 16-bit PCM
    const buffer = Buffer.alloc(44 + audioData.length * 2);
    
    // WAV header
    buffer.write('RIFF', 0);
    buffer.writeUInt32LE(36 + audioData.length * 2, 4);
    buffer.write('WAVE', 8);
    buffer.write('fmt ', 12);
    buffer.writeUInt32LE(16, 16);
    buffer.writeUInt16LE(1, 20); // PCM format
    buffer.writeUInt16LE(this.config.channels, 22);
    buffer.writeUInt32LE(this.config.sampleRate, 24);
    buffer.writeUInt32LE(this.config.sampleRate * this.config.channels * 2, 28);
    buffer.writeUInt16LE(this.config.channels * 2, 32);
    buffer.writeUInt16LE(16, 34); // 16-bit
    buffer.write('data', 36);
    buffer.writeUInt32LE(audioData.length * 2, 40);
    
    // Audio data
    for (let i = 0; i < audioData.length; i++) {
      buffer.writeInt16LE(audioData[i], 44 + i * 2);
    }
    
    return buffer;
  }

  /**
   * Extract word timestamps from Azure results
   */
  extractAzureWordTimestamps(result) {
    if (!result.NBest || !result.NBest[0] || !result.NBest[0].Words) {
      return null;
    }
    
    return result.NBest[0].Words.map(word => ({
      word: word.Word,
      startTime: word.Offset / 10000000, // Convert from 100-nanosecond units to seconds
      endTime: (word.Offset + word.Duration) / 10000000,
      confidence: word.Confidence
    }));
  }

  /**
   * Generate cache key for TTS
   */
  generateTTSCacheKey(text, language, voice, emotion, speed, pitch) {
    const crypto = require('crypto');
    const data = `${text}|${language}|${voice}|${emotion}|${speed}|${pitch}`;
    return crypto.createHash('md5').update(data).digest('hex');
  }

  /**
   * Save TTS cache to disk
   */
  async saveTTSCache() {
    try {
      const cacheFile = path.join(this.config.cacheDirectory, 'tts_cache.json');
      const cacheData = Object.fromEntries(
        Array.from(this.ttsCache.entries()).map(([key, value]) => [
          key,
          { ...value, audioBuffer: null } // Don't save audio buffers to JSON
        ])
      );
      await fs.writeFile(cacheFile, JSON.stringify(cacheData, null, 2));
    } catch (error) {
      console.warn('Failed to save TTS cache:', error.message);
    }
  }

  /**
   * Clean up expired cache entries
   */
  cleanupExpiredCache() {
    const now = Date.now();
    for (const [key, entry] of this.ttsCache.entries()) {
      if (now - entry.timestamp > this.config.cacheTimeout) {
        this.ttsCache.delete(key);
      }
    }
  }

  /**
   * Clean up old transcription sessions
   */
  cleanupOldSessions() {
    const now = Date.now();
    const maxAge = 60 * 60 * 1000; // 1 hour
    
    for (const [sessionId, session] of this.transcriptionSessions.entries()) {
      if (now - session.lastActivity > maxAge) {
        this.transcriptionSessions.delete(sessionId);
      }
    }
  }

  /**
   * Get supported voices
   */
  getSupportedVoices(provider = null) {
    if (provider) {
      return Array.from(this.supportedVoices.entries())
        .filter(([_, voice]) => voice.provider === provider)
        .map(([id, voice]) => ({ id, ...voice }));
    }
    
    return Array.from(this.supportedVoices.entries())
      .map(([id, voice]) => ({ id, ...voice }));
  }

  /**
   * Get supported languages
   */
  getSupportedLanguages() {
    return Array.from(this.supportedLanguages);
  }

  /**
   * Get transcription session status
   */
  getTranscriptionSessionStatus(sessionId) {
    const session = this.transcriptionSessions.get(sessionId);
    if (!session) {
      return { exists: false };
    }
    
    return {
      exists: true,
      language: session.language,
      totalAudioTime: session.totalAudioTime,
      partialResults: session.partialResults.length,
      finalResults: session.finalResults.length,
      lastActivity: session.lastActivity
    };
  }

  /**
   * End transcription session
   */
  endTranscriptionSession(sessionId) {
    const session = this.transcriptionSessions.get(sessionId);
    if (session) {
      this.transcriptionSessions.delete(sessionId);
      return {
        ended: true,
        finalResults: session.finalResults,
        totalAudioTime: session.totalAudioTime
      };
    }
    
    return { ended: false, reason: 'Session not found' };
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
      supportedVoices: this.supportedVoices.size,
      supportedLanguages: this.supportedLanguages.size,
      activeTranscriptionSessions: this.transcriptionSessions.size,
      cacheSize: this.ttsCache.size,
      features: {
        realTimeTranscription: this.config.enableRealTimeTranscription,
        emotionalTTS: this.config.enableEmotionalTTS,
        voiceEffects: this.config.enableVoiceEffects,
        caching: this.config.enableCaching
      }
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    console.log('Cleaning up Speech Service...');
    
    // Save TTS cache
    await this.saveTTSCache();
    
    // Clear data structures
    this.transcriptionSessions.clear();
    this.ttsCache.clear();
    this.supportedVoices.clear();
    this.supportedLanguages.clear();
    
    this.isInitialized = false;
    this.removeAllListeners();
  }
}

export default SpeechService;