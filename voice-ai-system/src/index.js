/**
 * Voice AI System Entry Point
 * Advanced voice and NLP interface with LLM integration
 */

import { VoiceServer } from './server/VoiceServer.js';
import dotenv from 'dotenv';
import logger from './server/logger.js';

// Load environment variables
dotenv.config();

/**
 * Main application entry point
 */
async function main() {
  logger.info('🎙️  Starting Advanced Voice AI System...');
  logger.info('=====================================');
  
  try {
    // Configuration for the voice server
    const serverConfig = {
      port: process.env.PORT || 8080,
      corsOrigin: process.env.CORS_ORIGIN || "*",
      
      // Voice processing settings
      sampleRate: parseInt(process.env.SAMPLE_RATE) || 16000,
      channels: parseInt(process.env.CHANNELS) || 1,
      
      // Provider API keys from environment
      // Emotion Detection
      humeApiKey: process.env.HUME_API_KEY,
      azureKey: process.env.AZURE_COGNITIVE_KEY,
      azureRegion: process.env.AZURE_REGION,
      
      // Language Processing & Translation
      googleApiKey: process.env.GOOGLE_TRANSLATE_API_KEY,
      deeplApiKey: process.env.DEEPL_API_KEY,
      
      // LLM Processing
      openaiApiKey: process.env.OPENAI_API_KEY,
      anthropicApiKey: process.env.ANTHROPIC_API_KEY,
      geminiApiKey: process.env.GEMINI_API_KEY,
      
      // Speech Services
      azureSpeechKey: process.env.AZURE_SPEECH_KEY,
      elevenLabsApiKey: process.env.ELEVENLABS_API_KEY,
      
      // Voice Biometrics
      encryptionKey: process.env.BIOMETRIC_ENCRYPTION_KEY,
      
      // Media Library
      mediaDirectory: process.env.MEDIA_DIRECTORY || './media',
      
      // Feature flags
      enableEmotionDetection: process.env.ENABLE_EMOTION_DETECTION !== 'false',
      enableBiometrics: process.env.ENABLE_BIOMETRICS !== 'false',
      enableRealTimeTranslation: process.env.ENABLE_REAL_TIME_TRANSLATION !== 'false',
      enableLLMProcessing: process.env.ENABLE_LLM_PROCESSING !== 'false',
      enableMediaLibrary: process.env.ENABLE_MEDIA_LIBRARY !== 'false'
    };

    // Create and start the voice server
    const voiceServer = new VoiceServer(serverConfig);
    
    // Start the server
    await voiceServer.start();
    
    // Display startup information
    displayStartupInfo(serverConfig);
    
    // Handle graceful shutdown
    setupGracefulShutdown(voiceServer);
    
  } catch (error) {
    logger.error('❌ Failed to start Voice AI System:', error);
    process.exit(1);
  }
}

/**
 * Display startup information
 */
function displayStartupInfo(config) {
  console.log('\n🎉 Voice AI System Successfully Started!');
  console.log('======================================');
  console.log(`🌐 Server: http://localhost:${config.port}`);
  console.log(`🎙️  WebSocket: ws://localhost:${config.port}/voice`);
  console.log(`📡 Socket.IO: http://localhost:${config.port}/socket.io/`);
  console.log(`📊 Health Check: http://localhost:${config.port}/health`);
  console.log('');
  
  // Display enabled features
  console.log('🔧 Enabled Features:');
  console.log('==================');
  
  if (config.enableEmotionDetection) {
    console.log('✅ Emotion Detection & Sentiment Analysis');
  }
  
  if (config.enableBiometrics) {
    console.log('✅ Voice Biometric Authentication');
  }
  
  if (config.enableRealTimeTranslation) {
    console.log('✅ Real-time Translation (100+ Languages)');
  }
  
  if (config.enableLLMProcessing) {
    console.log('✅ LLM-Powered Voice Responses');
  }
  
  if (config.enableMediaLibrary) {
    console.log('✅ Natural Language Media Library Interface');
  }
  
  console.log('✅ WebRTC Voice Processing');
  console.log('✅ Multi-Provider Speech Services');
  console.log('✅ Voice-Controlled Media Editing');
  console.log('');
  
  // Display API endpoints
  console.log('📋 Available Endpoints:');
  console.log('======================');
  console.log('GET  /health                    - Service health check');
  console.log('GET  /api/languages            - Supported languages');
  console.log('POST /api/tts                  - Text-to-speech');
  console.log('POST /api/translate            - Text translation');
  console.log('GET  /api/media/search         - Media library search');
  console.log('POST /api/biometric/enroll     - Voice biometric enrollment');
  console.log('');
  
  // Display WebSocket events
  console.log('🔌 WebSocket Events:');
  console.log('===================');
  console.log('session_start          - Initialize voice session');
  console.log('audio_chunk           - Send audio data for processing');
  console.log('start_listening       - Start voice recognition');
  console.log('stop_listening        - Stop voice recognition');
  console.log('translate_request     - Request real-time translation');
  console.log('llm_process          - Process text with LLM');
  console.log('enroll_biometric     - Enroll voice biometric');
  console.log('');
  
  // Display Socket.IO events
  console.log('🎯 Socket.IO Events:');
  console.log('===================');
  console.log('media_conversation    - Natural language media queries');
  console.log('voice_edit_command   - Voice-controlled media editing');
  console.log('');
  
  console.log('💡 Example Usage:');
  console.log('================');
  console.log('1. Connect to WebSocket: ws://localhost:' + config.port + '/voice');
  console.log('2. Send session_start message with language preferences');
  console.log('3. Stream audio_chunk messages with voice data');
  console.log('4. Receive real-time transcription, emotion, and translation');
  console.log('5. Use Socket.IO for conversational media library access');
  console.log('');
  
  console.log('🚀 System Ready for Voice Interactions!');
  console.log('Press Ctrl+C to stop the server');
  console.log('');
}

/**
 * Setup graceful shutdown handling
 */
function setupGracefulShutdown(voiceServer) {
  const signals = ['SIGINT', 'SIGTERM', 'SIGUSR2'];
  
  signals.forEach(signal => {
    process.on(signal, async () => {
      console.log(`\n📴 Received ${signal}, shutting down gracefully...`);
      
      try {
        await voiceServer.shutdown();
        console.log('✅ Voice AI System shutdown complete');
        process.exit(0);
      } catch (error) {
        console.error('❌ Error during shutdown:', error);
        process.exit(1);
      }
    });
  });
  
  // Handle uncaught exceptions
  process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    process.exit(1);
  });
  
  // Handle unhandled promise rejections
  process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
  });
}

/**
 * Display environment setup instructions
 */
function displayEnvironmentInstructions() {
  console.log('🔧 Environment Setup Instructions:');
  console.log('==================================');
  console.log('');
  console.log('Create a .env file in your project root with the following keys:');
  console.log('');
  console.log('# Server Configuration');
  console.log('PORT=8080');
  console.log('CORS_ORIGIN=*');
  console.log('');
  console.log('# Audio Processing');
  console.log('SAMPLE_RATE=16000');
  console.log('CHANNELS=1');
  console.log('');
  console.log('# Azure Services (Speech + Cognitive)');
  console.log('AZURE_SPEECH_KEY=your_azure_speech_key');
  console.log('AZURE_COGNITIVE_KEY=your_azure_cognitive_key');
  console.log('AZURE_REGION=your_azure_region');
  console.log('');
  console.log('# Google Services');
  console.log('GOOGLE_TRANSLATE_API_KEY=your_google_translate_key');
  console.log('GOOGLE_SPEECH_API_KEY=your_google_speech_key');
  console.log('');
  console.log('# OpenAI');
  console.log('OPENAI_API_KEY=your_openai_key');
  console.log('');
  console.log('# Anthropic Claude');
  console.log('ANTHROPIC_API_KEY=your_anthropic_key');
  console.log('');
  console.log('# Google Gemini');
  console.log('GEMINI_API_KEY=your_gemini_key');
  console.log('');
  console.log('# DeepL Translation');
  console.log('DEEPL_API_KEY=your_deepl_key');
  console.log('');
  console.log('# ElevenLabs TTS');
  console.log('ELEVENLABS_API_KEY=your_elevenlabs_key');
  console.log('');
  console.log('# Hume AI Emotion Detection');
  console.log('HUME_API_KEY=your_hume_key');
  console.log('');
  console.log('# Voice Biometrics');
  console.log('BIOMETRIC_ENCRYPTION_KEY=your_encryption_key');
  console.log('');
  console.log('# Media Library');
  console.log('MEDIA_DIRECTORY=./media');
  console.log('');
  console.log('# Feature Flags (optional - defaults to true)');
  console.log('ENABLE_EMOTION_DETECTION=true');
  console.log('ENABLE_BIOMETRICS=true');
  console.log('ENABLE_REAL_TIME_TRANSLATION=true');
  console.log('ENABLE_LLM_PROCESSING=true');
  console.log('ENABLE_MEDIA_LIBRARY=true');
  console.log('');
}

// Check if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  // Check for help flag
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    displayEnvironmentInstructions();
    process.exit(0);
  }
  
  // Start the application
  main().catch(error => {
    console.error('❌ Application startup failed:', error);
    process.exit(1);
  });
}

export default main;