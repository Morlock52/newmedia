/**
 * Voice Biometric Authentication Service
 * Implements speaker recognition and verification using voice characteristics
 */

import crypto from 'crypto';
import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';

export class BiometricAuth extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      // Storage configuration
      storageDir: config.storageDir || './data/voiceprints',
      
      // Feature extraction parameters
      frameSize: config.frameSize || 512,
      hopSize: config.hopSize || 256,
      sampleRate: config.sampleRate || 16000,
      
      // Authentication thresholds
      enrollmentThreshold: config.enrollmentThreshold || 0.85,
      verificationThreshold: config.verificationThreshold || 0.75,
      minEnrollmentSamples: config.minEnrollmentSamples || 3,
      maxEnrollmentSamples: config.maxEnrollmentSamples || 10,
      
      // Security settings
      encryptionKey: config.encryptionKey || process.env.BIOMETRIC_ENCRYPTION_KEY,
      saltRounds: config.saltRounds || 12,
      sessionTimeout: config.sessionTimeout || 3600000, // 1 hour
      
      // Feature extraction options
      extractMFCC: config.extractMFCC !== false,
      extractPitch: config.extractPitch !== false,
      extractFormants: config.extractFormants !== false,
      extractSpectralFeatures: config.extractSpectralFeatures !== false,
      
      // Anti-spoofing
      livenesDetection: config.livenesDetection !== false,
      replayAttackDetection: config.replayAttackDetection !== false,
      
      ...config
    };

    // Storage for voice prints and sessions
    this.voicePrints = new Map();
    this.activeSessions = new Map();
    this.enrollmentSessions = new Map();
    
    // Feature extraction components
    this.featureExtractor = null;
    this.modelMatcher = null;
    this.antiSpoofingDetector = null;
    
    this.isInitialized = false;
  }

  /**
   * Initialize the biometric authentication service
   */
  async initialize() {
    try {
      console.log('Initializing Voice Biometric Authentication Service...');
      
      // Create storage directory if it doesn't exist
      await this.ensureStorageDirectory();
      
      // Initialize feature extraction components
      this.featureExtractor = new VoiceFeatureExtractor(this.config);
      this.modelMatcher = new VoiceModelMatcher(this.config);
      this.antiSpoofingDetector = new AntiSpoofingDetector(this.config);
      
      // Load existing voice prints
      await this.loadVoicePrints();
      
      // Setup cleanup intervals
      this.setupCleanupIntervals();
      
      this.isInitialized = true;
      console.log('Voice Biometric Authentication Service initialized successfully');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize Voice Biometric Authentication Service:', error);
      throw error;
    }
  }

  /**
   * Ensure storage directory exists
   */
  async ensureStorageDirectory() {
    try {
      await fs.access(this.config.storageDir);
    } catch {
      await fs.mkdir(this.config.storageDir, { recursive: true });
    }
  }

  /**
   * Load existing voice prints from storage
   */
  async loadVoicePrints() {
    try {
      const files = await fs.readdir(this.config.storageDir);
      const voicePrintFiles = files.filter(file => file.endsWith('.vp'));
      
      for (const file of voicePrintFiles) {
        const filePath = path.join(this.config.storageDir, file);
        const data = await fs.readFile(filePath, 'utf8');
        const voicePrint = JSON.parse(data);
        
        // Decrypt if encrypted
        if (voicePrint.encrypted && this.config.encryptionKey) {
          voicePrint.features = this.decrypt(voicePrint.features);
        }
        
        this.voicePrints.set(voicePrint.speakerId, voicePrint);
      }
      
      console.log(`Loaded ${this.voicePrints.size} voice prints`);
    } catch (error) {
      console.warn('Failed to load voice prints:', error.message);
    }
  }

  /**
   * Setup cleanup intervals for sessions
   */
  setupCleanupIntervals() {
    // Clean expired sessions every 5 minutes
    setInterval(() => {
      this.cleanupExpiredSessions();
    }, 300000);
    
    // Clean old enrollment sessions every hour
    setInterval(() => {
      this.cleanupExpiredEnrollments();
    }, 3600000);
  }

  /**
   * Start voice biometric enrollment for a speaker
   */
  async startEnrollment(speakerId, options = {}) {
    if (!this.isInitialized) {
      throw new Error('Biometric service not initialized');
    }

    if (!speakerId) {
      throw new Error('Speaker ID is required');
    }

    const enrollmentId = crypto.randomUUID();
    const enrollmentSession = {
      enrollmentId,
      speakerId,
      samples: [],
      features: [],
      startTime: Date.now(),
      options: {
        minSamples: options.minSamples || this.config.minEnrollmentSamples,
        maxSamples: options.maxSamples || this.config.maxEnrollmentSamples,
        ...options
      }
    };

    this.enrollmentSessions.set(enrollmentId, enrollmentSession);
    
    this.emit('enrollmentStarted', { enrollmentId, speakerId });
    
    return {
      enrollmentId,
      speakerId,
      status: 'started',
      samplesNeeded: enrollmentSession.options.minSamples
    };
  }

  /**
   * Add audio sample to enrollment session
   */
  async addEnrollmentSample(enrollmentId, audioData, options = {}) {
    const enrollment = this.enrollmentSessions.get(enrollmentId);
    if (!enrollment) {
      throw new Error('Enrollment session not found');
    }

    try {
      // Perform liveness detection if enabled
      if (this.config.livenesDetection) {
        const livenessResult = await this.antiSpoofingDetector.detectLiveness(audioData);
        if (!livenessResult.isLive) {
          throw new Error('Liveness detection failed: recorded or synthesized audio detected');
        }
      }

      // Extract voice features
      const features = await this.featureExtractor.extract(audioData, {
        speakerId: enrollment.speakerId,
        sampleIndex: enrollment.samples.length,
        ...options
      });

      // Validate feature quality
      const qualityScore = this.assessFeatureQuality(features);
      if (qualityScore < 0.6) {
        return {
          enrollmentId,
          status: 'sample_rejected',
          reason: 'Poor audio quality',
          qualityScore,
          samplesCollected: enrollment.samples.length,
          samplesNeeded: enrollment.options.minSamples - enrollment.samples.length
        };
      }

      // Add sample to enrollment
      enrollment.samples.push({
        audioData: this.compressAudioData(audioData),
        features,
        qualityScore,
        timestamp: Date.now()
      });

      enrollment.features.push(features);

      const samplesCollected = enrollment.samples.length;
      const samplesNeeded = Math.max(0, enrollment.options.minSamples - samplesCollected);

      this.emit('enrollmentSampleAdded', {
        enrollmentId,
        speakerId: enrollment.speakerId,
        samplesCollected,
        samplesNeeded,
        qualityScore
      });

      // Check if we have enough samples for enrollment
      if (samplesCollected >= enrollment.options.minSamples) {
        return await this.completeEnrollment(enrollmentId);
      }

      return {
        enrollmentId,
        status: 'sample_accepted',
        samplesCollected,
        samplesNeeded,
        qualityScore
      };

    } catch (error) {
      console.error('Enrollment sample processing failed:', error);
      throw error;
    }
  }

  /**
   * Complete the enrollment process
   */
  async completeEnrollment(enrollmentId) {
    const enrollment = this.enrollmentSessions.get(enrollmentId);
    if (!enrollment) {
      throw new Error('Enrollment session not found');
    }

    try {
      // Generate voice print model from collected features
      const voicePrintModel = await this.modelMatcher.createModel(enrollment.features, {
        speakerId: enrollment.speakerId,
        samplesCount: enrollment.samples.length
      });

      // Calculate model quality and consistency
      const modelQuality = this.assessModelQuality(voicePrintModel, enrollment.features);
      
      if (modelQuality < this.config.enrollmentThreshold) {
        return {
          enrollmentId,
          status: 'enrollment_failed',
          reason: 'Insufficient voice consistency across samples',
          modelQuality,
          recommendation: 'Please provide additional clear voice samples'
        };
      }

      // Create voice print
      const voicePrint = {
        speakerId: enrollment.speakerId,
        model: voicePrintModel,
        enrollmentDate: new Date().toISOString(),
        samplesCount: enrollment.samples.length,
        modelQuality,
        version: '1.0',
        encrypted: !!this.config.encryptionKey,
        features: this.config.encryptionKey ? 
          this.encrypt(JSON.stringify(voicePrintModel)) : 
          voicePrintModel
      };

      // Save voice print
      await this.saveVoicePrint(voicePrint);
      
      // Store in memory
      this.voicePrints.set(enrollment.speakerId, voicePrint);
      
      // Clean up enrollment session
      this.enrollmentSessions.delete(enrollmentId);
      
      this.emit('enrollmentCompleted', {
        enrollmentId,
        speakerId: enrollment.speakerId,
        modelQuality,
        samplesUsed: enrollment.samples.length
      });

      return {
        enrollmentId,
        status: 'enrollment_completed',
        speakerId: enrollment.speakerId,
        modelQuality,
        samplesUsed: enrollment.samples.length
      };

    } catch (error) {
      console.error('Enrollment completion failed:', error);
      throw error;
    }
  }

  /**
   * Verify speaker identity using voice biometrics
   */
  async verify(audioData, speakerId, options = {}) {
    if (!this.isInitialized) {
      throw new Error('Biometric service not initialized');
    }

    if (!speakerId) {
      throw new Error('Speaker ID is required');
    }

    const voicePrint = this.voicePrints.get(speakerId);
    if (!voicePrint) {
      return {
        verified: false,
        speakerId,
        confidence: 0,
        reason: 'Speaker not enrolled'
      };
    }

    try {
      // Perform anti-spoofing checks
      if (this.config.replayAttackDetection) {
        const spoofingResult = await this.antiSpoofingDetector.detectReplayAttack(audioData);
        if (spoofingResult.isReplay) {
          return {
            verified: false,
            speakerId,
            confidence: 0,
            reason: 'Replay attack detected',
            securityFlag: true
          };
        }
      }

      // Extract features from verification audio
      const verificationFeatures = await this.featureExtractor.extract(audioData, {
        speakerId,
        verificationType: 'verification',
        ...options
      });

      // Assess feature quality
      const qualityScore = this.assessFeatureQuality(verificationFeatures);
      if (qualityScore < 0.5) {
        return {
          verified: false,
          speakerId,
          confidence: 0,
          reason: 'Poor audio quality for verification',
          qualityScore
        };
      }

      // Get model from voice print (decrypt if necessary)
      let model = voicePrint.model;
      if (voicePrint.encrypted && this.config.encryptionKey) {
        model = JSON.parse(this.decrypt(voicePrint.features));
      }

      // Perform voice matching
      const matchResult = await this.modelMatcher.match(verificationFeatures, model, {
        threshold: this.config.verificationThreshold,
        speakerId
      });

      const verified = matchResult.similarity >= this.config.verificationThreshold;
      const confidence = matchResult.similarity;

      // Create verification session if successful
      if (verified) {
        const sessionId = crypto.randomUUID();
        this.activeSessions.set(sessionId, {
          sessionId,
          speakerId,
          verificationTime: Date.now(),
          confidence,
          expiresAt: Date.now() + this.config.sessionTimeout
        });
      }

      this.emit('verificationCompleted', {
        speakerId,
        verified,
        confidence,
        qualityScore,
        sessionId: verified ? sessionId : null
      });

      return {
        verified,
        speakerId,
        confidence,
        qualityScore,
        sessionId: verified ? sessionId : null,
        voicePrintMatch: matchResult.similarity,
        threshold: this.config.verificationThreshold
      };

    } catch (error) {
      console.error('Voice verification failed:', error);
      return {
        verified: false,
        speakerId,
        confidence: 0,
        reason: error.message
      };
    }
  }

  /**
   * Perform identification (1:N matching) to find speaker among enrolled users
   */
  async identify(audioData, options = {}) {
    if (!this.isInitialized) {
      throw new Error('Biometric service not initialized');
    }

    if (this.voicePrints.size === 0) {
      return {
        identified: false,
        confidence: 0,
        reason: 'No enrolled speakers'
      };
    }

    try {
      // Extract features from identification audio
      const identificationFeatures = await this.featureExtractor.extract(audioData, {
        verificationType: 'identification',
        ...options
      });

      // Assess feature quality
      const qualityScore = this.assessFeatureQuality(identificationFeatures);
      if (qualityScore < 0.5) {
        return {
          identified: false,
          confidence: 0,
          reason: 'Poor audio quality for identification',
          qualityScore
        };
      }

      // Match against all enrolled voice prints
      const matchResults = [];
      
      for (const [speakerId, voicePrint] of this.voicePrints.entries()) {
        try {
          // Get model (decrypt if necessary)
          let model = voicePrint.model;
          if (voicePrint.encrypted && this.config.encryptionKey) {
            model = JSON.parse(this.decrypt(voicePrint.features));
          }

          const matchResult = await this.modelMatcher.match(identificationFeatures, model, {
            speakerId
          });

          matchResults.push({
            speakerId,
            similarity: matchResult.similarity,
            confidence: matchResult.similarity
          });
        } catch (error) {
          console.warn(`Failed to match against speaker ${speakerId}:`, error.message);
        }
      }

      // Sort by confidence and get best match
      matchResults.sort((a, b) => b.confidence - a.confidence);
      
      const bestMatch = matchResults[0];
      const identified = bestMatch && bestMatch.confidence >= this.config.verificationThreshold;

      if (identified) {
        // Create identification session
        const sessionId = crypto.randomUUID();
        this.activeSessions.set(sessionId, {
          sessionId,
          speakerId: bestMatch.speakerId,
          verificationTime: Date.now(),
          confidence: bestMatch.confidence,
          expiresAt: Date.now() + this.config.sessionTimeout
        });
      }

      this.emit('identificationCompleted', {
        identified,
        speakerId: identified ? bestMatch.speakerId : null,
        confidence: identified ? bestMatch.confidence : 0,
        qualityScore,
        alternatives: matchResults.slice(0, 3) // Top 3 matches
      });

      return {
        identified,
        speakerId: identified ? bestMatch.speakerId : null,
        confidence: identified ? bestMatch.confidence : 0,
        qualityScore,
        sessionId: identified ? sessionId : null,
        alternatives: matchResults.slice(0, 3),
        threshold: this.config.verificationThreshold
      };

    } catch (error) {
      console.error('Voice identification failed:', error);
      return {
        identified: false,
        confidence: 0,
        reason: error.message
      };
    }
  }

  /**
   * Validate an active biometric session
   */
  validateSession(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      return { valid: false, reason: 'Session not found' };
    }

    if (Date.now() > session.expiresAt) {
      this.activeSessions.delete(sessionId);
      return { valid: false, reason: 'Session expired' };
    }

    return {
      valid: true,
      speakerId: session.speakerId,
      confidence: session.confidence,
      expiresAt: session.expiresAt
    };
  }

  /**
   * Revoke/delete a voice print
   */
  async revokeVoicePrint(speakerId) {
    if (!this.voicePrints.has(speakerId)) {
      throw new Error('Speaker not found');
    }

    try {
      // Remove from memory
      this.voicePrints.delete(speakerId);
      
      // Remove from storage
      const filePath = path.join(this.config.storageDir, `${speakerId}.vp`);
      await fs.unlink(filePath);
      
      // Invalidate any active sessions
      for (const [sessionId, session] of this.activeSessions.entries()) {
        if (session.speakerId === speakerId) {
          this.activeSessions.delete(sessionId);
        }
      }

      this.emit('voicePrintRevoked', { speakerId });
      
      return { success: true, speakerId };
    } catch (error) {
      console.error('Failed to revoke voice print:', error);
      throw error;
    }
  }

  /**
   * Get enrolled speakers list
   */
  getEnrolledSpeakers() {
    return Array.from(this.voicePrints.keys()).map(speakerId => {
      const voicePrint = this.voicePrints.get(speakerId);
      return {
        speakerId,
        enrollmentDate: voicePrint.enrollmentDate,
        samplesCount: voicePrint.samplesCount,
        modelQuality: voicePrint.modelQuality
      };
    });
  }

  /**
   * Save voice print to storage
   */
  async saveVoicePrint(voicePrint) {
    const filePath = path.join(this.config.storageDir, `${voicePrint.speakerId}.vp`);
    await fs.writeFile(filePath, JSON.stringify(voicePrint, null, 2), 'utf8');
  }

  /**
   * Assess feature quality for enrollment/verification
   */
  assessFeatureQuality(features) {
    // Simple quality assessment based on feature completeness and variance
    let qualityScore = 0.5; // Base score

    // Check if we have all expected feature types
    if (features.mfcc && features.mfcc.length > 0) qualityScore += 0.2;
    if (features.pitch && features.pitch.length > 0) qualityScore += 0.1;
    if (features.formants && features.formants.length > 0) qualityScore += 0.1;
    if (features.spectral && Object.keys(features.spectral).length > 0) qualityScore += 0.1;

    // Check feature variance (avoid silence or constant signals)
    if (features.mfcc) {
      const variance = this.calculateVariance(features.mfcc.flat());
      if (variance > 0.01) qualityScore += 0.1;
    }

    return Math.max(0, Math.min(1, qualityScore));
  }

  /**
   * Assess model quality after enrollment
   */
  assessModelQuality(model, allFeatures) {
    if (!model || !allFeatures || allFeatures.length < 2) {
      return 0;
    }

    // Calculate consistency across samples
    let totalConsistency = 0;
    let comparisons = 0;

    for (let i = 0; i < allFeatures.length; i++) {
      for (let j = i + 1; j < allFeatures.length; j++) {
        const similarity = this.calculateFeatureSimilarity(allFeatures[i], allFeatures[j]);
        totalConsistency += similarity;
        comparisons++;
      }
    }

    const averageConsistency = comparisons > 0 ? totalConsistency / comparisons : 0;
    
    // Model quality is based on consistency and feature richness
    const featureRichness = this.assessFeatureRichness(model);
    
    return (averageConsistency * 0.7) + (featureRichness * 0.3);
  }

  /**
   * Calculate feature similarity between two feature sets
   */
  calculateFeatureSimilarity(features1, features2) {
    if (!features1.mfcc || !features2.mfcc) return 0;

    // Simple cosine similarity for MFCC features
    const vec1 = features1.mfcc.flat();
    const vec2 = features2.mfcc.flat();
    
    if (vec1.length !== vec2.length) return 0;

    const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
    const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
    const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));

    return mag1 && mag2 ? dotProduct / (mag1 * mag2) : 0;
  }

  /**
   * Assess feature richness of a model
   */
  assessFeatureRichness(model) {
    let richness = 0;
    
    if (model.mfcc) richness += 0.4;
    if (model.pitch) richness += 0.2;
    if (model.formants) richness += 0.2;
    if (model.spectral) richness += 0.2;
    
    return richness;
  }

  /**
   * Calculate variance of an array
   */
  calculateVariance(array) {
    const mean = array.reduce((sum, val) => sum + val, 0) / array.length;
    const variance = array.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / array.length;
    return variance;
  }

  /**
   * Compress audio data for storage
   */
  compressAudioData(audioData) {
    // Simple compression by downsampling - in production use proper audio compression
    const compressionRatio = 4;
    const compressed = new Int16Array(Math.floor(audioData.length / compressionRatio));
    
    for (let i = 0; i < compressed.length; i++) {
      compressed[i] = audioData[i * compressionRatio];
    }
    
    return Array.from(compressed);
  }

  /**
   * Encrypt sensitive data
   */
  encrypt(data) {
    if (!this.config.encryptionKey) return data;
    
    const cipher = crypto.createCipher('aes-256-cbc', this.config.encryptionKey);
    let encrypted = cipher.update(data, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return encrypted;
  }

  /**
   * Decrypt sensitive data
   */
  decrypt(encryptedData) {
    if (!this.config.encryptionKey) return encryptedData;
    
    const decipher = crypto.createDecipher('aes-256-cbc', this.config.encryptionKey);
    let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }

  /**
   * Clean up expired sessions
   */
  cleanupExpiredSessions() {
    const now = Date.now();
    for (const [sessionId, session] of this.activeSessions.entries()) {
      if (now > session.expiresAt) {
        this.activeSessions.delete(sessionId);
      }
    }
  }

  /**
   * Clean up expired enrollment sessions
   */
  cleanupExpiredEnrollments() {
    const now = Date.now();
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    
    for (const [enrollmentId, enrollment] of this.enrollmentSessions.entries()) {
      if (now - enrollment.startTime > maxAge) {
        this.enrollmentSessions.delete(enrollmentId);
      }
    }
  }

  /**
   * Check if service is ready
   */
  isReady() {
    return this.isInitialized && this.featureExtractor && this.modelMatcher;
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      initialized: this.isInitialized,
      enrolledSpeakers: this.voicePrints.size,
      activeSessions: this.activeSessions.size,
      activeEnrollments: this.enrollmentSessions.size,
      features: {
        livenesDetection: this.config.livenesDetection,
        replayAttackDetection: this.config.replayAttackDetection,
        encryption: !!this.config.encryptionKey
      }
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    console.log('Cleaning up Voice Biometric Authentication Service...');
    this.activeSessions.clear();
    this.enrollmentSessions.clear();
    this.voicePrints.clear();
    this.isInitialized = false;
    this.removeAllListeners();
  }
}

/**
 * Voice Feature Extractor
 * Extracts MFCC, pitch, formants, and spectral features from audio
 */
class VoiceFeatureExtractor {
  constructor(config) {
    this.config = config;
  }

  async extract(audioData, options = {}) {
    const features = {};

    // Convert to Float32Array for processing
    const samples = new Float32Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      samples[i] = audioData[i] / 32768.0; // Normalize to [-1, 1]
    }

    // Extract MFCC features
    if (this.config.extractMFCC) {
      features.mfcc = this.extractMFCC(samples);
    }

    // Extract pitch features
    if (this.config.extractPitch) {
      features.pitch = this.extractPitch(samples);
    }

    // Extract formant frequencies
    if (this.config.extractFormants) {
      features.formants = this.extractFormants(samples);
    }

    // Extract spectral features
    if (this.config.extractSpectralFeatures) {
      features.spectral = this.extractSpectralFeatures(samples);
    }

    return features;
  }

  extractMFCC(samples) {
    // Simplified MFCC extraction - in production use proper DSP library
    const numCoeffs = 13;
    const numFrames = Math.floor((samples.length - this.config.frameSize) / this.config.hopSize) + 1;
    const mfcc = [];

    for (let frame = 0; frame < numFrames; frame++) {
      const frameStart = frame * this.config.hopSize;
      const frameEnd = Math.min(frameStart + this.config.frameSize, samples.length);
      const frameSamples = samples.slice(frameStart, frameEnd);
      
      // Apply window function (Hamming)
      const windowed = this.applyHammingWindow(frameSamples);
      
      // Compute FFT (simplified)
      const spectrum = this.computeSpectrum(windowed);
      
      // Mel filter bank and DCT (simplified)
      const melFeatures = this.applyMelFilterBank(spectrum);
      const mfccFrame = this.computeDCT(melFeatures, numCoeffs);
      
      mfcc.push(mfccFrame);
    }

    return mfcc;
  }

  extractPitch(samples) {
    // Simplified pitch extraction using autocorrelation
    const pitch = [];
    const frameSize = 1024;
    const hopSize = 512;
    const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;

    for (let frame = 0; frame < numFrames; frame++) {
      const frameStart = frame * hopSize;
      const frameEnd = Math.min(frameStart + frameSize, samples.length);
      const frameSamples = samples.slice(frameStart, frameEnd);
      
      const pitchValue = this.autocorrelationPitch(frameSamples);
      pitch.push(pitchValue);
    }

    return pitch;
  }

  extractFormants(samples) {
    // Simplified formant extraction - in production use LPC analysis
    const formants = [];
    const frameSize = 1024;
    const hopSize = 512;
    const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;

    for (let frame = 0; frame < numFrames; frame++) {
      const frameStart = frame * hopSize;
      const frameEnd = Math.min(frameStart + frameSize, samples.length);
      const frameSamples = samples.slice(frameStart, frameEnd);
      
      // Simplified formant estimation using spectral peaks
      const spectrum = this.computeSpectrum(frameSamples);
      const formantFreqs = this.findSpectralPeaks(spectrum, 3); // First 3 formants
      formants.push(formantFreqs);
    }

    return formants;
  }

  extractSpectralFeatures(samples) {
    const spectrum = this.computeSpectrum(samples);
    
    return {
      spectralCentroid: this.computeSpectralCentroid(spectrum),
      spectralRolloff: this.computeSpectralRolloff(spectrum),
      spectralFlux: this.computeSpectralFlux(spectrum),
      zeroCrossingRate: this.computeZeroCrossingRate(samples)
    };
  }

  // Helper methods for feature extraction
  applyHammingWindow(samples) {
    const windowed = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      windowed[i] = samples[i] * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (samples.length - 1)));
    }
    return windowed;
  }

  computeSpectrum(samples) {
    // Simplified spectrum computation - use FFT in production
    const spectrum = new Float32Array(samples.length / 2);
    for (let k = 0; k < spectrum.length; k++) {
      let real = 0, imag = 0;
      for (let n = 0; n < samples.length; n++) {
        const angle = -2 * Math.PI * k * n / samples.length;
        real += samples[n] * Math.cos(angle);
        imag += samples[n] * Math.sin(angle);
      }
      spectrum[k] = Math.sqrt(real * real + imag * imag);
    }
    return spectrum;
  }

  applyMelFilterBank(spectrum) {
    // Simplified mel filter bank - use proper implementation in production
    const numFilters = 26;
    const melFilters = new Float32Array(numFilters);
    
    for (let i = 0; i < numFilters; i++) {
      let sum = 0;
      const startBin = Math.floor(i * spectrum.length / numFilters);
      const endBin = Math.floor((i + 1) * spectrum.length / numFilters);
      
      for (let j = startBin; j < endBin; j++) {
        sum += spectrum[j];
      }
      
      melFilters[i] = sum;
    }
    
    return melFilters;
  }

  computeDCT(melFeatures, numCoeffs) {
    const dct = new Float32Array(numCoeffs);
    
    for (let k = 0; k < numCoeffs; k++) {
      let sum = 0;
      for (let n = 0; n < melFeatures.length; n++) {
        sum += melFeatures[n] * Math.cos(Math.PI * k * (n + 0.5) / melFeatures.length);
      }
      dct[k] = sum;
    }
    
    return Array.from(dct);
  }

  autocorrelationPitch(samples) {
    const minPeriod = Math.floor(this.config.sampleRate / 500); // 500 Hz max
    const maxPeriod = Math.floor(this.config.sampleRate / 50);  // 50 Hz min
    
    let maxCorr = 0;
    let bestPeriod = 0;
    
    for (let period = minPeriod; period <= maxPeriod; period++) {
      let corr = 0;
      for (let i = 0; i < samples.length - period; i++) {
        corr += samples[i] * samples[i + period];
      }
      
      if (corr > maxCorr) {
        maxCorr = corr;
        bestPeriod = period;
      }
    }
    
    return bestPeriod > 0 ? this.config.sampleRate / bestPeriod : 0;
  }

  findSpectralPeaks(spectrum, numPeaks) {
    const peaks = [];
    const threshold = Math.max(...spectrum) * 0.1; // 10% of max
    
    for (let i = 1; i < spectrum.length - 1; i++) {
      if (spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1] && spectrum[i] > threshold) {
        peaks.push({
          frequency: i * this.config.sampleRate / (2 * spectrum.length),
          magnitude: spectrum[i]
        });
      }
    }
    
    // Sort by magnitude and return top peaks
    peaks.sort((a, b) => b.magnitude - a.magnitude);
    return peaks.slice(0, numPeaks).map(peak => peak.frequency);
  }

  computeSpectralCentroid(spectrum) {
    let weightedSum = 0;
    let magnitudeSum = 0;
    
    for (let i = 0; i < spectrum.length; i++) {
      const frequency = i * this.config.sampleRate / (2 * spectrum.length);
      weightedSum += frequency * spectrum[i];
      magnitudeSum += spectrum[i];
    }
    
    return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
  }

  computeSpectralRolloff(spectrum) {
    const totalEnergy = spectrum.reduce((sum, val) => sum + val, 0);
    const threshold = totalEnergy * 0.85; // 85% rolloff point
    
    let cumulativeEnergy = 0;
    for (let i = 0; i < spectrum.length; i++) {
      cumulativeEnergy += spectrum[i];
      if (cumulativeEnergy >= threshold) {
        return i * this.config.sampleRate / (2 * spectrum.length);
      }
    }
    
    return this.config.sampleRate / 2;
  }

  computeSpectralFlux(spectrum) {
    // Simple spectral flux calculation
    let flux = 0;
    for (let i = 1; i < spectrum.length; i++) {
      const diff = spectrum[i] - spectrum[i-1];
      flux += diff * diff;
    }
    return Math.sqrt(flux);
  }

  computeZeroCrossingRate(samples) {
    let crossings = 0;
    for (let i = 1; i < samples.length; i++) {
      if ((samples[i] >= 0) !== (samples[i-1] >= 0)) {
        crossings++;
      }
    }
    return crossings / samples.length;
  }
}

/**
 * Voice Model Matcher
 * Handles creation and matching of voice print models
 */
class VoiceModelMatcher {
  constructor(config) {
    this.config = config;
  }

  async createModel(featuresArray, options = {}) {
    if (!featuresArray || featuresArray.length === 0) {
      throw new Error('No features provided for model creation');
    }

    // Create statistical model from features
    const model = {
      speakerId: options.speakerId,
      samplesCount: featuresArray.length,
      createdAt: new Date().toISOString()
    };

    // Process MFCC features
    if (featuresArray[0].mfcc) {
      model.mfcc = this.createStatisticalModel(
        featuresArray.map(f => f.mfcc).flat()
      );
    }

    // Process pitch features
    if (featuresArray[0].pitch) {
      model.pitch = this.createStatisticalModel(
        featuresArray.map(f => f.pitch).flat()
      );
    }

    // Process formant features
    if (featuresArray[0].formants) {
      model.formants = this.createStatisticalModel(
        featuresArray.map(f => f.formants).flat()
      );
    }

    // Process spectral features
    if (featuresArray[0].spectral) {
      model.spectral = this.aggregateSpectralFeatures(
        featuresArray.map(f => f.spectral)
      );
    }

    return model;
  }

  async match(verificationFeatures, enrolledModel, options = {}) {
    let totalSimilarity = 0;
    let weightSum = 0;

    // Match MFCC features (highest weight)
    if (verificationFeatures.mfcc && enrolledModel.mfcc) {
      const mfccSimilarity = this.calculateGaussianSimilarity(
        verificationFeatures.mfcc.flat(),
        enrolledModel.mfcc
      );
      totalSimilarity += mfccSimilarity * 0.5;
      weightSum += 0.5;
    }

    // Match pitch features
    if (verificationFeatures.pitch && enrolledModel.pitch) {
      const pitchSimilarity = this.calculateGaussianSimilarity(
        verificationFeatures.pitch,
        enrolledModel.pitch
      );
      totalSimilarity += pitchSimilarity * 0.2;
      weightSum += 0.2;
    }

    // Match formant features
    if (verificationFeatures.formants && enrolledModel.formants) {
      const formantSimilarity = this.calculateGaussianSimilarity(
        verificationFeatures.formants.flat(),
        enrolledModel.formants
      );
      totalSimilarity += formantSimilarity * 0.2;
      weightSum += 0.2;
    }

    // Match spectral features
    if (verificationFeatures.spectral && enrolledModel.spectral) {
      const spectralSimilarity = this.calculateSpectralSimilarity(
        verificationFeatures.spectral,
        enrolledModel.spectral
      );
      totalSimilarity += spectralSimilarity * 0.1;
      weightSum += 0.1;
    }

    const similarity = weightSum > 0 ? totalSimilarity / weightSum : 0;

    return {
      similarity,
      confidence: similarity,
      components: {
        mfcc: verificationFeatures.mfcc && enrolledModel.mfcc ? 
          this.calculateGaussianSimilarity(verificationFeatures.mfcc.flat(), enrolledModel.mfcc) : null,
        pitch: verificationFeatures.pitch && enrolledModel.pitch ? 
          this.calculateGaussianSimilarity(verificationFeatures.pitch, enrolledModel.pitch) : null,
        formants: verificationFeatures.formants && enrolledModel.formants ? 
          this.calculateGaussianSimilarity(verificationFeatures.formants.flat(), enrolledModel.formants) : null,
        spectral: verificationFeatures.spectral && enrolledModel.spectral ? 
          this.calculateSpectralSimilarity(verificationFeatures.spectral, enrolledModel.spectral) : null
      }
    };
  }

  createStatisticalModel(data) {
    if (!data || data.length === 0) return null;

    const flatData = Array.isArray(data[0]) ? data.flat() : data;
    const mean = flatData.reduce((sum, val) => sum + val, 0) / flatData.length;
    const variance = flatData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flatData.length;
    const stdDev = Math.sqrt(variance);

    return {
      mean,
      variance,
      stdDev,
      min: Math.min(...flatData),
      max: Math.max(...flatData),
      count: flatData.length
    };
  }

  aggregateSpectralFeatures(spectralArray) {
    const aggregated = {};
    const features = ['spectralCentroid', 'spectralRolloff', 'spectralFlux', 'zeroCrossingRate'];

    features.forEach(feature => {
      const values = spectralArray.map(s => s[feature]).filter(v => v !== undefined);
      if (values.length > 0) {
        aggregated[feature] = this.createStatisticalModel(values);
      }
    });

    return aggregated;
  }

  calculateGaussianSimilarity(data, model) {
    if (!data || !model || data.length === 0) return 0;

    const flatData = Array.isArray(data[0]) ? data.flat() : data;
    const testMean = flatData.reduce((sum, val) => sum + val, 0) / flatData.length;
    
    // Calculate similarity based on mean difference and model variance
    const meanDiff = Math.abs(testMean - model.mean);
    const normalizedDiff = model.stdDev > 0 ? meanDiff / model.stdDev : meanDiff;
    
    // Convert to similarity score (closer means = higher similarity)
    const similarity = Math.exp(-normalizedDiff * normalizedDiff / 2);
    
    return Math.max(0, Math.min(1, similarity));
  }

  calculateSpectralSimilarity(spectral1, spectral2) {
    const features = ['spectralCentroid', 'spectralRolloff', 'spectralFlux', 'zeroCrossingRate'];
    let totalSimilarity = 0;
    let validFeatures = 0;

    features.forEach(feature => {
      if (spectral1[feature] !== undefined && spectral2[feature] !== undefined) {
        const sim = this.calculateGaussianSimilarity([spectral1[feature]], spectral2[feature]);
        totalSimilarity += sim;
        validFeatures++;
      }
    });

    return validFeatures > 0 ? totalSimilarity / validFeatures : 0;
  }
}

/**
 * Anti-Spoofing Detector
 * Detects liveness and replay attacks
 */
class AntiSpoofingDetector {
  constructor(config) {
    this.config = config;
  }

  async detectLiveness(audioData) {
    // Simple liveness detection based on audio characteristics
    const features = this.extractLivenessFeatures(audioData);
    
    return {
      isLive: features.hasNaturalVariation && features.hasProperDynamics,
      confidence: this.calculateLivenessConfidence(features),
      features
    };
  }

  async detectReplayAttack(audioData) {
    // Simple replay attack detection
    const features = this.extractReplayFeatures(audioData);
    
    return {
      isReplay: features.hasCompressionArtifacts || features.hasUniformBackground,
      confidence: this.calculateReplayConfidence(features),
      features
    };
  }

  extractLivenessFeatures(audioData) {
    const samples = new Float32Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      samples[i] = audioData[i] / 32768.0;
    }

    // Calculate basic liveness indicators
    const energy = this.calculateEnergy(samples);
    const variance = this.calculateVariance(Array.from(samples));
    const dynamicRange = this.calculateDynamicRange(samples);
    const spectralVariation = this.calculateSpectralVariation(samples);

    return {
      energy,
      variance,
      dynamicRange,
      spectralVariation,
      hasNaturalVariation: variance > 0.001,
      hasProperDynamics: dynamicRange > 0.1
    };
  }

  extractReplayFeatures(audioData) {
    const samples = new Float32Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      samples[i] = audioData[i] / 32768.0;
    }

    // Look for compression artifacts and uniform background noise
    const compressionRatio = this.estimateCompressionRatio(samples);
    const backgroundUniformity = this.calculateBackgroundUniformity(samples);
    const frequencyResponse = this.analyzeFrequencyResponse(samples);

    return {
      compressionRatio,
      backgroundUniformity,
      frequencyResponse,
      hasCompressionArtifacts: compressionRatio > 10,
      hasUniformBackground: backgroundUniformity > 0.8
    };
  }

  calculateEnergy(samples) {
    return samples.reduce((sum, sample) => sum + sample * sample, 0) / samples.length;
  }

  calculateVariance(array) {
    const mean = array.reduce((sum, val) => sum + val, 0) / array.length;
    return array.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / array.length;
  }

  calculateDynamicRange(samples) {
    const max = Math.max(...samples);
    const min = Math.min(...samples);
    return max - min;
  }

  calculateSpectralVariation(samples) {
    // Simplified spectral variation calculation
    const frameSize = 512;
    const numFrames = Math.floor(samples.length / frameSize);
    const spectralCentroids = [];

    for (let i = 0; i < numFrames; i++) {
      const frame = samples.slice(i * frameSize, (i + 1) * frameSize);
      const centroid = this.calculateSpectralCentroid(frame);
      spectralCentroids.push(centroid);
    }

    return this.calculateVariance(spectralCentroids);
  }

  calculateSpectralCentroid(frame) {
    // Simple spectral centroid calculation
    let weightedSum = 0;
    let magnitudeSum = 0;
    
    for (let i = 0; i < frame.length; i++) {
      const magnitude = Math.abs(frame[i]);
      weightedSum += i * magnitude;
      magnitudeSum += magnitude;
    }
    
    return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
  }

  estimateCompressionRatio(samples) {
    // Simple compression estimation based on dynamic range
    const sortedSamples = Array.from(samples).sort((a, b) => Math.abs(b) - Math.abs(a));
    const p99 = sortedSamples[Math.floor(sortedSamples.length * 0.01)];
    const p1 = sortedSamples[Math.floor(sortedSamples.length * 0.99)];
    
    return p99 !== 0 ? Math.abs(p1 / p99) : 1;
  }

  calculateBackgroundUniformity(samples) {
    // Analyze the uniformity of background noise
    const frameSize = 256;
    const numFrames = Math.floor(samples.length / frameSize);
    const frameEnergies = [];

    for (let i = 0; i < numFrames; i++) {
      const frame = samples.slice(i * frameSize, (i + 1) * frameSize);
      const energy = frame.reduce((sum, sample) => sum + sample * sample, 0) / frame.length;
      frameEnergies.push(energy);
    }

    // Calculate coefficient of variation
    const mean = frameEnergies.reduce((sum, e) => sum + e, 0) / frameEnergies.length;
    const stdDev = Math.sqrt(this.calculateVariance(frameEnergies));
    
    return mean > 0 ? 1 - (stdDev / mean) : 0; // High uniformity = low variation
  }

  analyzeFrequencyResponse(samples) {
    // Simplified frequency response analysis
    // In production, use proper FFT-based analysis
    const lowFreq = this.bandpassEnergy(samples, 0, 0.1);
    const midFreq = this.bandpassEnergy(samples, 0.1, 0.5);
    const highFreq = this.bandpassEnergy(samples, 0.5, 1.0);
    
    return {
      low: lowFreq,
      mid: midFreq,
      high: highFreq,
      balance: midFreq / (lowFreq + highFreq + 0.001)
    };
  }

  bandpassEnergy(samples, lowCutoff, highCutoff) {
    // Simplified bandpass energy calculation
    // In production, use proper filtering
    let energy = 0;
    const startIdx = Math.floor(lowCutoff * samples.length);
    const endIdx = Math.floor(highCutoff * samples.length);
    
    for (let i = startIdx; i < endIdx && i < samples.length; i++) {
      energy += samples[i] * samples[i];
    }
    
    return energy / (endIdx - startIdx);
  }

  calculateLivenessConfidence(features) {
    let confidence = 0.5; // Base confidence
    
    if (features.hasNaturalVariation) confidence += 0.2;
    if (features.hasProperDynamics) confidence += 0.2;
    if (features.spectralVariation > 0.01) confidence += 0.1;
    
    return Math.max(0, Math.min(1, confidence));
  }

  calculateReplayConfidence(features) {
    let confidence = 0.5; // Base confidence
    
    if (features.hasCompressionArtifacts) confidence += 0.3;
    if (features.hasUniformBackground) confidence += 0.2;
    
    return Math.max(0, Math.min(1, confidence));
  }
}

export default BiometricAuth;