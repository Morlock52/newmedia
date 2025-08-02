import * as tf from '@tensorflow/tfjs-node-gpu';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as toxicity from '@tensorflow-models/toxicity';
import * as faceapi from 'face-api.js';
import ffmpeg from 'fluent-ffmpeg';
import sharp from 'sharp';
import { pipeline } from '@xenova/transformers';
import Redis from 'ioredis';
import pino from 'pino';

const logger = pino({ transport: { target: 'pino-pretty' } });
const redis = new Redis();

/**
 * Advanced Content Analysis Pipeline with Computer Vision
 * Implements scene detection, object recognition, face analysis, and content moderation
 */
export class ContentAnalysisPipeline {
  constructor() {
    this.models = {
      objectDetection: null,
      imageClassification: null,
      faceDetection: null,
      toxicityClassifier: null,
      sceneClassifier: null,
      audioAnalyzer: null
    };
    this.initialized = false;
  }

  async initialize() {
    logger.info('Initializing Content Analysis Pipeline...');
    
    // Load pre-trained models
    this.models.objectDetection = await cocoSsd.load();
    this.models.imageClassification = await mobilenet.load();
    this.models.toxicityClassifier = await toxicity.load();
    
    // Load face detection models
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models/face');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./models/face');
    await faceapi.nets.faceRecognitionNet.loadFromDisk('./models/face');
    await faceapi.nets.faceExpressionNet.loadFromDisk('./models/face');
    await faceapi.nets.ageGenderNet.loadFromDisk('./models/face');
    
    // Load transformer models
    this.models.sceneClassifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
    this.models.audioAnalyzer = await pipeline('audio-classification', 'Xenova/wav2vec2-base-960h');
    
    this.initialized = true;
    logger.info('Content Analysis Pipeline initialized successfully');
  }

  /**
   * Analyze video content with multiple AI models
   */
  async analyzeVideo(videoPath, options = {}) {
    const {
      sampleRate = 1, // Sample every N seconds
      analyzeFaces = true,
      detectObjects = true,
      classifyScenes = true,
      moderateContent = true,
      extractAudio = true
    } = options;
    
    const analysis = {
      metadata: await this.extractMetadata(videoPath),
      frames: [],
      audio: null,
      summary: null,
      tags: new Set(),
      warnings: []
    };
    
    try {
      // Extract key frames
      const frames = await this.extractKeyFrames(videoPath, sampleRate);
      
      // Analyze each frame
      for (const frame of frames) {
        const frameAnalysis = {
          timestamp: frame.timestamp,
          objects: detectObjects ? await this.detectObjects(frame.data) : [],
          faces: analyzeFaces ? await this.analyzeFaces(frame.data) : [],
          scene: classifyScenes ? await this.classifyScene(frame.data) : null,
          moderation: moderateContent ? await this.moderateImage(frame.data) : null
        };
        
        analysis.frames.push(frameAnalysis);
        
        // Aggregate tags
        frameAnalysis.objects.forEach(obj => analysis.tags.add(obj.class));
        if (frameAnalysis.scene) analysis.tags.add(frameAnalysis.scene.label);
      }
      
      // Analyze audio track
      if (extractAudio) {
        analysis.audio = await this.analyzeAudio(videoPath);
      }
      
      // Generate video summary
      analysis.summary = await this.generateVideoSummary(analysis);
      
      // Check for content warnings
      analysis.warnings = this.checkContentWarnings(analysis);
      
      // Cache results
      await redis.setex(
        `analysis:${videoPath}`,
        86400,
        JSON.stringify(analysis)
      );
      
      return analysis;
    } catch (error) {
      logger.error('Error analyzing video:', error);
      throw error;
    }
  }

  /**
   * Extract metadata from video file
   */
  async extractMetadata(videoPath) {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(videoPath, (err, metadata) => {
        if (err) reject(err);
        else resolve({
          duration: metadata.format.duration,
          bitrate: metadata.format.bit_rate,
          size: metadata.format.size,
          format: metadata.format.format_name,
          streams: metadata.streams.map(s => ({
            type: s.codec_type,
            codec: s.codec_name,
            width: s.width,
            height: s.height,
            fps: s.r_frame_rate
          }))
        });
      });
    });
  }

  /**
   * Extract key frames from video
   */
  async extractKeyFrames(videoPath, sampleRate) {
    const frames = [];
    const metadata = await this.extractMetadata(videoPath);
    const duration = metadata.duration;
    
    return new Promise((resolve, reject) => {
      const timestamps = [];
      for (let t = 0; t < duration; t += sampleRate) {
        timestamps.push(t);
      }
      
      let completed = 0;
      timestamps.forEach(timestamp => {
        ffmpeg(videoPath)
          .screenshots({
            timestamps: [timestamp],
            filename: `frame-${timestamp}.jpg`,
            folder: '/tmp/frames',
            size: '1280x720'
          })
          .on('end', async () => {
            const framePath = `/tmp/frames/frame-${timestamp}.jpg`;
            const frameData = await sharp(framePath).raw().toBuffer();
            
            frames.push({
              timestamp,
              path: framePath,
              data: frameData
            });
            
            completed++;
            if (completed === timestamps.length) {
              resolve(frames);
            }
          })
          .on('error', reject);
      });
    });
  }

  /**
   * Detect objects in image using COCO-SSD
   */
  async detectObjects(imageData) {
    const imageTensor = tf.node.decodeImage(imageData);
    const predictions = await this.models.objectDetection.detect(imageTensor);
    imageTensor.dispose();
    
    return predictions.map(pred => ({
      class: pred.class,
      score: pred.score,
      bbox: pred.bbox
    }));
  }

  /**
   * Analyze faces in image
   */
  async analyzeFaces(imageData) {
    const img = await sharp(imageData).jpeg().toBuffer();
    const detections = await faceapi
      .detectAllFaces(img)
      .withFaceLandmarks()
      .withFaceDescriptors()
      .withFaceExpressions()
      .withAgeAndGender();
    
    return detections.map(detection => ({
      box: detection.detection.box,
      confidence: detection.detection.score,
      landmarks: detection.landmarks.positions,
      expressions: detection.expressions,
      age: detection.age,
      gender: detection.gender,
      genderProbability: detection.genderProbability,
      descriptor: Array.from(detection.descriptor)
    }));
  }

  /**
   * Classify scene using Vision Transformer
   */
  async classifyScene(imageData) {
    const img = await sharp(imageData).resize(224, 224).jpeg().toBuffer();
    const results = await this.models.sceneClassifier(img);
    
    return results[0]; // Top classification
  }

  /**
   * Moderate image content for inappropriate material
   */
  async moderateImage(imageData) {
    // Convert image to base64 for text extraction if needed
    const base64 = imageData.toString('base64');
    
    // Use MobileNet for general content classification
    const imageTensor = tf.node.decodeImage(imageData);
    const predictions = await this.models.imageClassification.classify(imageTensor);
    imageTensor.dispose();
    
    // Check for potentially inappropriate content
    const inappropriate = predictions.some(pred => 
      this.isInappropriateContent(pred.className)
    );
    
    return {
      safe: !inappropriate,
      predictions,
      confidence: predictions[0]?.probability || 0
    };
  }

  /**
   * Analyze audio track
   */
  async analyzeAudio(videoPath) {
    // Extract audio
    const audioPath = `/tmp/audio-${Date.now()}.wav`;
    
    await new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .output(audioPath)
        .audioCodec('pcm_s16le')
        .audioFrequency(16000)
        .audioChannels(1)
        .on('end', resolve)
        .on('error', reject)
        .run();
    });
    
    // Analyze audio content
    const audioBuffer = await sharp(audioPath).toBuffer();
    const audioAnalysis = await this.models.audioAnalyzer(audioBuffer);
    
    // Speech-to-text would go here
    // const transcript = await this.transcribeAudio(audioPath);
    
    return {
      classification: audioAnalysis,
      // transcript,
      audioPath
    };
  }

  /**
   * Generate comprehensive video summary
   */
  async generateVideoSummary(analysis) {
    const summary = {
      duration: analysis.metadata.duration,
      format: analysis.metadata.format,
      resolution: `${analysis.metadata.streams[0]?.width}x${analysis.metadata.streams[0]?.height}`,
      
      // Aggregate object detections
      detectedObjects: this.aggregateDetections(
        analysis.frames.flatMap(f => f.objects)
      ),
      
      // Face statistics
      faceStats: this.calculateFaceStats(
        analysis.frames.flatMap(f => f.faces)
      ),
      
      // Scene classifications
      dominantScenes: this.findDominantScenes(
        analysis.frames.map(f => f.scene).filter(Boolean)
      ),
      
      // Content safety
      contentRating: this.calculateContentRating(analysis),
      
      // Key moments
      keyMoments: this.identifyKeyMoments(analysis.frames),
      
      // Auto-generated tags
      tags: Array.from(analysis.tags),
      
      // Emotion timeline
      emotionTimeline: this.buildEmotionTimeline(analysis.frames)
    };
    
    return summary;
  }

  /**
   * Aggregate object detections across frames
   */
  aggregateDetections(detections) {
    const counts = {};
    detections.forEach(det => {
      counts[det.class] = (counts[det.class] || 0) + 1;
    });
    
    return Object.entries(counts)
      .sort(([, a], [, b]) => b - a)
      .map(([object, count]) => ({ object, count }));
  }

  /**
   * Calculate face statistics
   */
  calculateFaceStats(faces) {
    if (faces.length === 0) return null;
    
    const stats = {
      totalFaces: faces.length,
      averageAge: faces.reduce((sum, f) => sum + f.age, 0) / faces.length,
      genderDistribution: this.calculateGenderDistribution(faces),
      dominantEmotions: this.calculateDominantEmotions(faces),
      uniqueFaces: this.estimateUniqueFaces(faces)
    };
    
    return stats;
  }

  /**
   * Calculate gender distribution
   */
  calculateGenderDistribution(faces) {
    const dist = { male: 0, female: 0 };
    faces.forEach(face => {
      if (face.genderProbability > 0.7) {
        dist[face.gender]++;
      }
    });
    return dist;
  }

  /**
   * Calculate dominant emotions
   */
  calculateDominantEmotions(faces) {
    const emotions = {};
    faces.forEach(face => {
      Object.entries(face.expressions).forEach(([emotion, score]) => {
        emotions[emotion] = (emotions[emotion] || 0) + score;
      });
    });
    
    return Object.entries(emotions)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([emotion, score]) => ({ emotion, score: score / faces.length }));
  }

  /**
   * Estimate unique faces using face descriptors
   */
  estimateUniqueFaces(faces) {
    const threshold = 0.6;
    const uniqueDescriptors = [];
    
    faces.forEach(face => {
      const descriptor = face.descriptor;
      const isUnique = !uniqueDescriptors.some(unique => 
        faceapi.euclideanDistance(descriptor, unique) < threshold
      );
      
      if (isUnique) {
        uniqueDescriptors.push(descriptor);
      }
    });
    
    return uniqueDescriptors.length;
  }

  /**
   * Find dominant scenes
   */
  findDominantScenes(scenes) {
    const sceneCounts = {};
    scenes.forEach(scene => {
      sceneCounts[scene.label] = (sceneCounts[scene.label] || 0) + 1;
    });
    
    return Object.entries(sceneCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([scene, count]) => ({ scene, frequency: count / scenes.length }));
  }

  /**
   * Calculate content rating
   */
  calculateContentRating(analysis) {
    const hasViolence = analysis.frames.some(f => 
      f.objects.some(o => ['weapon', 'knife', 'gun'].includes(o.class))
    );
    
    const hasInappropriate = analysis.frames.some(f => 
      !f.moderation?.safe
    );
    
    if (hasInappropriate) return 'R';
    if (hasViolence) return 'PG-13';
    return 'G';
  }

  /**
   * Identify key moments in video
   */
  identifyKeyMoments(frames) {
    const moments = [];
    
    frames.forEach((frame, idx) => {
      // High emotion moments
      const maxEmotion = frame.faces.reduce((max, face) => {
        const emotions = Object.values(face.expressions);
        const maxFaceEmotion = Math.max(...emotions);
        return Math.max(max, maxFaceEmotion);
      }, 0);
      
      if (maxEmotion > 0.8) {
        moments.push({
          timestamp: frame.timestamp,
          type: 'high-emotion',
          confidence: maxEmotion
        });
      }
      
      // Scene changes
      if (idx > 0) {
        const prevScene = frames[idx - 1].scene?.label;
        const currScene = frame.scene?.label;
        
        if (prevScene !== currScene && currScene) {
          moments.push({
            timestamp: frame.timestamp,
            type: 'scene-change',
            newScene: currScene
          });
        }
      }
    });
    
    return moments;
  }

  /**
   * Build emotion timeline
   */
  buildEmotionTimeline(frames) {
    return frames.map(frame => ({
      timestamp: frame.timestamp,
      emotions: frame.faces.length > 0
        ? this.averageEmotions(frame.faces)
        : null
    })).filter(item => item.emotions !== null);
  }

  /**
   * Average emotions across faces
   */
  averageEmotions(faces) {
    const emotions = {
      neutral: 0, happy: 0, sad: 0, angry: 0,
      fearful: 0, disgusted: 0, surprised: 0
    };
    
    faces.forEach(face => {
      Object.entries(face.expressions).forEach(([emotion, score]) => {
        emotions[emotion] += score;
      });
    });
    
    Object.keys(emotions).forEach(emotion => {
      emotions[emotion] /= faces.length;
    });
    
    return emotions;
  }

  /**
   * Check content warnings
   */
  checkContentWarnings(analysis) {
    const warnings = [];
    
    // Violence detection
    if (analysis.frames.some(f => f.objects.some(o => 
      ['weapon', 'knife', 'gun', 'blood'].includes(o.class)
    ))) {
      warnings.push({ type: 'violence', severity: 'medium' });
    }
    
    // Inappropriate content
    if (analysis.frames.some(f => !f.moderation?.safe)) {
      warnings.push({ type: 'inappropriate', severity: 'high' });
    }
    
    // Flashing lights (check rapid scene changes)
    const rapidChanges = this.detectRapidSceneChanges(analysis.frames);
    if (rapidChanges) {
      warnings.push({ type: 'flashing-lights', severity: 'low' });
    }
    
    return warnings;
  }

  /**
   * Detect rapid scene changes
   */
  detectRapidSceneChanges(frames) {
    let changeCount = 0;
    const windowSize = 5; // 5 second window
    
    for (let i = 1; i < frames.length; i++) {
      if (frames[i].scene?.label !== frames[i-1].scene?.label) {
        changeCount++;
      }
      
      if (i % windowSize === 0) {
        if (changeCount > windowSize * 0.6) {
          return true;
        }
        changeCount = 0;
      }
    }
    
    return false;
  }

  /**
   * Check if content is inappropriate
   */
  isInappropriateContent(className) {
    const inappropriate = [
      'bikini', 'lingerie', 'nude', 'explicit',
      'violence', 'gore', 'weapon'
    ];
    
    return inappropriate.some(term => 
      className.toLowerCase().includes(term)
    );
  }
}

// API Server
import Fastify from 'fastify';
import cors from '@fastify/cors';

const fastify = Fastify({ logger: true });
await fastify.register(cors);

const pipeline = new ContentAnalysisPipeline();

fastify.post('/analyze/video', async (request, reply) => {
  const { videoPath, options } = request.body;
  
  const analysis = await pipeline.analyzeVideo(videoPath, options);
  
  return { analysis };
});

fastify.post('/analyze/image', async (request, reply) => {
  const { imagePath } = request.body;
  
  const imageData = await sharp(imagePath).toBuffer();
  
  const analysis = {
    objects: await pipeline.detectObjects(imageData),
    faces: await pipeline.analyzeFaces(imageData),
    scene: await pipeline.classifyScene(imageData),
    moderation: await pipeline.moderateImage(imageData)
  };
  
  return { analysis };
});

// Initialize and start server
async function start() {
  await pipeline.initialize();
  await fastify.listen({ port: 8082, host: '0.0.0.0' });
  logger.info('Content Analysis Pipeline running on port 8082');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  start().catch(console.error);
}

export default pipeline;