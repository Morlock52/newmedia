import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { motion, AnimatePresence } from 'framer-motion';
import './NeuralRecommendations.css';

interface MediaItem {
  id: string;
  title: string;
  type: 'movie' | 'tv' | 'music';
  genre: string[];
  rating: number;
  year: number;
  duration: number;
  views: number;
  lastWatched?: Date;
  thumbnail?: string;
  tags: string[];
}

interface UserPreferences {
  preferredGenres: string[];
  viewingHistory: string[];
  ratings: { [mediaId: string]: number };
  watchTime: { [hour: number]: number };
  devicePreference: string;
  contentLength: 'short' | 'medium' | 'long';
}

interface Recommendation {
  item: MediaItem;
  score: number;
  reason: string;
  confidence: number;
}

const NeuralRecommendations: React.FC = () => {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [userPreferences, setUserPreferences] = useState<UserPreferences | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [selectedRecommendation, setSelectedRecommendation] = useState<Recommendation | null>(null);
  const [autoDownloadEnabled, setAutoDownloadEnabled] = useState(false);
  const [qualityPreference, setQualityPreference] = useState<'auto' | '4k' | '1080p' | '720p'>('auto');
  const [modelAccuracy, setModelAccuracy] = useState(0);
  const [isModelReady, setIsModelReady] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const neuralNetworkRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadOrCreateModel();
    fetchUserPreferences();
    visualizeNeuralNetwork();
  }, []);

  useEffect(() => {
    if (model && userPreferences) {
      generateRecommendations();
    }
  }, [model, userPreferences]);

  const loadOrCreateModel = async () => {
    try {
      // Try to load existing model
      const loadedModel = await tf.loadLayersModel('localstorage://media-recommendation-model');
      setModel(loadedModel);
      setIsModelReady(true);
      console.log('Model loaded from storage');
    } catch (error) {
      console.log('Creating new model');
      createAndTrainModel();
    }
  };

  const createAndTrainModel = async () => {
    setIsTraining(true);
    
    // Create neural network architecture
    const model = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [20], // Features: genre, rating, year, duration, etc.
          units: 64,
          activation: 'relu',
          kernelInitializer: 'glorotNormal'
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({
          units: 128,
          activation: 'relu',
          kernelInitializer: 'glorotNormal'
        }),
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({
          units: 64,
          activation: 'relu'
        }),
        tf.layers.dense({
          units: 32,
          activation: 'relu'
        }),
        tf.layers.dense({
          units: 1,
          activation: 'sigmoid' // Output: recommendation score 0-1
        })
      ]
    });

    // Compile model
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    // Generate training data
    const trainingData = generateTrainingData();
    
    // Train model with progress tracking
    await model.fit(trainingData.xs, trainingData.ys, {
      epochs: 50,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          setTrainingProgress((epoch + 1) / 50 * 100);
          if (logs?.acc) {
            setModelAccuracy(logs.acc * 100);
          }
          visualizeTrainingProgress(epoch, logs);
        }
      }
    });

    // Save model
    await model.save('localstorage://media-recommendation-model');
    
    setModel(model);
    setIsTraining(false);
    setIsModelReady(true);
    
    // Clean up tensors
    trainingData.xs.dispose();
    trainingData.ys.dispose();
  };

  const generateTrainingData = () => {
    // Simulate training data based on viewing patterns
    const numSamples = 1000;
    const features: number[][] = [];
    const labels: number[] = [];

    for (let i = 0; i < numSamples; i++) {
      // Random feature vector (normalized)
      const feature = [
        Math.random(), // Genre match score
        Math.random(), // Rating
        Math.random(), // Recency
        Math.random(), // Popularity
        Math.random(), // Duration preference
        Math.random(), // Time of day match
        Math.random(), // Sequel/series continuation
        Math.random(), // Actor/director match
        Math.random(), // Similar users liked
        Math.random(), // Trending score
        Math.random(), // Critical acclaim
        Math.random(), // User mood match
        Math.random(), // Season match
        Math.random(), // Language preference
        Math.random(), // Content maturity
        Math.random(), // Visual style match
        Math.random(), // Soundtrack match
        Math.random(), // Pace preference
        Math.random(), // Complexity preference
        Math.random()  // Nostalgia factor
      ];
      
      features.push(feature);
      
      // Simulate label (1 = user would like, 0 = user wouldn't like)
      const label = Math.random() > 0.5 ? 1 : 0;
      labels.push(label);
    }

    return {
      xs: tf.tensor2d(features),
      ys: tf.tensor2d(labels, [numSamples, 1])
    };
  };

  const fetchUserPreferences = async () => {
    try {
      const response = await fetch('/api/user/preferences');
      const data = await response.json();
      setUserPreferences(data);
    } catch (error) {
      // Use mock data
      setUserPreferences({
        preferredGenres: ['sci-fi', 'action', 'thriller'],
        viewingHistory: ['movie1', 'movie2', 'tv1'],
        ratings: { 'movie1': 5, 'movie2': 4, 'tv1': 5 },
        watchTime: { 20: 0.3, 21: 0.4, 22: 0.3 }, // Peak hours
        devicePreference: 'smart-tv',
        contentLength: 'medium'
      });
    }
  };

  const generateRecommendations = async () => {
    if (!model || !userPreferences) return;

    // Fetch available media
    const mediaLibrary = await fetchMediaLibrary();
    
    const recommendations: Recommendation[] = [];
    
    for (const media of mediaLibrary) {
      // Create feature vector for this media
      const features = extractFeatures(media, userPreferences);
      const featureTensor = tf.tensor2d([features]);
      
      // Get prediction
      const prediction = model.predict(featureTensor) as tf.Tensor;
      const score = (await prediction.data())[0];
      
      // Generate explanation
      const reason = generateReason(media, userPreferences, score);
      
      recommendations.push({
        item: media,
        score: score,
        reason: reason,
        confidence: calculateConfidence(media, userPreferences)
      });
      
      // Clean up
      featureTensor.dispose();
      prediction.dispose();
    }
    
    // Sort by score and take top recommendations
    const topRecommendations = recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, 20);
    
    setRecommendations(topRecommendations);
    
    // Auto-download if enabled
    if (autoDownloadEnabled) {
      autoDownloadTopPicks(topRecommendations);
    }
  };

  const extractFeatures = (media: MediaItem, preferences: UserPreferences): number[] => {
    // Extract normalized features for neural network input
    const genreMatch = media.genre.filter(g => 
      preferences.preferredGenres.includes(g)
    ).length / Math.max(media.genre.length, 1);
    
    const rating = media.rating / 10;
    const recency = (2025 - media.year) / 50; // Normalize by 50 years
    const popularity = Math.min(media.views / 1000000, 1); // Cap at 1M views
    
    const durationMatch = 
      (preferences.contentLength === 'short' && media.duration < 30) ? 1 :
      (preferences.contentLength === 'medium' && media.duration >= 30 && media.duration <= 120) ? 1 :
      (preferences.contentLength === 'long' && media.duration > 120) ? 1 : 0;
    
    // Add more features as needed
    return [
      genreMatch,
      rating,
      recency,
      popularity,
      durationMatch,
      Math.random(), // Time of day match (would be calculated based on current time)
      Math.random(), // Sequel/series continuation
      Math.random(), // Actor/director match
      Math.random(), // Similar users liked
      Math.random(), // Trending score
      Math.random(), // Critical acclaim
      Math.random(), // User mood match
      Math.random(), // Season match
      Math.random(), // Language preference
      Math.random(), // Content maturity
      Math.random(), // Visual style match
      Math.random(), // Soundtrack match
      Math.random(), // Pace preference
      Math.random(), // Complexity preference
      Math.random()  // Nostalgia factor
    ];
  };

  const generateReason = (media: MediaItem, preferences: UserPreferences, score: number): string => {
    const reasons = [];
    
    if (media.genre.some(g => preferences.preferredGenres.includes(g))) {
      reasons.push(`matches your love for ${media.genre[0]}`);
    }
    
    if (media.rating > 8) {
      reasons.push('highly rated by critics');
    }
    
    if (score > 0.8) {
      reasons.push('perfect match for your taste');
    } else if (score > 0.6) {
      reasons.push('similar to what you\'ve enjoyed');
    }
    
    if (media.year === 2025) {
      reasons.push('brand new release');
    }
    
    return reasons.join(', ') || 'recommended for you';
  };

  const calculateConfidence = (media: MediaItem, preferences: UserPreferences): number => {
    // Calculate confidence based on available data
    let confidence = 0.5;
    
    if (preferences.viewingHistory.length > 10) confidence += 0.1;
    if (Object.keys(preferences.ratings).length > 5) confidence += 0.1;
    if (media.views > 10000) confidence += 0.1;
    if (media.rating > 7) confidence += 0.1;
    if (media.genre.some(g => preferences.preferredGenres.includes(g))) confidence += 0.1;
    
    return Math.min(confidence, 1);
  };

  const fetchMediaLibrary = async (): Promise<MediaItem[]> => {
    // Simulate fetching media library
    return [
      {
        id: '1',
        title: 'Cyberpunk 2077: Phoenix Protocol',
        type: 'movie',
        genre: ['sci-fi', 'action'],
        rating: 8.5,
        year: 2025,
        duration: 148,
        views: 1500000,
        tags: ['dystopian', 'ai', 'neon']
      },
      {
        id: '2',
        title: 'Neural Interface',
        type: 'tv',
        genre: ['sci-fi', 'thriller'],
        rating: 9.0,
        year: 2024,
        duration: 45,
        views: 2000000,
        tags: ['mind-upload', 'conspiracy']
      },
      {
        id: '3',
        title: 'Quantum Dreams',
        type: 'movie',
        genre: ['sci-fi', 'mystery'],
        rating: 7.8,
        year: 2025,
        duration: 132,
        views: 800000,
        tags: ['parallel-universe', 'quantum']
      }
    ];
  };

  const autoDownloadTopPicks = async (recommendations: Recommendation[]) => {
    const topPicks = recommendations.slice(0, 3);
    
    for (const pick of topPicks) {
      if (pick.score > 0.8 && pick.confidence > 0.7) {
        await initiateDownload(pick.item);
      }
    }
  };

  const initiateDownload = async (media: MediaItem) => {
    try {
      await fetch('/api/downloads/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mediaId: media.id,
          quality: qualityPreference,
          priority: 'high'
        })
      });
    } catch (error) {
      console.error('Failed to initiate download:', error);
    }
  };

  const visualizeNeuralNetwork = () => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    
    const drawNetwork = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw neural network structure
      const layers = [20, 64, 128, 64, 32, 1];
      const layerSpacing = canvas.width / (layers.length + 1);
      
      for (let l = 0; l < layers.length; l++) {
        const x = layerSpacing * (l + 1);
        const nodeSpacing = canvas.height / (layers[l] + 1);
        
        for (let n = 0; n < layers[l]; n++) {
          const y = nodeSpacing * (n + 1);
          
          // Draw connections to next layer
          if (l < layers.length - 1) {
            const nextX = layerSpacing * (l + 2);
            const nextNodeSpacing = canvas.height / (layers[l + 1] + 1);
            
            for (let nn = 0; nn < layers[l + 1]; nn++) {
              const nextY = nextNodeSpacing * (nn + 1);
              
              ctx.strokeStyle = `rgba(0, 255, 255, ${Math.random() * 0.3 + 0.1})`;
              ctx.lineWidth = 0.5;
              ctx.beginPath();
              ctx.moveTo(x, y);
              ctx.lineTo(nextX, nextY);
              ctx.stroke();
            }
          }
          
          // Draw node
          const activation = Math.random();
          const radius = 3 + activation * 2;
          
          ctx.fillStyle = `rgba(${activation * 255}, ${255 - activation * 100}, 255, ${activation})`;
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          ctx.fill();
          
          // Add glow
          ctx.shadowBlur = 10;
          ctx.shadowColor = '#00ffff';
        }
      }
      
      requestAnimationFrame(drawNetwork);
    };
    
    drawNetwork();
  };

  const visualizeTrainingProgress = (epoch: number, logs: any) => {
    // Update training visualization
    console.log(`Epoch ${epoch}: Loss = ${logs?.loss}, Accuracy = ${logs?.acc}`);
  };

  const retrain = () => {
    createAndTrainModel();
  };

  return (
    <div className="neural-recommendations cyberpunk-theme">
      <div className="recommendations-header">
        <h1 className="title glitch-text" data-text="NEURAL RECOMMENDATIONS">
          NEURAL RECOMMENDATIONS
        </h1>
        
        <div className="model-status">
          <div className={`status-indicator ${isModelReady ? 'ready' : 'loading'}`}></div>
          <span>Model: {isModelReady ? 'Ready' : isTraining ? 'Training' : 'Loading'}</span>
          <span className="accuracy">Accuracy: {modelAccuracy.toFixed(1)}%</span>
        </div>
      </div>

      {/* Neural Network Visualization */}
      <div className="neural-visualization">
        <canvas
          ref={canvasRef}
          className="neural-canvas"
          width={800}
          height={300}
        />
        
        {isTraining && (
          <div className="training-overlay">
            <div className="training-progress">
              <div className="progress-label">Training Neural Network...</div>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${trainingProgress}%` }}
                />
              </div>
              <div className="progress-text">{trainingProgress.toFixed(0)}%</div>
            </div>
          </div>
        )}
      </div>

      {/* Settings Panel */}
      <div className="settings-panel">
        <div className="setting-item">
          <label className="cyberpunk-checkbox">
            <input
              type="checkbox"
              checked={autoDownloadEnabled}
              onChange={(e) => setAutoDownloadEnabled(e.target.checked)}
            />
            <span className="checkmark"></span>
            Auto-download top picks
          </label>
        </div>
        
        <div className="setting-item">
          <label>Quality Preference:</label>
          <select
            value={qualityPreference}
            onChange={(e) => setQualityPreference(e.target.value as any)}
            className="cyberpunk-select"
          >
            <option value="auto">Auto</option>
            <option value="4k">4K</option>
            <option value="1080p">1080p</option>
            <option value="720p">720p</option>
          </select>
        </div>
        
        <button className="retrain-button" onClick={retrain}>
          🔄 Retrain Model
        </button>
      </div>

      {/* Recommendations Grid */}
      <div className="recommendations-grid">
        <AnimatePresence>
          {recommendations.map((rec, index) => (
            <motion.div
              key={rec.item.id}
              className="recommendation-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => setSelectedRecommendation(rec)}
              style={{
                borderColor: `rgba(0, 255, 255, ${rec.score})`,
                boxShadow: `0 0 ${20 * rec.score}px rgba(0, 255, 255, ${rec.score * 0.5})`
              }}
            >
              <div className="card-header">
                <div className="confidence-meter">
                  <div 
                    className="confidence-fill"
                    style={{ 
                      height: `${rec.confidence * 100}%`,
                      background: `linear-gradient(to top, #00ffff, #ff00ff)`
                    }}
                  />
                </div>
                
                <div className="card-info">
                  <h3 className="media-title">{rec.item.title}</h3>
                  <div className="media-meta">
                    <span className="media-type">{rec.item.type}</span>
                    <span className="media-year">{rec.item.year}</span>
                    <span className="media-rating">⭐ {rec.item.rating}</span>
                  </div>
                  <div className="media-genres">
                    {rec.item.genre.map(g => (
                      <span key={g} className="genre-tag">{g}</span>
                    ))}
                  </div>
                </div>
                
                <div className="score-display">
                  <div className="score-value">{(rec.score * 100).toFixed(0)}%</div>
                  <div className="score-label">Match</div>
                </div>
              </div>
              
              <div className="recommendation-reason">
                <span className="reason-icon">💡</span>
                {rec.reason}
              </div>
              
              <div className="card-actions">
                <button className="action-btn play-btn">▶ Play</button>
                <button className="action-btn download-btn">⬇ Download</button>
                <button className="action-btn info-btn">ℹ Info</button>
              </div>
              
              <div className="neural-activity">
                {[...Array(5)].map((_, i) => (
                  <div 
                    key={i} 
                    className="activity-bar"
                    style={{
                      animationDelay: `${i * 0.1}s`,
                      height: `${20 + Math.random() * 30}px`
                    }}
                  />
                ))}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Selected Recommendation Detail */}
      <AnimatePresence>
        {selectedRecommendation && (
          <motion.div
            className="recommendation-detail"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <div className="detail-content">
              <button 
                className="close-button"
                onClick={() => setSelectedRecommendation(null)}
              >
                ×
              </button>
              
              <h2>{selectedRecommendation.item.title}</h2>
              
              <div className="detail-stats">
                <div className="stat-item">
                  <span className="stat-label">Neural Score:</span>
                  <span className="stat-value">
                    {(selectedRecommendation.score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Confidence:</span>
                  <span className="stat-value">
                    {(selectedRecommendation.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="stat-item">
                  <span className="stat-label">Views:</span>
                  <span className="stat-value">
                    {selectedRecommendation.item.views.toLocaleString()}
                  </span>
                </div>
              </div>
              
              <div className="detail-tags">
                {selectedRecommendation.item.tags.map(tag => (
                  <span key={tag} className="tag">{tag}</span>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Cyberpunk decorations */}
      <div className="cyberpunk-decorations">
        <div className="grid-overlay"></div>
        <div className="scan-lines"></div>
      </div>
    </div>
  );
};

export default NeuralRecommendations;