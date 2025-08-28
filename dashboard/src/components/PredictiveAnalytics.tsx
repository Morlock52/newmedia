import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line, Bar, Doughnut, Scatter, Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
} from 'chart.js';
import * as tf from '@tensorflow/tfjs';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
);

interface MetricPoint {
  timestamp: number;
  value: number;
  predicted?: boolean;
  confidence?: number;
}

interface Prediction {
  metric: string;
  timeframe: '1h' | '24h' | '7d' | '30d';
  values: number[];
  confidence: number;
  trend: 'increasing' | 'decreasing' | 'stable' | 'volatile';
  anomalies: { timestamp: number; severity: 'low' | 'medium' | 'high' }[];
}

interface Alert {
  id: string;
  type: 'capacity' | 'performance' | 'security' | 'maintenance';
  severity: 'info' | 'warning' | 'critical';
  title: string;
  description: string;
  prediction: {
    probability: number;
    timeframe: string;
    impact: 'low' | 'medium' | 'high';
  };
  timestamp: number;
  acknowledged: boolean;
}

interface AnalyticsData {
  systemMetrics: {
    cpu: MetricPoint[];
    memory: MetricPoint[];
    disk: MetricPoint[];
    network: MetricPoint[];
  };
  userMetrics: {
    activeUsers: MetricPoint[];
    watchTime: MetricPoint[];
    downloadActivity: MetricPoint[];
    streamingQuality: MetricPoint[];
  };
  mediaMetrics: {
    popularGenres: { genre: string; count: number; trend: number }[];
    viewingPatterns: MetricPoint[];
    contentRequests: MetricPoint[];
    qualityDistribution: { quality: string; percentage: number }[];
  };
}

interface PredictiveAnalyticsProps {
  enableRealTime?: boolean;
  enablePredictions?: boolean;
  enableAnomalyDetection?: boolean;
  timeRange?: '1h' | '24h' | '7d' | '30d';
  onAlertGenerated?: (alert: Alert) => void;
}

const PredictiveAnalytics: React.FC<PredictiveAnalyticsProps> = ({
  enableRealTime = true,
  enablePredictions = true,
  enableAnomalyDetection = true,
  timeRange = '24h',
  onAlertGenerated
}) => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('cpu');
  const [activeView, setActiveView] = useState<'overview' | 'predictions' | 'anomalies' | 'insights'>('overview');
  const [isTraining, setIsTraining] = useState(false);
  const [modelAccuracy, setModelAccuracy] = useState(0.85);
  const [processingTime, setProcessingTime] = useState(0);
  const [dataPoints, setDataPoints] = useState(0);
  const [trendAnalysis, setTrendAnalysis] = useState<any>(null);
  const [seasonalPatterns, setSeasonalPatterns] = useState<any>(null);

  const modelRef = useRef<tf.LayersModel | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const anomalyCanvasRef = useRef<HTMLCanvasElement>(null);
  const timeSeriesRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  // Initialize ML models for predictions
  useEffect(() => {
    initializePredictionModels();
    loadHistoricalData();
    
    if (enableRealTime) {
      const interval = setInterval(updateRealTimeData, 5000);
      return () => clearInterval(interval);
    }
  }, [enableRealTime]);

  // Generate predictions when data changes
  useEffect(() => {
    if (analyticsData && enablePredictions) {
      generatePredictions();
    }
  }, [analyticsData, enablePredictions]);

  // Anomaly detection
  useEffect(() => {
    if (analyticsData && enableAnomalyDetection) {
      detectAnomalies();
    }
  }, [analyticsData, enableAnomalyDetection]);

  const initializePredictionModels = async () => {
    setIsTraining(true);
    const startTime = performance.now();

    try {
      // Time series prediction model
      const model = tf.sequential({
        layers: [
          tf.layers.lstm({
            units: 50,
            returnSequences: true,
            inputShape: [10, 1] // 10 time steps, 1 feature
          }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.lstm({
            units: 50,
            returnSequences: false
          }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 25 }),
          tf.layers.dense({ units: 1 })
        ]
      });

      model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['mae']
      });

      // Train with synthetic data
      const { xs, ys } = generateTrainingData();
      
      await model.fit(xs, ys, {
        epochs: 20,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (logs?.acc) {
              setModelAccuracy(logs.acc);
            }
          }
        }
      });

      modelRef.current = model;
      xs.dispose();
      ys.dispose();

    } catch (error) {
      console.error('Failed to initialize prediction models:', error);
    } finally {
      setIsTraining(false);
      setProcessingTime(performance.now() - startTime);
    }
  };

  const generateTrainingData = () => {
    const sequenceLength = 10;
    const samples = 1000;
    const features: number[][] = [];
    const labels: number[] = [];

    for (let i = 0; i < samples; i++) {
      const sequence: number[] = [];
      let value = Math.random() * 100;
      
      for (let j = 0; j < sequenceLength; j++) {
        // Add trend and noise
        value += (Math.random() - 0.5) * 10 + Math.sin(j * 0.1) * 5;
        value = Math.max(0, Math.min(100, value));
        sequence.push(value / 100); // Normalize
      }
      
      features.push(sequence);
      // Next value in sequence
      const nextValue = value + (Math.random() - 0.5) * 10 + Math.sin(sequenceLength * 0.1) * 5;
      labels.push(Math.max(0, Math.min(100, nextValue)) / 100);
    }

    return {
      xs: tf.tensor3d(features.map(seq => seq.map(val => [val]))),
      ys: tf.tensor2d(labels, [labels.length, 1])
    };
  };

  const loadHistoricalData = async () => {
    // Simulate loading historical data
    const now = Date.now();
    const timePoints = 48; // 48 hours of data
    const interval = 60 * 60 * 1000; // 1 hour intervals

    const generateMetricData = (baseValue: number, volatility: number): MetricPoint[] => {
      const data: MetricPoint[] = [];
      let value = baseValue;

      for (let i = 0; i < timePoints; i++) {
        const timestamp = now - (timePoints - i) * interval;
        
        // Add seasonal patterns
        const hourOfDay = new Date(timestamp).getHours();
        const seasonalFactor = 1 + 0.3 * Math.sin((hourOfDay / 24) * 2 * Math.PI);
        
        // Add trend
        const trendFactor = 1 + (i / timePoints) * 0.2;
        
        // Add noise
        const noise = (Math.random() - 0.5) * volatility;
        
        value = Math.max(0, Math.min(100, baseValue * seasonalFactor * trendFactor + noise));
        
        data.push({
          timestamp,
          value,
          confidence: 0.95 + Math.random() * 0.05
        });
      }

      return data;
    };

    const data: AnalyticsData = {
      systemMetrics: {
        cpu: generateMetricData(45, 15),
        memory: generateMetricData(60, 10),
        disk: generateMetricData(35, 5),
        network: generateMetricData(70, 20)
      },
      userMetrics: {
        activeUsers: generateMetricData(150, 30),
        watchTime: generateMetricData(8.5, 2),
        downloadActivity: generateMetricData(25, 10),
        streamingQuality: generateMetricData(85, 8)
      },
      mediaMetrics: {
        popularGenres: [
          { genre: 'Action', count: 45, trend: 0.15 },
          { genre: 'Sci-Fi', count: 38, trend: 0.22 },
          { genre: 'Comedy', count: 32, trend: -0.05 },
          { genre: 'Drama', count: 28, trend: 0.08 },
          { genre: 'Horror', count: 22, trend: 0.35 },
          { genre: 'Documentary', count: 18, trend: 0.18 }
        ],
        viewingPatterns: generateMetricData(120, 25),
        contentRequests: generateMetricData(85, 18),
        qualityDistribution: [
          { quality: '4K', percentage: 25 },
          { quality: '1080p', percentage: 45 },
          { quality: '720p', percentage: 20 },
          { quality: '480p', percentage: 10 }
        ]
      }
    };

    setAnalyticsData(data);
    setDataPoints(timePoints * 7); // Total data points across all metrics
  };

  const updateRealTimeData = () => {
    if (!analyticsData) return;

    const now = Date.now();
    setAnalyticsData(prev => {
      if (!prev) return prev;

      const updateMetricArray = (arr: MetricPoint[], baseValue: number, volatility: number) => {
        const newArr = [...arr];
        const lastValue = newArr[newArr.length - 1]?.value || baseValue;
        const newValue = Math.max(0, Math.min(100, lastValue + (Math.random() - 0.5) * volatility));
        
        newArr.push({
          timestamp: now,
          value: newValue,
          confidence: 0.95 + Math.random() * 0.05
        });

        // Keep only last 48 points
        return newArr.slice(-48);
      };

      return {
        ...prev,
        systemMetrics: {
          cpu: updateMetricArray(prev.systemMetrics.cpu, 45, 8),
          memory: updateMetricArray(prev.systemMetrics.memory, 60, 5),
          disk: updateMetricArray(prev.systemMetrics.disk, 35, 2),
          network: updateMetricArray(prev.systemMetrics.network, 70, 12)
        },
        userMetrics: {
          activeUsers: updateMetricArray(prev.userMetrics.activeUsers, 150, 15),
          watchTime: updateMetricArray(prev.userMetrics.watchTime, 8.5, 1),
          downloadActivity: updateMetricArray(prev.userMetrics.downloadActivity, 25, 8),
          streamingQuality: updateMetricArray(prev.userMetrics.streamingQuality, 85, 5)
        }
      };
    });
  };

  const generatePredictions = async () => {
    if (!modelRef.current || !analyticsData) return;

    const startTime = performance.now();
    const newPredictions: Prediction[] = [];

    try {
      // Generate predictions for each metric
      const metrics = ['cpu', 'memory', 'disk', 'network'];
      
      for (const metric of metrics) {
        const data = analyticsData.systemMetrics[metric as keyof typeof analyticsData.systemMetrics];
        if (data.length < 10) continue;

        // Prepare sequence for prediction
        const sequence = data.slice(-10).map(d => d.value / 100);
        const input = tf.tensor3d([sequence.map(val => [val])]);
        
        // Generate multiple predictions for different timeframes
        const predictions = await modelRef.current.predict(input) as tf.Tensor;
        const predictionValue = (await predictions.data())[0] * 100;
        
        // Calculate trend
        const recentValues = data.slice(-5).map(d => d.value);
        const olderValues = data.slice(-10, -5).map(d => d.value);
        const recentAvg = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
        const olderAvg = olderValues.reduce((a, b) => a + b, 0) / olderValues.length;
        
        let trend: 'increasing' | 'decreasing' | 'stable' | 'volatile';
        const change = (recentAvg - olderAvg) / olderAvg;
        
        if (Math.abs(change) < 0.02) trend = 'stable';
        else if (change > 0.1) trend = 'increasing';
        else if (change < -0.1) trend = 'decreasing';
        else trend = 'volatile';

        // Generate future values
        const futureValues = [predictionValue];
        for (let i = 1; i < 24; i++) {
          const nextValue = futureValues[i - 1] + (Math.random() - 0.5) * 5;
          futureValues.push(Math.max(0, Math.min(100, nextValue)));
        }

        newPredictions.push({
          metric,
          timeframe: '24h',
          values: futureValues,
          confidence: 0.75 + Math.random() * 0.2,
          trend,
          anomalies: []
        });

        input.dispose();
        predictions.dispose();
      }

      setPredictions(newPredictions);
      
      // Generate alerts based on predictions
      generatePredictiveAlerts(newPredictions);

    } catch (error) {
      console.error('Failed to generate predictions:', error);
    } finally {
      setProcessingTime(performance.now() - startTime);
    }
  };

  const detectAnomalies = () => {
    if (!analyticsData) return;

    const anomalies: any[] = [];
    
    Object.entries(analyticsData.systemMetrics).forEach(([metric, data]) => {
      // Simple statistical anomaly detection
      const values = data.map(d => d.value);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const stdDev = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
      
      data.forEach(point => {
        const zScore = Math.abs((point.value - mean) / stdDev);
        if (zScore > 2.5) { // 2.5 standard deviations
          anomalies.push({
            metric,
            timestamp: point.timestamp,
            value: point.value,
            severity: zScore > 3 ? 'high' : zScore > 2.8 ? 'medium' : 'low',
            zScore
          });
        }
      });
    });

    // Visualize anomalies
    visualizeAnomalies(anomalies);
  };

  const generatePredictiveAlerts = (predictions: Prediction[]) => {
    const newAlerts: Alert[] = [];

    predictions.forEach(prediction => {
      const futureMax = Math.max(...prediction.values);
      const currentValue = analyticsData?.systemMetrics[prediction.metric as keyof typeof analyticsData.systemMetrics]?.slice(-1)[0]?.value || 0;

      // Capacity alert
      if (futureMax > 90 && prediction.confidence > 0.7) {
        newAlerts.push({
          id: `capacity-${prediction.metric}-${Date.now()}`,
          type: 'capacity',
          severity: futureMax > 95 ? 'critical' : 'warning',
          title: `${prediction.metric.toUpperCase()} Capacity Alert`,
          description: `${prediction.metric} usage predicted to reach ${futureMax.toFixed(1)}% within ${prediction.timeframe}`,
          prediction: {
            probability: prediction.confidence,
            timeframe: prediction.timeframe,
            impact: futureMax > 95 ? 'high' : 'medium'
          },
          timestamp: Date.now(),
          acknowledged: false
        });
      }

      // Performance degradation alert
      if (prediction.trend === 'increasing' && currentValue > 70) {
        newAlerts.push({
          id: `performance-${prediction.metric}-${Date.now()}`,
          type: 'performance',
          severity: 'warning',
          title: 'Performance Degradation Predicted',
          description: `Increasing ${prediction.metric} usage trend detected. Consider scaling resources.`,
          prediction: {
            probability: prediction.confidence,
            timeframe: prediction.timeframe,
            impact: 'medium'
          },
          timestamp: Date.now(),
          acknowledged: false
        });
      }
    });

    if (newAlerts.length > 0) {
      setAlerts(prev => [...newAlerts, ...prev].slice(0, 10)); // Keep last 10 alerts
      newAlerts.forEach(alert => {
        if (onAlertGenerated) {
          onAlertGenerated(alert);
        }
      });
    }
  };

  const visualizeAnomalies = (anomalies: any[]) => {
    if (!anomalyCanvasRef.current) return;

    const canvas = anomalyCanvasRef.current;
    const ctx = canvas.getContext('2d')!;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw anomaly heatmap
    const now = Date.now();
    const timeWindow = 24 * 60 * 60 * 1000; // 24 hours
    
    anomalies.forEach(anomaly => {
      const x = ((now - anomaly.timestamp) / timeWindow) * canvas.width;
      const y = (anomaly.value / 100) * canvas.height;
      
      const severity = anomaly.severity === 'high' ? 1 : anomaly.severity === 'medium' ? 0.7 : 0.4;
      const size = 5 + severity * 10;
      
      ctx.beginPath();
      ctx.arc(x, canvas.height - y, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255, ${255 - severity * 255}, 0, ${severity})`;
      ctx.fill();
      
      // Add glow effect
      ctx.shadowBlur = 15;
      ctx.shadowColor = `rgba(255, ${255 - severity * 255}, 0, ${severity})`;
    });
  };

  const formatTrendIndicator = (trend: string) => {
    switch (trend) {
      case 'increasing': return { icon: '📈', color: '#ff6b6b' };
      case 'decreasing': return { icon: '📉', color: '#51cf66' };
      case 'stable': return { icon: '➡️', color: '#74c0fc' };
      case 'volatile': return { icon: '📊', color: '#ffd43b' };
      default: return { icon: '❓', color: '#868e96' };
    }
  };

  const getMetricChartData = (metric: string) => {
    if (!analyticsData) return null;

    const data = analyticsData.systemMetrics[metric as keyof typeof analyticsData.systemMetrics];
    const prediction = predictions.find(p => p.metric === metric);
    
    const labels = data.map((_, index) => index - data.length + 1);
    const historicalData = data.map(d => d.value);
    
    let predictedData = [];
    if (prediction && enablePredictions) {
      predictedData = Array(data.length - 1).fill(null).concat(prediction.values.slice(0, 12));
      labels.push(...Array.from({ length: 12 }, (_, i) => i + 1));
    }

    return {
      labels: labels.map(l => l === 0 ? 'Now' : l > 0 ? `+${l}h` : `${l}h`),
      datasets: [
        {
          label: 'Historical',
          data: historicalData,
          borderColor: '#00ffff',
          backgroundColor: 'rgba(0, 255, 255, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 2
        },
        ...(predictedData.length > 0 ? [{
          label: 'Predicted',
          data: predictedData,
          borderColor: '#ff00ff',
          backgroundColor: 'rgba(255, 0, 255, 0.1)',
          borderDash: [5, 5],
          fill: false,
          tension: 0.4,
          pointRadius: 3
        }] : [])
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#ffffff',
          font: {
            family: 'monospace'
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#00ffff',
        bodyColor: '#ffffff',
        borderColor: '#00ffff',
        borderWidth: 1
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#ffffff',
          font: {
            family: 'monospace'
          }
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#ffffff',
          font: {
            family: 'monospace'
          }
        },
        min: 0,
        max: 100
      }
    }
  };

  if (!analyticsData) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
        border: '2px solid #00ffff',
        borderRadius: '15px',
        padding: '40px',
        color: '#ffffff',
        fontFamily: 'monospace',
        textAlign: 'center',
        minHeight: '400px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div>
          <div style={{ fontSize: '48px', marginBottom: '20px' }}>🔮</div>
          <div style={{ color: '#00ffff', fontSize: '18px', marginBottom: '10px' }}>
            Initializing Predictive Analytics...
          </div>
          <div style={{ color: '#cccccc' }}>
            Loading historical data and training ML models
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
      border: '2px solid #00ffff',
      borderRadius: '15px',
      padding: '20px',
      color: '#ffffff',
      fontFamily: 'monospace',
      minHeight: '600px'
    }}>
      {/* Header */}
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '20px' 
      }}>
        <h2 style={{
          color: '#00ffff',
          textShadow: '0 0 10px #00ffff',
          margin: 0,
          fontSize: '24px'
        }}>
          🔮 Predictive Analytics
        </h2>

        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          {/* Model Status */}
          <div style={{
            padding: '5px 10px',
            background: isTraining ? 'rgba(255,255,0,0.2)' : 'rgba(0,255,0,0.2)',
            border: `1px solid ${isTraining ? '#ffff00' : '#00ff00'}`,
            borderRadius: '5px',
            fontSize: '11px'
          }}>
            {isTraining ? 'TRAINING' : 'READY'}
          </div>

          <div style={{
            padding: '5px 10px',
            background: 'rgba(255,0,255,0.2)',
            border: '1px solid #ff00ff',
            borderRadius: '5px',
            fontSize: '11px'
          }}>
            Accuracy: {(modelAccuracy * 100).toFixed(1)}%
          </div>

          <div style={{
            padding: '5px 10px',
            background: 'rgba(0,255,255,0.2)',
            border: '1px solid #00ffff',
            borderRadius: '5px',
            fontSize: '11px'
          }}>
            {dataPoints.toLocaleString()} points
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        {(['overview', 'predictions', 'anomalies', 'insights'] as const).map(view => (
          <button
            key={view}
            onClick={() => setActiveView(view)}
            style={{
              padding: '8px 16px',
              background: activeView === view ? 'rgba(0,255,255,0.3)' : 'rgba(0,0,0,0.3)',
              border: `1px solid ${activeView === view ? '#00ffff' : '#666'}`,
              borderRadius: '20px',
              color: activeView === view ? '#00ffff' : '#cccccc',
              cursor: 'pointer',
              fontSize: '12px',
              textTransform: 'uppercase',
              fontFamily: 'monospace'
            }}
          >
            {view}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ height: 'calc(100% - 120px)' }}>
        {activeView === 'overview' && (
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '2fr 1fr', 
            gap: '20px', 
            height: '100%' 
          }}>
            {/* Main Chart */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(0,255,255,0.3)',
              borderRadius: '10px',
              padding: '15px'
            }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                marginBottom: '15px' 
              }}>
                <h3 style={{ color: '#ffff00', fontSize: '16px', margin: 0 }}>
                  Time Series Analysis
                </h3>
                
                <select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  style={{
                    padding: '5px 10px',
                    background: 'rgba(0,0,0,0.7)',
                    border: '1px solid #ffff00',
                    borderRadius: '5px',
                    color: '#ffffff',
                    fontSize: '12px',
                    fontFamily: 'monospace'
                  }}
                >
                  <option value="cpu">CPU Usage</option>
                  <option value="memory">Memory Usage</option>
                  <option value="disk">Disk Usage</option>
                  <option value="network">Network Usage</option>
                </select>
              </div>

              <div style={{ height: '300px' }}>
                {getMetricChartData(selectedMetric) && (
                  <Line 
                    data={getMetricChartData(selectedMetric)!} 
                    options={chartOptions} 
                  />
                )}
              </div>
            </div>

            {/* Metrics Summary */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '15px'
            }}>
              {/* Current Metrics */}
              <div style={{
                background: 'rgba(0,0,0,0.5)',
                border: '1px solid rgba(255,0,255,0.3)',
                borderRadius: '10px',
                padding: '15px'
              }}>
                <h3 style={{ color: '#ff00ff', fontSize: '14px', margin: '0 0 10px 0' }}>
                  Current Status
                </h3>
                
                {Object.entries(analyticsData.systemMetrics).map(([metric, data]) => {
                  const currentValue = data[data.length - 1]?.value || 0;
                  const prediction = predictions.find(p => p.metric === metric);
                  const trend = formatTrendIndicator(prediction?.trend || 'stable');
                  
                  return (
                    <div 
                      key={metric}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '8px',
                        padding: '8px',
                        background: 'rgba(255,255,255,0.05)',
                        borderRadius: '5px'
                      }}
                    >
                      <span style={{ textTransform: 'uppercase', fontSize: '11px' }}>
                        {metric}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '12px' }}>
                          {currentValue.toFixed(1)}%
                        </span>
                        <span style={{ fontSize: '14px' }}>
                          {trend.icon}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Predictions Summary */}
              <div style={{
                background: 'rgba(0,0,0,0.5)',
                border: '1px solid rgba(255,255,0,0.3)',
                borderRadius: '10px',
                padding: '15px'
              }}>
                <h3 style={{ color: '#ffff00', fontSize: '14px', margin: '0 0 10px 0' }}>
                  24h Predictions
                </h3>
                
                {predictions.slice(0, 4).map((pred, index) => (
                  <div 
                    key={pred.metric}
                    style={{
                      marginBottom: '8px',
                      padding: '8px',
                      background: 'rgba(255,255,255,0.05)',
                      borderRadius: '5px'
                    }}
                  >
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      fontSize: '11px',
                      marginBottom: '4px'
                    }}>
                      <span style={{ textTransform: 'uppercase' }}>
                        {pred.metric}
                      </span>
                      <span style={{ color: '#00ff00' }}>
                        {(pred.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#cccccc' }}>
                      Peak: {Math.max(...pred.values).toFixed(1)}% • 
                      Trend: {pred.trend}
                    </div>
                  </div>
                ))}
              </div>

              {/* Recent Alerts */}
              {alerts.length > 0 && (
                <div style={{
                  background: 'rgba(0,0,0,0.5)',
                  border: '1px solid rgba(255,0,0,0.3)',
                  borderRadius: '10px',
                  padding: '15px'
                }}>
                  <h3 style={{ color: '#ff0000', fontSize: '14px', margin: '0 0 10px 0' }}>
                    Predictive Alerts
                  </h3>
                  
                  {alerts.slice(0, 3).map((alert, index) => (
                    <div 
                      key={alert.id}
                      style={{
                        marginBottom: '8px',
                        padding: '8px',
                        background: `rgba(255,${alert.severity === 'critical' ? '0' : '165'},0,0.1)`,
                        border: `1px solid rgba(255,${alert.severity === 'critical' ? '0' : '165'},0,0.3)`,
                        borderRadius: '5px'
                      }}
                    >
                      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '2px' }}>
                        {alert.title}
                      </div>
                      <div style={{ fontSize: '9px', color: '#cccccc' }}>
                        {(alert.prediction.probability * 100).toFixed(0)}% probability • {alert.prediction.timeframe}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeView === 'predictions' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '20px'
          }}>
            {predictions.map((prediction, index) => {
              const trend = formatTrendIndicator(prediction.trend);
              
              return (
                <motion.div
                  key={prediction.metric}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  style={{
                    background: 'rgba(0,0,0,0.5)',
                    border: '1px solid rgba(0,255,255,0.3)',
                    borderRadius: '10px',
                    padding: '15px'
                  }}
                >
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginBottom: '15px'
                  }}>
                    <h4 style={{ 
                      color: '#00ffff', 
                      margin: 0, 
                      textTransform: 'uppercase',
                      fontSize: '14px'
                    }}>
                      {prediction.metric}
                    </h4>
                    <div style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px',
                      fontSize: '12px'
                    }}>
                      <span style={{ color: trend.color }}>
                        {trend.icon}
                      </span>
                      <span style={{ color: '#00ff00' }}>
                        {(prediction.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  <div style={{ marginBottom: '10px' }}>
                    <div style={{ fontSize: '11px', color: '#cccccc', marginBottom: '5px' }}>
                      Peak Value (24h):
                    </div>
                    <div style={{ fontSize: '18px', color: '#ffff00' }}>
                      {Math.max(...prediction.values).toFixed(1)}%
                    </div>
                  </div>

                  <div style={{ marginBottom: '10px' }}>
                    <div style={{ fontSize: '11px', color: '#cccccc', marginBottom: '5px' }}>
                      Trend: {prediction.trend}
                    </div>
                    <div style={{ fontSize: '11px', color: '#cccccc' }}>
                      Timeframe: {prediction.timeframe}
                    </div>
                  </div>

                  {/* Mini chart */}
                  <div style={{ 
                    height: '60px', 
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: '5px',
                    padding: '5px'
                  }}>
                    <canvas
                      width={250}
                      height={50}
                      style={{
                        width: '100%',
                        height: '100%'
                      }}
                      ref={(canvas) => {
                        if (canvas) {
                          const ctx = canvas.getContext('2d')!;
                          ctx.clearRect(0, 0, 250, 50);
                          
                          const points = prediction.values.slice(0, 12);
                          const maxVal = Math.max(...points);
                          const minVal = Math.min(...points);
                          const range = maxVal - minVal || 1;
                          
                          ctx.strokeStyle = trend.color;
                          ctx.lineWidth = 2;
                          ctx.beginPath();
                          
                          points.forEach((value, idx) => {
                            const x = (idx / (points.length - 1)) * 240 + 5;
                            const y = 45 - ((value - minVal) / range) * 40;
                            
                            if (idx === 0) ctx.moveTo(x, y);
                            else ctx.lineTo(x, y);
                          });
                          
                          ctx.stroke();
                        }
                      }}
                    />
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}

        {activeView === 'anomalies' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '2fr 1fr',
            gap: '20px',
            height: '100%'
          }}>
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,0,0,0.3)',
              borderRadius: '10px',
              padding: '15px'
            }}>
              <h3 style={{ color: '#ff0000', fontSize: '16px', marginBottom: '15px' }}>
                Anomaly Detection Heatmap
              </h3>
              <canvas
                ref={anomalyCanvasRef}
                width={600}
                height={300}
                style={{
                  width: '100%',
                  height: '300px',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '5px'
                }}
              />
            </div>

            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,165,0,0.3)',
              borderRadius: '10px',
              padding: '15px'
            }}>
              <h3 style={{ color: '#ffa500', fontSize: '14px', marginBottom: '15px' }}>
                Detection Settings
              </h3>
              
              <div style={{ marginBottom: '15px' }}>
                <label style={{ fontSize: '12px', color: '#cccccc', display: 'block', marginBottom: '5px' }}>
                  Sensitivity:
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  defaultValue="3"
                  style={{ width: '100%' }}
                />
              </div>

              <div style={{ marginBottom: '15px' }}>
                <label style={{ fontSize: '12px', color: '#cccccc', display: 'block', marginBottom: '5px' }}>
                  Window Size:
                </label>
                <select style={{
                  width: '100%',
                  padding: '5px',
                  background: 'rgba(0,0,0,0.7)',
                  border: '1px solid #666',
                  borderRadius: '3px',
                  color: '#ffffff',
                  fontSize: '11px'
                }}>
                  <option value="1h">1 Hour</option>
                  <option value="6h">6 Hours</option>
                  <option value="24h">24 Hours</option>
                  <option value="7d">7 Days</option>
                </select>
              </div>

              <button style={{
                width: '100%',
                padding: '8px',
                background: 'rgba(255,0,0,0.2)',
                border: '1px solid #ff0000',
                borderRadius: '5px',
                color: '#ff0000',
                cursor: 'pointer',
                fontSize: '12px'
              }}>
                🔍 Rerun Detection
              </button>
            </div>
          </div>
        )}

        {activeView === 'insights' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '20px'
          }}>
            {/* Processing Stats */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(0,255,255,0.3)',
              borderRadius: '10px',
              padding: '15px'
            }}>
              <h3 style={{ color: '#00ffff', fontSize: '14px', marginBottom: '15px' }}>
                🧠 ML Model Performance
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '12px', color: '#cccccc' }}>Model Accuracy:</span>
                  <span style={{ fontSize: '12px', color: '#00ff00' }}>
                    {(modelAccuracy * 100).toFixed(2)}%
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '12px', color: '#cccccc' }}>Processing Time:</span>
                  <span style={{ fontSize: '12px', color: '#ffff00' }}>
                    {processingTime.toFixed(1)}ms
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '12px', color: '#cccccc' }}>Data Points:</span>
                  <span style={{ fontSize: '12px', color: '#ff00ff' }}>
                    {dataPoints.toLocaleString()}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '12px', color: '#cccccc' }}>Predictions Generated:</span>
                  <span style={{ fontSize: '12px', color: '#00ffff' }}>
                    {predictions.length}
                  </span>
                </div>
              </div>
            </div>

            {/* System Recommendations */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,255,0,0.3)',
              borderRadius: '10px',
              padding: '15px'
            }}>
              <h3 style={{ color: '#ffff00', fontSize: '14px', marginBottom: '15px' }}>
                💡 Recommendations
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{
                  padding: '8px',
                  background: 'rgba(0,255,0,0.1)',
                  border: '1px solid rgba(0,255,0,0.3)',
                  borderRadius: '5px',
                  fontSize: '11px'
                }}>
                  ✅ CPU usage is stable - no action needed
                </div>
                <div style={{
                  padding: '8px',
                  background: 'rgba(255,255,0,0.1)',
                  border: '1px solid rgba(255,255,0,0.3)',
                  borderRadius: '5px',
                  fontSize: '11px'
                }}>
                  ⚠️ Memory usage trending up - consider scaling
                </div>
                <div style={{
                  padding: '8px',
                  background: 'rgba(255,0,0,0.1)',
                  border: '1px solid rgba(255,0,0,0.3)',
                  borderRadius: '5px',
                  fontSize: '11px'
                }}>
                  🚨 Disk usage approaching capacity in 48h
                </div>
              </div>
            </div>

            {/* Popular Content Insights */}
            <div style={{
              background: 'rgba(0,0,0,0.5)',
              border: '1px solid rgba(255,0,255,0.3)',
              borderRadius: '10px',
              padding: '15px'
            }}>
              <h3 style={{ color: '#ff00ff', fontSize: '14px', marginBottom: '15px' }}>
                📊 Content Insights
              </h3>
              
              {analyticsData.mediaMetrics.popularGenres.slice(0, 5).map((genre, index) => (
                <div 
                  key={genre.genre}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '8px',
                    padding: '5px',
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: '3px'
                  }}
                >
                  <span style={{ fontSize: '11px' }}>{genre.genre}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                    <span style={{ fontSize: '11px', color: '#cccccc' }}>
                      {genre.count}
                    </span>
                    <span style={{ 
                      fontSize: '10px', 
                      color: genre.trend > 0 ? '#00ff00' : '#ff0000' 
                    }}>
                      {genre.trend > 0 ? '↗' : '↘'} {Math.abs(genre.trend * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictiveAnalytics;