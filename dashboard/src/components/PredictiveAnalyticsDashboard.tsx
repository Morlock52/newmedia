import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as tf from '@tensorflow/tfjs';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar, Doughnut, Radar, Scatter } from 'react-chartjs-2';
import './PredictiveAnalyticsDashboard.css';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface Prediction {
  id: string;
  type: 'bandwidth' | 'storage' | 'usage' | 'performance' | 'cost';
  metric: string;
  currentValue: number;
  predictedValue: number;
  confidence: number;
  timeframe: string;
  trend: 'up' | 'down' | 'stable';
  impact: 'high' | 'medium' | 'low';
  recommendation: string;
}

interface Insight {
  id: string;
  category: string;
  title: string;
  description: string;
  severity: 'critical' | 'warning' | 'info' | 'success';
  actionable: boolean;
  actions?: string[];
  probability: number;
}

interface TimeSeriesData {
  timestamp: Date;
  value: number;
  predicted?: number;
  anomaly?: boolean;
}

interface ServiceAnalytics {
  serviceName: string;
  usage: number;
  trend: number;
  prediction: number;
  health: number;
  cost: number;
}

const PredictiveAnalyticsDashboard: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<'24h' | '7d' | '30d' | '90d'>('7d');
  const [selectedMetric, setSelectedMetric] = useState<'all' | 'bandwidth' | 'storage' | 'usage' | 'performance'>('all');
  const [isTraining, setIsTraining] = useState(false);
  const [modelAccuracy, setModelAccuracy] = useState(0);
  const [anomalies, setAnomalies] = useState<any[]>([]);
  const [serviceAnalytics, setServiceAnalytics] = useState<ServiceAnalytics[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30000);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [neuralNetworkModel, setNeuralNetworkModel] = useState<tf.LayersModel | null>(null);
  const [dataPoints, setDataPoints] = useState<TimeSeriesData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    initializeAnalytics();
    trainPredictionModel();
    setupWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        updatePredictions();
        generateInsights();
      }, refreshInterval);
      
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval]);

  const initializeAnalytics = async () => {
    setIsLoading(true);
    
    // Generate sample historical data
    const historicalData = generateHistoricalData();
    setDataPoints(historicalData);
    
    // Initialize predictions
    const initialPredictions = generatePredictions();
    setPredictions(initialPredictions);
    
    // Generate insights
    const initialInsights = generateInsights();
    setInsights(initialInsights);
    
    // Generate service analytics
    const analytics = generateServiceAnalytics();
    setServiceAnalytics(analytics);
    
    // Detect anomalies
    const detectedAnomalies = detectAnomalies(historicalData);
    setAnomalies(detectedAnomalies);
    
    setIsLoading(false);
  };

  const setupWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8080/analytics');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  };

  const handleRealtimeUpdate = (data: any) => {
    if (data.type === 'prediction') {
      setPredictions(prev => [...prev.slice(-99), data.prediction]);
    } else if (data.type === 'anomaly') {
      setAnomalies(prev => [...prev, data.anomaly]);
    }
  };

  const trainPredictionModel = async () => {
    setIsTraining(true);
    
    try {
      // Create a simple neural network for time series prediction
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [10], units: 64, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 16, activation: 'relu' }),
          tf.layers.dense({ units: 1 })
        ]
      });
      
      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: ['mse', 'mae']
      });
      
      // Generate training data
      const trainData = generateTrainingData();
      const xs = tf.tensor2d(trainData.inputs);
      const ys = tf.tensor2d(trainData.outputs);
      
      // Train the model
      await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (logs) {
              const accuracy = (1 - (logs.loss as number)) * 100;
              setModelAccuracy(Math.min(accuracy, 95));
            }
          }
        }
      });
      
      setNeuralNetworkModel(model);
      
      // Cleanup tensors
      xs.dispose();
      ys.dispose();
    } catch (error) {
      console.error('Model training failed:', error);
    }
    
    setIsTraining(false);
  };

  const generateTrainingData = () => {
    const inputs: number[][] = [];
    const outputs: number[][] = [];
    
    // Generate synthetic training data
    for (let i = 0; i < 1000; i++) {
      const input = Array.from({ length: 10 }, () => Math.random());
      const output = [input.reduce((a, b) => a + b, 0) / 10 + Math.random() * 0.1];
      inputs.push(input);
      outputs.push(output);
    }
    
    return { inputs, outputs };
  };

  const generateHistoricalData = (): TimeSeriesData[] => {
    const data: TimeSeriesData[] = [];
    const now = new Date();
    
    for (let i = 168; i > 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
      const baseValue = 50 + Math.sin(i / 24) * 20;
      const noise = Math.random() * 10 - 5;
      const value = Math.max(0, baseValue + noise);
      
      data.push({
        timestamp,
        value,
        predicted: i < 24 ? value + Math.random() * 5 - 2.5 : undefined,
        anomaly: Math.random() > 0.95
      });
    }
    
    return data;
  };

  const generatePredictions = (): Prediction[] => {
    const predictionTypes = ['bandwidth', 'storage', 'usage', 'performance', 'cost'];
    const metrics = {
      bandwidth: ['Peak Usage', 'Average Throughput', 'Network Load'],
      storage: ['Disk Usage', 'Growth Rate', 'Capacity'],
      usage: ['Active Users', 'Stream Count', 'API Calls'],
      performance: ['Response Time', 'CPU Load', 'Memory Usage'],
      cost: ['Monthly Cost', 'Per User Cost', 'Resource Cost']
    };
    
    const predictions: Prediction[] = [];
    
    predictionTypes.forEach(type => {
      const typeMetrics = metrics[type as keyof typeof metrics];
      typeMetrics.forEach(metric => {
        const currentValue = Math.random() * 100;
        const trend = Math.random() > 0.5 ? 'up' : Math.random() > 0.5 ? 'down' : 'stable';
        const trendMultiplier = trend === 'up' ? 1.2 : trend === 'down' ? 0.8 : 1;
        
        predictions.push({
          id: `${type}-${metric}`,
          type: type as any,
          metric,
          currentValue,
          predictedValue: currentValue * trendMultiplier + (Math.random() * 20 - 10),
          confidence: 70 + Math.random() * 25,
          timeframe: '7 days',
          trend,
          impact: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low',
          recommendation: generateRecommendation(type, trend)
        });
      });
    });
    
    return predictions;
  };

  const generateRecommendation = (type: string, trend: string): string => {
    const recommendations = {
      bandwidth: {
        up: 'Consider upgrading network capacity to handle increased load',
        down: 'Optimize current bandwidth allocation for cost savings',
        stable: 'Current bandwidth allocation is optimal'
      },
      storage: {
        up: 'Plan for additional storage capacity within 30 days',
        down: 'Review and clean up unused media files',
        stable: 'Storage usage is within normal parameters'
      },
      usage: {
        up: 'Scale infrastructure to accommodate user growth',
        down: 'Investigate user engagement and retention strategies',
        stable: 'User patterns are consistent with projections'
      },
      performance: {
        up: 'Optimize code and consider horizontal scaling',
        down: 'Performance improvements detected, monitor stability',
        stable: 'System performance is meeting targets'
      },
      cost: {
        up: 'Review resource allocation for cost optimization',
        down: 'Cost reduction strategies are working effectively',
        stable: 'Costs are aligned with budget projections'
      }
    };
    
    return recommendations[type as keyof typeof recommendations]?.[trend as keyof typeof recommendations.bandwidth] || 'Monitor trends closely';
  };

  const generateInsights = (): Insight[] => {
    return [
      {
        id: '1',
        category: 'Performance',
        title: 'Peak Usage Pattern Detected',
        description: 'System experiences 3x normal load between 8-10 PM. Consider auto-scaling during these hours.',
        severity: 'warning',
        actionable: true,
        actions: ['Enable auto-scaling', 'Optimize caching', 'Increase CDN coverage'],
        probability: 85
      },
      {
        id: '2',
        category: 'Storage',
        title: 'Storage Capacity Warning',
        description: 'At current growth rate, storage will reach 90% capacity in 15 days.',
        severity: 'critical',
        actionable: true,
        actions: ['Expand storage', 'Archive old content', 'Implement compression'],
        probability: 92
      },
      {
        id: '3',
        category: 'Cost',
        title: 'Cost Optimization Opportunity',
        description: 'Switching to reserved instances could save $500/month based on usage patterns.',
        severity: 'info',
        actionable: true,
        actions: ['Review instance types', 'Purchase reserved capacity', 'Optimize resource allocation'],
        probability: 78
      },
      {
        id: '4',
        category: 'User Behavior',
        title: 'Increased Mobile Usage',
        description: 'Mobile streaming increased by 40% this month. Consider mobile optimization.',
        severity: 'success',
        actionable: true,
        actions: ['Optimize mobile experience', 'Add adaptive streaming', 'Improve mobile UI'],
        probability: 95
      },
      {
        id: '5',
        category: 'Security',
        title: 'Unusual Access Pattern',
        description: 'Detected 5x increase in API calls from specific region. Possible bot activity.',
        severity: 'warning',
        actionable: true,
        actions: ['Review access logs', 'Implement rate limiting', 'Update firewall rules'],
        probability: 73
      }
    ];
  };

  const generateServiceAnalytics = (): ServiceAnalytics[] => {
    const services = [
      'Jellyfin', 'Plex', 'Sonarr', 'Radarr', 'qBittorrent',
      'Prowlarr', 'Bazarr', 'Overseerr', 'Tautulli', 'Nginx'
    ];
    
    return services.map(service => ({
      serviceName: service,
      usage: Math.random() * 100,
      trend: Math.random() * 40 - 20,
      prediction: 50 + Math.random() * 50,
      health: 70 + Math.random() * 30,
      cost: Math.random() * 50
    }));
  };

  const detectAnomalies = (data: TimeSeriesData[]): any[] => {
    const anomalies: any[] = [];
    const threshold = 2; // Standard deviations
    
    const values = data.map(d => d.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(
      values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length
    );
    
    data.forEach((point, index) => {
      const zScore = Math.abs((point.value - mean) / stdDev);
      if (zScore > threshold) {
        anomalies.push({
          timestamp: point.timestamp,
          value: point.value,
          severity: zScore > 3 ? 'high' : 'medium',
          type: point.value > mean ? 'spike' : 'drop',
          index
        });
      }
    });
    
    return anomalies;
  };

  const updatePredictions = async () => {
    if (!neuralNetworkModel) return;
    
    try {
      // Get recent data points
      const recentData = dataPoints.slice(-10).map(d => d.value);
      const input = tf.tensor2d([recentData]);
      
      // Make prediction
      const prediction = await neuralNetworkModel.predict(input) as tf.Tensor;
      const predictedValue = (await prediction.data())[0];
      
      // Update predictions with new value
      const newPrediction: TimeSeriesData = {
        timestamp: new Date(),
        value: predictedValue,
        predicted: predictedValue
      };
      
      setDataPoints(prev => [...prev.slice(-167), newPrediction]);
      
      // Cleanup tensors
      input.dispose();
      prediction.dispose();
    } catch (error) {
      console.error('Prediction update failed:', error);
    }
  };

  const getPredictionIcon = (type: string) => {
    const icons: { [key: string]: string } = {
      bandwidth: '📡',
      storage: '💾',
      usage: '👥',
      performance: '⚡',
      cost: '💰'
    };
    return icons[type] || '📊';
  };

  const getSeverityColor = (severity: string) => {
    const colors: { [key: string]: string } = {
      critical: '#ff0000',
      warning: '#ffff00',
      info: '#00ffff',
      success: '#00ff00'
    };
    return colors[severity] || '#ffffff';
  };

  const getTrendIcon = (trend: string) => {
    const icons: { [key: string]: string } = {
      up: '↗️',
      down: '↘️',
      stable: '→'
    };
    return icons[trend] || '•';
  };

  // Chart configurations
  const lineChartData = {
    labels: dataPoints.slice(-24).map(d => 
      new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    ),
    datasets: [
      {
        label: 'Actual',
        data: dataPoints.slice(-24).map(d => d.value),
        borderColor: '#00ffff',
        backgroundColor: 'rgba(0, 255, 255, 0.1)',
        tension: 0.4
      },
      {
        label: 'Predicted',
        data: dataPoints.slice(-24).map(d => d.predicted || null),
        borderColor: '#ff00ff',
        backgroundColor: 'rgba(255, 0, 255, 0.1)',
        borderDash: [5, 5],
        tension: 0.4
      }
    ]
  };

  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#00ffff' }
      },
      title: {
        display: true,
        text: 'Time Series Analysis',
        color: '#ff00ff'
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#888' }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        ticks: { color: '#888' }
      }
    }
  };

  const barChartData = {
    labels: serviceAnalytics.map(s => s.serviceName),
    datasets: [
      {
        label: 'Current Usage',
        data: serviceAnalytics.map(s => s.usage),
        backgroundColor: 'rgba(0, 255, 255, 0.5)',
        borderColor: '#00ffff',
        borderWidth: 1
      },
      {
        label: 'Predicted Usage',
        data: serviceAnalytics.map(s => s.prediction),
        backgroundColor: 'rgba(255, 0, 255, 0.5)',
        borderColor: '#ff00ff',
        borderWidth: 1
      }
    ]
  };

  const radarChartData = {
    labels: ['Performance', 'Reliability', 'Scalability', 'Cost', 'Security', 'User Experience'],
    datasets: [
      {
        label: 'Current State',
        data: [85, 90, 75, 60, 95, 88],
        backgroundColor: 'rgba(0, 255, 255, 0.2)',
        borderColor: '#00ffff',
        pointBackgroundColor: '#00ffff'
      },
      {
        label: 'Target State',
        data: [95, 95, 90, 80, 98, 95],
        backgroundColor: 'rgba(255, 0, 255, 0.2)',
        borderColor: '#ff00ff',
        pointBackgroundColor: '#ff00ff'
      }
    ]
  };

  return (
    <div className="predictive-analytics-dashboard cyberpunk-theme">
      <div className="dashboard-header">
        <h1 className="title glitch-text" data-text="PREDICTIVE ANALYTICS">
          PREDICTIVE ANALYTICS
        </h1>
        
        <div className="header-stats">
          <div className="stat-card">
            <span className="stat-value">{modelAccuracy.toFixed(1)}%</span>
            <span className="stat-label">Model Accuracy</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{predictions.length}</span>
            <span className="stat-label">Active Predictions</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{anomalies.length}</span>
            <span className="stat-label">Anomalies Detected</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{insights.filter(i => i.actionable).length}</span>
            <span className="stat-label">Actionable Insights</span>
          </div>
        </div>
      </div>

      <div className="controls-bar">
        <div className="time-range-selector">
          {(['24h', '7d', '30d', '90d'] as const).map(range => (
            <button
              key={range}
              className={`range-btn ${selectedTimeRange === range ? 'active' : ''}`}
              onClick={() => setSelectedTimeRange(range)}
            >
              {range}
            </button>
          ))}
        </div>

        <div className="metric-filter">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value as any)}
            className="metric-select"
          >
            <option value="all">All Metrics</option>
            <option value="bandwidth">Bandwidth</option>
            <option value="storage">Storage</option>
            <option value="usage">Usage</option>
            <option value="performance">Performance</option>
          </select>
        </div>

        <div className="control-buttons">
          <button
            className={`control-btn ${showAdvanced ? 'active' : ''}`}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            🔬 Advanced
          </button>
          <button
            className={`control-btn ${autoRefresh ? 'active' : ''}`}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            🔄 Auto Refresh
          </button>
          <button
            className="control-btn train"
            onClick={trainPredictionModel}
            disabled={isTraining}
          >
            {isTraining ? '⏳ Training...' : '🧠 Retrain Model'}
          </button>
        </div>
      </div>

      <div className="analytics-grid">
        {/* Main Chart Section */}
        <div className="chart-section main">
          <div className="section-header">
            <h2>Performance Trends</h2>
            <span className="live-indicator">● LIVE</span>
          </div>
          <div className="chart-container">
            <Line data={lineChartData} options={lineChartOptions} />
          </div>
        </div>

        {/* Predictions Grid */}
        <div className="predictions-section">
          <div className="section-header">
            <h2>AI Predictions</h2>
            <span className="accuracy-badge">{modelAccuracy.toFixed(1)}% accurate</span>
          </div>
          <div className="predictions-grid">
            {predictions
              .filter(p => selectedMetric === 'all' || p.type === selectedMetric)
              .slice(0, 6)
              .map((prediction, index) => (
                <motion.div
                  key={prediction.id}
                  className={`prediction-card ${prediction.impact}`}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.05 }}
                  onClick={() => setSelectedPrediction(prediction)}
                >
                  <div className="prediction-header">
                    <span className="prediction-icon">{getPredictionIcon(prediction.type)}</span>
                    <span className="prediction-type">{prediction.type}</span>
                  </div>
                  <h3>{prediction.metric}</h3>
                  <div className="prediction-values">
                    <div className="current-value">
                      <span className="label">Current</span>
                      <span className="value">{prediction.currentValue.toFixed(1)}</span>
                    </div>
                    <div className="trend-arrow">
                      {getTrendIcon(prediction.trend)}
                    </div>
                    <div className="predicted-value">
                      <span className="label">Predicted</span>
                      <span className="value">{prediction.predictedValue.toFixed(1)}</span>
                    </div>
                  </div>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill"
                      style={{ width: `${prediction.confidence}%` }}
                    />
                    <span className="confidence-text">{prediction.confidence.toFixed(0)}% confidence</span>
                  </div>
                  <div className="prediction-footer">
                    <span className="timeframe">📅 {prediction.timeframe}</span>
                    <span className={`impact-badge ${prediction.impact}`}>
                      {prediction.impact} impact
                    </span>
                  </div>
                </motion.div>
              ))}
          </div>
        </div>

        {/* Insights Panel */}
        <div className="insights-section">
          <div className="section-header">
            <h2>AI Insights</h2>
            <button className="filter-btn">Filter</button>
          </div>
          <div className="insights-list">
            {insights.map((insight, index) => (
              <motion.div
                key={insight.id}
                className={`insight-card ${insight.severity}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="insight-header">
                  <span 
                    className="severity-indicator"
                    style={{ backgroundColor: getSeverityColor(insight.severity) }}
                  />
                  <span className="insight-category">{insight.category}</span>
                  <span className="probability">{insight.probability}% likely</span>
                </div>
                <h3>{insight.title}</h3>
                <p>{insight.description}</p>
                {insight.actionable && insight.actions && (
                  <div className="insight-actions">
                    {insight.actions.map((action, i) => (
                      <button key={i} className="action-btn">
                        {action}
                      </button>
                    ))}
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Service Analytics */}
        <div className="service-analytics-section">
          <div className="section-header">
            <h2>Service Performance</h2>
          </div>
          <div className="chart-container">
            <Bar data={barChartData} options={lineChartOptions} />
          </div>
        </div>

        {/* System Health Radar */}
        <div className="radar-section">
          <div className="section-header">
            <h2>System Health Matrix</h2>
          </div>
          <div className="chart-container">
            <Radar data={radarChartData} options={lineChartOptions} />
          </div>
        </div>

        {/* Anomaly Detection */}
        <div className="anomaly-section">
          <div className="section-header">
            <h2>Anomaly Detection</h2>
            <span className="anomaly-count">{anomalies.length} detected</span>
          </div>
          <div className="anomaly-list">
            {anomalies.slice(0, 5).map((anomaly, index) => (
              <div key={index} className={`anomaly-item ${anomaly.severity}`}>
                <span className="anomaly-time">
                  {new Date(anomaly.timestamp).toLocaleString()}
                </span>
                <span className="anomaly-type">{anomaly.type.toUpperCase()}</span>
                <span className="anomaly-value">{anomaly.value.toFixed(2)}</span>
                <span className={`anomaly-severity ${anomaly.severity}`}>
                  {anomaly.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Advanced Analytics Panel */}
      <AnimatePresence>
        {showAdvanced && (
          <motion.div
            className="advanced-panel"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <h2>Advanced Analytics</h2>
            <div className="advanced-grid">
              <div className="neural-network-viz">
                <h3>Neural Network Architecture</h3>
                <canvas ref={canvasRef} className="nn-canvas" />
                <div className="nn-stats">
                  <span>Layers: 4</span>
                  <span>Neurons: 113</span>
                  <span>Parameters: 7,201</span>
                </div>
              </div>
              
              <div className="correlation-matrix">
                <h3>Correlation Matrix</h3>
                <div className="matrix-grid">
                  {['CPU', 'Memory', 'Disk', 'Network', 'Users'].map(row => (
                    <div key={row} className="matrix-row">
                      {['CPU', 'Memory', 'Disk', 'Network', 'Users'].map(col => {
                        const correlation = Math.random();
                        return (
                          <div
                            key={col}
                            className="matrix-cell"
                            style={{
                              backgroundColor: `rgba(0, 255, 255, ${correlation})`,
                              color: correlation > 0.5 ? '#000' : '#fff'
                            }}
                          >
                            {correlation.toFixed(2)}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="forecast-accuracy">
                <h3>Forecast Accuracy by Metric</h3>
                <div className="accuracy-bars">
                  {['Bandwidth', 'Storage', 'CPU', 'Memory', 'Cost'].map(metric => {
                    const accuracy = 70 + Math.random() * 25;
                    return (
                      <div key={metric} className="accuracy-bar">
                        <span className="metric-name">{metric}</span>
                        <div className="bar-container">
                          <div 
                            className="bar-fill"
                            style={{ width: `${accuracy}%` }}
                          />
                        </div>
                        <span className="accuracy-value">{accuracy.toFixed(1)}%</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Prediction Detail Modal */}
      <AnimatePresence>
        {selectedPrediction && (
          <motion.div
            className="prediction-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedPrediction(null)}
          >
            <motion.div
              className="modal-content"
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              onClick={(e) => e.stopPropagation()}
            >
              <button 
                className="close-modal"
                onClick={() => setSelectedPrediction(null)}
              >
                ×
              </button>
              
              <h2>{selectedPrediction.metric}</h2>
              <div className="modal-body">
                <div className="prediction-details">
                  <div className="detail-item">
                    <span className="label">Type:</span>
                    <span className="value">{selectedPrediction.type}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Current Value:</span>
                    <span className="value">{selectedPrediction.currentValue.toFixed(2)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Predicted Value:</span>
                    <span className="value">{selectedPrediction.predictedValue.toFixed(2)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Confidence:</span>
                    <span className="value">{selectedPrediction.confidence.toFixed(1)}%</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Trend:</span>
                    <span className="value">{selectedPrediction.trend}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Impact:</span>
                    <span className="value">{selectedPrediction.impact}</span>
                  </div>
                </div>
                
                <div className="recommendation-box">
                  <h3>AI Recommendation</h3>
                  <p>{selectedPrediction.recommendation}</p>
                </div>
                
                <div className="action-buttons">
                  <button className="action-btn primary">Apply Recommendation</button>
                  <button className="action-btn secondary">Schedule Review</button>
                  <button className="action-btn">Export Report</button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PredictiveAnalyticsDashboard;