import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import './DataAnalyticsDashboard.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface AnalyticsData {
  viewingStats: {
    daily: number[];
    weekly: number[];
    monthly: number[];
  };
  storageUsage: {
    movies: number;
    tv: number;
    music: number;
    total: number;
  };
  popularContent: Array<{
    title: string;
    views: number;
    type: string;
  }>;
  userBehavior: {
    peakHours: number[];
    deviceTypes: { [key: string]: number };
    genres: { [key: string]: number };
  };
  bandwidth: {
    download: number[];
    upload: number[];
    timestamps: string[];
  };
}

const DataAnalyticsDashboard: React.FC = () => {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [selectedView, setSelectedView] = useState<'day' | 'week' | 'month'>('day');
  const [isLoading, setIsLoading] = useState(true);
  const threeDChartRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  useEffect(() => {
    fetchAnalyticsData();
    init3DVisualization();
    
    const interval = setInterval(fetchAnalyticsData, 30000); // Update every 30 seconds
    
    return () => {
      clearInterval(interval);
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  const fetchAnalyticsData = async () => {
    try {
      const response = await fetch('/api/analytics/dashboard');
      const data = await response.json();
      setAnalyticsData(data);
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
      setIsLoading(false);
    }
  };

  const init3DVisualization = () => {
    if (!threeDChartRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);

    const camera = new THREE.PerspectiveCamera(
      75,
      threeDChartRef.current.clientWidth / threeDChartRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 10, 20);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(threeDChartRef.current.clientWidth, threeDChartRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    threeDChartRef.current.appendChild(renderer.domElement);

    // Add holographic grid
    const gridHelper = new THREE.GridHelper(30, 30, 0x00ffff, 0x00ffff);
    scene.add(gridHelper);

    // Create 3D bar chart
    const createBar = (height: number, x: number, z: number, color: number) => {
      const geometry = new THREE.BoxGeometry(0.8, height, 0.8);
      const material = new THREE.MeshPhongMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.8,
      });
      const bar = new THREE.Mesh(geometry, material);
      bar.position.set(x, height / 2, z);
      return bar;
    };

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const pointLight = new THREE.PointLight(0x00ffff, 2, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    const pointLight2 = new THREE.PointLight(0xff00ff, 2, 100);
    pointLight2.position.set(-10, 10, -10);
    scene.add(pointLight2);

    sceneRef.current = scene;
    rendererRef.current = renderer;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      
      // Rotate camera around scene
      const time = Date.now() * 0.0005;
      camera.position.x = Math.cos(time) * 20;
      camera.position.z = Math.sin(time) * 20;
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    };
    animate();
  };

  // Chart configurations with cyberpunk theme
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#00ffff',
          font: {
            family: 'Orbitron, monospace',
          },
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        borderColor: '#00ffff',
        borderWidth: 1,
        titleColor: '#00ffff',
        bodyColor: '#ffffff',
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(0, 255, 255, 0.1)',
        },
        ticks: {
          color: '#00ffff',
        },
      },
      y: {
        grid: {
          color: 'rgba(0, 255, 255, 0.1)',
        },
        ticks: {
          color: '#00ffff',
        },
      },
    },
  };

  const viewingStatsData = {
    labels: selectedView === 'day' 
      ? ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
      : selectedView === 'week'
      ? ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
      : ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
    datasets: [
      {
        label: 'Viewing Hours',
        data: analyticsData?.viewingStats[selectedView === 'day' ? 'daily' : selectedView === 'week' ? 'weekly' : 'monthly'] || [],
        borderColor: '#00ffff',
        backgroundColor: 'rgba(0, 255, 255, 0.1)',
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const storageData = {
    labels: ['Movies', 'TV Shows', 'Music'],
    datasets: [
      {
        data: [
          analyticsData?.storageUsage.movies || 0,
          analyticsData?.storageUsage.tv || 0,
          analyticsData?.storageUsage.music || 0,
        ],
        backgroundColor: [
          'rgba(255, 0, 255, 0.8)',
          'rgba(0, 255, 255, 0.8)',
          'rgba(255, 255, 0, 0.8)',
        ],
        borderColor: [
          '#ff00ff',
          '#00ffff',
          '#ffff00',
        ],
        borderWidth: 2,
      },
    ],
  };

  const bandwidthData = {
    labels: analyticsData?.bandwidth.timestamps || [],
    datasets: [
      {
        label: 'Download',
        data: analyticsData?.bandwidth.download || [],
        borderColor: '#00ff00',
        backgroundColor: 'rgba(0, 255, 0, 0.1)',
        tension: 0.4,
      },
      {
        label: 'Upload',
        data: analyticsData?.bandwidth.upload || [],
        borderColor: '#ff00ff',
        backgroundColor: 'rgba(255, 0, 255, 0.1)',
        tension: 0.4,
      },
    ],
  };

  if (isLoading) {
    return (
      <div className="analytics-loading">
        <div className="holographic-loader">
          <div className="loader-ring"></div>
          <div className="loader-ring"></div>
          <div className="loader-ring"></div>
        </div>
        <p>Loading Analytics...</p>
      </div>
    );
  }

  return (
    <div className="analytics-dashboard cyberpunk-theme">
      <div className="dashboard-header">
        <h1 className="dashboard-title glitch-text" data-text="DATA ANALYTICS">
          DATA ANALYTICS
        </h1>
        <div className="view-selector">
          <button
            className={`view-btn ${selectedView === 'day' ? 'active' : ''}`}
            onClick={() => setSelectedView('day')}
          >
            Daily
          </button>
          <button
            className={`view-btn ${selectedView === 'week' ? 'active' : ''}`}
            onClick={() => setSelectedView('week')}
          >
            Weekly
          </button>
          <button
            className={`view-btn ${selectedView === 'month' ? 'active' : ''}`}
            onClick={() => setSelectedView('month')}
          >
            Monthly
          </button>
        </div>
      </div>

      <div className="analytics-grid">
        {/* 3D Visualization */}
        <div className="analytics-card full-width">
          <h2 className="card-title">3D Data Flow</h2>
          <div ref={threeDChartRef} className="three-d-chart"></div>
        </div>

        {/* Viewing Statistics */}
        <div className="analytics-card">
          <h2 className="card-title">Viewing Statistics</h2>
          <div className="chart-container">
            <Line data={viewingStatsData} options={chartOptions} />
          </div>
        </div>

        {/* Storage Usage */}
        <div className="analytics-card">
          <h2 className="card-title">Storage Distribution</h2>
          <div className="chart-container">
            <Doughnut data={storageData} options={chartOptions} />
          </div>
          <div className="storage-stats">
            <div className="stat-item">
              <span className="stat-label">Total Used:</span>
              <span className="stat-value">{analyticsData?.storageUsage.total || 0} TB</span>
            </div>
          </div>
        </div>

        {/* Bandwidth Usage */}
        <div className="analytics-card full-width">
          <h2 className="card-title">Network Bandwidth</h2>
          <div className="chart-container">
            <Line data={bandwidthData} options={chartOptions} />
          </div>
        </div>

        {/* Popular Content */}
        <div className="analytics-card">
          <h2 className="card-title">Trending Content</h2>
          <div className="content-list">
            {analyticsData?.popularContent.map((item, index) => (
              <div key={index} className="content-item">
                <span className="content-rank">#{index + 1}</span>
                <div className="content-info">
                  <span className="content-title">{item.title}</span>
                  <span className="content-type">{item.type}</span>
                </div>
                <span className="content-views">{item.views} views</span>
              </div>
            ))}
          </div>
        </div>

        {/* User Behavior Patterns */}
        <div className="analytics-card">
          <h2 className="card-title">User Patterns</h2>
          <div className="pattern-grid">
            <div className="pattern-item">
              <span className="pattern-label">Peak Hour</span>
              <span className="pattern-value">8:00 PM</span>
            </div>
            <div className="pattern-item">
              <span className="pattern-label">Top Device</span>
              <span className="pattern-value">Smart TV</span>
            </div>
            <div className="pattern-item">
              <span className="pattern-label">Favorite Genre</span>
              <span className="pattern-value">Sci-Fi</span>
            </div>
          </div>
        </div>
      </div>

      {/* Animated data flow particles */}
      <div className="data-particles">
        {[...Array(20)].map((_, i) => (
          <div key={i} className="particle" style={{
            animationDelay: `${i * 0.2}s`,
            left: `${Math.random() * 100}%`,
          }}></div>
        ))}
      </div>
    </div>
  );
};

export default DataAnalyticsDashboard;