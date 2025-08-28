import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { motion } from 'framer-motion';
import './ServiceVisualization3D.css';

interface Service {
  id: string;
  name: string;
  type: 'media' | 'download' | 'monitor' | 'management' | 'storage';
  status: 'online' | 'offline' | 'degraded' | 'maintenance';
  cpu: number;
  memory: number;
  connections: string[];
  metrics: {
    requests: number;
    errors: number;
    latency: number;
  };
}

const ServiceVisualization3D: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const servicesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const connectionsRef = useRef<THREE.Line[]>([]);
  const animationIdRef = useRef<number>(0);
  
  const [services, setServices] = useState<Service[]>([]);
  const [selectedService, setSelectedService] = useState<Service | null>(null);
  const [viewMode, setViewMode] = useState<'galaxy' | 'network' | 'hierarchy'>('galaxy');
  const [showConnections, setShowConnections] = useState(true);
  const [particlesEnabled, setParticlesEnabled] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);

  useEffect(() => {
    initializeServices();
    setupThreeJS();
    
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  useEffect(() => {
    updateVisualization();
  }, [viewMode, showConnections, services]);

  const initializeServices = () => {
    const serviceList: Service[] = [
      // Media Servers
      { id: 'jellyfin', name: 'Jellyfin', type: 'media', status: 'online', cpu: 45, memory: 60, connections: ['sonarr', 'radarr'], metrics: { requests: 1500, errors: 2, latency: 45 } },
      { id: 'plex', name: 'Plex', type: 'media', status: 'online', cpu: 55, memory: 70, connections: ['tautulli'], metrics: { requests: 2000, errors: 0, latency: 30 } },
      { id: 'emby', name: 'Emby', type: 'media', status: 'degraded', cpu: 60, memory: 65, connections: [], metrics: { requests: 800, errors: 5, latency: 60 } },
      
      // Content Management
      { id: 'sonarr', name: 'Sonarr', type: 'management', status: 'online', cpu: 30, memory: 40, connections: ['prowlarr', 'qbittorrent', 'jellyfin'], metrics: { requests: 500, errors: 0, latency: 20 } },
      { id: 'radarr', name: 'Radarr', type: 'management', status: 'online', cpu: 25, memory: 35, connections: ['prowlarr', 'qbittorrent', 'jellyfin'], metrics: { requests: 400, errors: 0, latency: 25 } },
      { id: 'lidarr', name: 'Lidarr', type: 'management', status: 'online', cpu: 20, memory: 30, connections: ['prowlarr', 'qbittorrent'], metrics: { requests: 200, errors: 0, latency: 30 } },
      { id: 'readarr', name: 'Readarr', type: 'management', status: 'offline', cpu: 0, memory: 0, connections: ['prowlarr'], metrics: { requests: 0, errors: 10, latency: 0 } },
      { id: 'bazarr', name: 'Bazarr', type: 'management', status: 'online', cpu: 15, memory: 25, connections: ['sonarr', 'radarr'], metrics: { requests: 300, errors: 1, latency: 15 } },
      { id: 'prowlarr', name: 'Prowlarr', type: 'management', status: 'online', cpu: 35, memory: 45, connections: ['sonarr', 'radarr', 'lidarr', 'readarr'], metrics: { requests: 1000, errors: 0, latency: 10 } },
      
      // Download Clients
      { id: 'qbittorrent', name: 'qBittorrent', type: 'download', status: 'online', cpu: 70, memory: 80, connections: ['sonarr', 'radarr', 'lidarr'], metrics: { requests: 800, errors: 0, latency: 5 } },
      { id: 'sabnzbd', name: 'SABnzbd', type: 'download', status: 'online', cpu: 50, memory: 55, connections: [], metrics: { requests: 400, errors: 0, latency: 8 } },
      { id: 'transmission', name: 'Transmission', type: 'download', status: 'maintenance', cpu: 0, memory: 0, connections: [], metrics: { requests: 0, errors: 0, latency: 0 } },
      
      // Monitoring
      { id: 'tautulli', name: 'Tautulli', type: 'monitor', status: 'online', cpu: 20, memory: 30, connections: ['plex'], metrics: { requests: 600, errors: 0, latency: 12 } },
      { id: 'grafana', name: 'Grafana', type: 'monitor', status: 'online', cpu: 25, memory: 35, connections: ['prometheus'], metrics: { requests: 1200, errors: 0, latency: 15 } },
      { id: 'prometheus', name: 'Prometheus', type: 'monitor', status: 'online', cpu: 40, memory: 50, connections: ['grafana'], metrics: { requests: 5000, errors: 0, latency: 2 } },
      { id: 'uptimekuma', name: 'Uptime Kuma', type: 'monitor', status: 'online', cpu: 15, memory: 20, connections: [], metrics: { requests: 300, errors: 0, latency: 20 } },
      
      // Storage
      { id: 'nextcloud', name: 'Nextcloud', type: 'storage', status: 'online', cpu: 35, memory: 45, connections: [], metrics: { requests: 700, errors: 2, latency: 35 } },
      { id: 'syncthing', name: 'Syncthing', type: 'storage', status: 'online', cpu: 30, memory: 40, connections: [], metrics: { requests: 400, errors: 0, latency: 25 } },
      { id: 'duplicati', name: 'Duplicati', type: 'storage', status: 'online', cpu: 25, memory: 30, connections: [], metrics: { requests: 100, errors: 0, latency: 40 } }
    ];
    
    setServices(serviceList);
  };

  const setupThreeJS = () => {
    if (!mountRef.current) return;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 50, 200);
    
    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 20, 50);
    camera.lookAt(0, 0, 0);
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 20;
    controls.maxDistance = 150;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 0.5;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0x00ffff, 0.5);
    directionalLight.position.set(50, 50, 50);
    directionalLight.castShadow = true;
    scene.add(directionalLight);
    
    const pointLight1 = new THREE.PointLight(0xff00ff, 1, 100);
    pointLight1.position.set(20, 20, 20);
    scene.add(pointLight1);
    
    const pointLight2 = new THREE.PointLight(0xffff00, 1, 100);
    pointLight2.position.set(-20, 20, -20);
    scene.add(pointLight2);
    
    // Post-processing
    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(mountRef.current.clientWidth, mountRef.current.clientHeight),
      1.5,
      0.4,
      0.85
    );
    bloomPass.threshold = 0.21;
    bloomPass.strength = 1.2;
    bloomPass.radius = 0.55;
    composer.addPass(bloomPass);
    
    // Grid
    const gridHelper = new THREE.GridHelper(100, 50, 0x00ffff, 0x004444);
    scene.add(gridHelper);
    
    // Add particles
    if (particlesEnabled) {
      createParticles(scene);
    }
    
    // Store references
    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    composerRef.current = composer;
    
    // Create service nodes
    createServiceNodes();
    
    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      
      controls.update();
      
      // Animate service nodes
      servicesRef.current.forEach((mesh, id) => {
        const service = services.find(s => s.id === id);
        if (service) {
          // Pulse based on CPU usage
          const scale = 1 + (service.cpu / 100) * 0.3 * Math.sin(Date.now() * 0.001);
          mesh.scale.setScalar(scale);
          
          // Rotate based on activity
          mesh.rotation.y += service.metrics.requests * 0.00001;
        }
      });
      
      // Animate connections
      connectionsRef.current.forEach((line, index) => {
        const material = line.material as THREE.LineBasicMaterial;
        material.opacity = 0.3 + 0.3 * Math.sin(Date.now() * 0.002 + index);
      });
      
      composer.render();
    };
    
    animate();
    
    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
      composer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      mountRef.current?.removeChild(renderer.domElement);
    };
  };

  const createParticles = (scene: THREE.Scene) => {
    const particleCount = 1000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 200;
      positions[i + 1] = Math.random() * 100;
      positions[i + 2] = (Math.random() - 0.5) * 200;
      
      const color = new THREE.Color();
      color.setHSL(180 + Math.random() * 60, 1, 0.5);
      colors[i] = color.r;
      colors[i + 1] = color.g;
      colors[i + 2] = color.b;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });
    
    const particles = new THREE.Points(geometry, material);
    scene.add(particles);
  };

  const createServiceNodes = () => {
    if (!sceneRef.current) return;
    
    // Clear existing nodes
    servicesRef.current.forEach(mesh => {
      sceneRef.current?.remove(mesh);
    });
    servicesRef.current.clear();
    
    // Clear existing connections
    connectionsRef.current.forEach(line => {
      sceneRef.current?.remove(line);
    });
    connectionsRef.current = [];
    
    services.forEach((service, index) => {
      const geometry = getGeometryForServiceType(service.type);
      const material = new THREE.MeshPhongMaterial({
        color: getColorForStatus(service.status),
        emissive: getColorForStatus(service.status),
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.8,
        wireframe: false
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      
      // Position based on view mode
      const position = getPositionForService(service, index);
      mesh.position.set(position.x, position.y, position.z);
      
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      
      // Add glow effect
      const glowGeometry = geometry.clone();
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: getColorForStatus(service.status),
        transparent: true,
        opacity: 0.2,
        side: THREE.BackSide
      });
      const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
      glowMesh.scale.multiplyScalar(1.2);
      mesh.add(glowMesh);
      
      // Add to scene and store reference
      sceneRef.current.add(mesh);
      servicesRef.current.set(service.id, mesh);
      
      // Add label
      createServiceLabel(service, mesh.position);
    });
    
    // Create connections
    if (showConnections) {
      createConnections();
    }
  };

  const getGeometryForServiceType = (type: string): THREE.BufferGeometry => {
    switch (type) {
      case 'media':
        return new THREE.SphereGeometry(2, 32, 32);
      case 'download':
        return new THREE.ConeGeometry(1.5, 3, 8);
      case 'monitor':
        return new THREE.OctahedronGeometry(1.5);
      case 'management':
        return new THREE.BoxGeometry(2.5, 2.5, 2.5);
      case 'storage':
        return new THREE.CylinderGeometry(1.5, 1.5, 2.5, 8);
      default:
        return new THREE.TetrahedronGeometry(1.5);
    }
  };

  const getColorForStatus = (status: string): number => {
    switch (status) {
      case 'online':
        return 0x00ff00;
      case 'offline':
        return 0xff0000;
      case 'degraded':
        return 0xffff00;
      case 'maintenance':
        return 0xff00ff;
      default:
        return 0x666666;
    }
  };

  const getPositionForService = (service: Service, index: number): THREE.Vector3 => {
    switch (viewMode) {
      case 'galaxy':
        const angle = (index / services.length) * Math.PI * 2;
        const radius = 20 + Math.random() * 20;
        const height = (Math.random() - 0.5) * 20;
        return new THREE.Vector3(
          Math.cos(angle) * radius,
          height,
          Math.sin(angle) * radius
        );
        
      case 'network':
        const layer = getLayerForType(service.type);
        const layerIndex = services.filter(s => getLayerForType(s.type) === layer).indexOf(service);
        const layerCount = services.filter(s => getLayerForType(s.type) === layer).length;
        const layerAngle = (layerIndex / layerCount) * Math.PI * 2;
        const layerRadius = 15 + layer * 10;
        return new THREE.Vector3(
          Math.cos(layerAngle) * layerRadius,
          layer * 5 - 10,
          Math.sin(layerAngle) * layerRadius
        );
        
      case 'hierarchy':
        const typeIndex = ['media', 'management', 'download', 'monitor', 'storage'].indexOf(service.type);
        const typeServices = services.filter(s => s.type === service.type);
        const serviceIndex = typeServices.indexOf(service);
        return new THREE.Vector3(
          (serviceIndex - typeServices.length / 2) * 5,
          typeIndex * 8 - 16,
          0
        );
        
      default:
        return new THREE.Vector3(0, 0, 0);
    }
  };

  const getLayerForType = (type: string): number => {
    switch (type) {
      case 'media':
        return 0;
      case 'management':
        return 1;
      case 'download':
        return 2;
      case 'monitor':
        return 3;
      case 'storage':
        return 4;
      default:
        return 0;
    }
  };

  const createServiceLabel = (service: Service, position: THREE.Vector3) => {
    // Labels would be created using CSS3DRenderer or sprites
    // For simplicity, we'll skip the actual implementation here
  };

  const createConnections = () => {
    if (!sceneRef.current) return;
    
    services.forEach(service => {
      service.connections.forEach(targetId => {
        const sourceMesh = servicesRef.current.get(service.id);
        const targetMesh = servicesRef.current.get(targetId);
        
        if (sourceMesh && targetMesh) {
          const points = [];
          points.push(sourceMesh.position);
          
          // Add curve for better visuals
          const midPoint = new THREE.Vector3();
          midPoint.addVectors(sourceMesh.position, targetMesh.position);
          midPoint.multiplyScalar(0.5);
          midPoint.y += 5;
          points.push(midPoint);
          
          points.push(targetMesh.position);
          
          const curve = new THREE.CatmullRomCurve3(points);
          const curvePoints = curve.getPoints(50);
          const geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
          
          const material = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            transparent: true,
            opacity: 0.3,
            blending: THREE.AdditiveBlending
          });
          
          const line = new THREE.Line(geometry, material);
          sceneRef.current.add(line);
          connectionsRef.current.push(line);
        }
      });
    });
  };

  const updateVisualization = () => {
    createServiceNodes();
  };

  const handleServiceClick = (service: Service) => {
    setSelectedService(service);
  };

  return (
    <div className="service-visualization-3d cyberpunk-theme">
      <div className="visualization-header">
        <h1 className="title glitch-text" data-text="3D SERVICE GALAXY">
          3D SERVICE GALAXY
        </h1>
        
        <div className="view-controls">
          <button 
            className={`view-btn ${viewMode === 'galaxy' ? 'active' : ''}`}
            onClick={() => setViewMode('galaxy')}
          >
            Galaxy View
          </button>
          <button 
            className={`view-btn ${viewMode === 'network' ? 'active' : ''}`}
            onClick={() => setViewMode('network')}
          >
            Network View
          </button>
          <button 
            className={`view-btn ${viewMode === 'hierarchy' ? 'active' : ''}`}
            onClick={() => setViewMode('hierarchy')}
          >
            Hierarchy View
          </button>
        </div>
      </div>

      <div className="visualization-container">
        <div ref={mountRef} className="threejs-mount" />
        
        {/* Legend */}
        <div className="legend-panel">
          <h3>Service Types</h3>
          <div className="legend-item">
            <span className="legend-shape sphere"></span>
            <span>Media Servers</span>
          </div>
          <div className="legend-item">
            <span className="legend-shape cube"></span>
            <span>Management</span>
          </div>
          <div className="legend-item">
            <span className="legend-shape cone"></span>
            <span>Download</span>
          </div>
          <div className="legend-item">
            <span className="legend-shape octahedron"></span>
            <span>Monitoring</span>
          </div>
          <div className="legend-item">
            <span className="legend-shape cylinder"></span>
            <span>Storage</span>
          </div>
          
          <h3>Status</h3>
          <div className="legend-item">
            <span className="status-dot online"></span>
            <span>Online</span>
          </div>
          <div className="legend-item">
            <span className="status-dot offline"></span>
            <span>Offline</span>
          </div>
          <div className="legend-item">
            <span className="status-dot degraded"></span>
            <span>Degraded</span>
          </div>
          <div className="legend-item">
            <span className="status-dot maintenance"></span>
            <span>Maintenance</span>
          </div>
        </div>
        
        {/* Controls */}
        <div className="visualization-controls">
          <label className="control-item">
            <input
              type="checkbox"
              checked={showConnections}
              onChange={(e) => setShowConnections(e.target.checked)}
            />
            <span>Show Connections</span>
          </label>
          <label className="control-item">
            <input
              type="checkbox"
              checked={particlesEnabled}
              onChange={(e) => setParticlesEnabled(e.target.checked)}
            />
            <span>Particles</span>
          </label>
          <label className="control-item">
            <input
              type="checkbox"
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
            />
            <span>Auto Rotate</span>
          </label>
        </div>
        
        {/* Service Details */}
        {selectedService && (
          <motion.div
            className="service-details"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            <h2>{selectedService.name}</h2>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="label">Status:</span>
                <span className={`value ${selectedService.status}`}>
                  {selectedService.status.toUpperCase()}
                </span>
              </div>
              <div className="detail-item">
                <span className="label">CPU:</span>
                <span className="value">{selectedService.cpu}%</span>
              </div>
              <div className="detail-item">
                <span className="label">Memory:</span>
                <span className="value">{selectedService.memory}%</span>
              </div>
              <div className="detail-item">
                <span className="label">Requests:</span>
                <span className="value">{selectedService.metrics.requests}</span>
              </div>
              <div className="detail-item">
                <span className="label">Errors:</span>
                <span className="value">{selectedService.metrics.errors}</span>
              </div>
              <div className="detail-item">
                <span className="label">Latency:</span>
                <span className="value">{selectedService.metrics.latency}ms</span>
              </div>
            </div>
            <div className="connections-list">
              <h3>Connections</h3>
              {selectedService.connections.map(conn => (
                <div key={conn} className="connection-item">{conn}</div>
              ))}
            </div>
            <button 
              className="close-details"
              onClick={() => setSelectedService(null)}
            >
              ×
            </button>
          </motion.div>
        )}
      </div>

      {/* Stats Overview */}
      <div className="stats-overview">
        <div className="stat-card">
          <span className="stat-value">{services.filter(s => s.status === 'online').length}</span>
          <span className="stat-label">Online</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">{services.length}</span>
          <span className="stat-label">Total Services</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">
            {services.reduce((sum, s) => sum + s.connections.length, 0)}
          </span>
          <span className="stat-label">Connections</span>
        </div>
        <div className="stat-card">
          <span className="stat-value">
            {Math.round(services.reduce((sum, s) => sum + s.cpu, 0) / services.length)}%
          </span>
          <span className="stat-label">Avg CPU</span>
        </div>
      </div>
    </div>
  );
};

export default ServiceVisualization3D;