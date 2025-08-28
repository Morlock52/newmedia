import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';

interface Service {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'degraded' | 'maintenance';
  type: 'media' | 'download' | 'automation' | 'monitoring' | 'proxy' | 'database';
  position: { x: number; y: number; z: number };
  connections: string[];
  metrics: {
    cpu: number;
    memory: number;
    network: number;
    uptime: number;
    responseTime: number;
  };
  version: string;
  port: number;
  url?: string;
}

interface ServiceGrid3DProps {
  services?: Service[];
  enableInteraction?: boolean;
  enableAnimation?: boolean;
  showConnections?: boolean;
  onServiceClick?: (service: Service) => void;
  viewMode?: 'grid' | 'sphere' | 'cylinder' | 'helix';
}

const ServiceGrid3D: React.FC<ServiceGrid3DProps> = ({
  services: initialServices,
  enableInteraction = true,
  enableAnimation = true,
  showConnections = true,
  onServiceClick,
  viewMode = 'grid'
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<any>(null);
  const frameRef = useRef<number>(0);
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());
  const serviceObjectsRef = useRef<Map<string, THREE.Group>>(new Map());
  const connectionLinesRef = useRef<THREE.Group>(new THREE.Group());

  const [services, setServices] = useState<Service[]>(initialServices || []);
  const [selectedService, setSelectedService] = useState<Service | null>(null);
  const [hoveredService, setHoveredService] = useState<Service | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [showMetrics, setShowMetrics] = useState(true);
  const [filterType, setFilterType] = useState<string>('all');

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    initializeScene();
    createServices();
    if (showConnections) {
      createConnections();
    }
    if (enableAnimation) {
      animate();
    }

    return () => {
      cleanup();
    };
  }, []);

  // Update services when props change
  useEffect(() => {
    if (initialServices) {
      setServices(initialServices);
      updateServiceObjects();
    }
  }, [initialServices]);

  // Update view mode
  useEffect(() => {
    updateServicePositions();
  }, [viewMode, services]);

  // Update connections when services change
  useEffect(() => {
    if (showConnections) {
      updateConnections();
    }
  }, [services, showConnections]);

  const initializeScene = () => {
    if (!mountRef.current) return;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.Fog(0x000000, 10, 50);

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 10, 20);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true,
      powerPreference: 'high-performance'
    });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    mountRef.current.appendChild(renderer.domElement);

    // Controls (simplified OrbitControls-like behavior)
    const controls = {
      enableDamping: true,
      dampingFactor: 0.05,
      enableZoom: true,
      enableRotate: true,
      enablePan: true
    };

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    // Neon lighting effects
    const neonLights = [
      { color: 0x00ffff, position: [15, 5, 0] },
      { color: 0xff00ff, position: [-15, 5, 0] },
      { color: 0xffff00, position: [0, 5, 15] }
    ];

    neonLights.forEach(({ color, position }) => {
      const light = new THREE.PointLight(color, 0.5, 30);
      light.position.set(position[0], position[1], position[2]);
      scene.add(light);
    });

    // Grid floor
    const gridHelper = new THREE.GridHelper(40, 40, 0x00ffff, 0x004444);
    gridHelper.position.y = -5;
    scene.add(gridHelper);

    // Store references
    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    controlsRef.current = controls;

    // Add connection lines group
    scene.add(connectionLinesRef.current);

    // Event listeners
    if (enableInteraction) {
      renderer.domElement.addEventListener('mousemove', onMouseMove);
      renderer.domElement.addEventListener('click', onMouseClick);
      window.addEventListener('resize', onWindowResize);
    }
  };

  const createServices = () => {
    if (!sceneRef.current) return;

    // Default services if none provided
    const defaultServices: Service[] = [
      {
        id: 'jellyfin',
        name: 'Jellyfin',
        status: 'online',
        type: 'media',
        position: { x: 0, y: 0, z: 0 },
        connections: ['sonarr', 'radarr'],
        metrics: { cpu: 15, memory: 45, network: 80, uptime: 99.9, responseTime: 50 },
        version: '10.8.13',
        port: 8096
      },
      {
        id: 'sonarr',
        name: 'Sonarr',
        status: 'online',
        type: 'automation',
        position: { x: -8, y: 0, z: -4 },
        connections: ['prowlarr', 'qbittorrent'],
        metrics: { cpu: 8, memory: 25, network: 40, uptime: 98.5, responseTime: 30 },
        version: '4.0.2',
        port: 8989
      },
      {
        id: 'radarr',
        name: 'Radarr',
        status: 'online',
        type: 'automation',
        position: { x: 8, y: 0, z: -4 },
        connections: ['prowlarr', 'qbittorrent'],
        metrics: { cpu: 10, memory: 30, network: 35, uptime: 99.1, responseTime: 35 },
        version: '5.3.6',
        port: 7878
      },
      {
        id: 'prowlarr',
        name: 'Prowlarr',
        status: 'online',
        type: 'automation',
        position: { x: 0, y: 0, z: -8 },
        connections: ['sonarr', 'radarr'],
        metrics: { cpu: 12, memory: 20, network: 60, uptime: 97.8, responseTime: 80 },
        version: '1.12.2',
        port: 9696
      },
      {
        id: 'qbittorrent',
        name: 'qBittorrent',
        status: 'online',
        type: 'download',
        position: { x: 0, y: 0, z: 8 },
        connections: ['sonarr', 'radarr'],
        metrics: { cpu: 25, memory: 35, network: 95, uptime: 99.5, responseTime: 15 },
        version: '4.6.3',
        port: 8080
      },
      {
        id: 'bazarr',
        name: 'Bazarr',
        status: 'online',
        type: 'automation',
        position: { x: -4, y: 2, z: 0 },
        connections: ['sonarr', 'radarr'],
        metrics: { cpu: 5, memory: 15, network: 20, uptime: 96.5, responseTime: 45 },
        version: '1.4.0',
        port: 6767
      }
    ];

    setServices(prev => prev.length > 0 ? prev : defaultServices);
    
    defaultServices.forEach(service => {
      createServiceObject(service);
    });
  };

  const createServiceObject = (service: Service) => {
    if (!sceneRef.current) return;

    const group = new THREE.Group();
    group.userData = { service };

    // Main service cube
    const geometry = new THREE.BoxGeometry(2, 2, 2);
    const material = new THREE.MeshPhongMaterial({
      color: getServiceColor(service.status),
      transparent: true,
      opacity: 0.8,
      emissive: getServiceColor(service.status),
      emissiveIntensity: 0.2
    });

    const cube = new THREE.Mesh(geometry, material);
    cube.castShadow = true;
    cube.receiveShadow = true;
    group.add(cube);

    // Status indicator ring
    const ringGeometry = new THREE.RingGeometry(1.5, 2, 16);
    const ringMaterial = new THREE.MeshBasicMaterial({
      color: getServiceColor(service.status),
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide
    });
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 1.5;
    group.add(ring);

    // Type indicator
    const typeGeometry = new THREE.SphereGeometry(0.3, 8, 8);
    const typeMaterial = new THREE.MeshBasicMaterial({
      color: getTypeColor(service.type),
      transparent: true,
      opacity: 0.9
    });
    const typeSphere = new THREE.Mesh(typeGeometry, typeMaterial);
    typeSphere.position.set(0, 2.5, 0);
    group.add(typeSphere);

    // Metrics visualization (floating bars)
    if (showMetrics) {
      const metrics = ['cpu', 'memory', 'network'];
      metrics.forEach((metric, index) => {
        const value = service.metrics[metric as keyof typeof service.metrics] as number;
        const height = (value / 100) * 2;
        
        const barGeometry = new THREE.BoxGeometry(0.2, height, 0.2);
        const barMaterial = new THREE.MeshBasicMaterial({
          color: getMetricColor(metric),
          transparent: true,
          opacity: 0.8
        });
        const bar = new THREE.Mesh(barGeometry, barMaterial);
        bar.position.set((index - 1) * 0.8, height / 2 + 2, 0);
        group.add(bar);
      });
    }

    // Holographic nameplate
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.width = 256;
    canvas.height = 64;
    
    context.fillStyle = 'rgba(0, 0, 0, 0.8)';
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    context.fillStyle = '#00ffff';
    context.font = 'bold 16px monospace';
    context.textAlign = 'center';
    context.fillText(service.name, canvas.width / 2, 25);
    
    context.fillStyle = '#ffff00';
    context.font = '12px monospace';
    context.fillText(`${service.type.toUpperCase()} | ${service.port}`, canvas.width / 2, 45);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(4, 1, 1);
    sprite.position.set(0, -3, 0);
    group.add(sprite);

    // Position based on view mode
    updateServicePosition(group, service);

    // Store reference
    serviceObjectsRef.current.set(service.id, group);
    sceneRef.current.add(group);
  };

  const updateServicePosition = (group: THREE.Group, service: Service) => {
    let position = { ...service.position };

    switch (viewMode) {
      case 'sphere':
        const sphereRadius = 12;
        const phi = Math.acos(-1 + (2 * services.indexOf(service)) / services.length);
        const theta = Math.sqrt(services.length * Math.PI) * phi;
        position = {
          x: sphereRadius * Math.cos(theta) * Math.sin(phi),
          y: sphereRadius * Math.cos(phi),
          z: sphereRadius * Math.sin(theta) * Math.sin(phi)
        };
        break;

      case 'cylinder':
        const cylinderRadius = 10;
        const height = 8;
        const angle = (services.indexOf(service) / services.length) * Math.PI * 2;
        position = {
          x: cylinderRadius * Math.cos(angle),
          y: (services.indexOf(service) % 3 - 1) * height / 2,
          z: cylinderRadius * Math.sin(angle)
        };
        break;

      case 'helix':
        const helixRadius = 8;
        const helixHeight = 12;
        const helixAngle = services.indexOf(service) * 0.5;
        position = {
          x: helixRadius * Math.cos(helixAngle),
          y: (services.indexOf(service) / services.length) * helixHeight - helixHeight / 2,
          z: helixRadius * Math.sin(helixAngle)
        };
        break;

      case 'grid':
      default:
        // Keep original grid positions
        break;
    }

    group.position.set(position.x, position.y, position.z);
  };

  const updateServicePositions = () => {
    services.forEach(service => {
      const group = serviceObjectsRef.current.get(service.id);
      if (group) {
        updateServicePosition(group, service);
      }
    });
  };

  const createConnections = () => {
    if (!sceneRef.current) return;

    // Clear existing connections
    connectionLinesRef.current.clear();

    services.forEach(service => {
      service.connections.forEach(connectionId => {
        const targetService = services.find(s => s.id === connectionId);
        if (targetService) {
          createConnectionLine(service, targetService);
        }
      });
    });
  };

  const createConnectionLine = (fromService: Service, toService: Service) => {
    const fromGroup = serviceObjectsRef.current.get(fromService.id);
    const toGroup = serviceObjectsRef.current.get(toService.id);
    
    if (!fromGroup || !toGroup) return;

    const fromPos = fromGroup.position;
    const toPos = toGroup.position;

    // Create curved connection line
    const curve = new THREE.QuadraticBezierCurve3(
      fromPos,
      new THREE.Vector3(
        (fromPos.x + toPos.x) / 2,
        Math.max(fromPos.y, toPos.y) + 3,
        (fromPos.z + toPos.z) / 2
      ),
      toPos
    );

    const points = curve.getPoints(20);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    
    const material = new THREE.LineBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.6,
      linewidth: 2
    });

    const line = new THREE.Line(geometry, material);
    connectionLinesRef.current.add(line);

    // Animated data flow particles
    createDataFlowParticles(curve, fromService, toService);
  };

  const createDataFlowParticles = (curve: THREE.QuadraticBezierCurve3, fromService: Service, toService: Service) => {
    const particleCount = 5;
    const particleGeometry = new THREE.SphereGeometry(0.1, 4, 4);
    
    for (let i = 0; i < particleCount; i++) {
      const particleMaterial = new THREE.MeshBasicMaterial({
        color: 0xffff00,
        transparent: true,
        opacity: 0.8
      });
      
      const particle = new THREE.Mesh(particleGeometry, particleMaterial);
      particle.userData = {
        curve,
        progress: i / particleCount,
        speed: 0.01 + Math.random() * 0.01
      };
      
      connectionLinesRef.current.add(particle);
    }
  };

  const updateConnections = () => {
    if (showConnections) {
      createConnections();
    }
  };

  const updateServiceObjects = () => {
    // Clear existing objects
    serviceObjectsRef.current.forEach((group, id) => {
      if (sceneRef.current) {
        sceneRef.current.remove(group);
      }
    });
    serviceObjectsRef.current.clear();

    // Create new objects
    services.forEach(service => {
      createServiceObject(service);
    });

    updateConnections();
  };

  const getServiceColor = (status: string): number => {
    switch (status) {
      case 'online': return 0x00ff00;
      case 'offline': return 0xff0000;
      case 'degraded': return 0xffff00;
      case 'maintenance': return 0xff00ff;
      default: return 0x888888;
    }
  };

  const getTypeColor = (type: string): number => {
    switch (type) {
      case 'media': return 0x00ffff;
      case 'download': return 0xff00ff;
      case 'automation': return 0xffff00;
      case 'monitoring': return 0x00ff00;
      case 'proxy': return 0xff6600;
      case 'database': return 0x6600ff;
      default: return 0x888888;
    }
  };

  const getMetricColor = (metric: string): number => {
    switch (metric) {
      case 'cpu': return 0xff0000;
      case 'memory': return 0x00ff00;
      case 'network': return 0x0000ff;
      default: return 0x888888;
    }
  };

  const animate = () => {
    if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;

    // Rotate camera around the scene
    if (enableAnimation) {
      const time = Date.now() * 0.0005 * animationSpeed;
      cameraRef.current.position.x = Math.cos(time) * 25;
      cameraRef.current.position.z = Math.sin(time) * 25;
      cameraRef.current.lookAt(0, 0, 0);
    }

    // Animate service objects
    serviceObjectsRef.current.forEach((group, id) => {
      if (enableAnimation) {
        group.rotation.y += 0.01 * animationSpeed;
        
        // Pulsing effect based on status
        const service = group.userData.service;
        if (service.status === 'online') {
          const pulse = Math.sin(Date.now() * 0.003) * 0.1 + 1;
          group.scale.setScalar(pulse);
        }
      }
    });

    // Animate data flow particles
    connectionLinesRef.current.children.forEach(child => {
      if (child.userData.curve) {
        child.userData.progress += child.userData.speed;
        if (child.userData.progress > 1) {
          child.userData.progress = 0;
        }
        
        const point = child.userData.curve.getPoint(child.userData.progress);
        child.position.copy(point);
      }
    });

    rendererRef.current.render(sceneRef.current, cameraRef.current);
    frameRef.current = requestAnimationFrame(animate);
  };

  const onMouseMove = useCallback((event: MouseEvent) => {
    if (!mountRef.current || !cameraRef.current || !sceneRef.current) return;

    const rect = mountRef.current.getBoundingClientRect();
    mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Raycast for hover detection
    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
    const intersects = raycasterRef.current.intersectObjects(
      Array.from(serviceObjectsRef.current.values()),
      true
    );

    if (intersects.length > 0) {
      const service = intersects[0].object.parent?.userData.service;
      if (service && service !== hoveredService) {
        setHoveredService(service);
        document.body.style.cursor = 'pointer';
      }
    } else {
      if (hoveredService) {
        setHoveredService(null);
        document.body.style.cursor = 'default';
      }
    }
  }, [hoveredService]);

  const onMouseClick = useCallback((event: MouseEvent) => {
    if (!cameraRef.current || !sceneRef.current) return;

    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
    const intersects = raycasterRef.current.intersectObjects(
      Array.from(serviceObjectsRef.current.values()),
      true
    );

    if (intersects.length > 0) {
      const service = intersects[0].object.parent?.userData.service;
      if (service) {
        setSelectedService(service);
        if (onServiceClick) {
          onServiceClick(service);
        }
      }
    }
  }, [onServiceClick]);

  const onWindowResize = useCallback(() => {
    if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

    cameraRef.current.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
  }, []);

  const cleanup = () => {
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current);
    }
    
    if (rendererRef.current) {
      rendererRef.current.dispose();
    }
    
    serviceObjectsRef.current.clear();
    
    if (mountRef.current && rendererRef.current) {
      mountRef.current.removeChild(rendererRef.current.domElement);
    }
    
    // Remove event listeners
    window.removeEventListener('resize', onWindowResize);
    document.body.style.cursor = 'default';
  };

  const filteredServices = filterType === 'all' 
    ? services 
    : services.filter(s => s.type === filterType);

  return (
    <div style={{
      position: 'relative',
      width: '100%',
      height: '600px',
      background: 'linear-gradient(135deg, rgba(0,0,0,0.95) 0%, rgba(20,20,40,0.95) 100%)',
      border: '2px solid #00ffff',
      borderRadius: '15px',
      overflow: 'hidden'
    }}>
      {/* Controls */}
      <div style={{
        position: 'absolute',
        top: '15px',
        left: '15px',
        zIndex: 10,
        display: 'flex',
        gap: '10px',
        flexWrap: 'wrap'
      }}>
        <div style={{
          background: 'rgba(0,0,0,0.8)',
          border: '1px solid #00ffff',
          borderRadius: '5px',
          padding: '8px 12px',
          color: '#ffffff',
          fontSize: '12px',
          fontFamily: 'monospace'
        }}>
          🌐 Service Grid 3D
        </div>

        {/* View Mode Selector */}
        <select
          value={viewMode}
          onChange={(e) => setViewMode(e.target.value as any)}
          style={{
            background: 'rgba(0,0,0,0.8)',
            border: '1px solid #ff00ff',
            borderRadius: '5px',
            padding: '5px 10px',
            color: '#ffffff',
            fontSize: '11px',
            fontFamily: 'monospace'
          }}
        >
          <option value="grid">Grid</option>
          <option value="sphere">Sphere</option>
          <option value="cylinder">Cylinder</option>
          <option value="helix">Helix</option>
        </select>

        {/* Filter */}
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value)}
          style={{
            background: 'rgba(0,0,0,0.8)',
            border: '1px solid #ffff00',
            borderRadius: '5px',
            padding: '5px 10px',
            color: '#ffffff',
            fontSize: '11px',
            fontFamily: 'monospace'
          }}
        >
          <option value="all">All Services</option>
          <option value="media">Media</option>
          <option value="download">Download</option>
          <option value="automation">Automation</option>
          <option value="monitoring">Monitoring</option>
        </select>

        {/* Animation Controls */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          background: 'rgba(0,0,0,0.8)',
          border: '1px solid #00ff00',
          borderRadius: '5px',
          padding: '5px 10px'
        }}>
          <button
            onClick={() => setAnimationSpeed(prev => prev === 0 ? 1 : 0)}
            style={{
              background: 'none',
              border: 'none',
              color: animationSpeed > 0 ? '#00ff00' : '#ff0000',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            {animationSpeed > 0 ? '⏸' : '▶'}
          </button>
          <input
            type="range"
            min="0"
            max="3"
            step="0.1"
            value={animationSpeed}
            onChange={(e) => setAnimationSpeed(parseFloat(e.target.value))}
            style={{
              width: '60px',
              height: '3px'
            }}
          />
        </div>
      </div>

      {/* Legend */}
      <div style={{
        position: 'absolute',
        top: '15px',
        right: '15px',
        zIndex: 10,
        background: 'rgba(0,0,0,0.8)',
        border: '1px solid #ffffff',
        borderRadius: '5px',
        padding: '10px',
        color: '#ffffff',
        fontSize: '10px',
        fontFamily: 'monospace'
      }}>
        <div style={{ marginBottom: '5px', color: '#00ffff' }}>Status:</div>
        <div>🟢 Online 🟡 Degraded 🔴 Offline 🟣 Maintenance</div>
        <div style={{ marginTop: '8px', marginBottom: '5px', color: '#ff00ff' }}>Types:</div>
        <div>📺 Media 📥 Download ⚙️ Auto 📊 Monitor</div>
      </div>

      {/* 3D Viewport */}
      <div
        ref={mountRef}
        style={{
          width: '100%',
          height: '100%',
          cursor: enableInteraction ? 'grab' : 'default'
        }}
      />

      {/* Service Details Panel */}
      <AnimatePresence>
        {selectedService && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            style={{
              position: 'absolute',
              top: '80px',
              right: '15px',
              width: '300px',
              background: 'rgba(0,0,0,0.9)',
              border: '2px solid #00ffff',
              borderRadius: '10px',
              padding: '15px',
              color: '#ffffff',
              fontFamily: 'monospace',
              fontSize: '12px',
              zIndex: 20,
              maxHeight: '400px',
              overflowY: 'auto'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h3 style={{ margin: 0, color: '#00ffff' }}>
                {selectedService.name}
              </h3>
              <button
                onClick={() => setSelectedService(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#ff0000',
                  cursor: 'pointer',
                  fontSize: '16px'
                }}
              >
                ✕
              </button>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>Status:</span>
                <span style={{ color: getServiceColor(selectedService.status) === 0x00ff00 ? '#00ff00' : 
                                    getServiceColor(selectedService.status) === 0xff0000 ? '#ff0000' :
                                    getServiceColor(selectedService.status) === 0xffff00 ? '#ffff00' : '#ff00ff' }}>
                  {selectedService.status.toUpperCase()}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>Type:</span>
                <span style={{ color: '#ffff00' }}>{selectedService.type}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>Port:</span>
                <span style={{ color: '#ff00ff' }}>{selectedService.port}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span>Version:</span>
                <span style={{ color: '#00ffff' }}>{selectedService.version}</span>
              </div>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <h4 style={{ margin: '0 0 10px 0', color: '#ffff00' }}>Metrics</h4>
              {Object.entries(selectedService.metrics).map(([key, value]) => (
                <div key={key} style={{ marginBottom: '8px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                    <span style={{ textTransform: 'capitalize' }}>{key}:</span>
                    <span>{typeof value === 'number' ? 
                      (key === 'uptime' ? `${value.toFixed(1)}%` :
                       key === 'responseTime' ? `${value}ms` :
                       `${value.toFixed(0)}%`) : value}
                    </span>
                  </div>
                  {typeof value === 'number' && key !== 'responseTime' && (
                    <div style={{
                      width: '100%',
                      height: '4px',
                      background: 'rgba(255,255,255,0.2)',
                      borderRadius: '2px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${Math.min(value, 100)}%`,
                        height: '100%',
                        background: `linear-gradient(90deg, ${
                          key === 'cpu' ? '#ff0000' :
                          key === 'memory' ? '#00ff00' :
                          key === 'network' ? '#0000ff' :
                          '#ffff00'
                        }, #ffffff)`,
                        transition: 'width 0.3s ease'
                      }} />
                    </div>
                  )}
                </div>
              ))}
            </div>

            {selectedService.connections.length > 0 && (
              <div>
                <h4 style={{ margin: '0 0 10px 0', color: '#ff00ff' }}>Connections</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                  {selectedService.connections.map(connId => {
                    const connService = services.find(s => s.id === connId);
                    return connService ? (
                      <span
                        key={connId}
                        style={{
                          background: 'rgba(0,255,255,0.2)',
                          border: '1px solid #00ffff',
                          borderRadius: '10px',
                          padding: '2px 6px',
                          fontSize: '10px'
                        }}
                      >
                        {connService.name}
                      </span>
                    ) : null;
                  })}
                </div>
              </div>
            )}

            {selectedService.url && (
              <div style={{ marginTop: '15px' }}>
                <button
                  onClick={() => window.open(selectedService.url, '_blank')}
                  style={{
                    width: '100%',
                    padding: '8px',
                    background: 'rgba(0,255,0,0.2)',
                    border: '1px solid #00ff00',
                    borderRadius: '5px',
                    color: '#00ff00',
                    cursor: 'pointer',
                    fontSize: '11px'
                  }}
                >
                  🔗 Open Service
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hover Tooltip */}
      <AnimatePresence>
        {hoveredService && !selectedService && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            style={{
              position: 'absolute',
              bottom: '20px',
              left: '20px',
              background: 'rgba(0,0,0,0.9)',
              border: '1px solid #ffffff',
              borderRadius: '5px',
              padding: '10px',
              color: '#ffffff',
              fontFamily: 'monospace',
              fontSize: '11px',
              zIndex: 15,
              pointerEvents: 'none'
            }}
          >
            <div style={{ color: '#00ffff', fontWeight: 'bold' }}>
              {hoveredService.name}
            </div>
            <div>Status: {hoveredService.status}</div>
            <div>Type: {hoveredService.type}</div>
            <div>Port: {hoveredService.port}</div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading Overlay */}
      {isLoading && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 30
        }}>
          <div style={{
            color: '#00ffff',
            fontSize: '18px',
            fontFamily: 'monospace',
            textAlign: 'center'
          }}>
            <div style={{ marginBottom: '10px' }}>🔄 Loading Services...</div>
            <div style={{
              width: '200px',
              height: '4px',
              background: 'rgba(255,255,255,0.2)',
              borderRadius: '2px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: '50%',
                height: '100%',
                background: 'linear-gradient(90deg, #00ffff, #ff00ff)',
                animation: 'loading 2s ease-in-out infinite'
              }} />
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
      `}</style>
    </div>
  );
};

export default ServiceGrid3D;