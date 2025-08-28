import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton';
import { ARButton } from 'three/examples/jsm/webxr/ARButton';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import './ARVRMediaExperience.css';

interface MediaItem {
  id: string;
  title: string;
  type: 'movie' | 'tv' | 'music';
  poster: string;
  year: number;
  rating: number;
  duration: number;
}

const ARVRMediaExperience: React.FC = () => {
  const [isVRSupported, setIsVRSupported] = useState(false);
  const [isARSupported, setIsARSupported] = useState(false);
  const [currentMode, setCurrentMode] = useState<'normal' | 'vr' | 'ar'>('normal');
  const [mediaLibrary, setMediaLibrary] = useState<MediaItem[]>([]);
  const [selectedMedia, setSelectedMedia] = useState<MediaItem | null>(null);
  const [cinemaMode, setCinemaMode] = useState(false);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    checkXRSupport();
    fetchMediaLibrary();
    initializeScene();

    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  const checkXRSupport = async () => {
    if ('xr' in navigator) {
      const nav = navigator as any;
      
      // Check VR support
      const vrSupported = await nav.xr.isSessionSupported('immersive-vr');
      setIsVRSupported(vrSupported);
      
      // Check AR support
      const arSupported = await nav.xr.isSessionSupported('immersive-ar');
      setIsARSupported(arSupported);
    }
  };

  const fetchMediaLibrary = async () => {
    try {
      const response = await fetch('/api/media/library');
      const data = await response.json();
      setMediaLibrary(data);
    } catch (error) {
      console.error('Failed to fetch media library:', error);
      // Use mock data for demo
      setMediaLibrary([
        {
          id: '1',
          title: 'Blade Runner 2049',
          type: 'movie',
          poster: '/posters/blade-runner.jpg',
          year: 2017,
          rating: 8.0,
          duration: 164
        },
        {
          id: '2',
          title: 'The Matrix',
          type: 'movie',
          poster: '/posters/matrix.jpg',
          year: 1999,
          rating: 8.7,
          duration: 136
        },
        {
          id: '3',
          title: 'Cyberpunk: Edgerunners',
          type: 'tv',
          poster: '/posters/edgerunners.jpg',
          year: 2022,
          rating: 8.3,
          duration: 24
        }
      ]);
    }
  };

  const initializeScene = () => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.Fog(0x000000, 5, 50);

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    );
    camera.position.set(0, 1.6, 5);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.xr.enabled = true;
    
    containerRef.current.appendChild(renderer.domElement);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1;
    controls.maxDistance = 20;
    controls.maxPolarAngle = Math.PI / 2;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const neonLight1 = new THREE.PointLight(0x00ffff, 2, 30);
    neonLight1.position.set(5, 5, 5);
    scene.add(neonLight1);

    const neonLight2 = new THREE.PointLight(0xff00ff, 2, 30);
    neonLight2.position.set(-5, 5, -5);
    scene.add(neonLight2);

    const spotLight = new THREE.SpotLight(0xffffff, 1);
    spotLight.position.set(0, 10, 0);
    spotLight.castShadow = true;
    scene.add(spotLight);

    // Create cyberpunk environment
    createCyberpunkEnvironment(scene);
    
    // Create media gallery
    createMediaGallery(scene);

    // VR/AR buttons
    if (isVRSupported) {
      const vrButton = VRButton.createButton(renderer);
      document.body.appendChild(vrButton);
    }

    if (isARSupported) {
      const arButton = ARButton.createButton(renderer, {
        requiredFeatures: ['hit-test'],
        optionalFeatures: ['dom-overlay'],
        domOverlay: { root: document.body }
      });
      document.body.appendChild(arButton);
    }

    // XR Controllers
    setupXRControllers(renderer, scene);

    // Store references
    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    controlsRef.current = controls;

    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      
      // Update controls
      controls.update();
      
      // Animate environment
      animateCyberpunkElements(scene);
      
      // Render
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    window.addEventListener('resize', handleResize);
  };

  const createCyberpunkEnvironment = (scene: THREE.Scene) => {
    // Grid floor
    const gridHelper = new THREE.GridHelper(50, 50, 0x00ffff, 0x00ffff);
    scene.add(gridHelper);

    // Wireframe buildings
    const buildingGeometry = new THREE.BoxGeometry(2, 8, 2);
    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      wireframe: true,
      transparent: true,
      opacity: 0.3
    });

    for (let i = 0; i < 20; i++) {
      const building = new THREE.Mesh(buildingGeometry, wireframeMaterial);
      building.position.set(
        (Math.random() - 0.5) * 40,
        4,
        (Math.random() - 0.5) * 40
      );
      building.scale.y = Math.random() * 2 + 0.5;
      scene.add(building);
    }

    // Particle system for atmosphere
    const particleCount = 1000;
    const particles = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 50;
      positions[i + 1] = Math.random() * 20;
      positions[i + 2] = (Math.random() - 0.5) * 50;

      const color = new THREE.Color();
      color.setHSL(Math.random() * 0.2 + 0.5, 1, 0.5);
      colors[i] = color.r;
      colors[i + 1] = color.g;
      colors[i + 2] = color.b;
    }

    particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });

    const particleSystem = new THREE.Points(particles, particleMaterial);
    scene.add(particleSystem);
  };

  const createMediaGallery = (scene: THREE.Scene) => {
    const radius = 8;
    const angleStep = (Math.PI * 2) / mediaLibrary.length;

    mediaLibrary.forEach((media, index) => {
      const angle = angleStep * index;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;

      // Create media poster
      const posterGeometry = new THREE.PlaneGeometry(2, 3);
      const posterMaterial = new THREE.MeshPhongMaterial({
        color: 0xffffff,
        emissive: 0x00ffff,
        emissiveIntensity: 0.2,
        side: THREE.DoubleSide
      });

      const poster = new THREE.Mesh(posterGeometry, posterMaterial);
      poster.position.set(x, 2, z);
      poster.lookAt(0, 2, 0);
      poster.userData = media;

      // Add holographic effect
      const hologramGeometry = new THREE.PlaneGeometry(2.2, 3.2);
      const hologramMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ffff,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide
      });

      const hologram = new THREE.Mesh(hologramGeometry, hologramMaterial);
      hologram.position.copy(poster.position);
      hologram.position.y += 0.01;
      hologram.lookAt(0, 2, 0);

      // Add title text (using sprite for simplicity)
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 64;
      const context = canvas.getContext('2d')!;
      context.fillStyle = '#00ffff';
      context.font = 'bold 24px Orbitron';
      context.textAlign = 'center';
      context.fillText(media.title, 128, 40);

      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ 
        map: texture,
        transparent: true
      });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.position.set(x, 4, z);
      sprite.scale.set(2, 0.5, 1);

      scene.add(poster);
      scene.add(hologram);
      scene.add(sprite);
    });
  };

  const createVirtualCinema = (scene: THREE.Scene) => {
    // Cinema screen
    const screenGeometry = new THREE.PlaneGeometry(16, 9);
    const screenMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      side: THREE.DoubleSide
    });
    const screen = new THREE.Mesh(screenGeometry, screenMaterial);
    screen.position.set(0, 5, -10);
    scene.add(screen);

    // Cinema seats
    const seatGeometry = new THREE.BoxGeometry(0.8, 0.8, 0.8);
    const seatMaterial = new THREE.MeshPhongMaterial({
      color: 0x800000
    });

    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 10; col++) {
        const seat = new THREE.Mesh(seatGeometry, seatMaterial);
        seat.position.set(
          (col - 4.5) * 1.2,
          0.5,
          row * 1.5
        );
        scene.add(seat);
      }
    }

    // Cinema walls
    const wallMaterial = new THREE.MeshPhongMaterial({
      color: 0x1a1a1a,
      side: THREE.DoubleSide
    });

    const leftWall = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 10),
      wallMaterial
    );
    leftWall.position.set(-10, 5, 0);
    leftWall.rotation.y = Math.PI / 2;
    scene.add(leftWall);

    const rightWall = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 10),
      wallMaterial
    );
    rightWall.position.set(10, 5, 0);
    rightWall.rotation.y = -Math.PI / 2;
    scene.add(rightWall);
  };

  const setupXRControllers = (renderer: THREE.WebGLRenderer, scene: THREE.Scene) => {
    const controllerModelFactory = new XRControllerModelFactory();

    // Controller 1
    const controller1 = renderer.xr.getController(0);
    scene.add(controller1);

    const controllerGrip1 = renderer.xr.getControllerGrip(0);
    controllerGrip1.add(controllerModelFactory.createControllerModel(controllerGrip1));
    scene.add(controllerGrip1);

    // Controller 2
    const controller2 = renderer.xr.getController(1);
    scene.add(controller2);

    const controllerGrip2 = renderer.xr.getControllerGrip(1);
    controllerGrip2.add(controllerModelFactory.createControllerModel(controllerGrip2));
    scene.add(controllerGrip2);

    // Add ray pointers
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -5)
    ]);
    const material = new THREE.LineBasicMaterial({
      color: 0x00ffff,
      linewidth: 2
    });

    const line1 = new THREE.Line(geometry, material);
    controller1.add(line1);

    const line2 = new THREE.Line(geometry, material.clone());
    controller2.add(line2);

    // Controller events
    controller1.addEventListener('selectstart', onSelectStart);
    controller1.addEventListener('selectend', onSelectEnd);
    controller2.addEventListener('selectstart', onSelectStart);
    controller2.addEventListener('selectend', onSelectEnd);
  };

  const onSelectStart = (event: any) => {
    const controller = event.target;
    
    // Raycasting for interaction
    const tempMatrix = new THREE.Matrix4();
    tempMatrix.identity().extractRotation(controller.matrixWorld);

    const raycaster = new THREE.Raycaster();
    raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
    raycaster.ray.direction.set(0, 0, -1).applyMatrix4(tempMatrix);

    if (sceneRef.current) {
      const intersects = raycaster.intersectObjects(sceneRef.current.children, true);
      
      if (intersects.length > 0) {
        const intersected = intersects[0].object;
        if (intersected.userData && intersected.userData.id) {
          setSelectedMedia(intersected.userData);
          playHapticFeedback(controller);
        }
      }
    }
  };

  const onSelectEnd = (event: any) => {
    // Handle select end
  };

  const playHapticFeedback = (controller: any) => {
    const session = rendererRef.current?.xr.getSession();
    if (session && session.inputSources) {
      for (const source of session.inputSources) {
        if (source.gamepad && source.gamepad.hapticActuators) {
          source.gamepad.hapticActuators[0].pulse(0.5, 100);
        }
      }
    }
  };

  const animateCyberpunkElements = (scene: THREE.Scene) => {
    const time = Date.now() * 0.001;

    // Animate particles
    scene.traverse((child) => {
      if (child instanceof THREE.Points) {
        child.rotation.y = time * 0.05;
      }
      
      // Animate holographic effects
      if (child instanceof THREE.Mesh && child.material && 
          (child.material as any).emissive) {
        (child.material as any).emissiveIntensity = 
          0.2 + Math.sin(time * 2) * 0.1;
      }
    });
  };

  const handleResize = () => {
    if (!cameraRef.current || !rendererRef.current) return;

    cameraRef.current.aspect = window.innerWidth / window.innerHeight;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(window.innerWidth, window.innerHeight);
  };

  const toggleCinemaMode = () => {
    setCinemaMode(!cinemaMode);
    if (sceneRef.current) {
      // Clear scene and create cinema
      while (sceneRef.current.children.length > 0) {
        sceneRef.current.remove(sceneRef.current.children[0]);
      }
      
      if (!cinemaMode) {
        createVirtualCinema(sceneRef.current);
      } else {
        createCyberpunkEnvironment(sceneRef.current);
        createMediaGallery(sceneRef.current);
      }
    }
  };

  return (
    <div className="ar-vr-media-experience">
      <div ref={containerRef} className="webxr-container" />
      
      <div className="xr-controls">
        <div className="xr-status">
          <div className="status-item">
            <span className="status-label">VR:</span>
            <span className={`status-value ${isVRSupported ? 'supported' : 'unsupported'}`}>
              {isVRSupported ? 'Supported' : 'Not Supported'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">AR:</span>
            <span className={`status-value ${isARSupported ? 'supported' : 'unsupported'}`}>
              {isARSupported ? 'Supported' : 'Not Supported'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Mode:</span>
            <span className="status-value">{currentMode.toUpperCase()}</span>
          </div>
        </div>

        <div className="xr-actions">
          <button
            className="xr-button cinema-toggle"
            onClick={toggleCinemaMode}
          >
            {cinemaMode ? '🎭 Gallery Mode' : '🎬 Cinema Mode'}
          </button>
        </div>
      </div>

      {selectedMedia && (
        <div className="media-info-overlay">
          <h2>{selectedMedia.title}</h2>
          <p>Year: {selectedMedia.year}</p>
          <p>Rating: ⭐ {selectedMedia.rating}</p>
          <p>Duration: {selectedMedia.duration} min</p>
          <button onClick={() => setSelectedMedia(null)}>Close</button>
        </div>
      )}

      <div className="xr-instructions">
        <p>🎮 Use controllers to point and select media</p>
        <p>👆 Pinch to grab and move objects in AR</p>
        <p>👀 Look around to explore the environment</p>
      </div>
    </div>
  );
};

export default ARVRMediaExperience;