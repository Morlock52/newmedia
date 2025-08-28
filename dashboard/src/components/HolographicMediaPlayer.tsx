import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import { Hands, Results as HandsResults } from '@mediapipe/hands';
import Webcam from 'react-webcam';

interface MediaPlayerProps {
  mediaUrl?: string;
  mediaType?: 'video' | 'audio' | 'stream';
  title?: string;
  onClose?: () => void;
  enableAR?: boolean;
  enableGestures?: boolean;
  onARStateChange?: (isActive: boolean) => void;
}

interface AROverlay {
  type: 'info' | 'controls' | 'effects';
  position: { x: number; y: number; z: number };
  content: string;
  visible: boolean;
}

interface WebXRSession {
  session: XRSession | null;
  isActive: boolean;
  referenceSpace: XRReferenceSpace | null;
}

interface GestureData {
  type: 'swipe' | 'pinch' | 'rotate' | 'tap' | 'hold';
  direction?: 'left' | 'right' | 'up' | 'down';
  scale?: number;
  rotation?: number;
  position?: { x: number; y: number };
}

const HolographicMediaPlayer: React.FC<MediaPlayerProps> = ({
  mediaUrl = '',
  mediaType = 'video',
  title = 'Untitled Media',
  onClose,
  enableAR = false,
  enableGestures = true,
  onARStateChange
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(0.8);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isGestureEnabled, setIsGestureEnabled] = useState(false);
  const [gestureIndicator, setGestureIndicator] = useState<GestureData | null>(null);
  const [visualizerIntensity, setVisualizerIntensity] = useState(0);
  const [isHologramActive, setIsHologramActive] = useState(true);
  const [arOverlays, setArOverlays] = useState<AROverlay[]>([]);
  const [webXRSession, setWebXRSession] = useState<WebXRSession>({ session: null, isActive: false, referenceSpace: null });
  const [handPoses, setHandPoses] = useState<HandsResults | null>(null);
  const [isARSupported, setIsARSupported] = useState(false);
  const [gestureConfidence, setGestureConfidence] = useState(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const gestureCanvasRef = useRef<HTMLCanvasElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const animationFrameRef = useRef<number>(0);
  const webcamRef = useRef<Webcam>(null);
  const handsRef = useRef<Hands | null>(null);
  const arCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const xrFrameRef = useRef<number>(0);

  useEffect(() => {
    setupMediaElement();
    setupAudioVisualizer();
    setupGestureRecognition();
    setupHolographicDisplay();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [mediaUrl]);

  const setupMediaElement = () => {
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (!mediaElement) return;

    mediaElement.addEventListener('loadedmetadata', () => {
      setDuration(mediaElement.duration);
    });

    mediaElement.addEventListener('timeupdate', () => {
      setCurrentTime(mediaElement.currentTime);
    });

    mediaElement.addEventListener('ended', () => {
      setIsPlaying(false);
    });
  };

  const setupAudioVisualizer = () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    
    audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = 256;

    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      const source = audioContextRef.current.createMediaElementSource(mediaElement);
      source.connect(analyserRef.current);
      analyserRef.current.connect(audioContextRef.current.destination);
    }

    const renderVisualizer = () => {
      if (!analyserRef.current) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(dataArray);

      ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height * 0.7;
        
        // Create gradient for bars
        const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
        gradient.addColorStop(0, '#00ffff');
        gradient.addColorStop(0.5, '#ff00ff');
        gradient.addColorStop(1, '#ffff00');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        // Add glow effect
        ctx.shadowBlur = 20;
        ctx.shadowColor = '#00ffff';
        
        x += barWidth + 1;
      }

      // Calculate average intensity for effects
      const average = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
      setVisualizerIntensity(average / 255);

      animationFrameRef.current = requestAnimationFrame(renderVisualizer);
    };

    renderVisualizer();
  };

  const setupGestureRecognition = () => {
    if (!gestureCanvasRef.current || !navigator.mediaDevices) return;

    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      .then(stream => {
        // Setup webcam for gesture detection
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();

        // Simulate gesture detection (in production, use TensorFlow.js HandPose)
        const detectGestures = () => {
          // Simulated gesture detection
          if (Math.random() > 0.98) {
            const gestures: GestureData[] = [
              { type: 'swipe', direction: 'right' },
              { type: 'swipe', direction: 'left' },
              { type: 'pinch', scale: 0.8 },
              { type: 'tap', position: { x: 50, y: 50 } }
            ];
            
            const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
            handleGesture(randomGesture);
          }

          if (isGestureEnabled) {
            requestAnimationFrame(detectGestures);
          }
        };

        if (isGestureEnabled) {
          detectGestures();
        }
      })
      .catch(err => console.log('Camera access denied for gestures:', err));
  };

  const setupHolographicDisplay = () => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    scene.background = null; // Transparent background

    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ 
      alpha: true,
      antialias: true 
    });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Create holographic frame
    const frameGeometry = new THREE.BoxGeometry(4, 2.25, 0.1);
    const frameMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      wireframe: true,
      transparent: true,
      opacity: 0.3
    });
    const frame = new THREE.Mesh(frameGeometry, frameMaterial);
    scene.add(frame);

    // Add holographic particles
    const particleCount = 500;
    const particles = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 10;
      positions[i + 1] = (Math.random() - 0.5) * 10;
      positions[i + 2] = (Math.random() - 0.5) * 10;
    }

    particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particleMaterial = new THREE.PointsMaterial({
      size: 0.02,
      color: 0x00ffff,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });

    const particleSystem = new THREE.Points(particles, particleMaterial);
    scene.add(particleSystem);

    sceneRef.current = scene;
    rendererRef.current = renderer;

    const animateHologram = () => {
      if (!isHologramActive) return;

      frame.rotation.y += 0.005;
      particleSystem.rotation.y += 0.002;

      // Pulse effect based on audio
      const scale = 1 + visualizerIntensity * 0.1;
      frame.scale.set(scale, scale, scale);

      renderer.render(scene, camera);
      requestAnimationFrame(animateHologram);
    };

    animateHologram();
  };

  const handleGesture = (gesture: GestureData) => {
    setGestureIndicator(gesture);
    setTimeout(() => setGestureIndicator(null), 1000);

    switch (gesture.type) {
      case 'swipe':
        if (gesture.direction === 'right') {
          skipForward();
        } else if (gesture.direction === 'left') {
          skipBackward();
        } else if (gesture.direction === 'up') {
          increaseVolume();
        } else if (gesture.direction === 'down') {
          decreaseVolume();
        }
        break;
      
      case 'tap':
        togglePlayPause();
        break;
      
      case 'pinch':
        if (gesture.scale && gesture.scale < 0.9) {
          exitFullscreen();
        } else if (gesture.scale && gesture.scale > 1.1) {
          enterFullscreen();
        }
        break;
      
      case 'rotate':
        if (gesture.rotation) {
          adjustPlaybackRate(gesture.rotation);
        }
        break;
    }
  };

  const togglePlayPause = () => {
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (!mediaElement) return;

    if (isPlaying) {
      mediaElement.pause();
    } else {
      mediaElement.play();
    }
    setIsPlaying(!isPlaying);
  };

  const skipForward = () => {
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      mediaElement.currentTime = Math.min(mediaElement.currentTime + 10, duration);
    }
  };

  const skipBackward = () => {
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      mediaElement.currentTime = Math.max(mediaElement.currentTime - 10, 0);
    }
  };

  const increaseVolume = () => {
    const newVolume = Math.min(volume + 0.1, 1);
    setVolume(newVolume);
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      mediaElement.volume = newVolume;
    }
  };

  const decreaseVolume = () => {
    const newVolume = Math.max(volume - 0.1, 0);
    setVolume(newVolume);
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      mediaElement.volume = newVolume;
    }
  };

  const adjustPlaybackRate = (rotation: number) => {
    const newRate = Math.max(0.5, Math.min(2, 1 + rotation));
    setPlaybackRate(newRate);
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      mediaElement.playbackRate = newRate;
    }
  };

  const enterFullscreen = () => {
    if (containerRef.current?.requestFullscreen) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    }
  };

  const exitFullscreen = () => {
    if (document.exitFullscreen) {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      mediaElement.currentTime = newTime;
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
    if (mediaElement) {
      mediaElement.volume = newVolume;
    }
  };

  return (
    <div className={`holographic-media-player ${isFullscreen ? 'fullscreen' : ''}`} ref={containerRef}>
      {/* Holographic background effect */}
      <div className="holographic-background">
        <div className="hologram-lines"></div>
        <div className="hologram-grid"></div>
      </div>

      {/* Media elements */}
      {mediaType === 'video' ? (
        <video
          ref={videoRef}
          src={mediaUrl}
          className="media-video"
          onClick={togglePlayPause}
        />
      ) : (
        <audio
          ref={audioRef}
          src={mediaUrl}
        />
      )}

      {/* Audio visualizer */}
      <canvas
        ref={canvasRef}
        className="audio-visualizer"
        width={800}
        height={200}
      />

      {/* Gesture canvas (hidden) */}
      <canvas
        ref={gestureCanvasRef}
        className="gesture-canvas"
        style={{ display: 'none' }}
      />

      {/* Holographic UI overlay */}
      <div className="player-overlay">
        {/* Title bar */}
        <div className="player-header">
          <h2 className="media-title glitch-text" data-text={title}>
            {title}
          </h2>
          {onClose && (
            <button className="close-button" onClick={onClose}>
              ×
            </button>
          )}
        </div>

        {/* Gesture indicator */}
        <AnimatePresence>
          {gestureIndicator && (
            <motion.div
              className="gesture-indicator"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0 }}
            >
              <div className="gesture-icon">
                {gestureIndicator.type === 'swipe' && `→ ${gestureIndicator.direction}`}
                {gestureIndicator.type === 'tap' && '👆'}
                {gestureIndicator.type === 'pinch' && '🤏'}
                {gestureIndicator.type === 'rotate' && '🔄'}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Control panel */}
        <div className="player-controls">
          {/* Play controls */}
          <div className="control-row">
            <button className="control-btn" onClick={skipBackward}>
              ⏮ -10s
            </button>
            
            <button className="control-btn play-btn" onClick={togglePlayPause}>
              {isPlaying ? '⏸' : '▶'}
            </button>
            
            <button className="control-btn" onClick={skipForward}>
              +10s ⏭
            </button>
          </div>

          {/* Progress bar */}
          <div className="progress-container">
            <span className="time-display">{formatTime(currentTime)}</span>
            <input
              type="range"
              className="progress-bar"
              min="0"
              max={duration}
              value={currentTime}
              onChange={handleSeek}
            />
            <span className="time-display">{formatTime(duration)}</span>
          </div>

          {/* Volume and settings */}
          <div className="control-row">
            <div className="volume-control">
              <span className="volume-icon">🔊</span>
              <input
                type="range"
                className="volume-slider"
                min="0"
                max="1"
                step="0.01"
                value={volume}
                onChange={handleVolumeChange}
              />
              <span className="volume-value">{Math.round(volume * 100)}%</span>
            </div>

            <div className="playback-rate">
              <span>Speed:</span>
              <select
                value={playbackRate}
                onChange={(e) => {
                  const rate = parseFloat(e.target.value);
                  setPlaybackRate(rate);
                  const mediaElement = mediaType === 'video' ? videoRef.current : audioRef.current;
                  if (mediaElement) {
                    mediaElement.playbackRate = rate;
                  }
                }}
                className="rate-selector"
              >
                <option value="0.5">0.5x</option>
                <option value="0.75">0.75x</option>
                <option value="1">1x</option>
                <option value="1.25">1.25x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2x</option>
              </select>
            </div>

            <button
              className={`control-btn ${isGestureEnabled ? 'active' : ''}`}
              onClick={() => setIsGestureEnabled(!isGestureEnabled)}
              title="Toggle gesture control"
            >
              👋
            </button>

            <button
              className={`control-btn ${isHologramActive ? 'active' : ''}`}
              onClick={() => setIsHologramActive(!isHologramActive)}
              title="Toggle holographic effects"
            >
              🔮
            </button>

            <button
              className="control-btn"
              onClick={isFullscreen ? exitFullscreen : enterFullscreen}
            >
              {isFullscreen ? '⛶' : '⛶'}
            </button>
          </div>
        </div>

        {/* Gesture control instructions */}
        {isGestureEnabled && (
          <div className="gesture-instructions">
            <p>👈 Swipe left/right: Skip backward/forward</p>
            <p>👆 Swipe up/down: Volume control</p>
            <p>👆 Tap: Play/Pause</p>
            <p>🤏 Pinch: Fullscreen toggle</p>
          </div>
        )}
      </div>

      {/* Cyberpunk effects */}
      <div className="cyberpunk-effects">
        <div className="scan-line"></div>
        <div className="glitch-effect"></div>
        <div className="neon-border"></div>
      </div>
    </div>
  );
};

export default HolographicMediaPlayer;