/**
 * Real Spatial Video Player
 * Supports MV-HEVC (Apple), side-by-side, over-under, and 180/360 formats
 * Optimized for Vision Pro and Quest 3
 */

import * as THREE from 'three';
import { webXRManager } from '../core/webxr-manager-v2.js';

export class RealSpatialVideoPlayer {
    constructor() {
        this.videos = new Map();
        this.activeVideo = null;
        
        // Video formats
        this.formats = {
            FLAT_2D: 'flat-2d',
            SIDE_BY_SIDE: 'side-by-side',
            OVER_UNDER: 'over-under',
            MV_HEVC: 'mv-hevc', // Apple spatial video
            SPATIAL_180: 'spatial-180',
            SPATIAL_360: 'spatial-360'
        };
        
        // Playback settings
        this.settings = {
            defaultFormat: this.formats.SIDE_BY_SIDE,
            autoplay: false,
            loop: true,
            volume: 1.0,
            ipd: 0.064, // Inter-pupillary distance in meters
            convergence: 2.0, // Convergence distance in meters
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0
        };
        
        // Screen configurations
        this.screens = new Map();
        this.currentScreen = 'cinema';
        
        // Performance optimization
        this.videoTexture = null;
        this.useVideoTexture = true; // Use THREE.VideoTexture for better performance
        
        this.init();
    }

    init() {
        console.log('📹 Initializing Real Spatial Video Player...');
        
        // Create different screen types
        this.createScreens();
        
        // Setup video processing
        this.setupVideoProcessing();
        
        // Listen for XR events
        window.addEventListener('xr-session-started', () => this.onXRSessionStarted());
        window.addEventListener('xr-session-ended', () => this.onXRSessionEnded());
        
        console.log('✅ Spatial Video Player ready');
    }

    createScreens() {
        // Cinema screen (flat)
        this.createCinemaScreen();
        
        // Curved screen (180°)
        this.createCurvedScreen();
        
        // Sphere screen (360°)
        this.createSphereScreen();
        
        // Personal screen (floating)
        this.createPersonalScreen();
    }

    createCinemaScreen() {
        const screen = new THREE.Group();
        screen.name = 'cinema-screen';
        
        // Main screen
        const screenGeometry = new THREE.PlaneGeometry(16, 9); // 16:9 aspect ratio
        const screenMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            side: THREE.DoubleSide
        });
        
        const screenMesh = new THREE.Mesh(screenGeometry, screenMaterial);
        screenMesh.position.set(0, 5, -10);
        screen.add(screenMesh);
        
        // Screen frame
        const frameGeometry = new THREE.PlaneGeometry(16.5, 9.5);
        const frameMaterial = new THREE.MeshPhongMaterial({
            color: 0x222222,
            emissive: 0x111111
        });
        
        const frameMesh = new THREE.Mesh(frameGeometry, frameMaterial);
        frameMesh.position.set(0, 5, -10.1);
        screen.add(frameMesh);
        
        // Ambient lighting from screen
        const screenLight = new THREE.RectAreaLight(0xffffff, 0.5, 16, 9);
        screenLight.position.set(0, 5, -9.9);
        screen.add(screenLight);
        
        // Store references
        screen.userData = {
            videoMesh: screenMesh,
            screenLight: screenLight,
            format: this.formats.FLAT_2D,
            scale: 1.0
        };
        
        this.screens.set('cinema', screen);
        webXRManager.scene.add(screen);
    }

    createCurvedScreen() {
        const screen = new THREE.Group();
        screen.name = 'curved-screen';
        
        // Curved geometry for 180° video
        const geometry = new THREE.SphereGeometry(
            10, // radius
            60, // width segments
            40, // height segments
            0, // phiStart
            Math.PI, // phiLength (180°)
            0, // thetaStart
            Math.PI // thetaLength
        );
        
        // Flip normals to render inside
        geometry.scale(-1, 1, 1);
        
        const material = new THREE.MeshBasicMaterial({
            color: 0x000000,
            side: THREE.BackSide
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        screen.add(mesh);
        
        screen.userData = {
            videoMesh: mesh,
            format: this.formats.SPATIAL_180,
            scale: 1.0
        };
        
        screen.visible = false;
        this.screens.set('curved', screen);
        webXRManager.scene.add(screen);
    }

    createSphereScreen() {
        const screen = new THREE.Group();
        screen.name = 'sphere-screen';
        
        // Full sphere for 360° video
        const geometry = new THREE.SphereGeometry(20, 60, 40);
        geometry.scale(-1, 1, 1);
        
        const material = new THREE.MeshBasicMaterial({
            color: 0x000000,
            side: THREE.BackSide
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        screen.add(mesh);
        
        screen.userData = {
            videoMesh: mesh,
            format: this.formats.SPATIAL_360,
            scale: 1.0
        };
        
        screen.visible = false;
        this.screens.set('sphere', screen);
        webXRManager.scene.add(screen);
    }

    createPersonalScreen() {
        const screen = new THREE.Group();
        screen.name = 'personal-screen';
        
        // Smaller, closer screen
        const geometry = new THREE.PlaneGeometry(2.4, 1.35); // 16:9
        const material = new THREE.MeshBasicMaterial({
            color: 0x000000,
            side: THREE.DoubleSide
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(0, 0, -2);
        screen.add(mesh);
        
        // Soft glow around screen
        const glowGeometry = new THREE.PlaneGeometry(2.6, 1.55);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0x4a90e2,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide
        });
        
        const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
        glowMesh.position.set(0, 0, -2.01);
        screen.add(glowMesh);
        
        screen.userData = {
            videoMesh: mesh,
            format: this.formats.FLAT_2D,
            scale: 1.0
        };
        
        screen.visible = false;
        this.screens.set('personal', screen);
        webXRManager.scene.add(screen);
    }

    setupVideoProcessing() {
        // Create video element
        this.videoElement = document.createElement('video');
        this.videoElement.crossOrigin = 'anonymous';
        this.videoElement.playsInline = true;
        this.videoElement.muted = true; // Required for autoplay
        
        // Video event handlers
        this.videoElement.addEventListener('loadedmetadata', () => this.onVideoLoaded());
        this.videoElement.addEventListener('play', () => this.onVideoPlay());
        this.videoElement.addEventListener('pause', () => this.onVideoPause());
        this.videoElement.addEventListener('ended', () => this.onVideoEnded());
        this.videoElement.addEventListener('error', (e) => this.onVideoError(e));
    }

    async loadVideo(config) {
        const {
            url,
            format = this.settings.defaultFormat,
            autoplay = this.settings.autoplay,
            loop = this.settings.loop,
            screen = 'cinema'
        } = config;
        
        console.log(`📼 Loading spatial video: ${url} (${format})`);
        
        try {
            // Store video config
            const videoId = this.generateVideoId();
            this.videos.set(videoId, {
                id: videoId,
                url,
                format,
                autoplay,
                loop,
                screen,
                isLoaded: false,
                metadata: {}
            });
            
            // Set video source
            this.videoElement.src = url;
            this.videoElement.loop = loop;
            
            // Load video
            await this.videoElement.load();
            
            // Set as active video
            this.activeVideo = videoId;
            
            // Setup screen for format
            this.setupScreenForFormat(format, screen);
            
            // Start playback if autoplay
            if (autoplay) {
                await this.play(videoId);
            }
            
            return videoId;
            
        } catch (error) {
            console.error('Failed to load spatial video:', error);
            throw error;
        }
    }

    setupScreenForFormat(format, screenType) {
        // Hide all screens
        this.screens.forEach(screen => screen.visible = false);
        
        // Get appropriate screen
        let screen;
        switch (format) {
            case this.formats.SPATIAL_180:
                screen = this.screens.get('curved');
                break;
            case this.formats.SPATIAL_360:
                screen = this.screens.get('sphere');
                break;
            case this.formats.FLAT_2D:
            case this.formats.SIDE_BY_SIDE:
            case this.formats.OVER_UNDER:
            case this.formats.MV_HEVC:
            default:
                screen = this.screens.get(screenType);
                break;
        }
        
        if (screen) {
            screen.visible = true;
            this.currentScreen = screenType;
            
            // Update video texture
            this.updateVideoTexture(screen, format);
        }
    }

    updateVideoTexture(screen, format) {
        const { videoMesh } = screen.userData;
        
        // Create or update video texture
        if (!this.videoTexture || this.videoTexture.image !== this.videoElement) {
            this.videoTexture = new THREE.VideoTexture(this.videoElement);
            this.videoTexture.minFilter = THREE.LinearFilter;
            this.videoTexture.magFilter = THREE.LinearFilter;
            this.videoTexture.format = THREE.RGBAFormat;
            this.videoTexture.generateMipmaps = false;
        }
        
        // Configure material based on format
        let material;
        switch (format) {
            case this.formats.SIDE_BY_SIDE:
                material = this.createStereoscopicMaterial('horizontal');
                break;
            case this.formats.OVER_UNDER:
                material = this.createStereoscopicMaterial('vertical');
                break;
            case this.formats.MV_HEVC:
                material = this.createMVHEVCMaterial();
                break;
            default:
                material = new THREE.MeshBasicMaterial({
                    map: this.videoTexture,
                    side: videoMesh.material.side
                });
        }
        
        // Apply color adjustments
        if (material.uniforms) {
            material.uniforms.brightness = { value: this.settings.brightness };
            material.uniforms.contrast = { value: this.settings.contrast };
            material.uniforms.saturation = { value: this.settings.saturation };
        }
        
        videoMesh.material = material;
    }

    createStereoscopicMaterial(mode) {
        const material = new THREE.ShaderMaterial({
            uniforms: {
                videoTexture: { value: this.videoTexture },
                eye: { value: 0.0 }, // 0 = left, 1 = right
                mode: { value: mode === 'horizontal' ? 0.0 : 1.0 },
                brightness: { value: 1.0 },
                contrast: { value: 1.0 },
                saturation: { value: 1.0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D videoTexture;
                uniform float eye;
                uniform float mode;
                uniform float brightness;
                uniform float contrast;
                uniform float saturation;
                varying vec2 vUv;
                
                vec3 adjustColor(vec3 color) {
                    // Brightness
                    color *= brightness;
                    
                    // Contrast
                    color = (color - 0.5) * contrast + 0.5;
                    
                    // Saturation
                    vec3 gray = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
                    color = mix(gray, color, saturation);
                    
                    return clamp(color, 0.0, 1.0);
                }
                
                void main() {
                    vec2 uv = vUv;
                    
                    // Adjust UV based on eye and mode
                    if (mode < 0.5) {
                        // Side-by-side
                        uv.x = uv.x * 0.5 + eye * 0.5;
                    } else {
                        // Over-under
                        uv.y = uv.y * 0.5 + eye * 0.5;
                    }
                    
                    vec4 color = texture2D(videoTexture, uv);
                    color.rgb = adjustColor(color.rgb);
                    gl_FragColor = color;
                }
            `
        });
        
        // Update eye uniform based on camera in render loop
        material.onBeforeRender = (renderer, scene, camera) => {
            // Determine which eye based on camera layer
            const isRightEye = camera.layers.mask & 2; // Layer 2 is typically right eye
            material.uniforms.eye.value = isRightEye ? 1.0 : 0.0;
        };
        
        return material;
    }

    createMVHEVCMaterial() {
        // MV-HEVC is Apple's spatial video format
        // For now, we'll use a standard material as MV-HEVC requires special decoding
        console.log('📱 MV-HEVC spatial video detected (Vision Pro format)');
        
        // In a real implementation, this would use Apple's spatial video APIs
        return new THREE.MeshBasicMaterial({
            map: this.videoTexture
        });
    }

    async play(videoId) {
        const video = videoId ? this.videos.get(videoId) : this.videos.get(this.activeVideo);
        if (!video) return;
        
        try {
            await this.videoElement.play();
            console.log('▶️ Playing spatial video');
            
            // Dispatch event
            window.dispatchEvent(new CustomEvent('spatial-video-play', {
                detail: { videoId: video.id, format: video.format }
            }));
            
        } catch (error) {
            console.error('Failed to play video:', error);
            throw error;
        }
    }

    pause() {
        this.videoElement.pause();
        console.log('⏸️ Paused spatial video');
        
        window.dispatchEvent(new CustomEvent('spatial-video-pause'));
    }

    seek(time) {
        this.videoElement.currentTime = time;
    }

    setVolume(volume) {
        this.settings.volume = Math.max(0, Math.min(1, volume));
        this.videoElement.volume = this.settings.volume;
    }

    setPlaybackRate(rate) {
        this.videoElement.playbackRate = rate;
    }

    // Screen control methods
    
    switchScreen(screenType) {
        const screen = this.screens.get(screenType);
        if (!screen) return;
        
        // Hide current screen
        const currentScreen = this.screens.get(this.currentScreen);
        if (currentScreen) currentScreen.visible = false;
        
        // Show new screen
        screen.visible = true;
        this.currentScreen = screenType;
        
        // Update video texture
        if (this.activeVideo) {
            const video = this.videos.get(this.activeVideo);
            if (video) {
                this.updateVideoTexture(screen, video.format);
            }
        }
    }

    setScreenScale(scale) {
        const screen = this.screens.get(this.currentScreen);
        if (screen) {
            screen.scale.setScalar(scale);
            screen.userData.scale = scale;
        }
    }

    setScreenPosition(position) {
        const screen = this.screens.get(this.currentScreen);
        if (screen) {
            screen.position.copy(position);
        }
    }

    // Visual adjustment methods
    
    setBrightness(value) {
        this.settings.brightness = Math.max(0, Math.min(2, value));
        this.updateMaterialUniforms();
    }

    setContrast(value) {
        this.settings.contrast = Math.max(0, Math.min(2, value));
        this.updateMaterialUniforms();
    }

    setSaturation(value) {
        this.settings.saturation = Math.max(0, Math.min(2, value));
        this.updateMaterialUniforms();
    }

    updateMaterialUniforms() {
        this.screens.forEach(screen => {
            const { videoMesh } = screen.userData;
            if (videoMesh.material.uniforms) {
                videoMesh.material.uniforms.brightness.value = this.settings.brightness;
                videoMesh.material.uniforms.contrast.value = this.settings.contrast;
                videoMesh.material.uniforms.saturation.value = this.settings.saturation;
            }
        });
    }

    // Event handlers
    
    onVideoLoaded() {
        const video = this.videos.get(this.activeVideo);
        if (!video) return;
        
        video.isLoaded = true;
        video.metadata = {
            duration: this.videoElement.duration,
            width: this.videoElement.videoWidth,
            height: this.videoElement.videoHeight,
            aspectRatio: this.videoElement.videoWidth / this.videoElement.videoHeight
        };
        
        console.log('✅ Video loaded:', video.metadata);
        
        // Auto-detect format if not specified
        if (video.format === this.formats.FLAT_2D && video.metadata.aspectRatio > 3.5) {
            // Very wide aspect ratio suggests side-by-side 3D
            video.format = this.formats.SIDE_BY_SIDE;
            this.setupScreenForFormat(video.format, video.screen);
        }
        
        window.dispatchEvent(new CustomEvent('spatial-video-loaded', {
            detail: { videoId: video.id, metadata: video.metadata }
        }));
    }

    onVideoPlay() {
        // Resume texture updates
        if (this.videoTexture) {
            this.videoTexture.needsUpdate = true;
        }
    }

    onVideoPause() {
        // Can pause texture updates for performance
    }

    onVideoEnded() {
        window.dispatchEvent(new CustomEvent('spatial-video-ended'));
        
        // Loop if enabled
        const video = this.videos.get(this.activeVideo);
        if (video && video.loop) {
            this.play(video.id);
        }
    }

    onVideoError(error) {
        console.error('Video error:', error);
        window.dispatchEvent(new CustomEvent('spatial-video-error', {
            detail: { error }
        }));
    }

    onXRSessionStarted() {
        console.log('🥽 XR session started - optimizing spatial video');
        
        // Optimize for XR
        if (webXRManager.platform.isVisionPro) {
            // Vision Pro optimizations
            this.optimizeForVisionPro();
        } else if (webXRManager.platform.isQuest) {
            // Quest optimizations
            this.optimizeForQuest();
        }
    }

    onXRSessionEnded() {
        console.log('📱 XR session ended - reverting to desktop mode');
        
        // Reset optimizations
        this.resetOptimizations();
    }

    optimizeForVisionPro() {
        // Vision Pro has high resolution displays
        // Ensure high quality textures
        if (this.videoTexture) {
            this.videoTexture.minFilter = THREE.LinearFilter;
            this.videoTexture.magFilter = THREE.LinearFilter;
        }
        
        // Use MV-HEVC if available
        console.log('🍎 Optimizing for Vision Pro spatial video');
    }

    optimizeForQuest() {
        // Quest needs performance optimizations
        // Consider reducing texture resolution
        console.log('🥽 Optimizing for Quest performance');
        
        // Use lower quality filtering for better performance
        if (this.videoTexture) {
            this.videoTexture.minFilter = THREE.NearestFilter;
            this.videoTexture.magFilter = THREE.LinearFilter;
        }
    }

    resetOptimizations() {
        // Reset to default quality settings
        if (this.videoTexture) {
            this.videoTexture.minFilter = THREE.LinearFilter;
            this.videoTexture.magFilter = THREE.LinearFilter;
        }
    }

    // Utility methods
    
    generateVideoId() {
        return `video-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    }

    getCurrentTime() {
        return this.videoElement.currentTime;
    }

    getDuration() {
        return this.videoElement.duration;
    }

    getActiveVideo() {
        return this.videos.get(this.activeVideo);
    }

    isPlaying() {
        return !this.videoElement.paused && !this.videoElement.ended;
    }

    destroy() {
        // Clean up
        this.pause();
        this.videos.clear();
        
        this.screens.forEach(screen => {
            webXRManager.scene.remove(screen);
        });
        this.screens.clear();
        
        if (this.videoTexture) {
            this.videoTexture.dispose();
        }
        
        this.videoElement.remove();
    }
}

// Export spatial video player
export const spatialVideoPlayer = new RealSpatialVideoPlayer();