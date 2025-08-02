/**
 * Spatial Video System - Advanced 3D video playback in spatial environments
 * Supports Apple Vision Pro spatial videos and volumetric content
 */

import { webXRManager } from '../core/webxr-manager.js';

export class SpatialVideo {
    constructor() {
        this.spatialVideos = new Map();
        this.volumetricPlayer = null;
        this.spatialAudioSystem = null;
        this.depthProcessor = null;
        this.stereoRenderer = null;
        this.spatialControls = null;
        this.playbackSynchronizer = null;
        this.adaptiveStreaming = null;
        
        this.supportedFormats = [
            'spatial-stereo', // Apple Vision Pro format
            'volumetric-mesh', // 3D mesh video
            'point-cloud', // Point cloud video
            'light-field', // Light field capture
            'holographic' // Holographic video
        ];
        
        this.init();
    }

    async init() {
        console.log('📹 Initializing Spatial Video System...');
        
        // Initialize spatial video components
        this.initSpatialVideoPlayer();
        
        // Setup stereo rendering
        this.setupStereoRenderer();
        
        // Initialize depth processing
        this.setupDepthProcessor();
        
        // Setup spatial audio
        this.setupSpatialAudioSystem();
        
        // Initialize playback controls
        this.setupSpatialControls();
        
        // Setup adaptive streaming
        this.setupAdaptiveStreaming();
        
        console.log('✅ Spatial Video System initialized');
    }

    initSpatialVideoPlayer() {
        this.spatialVideos = new Map();
        
        // Create main spatial video container
        this.spatialVideoContainer = new THREE.Group();
        webXRManager.scene.add(this.spatialVideoContainer);
        
        // Initialize volumetric player
        this.volumetricPlayer = {
            meshes: new Map(),
            pointClouds: new Map(),
            lightFields: new Map(),
            isPlaying: false,
            currentTime: 0,
            duration: 0
        };
    }

    setupStereoRenderer() {
        console.log('👁️ Setting up stereo renderer...');
        
        this.stereoRenderer = {
            leftEyeCamera: new THREE.PerspectiveCamera(75, 1, 0.1, 1000),
            rightEyeCamera: new THREE.PerspectiveCamera(75, 1, 0.1, 1000),
            eyeSeparation: 0.064, // Average human IPD in meters
            convergence: 1.0,
            renderTargetLeft: null,
            renderTargetRight: null
        };
        
        // Setup render targets for each eye
        this.setupStereoRenderTargets();
    }

    setupStereoRenderTargets() {
        const width = 2048;
        const height = 2048;
        
        this.stereoRenderer.renderTargetLeft = new THREE.WebGLRenderTarget(width, height, {
            format: THREE.RGBAFormat,
            type: THREE.FloatType,
            magFilter: THREE.LinearFilter,
            minFilter: THREE.LinearFilter
        });
        
        this.stereoRenderer.renderTargetRight = new THREE.WebGLRenderTarget(width, height, {
            format: THREE.RGBAFormat,
            type: THREE.FloatType,
            magFilter: THREE.LinearFilter,
            minFilter: THREE.LinearFilter
        });
    }

    setupDepthProcessor() {
        console.log('📏 Setting up depth processor...');
        
        this.depthProcessor = {
            depthTexture: null,
            depthScale: 1000, // mm to meters
            occlusionEnabled: true,
            depthCompositing: true,
            rgbdDecoder: null
        };
        
        // Initialize RGBD decoder for depth video
        this.initRGBDDecoder();
    }

    initRGBDDecoder() {
        // Shader for decoding RGBD spatial video
        const rgbdVertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        const rgbdFragmentShader = `
            uniform sampler2D colorTexture;
            uniform sampler2D depthTexture;
            uniform float depthScale;
            uniform float time;
            varying vec2 vUv;
            
            void main() {
                vec4 color = texture2D(colorTexture, vUv);
                float depth = texture2D(depthTexture, vUv).r * depthScale;
                
                // Apply depth-based effects
                float depthFactor = 1.0 - (depth / 10.0);
                color.rgb *= depthFactor;
                
                gl_FragColor = color;
            }
        `;
        
        this.depthProcessor.rgbdDecoder = new THREE.ShaderMaterial({
            uniforms: {
                colorTexture: { value: null },
                depthTexture: { value: null },
                depthScale: { value: this.depthProcessor.depthScale },
                time: { value: 0 }
            },
            vertexShader: rgbdVertexShader,
            fragmentShader: rgbdFragmentShader
        });
    }

    setupSpatialAudioSystem() {
        console.log('🔊 Setting up spatial audio system...');
        
        this.spatialAudioSystem = {
            listener: new THREE.AudioListener(),
            positionalSounds: new Map(),
            ambisonicDecoder: null,
            spatialSoundFields: new Map(),
            audioWorklet: null
        };
        
        webXRManager.camera.add(this.spatialAudioSystem.listener);
        
        // Initialize ambisonic audio decoder
        this.initAmbisonicDecoder();
        
        // Setup audio worklet for spatial processing
        this.setupAudioWorklet();
    }

    async initAmbisonicDecoder() {
        // Initialize higher-order ambisonic decoder
        this.spatialAudioSystem.ambisonicDecoder = {
            order: 3, // Third-order ambisonics
            channelCount: 16, // (order + 1)^2
            decoder: null
        };
        
        console.log('🎵 Ambisonic decoder initialized');
    }

    async setupAudioWorklet() {
        try {
            // Load custom audio worklet for spatial processing
            await this.spatialAudioSystem.listener.context.audioWorklet.addModule('webxr/spatial-video/spatial-audio-processor.js');
            
            this.spatialAudioSystem.audioWorklet = new AudioWorkletNode(
                this.spatialAudioSystem.listener.context,
                'spatial-audio-processor'
            );
            
            console.log('🎛️ Spatial audio worklet loaded');
        } catch (error) {
            console.warn('Audio worklet not supported:', error);
        }
    }

    setupSpatialControls() {
        console.log('🎮 Setting up spatial controls...');
        
        this.spatialControls = {
            playPauseGesture: null,
            volumeControl: null,
            seekGesture: null,
            spatialNavigation: null,
            handTracking: null
        };
        
        // Initialize gesture-based controls
        this.initGestureControls();
        
        // Setup 3D UI controls
        this.create3DControls();
    }

    initGestureControls() {
        this.spatialControls.handTracking = {
            playPauseGesture: {
                pattern: 'thumb_up',
                callback: () => this.togglePlayback()
            },
            volumeGesture: {
                pattern: 'pinch_drag_vertical',
                callback: (delta) => this.adjustVolume(delta)
            },
            seekGesture: {
                pattern: 'swipe_horizontal',
                callback: (direction) => this.seek(direction)
            }
        };
    }

    create3DControls() {
        const controlsGroup = new THREE.Group();
        
        // Play/Pause button
        const playButton = this.createSpatialButton('▶️', { x: 0, y: -2, z: -3 });
        playButton.userData.action = 'play-pause';
        controlsGroup.add(playButton);
        
        // Volume control
        const volumeSlider = this.createVolumeSlider({ x: -1.5, y: -2, z: -3 });
        controlsGroup.add(volumeSlider);
        
        // Progress bar
        const progressBar = this.createProgressBar({ x: 0, y: -2.5, z: -3 });
        controlsGroup.add(progressBar);
        
        // Quality selector
        const qualitySelector = this.createQualitySelector({ x: 1.5, y: -2, z: -3 });
        controlsGroup.add(qualitySelector);
        
        this.spatialControls.ui = controlsGroup;
        this.spatialVideoContainer.add(controlsGroup);
    }

    createSpatialButton(text, position) {
        const buttonGroup = new THREE.Group();
        
        // Button geometry
        const buttonGeometry = new THREE.CylinderGeometry(0.2, 0.2, 0.05, 16);
        const buttonMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x4a90e2,
            transparent: true,
            opacity: 0.8 
        });
        const button = new THREE.Mesh(buttonGeometry, buttonMaterial);
        
        // Button text
        const textTexture = this.createTextTexture(text, 64);
        const textGeometry = new THREE.PlaneGeometry(0.3, 0.3);
        const textMaterial = new THREE.MeshBasicMaterial({
            map: textTexture,
            transparent: true
        });
        const textMesh = new THREE.Mesh(textGeometry, textMaterial);
        textMesh.position.y = 0.03;
        
        buttonGroup.add(button);
        buttonGroup.add(textMesh);
        buttonGroup.position.set(position.x, position.y, position.z);
        
        // Add interaction data
        button.userData = {
            isInteractive: true,
            onSelect: () => this.handleButtonPress(button.userData.action)
        };
        
        return buttonGroup;
    }

    createVolumeSlider(position) {
        const sliderGroup = new THREE.Group();
        
        // Slider track
        const trackGeometry = new THREE.CylinderGeometry(0.01, 0.01, 1, 8);
        const trackMaterial = new THREE.MeshLambertMaterial({ color: 0x666666 });
        const track = new THREE.Mesh(trackGeometry, trackMaterial);
        track.rotation.z = Math.PI / 2;
        
        // Slider handle
        const handleGeometry = new THREE.SphereGeometry(0.05, 16, 8);
        const handleMaterial = new THREE.MeshLambertMaterial({ color: 0x4a90e2 });
        const handle = new THREE.Mesh(handleGeometry, handleMaterial);
        handle.position.x = 0; // Will be updated based on volume
        
        sliderGroup.add(track);
        sliderGroup.add(handle);
        sliderGroup.position.set(position.x, position.y, position.z);
        
        // Store references for updates
        sliderGroup.userData = {
            track: track,
            handle: handle,
            value: 0.8, // Default volume
            isSlider: true
        };
        
        return sliderGroup;
    }

    createProgressBar(position) {
        const progressGroup = new THREE.Group();
        
        // Background bar
        const bgGeometry = new THREE.BoxGeometry(3, 0.05, 0.05);
        const bgMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
        const background = new THREE.Mesh(bgGeometry, bgMaterial);
        
        // Progress bar
        const progressGeometry = new THREE.BoxGeometry(0, 0.06, 0.06);
        const progressMaterial = new THREE.MeshLambertMaterial({ color: 0x4a90e2 });
        const progress = new THREE.Mesh(progressGeometry, progressMaterial);
        progress.position.x = -1.5; // Start from left
        
        progressGroup.add(background);
        progressGroup.add(progress);
        progressGroup.position.set(position.x, position.y, position.z);
        
        // Store references for updates
        progressGroup.userData = {
            background: background,
            progress: progress,
            duration: 0,
            currentTime: 0,
            isProgressBar: true
        };
        
        return progressGroup;
    }

    createQualitySelector(position) {
        const qualityGroup = new THREE.Group();
        
        const qualities = ['4K', '2K', '1080p', 'Auto'];
        qualities.forEach((quality, index) => {
            const button = this.createSpatialButton(quality, { 
                x: 0, 
                y: index * 0.3, 
                z: 0 
            });
            button.userData.action = `quality-${quality}`;
            qualityGroup.add(button);
        });
        
        qualityGroup.position.set(position.x, position.y, position.z);
        return qualityGroup;
    }

    setupAdaptiveStreaming() {
        console.log('📡 Setting up adaptive streaming...');
        
        this.adaptiveStreaming = {
            qualityLevels: [
                { name: '4K', width: 3840, height: 2160, bitrate: 25000000 },
                { name: '2K', width: 2560, height: 1440, bitrate: 12000000 },
                { name: '1080p', width: 1920, height: 1080, bitrate: 6000000 },
                { name: '720p', width: 1280, height: 720, bitrate: 3000000 }
            ],
            currentQuality: '2K',
            autoAdapt: true,
            bandwidth: 0,
            bufferHealth: 0
        };
        
        // Monitor network conditions
        this.startBandwidthMonitoring();
    }

    startBandwidthMonitoring() {
        setInterval(() => {
            // Monitor bandwidth and adjust quality
            this.monitorNetworkConditions();
        }, 5000);
    }

    monitorNetworkConditions() {
        // Get connection info if available
        if (navigator.connection) {
            const connection = navigator.connection;
            this.adaptiveStreaming.bandwidth = connection.downlink * 1000000; // Convert to bps
            
            if (this.adaptiveStreaming.autoAdapt) {
                this.adaptQuality();
            }
        }
    }

    adaptQuality() {
        const bandwidth = this.adaptiveStreaming.bandwidth;
        let targetQuality = '720p'; // Default fallback
        
        for (const quality of this.adaptiveStreaming.qualityLevels) {
            if (bandwidth > quality.bitrate * 1.5) { // 1.5x buffer
                targetQuality = quality.name;
                break;
            }
        }
        
        if (targetQuality !== this.adaptiveStreaming.currentQuality) {
            this.switchQuality(targetQuality);
        }
    }

    async loadSpatialVideo(videoConfig) {
        console.log('📼 Loading spatial video:', videoConfig.url);
        
        const videoId = `video_${Date.now()}`;
        
        try {
            let spatialVideoObject;
            
            switch (videoConfig.format) {
                case 'spatial-stereo':
                    spatialVideoObject = await this.loadSpatialStereoVideo(videoConfig);
                    break;
                case 'volumetric-mesh':
                    spatialVideoObject = await this.loadVolumetricVideo(videoConfig);
                    break;
                case 'point-cloud':
                    spatialVideoObject = await this.loadPointCloudVideo(videoConfig);
                    break;
                case 'light-field':
                    spatialVideoObject = await this.loadLightFieldVideo(videoConfig);
                    break;
                default:
                    throw new Error(`Unsupported format: ${videoConfig.format}`);
            }
            
            this.spatialVideos.set(videoId, spatialVideoObject);
            this.spatialVideoContainer.add(spatialVideoObject.mesh);
            
            console.log('✅ Spatial video loaded:', videoId);
            return videoId;
            
        } catch (error) {
            console.error('Failed to load spatial video:', error);
            throw error;
        }
    }

    async loadSpatialStereoVideo(config) {
        // Load Apple Vision Pro spatial video format
        const video = document.createElement('video');
        video.src = config.url;
        video.crossOrigin = 'anonymous';
        video.loop = config.loop || false;
        
        const videoTexture = new THREE.VideoTexture(video);
        videoTexture.minFilter = THREE.LinearFilter;
        videoTexture.magFilter = THREE.LinearFilter;
        
        // Create stereo geometry
        const geometry = new THREE.SphereGeometry(10, 64, 32);
        
        // Modify UV mapping for stereo
        this.applyStereoUVMapping(geometry, config.stereoMode || 'side-by-side');
        
        const material = new THREE.MeshBasicMaterial({
            map: videoTexture,
            side: THREE.BackSide
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        return {
            type: 'spatial-stereo',
            video: video,
            texture: videoTexture,
            mesh: mesh,
            config: config
        };
    }

    async loadVolumetricVideo(config) {
        console.log('🎭 Loading volumetric video...');
        
        // Load volumetric mesh sequence
        const meshSequence = await this.loadMeshSequence(config.meshUrl);
        const textureSequence = await this.loadTextureSequence(config.textureUrl);
        
        const geometry = meshSequence[0]; // First frame
        const material = new THREE.MeshLambertMaterial({
            map: textureSequence[0]
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        return {
            type: 'volumetric-mesh',
            meshSequence: meshSequence,
            textureSequence: textureSequence,
            mesh: mesh,
            currentFrame: 0,
            frameCount: meshSequence.length,
            config: config
        };
    }

    async loadPointCloudVideo(config) {
        console.log('☁️ Loading point cloud video...');
        
        // Load point cloud sequence
        const pointCloudData = await this.loadPointCloudSequence(config.dataUrl);
        
        const geometry = new THREE.BufferGeometry();
        const positions = pointCloudData[0].positions;
        const colors = pointCloudData[0].colors;
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: config.pointSize || 0.01,
            vertexColors: true,
            sizeAttenuation: true
        });
        
        const pointCloud = new THREE.Points(geometry, material);
        
        return {
            type: 'point-cloud',
            pointCloudData: pointCloudData,
            mesh: pointCloud,
            currentFrame: 0,
            frameCount: pointCloudData.length,
            config: config
        };
    }

    async loadLightFieldVideo(config) {
        console.log('💡 Loading light field video...');
        
        // Load light field data
        const lightFieldData = await this.loadLightFieldData(config.lightFieldUrl);
        
        // Create light field renderer
        const lightFieldRenderer = this.createLightFieldRenderer(lightFieldData);
        
        return {
            type: 'light-field',
            lightFieldData: lightFieldData,
            renderer: lightFieldRenderer,
            mesh: lightFieldRenderer.mesh,
            config: config
        };
    }

    applyStereoUVMapping(geometry, stereoMode) {
        const uvAttribute = geometry.attributes.uv;
        const uvArray = uvAttribute.array;
        
        if (stereoMode === 'side-by-side') {
            // Map each eye to its respective half
            for (let i = 0; i < uvArray.length; i += 2) {
                const u = uvArray[i];
                const v = uvArray[i + 1];
                
                // For sphere geometry, map based on position
                if (u < 0.5) {
                    // Left eye
                    uvArray[i] = u * 2;
                } else {
                    // Right eye
                    uvArray[i] = (u - 0.5) * 2;
                }
            }
        } else if (stereoMode === 'over-under') {
            // Top-bottom stereo format
            for (let i = 0; i < uvArray.length; i += 2) {
                const u = uvArray[i];
                const v = uvArray[i + 1];
                
                if (v > 0.5) {
                    // Top half (left eye)
                    uvArray[i + 1] = (v - 0.5) * 2;
                } else {
                    // Bottom half (right eye)
                    uvArray[i + 1] = v * 2;
                }
            }
        }
        
        uvAttribute.needsUpdate = true;
    }

    play(videoId) {
        const spatialVideo = this.spatialVideos.get(videoId);
        if (!spatialVideo) return false;
        
        switch (spatialVideo.type) {
            case 'spatial-stereo':
                spatialVideo.video.play();
                break;
            case 'volumetric-mesh':
            case 'point-cloud':
                this.startFrameAnimation(spatialVideo);
                break;
            case 'light-field':
                spatialVideo.renderer.play();
                break;
        }
        
        this.volumetricPlayer.isPlaying = true;
        console.log('▶️ Playing spatial video:', videoId);
        return true;
    }

    pause(videoId) {
        const spatialVideo = this.spatialVideos.get(videoId);
        if (!spatialVideo) return false;
        
        switch (spatialVideo.type) {
            case 'spatial-stereo':
                spatialVideo.video.pause();
                break;
            case 'volumetric-mesh':
            case 'point-cloud':
                this.stopFrameAnimation(spatialVideo);
                break;
            case 'light-field':
                spatialVideo.renderer.pause();
                break;
        }
        
        this.volumetricPlayer.isPlaying = false;
        console.log('⏸️ Paused spatial video:', videoId);
        return true;
    }

    togglePlayback() {
        if (this.volumetricPlayer.isPlaying) {
            this.pauseAll();
        } else {
            this.playAll();
        }
    }

    playAll() {
        this.spatialVideos.forEach((video, id) => {
            this.play(id);
        });
    }

    pauseAll() {
        this.spatialVideos.forEach((video, id) => {
            this.pause(id);
        });
    }

    adjustVolume(delta) {
        this.spatialAudioSystem.positionalSounds.forEach(sound => {
            const newVolume = Math.max(0, Math.min(1, sound.getVolume() + delta));
            sound.setVolume(newVolume);
        });
        
        // Update volume slider UI
        if (this.spatialControls.ui) {
            const volumeSlider = this.spatialControls.ui.children.find(
                child => child.userData.isSlider
            );
            if (volumeSlider) {
                volumeSlider.userData.value = this.getAverageVolume();
                this.updateVolumeSliderUI(volumeSlider);
            }
        }
    }

    seek(direction) {
        const seekAmount = direction === 'forward' ? 10 : -10; // seconds
        
        this.spatialVideos.forEach(video => {
            if (video.type === 'spatial-stereo' && video.video) {
                video.video.currentTime += seekAmount;
            }
        });
    }

    switchQuality(qualityName) {
        console.log(`📺 Switching to quality: ${qualityName}`);
        
        this.adaptiveStreaming.currentQuality = qualityName;
        
        // Reload videos with new quality
        this.spatialVideos.forEach((video, id) => {
            if (video.config.qualityUrls && video.config.qualityUrls[qualityName]) {
                this.reloadVideoWithQuality(id, qualityName);
            }
        });
    }

    async reloadVideoWithQuality(videoId, quality) {
        const video = this.spatialVideos.get(videoId);
        if (!video) return;
        
        const newUrl = video.config.qualityUrls[quality];
        if (newUrl && video.video) {
            const currentTime = video.video.currentTime;
            video.video.src = newUrl;
            video.video.currentTime = currentTime;
        }
    }

    updateSpatialVideo(frame) {
        if (!webXRManager.xrSession) return;
        
        // Update all spatial videos
        this.spatialVideos.forEach((video, id) => {
            this.updateVideoFrame(video);
        });
        
        // Update 3D controls
        this.updateSpatialControls();
        
        // Update audio spatialization
        this.updateSpatialAudio();
    }

    updateVideoFrame(video) {
        switch (video.type) {
            case 'volumetric-mesh':
                this.updateVolumetricFrame(video);
                break;
            case 'point-cloud':
                this.updatePointCloudFrame(video);
                break;
            case 'light-field':
                this.updateLightFieldFrame(video);
                break;
        }
    }

    updateVolumetricFrame(video) {
        if (!this.volumetricPlayer.isPlaying) return;
        
        const frameIndex = Math.floor(this.volumetricPlayer.currentTime * 30) % video.frameCount;
        
        if (frameIndex !== video.currentFrame) {
            // Update geometry and texture
            video.mesh.geometry = video.meshSequence[frameIndex];
            video.mesh.material.map = video.textureSequence[frameIndex];
            video.currentFrame = frameIndex;
        }
    }

    updatePointCloudFrame(video) {
        if (!this.volumetricPlayer.isPlaying) return;
        
        const frameIndex = Math.floor(this.volumetricPlayer.currentTime * 30) % video.frameCount;
        const pointCloudFrame = video.pointCloudData[frameIndex];
        
        // Update point positions and colors
        video.mesh.geometry.setAttribute(
            'position', 
            new THREE.Float32BufferAttribute(pointCloudFrame.positions, 3)
        );
        video.mesh.geometry.setAttribute(
            'color', 
            new THREE.Float32BufferAttribute(pointCloudFrame.colors, 3)
        );
        
        video.mesh.geometry.attributes.position.needsUpdate = true;
        video.mesh.geometry.attributes.color.needsUpdate = true;
    }

    updateSpatialControls() {
        // Update progress bar
        if (this.spatialControls.ui) {
            const progressBar = this.spatialControls.ui.children.find(
                child => child.userData.isProgressBar
            );
            if (progressBar) {
                this.updateProgressBarUI(progressBar);
            }
        }
    }

    updateProgressBarUI(progressBar) {
        const progress = progressBar.userData.progress;
        const duration = this.getMaxDuration();
        const currentTime = this.getCurrentTime();
        
        if (duration > 0) {
            const progressRatio = currentTime / duration;
            progress.scale.x = progressRatio;
            progress.position.x = -1.5 + (progressRatio * 1.5);
        }
    }

    updateVolumeSliderUI(volumeSlider) {
        const handle = volumeSlider.userData.handle;
        const value = volumeSlider.userData.value;
        handle.position.x = (value - 0.5) * 1; // Map 0-1 to -0.5 to 0.5
    }

    updateSpatialAudio() {
        // Update positional audio based on head position
        const headPosition = webXRManager.camera.position;
        
        this.spatialAudioSystem.positionalSounds.forEach((sound, id) => {
            // Update 3D audio positioning
            sound.updateMatrixWorld();
        });
    }

    // Utility methods
    getCurrentTime() {
        // Get current playback time from active videos
        for (const video of this.spatialVideos.values()) {
            if (video.type === 'spatial-stereo' && video.video) {
                return video.video.currentTime;
            }
        }
        return this.volumetricPlayer.currentTime;
    }

    getMaxDuration() {
        let maxDuration = 0;
        for (const video of this.spatialVideos.values()) {
            if (video.type === 'spatial-stereo' && video.video) {
                maxDuration = Math.max(maxDuration, video.video.duration || 0);
            }
        }
        return maxDuration || this.volumetricPlayer.duration;
    }

    getAverageVolume() {
        let totalVolume = 0;
        let count = 0;
        
        this.spatialAudioSystem.positionalSounds.forEach(sound => {
            totalVolume += sound.getVolume();
            count++;
        });
        
        return count > 0 ? totalVolume / count : 0.8;
    }

    createTextTexture(text, size = 64, color = '#ffffff') {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = size * 2;
        canvas.height = size;
        
        context.fillStyle = 'transparent';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = color;
        context.font = `${size/2}px Arial`;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width/2, canvas.height/2);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    // Mock data loaders (replace with actual implementations)
    async loadMeshSequence(url) {
        console.log('Loading mesh sequence from:', url);
        // Return mock mesh sequence
        return [new THREE.BoxGeometry(1, 1, 1)];
    }

    async loadTextureSequence(url) {
        console.log('Loading texture sequence from:', url);
        // Return mock texture sequence
        return [new THREE.TextureLoader().load('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')];
    }

    async loadPointCloudSequence(url) {
        console.log('Loading point cloud sequence from:', url);
        // Return mock point cloud data
        return [{
            positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        }];
    }

    async loadLightFieldData(url) {
        console.log('Loading light field data from:', url);
        return { views: [], lightField: null };
    }

    createLightFieldRenderer(data) {
        return {
            mesh: new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.MeshBasicMaterial()),
            play: () => console.log('Light field playing'),
            pause: () => console.log('Light field paused')
        };
    }

    startFrameAnimation(video) {
        video.animationId = setInterval(() => {
            this.volumetricPlayer.currentTime += 1/30; // 30 FPS
        }, 1000/30);
    }

    stopFrameAnimation(video) {
        if (video.animationId) {
            clearInterval(video.animationId);
            video.animationId = null;
        }
    }

    handleButtonPress(action) {
        console.log('Button pressed:', action);
        
        // Trigger haptic feedback
        webXRManager.pulseHaptic(0.5, 100);
        
        switch (action) {
            case 'play-pause':
                this.togglePlayback();
                break;
            default:
                if (action.startsWith('quality-')) {
                    const quality = action.replace('quality-', '');
                    this.switchQuality(quality);
                }
                break;
        }
    }
}

// Export Spatial Video class
export default SpatialVideo;