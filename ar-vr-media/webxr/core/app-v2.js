/**
 * Enhanced AR/VR Media Platform App V2
 * Production-ready WebXR application with full feature integration
 */

import * as THREE from 'three';
import { webXRManager } from './webxr-manager-v2.js';
import { handTracking } from '../hand-tracking/real-hand-tracking.js';
import { spatialVideoPlayer } from '../spatial-video/real-spatial-video.js';
import { mixedReality } from '../mixed-reality/real-mixed-reality.js';
import { ModelGenerator, AnimationHelpers } from '../../assets/3d-models.js';

class ARVRMediaAppV2 {
    constructor() {
        this.isInitialized = false;
        this.currentDemo = null;
        this.demos = new Map();
        
        // Feature flags based on device capabilities
        this.features = {
            handTracking: false,
            eyeTracking: false,
            spatialVideo: false,
            mixedReality: false,
            haptics: false
        };
        
        // Demo objects
        this.demoObjects = new Map();
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Enhanced AR/VR Media Platform V2...');
        
        try {
            // Wait for WebXR manager
            await this.waitForWebXR();
            
            // Check feature support
            this.checkFeatureSupport();
            
            // Setup UI
            this.setupUI();
            
            // Setup demos
            this.setupDemos();
            
            // Setup event handlers
            this.setupEventHandlers();
            
            // Create initial scene
            this.createInitialScene();
            
            this.isInitialized = true;
            console.log('✅ Platform initialized successfully');
            
            // Show platform info
            this.showPlatformInfo();
            
        } catch (error) {
            console.error('❌ Failed to initialize platform:', error);
            this.showError('Initialization Failed', error.message);
        }
    }

    async waitForWebXR() {
        // Wait for WebXR manager to be ready
        return new Promise((resolve) => {
            const check = () => {
                if (webXRManager && webXRManager.features) {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    }

    checkFeatureSupport() {
        const features = webXRManager.getSupportedFeatures();
        const platform = webXRManager.getPlatformInfo();
        
        this.features = {
            handTracking: features.handTracking || platform.platform.isVisionPro,
            eyeTracking: features.eyeTracking || platform.platform.isVisionPro,
            spatialVideo: true, // Always available
            mixedReality: features.ar || features.vr,
            haptics: features.vr || features.ar
        };
        
        console.log('📋 Feature support:', this.features);
    }

    setupUI() {
        // Update UI based on feature support
        this.updateFeatureButtons();
        
        // Add XR enter button
        this.createXRButton();
        
        // Add info panel
        this.createInfoPanel();
    }

    updateFeatureButtons() {
        // Enable/disable buttons based on support
        Object.entries(this.features).forEach(([feature, supported]) => {
            const button = document.querySelector(`[data-feature="${feature}"]`);
            if (button) {
                button.disabled = !supported;
                button.classList.toggle('supported', supported);
                button.classList.toggle('unsupported', !supported);
            }
        });
    }

    createXRButton() {
        const xrButton = document.createElement('button');
        xrButton.id = 'xr-button';
        xrButton.className = 'xr-button';
        xrButton.innerHTML = '🥽 Enter VR';
        
        xrButton.addEventListener('click', () => this.toggleXR());
        
        document.body.appendChild(xrButton);
        
        // Update button based on support
        if (webXRManager.features.vr) {
            xrButton.innerHTML = '🥽 Enter VR';
        } else if (webXRManager.features.ar) {
            xrButton.innerHTML = '📱 Enter AR';
        } else {
            xrButton.innerHTML = '❌ WebXR Not Supported';
            xrButton.disabled = true;
        }
    }

    createInfoPanel() {
        const panel = document.createElement('div');
        panel.id = 'info-panel';
        panel.className = 'info-panel';
        panel.innerHTML = `
            <h3>Platform Info</h3>
            <div id="platform-name"></div>
            <div id="session-info"></div>
            <div id="performance-info"></div>
        `;
        
        document.body.appendChild(panel);
    }

    setupDemos() {
        // Register all demos
        this.demos.set('handTracking', {
            name: 'Hand Tracking Demo',
            setup: () => this.setupHandTrackingDemo(),
            cleanup: () => this.cleanupHandTrackingDemo()
        });
        
        this.demos.set('spatialVideo', {
            name: 'Spatial Video Cinema',
            setup: () => this.setupSpatialVideoDemo(),
            cleanup: () => this.cleanupSpatialVideoDemo()
        });
        
        this.demos.set('mixedReality', {
            name: 'Mixed Reality Space',
            setup: () => this.setupMixedRealityDemo(),
            cleanup: () => this.cleanupMixedRealityDemo()
        });
        
        this.demos.set('haptics', {
            name: 'Haptic Feedback Demo',
            setup: () => this.setupHapticsDemo(),
            cleanup: () => this.cleanupHapticsDemo()
        });
        
        this.demos.set('portal', {
            name: 'AR Portal Demo',
            setup: () => this.setupPortalDemo(),
            cleanup: () => this.cleanupPortalDemo()
        });
    }

    setupEventHandlers() {
        // Feature buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-feature]')) {
                const feature = e.target.dataset.feature;
                this.activateDemo(feature);
            }
        });
        
        // WebXR events
        window.addEventListener('xr-session-started', (e) => this.onXRSessionStarted(e));
        window.addEventListener('xr-session-ended', () => this.onXRSessionEnded());
        
        // Hand tracking events
        window.addEventListener('hand-gesture-start', (e) => this.onGestureStart(e));
        window.addEventListener('hand-gesture-end', (e) => this.onGestureEnd(e));
        
        // Spatial video events
        window.addEventListener('spatial-video-loaded', (e) => this.onVideoLoaded(e));
        
        // AR events
        window.addEventListener('ar-plane-detected', (e) => this.onPlaneDetected(e));
        window.addEventListener('ar-anchor-created', (e) => this.onAnchorCreated(e));
        
        // Performance monitoring
        setInterval(() => this.updatePerformanceInfo(), 1000);
    }

    createInitialScene() {
        // Add ambient lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        webXRManager.scene.add(ambientLight);
        
        // Add directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.far = 20;
        directionalLight.shadow.mapSize.set(2048, 2048);
        webXRManager.scene.add(directionalLight);
        
        // Add floor
        const floorGeometry = new THREE.PlaneGeometry(10, 10);
        const floorMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x808080,
            side: THREE.DoubleSide
        });
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        webXRManager.scene.add(floor);
        
        // Add welcome text (in real app, use text geometry or sprites)
        const welcomeBoard = ModelGenerator.createVirtualMonitor({
            width: 2,
            height: 1,
            screenColor: 0x111111
        });
        welcomeBoard.position.set(0, 1.5, -3);
        webXRManager.scene.add(welcomeBoard);
        
        // Store for cleanup
        this.demoObjects.set('initial', [ambientLight, directionalLight, floor, welcomeBoard]);
    }

    async toggleXR() {
        if (webXRManager.isSessionActive()) {
            webXRManager.endSession();
        } else {
            try {
                if (webXRManager.features.vr) {
                    await this.enterVR();
                } else if (webXRManager.features.ar) {
                    await this.enterAR();
                }
            } catch (error) {
                console.error('Failed to start XR session:', error);
                this.showError('XR Error', error.message);
            }
        }
    }

    async enterVR() {
        const requiredFeatures = [];
        const optionalFeatures = [];
        
        // Add features based on active demo
        if (this.currentDemo === 'handTracking') {
            optionalFeatures.push('hand-tracking');
        }
        
        await webXRManager.startVR(requiredFeatures, optionalFeatures);
    }

    async enterAR() {
        const requiredFeatures = ['hit-test'];
        const optionalFeatures = ['plane-detection', 'anchors'];
        
        await webXRManager.startAR(requiredFeatures, optionalFeatures);
    }

    activateDemo(demoName) {
        // Cleanup current demo
        if (this.currentDemo) {
            const currentDemoConfig = this.demos.get(this.currentDemo);
            if (currentDemoConfig && currentDemoConfig.cleanup) {
                currentDemoConfig.cleanup();
            }
        }
        
        // Setup new demo
        const demo = this.demos.get(demoName);
        if (demo && demo.setup) {
            console.log(`🎯 Activating demo: ${demo.name}`);
            demo.setup();
            this.currentDemo = demoName;
            
            // Update UI
            document.querySelectorAll('[data-feature]').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.feature === demoName);
            });
        }
    }

    // Demo implementations

    setupHandTrackingDemo() {
        console.log('👋 Setting up hand tracking demo');
        
        // Create hand interaction targets
        const targets = ModelGenerator.createHandTrackingDemo();
        targets.position.set(0, 1.5, -2);
        webXRManager.scene.add(targets);
        
        // Add floating crystals
        for (let i = 0; i < 5; i++) {
            const crystal = ModelGenerator.createFloatingCrystal({
                size: 0.1 + Math.random() * 0.1,
                color: new THREE.Color().setHSL(Math.random(), 0.8, 0.5)
            });
            
            crystal.position.set(
                (Math.random() - 0.5) * 3,
                1 + Math.random() * 0.5,
                -3 + (Math.random() - 0.5) * 2
            );
            
            AnimationHelpers.animateFloat(crystal, 0.1, crystal.userData.floatSpeed);
            AnimationHelpers.animateRotation(crystal, { x: 0, y: crystal.userData.rotationSpeed, z: 0 });
            
            webXRManager.scene.add(crystal);
            handTracking.addInteractableObject(crystal);
            
            // Store for cleanup
            if (!this.demoObjects.has('handTracking')) {
                this.demoObjects.set('handTracking', []);
            }
            this.demoObjects.get('handTracking').push(crystal);
        }
        
        this.demoObjects.get('handTracking').push(targets);
        
        // Make targets interactable
        targets.children.forEach(target => {
            handTracking.addInteractableObject(target);
            
            // Add interaction handlers
            target.addEventListener('hover-start', () => {
                target.scale.setScalar(1.2);
                target.material.emissiveIntensity = 0.5;
            });
            
            target.addEventListener('hover-end', () => {
                target.scale.setScalar(1);
                target.material.emissiveIntensity = 0.2;
            });
            
            target.addEventListener('select-start', () => {
                target.userData.isActivated = !target.userData.isActivated;
                target.material.emissive = target.userData.isActivated ? 
                    new THREE.Color(0xffffff) : 
                    new THREE.Color(target.userData.originalColor);
            });
        });
        
        // Show hand visualization
        handTracking.setVisualizationOptions({
            showJoints: true,
            showBones: true,
            jointScale: 1.2
        });
    }

    cleanupHandTrackingDemo() {
        const objects = this.demoObjects.get('handTracking');
        if (objects) {
            objects.forEach(obj => {
                webXRManager.scene.remove(obj);
                handTracking.removeInteractableObject(obj);
            });
            this.demoObjects.delete('handTracking');
        }
    }

    async setupSpatialVideoDemo() {
        console.log('📹 Setting up spatial video demo');
        
        // Load a sample video
        try {
            await spatialVideoPlayer.loadVideo({
                url: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
                format: spatialVideoPlayer.formats.FLAT_2D,
                autoplay: false,
                screen: 'cinema'
            });
            
            // Create cinema environment
            const cinemaFloor = new THREE.Mesh(
                new THREE.PlaneGeometry(20, 20),
                new THREE.MeshPhongMaterial({ 
                    color: 0x1a0f08,
                    side: THREE.DoubleSide
                })
            );
            cinemaFloor.rotation.x = -Math.PI / 2;
            cinemaFloor.receiveShadow = true;
            webXRManager.scene.add(cinemaFloor);
            
            // Cinema walls
            const wallMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x2a1810,
                side: THREE.DoubleSide
            });
            
            const backWall = new THREE.Mesh(
                new THREE.PlaneGeometry(20, 12),
                wallMaterial
            );
            backWall.position.set(0, 6, -10);
            webXRManager.scene.add(backWall);
            
            // Add play button
            const playButton = ModelGenerator.createInteractiveButton({
                text: 'Play Video',
                color: 0x00ff00
            });
            playButton.position.set(0, 1, -1);
            webXRManager.scene.add(playButton);
            
            handTracking.addInteractableObject(playButton.children[0]);
            
            playButton.children[0].addEventListener('select-start', () => {
                if (spatialVideoPlayer.isPlaying()) {
                    spatialVideoPlayer.pause();
                    playButton.children[0].material.color.setHex(0x00ff00);
                } else {
                    spatialVideoPlayer.play();
                    playButton.children[0].material.color.setHex(0xff0000);
                }
            });
            
            // Store for cleanup
            this.demoObjects.set('spatialVideo', [cinemaFloor, backWall, playButton]);
            
        } catch (error) {
            console.error('Failed to load video:', error);
            this.showError('Video Error', 'Failed to load sample video');
        }
    }

    cleanupSpatialVideoDemo() {
        spatialVideoPlayer.pause();
        
        const objects = this.demoObjects.get('spatialVideo');
        if (objects) {
            objects.forEach(obj => {
                webXRManager.scene.remove(obj);
                handTracking.removeInteractableObject(obj);
            });
            this.demoObjects.delete('spatialVideo');
        }
    }

    setupMixedRealityDemo() {
        console.log('🌍 Setting up mixed reality demo');
        
        // Enable passthrough if in VR mode
        if (webXRManager.isSessionActive() && webXRManager.sessionConfig.mode === 'immersive-vr') {
            mixedReality.enablePassthrough();
        }
        
        // Create AR content
        const arContent = new THREE.Group();
        
        // Virtual monitor
        const monitor = ModelGenerator.createVirtualMonitor();
        monitor.position.set(0, 1, -2);
        arContent.add(monitor);
        
        // AR Portal
        const portal = ModelGenerator.createARPortal({
            radius: 0.8,
            color: 0x00ffff,
            destination: 'space'
        });
        portal.position.set(2, 1, -2);
        AnimationHelpers.animatePortal(portal);
        arContent.add(portal);
        
        // Floating UI panels
        for (let i = 0; i < 3; i++) {
            const panel = new THREE.Mesh(
                new THREE.PlaneGeometry(0.5, 0.3),
                new THREE.MeshBasicMaterial({
                    color: 0x4a90e2,
                    transparent: true,
                    opacity: 0.8
                })
            );
            
            panel.position.set(
                -1.5 + i * 0.6,
                1.5,
                -1.5
            );
            
            handTracking.addInteractableObject(panel);
            arContent.add(panel);
        }
        
        webXRManager.scene.add(arContent);
        
        // If in AR mode, place objects with plane detection
        if (webXRManager.sessionConfig.mode === 'immersive-ar') {
            // Objects will snap to detected planes
            arContent.children.forEach(child => {
                mixedReality.placeObject(child, child.position, {
                    snapToPlane: true,
                    createAnchor: true
                });
            });
        }
        
        this.demoObjects.set('mixedReality', [arContent]);
        
        // Show plane visualization
        mixedReality.setShowPlanes(true);
    }

    cleanupMixedRealityDemo() {
        const objects = this.demoObjects.get('mixedReality');
        if (objects) {
            objects.forEach(obj => {
                webXRManager.scene.remove(obj);
            });
            this.demoObjects.delete('mixedReality');
        }
        
        mixedReality.setShowPlanes(false);
    }

    setupHapticsDemo() {
        console.log('🤝 Setting up haptics demo');
        
        // Create haptic test objects
        const hapticDemo = ModelGenerator.createHapticDemoObject();
        hapticDemo.position.set(0, 1, -2);
        webXRManager.scene.add(hapticDemo);
        
        // Make objects interactable
        hapticDemo.children.forEach(obj => {
            handTracking.addInteractableObject(obj);
            
            obj.addEventListener('hover-start', (e) => {
                const pattern = obj.userData.hapticPattern;
                webXRManager.vibrate(e.handedness, pattern.intensity, pattern.duration);
                
                // Visual feedback
                obj.material.emissiveIntensity = 0.3;
            });
            
            obj.addEventListener('hover-end', () => {
                obj.material.emissiveIntensity = 0;
            });
            
            obj.addEventListener('select-start', (e) => {
                const pattern = obj.userData.hapticPattern;
                
                // Stronger haptic for selection
                webXRManager.vibrate(e.handedness, pattern.intensity * 1.5, pattern.duration * 2);
                
                // Rotate object
                obj.rotation.y += Math.PI / 4;
            });
        });
        
        this.demoObjects.set('haptics', [hapticDemo]);
    }

    cleanupHapticsDemo() {
        const objects = this.demoObjects.get('haptics');
        if (objects) {
            objects.forEach(obj => {
                webXRManager.scene.remove(obj);
            });
            this.demoObjects.delete('haptics');
        }
    }

    setupPortalDemo() {
        console.log('🌀 Setting up portal demo');
        
        // Create main portal
        const portal = ModelGenerator.createARPortal({
            radius: 1.5,
            color: 0xff00ff,
            destination: 'space'
        });
        portal.position.set(0, 1.5, -3);
        AnimationHelpers.animatePortal(portal);
        webXRManager.scene.add(portal);
        
        // Create space environment (hidden initially)
        const spaceEnv = ModelGenerator.createSpaceEnvironment();
        spaceEnv.visible = false;
        webXRManager.scene.add(spaceEnv);
        AnimationHelpers.animateSpace(spaceEnv);
        
        // Portal interaction
        const portalRing = portal.children[0];
        handTracking.addInteractableObject(portalRing);
        
        portalRing.addEventListener('select-start', () => {
            // Toggle between environments
            const inSpace = spaceEnv.visible;
            
            if (!inSpace) {
                // Enter space
                this.demoObjects.get('initial')?.forEach(obj => obj.visible = false);
                spaceEnv.visible = true;
                
                // Move camera
                webXRManager.cameraGroup.position.set(0, 0, 0);
            } else {
                // Exit space
                this.demoObjects.get('initial')?.forEach(obj => obj.visible = true);
                spaceEnv.visible = false;
            }
            
            // Haptic feedback
            webXRManager.vibrate('left', 1.0, 200);
            webXRManager.vibrate('right', 1.0, 200);
        });
        
        this.demoObjects.set('portal', [portal, spaceEnv]);
    }

    cleanupPortalDemo() {
        const objects = this.demoObjects.get('portal');
        if (objects) {
            objects.forEach(obj => {
                webXRManager.scene.remove(obj);
            });
            this.demoObjects.delete('portal');
        }
        
        // Restore initial scene visibility
        this.demoObjects.get('initial')?.forEach(obj => obj.visible = true);
    }

    // Event handlers

    onXRSessionStarted(event) {
        console.log('🥽 XR Session started');
        
        const button = document.getElementById('xr-button');
        if (button) {
            button.innerHTML = '❌ Exit XR';
        }
        
        // Update info
        const sessionInfo = document.getElementById('session-info');
        if (sessionInfo) {
            sessionInfo.innerHTML = `Mode: ${event.detail.mode}`;
        }
        
        // Auto-activate appropriate demo
        if (event.detail.mode === 'immersive-ar') {
            this.activateDemo('mixedReality');
        } else if (this.features.handTracking) {
            this.activateDemo('handTracking');
        }
    }

    onXRSessionEnded() {
        console.log('📱 XR Session ended');
        
        const button = document.getElementById('xr-button');
        if (button) {
            button.innerHTML = webXRManager.features.vr ? '🥽 Enter VR' : '📱 Enter AR';
        }
        
        // Update info
        const sessionInfo = document.getElementById('session-info');
        if (sessionInfo) {
            sessionInfo.innerHTML = 'No active session';
        }
    }

    onGestureStart(event) {
        const { handedness, gesture } = event.detail;
        console.log(`Gesture started: ${gesture.type} (${handedness})`);
    }

    onGestureEnd(event) {
        const { handedness, gesture } = event.detail;
        console.log(`Gesture ended: ${gesture.type} (${handedness})`);
    }

    onVideoLoaded(event) {
        const { metadata } = event.detail;
        console.log('Video loaded:', metadata);
    }

    onPlaneDetected(event) {
        const { plane } = event.detail;
        console.log('Plane detected:', plane.orientation);
    }

    onAnchorCreated(event) {
        console.log('Anchor created:', event.detail);
    }

    updatePerformanceInfo() {
        const perfInfo = document.getElementById('performance-info');
        if (!perfInfo || !webXRManager.isSessionActive()) return;
        
        const stats = webXRManager.getPerformanceStats();
        perfInfo.innerHTML = `FPS: ${stats.fps} | Frame: ${stats.frameTime.toFixed(1)}ms`;
    }

    showPlatformInfo() {
        const platform = webXRManager.getPlatformInfo();
        const platformName = document.getElementById('platform-name');
        
        if (platformName) {
            platformName.innerHTML = `Device: ${platform.name}`;
        }
        
        console.log(`
╔════════════════════════════════════════╗
║     AR/VR Immersive Media Platform     ║
║              Version 2.0               ║
╠════════════════════════════════════════╣
║ Device: ${platform.name.padEnd(30)} ║
║ WebXR: ${(webXRManager.features.vr || webXRManager.features.ar ? '✅' : '❌').padEnd(31)} ║
║ Hand Tracking: ${(this.features.handTracking ? '✅' : '❌').padEnd(23)} ║
║ Eye Tracking: ${(this.features.eyeTracking ? '✅' : '❌').padEnd(24)} ║
║ Mixed Reality: ${(this.features.mixedReality ? '✅' : '❌').padEnd(23)} ║
╚════════════════════════════════════════╝
        `);
    }

    showError(title, message) {
        console.error(`${title}: ${message}`);
        
        // Create error toast
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.innerHTML = `
            <strong>${title}</strong>
            <p>${message}</p>
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Add styles
    const style = document.createElement('style');
    style.textContent = `
        .xr-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px 30px;
            font-size: 18px;
            font-weight: bold;
            background: #4a90e2;
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            z-index: 1000;
            transition: all 0.3s ease;
        }
        
        .xr-button:hover {
            background: #357abd;
            transform: scale(1.05);
        }
        
        .xr-button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        
        .info-panel {
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-family: monospace;
            z-index: 1000;
        }
        
        .error-toast {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ff4444;
            color: white;
            padding: 15px;
            border-radius: 10px;
            max-width: 300px;
            z-index: 10000;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        [data-feature] {
            transition: all 0.3s ease;
        }
        
        [data-feature].active {
            background: #4a90e2;
            color: white;
        }
        
        [data-feature].unsupported {
            opacity: 0.5;
            cursor: not-allowed;
        }
    `;
    document.head.appendChild(style);
    
    // Create app instance
    window.arvrApp = new ARVRMediaAppV2();
});

export default ARVRMediaAppV2;