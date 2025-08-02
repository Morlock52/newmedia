/**
 * Enhanced WebXR Manager V2 - Production-ready WebXR implementation
 * Full support for Apple Vision Pro, Meta Quest 3, and spatial computing devices
 * Implements latest WebXR Device API features (2025)
 */

import * as THREE from 'three';
import WebXRPolyfill from 'webxr-polyfill';

// Initialize WebXR polyfill for broader compatibility
if (!('xr' in navigator)) {
    const polyfill = new WebXRPolyfill();
}

export class WebXRManagerV2 {
    constructor() {
        // Core WebXR properties
        this.xrSession = null;
        this.xrRefSpace = null;
        this.xrViewerSpace = null;
        this.xrHitTestSource = null;
        this.gl = null;
        this.baseLayer = null;
        
        // Three.js components
        this.renderer = null;
        this.scene = null;
        this.camera = null;
        this.cameraGroup = null;
        
        // Platform detection
        this.platform = {
            isVisionPro: false,
            isQuest: false,
            isPico: false,
            isLynx: false,
            isMagicLeap: false
        };
        
        // Feature support flags
        this.features = {
            vr: false,
            ar: false,
            handTracking: false,
            eyeTracking: false,
            planeDetection: false,
            meshDetection: false,
            anchors: false,
            depthSensing: false,
            lightEstimation: false,
            domOverlay: false,
            layers: false,
            secondaryViews: false
        };
        
        // Input sources
        this.inputSources = new Map();
        this.transientInputSources = new Map();
        this.controllers = [];
        this.hands = new Map();
        
        // Frame data
        this.xrFrame = null;
        this.xrPose = null;
        
        // Session configuration
        this.sessionConfig = {
            mode: null,
            requiredFeatures: [],
            optionalFeatures: []
        };
        
        // Event handlers
        this.eventHandlers = new Map();
        
        // Performance monitoring
        this.performance = {
            fps: 0,
            frameTime: 0,
            droppedFrames: 0
        };
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Enhanced WebXR Manager V2...');
        
        try {
            // Detect platform
            this.detectPlatform();
            
            // Check WebXR support
            await this.checkWebXRSupport();
            
            // Initialize Three.js
            this.initializeThreeJS();
            
            // Setup WebXR event handlers
            this.setupWebXREventHandlers();
            
            console.log('✅ WebXR Manager V2 initialized successfully');
            console.log('📱 Platform:', this.getPlatformName());
            console.log('🎯 Features:', this.features);
            
        } catch (error) {
            console.error('❌ WebXR Manager initialization failed:', error);
            throw error;
        }
    }

    detectPlatform() {
        const ua = navigator.userAgent.toLowerCase();
        
        // Apple Vision Pro detection
        if (ua.includes('visionos') || ua.includes('vision pro') || 
            (ua.includes('safari') && 'xr' in navigator && ua.includes('cpu os'))) {
            this.platform.isVisionPro = true;
        }
        
        // Meta Quest detection
        if (ua.includes('quest') || ua.includes('oculus')) {
            this.platform.isQuest = true;
        }
        
        // Pico detection
        if (ua.includes('pico')) {
            this.platform.isPico = true;
        }
        
        // Magic Leap detection
        if (ua.includes('magic leap') || ua.includes('ml1')) {
            this.platform.isMagicLeap = true;
        }
        
        // Lynx detection
        if (ua.includes('lynx')) {
            this.platform.isLynx = true;
        }
    }

    getPlatformName() {
        if (this.platform.isVisionPro) return 'Apple Vision Pro';
        if (this.platform.isQuest) return 'Meta Quest';
        if (this.platform.isPico) return 'Pico';
        if (this.platform.isMagicLeap) return 'Magic Leap';
        if (this.platform.isLynx) return 'Lynx';
        return 'Unknown XR Device';
    }

    async checkWebXRSupport() {
        if (!('xr' in navigator)) {
            console.warn('WebXR not supported in this browser');
            return;
        }

        // Check VR support
        try {
            this.features.vr = await navigator.xr.isSessionSupported('immersive-vr');
        } catch (e) {
            console.warn('VR support check failed:', e);
        }

        // Check AR support
        try {
            this.features.ar = await navigator.xr.isSessionSupported('immersive-ar');
        } catch (e) {
            console.warn('AR support check failed:', e);
        }

        // Check for specific features
        await this.checkFeatureSupport();
    }

    async checkFeatureSupport() {
        const featuresToCheck = [
            { name: 'hand-tracking', flag: 'handTracking' },
            { name: 'plane-detection', flag: 'planeDetection' },
            { name: 'mesh-detection', flag: 'meshDetection' },
            { name: 'anchors', flag: 'anchors' },
            { name: 'depth-sensing', flag: 'depthSensing' },
            { name: 'light-estimation', flag: 'lightEstimation' },
            { name: 'dom-overlay', flag: 'domOverlay' },
            { name: 'layers', flag: 'layers' },
            { name: 'secondary-views', flag: 'secondaryViews' }
        ];

        // Special handling for eye tracking (not standard yet)
        if (this.platform.isVisionPro) {
            this.features.eyeTracking = true; // Via transient-pointer
        }

        // Test each feature
        for (const feature of featuresToCheck) {
            try {
                // Try with VR first
                if (this.features.vr) {
                    const vrSupported = await this.isFeatureSupported('immersive-vr', feature.name);
                    if (vrSupported) {
                        this.features[feature.flag] = true;
                        continue;
                    }
                }
                
                // Try with AR
                if (this.features.ar) {
                    const arSupported = await this.isFeatureSupported('immersive-ar', feature.name);
                    if (arSupported) {
                        this.features[feature.flag] = true;
                    }
                }
            } catch (e) {
                console.debug(`Feature ${feature.name} check failed:`, e);
            }
        }
    }

    async isFeatureSupported(mode, feature) {
        try {
            return await navigator.xr.isSessionSupported(mode, {
                optionalFeatures: [feature]
            });
        } catch (e) {
            return false;
        }
    }

    initializeThreeJS() {
        // Create canvas if not exists
        this.canvas = document.getElementById('webxr-canvas');
        if (!this.canvas) {
            this.canvas = document.createElement('canvas');
            this.canvas.id = 'webxr-canvas';
            document.body.appendChild(this.canvas);
        }

        // Initialize renderer with WebXR support
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: false,
            powerPreference: 'high-performance'
        });
        
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.xr.enabled = true;
        
        // Configure shadows for better quality
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x000000, 0.00025);
        
        // Create camera group for XR
        this.cameraGroup = new THREE.Group();
        this.cameraGroup.name = 'CameraGroup';
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            70,
            window.innerWidth / window.innerHeight,
            0.01,
            1000
        );
        this.cameraGroup.add(this.camera);
        this.scene.add(this.cameraGroup);
        
        // Add default lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = 50;
        directionalLight.shadow.camera.left = -10;
        directionalLight.shadow.camera.right = 10;
        directionalLight.shadow.camera.top = 10;
        directionalLight.shadow.camera.bottom = -10;
        directionalLight.shadow.mapSize.set(2048, 2048);
        this.scene.add(directionalLight);
        
        // Setup resize handler
        window.addEventListener('resize', () => this.onWindowResize());
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    setupWebXREventHandlers() {
        // Input source events
        const inputEvents = ['selectstart', 'selectend', 'select', 'squeezestart', 'squeezeend', 'squeeze'];
        
        inputEvents.forEach(eventType => {
            this.eventHandlers.set(eventType, (event) => this.handleInputEvent(eventType, event));
        });
    }

    async startSession(mode = 'immersive-vr', requiredFeatures = [], optionalFeatures = []) {
        if (!navigator.xr) {
            throw new Error('WebXR not supported');
        }

        // Build feature lists based on mode and platform
        const features = this.buildFeatureList(mode, requiredFeatures, optionalFeatures);
        
        this.sessionConfig = {
            mode,
            requiredFeatures: features.required,
            optionalFeatures: features.optional
        };

        try {
            // Request session
            this.xrSession = await navigator.xr.requestSession(mode, {
                requiredFeatures: features.required,
                optionalFeatures: features.optional
            });

            // Initialize session
            await this.onSessionStarted();
            
            return this.xrSession;
            
        } catch (error) {
            console.error('Failed to start WebXR session:', error);
            throw error;
        }
    }

    buildFeatureList(mode, requiredFeatures, optionalFeatures) {
        const features = {
            required: [...requiredFeatures],
            optional: [...optionalFeatures]
        };

        // Add default required features based on mode
        if (mode === 'immersive-vr') {
            if (!features.required.includes('local-floor')) {
                features.required.push('local-floor');
            }
        } else if (mode === 'immersive-ar') {
            if (!features.required.includes('local')) {
                features.required.push('local');
            }
        }

        // Add platform-specific optional features
        const platformFeatures = this.getPlatformSpecificFeatures();
        features.optional.push(...platformFeatures.filter(f => !features.optional.includes(f)));

        return features;
    }

    getPlatformSpecificFeatures() {
        const features = [];

        // Common features for all platforms
        features.push('bounded-floor', 'unbounded');

        // Hand tracking
        if (this.features.handTracking) {
            features.push('hand-tracking');
        }

        // Layers for better performance
        if (this.features.layers) {
            features.push('layers');
        }

        // Platform-specific features
        if (this.platform.isVisionPro) {
            // Vision Pro specific features
            features.push('secondary-views'); // For window management
        } else if (this.platform.isQuest) {
            // Quest specific features
            features.push('depth-sensing', 'mesh-detection', 'plane-detection');
        }

        // AR-specific features
        if (this.sessionConfig.mode === 'immersive-ar') {
            features.push('hit-test', 'anchors', 'light-estimation', 'dom-overlay');
        }

        return features;
    }

    async onSessionStarted() {
        console.log('🎮 WebXR session started');

        // Add event listeners
        this.xrSession.addEventListener('end', () => this.onSessionEnded());
        this.xrSession.addEventListener('inputsourceschange', (event) => this.onInputSourcesChange(event));
        
        // Add input event listeners
        this.eventHandlers.forEach((handler, eventType) => {
            this.xrSession.addEventListener(eventType, handler);
        });

        // Setup WebGL layer
        this.gl = this.renderer.getContext();
        this.baseLayer = new XRWebGLLayer(this.xrSession, this.gl, {
            antialias: true,
            alpha: true,
            framebufferScaleFactor: 1.0
        });

        await this.xrSession.updateRenderState({
            baseLayer: this.baseLayer,
            depthNear: 0.01,
            depthFar: 1000.0
        });

        // Get reference spaces
        try {
            this.xrRefSpace = await this.xrSession.requestReferenceSpace('local-floor');
        } catch (e) {
            console.warn('local-floor not available, falling back to local');
            this.xrRefSpace = await this.xrSession.requestReferenceSpace('local');
        }

        // Get viewer reference space for eye tracking
        this.xrViewerSpace = await this.xrSession.requestReferenceSpace('viewer');

        // Setup controllers and hands
        this.setupControllers();
        
        // Setup hit testing if available
        if (this.sessionConfig.mode === 'immersive-ar') {
            await this.setupHitTesting();
        }

        // Start render loop
        this.renderer.setAnimationLoop((timestamp, frame) => this.onXRFrame(timestamp, frame));

        // Dispatch custom event
        window.dispatchEvent(new CustomEvent('xr-session-started', {
            detail: {
                mode: this.sessionConfig.mode,
                features: this.sessionConfig
            }
        }));
    }

    onSessionEnded() {
        console.log('🛑 WebXR session ended');

        this.xrSession = null;
        this.xrRefSpace = null;
        this.xrViewerSpace = null;
        this.xrHitTestSource = null;

        // Clear controllers
        this.controllers.forEach(controller => {
            if (controller.parent) {
                controller.parent.remove(controller);
            }
        });
        this.controllers = [];
        this.hands.clear();
        this.inputSources.clear();

        // Stop render loop
        this.renderer.setAnimationLoop(null);

        // Dispatch custom event
        window.dispatchEvent(new CustomEvent('xr-session-ended'));
    }

    onInputSourcesChange(event) {
        console.log('🎮 Input sources changed:', event);

        // Handle added input sources
        event.added.forEach(inputSource => {
            this.handleInputSourceAdded(inputSource);
        });

        // Handle removed input sources
        event.removed.forEach(inputSource => {
            this.handleInputSourceRemoved(inputSource);
        });
    }

    handleInputSourceAdded(inputSource) {
        console.log('➕ Input source added:', inputSource.profiles, inputSource.handedness);

        // Store input source
        this.inputSources.set(inputSource, {
            profiles: inputSource.profiles,
            handedness: inputSource.handedness,
            targetRayMode: inputSource.targetRayMode,
            hand: inputSource.hand,
            gamepad: inputSource.gamepad
        });

        // Handle transient-pointer (Vision Pro)
        if (inputSource.targetRayMode === 'transient-pointer') {
            this.handleTransientPointerAdded(inputSource);
        }
        
        // Handle hand tracking
        else if (inputSource.hand) {
            this.handleHandTrackingAdded(inputSource);
        }
        
        // Handle controllers
        else if (inputSource.targetRayMode === 'tracked-pointer') {
            this.handleControllerAdded(inputSource);
        }
    }

    handleInputSourceRemoved(inputSource) {
        console.log('➖ Input source removed:', inputSource.handedness);
        
        // Clean up stored data
        this.inputSources.delete(inputSource);
        
        // Clean up specific input type
        if (inputSource.hand) {
            this.hands.delete(inputSource.handedness);
        }
    }

    handleTransientPointerAdded(inputSource) {
        // Vision Pro eye + pinch input
        console.log('👁️ Transient pointer detected (Vision Pro style input)');
        
        this.transientInputSources.set(inputSource.handedness || 'none', {
            inputSource,
            lastRay: null,
            isActive: false
        });
    }

    handleHandTrackingAdded(inputSource) {
        console.log('🖐️ Hand tracking detected:', inputSource.handedness);
        
        // Create hand model
        const handModel = this.createHandModel(inputSource);
        this.hands.set(inputSource.handedness, {
            inputSource,
            model: handModel,
            joints: new Map()
        });
    }

    handleControllerAdded(inputSource) {
        console.log('🎮 Controller detected:', inputSource.handedness);
        
        // Create controller representation
        const controller = this.createController(inputSource);
        this.controllers.push(controller);
    }

    setupControllers() {
        // Setup controller models for tracked devices
        for (let i = 0; i < 2; i++) {
            const controller = this.renderer.xr.getController(i);
            this.cameraGroup.add(controller);
            
            // Add ray visualization
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(0, 0, -1)
            ]);
            const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({
                color: 0xffffff,
                opacity: 0.5,
                transparent: true
            }));
            line.scale.z = 5;
            controller.add(line);
            
            this.controllers.push(controller);
        }
    }

    createHandModel(inputSource) {
        const handGroup = new THREE.Group();
        handGroup.name = `hand-${inputSource.handedness}`;
        
        // Create joint visualizations
        const jointNames = [
            'wrist',
            'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
            'index-finger-metacarpal', 'index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate', 
            'index-finger-phalanx-distal', 'index-finger-tip',
            'middle-finger-metacarpal', 'middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate',
            'middle-finger-phalanx-distal', 'middle-finger-tip',
            'ring-finger-metacarpal', 'ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate',
            'ring-finger-phalanx-distal', 'ring-finger-tip',
            'pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate',
            'pinky-finger-phalanx-distal', 'pinky-finger-tip'
        ];
        
        jointNames.forEach(jointName => {
            const joint = new THREE.Mesh(
                new THREE.SphereGeometry(0.008, 12, 8),
                new THREE.MeshPhongMaterial({
                    color: inputSource.handedness === 'left' ? 0x4a90e2 : 0xe24a90,
                    emissive: 0x000000,
                    shininess: 100
                })
            );
            joint.name = jointName;
            joint.castShadow = true;
            joint.receiveShadow = true;
            handGroup.add(joint);
        });
        
        this.scene.add(handGroup);
        return handGroup;
    }

    createController(inputSource) {
        const controller = new THREE.Group();
        controller.name = `controller-${inputSource.handedness}`;
        
        // Create controller mesh
        const geometry = new THREE.CylinderGeometry(0.02, 0.03, 0.1, 16);
        const material = new THREE.MeshPhongMaterial({
            color: inputSource.handedness === 'left' ? 0x0000ff : 0xff0000
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        controller.add(mesh);
        
        this.scene.add(controller);
        return controller;
    }

    async setupHitTesting() {
        if (!this.features.ar) return;
        
        try {
            const hitTestOptions = {
                space: this.xrViewerSpace,
                entityTypes: ['plane', 'mesh'],
                offsetRay: new XRRay()
            };
            
            this.xrHitTestSource = await this.xrSession.requestHitTestSource(hitTestOptions);
            console.log('✅ Hit testing initialized');
        } catch (e) {
            console.warn('Hit testing not available:', e);
        }
    }

    handleInputEvent(eventType, event) {
        const inputSource = event.inputSource;
        const data = this.inputSources.get(inputSource);
        
        if (!data) return;
        
        // Handle transient-pointer events (Vision Pro)
        if (inputSource.targetRayMode === 'transient-pointer') {
            this.handleTransientPointerEvent(eventType, event);
        }
        
        // Dispatch custom event for app to handle
        window.dispatchEvent(new CustomEvent(`xr-${eventType}`, {
            detail: {
                inputSource,
                frame: event.frame,
                data
            }
        }));
    }

    handleTransientPointerEvent(eventType, event) {
        const inputSource = event.inputSource;
        const frame = event.frame;
        
        if (!this.xrRefSpace) return;
        
        // Get the ray pose
        const rayPose = frame.getPose(inputSource.targetRaySpace, this.xrRefSpace);
        
        if (rayPose) {
            // This is where the user is looking when they pinch
            const ray = {
                origin: new THREE.Vector3().fromArray(rayPose.transform.position),
                direction: new THREE.Vector3(0, 0, -1).applyQuaternion(
                    new THREE.Quaternion().fromArray(rayPose.transform.orientation)
                )
            };
            
            console.log(`👁️ Transient pointer ${eventType} at:`, ray);
            
            // Store for app usage
            const transientData = this.transientInputSources.get(inputSource.handedness || 'none');
            if (transientData) {
                transientData.lastRay = ray;
                transientData.isActive = eventType.includes('start');
            }
        }
    }

    onXRFrame(timestamp, xrFrame) {
        this.xrFrame = xrFrame;
        
        if (!this.xrSession) return;
        
        // Get pose
        this.xrPose = xrFrame.getViewerPose(this.xrRefSpace);
        
        if (this.xrPose) {
            // Update camera position/orientation
            const transform = this.xrPose.transform;
            this.cameraGroup.position.fromArray(transform.position);
            this.cameraGroup.quaternion.fromArray(transform.orientation);
            
            // Update hand tracking
            this.updateHandTracking(xrFrame);
            
            // Update transient pointers
            this.updateTransientPointers(xrFrame);
            
            // Update performance stats
            this.updatePerformanceStats(timestamp);
        }
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }

    updateHandTracking(frame) {
        this.hands.forEach((handData, handedness) => {
            const inputSource = handData.inputSource;
            
            if (!inputSource.hand) return;
            
            // Update each joint
            for (const [jointName, joint] of inputSource.hand.entries()) {
                const jointPose = frame.getJointPose(joint, this.xrRefSpace);
                
                if (jointPose) {
                    // Find the corresponding mesh
                    const jointMesh = handData.model.getObjectByName(jointName);
                    
                    if (jointMesh) {
                        jointMesh.position.fromArray(jointPose.transform.position);
                        jointMesh.quaternion.fromArray(jointPose.transform.orientation);
                        jointMesh.visible = true;
                        
                        // Adjust joint size based on radius
                        if (jointPose.radius) {
                            const scale = jointPose.radius / 0.008;
                            jointMesh.scale.setScalar(scale);
                        }
                    }
                    
                    // Store joint data
                    handData.joints.set(jointName, {
                        position: new THREE.Vector3().fromArray(jointPose.transform.position),
                        orientation: new THREE.Quaternion().fromArray(jointPose.transform.orientation),
                        radius: jointPose.radius || 0.008
                    });
                }
            }
        });
    }

    updateTransientPointers(frame) {
        this.transientInputSources.forEach((transientData, handedness) => {
            const inputSource = transientData.inputSource;
            
            if (inputSource.targetRaySpace && inputSource.gripSpace) {
                // Update ray pose
                const rayPose = frame.getPose(inputSource.targetRaySpace, this.xrRefSpace);
                const gripPose = frame.getPose(inputSource.gripSpace, this.xrRefSpace);
                
                if (rayPose) {
                    transientData.lastRay = {
                        origin: new THREE.Vector3().fromArray(rayPose.transform.position),
                        direction: new THREE.Vector3(0, 0, -1).applyQuaternion(
                            new THREE.Quaternion().fromArray(rayPose.transform.orientation)
                        )
                    };
                }
                
                if (gripPose) {
                    transientData.gripPosition = new THREE.Vector3().fromArray(gripPose.transform.position);
                }
            }
        });
    }

    updatePerformanceStats(timestamp) {
        if (!this.lastFrameTime) {
            this.lastFrameTime = timestamp;
            return;
        }
        
        const frameTime = timestamp - this.lastFrameTime;
        this.lastFrameTime = timestamp;
        
        // Update FPS
        this.performance.fps = Math.round(1000 / frameTime);
        this.performance.frameTime = frameTime;
        
        // Check for dropped frames
        if (frameTime > 16.67) { // More than 60fps frame time
            this.performance.droppedFrames++;
        }
    }

    // Public API methods
    
    async startVR(requiredFeatures = [], optionalFeatures = []) {
        return this.startSession('immersive-vr', requiredFeatures, optionalFeatures);
    }

    async startAR(requiredFeatures = [], optionalFeatures = []) {
        return this.startSession('immersive-ar', requiredFeatures, optionalFeatures);
    }

    endSession() {
        if (this.xrSession) {
            this.xrSession.end();
        }
    }

    isSessionActive() {
        return this.xrSession !== null;
    }

    getHandJointPose(handedness, jointName) {
        const handData = this.hands.get(handedness);
        if (!handData) return null;
        
        return handData.joints.get(jointName);
    }

    getTransientPointerRay(handedness = 'none') {
        const transientData = this.transientInputSources.get(handedness);
        return transientData ? transientData.lastRay : null;
    }

    isTransientPointerActive(handedness = 'none') {
        const transientData = this.transientInputSources.get(handedness);
        return transientData ? transientData.isActive : false;
    }

    getControllerPose(handedness) {
        // Implementation for getting controller pose
        for (const inputSource of this.xrSession.inputSources) {
            if (inputSource.handedness === handedness && inputSource.gripSpace) {
                const pose = this.xrFrame?.getPose(inputSource.gripSpace, this.xrRefSpace);
                if (pose) {
                    return {
                        position: new THREE.Vector3().fromArray(pose.transform.position),
                        orientation: new THREE.Quaternion().fromArray(pose.transform.orientation)
                    };
                }
            }
        }
        return null;
    }

    vibrate(handedness, intensity = 1.0, duration = 100) {
        for (const inputSource of this.xrSession.inputSources) {
            if (inputSource.handedness === handedness && 
                inputSource.gamepad && 
                inputSource.gamepad.hapticActuators) {
                
                inputSource.gamepad.hapticActuators.forEach(actuator => {
                    if (actuator.pulse) {
                        actuator.pulse(intensity, duration);
                    }
                });
            }
        }
    }

    getHitTestResults() {
        if (!this.xrHitTestSource || !this.xrFrame) return [];
        
        const results = this.xrFrame.getHitTestResults(this.xrHitTestSource);
        return results.map(result => ({
            pose: result.getPose(this.xrRefSpace),
            result
        }));
    }

    async createAnchor(position, orientation) {
        if (!this.xrSession || !this.features.anchors) {
            throw new Error('Anchors not supported');
        }
        
        const anchorPose = new XRRigidTransform(
            { x: position.x, y: position.y, z: position.z },
            { x: orientation.x, y: orientation.y, z: orientation.z, w: orientation.w }
        );
        
        return await this.xrFrame.createAnchor(anchorPose, this.xrRefSpace);
    }

    getPerformanceStats() {
        return { ...this.performance };
    }

    getSupportedFeatures() {
        return { ...this.features };
    }

    getPlatformInfo() {
        return {
            platform: { ...this.platform },
            name: this.getPlatformName()
        };
    }
}

// Export singleton instance
export const webXRManager = new WebXRManagerV2();