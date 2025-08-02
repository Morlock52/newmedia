/**
 * WebXR Manager - Core WebXR functionality for AR/VR Media Platform
 * Supports Apple Vision Pro, Meta Quest 3, and spatial computing devices
 */

export class WebXRManager {
    constructor() {
        this.xrSession = null;
        this.xrRefSpace = null;
        this.gl = null;
        this.canvas = null;
        this.renderer = null;
        this.scene = null;
        this.camera = null;
        this.handTracking = null;
        this.eyeTracking = null;
        this.hapticActuators = [];
        this.spatialAudio = null;
        
        this.isVRSupported = false;
        this.isARSupported = false;
        this.isHandTrackingSupported = false;
        this.isEyeTrackingSupported = false;
        this.isHapticSupported = false;
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing WebXR Manager...');
        
        // Check WebXR support
        await this.checkWebXRSupport();
        
        // Initialize Three.js components
        this.initThreeJS();
        
        // Setup device detection
        this.detectDeviceCapabilities();
        
        console.log('✅ WebXR Manager initialized');
    }

    async checkWebXRSupport() {
        if (!('xr' in navigator)) {
            console.warn('WebXR not supported');
            return;
        }

        try {
            this.isVRSupported = await navigator.xr.isSessionSupported('immersive-vr');
            this.isARSupported = await navigator.xr.isSessionSupported('immersive-ar');
            
            console.log(`VR Support: ${this.isVRSupported}`);
            console.log(`AR Support: ${this.isARSupported}`);
            
            // Check for optional features
            if (this.isVRSupported || this.isARSupported) {
                await this.checkOptionalFeatures();
            }
        } catch (error) {
            console.error('Error checking WebXR support:', error);
        }
    }

    async checkOptionalFeatures() {
        const optionalFeatures = [
            'hand-tracking',
            'eye-tracking', 
            'local-floor',
            'bounded-floor',
            'unbounded',
            'layers',
            'hit-test',
            'anchors',
            'depth-sensing',
            'dom-overlay'
        ];

        for (const feature of optionalFeatures) {
            try {
                const supported = await navigator.xr.isSessionSupported('immersive-vr', {
                    optionalFeatures: [feature]
                });
                
                if (supported) {
                    switch (feature) {
                        case 'hand-tracking':
                            this.isHandTrackingSupported = true;
                            break;
                        case 'eye-tracking':
                            this.isEyeTrackingSupported = true;
                            break;
                    }
                }
                
                console.log(`${feature}: ${supported}`);
            } catch (error) {
                console.log(`${feature}: not supported`);
            }
        }
    }

    initThreeJS() {
        this.canvas = document.getElementById('webxr-canvas');
        
        // Initialize Three.js renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.xr.enabled = true;
        
        // Create scene
        this.scene = new THREE.Scene();
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        
        // Add ambient lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Add directional lighting
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Start render loop
        this.renderer.setAnimationLoop(() => this.render());
    }

    detectDeviceCapabilities() {
        const userAgent = navigator.userAgent.toLowerCase();
        
        // Detect Apple Vision Pro
        if (userAgent.includes('vision') || userAgent.includes('visionos')) {
            this.updateDeviceStatus('apple-vision-pro', 'Optimized ✅');
            console.log('🥽 Apple Vision Pro detected');
        }
        
        // Detect Meta Quest
        if (userAgent.includes('quest') || userAgent.includes('oculus')) {
            this.updateDeviceStatus('meta-quest', 'Enhanced MR ✅');
            console.log('🥽 Meta Quest detected');
        }
        
        // General spatial computing support
        if (this.isVRSupported || this.isARSupported) {
            this.updateDeviceStatus('spatial-computing', 'Ready ✅');
            console.log('🌐 Spatial computing ready');
        }
    }

    updateDeviceStatus(deviceClass, status) {
        const deviceElement = document.querySelector(`.${deviceClass} .support-status`);
        if (deviceElement) {
            deviceElement.innerHTML = `<span>${status}</span>`;
        }
    }

    async startVRSession(requiredFeatures = [], optionalFeatures = []) {
        if (!this.isVRSupported) {
            throw new Error('VR not supported');
        }

        const sessionInit = {
            requiredFeatures: ['local-floor', ...requiredFeatures],
            optionalFeatures: [
                'hand-tracking',
                'eye-tracking',
                'bounded-floor',
                'layers',
                'dom-overlay',
                ...optionalFeatures
            ]
        };

        try {
            this.xrSession = await navigator.xr.requestSession('immersive-vr', sessionInit);
            await this.initXRSession();
            console.log('🥽 VR Session started');
            return this.xrSession;
        } catch (error) {
            console.error('Failed to start VR session:', error);
            throw error;
        }
    }

    async startARSession(requiredFeatures = [], optionalFeatures = []) {
        if (!this.isARSupported) {
            throw new Error('AR not supported');
        }

        const sessionInit = {
            requiredFeatures: ['local', ...requiredFeatures],
            optionalFeatures: [
                'hand-tracking',
                'hit-test',
                'anchors',
                'dom-overlay',
                'depth-sensing',
                ...optionalFeatures
            ]
        };

        try {
            this.xrSession = await navigator.xr.requestSession('immersive-ar', sessionInit);
            await this.initXRSession();
            console.log('📱 AR Session started');
            return this.xrSession;
        } catch (error) {
            console.error('Failed to start AR session:', error);
            throw error;
        }
    }

    async initXRSession() {
        // Set up WebGL context
        this.gl = this.renderer.getContext();
        await this.xrSession.updateRenderState({
            baseLayer: new XRWebGLLayer(this.xrSession, this.gl)
        });

        // Get reference space
        this.xrRefSpace = await this.xrSession.requestReferenceSpace('local-floor')
            .catch(() => this.xrSession.requestReferenceSpace('local'));

        // Initialize hand tracking if supported
        if (this.isHandTrackingSupported) {
            this.initHandTracking();
        }

        // Initialize eye tracking if supported
        if (this.isEyeTrackingSupported) {
            this.initEyeTracking();
        }

        // Initialize haptic feedback
        this.initHapticFeedback();

        // Set up session event handlers
        this.xrSession.addEventListener('end', () => {
            this.xrSession = null;
            this.xrRefSpace = null;
            console.log('XR Session ended');
        });

        // Initialize spatial audio
        this.initSpatialAudio();
    }

    initHandTracking() {
        if (!this.xrSession.inputSources) return;

        console.log('👋 Initializing hand tracking...');
        this.handTracking = {
            left: null,
            right: null,
            joints: new Map()
        };
    }

    initEyeTracking() {
        console.log('👁️ Initializing eye tracking...');
        this.eyeTracking = {
            gazeDirection: new THREE.Vector3(),
            gazeOrigin: new THREE.Vector3(),
            isTracking: false
        };
    }

    initHapticFeedback() {
        console.log('🤝 Initializing haptic feedback...');
        this.hapticActuators = [];
        
        // Check for haptic actuators on input sources
        if (this.xrSession.inputSources) {
            this.xrSession.inputSources.forEach(inputSource => {
                if (inputSource.gamepad && inputSource.gamepad.hapticActuators) {
                    this.hapticActuators.push(...inputSource.gamepad.hapticActuators);
                }
            });
        }
    }

    initSpatialAudio() {
        console.log('🔊 Initializing spatial audio...');
        
        // Create Audio Context with spatial capabilities
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.spatialAudio = {
            context: new AudioContext(),
            listener: null,
            sources: new Map()
        };

        // Enable spatial audio in Three.js
        this.camera.add(new THREE.AudioListener());
    }

    updateHandTracking(frame) {
        if (!this.handTracking || !this.xrSession.inputSources) return;

        for (const inputSource of this.xrSession.inputSources) {
            if (inputSource.hand) {
                const hand = inputSource.hand;
                const handedness = inputSource.handedness;
                
                for (const [jointName, joint] of hand.entries()) {
                    const jointPose = frame.getJointPose(joint, this.xrRefSpace);
                    if (jointPose) {
                        const position = jointPose.transform.position;
                        const orientation = jointPose.transform.orientation;
                        
                        // Store joint data
                        this.handTracking.joints.set(`${handedness}-${jointName}`, {
                            position: new THREE.Vector3(position.x, position.y, position.z),
                            orientation: new THREE.Quaternion(
                                orientation.x, orientation.y, orientation.z, orientation.w
                            ),
                            radius: jointPose.radius
                        });
                    }
                }
            }
        }
    }

    updateEyeTracking(frame) {
        if (!this.eyeTracking) return;

        // Eye tracking implementation would go here
        // Note: Eye tracking API is still experimental
        console.debug('Eye tracking update...');
    }

    pulseHaptic(intensity = 1.0, duration = 100) {
        this.hapticActuators.forEach(actuator => {
            if (actuator && actuator.pulse) {
                actuator.pulse(intensity, duration);
            }
        });
    }

    render() {
        if (this.xrSession) {
            // XR rendering handled by WebXR
        } else {
            // Fallback 2D rendering
            this.renderer.render(this.scene, this.camera);
        }
    }

    endSession() {
        if (this.xrSession) {
            this.xrSession.end();
        }
    }

    // Utility methods
    getControllerPose(handedness) {
        if (!this.xrSession) return null;
        
        for (const inputSource of this.xrSession.inputSources) {
            if (inputSource.handedness === handedness && inputSource.gripSpace) {
                return inputSource.gripSpace;
            }
        }
        return null;
    }

    getHandJoint(handedness, jointName) {
        if (!this.handTracking) return null;
        return this.handTracking.joints.get(`${handedness}-${jointName}`);
    }

    isSupported() {
        return this.isVRSupported || this.isARSupported;
    }

    getSupportedFeatures() {
        return {
            vr: this.isVRSupported,
            ar: this.isARSupported,
            handTracking: this.isHandTrackingSupported,
            eyeTracking: this.isEyeTrackingSupported,
            haptic: this.isHapticSupported
        };
    }
}

// Export singleton instance
export const webXRManager = new WebXRManager();