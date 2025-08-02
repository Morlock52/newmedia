/**
 * Main Application - AR/VR Immersive Media Platform
 * Orchestrates all WebXR features and provides unified interface
 */

import { webXRManager } from './webxr-manager.js';
import VRCinema from '../vr-cinema/vr-cinema.js';
import AROverlay from '../ar-overlay/ar-overlay.js';
import SpatialVideo from '../spatial-video/spatial-video.js';
import HandTracking from '../hand-tracking/hand-tracking.js';
import EyeTracking from '../eye-tracking/eye-tracking.js';
import HapticFeedback from '../haptic-feedback/haptic-feedback.js';
import MixedRealityCollaborativeSpaces from '../mixed-reality/mixed-reality.js';

class ARVRMediaApp {
    constructor() {
        this.features = {
            vrCinema: null,
            arOverlay: null,
            spatialVideo: null,
            handTracking: null,
            eyeTracking: null,
            hapticFeedback: null,
            mixedReality: null
        };
        
        this.currentMode = 'desktop'; // desktop, vr, ar, mr
        this.isInitialized = false;
        this.activeFeatures = new Set();
        this.deviceCapabilities = null;
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing AR/VR Immersive Media Platform...');
        
        try {
            // Wait for WebXR Manager to initialize
            await this.waitForWebXRManager();
            
            // Initialize all features
            await this.initializeFeatures();
            
            // Setup device detection
            this.detectDeviceCapabilities();
            
            // Setup UI event handlers
            this.setupEventHandlers();
            
            // Setup main render loop
            this.setupRenderLoop();
            
            // Update UI based on capabilities
            this.updateUICapabilities();
            
            this.isInitialized = true;
            console.log('✅ AR/VR Immersive Media Platform initialized successfully');
            
            // Show welcome message
            this.showWelcomeMessage();
            
        } catch (error) {
            console.error('❌ Failed to initialize AR/VR Media Platform:', error);
            this.showErrorMessage('Failed to initialize platform', error.message);
        }
    }

    async waitForWebXRManager() {
        return new Promise((resolve) => {
            const checkWebXR = () => {
                if (webXRManager && webXRManager.isSupported !== undefined) {
                    resolve();
                } else {
                    setTimeout(checkWebXR, 100);
                }
            };
            checkWebXR();
        });
    }

    async initializeFeatures() {
        console.log('🔧 Initializing features...');
        
        // Initialize all features in parallel
        const initPromises = [
            this.initVRCinema(),
            this.initAROverlay(),
            this.initSpatialVideo(),
            this.initHandTracking(),
            this.initEyeTracking(),
            this.initHapticFeedback(),
            this.initMixedReality()
        ];
        
        await Promise.all(initPromises);
        console.log('✅ All features initialized');
    }

    async initVRCinema() {
        try {
            this.features.vrCinema = new VRCinema();
            console.log('🎬 VR Cinema initialized');
        } catch (error) {
            console.warn('⚠️ VR Cinema initialization failed:', error);
        }
    }

    async initAROverlay() {
        try {
            this.features.arOverlay = new AROverlay();
            console.log('🔍 AR Overlay initialized');
        } catch (error) {
            console.warn('⚠️ AR Overlay initialization failed:', error);
        }
    }

    async initSpatialVideo() {
        try {
            this.features.spatialVideo = new SpatialVideo();
            console.log('📹 Spatial Video initialized');
        } catch (error) {
            console.warn('⚠️ Spatial Video initialization failed:', error);
        }
    }

    async initHandTracking() {
        try {
            this.features.handTracking = new HandTracking();
            console.log('👋 Hand Tracking initialized');
        } catch (error) {
            console.warn('⚠️ Hand Tracking initialization failed:', error);
        }
    }

    async initEyeTracking() {
        try {
            this.features.eyeTracking = new EyeTracking();
            console.log('👁️ Eye Tracking initialized');
        } catch (error) {
            console.warn('⚠️ Eye Tracking initialization failed:', error);
        }
    }

    async initHapticFeedback() {
        try {
            this.features.hapticFeedback = new HapticFeedback();
            console.log('🤝 Haptic Feedback initialized');
        } catch (error) {
            console.warn('⚠️ Haptic Feedback initialization failed:', error);
        }
    }

    async initMixedReality() {
        try {
            this.features.mixedReality = new MixedRealityCollaborativeSpaces();
            console.log('🌍 Mixed Reality initialized');
        } catch (error) {
            console.warn('⚠️ Mixed Reality initialization failed:', error);
        }
    }

    detectDeviceCapabilities() {
        this.deviceCapabilities = {
            webxr: webXRManager.isSupported(),
            vr: webXRManager.isVRSupported,
            ar: webXRManager.isARSupported,
            handTracking: webXRManager.isHandTrackingSupported,
            eyeTracking: webXRManager.isEyeTrackingSupported,
            haptic: webXRManager.isHapticSupported,
            device: this.detectDeviceType()
        };
        
        console.log('📱 Device capabilities:', this.deviceCapabilities);
    }

    detectDeviceType() {
        const userAgent = navigator.userAgent.toLowerCase();
        
        if (userAgent.includes('vision') || userAgent.includes('visionos')) {
            return 'apple-vision-pro';
        } else if (userAgent.includes('quest') || userAgent.includes('oculus')) {
            return 'meta-quest';
        } else if (userAgent.includes('hololens')) {
            return 'hololens';
        } else if (userAgent.includes('magic leap')) {
            return 'magic-leap';
        }
        
        return 'desktop';
    }

    setupEventHandlers() {
        console.log('🎛️ Setting up event handlers...');
        
        // Feature navigation buttons
        const buttons = {
            'vr-cinema-btn': () => this.activateFeature('vrCinema'),
            'ar-overlay-btn': () => this.activateFeature('arOverlay'),
            'spatial-video-btn': () => this.activateFeature('spatialVideo'),
            'hand-tracking-btn': () => this.activateFeature('handTracking'),
            'eye-tracking-btn': () => this.activateFeature('eyeTracking'),
            'haptic-feedback-btn': () => this.activateFeature('hapticFeedback'),
            'mixed-reality-btn': () => this.activateFeature('mixedReality')
        };
        
        Object.entries(buttons).forEach(([buttonId, handler]) => {
            const button = document.getElementById(buttonId);
            if (button) {
                button.addEventListener('click', handler);
            }
        });
        
        // Global keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            this.handleKeyboardShortcuts(event);
        });
        
        // Window resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // WebXR session events
        this.setupWebXREventHandlers();
    }

    handleKeyboardShortcuts(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case '1':
                    event.preventDefault();
                    this.activateFeature('vrCinema');
                    break;
                case '2':
                    event.preventDefault();
                    this.activateFeature('arOverlay');
                    break;
                case '3':
                    event.preventDefault();
                    this.activateFeature('spatialVideo');
                    break;
                case '4':
                    event.preventDefault();
                    this.activateFeature('handTracking');
                    break;
                case '5':
                    event.preventDefault();
                    this.activateFeature('eyeTracking');
                    break;
                case '6':
                    event.preventDefault();
                    this.activateFeature('hapticFeedback');
                    break;
                case '7':
                    event.preventDefault();
                    this.activateFeature('mixedReality');
                    break;
                case 'Escape':
                    event.preventDefault();
                    this.exitXR();
                    break;
            }
        }
    }

    handleResize() {
        if (webXRManager.renderer) {
            const canvas = webXRManager.canvas;
            webXRManager.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
            webXRManager.camera.aspect = canvas.clientWidth / canvas.clientHeight;
            webXRManager.camera.updateProjectionMatrix();
        }
    }

    setupWebXREventHandlers() {
        // Listen for XR session start/end events
        document.addEventListener('xr-session-started', (event) => {
            this.onXRSessionStarted(event.detail);
        });
        
        document.addEventListener('xr-session-ended', (event) => {
            this.onXRSessionEnded(event.detail);
        });
    }

    onXRSessionStarted(sessionInfo) {
        console.log('🥽 XR Session started:', sessionInfo);
        
        this.currentMode = sessionInfo.mode; // 'vr', 'ar', etc.
        
        // Update UI for XR mode
        this.updateUIForXRMode(true);
        
        // Start XR-specific features
        this.startXRFeatures();
    }

    onXRSessionEnded(sessionInfo) {
        console.log('🏁 XR Session ended:', sessionInfo);
        
        this.currentMode = 'desktop';
        
        // Update UI back to desktop mode
        this.updateUIForXRMode(false);
        
        // Stop XR-specific features
        this.stopXRFeatures();
    }

    startXRFeatures() {
        // Start hand tracking if available
        if (this.features.handTracking && webXRManager.isHandTrackingSupported) {
            this.features.handTracking.setHandVisibility(true);
        }
        
        // Start eye tracking if available
        if (this.features.eyeTracking && webXRManager.isEyeTrackingSupported) {
            this.features.eyeTracking.startEyeTracking().catch(console.warn);
        }
        
        // Enable haptic feedback
        if (this.features.hapticFeedback) {
            this.features.hapticFeedback.enableHapticType('vibration', true);
        }
    }

    stopXRFeatures() {
        // Stop hand tracking
        if (this.features.handTracking) {
            this.features.handTracking.setHandVisibility(false);
        }
        
        // Stop eye tracking
        if (this.features.eyeTracking) {
            this.features.eyeTracking.stopEyeTracking();
        }
        
        // Stop haptic patterns
        if (this.features.hapticFeedback) {
            this.features.hapticFeedback.stopAllPatterns();
        }
    }

    setupRenderLoop() {
        console.log('🔄 Setting up render loop...');
        
        const animate = (timestamp, frame) => {
            // Update all active features
            this.updateFeatures(frame);
            
            // Continue animation loop
            if (webXRManager.renderer) {
                webXRManager.renderer.setAnimationLoop(animate);
            }
        };
        
        // Start the render loop
        if (webXRManager.renderer) {
            webXRManager.renderer.setAnimationLoop(animate);
        }
    }

    updateFeatures(frame) {
        // Update hand tracking
        if (this.features.handTracking && this.activeFeatures.has('handTracking')) {
            this.features.handTracking.updateHandTracking(frame);
        }
        
        // Update eye tracking
        if (this.features.eyeTracking && this.activeFeatures.has('eyeTracking')) {
            // Eye tracking updates are handled internally
        }
        
        // Update haptic feedback
        if (this.features.hapticFeedback && this.activeFeatures.has('hapticFeedback')) {
            this.features.hapticFeedback.updateHapticActuators(frame);
        }
        
        // Update spatial video
        if (this.features.spatialVideo && this.activeFeatures.has('spatialVideo')) {
            this.features.spatialVideo.updateSpatialVideo(frame);
        }
        
        // Update AR overlay
        if (this.features.arOverlay && this.activeFeatures.has('arOverlay')) {
            this.features.arOverlay.updateOverlays(frame);
        }
        
        // Update mixed reality
        if (this.features.mixedReality && this.activeFeatures.has('mixedReality')) {
            this.features.mixedReality.updateMixedReality(frame);
        }
    }

    async activateFeature(featureName) {
        console.log(`🎯 Activating feature: ${featureName}`);
        
        try {
            const feature = this.features[featureName];
            if (!feature) {
                throw new Error(`Feature not available: ${featureName}`);
            }
            
            // Deactivate other features first
            this.deactivateAllFeatures();
            
            // Activate the selected feature
            await this.activateSpecificFeature(featureName, feature);
            
            // Update UI
            this.updateActiveFeatureUI(featureName);
            
            // Add to active features
            this.activeFeatures.add(featureName);
            
            console.log(`✅ Feature activated: ${featureName}`);
            
        } catch (error) {
            console.error(`❌ Failed to activate feature ${featureName}:`, error);
            this.showErrorMessage(`Failed to activate ${featureName}`, error.message);
        }
    }

    async activateSpecificFeature(featureName, feature) {
        switch (featureName) {
            case 'vrCinema':
                await this.activateVRCinema(feature);
                break;
            case 'arOverlay':
                await this.activateAROverlay(feature);
                break;
            case 'spatialVideo':
                await this.activateSpatialVideo(feature);
                break;
            case 'handTracking':
                await this.activateHandTracking(feature);
                break;
            case 'eyeTracking':
                await this.activateEyeTracking(feature);
                break;
            case 'hapticFeedback':
                await this.activateHapticFeedback(feature);
                break;
            case 'mixedReality':
                await this.activateMixedReality(feature);
                break;
            default:
                throw new Error(`Unknown feature: ${featureName}`);
        }
    }

    async activateVRCinema(feature) {
        // Start VR session
        if (webXRManager.isVRSupported) {
            await feature.enterVRMode();
            this.currentMode = 'vr';
        } else {
            // Fallback to desktop VR cinema
            feature.playVideo();
            this.showInfoMessage('VR Cinema', 'Running in desktop mode. VR headset not detected.');
        }
    }

    async activateAROverlay(feature) {
        // Start AR session
        if (webXRManager.isARSupported) {
            await feature.startARSession();
            this.currentMode = 'ar';
        } else {
            this.showInfoMessage('AR Overlay', 'AR not supported on this device.');
        }
    }

    async activateSpatialVideo(feature) {
        // Load and play spatial video
        const spatialVideoConfig = {
            url: 'assets/spatial-video-sample.mp4',
            format: 'spatial-stereo',
            stereoMode: 'side-by-side'
        };
        
        try {
            const videoId = await feature.loadSpatialVideo(spatialVideoConfig);
            feature.play(videoId);
        } catch (error) {
            console.warn('Loading sample spatial video failed, using demo mode');
        }
    }

    async activateHandTracking(feature) {
        feature.setHandVisibility(true);
        feature.setJointVisibility(true);
        
        if (webXRManager.isHandTrackingSupported) {
            this.showInfoMessage('Hand Tracking', 'Hand tracking is active. Use gestures to interact.');
        } else {
            this.showInfoMessage('Hand Tracking', 'Hand tracking simulation active. Real hand tracking not available.');
        }
    }

    async activateEyeTracking(feature) {
        try {
            await feature.startEyeTracking();
            feature.setGazeIndicatorVisibility(true);
            this.showInfoMessage('Eye Tracking', 'Eye tracking is active. Look at objects to interact.');
        } catch (error) {
            feature.setGazeIndicatorVisibility(true); // Show mock indicator
            this.showInfoMessage('Eye Tracking', 'Eye tracking simulation active. Real eye tracking not available.');
        }
    }

    async activateHapticFeedback(feature) {
        // Test haptic feedback
        feature.playPattern('success_chime');
        
        // Show haptic demo
        this.startHapticDemo(feature);
        
        this.showInfoMessage('Haptic Feedback', 'Haptic feedback system is active. Interact with objects to feel responses.');
    }

    startHapticDemo(feature) {
        // Create interactive objects for haptic demo
        const demoObjects = this.createHapticDemoObjects();
        
        demoObjects.forEach(obj => {
            // Add haptic interaction handlers
            obj.userData.onHover = () => {
                feature.pulse('both', 0.3, 50);
            };
            
            obj.userData.onSelect = () => {
                feature.playPattern('button_click');
            };
            
            // Add to hand tracking interactions
            if (this.features.handTracking) {
                this.features.handTracking.addInteractableObject(obj);
            }
        });
    }

    createHapticDemoObjects() {
        const objects = [];
        
        // Create demo buttons
        for (let i = 0; i < 3; i++) {
            const buttonGeometry = new THREE.BoxGeometry(0.2, 0.1, 0.2);
            const buttonMaterial = new THREE.MeshLambertMaterial({
                color: 0x4a90e2
            });
            
            const button = new THREE.Mesh(buttonGeometry, buttonMaterial);
            button.position.set(-1 + i, 1, -2);
            button.name = `haptic-demo-button-${i}`;
            
            webXRManager.scene.add(button);
            objects.push(button);
        }
        
        return objects;
    }

    async activateMixedReality(feature) {
        // Enable passthrough if available
        feature.enablePassthrough(true);
        
        // Create a demo collaborative space
        const spacePosition = new THREE.Vector3(0, 0, -3);
        const spaceId = await feature.createCollaborativeSpace('meeting_room', spacePosition);
        
        // Join the space as the main user
        const participantInfo = {
            id: 'user_main',
            name: 'You',
            avatarColor: 0x4a90e2
        };
        
        await feature.joinCollaborativeSpace(spaceId, participantInfo);
        
        this.showInfoMessage('Mixed Reality', 'Mixed reality collaborative space created. Use gestures to interact.');
    }

    deactivateAllFeatures() {
        this.activeFeatures.forEach(featureName => {
            this.deactivateFeature(featureName);
        });
        this.activeFeatures.clear();
    }

    deactivateFeature(featureName) {
        const feature = this.features[featureName];
        if (!feature) return;
        
        switch (featureName) {
            case 'vrCinema':
                feature.pauseVideo();
                break;
            case 'arOverlay':
                feature.clearAllOverlays();
                break;
            case 'spatialVideo':
                feature.pauseAll();
                break;
            case 'handTracking':
                feature.setHandVisibility(false);
                feature.setJointVisibility(false);
                break;
            case 'eyeTracking':
                feature.setGazeIndicatorVisibility(false);
                break;
            case 'hapticFeedback':
                feature.stopAllPatterns();
                break;
            case 'mixedReality':
                feature.enablePassthrough(false);
                break;
        }
        
        console.log(`🔄 Deactivated feature: ${featureName}`);
    }

    updateUICapabilities() {
        const capabilities = this.deviceCapabilities;
        
        // Update device status indicators
        this.updateDeviceStatus('apple-vision-pro', capabilities.device === 'apple-vision-pro');
        this.updateDeviceStatus('meta-quest', capabilities.device === 'meta-quest');
        this.updateDeviceStatus('spatial-computing', capabilities.webxr);
        
        // Enable/disable feature buttons based on capabilities
        this.updateFeatureButton('vr-cinema-btn', capabilities.vr);
        this.updateFeatureButton('ar-overlay-btn', capabilities.ar);
        this.updateFeatureButton('spatial-video-btn', true); // Always available
        this.updateFeatureButton('hand-tracking-btn', capabilities.handTracking);
        this.updateFeatureButton('eye-tracking-btn', capabilities.eyeTracking);
        this.updateFeatureButton('haptic-feedback-btn', capabilities.haptic);
        this.updateFeatureButton('mixed-reality-btn', capabilities.ar || capabilities.vr);
    }

    updateDeviceStatus(deviceClass, isSupported) {
        const statusElement = document.querySelector(`.${deviceClass} .support-status`);
        if (statusElement) {
            statusElement.textContent = isSupported ? 'Supported ✅' : 'Not Available ❌';
            statusElement.style.color = isSupported ? '#00ff00' : '#ff0000';
        }
    }

    updateFeatureButton(buttonId, isSupported) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = !isSupported;
            button.style.opacity = isSupported ? '1' : '0.5';
            
            if (!isSupported) {
                button.title = 'Not supported on this device';
            }
        }
    }

    updateActiveFeatureUI(activeFeature) {
        // Remove active class from all buttons
        document.querySelectorAll('.feature-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Add active class to current feature button
        const activeButton = document.getElementById(`${activeFeature.replace(/([A-Z])/g, '-$1').toLowerCase()}-btn`);
        if (activeButton) {
            activeButton.classList.add('active');
        }
    }

    updateUIForXRMode(isXRMode) {
        const header = document.querySelector('.header');
        const nav = document.querySelector('.feature-nav');
        const footer = document.querySelector('.footer');
        
        if (isXRMode) {
            // Hide UI elements in XR mode
            if (header) header.style.display = 'none';
            if (nav) nav.style.display = 'none';
            if (footer) footer.style.display = 'none';
        } else {
            // Show UI elements in desktop mode
            if (header) header.style.display = 'block';
            if (nav) nav.style.display = 'flex';
            if (footer) footer.style.display = 'block';
        }
    }

    async exitXR() {
        if (webXRManager.xrSession) {
            webXRManager.endSession();
        }
        
        this.deactivateAllFeatures();
        this.currentMode = 'desktop';
        
        console.log('🏁 Exited XR mode');
    }

    showWelcomeMessage() {
        this.showInfoMessage(
            'Welcome to AR/VR Immersive Media Platform',
            `Platform initialized successfully! Device: ${this.deviceCapabilities.device}\n` +
            'Select a feature to begin your immersive experience.\n\n' +
            'Keyboard shortcuts:\n' +
            'Ctrl/Cmd + 1-7: Activate features\n' +
            'Esc: Exit XR mode'
        );
    }

    showInfoMessage(title, message) {
        console.log(`ℹ️ ${title}: ${message}`);
        
        // Show toast notification
        this.showToast(title, message, 'info');
    }

    showErrorMessage(title, message) {
        console.error(`❌ ${title}: ${message}`);
        
        // Show error toast
        this.showToast(title, message, 'error');
    }

    showToast(title, message, type = 'info') {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-header">
                <strong>${title}</strong>
                <button class="toast-close">&times;</button>
            </div>
            <div class="toast-body">${message}</div>
        `;
        
        // Add styles
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            max-width: 400px;
            background: ${type === 'error' ? '#ff4444' : '#4a90e2'};
            color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10000;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
        `;
        
        // Add to DOM
        document.body.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
        }, 100);
        
        // Add close handler
        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => {
            this.removeToast(toast);
        });
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            this.removeToast(toast);
        }, 5000);
    }

    removeToast(toast) {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }

    // Public API methods
    getActiveFeatures() {
        return Array.from(this.activeFeatures);
    }

    getCurrentMode() {
        return this.currentMode;
    }

    getDeviceCapabilities() {
        return { ...this.deviceCapabilities };
    }

    isFeatureSupported(featureName) {
        switch (featureName) {
            case 'vrCinema': return this.deviceCapabilities.vr;
            case 'arOverlay': return this.deviceCapabilities.ar;
            case 'spatialVideo': return true;
            case 'handTracking': return this.deviceCapabilities.handTracking;
            case 'eyeTracking': return this.deviceCapabilities.eyeTracking;
            case 'hapticFeedback': return this.deviceCapabilities.haptic;
            case 'mixedReality': return this.deviceCapabilities.ar || this.deviceCapabilities.vr;
            default: return false;
        }
    }

    async switchFeature(featureName) {
        if (this.isFeatureSupported(featureName)) {
            await this.activateFeature(featureName);
        } else {
            this.showErrorMessage('Feature Not Supported', `${featureName} is not supported on this device.`);
        }
    }

    // Demo methods for showcasing capabilities
    async runDemoSequence() {
        console.log('🎪 Running demo sequence...');
        
        const demos = [
            'hapticFeedback',
            'handTracking',
            'spatialVideo',
            'eyeTracking'
        ];
        
        for (const demo of demos) {
            if (this.isFeatureSupported(demo)) {
                await this.activateFeature(demo);
                await this.wait(3000); // Show each demo for 3 seconds
            }
        }
        
        this.showInfoMessage('Demo Complete', 'Demo sequence finished. Select any feature to continue exploring.');
    }

    wait(milliseconds) {
        return new Promise(resolve => setTimeout(resolve, milliseconds));
    }

    // Cleanup method
    destroy() {
        console.log('🗑️ Destroying AR/VR Media Platform...');
        
        // Deactivate all features
        this.deactivateAllFeatures();
        
        // End XR session if active
        if (webXRManager.xrSession) {
            webXRManager.endSession();
        }
        
        // Stop render loop
        if (webXRManager.renderer) {
            webXRManager.renderer.setAnimationLoop(null);
        }
        
        // Remove event listeners
        document.removeEventListener('keydown', this.handleKeyboardShortcuts);
        window.removeEventListener('resize', this.handleResize);
        
        console.log('✅ AR/VR Media Platform destroyed');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global app instance
    window.arvrMediaApp = new ARVRMediaApp();
});

// Export for external use
export default ARVRMediaApp;