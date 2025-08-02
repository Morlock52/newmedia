/**
 * AR Overlay System - Advanced AR information overlay for real objects
 * Optimized for Apple Vision Pro spatial computing capabilities
 */

import { webXRManager } from '../core/webxr-manager.js';

export class AROverlay {
    constructor() {
        this.overlays = new Map();
        this.trackedObjects = new Map();
        this.hitTestSource = null;
        this.anchors = new Map();
        this.objectDetector = null;
        this.depthSensing = null;
        this.planeDetection = null;
        this.handGestureRecognizer = null;
        this.spatialMapping = null;
        
        this.init();
    }

    async init() {
        console.log('🔍 Initializing AR Overlay System...');
        
        // Initialize object detection
        await this.initObjectDetection();
        
        // Setup plane detection
        this.setupPlaneDetection();
        
        // Initialize hit testing
        this.setupHitTesting();
        
        // Setup depth sensing
        this.setupDepthSensing();
        
        // Initialize hand gesture recognition
        this.setupHandGestureRecognition();
        
        // Setup spatial mapping
        this.setupSpatialMapping();
        
        console.log('✅ AR Overlay System initialized');
    }

    async initObjectDetection() {
        console.log('🎯 Initializing object detection...');
        
        // Object detection would use WebXR's computer vision APIs
        // or integrate with external ML services
        this.objectDetector = {
            isActive: false,
            detectedObjects: new Map(),
            confidenceThreshold: 0.7,
            supportedObjects: [
                'person', 'car', 'chair', 'table', 'laptop', 'phone',
                'book', 'bottle', 'cup', 'plant', 'tv', 'painting'
            ]
        };
        
        // Mock object detection for demonstration
        this.startMockObjectDetection();
    }

    startMockObjectDetection() {
        // Simulate detecting objects in the environment
        setInterval(() => {
            if (this.objectDetector.isActive) {
                this.simulateObjectDetection();
            }
        }, 2000);
    }

    simulateObjectDetection() {
        const objects = [
            { type: 'table', position: { x: 0, y: -0.5, z: -2 }, confidence: 0.9 },
            { type: 'chair', position: { x: -1, y: 0, z: -1.5 }, confidence: 0.8 },
            { type: 'laptop', position: { x: 0.2, y: -0.4, z: -1.8 }, confidence: 0.85 }
        ];
        
        objects.forEach(obj => {
            if (obj.confidence > this.objectDetector.confidenceThreshold) {
                this.addObjectOverlay(obj);
            }
        });
    }

    setupPlaneDetection() {
        console.log('📐 Setting up plane detection...');
        
        this.planeDetection = {
            detectedPlanes: new Map(),
            planeTypes: ['horizontal', 'vertical'],
            visualizePlanes: true
        };
    }

    setupHitTesting() {
        console.log('🎯 Setting up hit testing...');
        
        // Hit testing for placing overlays on surfaces
        this.hitTestSource = null;
    }

    setupDepthSensing() {
        console.log('📏 Setting up depth sensing...');
        
        this.depthSensing = {
            isSupported: false,
            depthTexture: null,
            occlusionEnabled: true
        };
    }

    setupHandGestureRecognition() {
        console.log('👋 Setting up hand gesture recognition...');
        
        this.handGestureRecognizer = {
            gestures: new Map(),
            recognizedGestures: [],
            confidenceThreshold: 0.8
        };
        
        // Define gesture patterns
        this.defineGesturePatterns();
    }

    defineGesturePatterns() {
        const gestures = {
            'point': {
                description: 'Pointing gesture to select objects',
                pattern: 'index_extended',
                callback: (target) => this.handlePointGesture(target)
            },
            'pinch': {
                description: 'Pinch gesture to grab/manipulate',
                pattern: 'thumb_index_close',
                callback: (target) => this.handlePinchGesture(target)
            },
            'tap': {
                description: 'Air tap gesture for activation',
                pattern: 'quick_pinch',
                callback: (target) => this.handleTapGesture(target)
            },
            'swipe': {
                description: 'Swipe gesture for navigation',
                pattern: 'hand_movement',
                callback: (direction) => this.handleSwipeGesture(direction)
            }
        };
        
        Object.entries(gestures).forEach(([name, gesture]) => {
            this.handGestureRecognizer.gestures.set(name, gesture);
        });
    }

    setupSpatialMapping() {
        console.log('🗺️ Setting up spatial mapping...');
        
        this.spatialMapping = {
            meshes: new Map(),
            spatialAnchors: new Map(),
            persistentAnchors: new Map(),
            roomScale: true
        };
    }

    async startARSession() {
        try {
            const session = await webXRManager.startARSession(
                ['local'], // required features
                [
                    'hit-test',
                    'anchors',
                    'depth-sensing',
                    'hand-tracking',
                    'dom-overlay',
                    'plane-detection'
                ] // optional features
            );
            
            // Initialize hit test source
            if (session.requestHitTestSource) {
                this.hitTestSource = await session.requestHitTestSource({
                    space: webXRManager.xrRefSpace
                });
            }
            
            // Start object detection
            this.objectDetector.isActive = true;
            
            console.log('📱 AR Overlay session started');
            return session;
        } catch (error) {
            console.error('Failed to start AR session:', error);
            throw error;
        }
    }

    addObjectOverlay(detectedObject) {
        const overlayId = `overlay_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        // Create overlay content based on object type
        const overlayData = this.generateOverlayContent(detectedObject);
        
        // Create 3D overlay element
        const overlay = this.create3DOverlay(overlayData, detectedObject.position);
        
        // Store overlay
        this.overlays.set(overlayId, {
            object: detectedObject,
            overlay: overlay,
            data: overlayData,
            isVisible: true,
            lastUpdate: Date.now()
        });
        
        // Add to scene
        webXRManager.scene.add(overlay);
        
        console.log(`📋 Added overlay for ${detectedObject.type}:`, overlayId);
        return overlayId;
    }

    generateOverlayContent(detectedObject) {
        const contentTemplates = {
            'table': {
                title: 'Coffee Table',
                info: 'Wooden surface • Perfect for drinks',
                actions: ['Get similar items', 'Measure dimensions', 'Add to wishlist'],
                color: 0x8b4513,
                icon: '🪑'
            },
            'chair': {
                title: 'Ergonomic Chair',
                info: 'Comfortable seating • Adjustable height',
                actions: ['Check reviews', 'Find replacement', 'Adjust settings'],
                color: 0x654321,
                icon: '🪑'
            },
            'laptop': {
                title: 'MacBook Pro',
                info: 'Apple M3 • 16GB RAM • 512GB SSD',
                actions: ['Check specs', 'Monitor usage', 'System info'],
                color: 0x666666,
                icon: '💻'
            },
            'plant': {
                title: 'Monstera Deliciosa',
                info: 'Indoor plant • Needs water every 7 days',
                actions: ['Care instructions', 'Watering schedule', 'Plant info'],
                color: 0x228b22,
                icon: '🌿'
            },
            'book': {
                title: 'Technical Manual',
                info: 'Last read: Chapter 5 • 67% complete',
                actions: ['Continue reading', 'Bookmark page', 'Find similar'],
                color: 0x8b0000,
                icon: '📚'
            }
        };
        
        return contentTemplates[detectedObject.type] || {
            title: 'Unknown Object',
            info: `Detected: ${detectedObject.type}`,
            actions: ['Identify object', 'Get info'],
            color: 0x808080,
            icon: '❓'
        };
    }

    create3DOverlay(overlayData, position) {
        const overlayGroup = new THREE.Group();
        
        // Main panel
        const panelGeometry = new THREE.PlaneGeometry(2, 1.2);
        const panelMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            transparent: true,
            opacity: 0.8
        });
        const panel = new THREE.Mesh(panelGeometry, panelMaterial);
        overlayGroup.add(panel);
        
        // Border
        const borderGeometry = new THREE.PlaneGeometry(2.1, 1.3);
        const borderMaterial = new THREE.MeshBasicMaterial({
            color: overlayData.color,
            transparent: true,
            opacity: 0.6
        });
        const border = new THREE.Mesh(borderGeometry, borderMaterial);
        border.position.z = -0.001;
        overlayGroup.add(border);
        
        // Icon
        const iconGeometry = new THREE.PlaneGeometry(0.3, 0.3);
        const iconTexture = this.createTextTexture(overlayData.icon, 64, '#ffffff');
        const iconMaterial = new THREE.MeshBasicMaterial({
            map: iconTexture,
            transparent: true
        });
        const icon = new THREE.Mesh(iconGeometry, iconMaterial);
        icon.position.set(-0.7, 0.3, 0.001);
        overlayGroup.add(icon);
        
        // Title text
        const titleTexture = this.createTextTexture(overlayData.title, 32, '#ffffff');
        const titleGeometry = new THREE.PlaneGeometry(1.2, 0.2);
        const titleMaterial = new THREE.MeshBasicMaterial({
            map: titleTexture,
            transparent: true
        });
        const title = new THREE.Mesh(titleGeometry, titleMaterial);
        title.position.set(0.2, 0.3, 0.001);
        overlayGroup.add(title);
        
        // Info text
        const infoTexture = this.createTextTexture(overlayData.info, 16, '#cccccc');
        const infoGeometry = new THREE.PlaneGeometry(1.4, 0.15);
        const infoMaterial = new THREE.MeshBasicMaterial({
            map: infoTexture,
            transparent: true
        });
        const info = new THREE.Mesh(infoGeometry, infoMaterial);
        info.position.set(0.2, 0.05, 0.001);
        overlayGroup.add(info);
        
        // Action buttons
        overlayData.actions.forEach((action, index) => {
            const buttonY = -0.2 - (index * 0.25);
            const buttonGroup = this.createActionButton(action, buttonY);
            overlayGroup.add(buttonGroup);
        });
        
        // Position overlay
        overlayGroup.position.set(
            position.x,
            position.y + 0.5, // Float above object
            position.z
        );
        
        // Billboard effect - always face camera
        overlayGroup.userData.isBillboard = true;
        
        return overlayGroup;
    }

    createActionButton(text, yPosition) {
        const buttonGroup = new THREE.Group();
        
        // Button background
        const buttonGeometry = new THREE.PlaneGeometry(1.6, 0.2);
        const buttonMaterial = new THREE.MeshBasicMaterial({
            color: 0x4a90e2,
            transparent: true,
            opacity: 0.7
        });
        const button = new THREE.Mesh(buttonGeometry, buttonMaterial);
        button.position.y = yPosition;
        buttonGroup.add(button);
        
        // Button text
        const textTexture = this.createTextTexture(text, 16, '#ffffff');
        const textGeometry = new THREE.PlaneGeometry(1.4, 0.15);
        const textMaterial = new THREE.MeshBasicMaterial({
            map: textTexture,
            transparent: true
        });
        const textMesh = new THREE.Mesh(textGeometry, textMaterial);
        textMesh.position.set(0, yPosition, 0.001);
        buttonGroup.add(textMesh);
        
        // Add interaction data
        button.userData = {
            isButton: true,
            action: text,
            onSelect: () => this.handleButtonAction(text)
        };
        
        return buttonGroup;
    }

    createTextTexture(text, fontSize = 32, color = '#ffffff') {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = 512;
        canvas.height = 128;
        
        // Setup text rendering
        context.fillStyle = 'transparent';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = color;
        context.font = `${fontSize}px Arial, sans-serif`;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        
        // Draw text
        context.fillText(text, canvas.width / 2, canvas.height / 2);
        
        // Create texture
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        
        return texture;
    }

    updateOverlays(frame) {
        if (!webXRManager.xrSession) return;
        
        // Update billboard overlays to face camera
        this.overlays.forEach((overlayInfo, overlayId) => {
            if (overlayInfo.overlay.userData.isBillboard) {
                overlayInfo.overlay.lookAt(webXRManager.camera.position);
            }
        });
        
        // Update hit testing
        this.updateHitTesting(frame);
        
        // Update hand gesture recognition
        this.updateHandGestureRecognition(frame);
        
        // Update spatial anchors
        this.updateSpatialAnchors(frame);
    }

    updateHitTesting(frame) {
        if (!this.hitTestSource) return;
        
        const hitTestResults = frame.getHitTestResults(this.hitTestSource);
        if (hitTestResults.length > 0) {
            const hit = hitTestResults[0];
            const pose = hit.getPose(webXRManager.xrRefSpace);
            
            // Process hit test results for overlay placement
            this.processHitTestResult(pose);
        }
    }

    processHitTestResult(pose) {
        // Handle hit test results for interactive overlay placement
        console.debug('Hit test result:', pose.transform.position);
    }

    updateHandGestureRecognition(frame) {
        if (!webXRManager.isHandTrackingSupported) return;
        
        webXRManager.updateHandTracking(frame);
        
        // Analyze hand poses for gesture recognition
        this.analyzeHandGestures();
    }

    analyzeHandGestures() {
        const leftHand = webXRManager.getHandJoint('left', 'index-finger-tip');
        const rightHand = webXRManager.getHandJoint('right', 'index-finger-tip');
        
        if (leftHand || rightHand) {
            // Implement gesture recognition logic
            this.recognizePointingGesture(leftHand, rightHand);
            this.recognizePinchGesture();
        }
    }

    recognizePointingGesture(leftHand, rightHand) {
        // Check if user is pointing at an overlay
        const activeHand = rightHand || leftHand;
        if (!activeHand) return;
        
        const raycaster = new THREE.Raycaster();
        const handPosition = activeHand.position;
        const handDirection = new THREE.Vector3(0, 0, -1); // Simplified pointing direction
        
        raycaster.set(handPosition, handDirection);
        
        // Check intersection with overlays
        const overlayMeshes = Array.from(this.overlays.values()).map(info => info.overlay);
        const intersections = raycaster.intersectObjects(overlayMeshes, true);
        
        if (intersections.length > 0) {
            this.handleOverlayInteraction(intersections[0]);
        }
    }

    recognizePinchGesture() {
        // Implement pinch gesture recognition
        console.debug('Analyzing pinch gesture...');
    }

    handleOverlayInteraction(intersection) {
        const object = intersection.object;
        
        if (object.userData.isButton) {
            // Highlight button
            object.material.opacity = 1.0;
            
            // Trigger haptic feedback
            webXRManager.pulseHaptic(0.3, 50);
            
            console.log('Button highlighted:', object.userData.action);
        }
    }

    handlePointGesture(target) {
        console.log('Point gesture detected at:', target);
        // Implement pointing gesture handler
    }

    handlePinchGesture(target) {
        console.log('Pinch gesture detected at:', target);
        // Implement pinch gesture handler
    }

    handleTapGesture(target) {
        console.log('Tap gesture detected at:', target);
        // Implement tap gesture handler
    }

    handleSwipeGesture(direction) {
        console.log('Swipe gesture detected:', direction);
        // Implement swipe gesture handler
    }

    handleButtonAction(action) {
        console.log('Button action:', action);
        
        // Trigger haptic feedback
        webXRManager.pulseHaptic(0.5, 100);
        
        // Execute action based on type
        switch (action) {
            case 'Get similar items':
                this.showSimilarItems();
                break;
            case 'Measure dimensions':
                this.startMeasurement();
                break;
            case 'Check specs':
                this.showSpecifications();
                break;
            case 'Care instructions':
                this.showCareInstructions();
                break;
            default:
                console.log('Action not implemented:', action);
        }
    }

    showSimilarItems() {
        console.log('🛒 Showing similar items...');
        // Implement shopping integration
    }

    startMeasurement() {
        console.log('📏 Starting AR measurement...');
        // Implement AR measurement tool
    }

    showSpecifications() {
        console.log('📊 Showing device specifications...');
        // Implement device info retrieval
    }

    showCareInstructions() {
        console.log('🌱 Showing plant care instructions...');
        // Implement care instruction overlay
    }

    updateSpatialAnchors(frame) {
        // Update persistent spatial anchors
        this.spatialMapping.spatialAnchors.forEach((anchor, id) => {
            // Update anchor positions and maintain persistence
            console.debug('Updating spatial anchor:', id);
        });
    }

    removeOverlay(overlayId) {
        const overlayInfo = this.overlays.get(overlayId);
        if (overlayInfo) {
            webXRManager.scene.remove(overlayInfo.overlay);
            this.overlays.delete(overlayId);
            console.log('🗑️ Removed overlay:', overlayId);
        }
    }

    clearAllOverlays() {
        this.overlays.forEach((overlayInfo, overlayId) => {
            webXRManager.scene.remove(overlayInfo.overlay);
        });
        this.overlays.clear();
        console.log('🧹 Cleared all overlays');
    }

    setOverlayVisibility(overlayId, visible) {
        const overlayInfo = this.overlays.get(overlayId);
        if (overlayInfo) {
            overlayInfo.overlay.visible = visible;
            overlayInfo.isVisible = visible;
        }
    }

    // Apple Vision Pro specific optimizations
    optimizeForVisionPro() {
        console.log('🥽 Optimizing for Apple Vision Pro...');
        
        // Enable high-resolution rendering
        webXRManager.renderer.setPixelRatio(2);
        
        // Optimize overlay rendering for spatial computing
        this.overlays.forEach((overlayInfo) => {
            // Enable higher quality materials for Vision Pro
            overlayInfo.overlay.traverse((child) => {
                if (child.material) {
                    child.material.precision = 'highp';
                }
            });
        });
    }

    // Meta Quest 3 specific optimizations
    optimizeForQuest3() {
        console.log('🥽 Optimizing for Meta Quest 3...');
        
        // Balance quality and performance for Quest 3
        webXRManager.renderer.setPixelRatio(1.5);
        
        // Optimize for mixed reality passthrough
        this.enablePassthroughOptimization();
    }

    enablePassthroughOptimization() {
        // Optimize overlays for passthrough mixed reality
        console.log('🌐 Passthrough optimization enabled');
    }
}

// Export AR Overlay class
export default AROverlay;