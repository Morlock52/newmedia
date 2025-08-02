/**
 * Real Hand Tracking Implementation
 * Production-ready hand tracking with gesture recognition
 * Supports Apple Vision Pro transient-pointer and Meta Quest full skeletal tracking
 */

import * as THREE from 'three';
import { webXRManager } from '../core/webxr-manager-v2.js';

// Hand joint indices for WebXR Hand Input
const JOINT_INDICES = {
    WRIST: 0,
    THUMB_METACARPAL: 1, THUMB_PHALANX_PROXIMAL: 2, THUMB_PHALANX_DISTAL: 3, THUMB_TIP: 4,
    INDEX_METACARPAL: 5, INDEX_PHALANX_PROXIMAL: 6, INDEX_PHALANX_INTERMEDIATE: 7, 
    INDEX_PHALANX_DISTAL: 8, INDEX_TIP: 9,
    MIDDLE_METACARPAL: 10, MIDDLE_PHALANX_PROXIMAL: 11, MIDDLE_PHALANX_INTERMEDIATE: 12,
    MIDDLE_PHALANX_DISTAL: 13, MIDDLE_TIP: 14,
    RING_METACARPAL: 15, RING_PHALANX_PROXIMAL: 16, RING_PHALANX_INTERMEDIATE: 17,
    RING_PHALANX_DISTAL: 18, RING_TIP: 19,
    PINKY_METACARPAL: 20, PINKY_PHALANX_PROXIMAL: 21, PINKY_PHALANX_INTERMEDIATE: 22,
    PINKY_PHALANX_DISTAL: 23, PINKY_TIP: 24
};

export class RealHandTracking {
    constructor() {
        // Hand data storage
        this.hands = new Map();
        this.handModels = new Map();
        
        // Gesture recognition
        this.gestureRecognizer = new GestureRecognizer();
        this.activeGestures = new Map();
        
        // Interaction system
        this.interactionSystem = new HandInteractionSystem();
        
        // Performance optimization
        this.updateRate = 90; // Hz
        this.lastUpdateTime = 0;
        
        // Visualization options
        this.options = {
            showJoints: true,
            showBones: true,
            showMesh: false,
            jointScale: 1.0,
            meshOpacity: 0.7
        };
        
        // Materials
        this.materials = {
            leftJoint: new THREE.MeshPhongMaterial({
                color: 0x4a90e2,
                emissive: 0x1a4a8a,
                shininess: 100
            }),
            rightJoint: new THREE.MeshPhongMaterial({
                color: 0xe24a90,
                emissive: 0x8a1a4a,
                shininess: 100
            }),
            bone: new THREE.LineBasicMaterial({
                color: 0xffffff,
                opacity: 0.6,
                transparent: true
            }),
            mesh: new THREE.MeshPhongMaterial({
                color: 0xfdbcb4,
                transparent: true,
                opacity: 0.7,
                side: THREE.DoubleSide
            })
        };
        
        this.init();
    }

    init() {
        console.log('🖐️ Initializing Real Hand Tracking System...');
        
        // Listen for WebXR events
        window.addEventListener('xr-session-started', () => this.onSessionStarted());
        window.addEventListener('xr-session-ended', () => this.onSessionEnded());
        
        // Setup hand models
        this.setupHandModels();
        
        console.log('✅ Hand Tracking System ready');
    }

    setupHandModels() {
        ['left', 'right'].forEach(handedness => {
            const handModel = this.createHandModel(handedness);
            this.handModels.set(handedness, handModel);
            
            // Add to scene but keep hidden initially
            handModel.visible = false;
            webXRManager.scene.add(handModel);
        });
    }

    createHandModel(handedness) {
        const handGroup = new THREE.Group();
        handGroup.name = `hand-${handedness}`;
        
        // Create joint meshes
        const joints = new Map();
        const jointMeshes = [];
        
        for (let i = 0; i < 25; i++) {
            // Different sizes for different joints
            let radius = 0.008;
            if (i === JOINT_INDICES.WRIST) radius = 0.012;
            else if (i % 5 === 4) radius = 0.006; // Fingertips
            
            const jointMesh = new THREE.Mesh(
                new THREE.SphereGeometry(radius, 16, 12),
                handedness === 'left' ? this.materials.leftJoint : this.materials.rightJoint
            );
            
            jointMesh.castShadow = true;
            jointMesh.receiveShadow = true;
            jointMesh.name = `joint-${i}`;
            
            joints.set(i, jointMesh);
            jointMeshes.push(jointMesh);
            handGroup.add(jointMesh);
        }
        
        // Create bone connections
        const bones = this.createBoneConnections(jointMeshes);
        bones.forEach(bone => handGroup.add(bone));
        
        // Store references
        handGroup.userData = {
            joints,
            jointMeshes,
            bones,
            handedness
        };
        
        return handGroup;
    }

    createBoneConnections(jointMeshes) {
        const bones = [];
        
        // Define bone connections
        const connections = [
            // Thumb
            [JOINT_INDICES.WRIST, JOINT_INDICES.THUMB_METACARPAL],
            [JOINT_INDICES.THUMB_METACARPAL, JOINT_INDICES.THUMB_PHALANX_PROXIMAL],
            [JOINT_INDICES.THUMB_PHALANX_PROXIMAL, JOINT_INDICES.THUMB_PHALANX_DISTAL],
            [JOINT_INDICES.THUMB_PHALANX_DISTAL, JOINT_INDICES.THUMB_TIP],
            
            // Index finger
            [JOINT_INDICES.WRIST, JOINT_INDICES.INDEX_METACARPAL],
            [JOINT_INDICES.INDEX_METACARPAL, JOINT_INDICES.INDEX_PHALANX_PROXIMAL],
            [JOINT_INDICES.INDEX_PHALANX_PROXIMAL, JOINT_INDICES.INDEX_PHALANX_INTERMEDIATE],
            [JOINT_INDICES.INDEX_PHALANX_INTERMEDIATE, JOINT_INDICES.INDEX_PHALANX_DISTAL],
            [JOINT_INDICES.INDEX_PHALANX_DISTAL, JOINT_INDICES.INDEX_TIP],
            
            // Middle finger
            [JOINT_INDICES.WRIST, JOINT_INDICES.MIDDLE_METACARPAL],
            [JOINT_INDICES.MIDDLE_METACARPAL, JOINT_INDICES.MIDDLE_PHALANX_PROXIMAL],
            [JOINT_INDICES.MIDDLE_PHALANX_PROXIMAL, JOINT_INDICES.MIDDLE_PHALANX_INTERMEDIATE],
            [JOINT_INDICES.MIDDLE_PHALANX_INTERMEDIATE, JOINT_INDICES.MIDDLE_PHALANX_DISTAL],
            [JOINT_INDICES.MIDDLE_PHALANX_DISTAL, JOINT_INDICES.MIDDLE_TIP],
            
            // Ring finger
            [JOINT_INDICES.WRIST, JOINT_INDICES.RING_METACARPAL],
            [JOINT_INDICES.RING_METACARPAL, JOINT_INDICES.RING_PHALANX_PROXIMAL],
            [JOINT_INDICES.RING_PHALANX_PROXIMAL, JOINT_INDICES.RING_PHALANX_INTERMEDIATE],
            [JOINT_INDICES.RING_PHALANX_INTERMEDIATE, JOINT_INDICES.RING_PHALANX_DISTAL],
            [JOINT_INDICES.RING_PHALANX_DISTAL, JOINT_INDICES.RING_TIP],
            
            // Pinky finger
            [JOINT_INDICES.WRIST, JOINT_INDICES.PINKY_METACARPAL],
            [JOINT_INDICES.PINKY_METACARPAL, JOINT_INDICES.PINKY_PHALANX_PROXIMAL],
            [JOINT_INDICES.PINKY_PHALANX_PROXIMAL, JOINT_INDICES.PINKY_PHALANX_INTERMEDIATE],
            [JOINT_INDICES.PINKY_PHALANX_INTERMEDIATE, JOINT_INDICES.PINKY_PHALANX_DISTAL],
            [JOINT_INDICES.PINKY_PHALANX_DISTAL, JOINT_INDICES.PINKY_TIP]
        ];
        
        connections.forEach(([start, end]) => {
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(6); // 2 vertices * 3 coordinates
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const bone = new THREE.Line(geometry, this.materials.bone);
            bone.name = `bone-${start}-${end}`;
            bone.userData = {
                startJoint: jointMeshes[start],
                endJoint: jointMeshes[end]
            };
            
            bones.push(bone);
        });
        
        return bones;
    }

    onSessionStarted() {
        console.log('🎮 Hand tracking session started');
        
        // Check if hand tracking is available
        if (webXRManager.features.handTracking) {
            console.log('✅ Native hand tracking available');
        } else if (webXRManager.platform.isVisionPro) {
            console.log('👁️ Using Vision Pro transient-pointer tracking');
        } else {
            console.warn('⚠️ Limited hand tracking available');
        }
        
        // Start update loop
        this.startUpdateLoop();
    }

    onSessionEnded() {
        console.log('🛑 Hand tracking session ended');
        
        // Hide all hand models
        this.handModels.forEach(model => {
            model.visible = false;
        });
        
        // Clear hand data
        this.hands.clear();
        this.activeGestures.clear();
    }

    startUpdateLoop() {
        const update = (timestamp) => {
            if (!webXRManager.isSessionActive()) return;
            
            // Throttle updates for performance
            if (timestamp - this.lastUpdateTime < 1000 / this.updateRate) {
                requestAnimationFrame(update);
                return;
            }
            
            this.lastUpdateTime = timestamp;
            
            // Update hand tracking
            this.updateHandTracking();
            
            // Continue loop
            requestAnimationFrame(update);
        };
        
        requestAnimationFrame(update);
    }

    updateHandTracking() {
        if (!webXRManager.xrFrame) return;
        
        // Update each tracked hand
        webXRManager.hands.forEach((handData, handedness) => {
            this.updateHand(handedness, handData);
        });
        
        // Update transient pointers (Vision Pro)
        if (webXRManager.platform.isVisionPro) {
            this.updateTransientPointers();
        }
        
        // Process gestures
        this.processGestures();
        
        // Update interactions
        this.interactionSystem.update(this.hands);
    }

    updateHand(handedness, handData) {
        const handModel = this.handModels.get(handedness);
        if (!handModel) return;
        
        const { joints, jointMeshes, bones } = handModel.userData;
        let handVisible = false;
        
        // Update joint positions
        handData.joints.forEach((jointPose, jointName) => {
            const jointIndex = this.getJointIndex(jointName);
            if (jointIndex === -1) return;
            
            const jointMesh = joints.get(jointIndex);
            if (!jointMesh) return;
            
            // Update position and orientation
            jointMesh.position.copy(jointPose.position);
            jointMesh.quaternion.copy(jointPose.orientation);
            
            // Update scale based on confidence or radius
            if (jointPose.radius) {
                const scale = (jointPose.radius / 0.008) * this.options.jointScale;
                jointMesh.scale.setScalar(scale);
            }
            
            jointMesh.visible = this.options.showJoints;
            handVisible = true;
        });
        
        // Update bone connections
        if (this.options.showBones) {
            bones.forEach(bone => {
                const positions = bone.geometry.attributes.position.array;
                const start = bone.userData.startJoint.position;
                const end = bone.userData.endJoint.position;
                
                positions[0] = start.x;
                positions[1] = start.y;
                positions[2] = start.z;
                positions[3] = end.x;
                positions[4] = end.y;
                positions[5] = end.z;
                
                bone.geometry.attributes.position.needsUpdate = true;
                bone.visible = this.options.showBones && handVisible;
            });
        }
        
        // Update hand visibility
        handModel.visible = handVisible;
        
        // Store processed hand data
        if (!this.hands.has(handedness)) {
            this.hands.set(handedness, new HandData(handedness));
        }
        
        const hand = this.hands.get(handedness);
        hand.updateFromJoints(handData.joints);
    }

    updateTransientPointers() {
        // Handle Vision Pro eye + pinch input
        ['left', 'right', 'none'].forEach(handedness => {
            const ray = webXRManager.getTransientPointerRay(handedness);
            const isActive = webXRManager.isTransientPointerActive(handedness);
            
            if (ray && isActive) {
                // Simulate pinch gesture for transient pointer
                if (!this.hands.has(handedness)) {
                    this.hands.set(handedness, new HandData(handedness));
                }
                
                const hand = this.hands.get(handedness);
                hand.simulateTransientPointer(ray, isActive);
            }
        });
    }

    processGestures() {
        this.hands.forEach((hand, handedness) => {
            const gestures = this.gestureRecognizer.recognize(hand);
            
            gestures.forEach(gesture => {
                const gestureKey = `${handedness}-${gesture.type}`;
                const wasActive = this.activeGestures.has(gestureKey);
                
                if (gesture.confidence > 0.7) {
                    if (!wasActive) {
                        // Gesture started
                        this.onGestureStart(handedness, gesture);
                    } else {
                        // Gesture continuing
                        this.onGestureUpdate(handedness, gesture);
                    }
                    
                    this.activeGestures.set(gestureKey, gesture);
                } else if (wasActive) {
                    // Gesture ended
                    this.onGestureEnd(handedness, gesture);
                    this.activeGestures.delete(gestureKey);
                }
            });
        });
    }

    onGestureStart(handedness, gesture) {
        console.log(`✋ Gesture started: ${gesture.type} (${handedness})`);
        
        // Haptic feedback
        webXRManager.vibrate(handedness, 0.5, 50);
        
        // Dispatch event
        window.dispatchEvent(new CustomEvent('hand-gesture-start', {
            detail: { handedness, gesture }
        }));
    }

    onGestureUpdate(handedness, gesture) {
        // Dispatch event
        window.dispatchEvent(new CustomEvent('hand-gesture-update', {
            detail: { handedness, gesture }
        }));
    }

    onGestureEnd(handedness, gesture) {
        console.log(`✋ Gesture ended: ${gesture.type} (${handedness})`);
        
        // Haptic feedback
        webXRManager.vibrate(handedness, 0.3, 30);
        
        // Dispatch event
        window.dispatchEvent(new CustomEvent('hand-gesture-end', {
            detail: { handedness, gesture }
        }));
    }

    getJointIndex(jointName) {
        // Convert joint name to index
        const nameToIndex = {
            'wrist': JOINT_INDICES.WRIST,
            'thumb-metacarpal': JOINT_INDICES.THUMB_METACARPAL,
            'thumb-phalanx-proximal': JOINT_INDICES.THUMB_PHALANX_PROXIMAL,
            'thumb-phalanx-distal': JOINT_INDICES.THUMB_PHALANX_DISTAL,
            'thumb-tip': JOINT_INDICES.THUMB_TIP,
            'index-finger-metacarpal': JOINT_INDICES.INDEX_METACARPAL,
            'index-finger-phalanx-proximal': JOINT_INDICES.INDEX_PHALANX_PROXIMAL,
            'index-finger-phalanx-intermediate': JOINT_INDICES.INDEX_PHALANX_INTERMEDIATE,
            'index-finger-phalanx-distal': JOINT_INDICES.INDEX_PHALANX_DISTAL,
            'index-finger-tip': JOINT_INDICES.INDEX_TIP,
            'middle-finger-metacarpal': JOINT_INDICES.MIDDLE_METACARPAL,
            'middle-finger-phalanx-proximal': JOINT_INDICES.MIDDLE_PHALANX_PROXIMAL,
            'middle-finger-phalanx-intermediate': JOINT_INDICES.MIDDLE_PHALANX_INTERMEDIATE,
            'middle-finger-phalanx-distal': JOINT_INDICES.MIDDLE_PHALANX_DISTAL,
            'middle-finger-tip': JOINT_INDICES.MIDDLE_TIP,
            'ring-finger-metacarpal': JOINT_INDICES.RING_METACARPAL,
            'ring-finger-phalanx-proximal': JOINT_INDICES.RING_PHALANX_PROXIMAL,
            'ring-finger-phalanx-intermediate': JOINT_INDICES.RING_PHALANX_INTERMEDIATE,
            'ring-finger-phalanx-distal': JOINT_INDICES.RING_PHALANX_DISTAL,
            'ring-finger-tip': JOINT_INDICES.RING_TIP,
            'pinky-finger-metacarpal': JOINT_INDICES.PINKY_METACARPAL,
            'pinky-finger-phalanx-proximal': JOINT_INDICES.PINKY_PHALANX_PROXIMAL,
            'pinky-finger-phalanx-intermediate': JOINT_INDICES.PINKY_PHALANX_INTERMEDIATE,
            'pinky-finger-phalanx-distal': JOINT_INDICES.PINKY_PHALANX_DISTAL,
            'pinky-finger-tip': JOINT_INDICES.PINKY_TIP
        };
        
        return nameToIndex[jointName] ?? -1;
    }

    // Public API
    
    setVisualizationOptions(options) {
        Object.assign(this.options, options);
    }
    
    getHandData(handedness) {
        return this.hands.get(handedness);
    }
    
    getActiveGestures() {
        return Array.from(this.activeGestures.values());
    }
    
    addInteractableObject(object) {
        this.interactionSystem.addObject(object);
    }
    
    removeInteractableObject(object) {
        this.interactionSystem.removeObject(object);
    }
}

// Hand data class
class HandData {
    constructor(handedness) {
        this.handedness = handedness;
        this.joints = new Map();
        this.velocity = new THREE.Vector3();
        this.lastPosition = new THREE.Vector3();
        this.lastUpdateTime = 0;
        
        // Gesture-specific data
        this.fingerCurl = new Map();
        this.fingerSplay = new Map();
        this.palmNormal = new THREE.Vector3(0, 1, 0);
        this.palmPosition = new THREE.Vector3();
    }
    
    updateFromJoints(jointData) {
        // Store joint data
        jointData.forEach((pose, jointName) => {
            this.joints.set(jointName, {
                position: pose.position.clone(),
                orientation: pose.orientation.clone(),
                radius: pose.radius
            });
        });
        
        // Calculate derived data
        this.calculateHandMetrics();
        
        // Update velocity
        const currentTime = performance.now();
        if (this.lastUpdateTime > 0) {
            const deltaTime = (currentTime - this.lastUpdateTime) / 1000;
            const wrist = this.joints.get('wrist');
            
            if (wrist) {
                this.velocity.subVectors(wrist.position, this.lastPosition);
                this.velocity.divideScalar(deltaTime);
                this.lastPosition.copy(wrist.position);
            }
        }
        this.lastUpdateTime = currentTime;
    }
    
    simulateTransientPointer(ray, isActive) {
        // Simulate hand data for Vision Pro transient pointer
        // This creates a virtual pinch gesture at the ray origin
        
        const pinchPosition = ray.origin;
        
        // Simulate thumb and index finger positions for pinch
        this.joints.set('thumb-tip', {
            position: pinchPosition.clone(),
            orientation: new THREE.Quaternion(),
            radius: 0.008
        });
        
        this.joints.set('index-finger-tip', {
            position: pinchPosition.clone().add(new THREE.Vector3(0.01, 0, 0)),
            orientation: new THREE.Quaternion(),
            radius: 0.008
        });
        
        // Set pinch state
        this.fingerCurl.set('thumb', isActive ? 1.0 : 0.0);
        this.fingerCurl.set('index', isActive ? 1.0 : 0.0);
    }
    
    calculateHandMetrics() {
        // Calculate finger curl (0 = extended, 1 = fully curled)
        this.calculateFingerCurl();
        
        // Calculate finger splay (spread)
        this.calculateFingerSplay();
        
        // Calculate palm orientation
        this.calculatePalmOrientation();
    }
    
    calculateFingerCurl() {
        const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
        
        fingers.forEach(finger => {
            const joints = this.getFingerJoints(finger);
            if (joints.length < 3) return;
            
            // Calculate angle between joints
            let totalAngle = 0;
            for (let i = 0; i < joints.length - 2; i++) {
                const v1 = new THREE.Vector3().subVectors(joints[i + 1].position, joints[i].position);
                const v2 = new THREE.Vector3().subVectors(joints[i + 2].position, joints[i + 1].position);
                const angle = v1.angleTo(v2);
                totalAngle += angle;
            }
            
            // Normalize to 0-1 range
            const maxCurl = Math.PI * 0.8; // ~144 degrees
            const curl = Math.min(totalAngle / maxCurl, 1.0);
            this.fingerCurl.set(finger, curl);
        });
    }
    
    calculateFingerSplay() {
        // Calculate spread between adjacent fingers
        const fingerPairs = [
            ['index', 'middle'],
            ['middle', 'ring'],
            ['ring', 'pinky']
        ];
        
        fingerPairs.forEach(([finger1, finger2]) => {
            const tip1 = this.joints.get(`${finger1}-finger-tip`);
            const tip2 = this.joints.get(`${finger2}-finger-tip`);
            
            if (tip1 && tip2) {
                const distance = tip1.position.distanceTo(tip2.position);
                const normalizedDistance = distance / 0.1; // Normalize to ~10cm max spread
                this.fingerSplay.set(`${finger1}-${finger2}`, normalizedDistance);
            }
        });
    }
    
    calculatePalmOrientation() {
        const wrist = this.joints.get('wrist');
        const indexBase = this.joints.get('index-finger-metacarpal');
        const pinkyBase = this.joints.get('pinky-finger-metacarpal');
        
        if (wrist && indexBase && pinkyBase) {
            // Calculate palm position (center of palm)
            this.palmPosition.copy(wrist.position);
            this.palmPosition.add(indexBase.position);
            this.palmPosition.add(pinkyBase.position);
            this.palmPosition.divideScalar(3);
            
            // Calculate palm normal
            const v1 = new THREE.Vector3().subVectors(indexBase.position, wrist.position);
            const v2 = new THREE.Vector3().subVectors(pinkyBase.position, wrist.position);
            this.palmNormal.crossVectors(v1, v2).normalize();
            
            // Adjust for handedness
            if (this.handedness === 'right') {
                this.palmNormal.negate();
            }
        }
    }
    
    getFingerJoints(finger) {
        const jointNames = {
            'thumb': ['thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip'],
            'index': ['index-finger-metacarpal', 'index-finger-phalanx-proximal', 
                     'index-finger-phalanx-intermediate', 'index-finger-phalanx-distal', 'index-finger-tip'],
            'middle': ['middle-finger-metacarpal', 'middle-finger-phalanx-proximal',
                      'middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal', 'middle-finger-tip'],
            'ring': ['ring-finger-metacarpal', 'ring-finger-phalanx-proximal',
                    'ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal', 'ring-finger-tip'],
            'pinky': ['pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal',
                     'pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal', 'pinky-finger-tip']
        };
        
        const names = jointNames[finger] || [];
        return names.map(name => this.joints.get(name)).filter(joint => joint);
    }
}

// Gesture recognizer class
class GestureRecognizer {
    constructor() {
        this.gestures = new Map();
        this.initializeGestures();
    }
    
    initializeGestures() {
        // Define gesture patterns
        this.gestures.set('pinch', {
            check: (hand) => this.checkPinch(hand),
            priority: 1
        });
        
        this.gestures.set('point', {
            check: (hand) => this.checkPoint(hand),
            priority: 2
        });
        
        this.gestures.set('thumbs_up', {
            check: (hand) => this.checkThumbsUp(hand),
            priority: 3
        });
        
        this.gestures.set('fist', {
            check: (hand) => this.checkFist(hand),
            priority: 4
        });
        
        this.gestures.set('open_palm', {
            check: (hand) => this.checkOpenPalm(hand),
            priority: 5
        });
        
        this.gestures.set('peace', {
            check: (hand) => this.checkPeace(hand),
            priority: 6
        });
    }
    
    recognize(hand) {
        const detectedGestures = [];
        
        // Check each gesture
        this.gestures.forEach((gesture, type) => {
            const result = gesture.check(hand);
            if (result.detected) {
                detectedGestures.push({
                    type,
                    confidence: result.confidence,
                    data: result.data || {},
                    priority: gesture.priority
                });
            }
        });
        
        // Sort by priority and confidence
        detectedGestures.sort((a, b) => {
            if (a.priority !== b.priority) return a.priority - b.priority;
            return b.confidence - a.confidence;
        });
        
        return detectedGestures;
    }
    
    checkPinch(hand) {
        const thumbTip = hand.joints.get('thumb-tip');
        const indexTip = hand.joints.get('index-finger-tip');
        
        if (!thumbTip || !indexTip) {
            return { detected: false };
        }
        
        const distance = thumbTip.position.distanceTo(indexTip.position);
        const isPinching = distance < 0.03; // 3cm threshold
        
        return {
            detected: isPinching,
            confidence: isPinching ? Math.max(0, 1 - distance / 0.03) : 0,
            data: { distance }
        };
    }
    
    checkPoint(hand) {
        const indexCurl = hand.fingerCurl.get('index') || 1;
        const othersCurled = ['middle', 'ring', 'pinky'].every(finger => 
            (hand.fingerCurl.get(finger) || 0) > 0.7
        );
        
        const isPointing = indexCurl < 0.3 && othersCurled;
        
        return {
            detected: isPointing,
            confidence: isPointing ? (1 - indexCurl) * 0.9 : 0
        };
    }
    
    checkThumbsUp(hand) {
        const thumbCurl = hand.fingerCurl.get('thumb') || 1;
        const othersCurled = ['index', 'middle', 'ring', 'pinky'].every(finger =>
            (hand.fingerCurl.get(finger) || 0) > 0.7
        );
        
        // Check if thumb is pointing up
        const thumbTip = hand.joints.get('thumb-tip');
        const wrist = hand.joints.get('wrist');
        let thumbUp = false;
        
        if (thumbTip && wrist) {
            const thumbDir = new THREE.Vector3().subVectors(thumbTip.position, wrist.position).normalize();
            thumbUp = thumbDir.y > 0.7;
        }
        
        const isThumbsUp = thumbCurl < 0.3 && othersCurled && thumbUp;
        
        return {
            detected: isThumbsUp,
            confidence: isThumbsUp ? 0.9 : 0
        };
    }
    
    checkFist(hand) {
        const allCurled = ['thumb', 'index', 'middle', 'ring', 'pinky'].every(finger =>
            (hand.fingerCurl.get(finger) || 0) > 0.8
        );
        
        return {
            detected: allCurled,
            confidence: allCurled ? 0.95 : 0
        };
    }
    
    checkOpenPalm(hand) {
        const allExtended = ['thumb', 'index', 'middle', 'ring', 'pinky'].every(finger =>
            (hand.fingerCurl.get(finger) || 1) < 0.2
        );
        
        return {
            detected: allExtended,
            confidence: allExtended ? 0.9 : 0
        };
    }
    
    checkPeace(hand) {
        const indexCurl = hand.fingerCurl.get('index') || 1;
        const middleCurl = hand.fingerCurl.get('middle') || 1;
        const othersCurled = ['thumb', 'ring', 'pinky'].every(finger =>
            (hand.fingerCurl.get(finger) || 0) > 0.7
        );
        
        const isPeace = indexCurl < 0.3 && middleCurl < 0.3 && othersCurled;
        
        return {
            detected: isPeace,
            confidence: isPeace ? 0.85 : 0
        };
    }
}

// Hand interaction system
class HandInteractionSystem {
    constructor() {
        this.interactables = [];
        this.hoveredObjects = new Map();
        this.selectedObjects = new Map();
        this.raycaster = new THREE.Raycaster();
    }
    
    addObject(object) {
        if (!this.interactables.includes(object)) {
            this.interactables.push(object);
            object.userData.isInteractable = true;
        }
    }
    
    removeObject(object) {
        const index = this.interactables.indexOf(object);
        if (index > -1) {
            this.interactables.splice(index, 1);
            delete object.userData.isInteractable;
        }
    }
    
    update(hands) {
        hands.forEach((hand, handedness) => {
            this.updateHandInteraction(hand, handedness);
        });
    }
    
    updateHandInteraction(hand, handedness) {
        const indexTip = hand.joints.get('index-finger-tip');
        if (!indexTip) return;
        
        // Cast ray from index finger
        const rayOrigin = indexTip.position;
        const rayDirection = hand.velocity.clone().normalize();
        
        // If velocity is too low, use finger direction
        if (rayDirection.length() < 0.1) {
            const indexBase = hand.joints.get('index-finger-metacarpal');
            if (indexBase) {
                rayDirection.subVectors(indexTip.position, indexBase.position).normalize();
            }
        }
        
        this.raycaster.set(rayOrigin, rayDirection);
        
        // Check intersections
        const intersects = this.raycaster.intersectObjects(this.interactables, true);
        
        if (intersects.length > 0) {
            const closest = intersects[0].object;
            this.handleHover(closest, handedness, intersects[0]);
            
            // Check for selection (pinch gesture)
            const pinchGesture = hand.getActiveGesture?.('pinch');
            if (pinchGesture && pinchGesture.confidence > 0.8) {
                this.handleSelect(closest, handedness, intersects[0]);
            }
        } else {
            // Clear hover if nothing intersected
            this.clearHover(handedness);
        }
    }
    
    handleHover(object, handedness, intersection) {
        const wasHovered = this.hoveredObjects.has(handedness);
        const previousObject = this.hoveredObjects.get(handedness);
        
        if (previousObject !== object) {
            // Clear previous hover
            if (previousObject) {
                this.onHoverEnd(previousObject, handedness);
            }
            
            // Start new hover
            this.hoveredObjects.set(handedness, object);
            this.onHoverStart(object, handedness, intersection);
        } else if (wasHovered) {
            // Continue hover
            this.onHoverUpdate(object, handedness, intersection);
        }
    }
    
    clearHover(handedness) {
        const object = this.hoveredObjects.get(handedness);
        if (object) {
            this.onHoverEnd(object, handedness);
            this.hoveredObjects.delete(handedness);
        }
    }
    
    handleSelect(object, handedness, intersection) {
        const wasSelected = this.selectedObjects.has(handedness);
        
        if (!wasSelected) {
            this.selectedObjects.set(handedness, object);
            this.onSelectStart(object, handedness, intersection);
        } else {
            this.onSelectUpdate(object, handedness, intersection);
        }
    }
    
    onHoverStart(object, handedness, intersection) {
        // Visual feedback
        if (object.material) {
            object.userData.originalEmissive = object.material.emissive?.getHex();
            object.material.emissive?.setHex(0x444444);
        }
        
        // Haptic feedback
        webXRManager.vibrate(handedness, 0.1, 20);
        
        // Dispatch event
        object.dispatchEvent({
            type: 'hover-start',
            handedness,
            intersection
        });
    }
    
    onHoverUpdate(object, handedness, intersection) {
        object.dispatchEvent({
            type: 'hover-update',
            handedness,
            intersection
        });
    }
    
    onHoverEnd(object, handedness) {
        // Restore visual state
        if (object.material && object.userData.originalEmissive !== undefined) {
            object.material.emissive?.setHex(object.userData.originalEmissive);
        }
        
        // Dispatch event
        object.dispatchEvent({
            type: 'hover-end',
            handedness
        });
    }
    
    onSelectStart(object, handedness, intersection) {
        // Strong haptic feedback
        webXRManager.vibrate(handedness, 0.5, 100);
        
        // Dispatch event
        object.dispatchEvent({
            type: 'select-start',
            handedness,
            intersection
        });
    }
    
    onSelectUpdate(object, handedness, intersection) {
        object.dispatchEvent({
            type: 'select-update',
            handedness,
            intersection
        });
    }
}

// Export hand tracking system
export const handTracking = new RealHandTracking();