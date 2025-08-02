/**
 * Advanced Hand Tracking System - Gesture recognition and control
 * Optimized for Apple Vision Pro and Meta Quest 3 hand tracking
 */

import { webXRManager } from '../core/webxr-manager.js';

export class HandTracking {
    constructor() {
        this.handModels = new Map();
        this.handJoints = new Map();
        this.gestureRecognizer = null;
        this.handMeshes = new Map();
        this.virtualHands = new Map();
        this.interactionSphere = null;
        this.raycastHandler = null;
        this.gestureHistory = [];
        this.handPhysics = null;
        
        this.jointNames = [
            'wrist',
            'thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip',
            'index-finger-metacarpal', 'index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate', 'index-finger-phalanx-distal', 'index-finger-tip',
            'middle-finger-metacarpal', 'middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal', 'middle-finger-tip',
            'ring-finger-metacarpal', 'ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal', 'ring-finger-tip',
            'pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal', 'pinky-finger-tip'
        ];
        
        this.init();
    }

    async init() {
        console.log('👋 Initializing Advanced Hand Tracking...');
        
        // Initialize gesture recognition system
        this.initGestureRecognizer();
        
        // Setup hand visualization
        this.setupHandVisualization();
        
        // Initialize interaction system
        this.setupInteractionSystem();
        
        // Setup hand physics
        this.setupHandPhysics();
        
        // Initialize gesture learning
        this.initGestureLearning();
        
        console.log('✅ Advanced Hand Tracking initialized');
    }

    initGestureRecognizer() {
        console.log('🤖 Initializing gesture recognizer...');
        
        this.gestureRecognizer = {
            recognizedGestures: new Map(),
            gestureTemplates: new Map(),
            confidenceThreshold: 0.75,
            gestureBuffer: [],
            bufferSize: 30, // 30 frames of gesture data
            isLearning: false,
            neuralNetwork: null
        };
        
        // Define built-in gesture templates
        this.defineBasicGestures();
        
        // Initialize neural network for gesture recognition
        this.initGestureNeuralNetwork();
    }

    defineBasicGestures() {
        const gestures = {
            'point': {
                description: 'Pointing with index finger',
                pattern: {
                    thumb: 'closed',
                    index: 'extended',
                    middle: 'closed',
                    ring: 'closed',
                    pinky: 'closed'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handlePointGesture(hand, data)
            },
            'thumbs_up': {
                description: 'Thumbs up approval gesture',
                pattern: {
                    thumb: 'extended',
                    index: 'closed',
                    middle: 'closed',
                    ring: 'closed',
                    pinky: 'closed'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleThumbsUpGesture(hand, data)
            },
            'peace': {
                description: 'Peace sign with index and middle finger',
                pattern: {
                    thumb: 'closed',
                    index: 'extended',
                    middle: 'extended',
                    ring: 'closed',
                    pinky: 'closed'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handlePeaceGesture(hand, data)
            },
            'pinch': {
                description: 'Pinch gesture with thumb and index',
                pattern: {
                    thumbIndexDistance: '<0.02',
                    precision: 'high'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handlePinchGesture(hand, data)
            },
            'grab': {
                description: 'Grabbing gesture with all fingers',
                pattern: {
                    thumb: 'closed',
                    index: 'closed',
                    middle: 'closed',
                    ring: 'closed',
                    pinky: 'closed',
                    handClosure: '>0.8'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleGrabGesture(hand, data)
            },
            'open_palm': {
                description: 'Open palm gesture',
                pattern: {
                    thumb: 'extended',
                    index: 'extended',
                    middle: 'extended',
                    ring: 'extended',
                    pinky: 'extended',
                    handClosure: '<0.2'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleOpenPalmGesture(hand, data)
            },
            'swipe_left': {
                description: 'Swipe left motion',
                pattern: {
                    motionDirection: 'left',
                    velocity: '>0.5',
                    duration: '<1.0'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleSwipeLeftGesture(hand, data)
            },
            'swipe_right': {
                description: 'Swipe right motion',
                pattern: {
                    motionDirection: 'right',
                    velocity: '>0.5',
                    duration: '<1.0'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleSwipeRightGesture(hand, data)
            },
            'tap': {
                description: 'Air tap gesture',
                pattern: {
                    indexMotion: 'forward_back',
                    velocity: '>1.0',
                    duration: '<0.5'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleTapGesture(hand, data)
            },
            'rotate_cw': {
                description: 'Clockwise rotation gesture',
                pattern: {
                    handRotation: 'clockwise',
                    rotationAngle: '>45',
                    duration: '<2.0'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleRotateClockwiseGesture(hand, data)
            },
            'rotate_ccw': {
                description: 'Counter-clockwise rotation gesture',
                pattern: {
                    handRotation: 'counter_clockwise',
                    rotationAngle: '>45',
                    duration: '<2.0'
                },
                confidence: 0.0,
                callback: (hand, data) => this.handleRotateCounterClockwiseGesture(hand, data)
            }
        };
        
        Object.entries(gestures).forEach(([name, gesture]) => {
            this.gestureRecognizer.gestureTemplates.set(name, gesture);
        });
        
        console.log(`📋 Loaded ${Object.keys(gestures).length} gesture templates`);
    }

    initGestureNeuralNetwork() {
        // Simple neural network for gesture classification
        this.gestureRecognizer.neuralNetwork = {
            layers: [
                { size: 75, type: 'input' }, // 25 joints * 3 coordinates
                { size: 50, type: 'hidden', activation: 'relu' },
                { size: 25, type: 'hidden', activation: 'relu' },
                { size: 11, type: 'output', activation: 'softmax' } // Number of gestures
            ],
            weights: new Map(),
            biases: new Map(),
            trained: false
        };
        
        // Initialize random weights
        this.initializeNetworkWeights();
    }

    initializeNetworkWeights() {
        // Initialize weights and biases with random values
        const network = this.gestureRecognizer.neuralNetwork;
        
        for (let i = 0; i < network.layers.length - 1; i++) {
            const currentLayer = network.layers[i];
            const nextLayer = network.layers[i + 1];
            
            const weights = [];
            for (let j = 0; j < currentLayer.size; j++) {
                const row = [];
                for (let k = 0; k < nextLayer.size; k++) {
                    row.push((Math.random() - 0.5) * 2);
                }
                weights.push(row);
            }
            
            network.weights.set(`layer_${i}`, weights);
            
            const biases = [];
            for (let j = 0; j < nextLayer.size; j++) {
                biases.push((Math.random() - 0.5) * 2);
            }
            
            network.biases.set(`layer_${i}`, biases);
        }
    }

    setupHandVisualization() {
        console.log('🖐️ Setting up hand visualization...');
        
        // Create hand models for both hands
        ['left', 'right'].forEach(handedness => {
            const handModel = this.createHandModel(handedness);
            this.handModels.set(handedness, handModel);
            webXRManager.scene.add(handModel);
        });
        
        // Setup interaction spheres
        this.setupInteractionSpheres();
    }

    createHandModel(handedness) {
        const handGroup = new THREE.Group();
        handGroup.name = `hand-${handedness}`;
        
        // Create joint visualizations
        const jointMeshes = new Map();
        
        this.jointNames.forEach(jointName => {
            const jointGeometry = new THREE.SphereGeometry(0.008, 8, 8);
            const jointMaterial = new THREE.MeshLambertMaterial({
                color: handedness === 'left' ? 0x4a90e2 : 0xe24a90,
                transparent: true,
                opacity: 0.8
            });
            
            const jointMesh = new THREE.Mesh(jointGeometry, jointMaterial);
            jointMesh.name = jointName;
            jointMesh.visible = false; // Hidden by default
            
            handGroup.add(jointMesh);
            jointMeshes.set(jointName, jointMesh);
        });
        
        // Create hand mesh (optional detailed mesh)
        const handMesh = this.createDetailedHandMesh(handedness);
        if (handMesh) {
            handGroup.add(handMesh);
            this.handMeshes.set(handedness, handMesh);
        }
        
        // Store joint meshes
        this.handJoints.set(handedness, jointMeshes);
        
        return handGroup;
    }

    createDetailedHandMesh(handedness) {
        // Create a more detailed hand mesh (simplified version)
        const handGeometry = new THREE.BoxGeometry(0.08, 0.02, 0.15);
        const handMaterial = new THREE.MeshLambertMaterial({
            color: 0xfdbcb4,
            transparent: true,
            opacity: 0.7
        });
        
        const handMesh = new THREE.Mesh(handGeometry, handMaterial);
        handMesh.name = `hand-mesh-${handedness}`;
        
        return handMesh;
    }

    setupInteractionSpheres() {
        // Create interaction spheres for fingertips
        ['left', 'right'].forEach(handedness => {
            const fingertips = ['thumb-tip', 'index-finger-tip', 'middle-finger-tip', 'ring-finger-tip', 'pinky-finger-tip'];
            
            fingertips.forEach(fingertip => {
                const sphereGeometry = new THREE.SphereGeometry(0.015, 16, 16);
                const sphereMaterial = new THREE.MeshLambertMaterial({
                    color: 0x00ff00,
                    transparent: true,
                    opacity: 0.3,
                    visible: false
                });
                
                const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
                sphere.name = `interaction-${handedness}-${fingertip}`;
                
                const handModel = this.handModels.get(handedness);
                handModel.add(sphere);
            });
        });
    }

    setupInteractionSystem() {
        console.log('🎯 Setting up interaction system...');
        
        this.raycastHandler = {
            raycaster: new THREE.Raycaster(),
            intersectableObjects: [],
            activeInteractions: new Map(),
            interactionDistance: 0.1
        };
        
        // Initialize interaction callbacks
        this.initInteractionCallbacks();
    }

    initInteractionCallbacks() {
        this.interactionCallbacks = {
            onHover: (object, hand) => this.handleObjectHover(object, hand),
            onSelect: (object, hand) => this.handleObjectSelect(object, hand),
            onGrab: (object, hand) => this.handleObjectGrab(object, hand),
            onRelease: (object, hand) => this.handleObjectRelease(object, hand)
        };
    }

    setupHandPhysics() {
        console.log('⚛️ Setting up hand physics...');
        
        this.handPhysics = {
            enabled: true,
            collisionBodies: new Map(),
            handColliders: new Map(),
            physicsWorld: null // Would integrate with physics engine
        };
        
        // Create collision bodies for hands
        ['left', 'right'].forEach(handedness => {
            const collisionBody = this.createHandCollisionBody(handedness);
            this.handPhysics.handColliders.set(handedness, collisionBody);
        });
    }

    createHandCollisionBody(handedness) {
        // Simplified collision body for hand
        return {
            position: new THREE.Vector3(),
            rotation: new THREE.Quaternion(),
            velocity: new THREE.Vector3(),
            boundingBox: new THREE.Box3()
        };
    }

    initGestureLearning() {
        console.log('🧠 Initializing gesture learning system...');
        
        this.gestureLearning = {
            isRecording: false,
            recordedSamples: new Map(),
            learningMode: false,
            customGestures: new Map(),
            adaptiveThreshold: true
        };
    }

    updateHandTracking(frame) {
        if (!webXRManager.xrSession || !webXRManager.isHandTrackingSupported) return;
        
        // Update hand tracking data
        webXRManager.updateHandTracking(frame);
        
        // Update hand visualizations
        this.updateHandVisualization();
        
        // Perform gesture recognition
        this.performGestureRecognition();
        
        // Process interactions
        this.processHandInteractions();
        
        // Update hand physics
        this.updateHandPhysics();
    }

    updateHandVisualization() {
        ['left', 'right'].forEach(handedness => {
            const handModel = this.handModels.get(handedness);
            const jointMeshes = this.handJoints.get(handedness);
            
            if (!handModel || !jointMeshes) return;
            
            let handVisible = false;
            
            this.jointNames.forEach(jointName => {
                const jointData = webXRManager.getHandJoint(handedness, jointName);
                const jointMesh = jointMeshes.get(jointName);
                
                if (jointData && jointMesh) {
                    // Update joint position
                    jointMesh.position.copy(jointData.position);
                    jointMesh.quaternion.copy(jointData.orientation);
                    jointMesh.visible = true;
                    handVisible = true;
                } else if (jointMesh) {
                    jointMesh.visible = false;
                }
            });
            
            // Update hand mesh if available
            const handMesh = this.handMeshes.get(handedness);
            if (handMesh) {
                const wristJoint = webXRManager.getHandJoint(handedness, 'wrist');
                if (wristJoint) {
                    handMesh.position.copy(wristJoint.position);
                    handMesh.quaternion.copy(wristJoint.orientation);
                    handMesh.visible = handVisible;
                }
            }
            
            handModel.visible = handVisible;
        });
    }

    performGestureRecognition() {
        ['left', 'right'].forEach(handedness => {
            const gestureData = this.extractGestureFeatures(handedness);
            if (gestureData) {
                const recognizedGesture = this.recognizeGesture(gestureData, handedness);
                if (recognizedGesture) {
                    this.handleRecognizedGesture(recognizedGesture, handedness, gestureData);
                }
            }
        });
    }

    extractGestureFeatures(handedness) {
        const features = {
            handedness: handedness,
            timestamp: Date.now(),
            joints: new Map(),
            fingerStates: new Map(),
            handClosure: 0,
            handOrientation: new THREE.Quaternion(),
            velocity: new THREE.Vector3(),
            confidence: 0
        };
        
        // Extract joint positions and orientations
        let validJoints = 0;
        this.jointNames.forEach(jointName => {
            const jointData = webXRManager.getHandJoint(handedness, jointName);
            if (jointData) {
                features.joints.set(jointName, {
                    position: jointData.position.clone(),
                    orientation: jointData.orientation.clone(),
                    radius: jointData.radius || 0.01
                });
                validJoints++;
            }
        });
        
        if (validJoints < 5) return null; // Not enough data
        
        // Calculate finger states
        this.calculateFingerStates(features);
        
        // Calculate hand closure
        features.handClosure = this.calculateHandClosure(features);
        
        // Calculate hand orientation
        const wristJoint = features.joints.get('wrist');
        if (wristJoint) {
            features.handOrientation.copy(wristJoint.orientation);
        }
        
        // Calculate velocity (if previous frame data available)
        features.velocity = this.calculateHandVelocity(handedness, features);
        
        return features;
    }

    calculateFingerStates(features) {
        const fingers = {
            'thumb': ['thumb-metacarpal', 'thumb-phalanx-proximal', 'thumb-phalanx-distal', 'thumb-tip'],
            'index': ['index-finger-metacarpal', 'index-finger-phalanx-proximal', 'index-finger-phalanx-intermediate', 'index-finger-phalanx-distal', 'index-finger-tip'],
            'middle': ['middle-finger-metacarpal', 'middle-finger-phalanx-proximal', 'middle-finger-phalanx-intermediate', 'middle-finger-phalanx-distal', 'middle-finger-tip'],
            'ring': ['ring-finger-metacarpal', 'ring-finger-phalanx-proximal', 'ring-finger-phalanx-intermediate', 'ring-finger-phalanx-distal', 'ring-finger-tip'],
            'pinky': ['pinky-finger-metacarpal', 'pinky-finger-phalanx-proximal', 'pinky-finger-phalanx-intermediate', 'pinky-finger-phalanx-distal', 'pinky-finger-tip']
        };
        
        Object.entries(fingers).forEach(([fingerName, joints]) => {
            const fingerState = this.calculateFingerExtension(features, joints);
            features.fingerStates.set(fingerName, fingerState);
        });
    }

    calculateFingerExtension(features, fingerJoints) {
        // Calculate if finger is extended or closed
        // Simplified calculation based on joint angles
        
        let totalCurvature = 0;
        let validJoints = 0;
        
        for (let i = 1; i < fingerJoints.length - 1; i++) {
            const joint1 = features.joints.get(fingerJoints[i]);
            const joint2 = features.joints.get(fingerJoints[i + 1]);
            
            if (joint1 && joint2) {
                // Calculate angle between joints (simplified)
                const distance = joint1.position.distanceTo(joint2.position);
                totalCurvature += distance;
                validJoints++;
            }
        }
        
        if (validJoints === 0) return 'unknown';
        
        const averageCurvature = totalCurvature / validJoints;
        return averageCurvature > 0.03 ? 'extended' : 'closed';
    }

    calculateHandClosure(features) {
        let closedFingers = 0;
        const totalFingers = 5;
        
        features.fingerStates.forEach((state, finger) => {
            if (state === 'closed') closedFingers++;
        });
        
        return closedFingers / totalFingers;
    }

    calculateHandVelocity(handedness, currentFeatures) {
        const previousFeatures = this.gestureHistory.find(
            h => h.handedness === handedness && h.timestamp > Date.now() - 100
        );
        
        if (!previousFeatures) return new THREE.Vector3();
        
        const wristCurrent = currentFeatures.joints.get('wrist');
        const wristPrevious = previousFeatures.joints.get('wrist');
        
        if (wristCurrent && wristPrevious) {
            const timeDelta = (currentFeatures.timestamp - previousFeatures.timestamp) / 1000;
            if (timeDelta > 0) {
                return wristCurrent.position.clone().sub(wristPrevious.position).divideScalar(timeDelta);
            }
        }
        
        return new THREE.Vector3();
    }

    recognizeGesture(gestureData, handedness) {
        let bestMatch = null;
        let bestConfidence = 0;
        
        // Check against all gesture templates
        this.gestureRecognizer.gestureTemplates.forEach((template, gestureName) => {
            const confidence = this.calculateGestureConfidence(gestureData, template);
            
            if (confidence > bestConfidence && confidence > this.gestureRecognizer.confidenceThreshold) {
                bestMatch = {
                    name: gestureName,
                    template: template,
                    confidence: confidence,
                    handedness: handedness
                };
                bestConfidence = confidence;
            }
        });
        
        // Use neural network for additional recognition
        if (this.gestureRecognizer.neuralNetwork.trained) {
            const nnResult = this.recognizeWithNeuralNetwork(gestureData);
            if (nnResult.confidence > bestConfidence) {
                bestMatch = nnResult;
            }
        }
        
        return bestMatch;
    }

    calculateGestureConfidence(gestureData, template) {
        let confidence = 0;
        let totalChecks = 0;
        
        // Check finger states
        if (template.pattern.thumb) {
            const thumbState = gestureData.fingerStates.get('thumb');
            if (thumbState === template.pattern.thumb) confidence += 0.2;
            totalChecks += 0.2;
        }
        
        if (template.pattern.index) {
            const indexState = gestureData.fingerStates.get('index');
            if (indexState === template.pattern.index) confidence += 0.2;
            totalChecks += 0.2;
        }
        
        if (template.pattern.middle) {
            const middleState = gestureData.fingerStates.get('middle');
            if (middleState === template.pattern.middle) confidence += 0.2;
            totalChecks += 0.2;
        }
        
        if (template.pattern.ring) {
            const ringState = gestureData.fingerStates.get('ring');
            if (ringState === template.pattern.ring) confidence += 0.2;
            totalChecks += 0.2;
        }
        
        if (template.pattern.pinky) {
            const pinkyState = gestureData.fingerStates.get('pinky');
            if (pinkyState === template.pattern.pinky) confidence += 0.2;
            totalChecks += 0.2;
        }
        
        // Check hand closure
        if (template.pattern.handClosure) {
            const targetClosure = parseFloat(template.pattern.handClosure.replace(/[<>]/, ''));
            const operator = template.pattern.handClosure.charAt(0);
            
            let closureMatch = false;
            if (operator === '<' && gestureData.handClosure < targetClosure) closureMatch = true;
            if (operator === '>' && gestureData.handClosure > targetClosure) closureMatch = true;
            if (!operator && Math.abs(gestureData.handClosure - targetClosure) < 0.1) closureMatch = true;
            
            if (closureMatch) confidence += 0.3;
            totalChecks += 0.3;
        }
        
        // Check special patterns (pinch, distances, etc.)
        if (template.pattern.thumbIndexDistance) {
            const thumbTip = gestureData.joints.get('thumb-tip');
            const indexTip = gestureData.joints.get('index-finger-tip');
            
            if (thumbTip && indexTip) {
                const distance = thumbTip.position.distanceTo(indexTip.position);
                const targetDistance = parseFloat(template.pattern.thumbIndexDistance.replace('<', ''));
                
                if (distance < targetDistance) {
                    confidence += 0.4;
                }
            }
            totalChecks += 0.4;
        }
        
        return totalChecks > 0 ? confidence / totalChecks : 0;
    }

    recognizeWithNeuralNetwork(gestureData) {
        // Convert gesture data to neural network input
        const input = this.convertGestureToNNInput(gestureData);
        
        // Forward pass through network
        const output = this.forwardPassNN(input);
        
        // Find highest confidence gesture
        let maxIndex = 0;
        let maxValue = output[0];
        
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
        
        const gestureNames = Array.from(this.gestureRecognizer.gestureTemplates.keys());
        
        return {
            name: gestureNames[maxIndex] || 'unknown',
            confidence: maxValue,
            source: 'neural_network'
        };
    }

    convertGestureToNNInput(gestureData) {
        const input = [];
        
        // Convert joint positions to input vector
        this.jointNames.forEach(jointName => {
            const joint = gestureData.joints.get(jointName);
            if (joint) {
                input.push(joint.position.x, joint.position.y, joint.position.z);
            } else {
                input.push(0, 0, 0);
            }
        });
        
        return input;
    }

    forwardPassNN(input) {
        const network = this.gestureRecognizer.neuralNetwork;
        let currentInput = input;
        
        for (let i = 0; i < network.layers.length - 1; i++) {
            const weights = network.weights.get(`layer_${i}`);
            const biases = network.biases.get(`layer_${i}`);
            const nextLayer = network.layers[i + 1];
            
            const output = [];
            
            for (let j = 0; j < nextLayer.size; j++) {
                let sum = biases[j];
                
                for (let k = 0; k < currentInput.length; k++) {
                    sum += currentInput[k] * weights[k][j];
                }
                
                // Apply activation function
                if (nextLayer.activation === 'relu') {
                    output.push(Math.max(0, sum));
                } else if (nextLayer.activation === 'softmax') {
                    output.push(Math.exp(sum));
                } else {
                    output.push(sum);
                }
            }
            
            // Normalize softmax
            if (nextLayer.activation === 'softmax') {
                const sumExp = output.reduce((a, b) => a + b, 0);
                for (let j = 0; j < output.length; j++) {
                    output[j] /= sumExp;
                }
            }
            
            currentInput = output;
        }
        
        return currentInput;
    }

    handleRecognizedGesture(recognizedGesture, handedness, gestureData) {
        console.log(`🎯 Recognized gesture: ${recognizedGesture.name} (${handedness}) - Confidence: ${recognizedGesture.confidence.toFixed(2)}`);
        
        // Store in history
        this.gestureHistory.push({
            ...gestureData,
            recognizedGesture: recognizedGesture
        });
        
        // Keep history limited
        if (this.gestureHistory.length > 100) {
            this.gestureHistory.shift();
        }
        
        // Execute gesture callback
        if (recognizedGesture.template && recognizedGesture.template.callback) {
            recognizedGesture.template.callback(handedness, gestureData);
        }
        
        // Trigger haptic feedback
        webXRManager.pulseHaptic(0.2, 50);
        
        // Update UI indicators
        this.updateGestureUI(recognizedGesture, handedness);
    }

    processHandInteractions() {
        const interactableObjects = this.raycastHandler.intersectableObjects;
        if (interactableObjects.length === 0) return;
        
        ['left', 'right'].forEach(handedness => {
            const indexTip = webXRManager.getHandJoint(handedness, 'index-finger-tip');
            if (!indexTip) return;
            
            // Check for object intersections
            this.raycastHandler.raycaster.set(
                indexTip.position,
                new THREE.Vector3(0, 0, -1).applyQuaternion(indexTip.orientation)
            );
            
            const intersections = this.raycastHandler.raycaster.intersectObjects(interactableObjects, true);
            
            if (intersections.length > 0) {
                const closestObject = intersections[0].object;
                
                // Check if within interaction distance
                if (intersections[0].distance <= this.raycastHandler.interactionDistance) {
                    this.handleObjectInteraction(closestObject, handedness, indexTip);
                }
            }
        });
    }

    handleObjectInteraction(object, handedness, jointData) {
        const interactionKey = `${object.uuid}-${handedness}`;
        
        if (!this.raycastHandler.activeInteractions.has(interactionKey)) {
            // New interaction
            this.raycastHandler.activeInteractions.set(interactionKey, {
                object: object,
                handedness: handedness,
                startTime: Date.now(),
                type: 'hover'
            });
            
            // Trigger hover callback
            if (this.interactionCallbacks.onHover) {
                this.interactionCallbacks.onHover(object, handedness);
            }
        }
    }

    updateHandPhysics() {
        if (!this.handPhysics.enabled) return;
        
        ['left', 'right'].forEach(handedness => {
            const collider = this.handPhysics.handColliders.get(handedness);
            const wristJoint = webXRManager.getHandJoint(handedness, 'wrist');
            
            if (collider && wristJoint) {
                // Update collision body position
                const previousPosition = collider.position.clone();
                collider.position.copy(wristJoint.position);
                
                // Calculate velocity
                collider.velocity.copy(collider.position).sub(previousPosition);
                
                // Update bounding box
                collider.boundingBox.setFromCenterAndSize(
                    collider.position,
                    new THREE.Vector3(0.1, 0.1, 0.2)
                );
            }
        });
    }

    // Gesture Handler Methods
    handlePointGesture(handedness, gestureData) {
        console.log(`👉 Point gesture (${handedness})`);
        
        const indexTip = gestureData.joints.get('index-finger-tip');
        if (indexTip) {
            // Create pointing ray visualization
            this.createPointingRay(indexTip.position, indexTip.orientation, handedness);
        }
        
        // Custom pointing logic here
        this.onPointingGesture?.(handedness, gestureData);
    }

    handleThumbsUpGesture(handedness, gestureData) {
        console.log(`👍 Thumbs up gesture (${handedness})`);
        
        // Custom thumbs up logic here
        this.onThumbsUpGesture?.(handedness, gestureData);
    }

    handlePeaceGesture(handedness, gestureData) {
        console.log(`✌️ Peace gesture (${handedness})`);
        
        // Custom peace gesture logic here
        this.onPeaceGesture?.(handedness, gestureData);
    }

    handlePinchGesture(handedness, gestureData) {
        console.log(`🤏 Pinch gesture (${handedness})`);
        
        const thumbTip = gestureData.joints.get('thumb-tip');
        const indexTip = gestureData.joints.get('index-finger-tip');
        
        if (thumbTip && indexTip) {
            const pinchPosition = thumbTip.position.clone().add(indexTip.position).multiplyScalar(0.5);
            
            // Create pinch visualization
            this.createPinchVisualization(pinchPosition, handedness);
        }
        
        // Custom pinch logic here
        this.onPinchGesture?.(handedness, gestureData);
    }

    handleGrabGesture(handedness, gestureData) {
        console.log(`✊ Grab gesture (${handedness})`);
        
        // Custom grab logic here
        this.onGrabGesture?.(handedness, gestureData);
    }

    handleOpenPalmGesture(handedness, gestureData) {
        console.log(`🖐️ Open palm gesture (${handedness})`);
        
        // Custom open palm logic here
        this.onOpenPalmGesture?.(handedness, gestureData);
    }

    handleSwipeLeftGesture(handedness, gestureData) {
        console.log(`👈 Swipe left gesture (${handedness})`);
        
        // Custom swipe left logic here
        this.onSwipeLeftGesture?.(handedness, gestureData);
    }

    handleSwipeRightGesture(handedness, gestureData) {
        console.log(`👉 Swipe right gesture (${handedness})`);
        
        // Custom swipe right logic here
        this.onSwipeRightGesture?.(handedness, gestureData);
    }

    handleTapGesture(handedness, gestureData) {
        console.log(`👆 Tap gesture (${handedness})`);
        
        // Custom tap logic here
        this.onTapGesture?.(handedness, gestureData);
    }

    handleRotateClockwiseGesture(handedness, gestureData) {
        console.log(`↻ Rotate clockwise gesture (${handedness})`);
        
        // Custom rotate clockwise logic here
        this.onRotateClockwiseGesture?.(handedness, gestureData);
    }

    handleRotateCounterClockwiseGesture(handedness, gestureData) {
        console.log(`↺ Rotate counter-clockwise gesture (${handedness})`);
        
        // Custom rotate counter-clockwise logic here
        this.onRotateCounterClockwiseGesture?.(handedness, gestureData);
    }

    // Object Interaction Handlers
    handleObjectHover(object, handedness) {
        console.log(`🎯 Hovering over object with ${handedness} hand`);
        
        // Highlight object
        if (object.material) {
            object.material.emissive.setHex(0x444444);
        }
    }

    handleObjectSelect(object, handedness) {
        console.log(`👆 Selected object with ${handedness} hand`);
        
        // Custom selection logic here
        webXRManager.pulseHaptic(0.5, 100);
    }

    handleObjectGrab(object, handedness) {
        console.log(`✊ Grabbed object with ${handedness} hand`);
        
        // Custom grab logic here
        webXRManager.pulseHaptic(0.8, 200);
    }

    handleObjectRelease(object, handedness) {
        console.log(`🖐️ Released object with ${handedness} hand`);
        
        // Custom release logic here
        webXRManager.pulseHaptic(0.3, 50);
    }

    // Visualization Methods
    createPointingRay(position, orientation, handedness) {
        const rayGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 0, -2)
        ]);
        
        const rayMaterial = new THREE.LineBasicMaterial({
            color: handedness === 'left' ? 0x4a90e2 : 0xe24a90,
            transparent: true,
            opacity: 0.6
        });
        
        const ray = new THREE.Line(rayGeometry, rayMaterial);
        ray.position.copy(position);
        ray.quaternion.copy(orientation);
        
        webXRManager.scene.add(ray);
        
        // Remove ray after short duration
        setTimeout(() => {
            webXRManager.scene.remove(ray);
        }, 500);
    }

    createPinchVisualization(position, handedness) {
        const pinchGeometry = new THREE.SphereGeometry(0.02, 16, 16);
        const pinchMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.8
        });
        
        const pinchSphere = new THREE.Mesh(pinchGeometry, pinchMaterial);
        pinchSphere.position.copy(position);
        
        webXRManager.scene.add(pinchSphere);
        
        // Animate and remove
        const startScale = pinchSphere.scale.clone();
        const animate = () => {
            pinchSphere.scale.multiplyScalar(1.05);
            pinchMaterial.opacity *= 0.95;
            
            if (pinchMaterial.opacity > 0.1) {
                requestAnimationFrame(animate);
            } else {
                webXRManager.scene.remove(pinchSphere);
            }
        };
        animate();
    }

    updateGestureUI(recognizedGesture, handedness) {
        // Update gesture indicator in UI
        const indicator = document.querySelector('.gesture-indicator');
        if (indicator) {
            indicator.textContent = `${recognizedGesture.name} (${handedness})`;
            indicator.style.opacity = '1';
            
            setTimeout(() => {
                indicator.style.opacity = '0';
            }, 2000);
        }
    }

    // Public API Methods
    addInteractableObject(object) {
        this.raycastHandler.intersectableObjects.push(object);
    }

    removeInteractableObject(object) {
        const index = this.raycastHandler.intersectableObjects.indexOf(object);
        if (index > -1) {
            this.raycastHandler.intersectableObjects.splice(index, 1);
        }
    }

    setHandVisibility(visible) {
        this.handModels.forEach(handModel => {
            handModel.visible = visible;
        });
    }

    setJointVisibility(visible) {
        this.handJoints.forEach(jointMeshes => {
            jointMeshes.forEach(jointMesh => {
                jointMesh.visible = visible;
            });
        });
    }

    startGestureRecording(gestureName) {
        this.gestureLearning.isRecording = true;
        this.gestureLearning.currentGestureName = gestureName;
        this.gestureLearning.recordedSamples.set(gestureName, []);
        
        console.log(`📹 Started recording gesture: ${gestureName}`);
    }

    stopGestureRecording() {
        if (this.gestureLearning.isRecording) {
            this.gestureLearning.isRecording = false;
            console.log(`⏹️ Stopped recording gesture: ${this.gestureLearning.currentGestureName}`);
            
            // Process recorded samples
            this.processRecordedGesture();
        }
    }

    processRecordedGesture() {
        const gestureName = this.gestureLearning.currentGestureName;
        const samples = this.gestureLearning.recordedSamples.get(gestureName);
        
        if (samples && samples.length > 10) {
            // Create new gesture template from samples
            const gestureTemplate = this.createGestureTemplate(samples);
            this.gestureRecognizer.gestureTemplates.set(gestureName, gestureTemplate);
            
            console.log(`✅ Created new gesture template: ${gestureName}`);
        }
    }

    createGestureTemplate(samples) {
        // Analyze samples to create gesture template
        // This is a simplified version
        return {
            description: 'Custom learned gesture',
            pattern: this.analyzeGestureSamples(samples),
            confidence: 0.0,
            callback: (hand, data) => console.log(`Custom gesture recognized: ${hand}`)
        };
    }

    analyzeGestureSamples(samples) {
        // Analyze gesture samples to extract pattern
        // Simplified analysis
        const pattern = {};
        
        // Analyze finger states across samples
        const fingerStates = { thumb: [], index: [], middle: [], ring: [], pinky: [] };
        
        samples.forEach(sample => {
            sample.fingerStates.forEach((state, finger) => {
                if (fingerStates[finger]) {
                    fingerStates[finger].push(state);
                }
            });
        });
        
        // Determine most common finger states
        Object.entries(fingerStates).forEach(([finger, states]) => {
            const counts = {};
            states.forEach(state => {
                counts[state] = (counts[state] || 0) + 1;
            });
            
            const mostCommon = Object.keys(counts).reduce((a, b) => 
                counts[a] > counts[b] ? a : b
            );
            
            pattern[finger] = mostCommon;
        });
        
        return pattern;
    }

    getGestureHistory() {
        return this.gestureHistory.slice(-10); // Return last 10 gestures
    }

    getCurrentHandPoses() {
        const poses = {};
        
        ['left', 'right'].forEach(handedness => {
            const pose = {};
            this.jointNames.forEach(jointName => {
                const jointData = webXRManager.getHandJoint(handedness, jointName);
                if (jointData) {
                    pose[jointName] = {
                        position: jointData.position.toArray(),
                        orientation: jointData.orientation.toArray()
                    };
                }
            });
            poses[handedness] = pose;
        });
        
        return poses;
    }
}

// Export Hand Tracking class
export default HandTracking;