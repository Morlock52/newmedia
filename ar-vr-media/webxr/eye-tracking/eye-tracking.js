/**
 * Advanced Eye Tracking System - Gaze-based navigation and interaction
 * Optimized for Apple Vision Pro's advanced eye tracking capabilities
 */

import { webXRManager } from '../core/webxr-manager.js';

export class EyeTracking {
    constructor() {
        this.isSupported = false;
        this.isActive = false;
        this.gazeData = {
            position: new THREE.Vector3(),
            direction: new THREE.Vector3(),
            confidence: 0,
            timestamp: 0
        };
        
        this.gazeHistory = [];
        this.gazeTarget = null;
        this.gazeIndicator = null;
        this.fixationDetector = null;
        this.saccadeDetector = null;
        this.gazeCalibration = null;
        this.privacyMode = true;
        
        // Gaze interaction system
        this.gazeInteraction = {
            dwellTime: 800, // ms for dwell selection
            currentTarget: null,
            dwellStartTime: 0,
            isSelecting: false,
            selectionProgress: 0
        };
        
        // Foveated rendering
        this.foveatedRendering = {
            enabled: false,
            foveaRadius: 15, // degrees
            peripheralReduction: 0.5,
            renderTargets: new Map()
        };
        
        // Attention tracking
        this.attentionTracking = {
            enabled: false,
            focusAreas: new Map(),
            heatmapData: new Map(),
            attentionMetrics: {
                totalGazeTime: 0,
                averageFixationDuration: 0,
                saccadeCount: 0,
                blinkCount: 0
            }
        };
        
        this.init();
    }

    async init() {
        console.log('👁️ Initializing Advanced Eye Tracking...');
        
        // Check eye tracking support
        await this.checkEyeTrackingSupport();
        
        if (this.isSupported) {
            // Initialize eye tracking components
            this.initGazeVisualization();
            this.initFixationDetection();
            this.initSaccadeDetection();
            this.initGazeCalibration();
            this.initFoveatedRendering();
            this.initAttentionTracking();
            this.initPrivacyControls();
            
            console.log('✅ Advanced Eye Tracking initialized');
        } else {
            console.warn('⚠️ Eye tracking not supported on this device');
            this.initMockEyeTracking(); // For development/testing
        }
    }

    async checkEyeTrackingSupport() {
        // Check for WebXR eye tracking support
        if (!navigator.xr) {
            this.isSupported = false;
            return;
        }
        
        try {
            // Check if eye tracking is supported as an optional feature
            const session = await navigator.xr.requestSession('immersive-vr', {
                optionalFeatures: ['eye-tracking']
            }).catch(() => null);
            
            if (session) {
                this.isSupported = session.inputSources.some(
                    source => source.eyes && source.eyes.length > 0
                );
                session.end();
            } else {
                // Fallback: check user agent for known eye tracking devices
                const userAgent = navigator.userAgent.toLowerCase();
                this.isSupported = userAgent.includes('vision') || 
                                 userAgent.includes('visionos') ||
                                 userAgent.includes('eye-tracking');
            }
        } catch (error) {
            console.warn('Eye tracking support check failed:', error);
            this.isSupported = false;
        }
        
        console.log(`Eye tracking supported: ${this.isSupported}`);
    }

    initMockEyeTracking() {
        // Mock eye tracking for development when real hardware isn't available
        console.log('🎭 Initializing mock eye tracking for development...');
        
        this.isSupported = true;
        this.mockMode = true;
        
        // Simulate eye movement
        this.startMockEyeMovement();
    }

    startMockEyeMovement() {
        let angle = 0;
        const mockUpdate = () => {
            if (!this.isActive) {
                requestAnimationFrame(mockUpdate);
                return;
            }
            
            // Generate mock gaze data
            angle += 0.02;
            
            this.gazeData.position.set(
                Math.sin(angle) * 2,
                Math.cos(angle * 0.7) * 1.5,
                -3 + Math.sin(angle * 0.3) * 1
            );
            
            this.gazeData.direction.set(
                Math.sin(angle + 0.1) * 0.1,
                Math.cos(angle * 0.8 + 0.1) * 0.1,
                -1
            ).normalize();
            
            this.gazeData.confidence = 0.8 + Math.random() * 0.2;
            this.gazeData.timestamp = Date.now();
            
            requestAnimationFrame(mockUpdate);
        };
        
        mockUpdate();
    }

    initGazeVisualization() {
        console.log('👀 Initializing gaze visualization...');
        
        // Create gaze indicator (cursor)
        const indicatorGeometry = new THREE.RingGeometry(0.01, 0.02, 16);
        const indicatorMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        
        this.gazeIndicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
        this.gazeIndicator.name = 'gaze-indicator';
        this.gazeIndicator.visible = false;
        
        webXRManager.scene.add(this.gazeIndicator);
        
        // Create gaze ray visualization
        this.gazeRay = this.createGazeRay();
        webXRManager.scene.add(this.gazeRay);
    }

    createGazeRay() {
        const rayGeometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 0, -5)
        ]);
        
        const rayMaterial = new THREE.LineBasicMaterial({
            color: 0x00aaff,
            transparent: true,
            opacity: 0.3,
            linewidth: 2
        });
        
        const ray = new THREE.Line(rayGeometry, rayMaterial);
        ray.name = 'gaze-ray';
        ray.visible = false;
        
        return ray;
    }

    initFixationDetection() {
        console.log('🎯 Initializing fixation detection...');
        
        this.fixationDetector = {
            isFixating: false,
            fixationStartTime: 0,
            fixationDuration: 0,
            fixationPosition: new THREE.Vector3(),
            fixationThreshold: 2.0, // degrees
            minimumFixationTime: 100, // ms
            currentFixation: null,
            fixationHistory: []
        };
    }

    initSaccadeDetection() {
        console.log('⚡ Initializing saccade detection...');
        
        this.saccadeDetector = {
            isSaccading: false,
            saccadeStartTime: 0,
            saccadeVelocity: new THREE.Vector3(),
            saccadeThreshold: 30, // degrees/second
            saccadeHistory: [],
            peakVelocity: 0
        };
    }

    initGazeCalibration() {
        console.log('🎯 Initializing gaze calibration...');
        
        this.gazeCalibration = {
            isCalibrated: false,
            calibrationPoints: [
                { x: -0.8, y: 0.6, z: -2 },   // Top left
                { x: 0.8, y: 0.6, z: -2 },    // Top right
                { x: -0.8, y: -0.6, z: -2 },  // Bottom left
                { x: 0.8, y: -0.6, z: -2 },   // Bottom right
                { x: 0, y: 0, z: -2 }          // Center
            ],
            currentCalibrationPoint: 0,
            calibrationData: [],
            offsetCorrection: new THREE.Vector3(),
            accuracyRadius: 0.1
        };
    }

    initFoveatedRendering() {
        console.log('🔍 Initializing foveated rendering...');
        
        this.foveatedRendering = {
            enabled: false,
            foveaRadius: 15, // degrees
            peripheralReduction: 0.5,
            renderTargets: new Map(),
            qualityLevels: [
                { distance: 0, quality: 1.0 },     // Fovea - full quality
                { distance: 5, quality: 0.8 },     // Near periphery
                { distance: 15, quality: 0.5 },    // Mid periphery
                { distance: 30, quality: 0.25 }    // Far periphery
            ]
        };
        
        // Create render targets for different quality levels
        this.createFoveatedRenderTargets();
    }

    createFoveatedRenderTargets() {
        const sizes = [
            { name: 'high', width: 2048, height: 2048 },
            { name: 'medium', width: 1024, height: 1024 },
            { name: 'low', width: 512, height: 512 }
        ];
        
        sizes.forEach(size => {
            const renderTarget = new THREE.WebGLRenderTarget(size.width, size.height, {
                format: THREE.RGBAFormat,
                type: THREE.FloatType,
                magFilter: THREE.LinearFilter,
                minFilter: THREE.LinearFilter
            });
            
            this.foveatedRendering.renderTargets.set(size.name, renderTarget);
        });
    }

    initAttentionTracking() {
        console.log('🧠 Initializing attention tracking...');
        
        this.attentionTracking = {
            enabled: false,
            focusAreas: new Map(),
            heatmapData: new Map(),
            attentionMetrics: {
                totalGazeTime: 0,
                averageFixationDuration: 0,
                saccadeCount: 0,
                blinkCount: 0,
                cognitiveLoad: 0,
                engagementLevel: 0
            },
            heatmapVisualization: null
        };
        
        // Create heatmap visualization
        this.createAttentionHeatmap();
    }

    createAttentionHeatmap() {
        const heatmapGeometry = new THREE.PlaneGeometry(10, 6);
        const heatmapMaterial = new THREE.ShaderMaterial({
            uniforms: {
                heatmapTexture: { value: null },
                opacity: { value: 0.5 },
                time: { value: 0 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D heatmapTexture;
                uniform float opacity;
                uniform float time;
                varying vec2 vUv;
                
                void main() {
                    vec4 heatmap = texture2D(heatmapTexture, vUv);
                    
                    // Create heat visualization
                    vec3 coldColor = vec3(0.0, 0.0, 1.0);  // Blue
                    vec3 warmColor = vec3(1.0, 1.0, 0.0);  // Yellow
                    vec3 hotColor = vec3(1.0, 0.0, 0.0);   // Red
                    
                    vec3 color;
                    if (heatmap.r < 0.5) {
                        color = mix(coldColor, warmColor, heatmap.r * 2.0);
                    } else {
                        color = mix(warmColor, hotColor, (heatmap.r - 0.5) * 2.0);
                    }
                    
                    gl_FragColor = vec4(color, heatmap.r * opacity);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending
        });
        
        this.attentionTracking.heatmapVisualization = new THREE.Mesh(heatmapGeometry, heatmapMaterial);
        this.attentionTracking.heatmapVisualization.position.set(0, 0, -5);
        this.attentionTracking.heatmapVisualization.visible = false;
        
        webXRManager.scene.add(this.attentionTracking.heatmapVisualization);
    }

    initPrivacyControls() {
        console.log('🔒 Initializing privacy controls...');
        
        this.privacySettings = {
            recordGazeData: false,
            shareGazeData: false,
            anonymizeData: true,
            dataRetentionDays: 7,
            consentGiven: false
        };
    }

    async startEyeTracking() {
        if (!this.isSupported) {
            throw new Error('Eye tracking not supported');
        }
        
        try {
            // Request eye tracking permission
            if (!this.privacySettings.consentGiven) {
                const consent = await this.requestEyeTrackingConsent();
                if (!consent) {
                    throw new Error('Eye tracking permission denied');
                }
            }
            
            this.isActive = true;
            
            // Start tracking loops
            this.startGazeTracking();
            this.startFixationDetection();
            this.startSaccadeDetection();
            this.startAttentionTracking();
            
            // Show gaze indicator
            this.gazeIndicator.visible = true;
            
            console.log('👁️ Eye tracking started');
            return true;
            
        } catch (error) {
            console.error('Failed to start eye tracking:', error);
            throw error;
        }
    }

    async requestEyeTrackingConsent() {
        return new Promise((resolve) => {
            // In a real implementation, show a proper consent dialog
            const consent = confirm(
                'This application would like to use eye tracking for enhanced interaction. ' +
                'Your gaze data will be processed locally and not shared. Do you consent?'
            );
            
            this.privacySettings.consentGiven = consent;
            resolve(consent);
        });
    }

    startGazeTracking() {
        const trackGaze = () => {
            if (!this.isActive) return;
            
            if (!this.mockMode) {
                this.updateRealGazeData();
            }
            
            // Update gaze visualization
            this.updateGazeVisualization();
            
            // Process gaze interactions
            this.processGazeInteractions();
            
            // Update attention tracking
            if (this.attentionTracking.enabled) {
                this.updateAttentionData();
            }
            
            // Store gaze history
            this.gazeHistory.push({
                ...this.gazeData,
                timestamp: Date.now()
            });
            
            // Keep history limited
            if (this.gazeHistory.length > 1000) {
                this.gazeHistory.shift();
            }
            
            requestAnimationFrame(trackGaze);
        };
        
        trackGaze();
    }

    updateRealGazeData() {
        // Get real eye tracking data from WebXR
        if (webXRManager.xrSession && webXRManager.xrSession.inputSources) {
            for (const inputSource of webXRManager.xrSession.inputSources) {
                if (inputSource.eyes) {
                    const eyeData = inputSource.eyes[0]; // Combined gaze
                    if (eyeData) {
                        // Update gaze data from real hardware
                        this.gazeData.position.copy(eyeData.position || this.gazeData.position);
                        this.gazeData.direction.copy(eyeData.direction || this.gazeData.direction);
                        this.gazeData.confidence = eyeData.confidence || 0.8;
                        this.gazeData.timestamp = Date.now();
                    }
                }
            }
        }
    }

    updateGazeVisualization() {
        if (!this.gazeIndicator) return;
        
        // Cast ray to find gaze intersection
        const raycaster = new THREE.Raycaster();
        raycaster.set(this.gazeData.position, this.gazeData.direction);
        
        // Find intersection with scene objects
        const intersections = raycaster.intersectObjects(webXRManager.scene.children, true);
        
        if (intersections.length > 0) {
            const intersection = intersections[0];
            
            // Position gaze indicator at intersection point
            this.gazeIndicator.position.copy(intersection.point);
            this.gazeIndicator.lookAt(
                intersection.point.clone().add(intersection.face.normal)
            );
            
            // Update gaze ray
            this.gazeRay.position.copy(this.gazeData.position);
            this.gazeRay.lookAt(intersection.point);
            this.gazeRay.scale.z = intersection.distance;
        } else {
            // No intersection, place indicator at default distance
            const targetPoint = this.gazeData.position.clone()
                .add(this.gazeData.direction.clone().multiplyScalar(2));
                
            this.gazeIndicator.position.copy(targetPoint);
            
            this.gazeRay.position.copy(this.gazeData.position);
            this.gazeRay.lookAt(targetPoint);
            this.gazeRay.scale.z = 2;
        }
        
        // Update indicator opacity based on confidence
        this.gazeIndicator.material.opacity = this.gazeData.confidence * 0.8;
    }

    startFixationDetection() {
        const detectFixations = () => {
            if (!this.isActive) return;
            
            const currentTime = Date.now();
            const gazePosition = this.gazeData.position;
            
            if (this.fixationDetector.currentFixation) {
                // Check if still fixating
                const distance = gazePosition.distanceTo(this.fixationDetector.fixationPosition);
                const angularDistance = this.calculateAngularDistance(distance);
                
                if (angularDistance < this.fixationDetector.fixationThreshold) {
                    // Still fixating
                    this.fixationDetector.fixationDuration = currentTime - this.fixationDetector.fixationStartTime;
                    this.fixationDetector.isFixating = true;
                } else {
                    // Fixation ended
                    this.endFixation();
                }
            } else {
                // Check if starting new fixation
                if (this.gazeHistory.length > 5) {
                    const recentPositions = this.gazeHistory.slice(-5);
                    const isStable = this.checkGazeStability(recentPositions);
                    
                    if (isStable) {
                        this.startFixation(gazePosition, currentTime);
                    }
                }
            }
            
            requestAnimationFrame(detectFixations);
        };
        
        detectFixations();
    }

    startFixation(position, timestamp) {
        this.fixationDetector.currentFixation = {
            id: `fixation_${timestamp}`,
            startTime: timestamp,
            position: position.clone(),
            duration: 0
        };
        
        this.fixationDetector.fixationStartTime = timestamp;
        this.fixationDetector.fixationPosition.copy(position);
        this.fixationDetector.isFixating = true;
        
        console.log('👁️ Fixation started');
        
        // Trigger fixation start callback
        this.onFixationStart?.(this.fixationDetector.currentFixation);
    }

    endFixation() {
        if (this.fixationDetector.currentFixation) {
            const fixation = this.fixationDetector.currentFixation;
            fixation.duration = this.fixationDetector.fixationDuration;
            
            // Store in history
            this.fixationDetector.fixationHistory.push(fixation);
            
            // Update metrics
            this.attentionTracking.attentionMetrics.averageFixationDuration = 
                this.calculateAverageFixationDuration();
            
            console.log(`👁️ Fixation ended (${fixation.duration}ms)`);
            
            // Trigger fixation end callback
            this.onFixationEnd?.(fixation);
            
            // Reset fixation state
            this.fixationDetector.currentFixation = null;
            this.fixationDetector.isFixating = false;
        }
    }

    checkGazeStability(positions) {
        if (positions.length < 2) return false;
        
        let totalVariation = 0;
        for (let i = 1; i < positions.length; i++) {
            const distance = positions[i].position.distanceTo(positions[i-1].position);
            totalVariation += distance;
        }
        
        const averageVariation = totalVariation / (positions.length - 1);
        return averageVariation < 0.05; // Stable if small movement
    }

    startSaccadeDetection() {
        let previousPosition = null;
        let previousTime = null;
        
        const detectSaccades = () => {
            if (!this.isActive) return;
            
            const currentPosition = this.gazeData.position;
            const currentTime = Date.now();
            
            if (previousPosition && previousTime) {
                const timeDelta = (currentTime - previousTime) / 1000; // seconds
                
                if (timeDelta > 0) {
                    const distance = currentPosition.distanceTo(previousPosition);
                    const angularDistance = this.calculateAngularDistance(distance);
                    const velocity = angularDistance / timeDelta; // degrees/second
                    
                    this.saccadeDetector.saccadeVelocity.copy(currentPosition)
                        .sub(previousPosition).divideScalar(timeDelta);
                    
                    if (velocity > this.saccadeDetector.saccadeThreshold) {
                        // Saccade detected
                        if (!this.saccadeDetector.isSaccading) {
                            this.startSaccade(currentTime, velocity);
                        }
                        
                        this.saccadeDetector.peakVelocity = Math.max(
                            this.saccadeDetector.peakVelocity, 
                            velocity
                        );
                    } else if (this.saccadeDetector.isSaccading) {
                        // Saccade ended
                        this.endSaccade(currentTime);
                    }
                }
            }
            
            previousPosition = currentPosition.clone();
            previousTime = currentTime;
            
            requestAnimationFrame(detectSaccades);
        };
        
        detectSaccades();
    }

    startSaccade(timestamp, velocity) {
        this.saccadeDetector.isSaccading = true;
        this.saccadeDetector.saccadeStartTime = timestamp;
        this.saccadeDetector.peakVelocity = velocity;
        
        console.log(`⚡ Saccade started (${velocity.toFixed(1)}°/s)`);
        
        // Trigger saccade start callback
        this.onSaccadeStart?.(velocity);
    }

    endSaccade(timestamp) {
        if (this.saccadeDetector.isSaccading) {
            const duration = timestamp - this.saccadeDetector.saccadeStartTime;
            
            const saccade = {
                startTime: this.saccadeDetector.saccadeStartTime,
                endTime: timestamp,
                duration: duration,
                peakVelocity: this.saccadeDetector.peakVelocity
            };
            
            this.saccadeDetector.saccadeHistory.push(saccade);
            this.attentionTracking.attentionMetrics.saccadeCount++;
            
            console.log(`⚡ Saccade ended (${duration}ms, peak: ${this.saccadeDetector.peakVelocity.toFixed(1)}°/s)`);
            
            // Trigger saccade end callback
            this.onSaccadeEnd?.(saccade);
            
            // Reset saccade state
            this.saccadeDetector.isSaccading = false;
            this.saccadeDetector.peakVelocity = 0;
        }
    }

    startAttentionTracking() {
        this.attentionTracking.enabled = true;
        
        const trackAttention = () => {
            if (!this.isActive || !this.attentionTracking.enabled) return;
            
            // Update attention metrics
            this.updateAttentionMetrics();
            
            // Update heatmap
            this.updateAttentionHeatmap();
            
            // Calculate cognitive load
            this.calculateCognitiveLoad();
            
            // Calculate engagement level
            this.calculateEngagementLevel();
            
            setTimeout(trackAttention, 100); // Update every 100ms
        };
        
        trackAttention();
    }

    updateAttentionData() {
        const gazePosition = this.gazeData.position;
        
        // Update heatmap data
        const heatmapKey = this.positionToHeatmapKey(gazePosition);
        const currentValue = this.attentionTracking.heatmapData.get(heatmapKey) || 0;
        this.attentionTracking.heatmapData.set(heatmapKey, currentValue + 1);
        
        // Update total gaze time
        this.attentionTracking.attentionMetrics.totalGazeTime += 16; // ~60fps
    }

    updateAttentionMetrics() {
        const metrics = this.attentionTracking.attentionMetrics;
        
        // Update average fixation duration
        metrics.averageFixationDuration = this.calculateAverageFixationDuration();
        
        // Update other metrics as needed
        // This is where you'd implement more sophisticated attention analysis
    }

    updateAttentionHeatmap() {
        // Update the visual heatmap based on accumulated gaze data
        // This would involve updating the heatmap texture
        
        if (this.attentionTracking.heatmapVisualization) {
            const material = this.attentionTracking.heatmapVisualization.material;
            material.uniforms.time.value = Date.now() * 0.001;
        }
    }

    calculateCognitiveLoad() {
        // Calculate cognitive load based on gaze patterns
        const recentSaccades = this.saccadeDetector.saccadeHistory.slice(-10);
        const recentFixations = this.fixationDetector.fixationHistory.slice(-10);
        
        let cognitiveLoad = 0;
        
        // High saccade frequency indicates higher cognitive load
        if (recentSaccades.length > 0) {
            const avgSaccadeDuration = recentSaccades.reduce((sum, s) => sum + s.duration, 0) / recentSaccades.length;
            cognitiveLoad += Math.min(1, recentSaccades.length / 10) * 0.4;
        }
        
        // Short fixations indicate higher cognitive load
        if (recentFixations.length > 0) {
            const avgFixationDuration = recentFixations.reduce((sum, f) => sum + f.duration, 0) / recentFixations.length;
            cognitiveLoad += Math.max(0, (500 - avgFixationDuration) / 500) * 0.6;
        }
        
        this.attentionTracking.attentionMetrics.cognitiveLoad = Math.min(1, cognitiveLoad);
    }

    calculateEngagementLevel() {
        // Calculate engagement based on gaze stability and fixation patterns
        const recentFixations = this.fixationDetector.fixationHistory.slice(-5);
        
        let engagement = 0;
        
        if (recentFixations.length > 0) {
            // Longer fixations indicate higher engagement
            const avgFixationDuration = recentFixations.reduce((sum, f) => sum + f.duration, 0) / recentFixations.length;
            engagement += Math.min(1, avgFixationDuration / 1000) * 0.7;
            
            // Consistent fixation positions indicate focus
            const positionVariability = this.calculateFixationVariability(recentFixations);
            engagement += Math.max(0, (1 - positionVariability)) * 0.3;
        }
        
        this.attentionTracking.attentionMetrics.engagementLevel = Math.min(1, engagement);
    }

    processGazeInteractions() {
        if (!this.gazeInteraction.currentTarget) {
            // Check for new gaze target
            const target = this.findGazeTarget();
            if (target) {
                this.startGazeInteraction(target);
            }
        } else {
            // Check if still looking at current target
            const target = this.findGazeTarget();
            if (target === this.gazeInteraction.currentTarget) {
                this.updateGazeInteraction();
            } else {
                this.endGazeInteraction();
                if (target) {
                    this.startGazeInteraction(target);
                }
            }
        }
    }

    findGazeTarget() {
        // Cast ray to find interactive objects
        const raycaster = new THREE.Raycaster();
        raycaster.set(this.gazeData.position, this.gazeData.direction);
        
        // Find interactive objects in scene
        const interactiveObjects = webXRManager.scene.children.filter(
            obj => obj.userData.gazeInteractive
        );
        
        const intersections = raycaster.intersectObjects(interactiveObjects, true);
        
        return intersections.length > 0 ? intersections[0].object : null;
    }

    startGazeInteraction(target) {
        this.gazeInteraction.currentTarget = target;
        this.gazeInteraction.dwellStartTime = Date.now();
        this.gazeInteraction.isSelecting = false;
        this.gazeInteraction.selectionProgress = 0;
        
        // Trigger hover callback
        if (target.userData.onGazeEnter) {
            target.userData.onGazeEnter(target);
        }
        
        console.log('👁️ Gaze interaction started');
    }

    updateGazeInteraction() {
        const currentTime = Date.now();
        const dwellTime = currentTime - this.gazeInteraction.dwellStartTime;
        
        this.gazeInteraction.selectionProgress = Math.min(1, dwellTime / this.gazeInteraction.dwellTime);
        
        // Trigger progress callback
        if (this.gazeInteraction.currentTarget.userData.onGazeProgress) {
            this.gazeInteraction.currentTarget.userData.onGazeProgress(
                this.gazeInteraction.currentTarget,
                this.gazeInteraction.selectionProgress
            );
        }
        
        // Check if dwell time reached
        if (dwellTime >= this.gazeInteraction.dwellTime && !this.gazeInteraction.isSelecting) {
            this.gazeInteraction.isSelecting = true;
            
            // Trigger selection
            if (this.gazeInteraction.currentTarget.userData.onGazeSelect) {
                this.gazeInteraction.currentTarget.userData.onGazeSelect(this.gazeInteraction.currentTarget);
            }
            
            // Trigger haptic feedback
            webXRManager.pulseHaptic(0.6, 100);
            
            console.log('👁️ Gaze selection triggered');
        }
    }

    endGazeInteraction() {
        if (this.gazeInteraction.currentTarget) {
            // Trigger exit callback
            if (this.gazeInteraction.currentTarget.userData.onGazeExit) {
                this.gazeInteraction.currentTarget.userData.onGazeExit(this.gazeInteraction.currentTarget);
            }
            
            this.gazeInteraction.currentTarget = null;
            this.gazeInteraction.isSelecting = false;
            this.gazeInteraction.selectionProgress = 0;
            
            console.log('👁️ Gaze interaction ended');
        }
    }

    // Utility methods
    calculateAngularDistance(linearDistance) {
        // Convert linear distance to angular distance (simplified)
        const viewingDistance = 2; // meters
        return (linearDistance / viewingDistance) * (180 / Math.PI);
    }

    calculateAverageFixationDuration() {
        const fixations = this.fixationDetector.fixationHistory;
        if (fixations.length === 0) return 0;
        
        const totalDuration = fixations.reduce((sum, fixation) => sum + fixation.duration, 0);
        return totalDuration / fixations.length;
    }

    calculateFixationVariability(fixations) {
        if (fixations.length < 2) return 0;
        
        const positions = fixations.map(f => f.position);
        const centroid = new THREE.Vector3();
        
        positions.forEach(pos => centroid.add(pos));
        centroid.divideScalar(positions.length);
        
        let totalVariation = 0;
        positions.forEach(pos => {
            totalVariation += pos.distanceTo(centroid);
        });
        
        return totalVariation / positions.length;
    }

    positionToHeatmapKey(position) {
        // Convert 3D position to heatmap grid key
        const gridSize = 0.2;
        const x = Math.floor(position.x / gridSize);
        const y = Math.floor(position.y / gridSize);
        const z = Math.floor(position.z / gridSize);
        return `${x},${y},${z}`;
    }

    // Public API methods
    setGazeIndicatorVisibility(visible) {
        if (this.gazeIndicator) {
            this.gazeIndicator.visible = visible;
        }
        if (this.gazeRay) {
            this.gazeRay.visible = visible;
        }
    }

    setDwellTime(milliseconds) {
        this.gazeInteraction.dwellTime = milliseconds;
    }

    enableFoveatedRendering(enabled = true) {
        this.foveatedRendering.enabled = enabled;
        console.log(`🔍 Foveated rendering ${enabled ? 'enabled' : 'disabled'}`);
    }

    enableAttentionHeatmap(enabled = true) {
        if (this.attentionTracking.heatmapVisualization) {
            this.attentionTracking.heatmapVisualization.visible = enabled;
        }
    }

    getAttentionMetrics() {
        return { ...this.attentionTracking.attentionMetrics };
    }

    getGazeHistory(count = 100) {
        return this.gazeHistory.slice(-count);
    }

    getCurrentGazeData() {
        return { ...this.gazeData };
    }

    addGazeInteractiveObject(object, callbacks = {}) {
        object.userData.gazeInteractive = true;
        object.userData.onGazeEnter = callbacks.onEnter;
        object.userData.onGazeExit = callbacks.onExit;
        object.userData.onGazeSelect = callbacks.onSelect;
        object.userData.onGazeProgress = callbacks.onProgress;
    }

    removeGazeInteractiveObject(object) {
        object.userData.gazeInteractive = false;
        delete object.userData.onGazeEnter;
        delete object.userData.onGazeExit;
        delete object.userData.onGazeSelect;
        delete object.userData.onGazeProgress;
    }

    stopEyeTracking() {
        this.isActive = false;
        
        if (this.gazeIndicator) {
            this.gazeIndicator.visible = false;
        }
        
        if (this.gazeRay) {
            this.gazeRay.visible = false;
        }
        
        console.log('👁️ Eye tracking stopped');
    }

    // Calibration methods
    async startCalibration() {
        console.log('🎯 Starting gaze calibration...');
        
        this.gazeCalibration.currentCalibrationPoint = 0;
        this.gazeCalibration.calibrationData = [];
        
        return this.showNextCalibrationPoint();
    }

    showNextCalibrationPoint() {
        const point = this.gazeCalibration.calibrationPoints[this.gazeCalibration.currentCalibrationPoint];
        
        if (!point) {
            // Calibration complete
            return this.completeCalibration();
        }
        
        // Show calibration point visualization
        this.showCalibrationTarget(point);
        
        return new Promise((resolve) => {
            setTimeout(() => {
                this.recordCalibrationData(point);
                this.gazeCalibration.currentCalibrationPoint++;
                resolve(this.showNextCalibrationPoint());
            }, 2000);
        });
    }

    showCalibrationTarget(point) {
        // Create calibration target
        const targetGeometry = new THREE.RingGeometry(0.02, 0.04, 16);
        const targetMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.8
        });
        
        const target = new THREE.Mesh(targetGeometry, targetMaterial);
        target.position.set(point.x, point.y, point.z);
        target.name = 'calibration-target';
        
        webXRManager.scene.add(target);
        
        // Remove previous target
        const previousTarget = webXRManager.scene.getObjectByName('calibration-target');
        if (previousTarget) {
            webXRManager.scene.remove(previousTarget);
        }
        
        webXRManager.scene.add(target);
        
        // Animate target
        const animate = () => {
            target.scale.setScalar(1 + Math.sin(Date.now() * 0.01) * 0.2);
            requestAnimationFrame(animate);
        };
        animate();
    }

    recordCalibrationData(targetPoint) {
        const gazeData = { ...this.gazeData };
        
        this.gazeCalibration.calibrationData.push({
            target: targetPoint,
            gaze: gazeData,
            timestamp: Date.now()
        });
        
        console.log(`📍 Recorded calibration data for point ${this.gazeCalibration.currentCalibrationPoint + 1}`);
    }

    completeCalibration() {
        // Calculate calibration offset
        this.calculateCalibrationOffset();
        
        // Remove calibration target
        const target = webXRManager.scene.getObjectByName('calibration-target');
        if (target) {
            webXRManager.scene.remove(target);
        }
        
        this.gazeCalibration.isCalibrated = true;
        
        console.log('✅ Gaze calibration completed');
        
        return {
            success: true,
            accuracy: this.gazeCalibration.accuracyRadius,
            offset: this.gazeCalibration.offsetCorrection
        };
    }

    calculateCalibrationOffset() {
        const data = this.gazeCalibration.calibrationData;
        
        let totalOffset = new THREE.Vector3();
        
        data.forEach(record => {
            const targetPos = new THREE.Vector3(record.target.x, record.target.y, record.target.z);
            const gazePos = record.gaze.position;
            
            const offset = targetPos.clone().sub(gazePos);
            totalOffset.add(offset);
        });
        
        this.gazeCalibration.offsetCorrection = totalOffset.divideScalar(data.length);
        
        // Calculate accuracy
        let totalError = 0;
        data.forEach(record => {
            const targetPos = new THREE.Vector3(record.target.x, record.target.y, record.target.z);
            const correctedGaze = record.gaze.position.clone().add(this.gazeCalibration.offsetCorrection);
            
            totalError += targetPos.distanceTo(correctedGaze);
        });
        
        this.gazeCalibration.accuracyRadius = totalError / data.length;
    }
}

// Export Eye Tracking class
export default EyeTracking;