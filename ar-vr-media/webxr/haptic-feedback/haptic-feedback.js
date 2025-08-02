/**
 * Advanced Haptic Feedback System - Multi-modal tactile experiences
 * Supports controller vibration, hand haptics, and spatial tactile feedback
 */

import { webXRManager } from '../core/webxr-manager.js';

export class HapticFeedback {
    constructor() {
        this.hapticActuators = new Map();
        this.audioHaptics = null;
        this.spatialHaptics = null;
        this.hapticPatterns = new Map();
        this.hapticEffects = new Map();
        this.hapticSequencer = null;
        this.ultrasonicHaptics = null;
        this.thermalFeedback = null;
        
        // Haptic feedback types
        this.feedbackTypes = {
            VIBRATION: 'vibration',
            AUDIO_HAPTIC: 'audio_haptic',
            ULTRASONIC: 'ultrasonic',
            THERMAL: 'thermal',
            FORCE_FEEDBACK: 'force_feedback',
            TEXTURE: 'texture'
        };
        
        // Device-specific optimizations
        this.deviceOptimizations = {
            appleVisionPro: {
                spatialAudio: true,
                thermalFeedback: false // Not yet available
            },
            metaQuest3: {
                controllerVibration: true,
                handTracking: true,
                spatialAudio: true
            },
            generic: {
                controllerVibration: true,
                audioHaptics: true
            }
        };
        
        this.init();
    }

    async init() {
        console.log('🤝 Initializing Advanced Haptic Feedback System...');
        
        // Initialize haptic actuators
        this.initHapticActuators();
        
        // Setup audio haptics
        this.setupAudioHaptics();
        
        // Initialize spatial haptics
        this.setupSpatialHaptics();
        
        // Setup haptic patterns
        this.defineHapticPatterns();
        
        // Initialize haptic effects
        this.setupHapticEffects();
        
        // Setup haptic sequencer
        this.setupHapticSequencer();
        
        // Initialize advanced haptic technologies
        this.initAdvancedHaptics();
        
        // Detect and optimize for device
        this.optimizeForDevice();
        
        console.log('✅ Advanced Haptic Feedback System initialized');
    }

    initHapticActuators() {
        console.log('📳 Initializing haptic actuators...');
        
        // Initialize controller haptic actuators
        this.initControllerHaptics();
        
        // Initialize hand haptic points
        this.initHandHaptics();
        
        // Setup haptic feedback mapping
        this.setupHapticMapping();
    }

    initControllerHaptics() {
        // Controller haptic actuators will be populated when session starts
        this.controllerHaptics = {
            left: null,
            right: null,
            capabilities: new Map()
        };
    }

    initHandHaptics() {
        // Define haptic zones on hands for spatial feedback
        this.handHapticZones = {
            left: {
                palm: { position: new THREE.Vector3(-0.02, 0, 0), intensity: 1.0 },
                thumb: { position: new THREE.Vector3(-0.03, 0.02, 0.01), intensity: 0.8 },
                index: { position: new THREE.Vector3(-0.01, 0.03, -0.02), intensity: 0.9 },
                middle: { position: new THREE.Vector3(0, 0.035, -0.02), intensity: 0.9 },
                ring: { position: new THREE.Vector3(0.01, 0.03, -0.02), intensity: 0.8 },
                pinky: { position: new THREE.Vector3(0.02, 0.025, -0.015), intensity: 0.7 }
            },
            right: {
                palm: { position: new THREE.Vector3(0.02, 0, 0), intensity: 1.0 },
                thumb: { position: new THREE.Vector3(0.03, 0.02, 0.01), intensity: 0.8 },
                index: { position: new THREE.Vector3(0.01, 0.03, -0.02), intensity: 0.9 },
                middle: { position: new THREE.Vector3(0, 0.035, -0.02), intensity: 0.9 },
                ring: { position: new THREE.Vector3(-0.01, 0.03, -0.02), intensity: 0.8 },
                pinky: { position: new THREE.Vector3(-0.02, 0.025, -0.015), intensity: 0.7 }
            }
        };
    }

    setupHapticMapping() {
        // Map physical interaction types to haptic responses
        this.hapticMapping = {
            button_press: { type: 'vibration', intensity: 0.6, duration: 50 },
            button_release: { type: 'vibration', intensity: 0.3, duration: 30 },
            slider_drag: { type: 'texture', intensity: 0.4, frequency: 20 },
            object_grab: { type: 'vibration', intensity: 0.8, duration: 100 },
            object_release: { type: 'vibration', intensity: 0.4, duration: 60 },
            collision: { type: 'vibration', intensity: 1.0, duration: 80 },
            surface_touch: { type: 'texture', intensity: 0.5, frequency: 15 },
            air_tap: { type: 'vibration', intensity: 0.5, duration: 40 },
            pinch: { type: 'vibration', intensity: 0.7, duration: 70 },
            swipe: { type: 'audio_haptic', intensity: 0.6, duration: 150 },
            scroll: { type: 'texture', intensity: 0.3, frequency: 10 },
            notification: { type: 'vibration', intensity: 0.4, duration: 200 },
            success: { type: 'audio_haptic', pattern: 'success_chime' },
            error: { type: 'audio_haptic', pattern: 'error_buzz' },
            warning: { type: 'vibration', intensity: 0.8, duration: 300 }
        };
    }

    setupAudioHaptics() {
        console.log('🔊 Setting up audio haptics...');
        
        this.audioHaptics = {
            context: null,
            lowFreqOscillator: null,
            subBassFilter: null,
            spatialPanner: null,
            hapticGain: null,
            isEnabled: true
        };
        
        // Initialize Web Audio API for haptic feedback
        this.initAudioHapticEngine();
    }

    async initAudioHapticEngine() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioHaptics.context = new AudioContext();
            
            // Create low-frequency oscillator for haptic effects
            this.audioHaptics.lowFreqOscillator = this.audioHaptics.context.createOscillator();
            this.audioHaptics.lowFreqOscillator.type = 'sine';
            this.audioHaptics.lowFreqOscillator.frequency.setValueAtTime(40, this.audioHaptics.context.currentTime);
            
            // Create sub-bass filter
            this.audioHaptics.subBassFilter = this.audioHaptics.context.createBiquadFilter();
            this.audioHaptics.subBassFilter.type = 'lowpass';
            this.audioHaptics.subBassFilter.frequency.setValueAtTime(80, this.audioHaptics.context.currentTime);
            this.audioHaptics.subBassFilter.Q.setValueAtTime(2, this.audioHaptics.context.currentTime);
            
            // Create spatial panner for directional haptics
            this.audioHaptics.spatialPanner = this.audioHaptics.context.createPanner();
            this.audioHaptics.spatialPanner.panningModel = 'HRTF';
            this.audioHaptics.spatialPanner.distanceModel = 'inverse';
            this.audioHaptics.spatialPanner.refDistance = 1;
            this.audioHaptics.spatialPanner.maxDistance = 10;
            
            // Create gain node for haptic intensity control
            this.audioHaptics.hapticGain = this.audioHaptics.context.createGain();
            this.audioHaptics.hapticGain.gain.setValueAtTime(0, this.audioHaptics.context.currentTime);
            
            // Connect audio graph
            this.audioHaptics.lowFreqOscillator.connect(this.audioHaptics.subBassFilter);
            this.audioHaptics.subBassFilter.connect(this.audioHaptics.spatialPanner);
            this.audioHaptics.spatialPanner.connect(this.audioHaptics.hapticGain);
            this.audioHaptics.hapticGain.connect(this.audioHaptics.context.destination);
            
            // Start oscillator
            this.audioHaptics.lowFreqOscillator.start();
            
            console.log('🎵 Audio haptic engine initialized');
            
        } catch (error) {
            console.warn('Audio haptics not available:', error);
            this.audioHaptics.isEnabled = false;
        }
    }

    setupSpatialHaptics() {
        console.log('🌐 Setting up spatial haptics...');
        
        this.spatialHaptics = {
            hapticField: new Map(),
            activeFeedbackZones: new Map(),
            hapticObjects: new Map(),
            spatialResolution: 0.01, // 1cm resolution
            maxDistance: 5.0, // 5 meter range
            falloffExponent: 2.0
        };
        
        // Initialize spatial haptic field
        this.initSpatialHapticField();
    }

    initSpatialHapticField() {
        // Create a spatial grid for haptic feedback zones
        const fieldSize = 10; // 10x10x10 meter space
        const resolution = this.spatialHaptics.spatialResolution;
        const gridSize = Math.ceil(fieldSize / resolution);
        
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                for (let z = 0; z < gridSize; z++) {
                    const key = `${x},${y},${z}`;
                    this.spatialHaptics.hapticField.set(key, {
                        position: new THREE.Vector3(
                            (x - gridSize/2) * resolution,
                            (y - gridSize/2) * resolution,
                            (z - gridSize/2) * resolution
                        ),
                        intensity: 0,
                        frequency: 0,
                        type: 'none'
                    });
                }
            }
        }
    }

    defineHapticPatterns() {
        console.log('🎼 Defining haptic patterns...');
        
        // Define common haptic patterns
        const patterns = {
            // Basic patterns
            single_pulse: {
                name: 'Single Pulse',
                sequence: [{ intensity: 1.0, duration: 100, delay: 0 }]
            },
            double_pulse: {
                name: 'Double Pulse',
                sequence: [
                    { intensity: 0.8, duration: 80, delay: 0 },
                    { intensity: 0.8, duration: 80, delay: 150 }
                ]
            },
            triple_pulse: {
                name: 'Triple Pulse',
                sequence: [
                    { intensity: 0.7, duration: 60, delay: 0 },
                    { intensity: 0.7, duration: 60, delay: 100 },
                    { intensity: 0.7, duration: 60, delay: 200 }
                ]
            },
            
            // Rhythmic patterns
            heartbeat: {
                name: 'Heartbeat',
                sequence: [
                    { intensity: 0.8, duration: 100, delay: 0 },
                    { intensity: 0.6, duration: 80, delay: 120 },
                    { intensity: 0.4, duration: 60, delay: 800 }
                ],
                loop: true
            },
            breathing: {
                name: 'Breathing',
                sequence: [
                    { intensity: 0.3, duration: 2000, delay: 0, envelope: 'sine' },
                    { intensity: 0.0, duration: 1000, delay: 2000 }
                ],
                loop: true
            },
            
            // Notification patterns
            success_chime: {
                name: 'Success Chime',
                sequence: [
                    { intensity: 0.6, duration: 50, delay: 0, frequency: 60 },
                    { intensity: 0.8, duration: 100, delay: 60, frequency: 80 },
                    { intensity: 0.4, duration: 150, delay: 180, frequency: 100 }
                ]
            },
            error_buzz: {
                name: 'Error Buzz',
                sequence: [
                    { intensity: 1.0, duration: 200, delay: 0, frequency: 20 },
                    { intensity: 0.8, duration: 200, delay: 220, frequency: 25 },
                    { intensity: 1.0, duration: 200, delay: 440, frequency: 20 }
                ]
            },
            
            // Interactive patterns
            button_click: {
                name: 'Button Click',
                sequence: [
                    { intensity: 0.8, duration: 30, delay: 0, frequency: 150 },
                    { intensity: 0.4, duration: 20, delay: 35, frequency: 100 }
                ]
            },
            drag_texture: {
                name: 'Drag Texture',
                sequence: [
                    { intensity: 0.5, duration: 50, delay: 0, frequency: 30 }
                ],
                continuous: true
            },
            collision_impact: {
                name: 'Collision Impact',
                sequence: [
                    { intensity: 1.0, duration: 20, delay: 0, frequency: 200 },
                    { intensity: 0.6, duration: 80, delay: 25, frequency: 100 },
                    { intensity: 0.3, duration: 150, delay: 110, frequency: 50 }
                ]
            },
            
            // Spatial patterns
            wave_left_right: {
                name: 'Wave Left to Right',
                sequence: [
                    { intensity: 0.8, duration: 100, delay: 0, spatial: 'left' },
                    { intensity: 0.8, duration: 100, delay: 100, spatial: 'center' },
                    { intensity: 0.8, duration: 100, delay: 200, spatial: 'right' }
                ]
            },
            circular_wave: {
                name: 'Circular Wave',
                sequence: [
                    { intensity: 0.6, duration: 80, delay: 0, spatial: 'north' },
                    { intensity: 0.6, duration: 80, delay: 80, spatial: 'northeast' },
                    { intensity: 0.6, duration: 80, delay: 160, spatial: 'east' },
                    { intensity: 0.6, duration: 80, delay: 240, spatial: 'southeast' },
                    { intensity: 0.6, duration: 80, delay: 320, spatial: 'south' },
                    { intensity: 0.6, duration: 80, delay: 400, spatial: 'southwest' },
                    { intensity: 0.6, duration: 80, delay: 480, spatial: 'west' },
                    { intensity: 0.6, duration: 80, delay: 560, spatial: 'northwest' }
                ]
            },
            
            // Emotional patterns
            gentle_comfort: {
                name: 'Gentle Comfort',
                sequence: [
                    { intensity: 0.3, duration: 1000, delay: 0, envelope: 'sine_soft' },
                    { intensity: 0.2, duration: 1500, delay: 1100, envelope: 'sine_soft' },
                    { intensity: 0.3, duration: 1000, delay: 2700, envelope: 'sine_soft' }
                ],
                loop: true
            },
            urgent_alert: {
                name: 'Urgent Alert',
                sequence: [
                    { intensity: 1.0, duration: 100, delay: 0 },
                    { intensity: 0.8, duration: 100, delay: 120 },
                    { intensity: 1.0, duration: 100, delay: 240 },
                    { intensity: 0.0, duration: 200, delay: 360 }
                ],
                loop: true,
                priority: 'high'
            }
        };
        
        // Store patterns
        Object.entries(patterns).forEach(([key, pattern]) => {
            this.hapticPatterns.set(key, pattern);
        });
        
        console.log(`📋 Loaded ${Object.keys(patterns).length} haptic patterns`);
    }

    setupHapticEffects() {
        console.log('✨ Setting up haptic effects...');
        
        this.hapticEffects = new Map([
            ['fade_in', this.createFadeInEffect()],
            ['fade_out', this.createFadeOutEffect()],
            ['pulse', this.createPulseEffect()],
            ['crescendo', this.createCrescendoEffect()],
            ['diminuendo', this.createDiminuendoEffect()],
            ['tremolo', this.createTremoloEffect()],
            ['staccato', this.createStaccatoEffect()],
            ['legato', this.createLegatoEffect()]
        ]);
    }

    createFadeInEffect() {
        return {
            name: 'Fade In',
            apply: (intensity, duration, elapsed) => {
                const progress = Math.min(elapsed / duration, 1);
                return intensity * this.easeInQuad(progress);
            }
        };
    }

    createFadeOutEffect() {
        return {
            name: 'Fade Out',
            apply: (intensity, duration, elapsed) => {
                const progress = Math.min(elapsed / duration, 1);
                return intensity * (1 - this.easeOutQuad(progress));
            }
        };
    }

    createPulseEffect() {
        return {
            name: 'Pulse',
            apply: (intensity, duration, elapsed, frequency = 5) => {
                const pulseValue = Math.sin((elapsed / duration) * Math.PI * 2 * frequency);
                return intensity * (0.5 + 0.5 * pulseValue);
            }
        };
    }

    createCrescendoEffect() {
        return {
            name: 'Crescendo',
            apply: (intensity, duration, elapsed) => {
                const progress = Math.min(elapsed / duration, 1);
                return intensity * Math.pow(progress, 0.5);
            }
        };
    }

    createDiminuendoEffect() {
        return {
            name: 'Diminuendo',
            apply: (intensity, duration, elapsed) => {
                const progress = Math.min(elapsed / duration, 1);
                return intensity * Math.pow(1 - progress, 0.5);
            }
        };
    }

    createTremoloEffect() {
        return {
            name: 'Tremolo',
            apply: (intensity, duration, elapsed, frequency = 10) => {
                const tremolo = Math.sin((elapsed / 1000) * Math.PI * 2 * frequency);
                return intensity * (0.7 + 0.3 * tremolo);
            }
        };
    }

    createStaccatoEffect() {
        return {
            name: 'Staccato',
            apply: (intensity, duration, elapsed) => {
                const noteLength = 100; // ms
                const silenceLength = 50; // ms
                const cycleLength = noteLength + silenceLength;
                const cyclePosition = elapsed % cycleLength;
                
                return cyclePosition < noteLength ? intensity : 0;
            }
        };
    }

    createLegatoEffect() {
        return {
            name: 'Legato',
            apply: (intensity, duration, elapsed) => {
                // Smooth continuous effect
                return intensity;
            }
        };
    }

    setupHapticSequencer() {
        console.log('🎛️ Setting up haptic sequencer...');
        
        this.hapticSequencer = {
            activeSequences: new Map(),
            sequenceId: 0,
            isRunning: false,
            frameRate: 60 // 60 FPS for smooth haptics
        };
        
        // Start sequencer loop
        this.startHapticSequencer();
    }

    startHapticSequencer() {
        this.hapticSequencer.isRunning = true;
        
        const sequencerLoop = () => {
            if (!this.hapticSequencer.isRunning) return;
            
            const currentTime = Date.now();
            
            // Update all active sequences
            this.hapticSequencer.activeSequences.forEach((sequence, id) => {
                this.updateHapticSequence(sequence, currentTime);
                
                // Remove completed sequences
                if (sequence.completed) {
                    this.hapticSequencer.activeSequences.delete(id);
                }
            });
            
            // Schedule next frame
            setTimeout(sequencerLoop, 1000 / this.hapticSequencer.frameRate);
        };
        
        sequencerLoop();
    }

    updateHapticSequence(sequence, currentTime) {
        const elapsed = currentTime - sequence.startTime;
        
        sequence.steps.forEach((step, index) => {
            const stepStartTime = step.delay;
            const stepEndTime = stepStartTime + step.duration;
            
            if (elapsed >= stepStartTime && elapsed <= stepEndTime) {
                // Step is active
                const stepElapsed = elapsed - stepStartTime;
                let intensity = step.intensity;
                
                // Apply envelope/effect
                if (step.envelope) {
                    const effect = this.hapticEffects.get(step.envelope);
                    if (effect) {
                        intensity = effect.apply(intensity, step.duration, stepElapsed, step.frequency);
                    }
                }
                
                // Trigger haptic feedback
                this.triggerHapticFeedback(step.type || 'vibration', {
                    intensity: intensity,
                    duration: Math.min(50, step.duration - stepElapsed),
                    frequency: step.frequency,
                    spatial: step.spatial
                });
                
                step.active = true;
            } else if (step.active && elapsed > stepEndTime) {
                step.active = false;
                step.completed = true;
            }
        });
        
        // Check if sequence is completed
        const allStepsCompleted = sequence.steps.every(step => step.completed);
        if (allStepsCompleted && !sequence.loop) {
            sequence.completed = true;
        } else if (allStepsCompleted && sequence.loop) {
            // Restart sequence
            sequence.startTime = currentTime;
            sequence.steps.forEach(step => {
                step.active = false;
                step.completed = false;
            });
        }
    }

    initAdvancedHaptics() {
        console.log('🚀 Initializing advanced haptic technologies...');
        
        // Initialize ultrasonic haptics (if available)
        this.initUltrasonicHaptics();
        
        // Initialize thermal feedback (experimental)
        this.initThermalFeedback();
        
        // Initialize force feedback
        this.initForceFeedback();
    }

    initUltrasonicHaptics() {
        // Ultrasonic haptics for mid-air tactile feedback
        this.ultrasonicHaptics = {
            isSupported: false,
            focusPoints: new Map(),
            ultrasonicArray: null,
            maxFocusPoints: 16,
            focusResolution: 0.001 // 1mm resolution
        };
        
        // Check for ultrasonic haptic support (hypothetical API)
        if (navigator.ultrasonicHaptics) {
            this.ultrasonicHaptics.isSupported = true;
            console.log('🌊 Ultrasonic haptics available');
        }
    }

    initThermalFeedback() {
        // Thermal haptic feedback
        this.thermalFeedback = {
            isSupported: false,
            thermalZones: new Map(),
            temperatureRange: { min: 15, max: 40 }, // Celsius
            thermalActuators: []
        };
        
        // Check for thermal feedback support (hypothetical)
        if (navigator.thermalHaptics) {
            this.thermalFeedback.isSupported = true;
            console.log('🌡️ Thermal feedback available');
        }
    }

    initForceFeedback() {
        // Force feedback for resistance and weight simulation
        this.forceFeedback = {
            isSupported: false,
            forceActuators: new Map(),
            maxForce: 10, // Newtons
            forceResolution: 0.1 // 0.1N resolution
        };
        
        // Check for force feedback support
        if (navigator.forceFeedback) {
            this.forceFeedback.isSupported = true;
            console.log('💪 Force feedback available');
        }
    }

    optimizeForDevice() {
        const userAgent = navigator.userAgent.toLowerCase();
        
        if (userAgent.includes('vision') || userAgent.includes('visionos')) {
            this.optimizeForAppleVisionPro();
        } else if (userAgent.includes('quest') || userAgent.includes('oculus')) {
            this.optimizeForMetaQuest();
        } else {
            this.optimizeForGenericDevice();
        }
    }

    optimizeForAppleVisionPro() {
        console.log('🥽 Optimizing for Apple Vision Pro...');
        
        // Enable spatial audio haptics
        this.audioHaptics.isEnabled = true;
        
        // Enable high-resolution haptic feedback
        this.spatialHaptics.spatialResolution = 0.005; // 5mm resolution
        
        // Configure for precision interactions
        this.hapticMapping.air_tap = { type: 'audio_haptic', intensity: 0.4, duration: 30 };
        this.hapticMapping.pinch = { type: 'audio_haptic', intensity: 0.6, duration: 50 };
    }

    optimizeForMetaQuest() {
        console.log('🥽 Optimizing for Meta Quest...');
        
        // Enable controller vibration
        this.controllerHaptics.capabilities.set('vibration', true);
        
        // Configure for hand tracking
        this.handHapticZones.left.palm.intensity = 0.8;
        this.handHapticZones.right.palm.intensity = 0.8;
        
        // Optimize for mixed reality
        this.spatialHaptics.maxDistance = 3.0; // Closer range for MR
    }

    optimizeForGenericDevice() {
        console.log('🔧 Optimizing for generic device...');
        
        // Use conservative settings
        this.audioHaptics.isEnabled = true;
        this.spatialHaptics.spatialResolution = 0.02; // 2cm resolution
    }

    // Public API Methods

    updateHapticActuators(frame) {
        if (!webXRManager.xrSession) return;
        
        // Update controller haptic actuators
        this.updateControllerHaptics();
        
        // Update spatial haptic field
        this.updateSpatialHapticField();
        
        // Update hand haptic zones
        this.updateHandHapticZones();
    }

    updateControllerHaptics() {
        if (!webXRManager.xrSession.inputSources) return;
        
        webXRManager.xrSession.inputSources.forEach(inputSource => {
            if (inputSource.gamepad && inputSource.gamepad.hapticActuators) {
                const handedness = inputSource.handedness;
                
                // Store haptic actuators
                this.controllerHaptics[handedness] = inputSource.gamepad.hapticActuators[0];
                
                // Update capabilities
                if (inputSource.gamepad.hapticActuators.length > 0) {
                    this.controllerHaptics.capabilities.set(`${handedness}_vibration`, true);
                }
            }
        });
    }

    updateSpatialHapticField() {
        // Update spatial haptic field based on active haptic objects
        this.spatialHaptics.hapticObjects.forEach((object, id) => {
            if (object.active) {
                this.calculateSpatialHapticInfluence(object);
            }
        });
    }

    calculateSpatialHapticInfluence(hapticObject) {
        const position = hapticObject.position;
        const intensity = hapticObject.intensity;
        const radius = hapticObject.radius;
        
        // Calculate influence on nearby grid points
        this.spatialHaptics.hapticField.forEach((gridPoint, key) => {
            const distance = gridPoint.position.distanceTo(position);
            
            if (distance <= radius) {
                const falloff = Math.pow(1 - (distance / radius), this.spatialHaptics.falloffExponent);
                const influence = intensity * falloff;
                
                gridPoint.intensity = Math.max(gridPoint.intensity, influence);
                gridPoint.type = hapticObject.type;
                gridPoint.frequency = hapticObject.frequency;
            }
        });
    }

    updateHandHapticZones() {
        // Update hand haptic zones based on hand tracking
        ['left', 'right'].forEach(handedness => {
            const wristJoint = webXRManager.getHandJoint(handedness, 'wrist');
            if (wristJoint) {
                // Update hand haptic zone positions
                Object.entries(this.handHapticZones[handedness]).forEach(([zone, data]) => {
                    const worldPosition = wristJoint.position.clone().add(data.position);
                    data.worldPosition = worldPosition;
                });
            }
        });
    }

    // Haptic Feedback Methods

    pulse(handedness = 'both', intensity = 0.5, duration = 100) {
        this.triggerHapticFeedback('vibration', {
            handedness: handedness,
            intensity: intensity,
            duration: duration
        });
    }

    playPattern(patternName, handedness = 'both', options = {}) {
        const pattern = this.hapticPatterns.get(patternName);
        if (!pattern) {
            console.warn(`Haptic pattern not found: ${patternName}`);
            return;
        }
        
        const sequence = {
            id: ++this.hapticSequencer.sequenceId,
            pattern: pattern,
            handedness: handedness,
            startTime: Date.now(),
            steps: pattern.sequence.map(step => ({
                ...step,
                active: false,
                completed: false,
                handedness: handedness
            })),
            loop: pattern.loop || options.loop || false,
            completed: false
        };
        
        this.hapticSequencer.activeSequences.set(sequence.id, sequence);
        
        console.log(`🎵 Playing haptic pattern: ${patternName}`);
        return sequence.id;
    }

    stopPattern(sequenceId) {
        this.hapticSequencer.activeSequences.delete(sequenceId);
        console.log(`⏹️ Stopped haptic sequence: ${sequenceId}`);
    }

    stopAllPatterns() {
        this.hapticSequencer.activeSequences.clear();
        console.log('⏹️ Stopped all haptic sequences');
    }

    triggerHapticFeedback(type, options = {}) {
        switch (type) {
            case this.feedbackTypes.VIBRATION:
                this.triggerVibration(options);
                break;
            case this.feedbackTypes.AUDIO_HAPTIC:
                this.triggerAudioHaptic(options);
                break;
            case this.feedbackTypes.ULTRASONIC:
                this.triggerUltrasonicHaptic(options);
                break;
            case this.feedbackTypes.THERMAL:
                this.triggerThermalFeedback(options);
                break;
            case this.feedbackTypes.FORCE_FEEDBACK:
                this.triggerForceFeedback(options);
                break;
            case this.feedbackTypes.TEXTURE:
                this.triggerTextureFeedback(options);
                break;
            default:
                console.warn(`Unknown haptic feedback type: ${type}`);
        }
    }

    triggerVibration(options) {
        const {
            handedness = 'both',
            intensity = 0.5,
            duration = 100,
            frequency = 0
        } = options;
        
        const hands = handedness === 'both' ? ['left', 'right'] : [handedness];
        
        hands.forEach(hand => {
            const actuator = this.controllerHaptics[hand];
            if (actuator && actuator.pulse) {
                actuator.pulse(intensity, duration).catch(error => {
                    console.debug(`Haptic pulse failed for ${hand}:`, error);
                });
            }
        });
    }

    triggerAudioHaptic(options) {
        if (!this.audioHaptics.isEnabled || !this.audioHaptics.context) return;
        
        const {
            intensity = 0.5,
            duration = 100,
            frequency = 40,
            position = null
        } = options;
        
        try {
            const currentTime = this.audioHaptics.context.currentTime;
            
            // Set frequency
            this.audioHaptics.lowFreqOscillator.frequency.setValueAtTime(
                frequency, 
                currentTime
            );
            
            // Set spatial position if provided
            if (position && this.audioHaptics.spatialPanner) {
                this.audioHaptics.spatialPanner.positionX.setValueAtTime(
                    position.x, 
                    currentTime
                );
                this.audioHaptics.spatialPanner.positionY.setValueAtTime(
                    position.y, 
                    currentTime
                );
                this.audioHaptics.spatialPanner.positionZ.setValueAtTime(
                    position.z, 
                    currentTime
                );
            }
            
            // Trigger haptic pulse
            this.audioHaptics.hapticGain.gain.setValueAtTime(0, currentTime);
            this.audioHaptics.hapticGain.gain.linearRampToValueAtTime(
                intensity, 
                currentTime + 0.01
            );
            this.audioHaptics.hapticGain.gain.exponentialRampToValueAtTime(
                0.001, 
                currentTime + (duration / 1000)
            );
            
        } catch (error) {
            console.debug('Audio haptic error:', error);
        }
    }

    triggerUltrasonicHaptic(options) {
        if (!this.ultrasonicHaptics.isSupported) return;
        
        const {
            position,
            intensity = 0.5,
            duration = 100,
            focusRadius = 0.01
        } = options;
        
        // Create ultrasonic focus point
        const focusId = `focus_${Date.now()}`;
        this.ultrasonicHaptics.focusPoints.set(focusId, {
            position: position.clone(),
            intensity: intensity,
            radius: focusRadius,
            startTime: Date.now(),
            duration: duration
        });
        
        // Remove focus point after duration
        setTimeout(() => {
            this.ultrasonicHaptics.focusPoints.delete(focusId);
        }, duration);
        
        console.log('🌊 Ultrasonic haptic triggered');
    }

    triggerThermalFeedback(options) {
        if (!this.thermalFeedback.isSupported) return;
        
        const {
            position,
            temperature = 25, // Celsius
            duration = 1000
        } = options;
        
        console.log(`🌡️ Thermal feedback: ${temperature}°C`);
    }

    triggerForceFeedback(options) {
        if (!this.forceFeedback.isSupported) return;
        
        const {
            direction,
            magnitude = 1.0, // Newtons
            duration = 100
        } = options;
        
        console.log(`💪 Force feedback: ${magnitude}N`);
    }

    triggerTextureFeedback(options) {
        const {
            textureType = 'rough',
            intensity = 0.4,
            frequency = 20,
            duration = 50
        } = options;
        
        // Simulate texture through rapid micro-vibrations
        this.triggerVibration({
            ...options,
            intensity: intensity,
            duration: duration
        });
        
        // Add audio component for texture
        this.triggerAudioHaptic({
            ...options,
            frequency: frequency,
            intensity: intensity * 0.3,
            duration: duration
        });
    }

    // Interaction-based haptic feedback
    onButtonPress(button) {
        const mapping = this.hapticMapping.button_press;
        this.triggerHapticFeedback(mapping.type, mapping);
    }

    onButtonRelease(button) {
        const mapping = this.hapticMapping.button_release;
        this.triggerHapticFeedback(mapping.type, mapping);
    }

    onObjectGrab(object, handedness) {
        const mapping = this.hapticMapping.object_grab;
        this.triggerHapticFeedback(mapping.type, { ...mapping, handedness });
        
        // Add object-specific haptic properties
        if (object.userData.hapticProperties) {
            this.applyObjectHaptics(object, handedness);
        }
    }

    onObjectRelease(object, handedness) {
        const mapping = this.hapticMapping.object_release;
        this.triggerHapticFeedback(mapping.type, { ...mapping, handedness });
    }

    onSurfaceTouch(surface, handedness, contactPoint) {
        const mapping = this.hapticMapping.surface_touch;
        this.triggerHapticFeedback(mapping.type, { 
            ...mapping, 
            handedness,
            position: contactPoint 
        });
    }

    onCollision(object1, object2, impactVelocity) {
        const mapping = this.hapticMapping.collision;
        const intensityMultiplier = Math.min(impactVelocity / 5.0, 1.0);
        
        this.triggerHapticFeedback(mapping.type, {
            ...mapping,
            intensity: mapping.intensity * intensityMultiplier
        });
    }

    applyObjectHaptics(object, handedness) {
        const props = object.userData.hapticProperties;
        
        if (props.texture) {
            this.triggerTextureFeedback({
                textureType: props.texture,
                handedness: handedness,
                intensity: props.textureIntensity || 0.4
            });
        }
        
        if (props.temperature) {
            this.triggerThermalFeedback({
                temperature: props.temperature,
                handedness: handedness
            });
        }
        
        if (props.weight) {
            this.triggerForceFeedback({
                magnitude: props.weight,
                direction: new THREE.Vector3(0, -1, 0),
                handedness: handedness
            });
        }
    }

    // Spatial haptic methods
    addSpatialHapticObject(position, options = {}) {
        const id = `haptic_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const hapticObject = {
            id: id,
            position: position.clone(),
            intensity: options.intensity || 0.5,
            radius: options.radius || 1.0,
            type: options.type || 'vibration',
            frequency: options.frequency || 40,
            active: true,
            startTime: Date.now(),
            duration: options.duration || Infinity
        };
        
        this.spatialHaptics.hapticObjects.set(id, hapticObject);
        
        // Remove after duration if not infinite
        if (hapticObject.duration !== Infinity) {
            setTimeout(() => {
                this.removeSpatialHapticObject(id);
            }, hapticObject.duration);
        }
        
        return id;
    }

    removeSpatialHapticObject(id) {
        this.spatialHaptics.hapticObjects.delete(id);
    }

    clearSpatialHapticObjects() {
        this.spatialHaptics.hapticObjects.clear();
    }

    // Utility methods
    easeInQuad(t) {
        return t * t;
    }

    easeOutQuad(t) {
        return t * (2 - t);
    }

    // Configuration methods
    setHapticIntensity(intensity) {
        // Global haptic intensity multiplier
        this.globalIntensity = Math.max(0, Math.min(1, intensity));
    }

    enableHapticType(type, enabled = true) {
        switch (type) {
            case 'vibration':
                this.controllerHaptics.capabilities.set('vibration', enabled);
                break;
            case 'audio':
                this.audioHaptics.isEnabled = enabled;
                break;
            case 'spatial':
                this.spatialHaptics.enabled = enabled;
                break;
        }
    }

    getHapticCapabilities() {
        return {
            vibration: this.controllerHaptics.capabilities.get('vibration') || false,
            audioHaptic: this.audioHaptics.isEnabled,
            spatialHaptic: this.spatialHaptics.enabled || false,
            ultrasonic: this.ultrasonicHaptics.isSupported,
            thermal: this.thermalFeedback.isSupported,
            forceFeedback: this.forceFeedback.isSupported
        };
    }
}

// Export Haptic Feedback class
export default HapticFeedback;