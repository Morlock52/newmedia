// Configuration for Holographic Media Dashboard

// Wait for THREE to be available
if (typeof window !== 'undefined') {
    window.CONFIG = window.CONFIG || {};
}

const CONFIG = {
    // WebSocket Configuration
    websocket: {
        url: 'ws://localhost:9998',
        reconnectInterval: 5000,
        maxReconnectAttempts: 10
    },

    // Three.js Scene Configuration
    scene: {
        backgroundColor: 0x0A0A14,
        fogColor: 0x0A0A14,
        fogNear: 100,
        fogFar: 1000,
        ambientLightColor: 0x404040,
        ambientLightIntensity: 0.5
    },

    // Camera Configuration
    camera: {
        fov: 75,
        near: 0.1,
        far: 2000,
        position: { x: 0, y: 10, z: 50 },
        lookAt: { x: 0, y: 0, z: 0 },
        moveSpeed: 0.5,
        rotateSpeed: 0.002
    },

    // Holographic Effects
    holographic: {
        glowIntensity: 1.0,
        scanlineSpeed: 0.001,
        glitchIntensity: 0.3,
        particleCount: 1000,
        particleSize: 2,
        particleSpeed: 0.5
    },

    // Media Cards Configuration
    mediaCards: {
        rows: 3,
        columns: 4,
        spacing: 15,
        cardWidth: 10,
        cardHeight: 6,
        cardDepth: 0.5,
        hoverScale: 1.1,
        hoverHeight: 2,
        animationSpeed: 0.02
    },

    // Audio Visualizer
    audioVisualizer: {
        enabled: true,
        fftSize: 2048,
        smoothingTimeConstant: 0.8,
        barCount: 64,
        barWidth: 1,
        barSpacing: 0.5,
        maxHeight: 20,
        colorStart: 0x00FFFF,
        colorEnd: 0xFF00FF
    },

    // Particle System
    particles: {
        count: 2000,
        size: 1,
        sizeVariation: 0.5,
        speed: 0.2,
        speedVariation: 0.1,
        color: 0x00FFFF,
        opacity: 0.6,
        blending: typeof THREE !== 'undefined' ? THREE.AdditiveBlending : 1,
        spread: 100
    },

    // Post-processing Effects
    postProcessing: {
        bloom: {
            enabled: true,
            strength: 1.5,
            radius: 0.4,
            threshold: 0.85
        },
        filmGrain: {
            enabled: true,
            intensity: 0.35,
            speed: 0.5
        },
        chromatic: {
            enabled: true,
            offset: 0.002
        },
        vignette: {
            enabled: true,
            darkness: 0.5,
            offset: 1.0
        }
    },

    // UI Configuration
    ui: {
        animationDuration: 300,
        updateInterval: 1000,
        statsUpdateInterval: 2000,
        activityFeedLimit: 20,
        notificationDuration: 3000
    },

    // Performance Settings
    performance: {
        shadowsEnabled: true,
        antialias: true,
        pixelRatio: window.devicePixelRatio > 2 ? 2 : window.devicePixelRatio,
        maxFPS: 60,
        adaptiveQuality: true,
        qualityLevels: {
            low: {
                particleCount: 500,
                shadowMapSize: 512,
                bloomStrength: 0.5
            },
            medium: {
                particleCount: 1000,
                shadowMapSize: 1024,
                bloomStrength: 1.0
            },
            high: {
                particleCount: 2000,
                shadowMapSize: 2048,
                bloomStrength: 1.5
            }
        }
    },

    // Media Types
    mediaTypes: {
        movie: { icon: '🎬', color: 0x00FFFF },
        series: { icon: '📺', color: 0xFF00FF },
        music: { icon: '🎵', color: 0xFFFF00 },
        live: { icon: '📡', color: 0x0FF1CE },
        documentary: { icon: '📹', color: 0xFF10F0 }
    },

    // Debug Settings
    debug: {
        enabled: false,
        showStats: false,
        showWireframe: false,
        logWebSocket: false,
        logPerformance: false
    }
};

// Quality preset helper
CONFIG.setQuality = function(level) {
    const preset = this.performance.qualityLevels[level];
    if (preset) {
        this.particles.count = preset.particleCount;
        this.postProcessing.bloom.strength = preset.bloomStrength;
        // Update shadow map size if renderer exists
        if (window.renderer) {
            window.renderer.shadowMap.size = preset.shadowMapSize;
        }
    }
};

// Dynamic configuration based on device capabilities
CONFIG.autoDetectQuality = function() {
    const gpu = this.detectGPU();
    const memory = navigator.deviceMemory || 4;
    
    if (gpu.tier >= 3 && memory >= 8) {
        this.setQuality('high');
    } else if (gpu.tier >= 2 && memory >= 4) {
        this.setQuality('medium');
    } else {
        this.setQuality('low');
    }
};

// Simple GPU detection
CONFIG.detectGPU = function() {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!gl) {
        return { tier: 1, renderer: 'unknown' };
    }
    
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = debugInfo ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'unknown';
    
    // Simple tier system based on renderer string
    let tier = 2; // Default to medium
    if (renderer.includes('NVIDIA') || renderer.includes('AMD')) {
        tier = 3;
    } else if (renderer.includes('Intel')) {
        tier = 1;
    }
    
    return { tier, renderer };
};

// Export for use in other modules
window.CONFIG = CONFIG;