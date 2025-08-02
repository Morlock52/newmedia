// WebGL Fallback System - Provides 2D Canvas fallback when WebGL is unavailable
// Ensures the dashboard works on older devices and browsers

class WebGLFallbackManager {
    constructor() {
        this.webglSupported = this.checkWebGLSupport();
        this.fallbackActive = false;
        this.canvas2d = null;
        this.ctx2d = null;
        this.animationId = null;
        this.particles = [];
        this.time = 0;
        
        console.log('WebGL supported:', this.webglSupported);
        
        if (!this.webglSupported) {
            console.warn('WebGL not supported, initializing 2D fallback');
            this.initializeFallback();
        }
    }

    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            
            if (!gl) {
                return false;
            }

            // Check for required extensions
            const requiredExtensions = [
                'OES_texture_float',
                'WEBGL_depth_texture'
            ];

            for (const ext of requiredExtensions) {
                if (!gl.getExtension(ext)) {
                    console.warn(`Missing WebGL extension: ${ext}`);
                }
            }

            // Test basic WebGL functionality
            const vertexShader = gl.createShader(gl.VERTEX_SHADER);
            gl.shaderSource(vertexShader, `
                attribute vec2 position;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            `);
            gl.compileShader(vertexShader);

            if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
                console.error('WebGL shader compilation failed');
                return false;
            }

            gl.deleteShader(vertexShader);
            return true;

        } catch (error) {
            console.error('WebGL support check failed:', error);
            return false;
        }
    }

    initializeFallback() {
        this.fallbackActive = true;
        this.setupFallbackUI();
        this.create2DRenderer();
        this.setupParticleSystem();
        this.startFallbackAnimation();
    }

    setupFallbackUI() {
        // Show fallback notification
        const notification = document.createElement('div');
        notification.className = 'webgl-fallback-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">ℹ️</span>
                <span class="notification-text">Running in compatibility mode</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 8px;
            padding: 12px;
            color: #00FFFF;
            font-family: 'Orbitron', monospace;
            font-size: 12px;
            z-index: 10000;
            backdrop-filter: blur(10px);
            animation: slideInRight 0.3s ease-out;
        `;

        document.body.appendChild(notification);

        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.animation = 'slideOutRight 0.3s ease-out';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);

        // Add necessary CSS animations
        this.addFallbackStyles();
    }

    addFallbackStyles() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
            
            .notification-content {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .notification-close {
                background: none;
                border: none;
                color: #00FFFF;
                cursor: pointer;
                font-size: 16px;
                margin-left: auto;
            }
            
            .notification-close:hover {
                color: #ffffff;
            }
            
            .fallback-particle {
                position: absolute;
                background: radial-gradient(circle, rgba(0,255,255,0.8) 0%, rgba(0,255,255,0) 70%);
                border-radius: 50%;
                pointer-events: none;
            }
            
            .fallback-canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
            }
        `;
        document.head.appendChild(style);
    }

    create2DRenderer() {
        const container = document.getElementById('webgl-container');
        if (!container) {
            console.error('WebGL container not found for fallback');
            return;
        }

        // Create 2D canvas
        this.canvas2d = document.createElement('canvas');
        this.canvas2d.className = 'fallback-canvas';
        this.ctx2d = this.canvas2d.getContext('2d');
        
        container.appendChild(this.canvas2d);
        
        // Setup canvas size
        this.resizeFallbackCanvas();
        
        // Handle resize
        window.addEventListener('resize', () => {
            this.resizeFallbackCanvas();
        });
    }

    resizeFallbackCanvas() {
        if (!this.canvas2d) return;
        
        const rect = this.canvas2d.parentElement.getBoundingClientRect();
        this.canvas2d.width = rect.width;
        this.canvas2d.height = rect.height;
        this.canvas2d.style.width = rect.width + 'px';
        this.canvas2d.style.height = rect.height + 'px';
    }

    setupParticleSystem() {
        const particleCount = 50; // Reduced for performance
        
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * (this.canvas2d?.width || window.innerWidth),
                y: Math.random() * (this.canvas2d?.height || window.innerHeight),
                z: Math.random() * 100,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                vz: (Math.random() - 0.5) * 0.2,
                size: Math.random() * 3 + 1,
                opacity: Math.random() * 0.8 + 0.2,
                color: this.getRandomColor()
            });
        }
    }

    getRandomColor() {
        const colors = [
            'rgba(0, 255, 255, ',    // Cyan
            'rgba(255, 0, 255, ',    // Magenta
            'rgba(0, 255, 0, ',      // Green
            'rgba(255, 255, 0, ',    // Yellow
            'rgba(255, 255, 255, '   // White
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    startFallbackAnimation() {
        if (!this.canvas2d || !this.ctx2d) return;
        
        const animate = () => {
            this.time += 0.016; // ~60fps
            
            // Clear canvas
            this.ctx2d.clearRect(0, 0, this.canvas2d.width, this.canvas2d.height);
            
            // Create background gradient
            const gradient = this.ctx2d.createRadialGradient(
                this.canvas2d.width / 2, this.canvas2d.height / 2, 0,
                this.canvas2d.width / 2, this.canvas2d.height / 2, Math.min(this.canvas2d.width, this.canvas2d.height) / 2
            );
            gradient.addColorStop(0, 'rgba(0, 20, 40, 0.8)');
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0.9)');
            
            this.ctx2d.fillStyle = gradient;
            this.ctx2d.fillRect(0, 0, this.canvas2d.width, this.canvas2d.height);
            
            // Update and draw particles
            this.updateParticles();
            this.drawParticles();
            
            // Draw connecting lines between nearby particles
            this.drawConnections();
            
            // Add some holographic effect
            this.drawHolographicGrid();
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }

    updateParticles() {
        const width = this.canvas2d.width;
        const height = this.canvas2d.height;
        
        this.particles.forEach(particle => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.z += particle.vz;
            
            // Wave motion
            particle.x += Math.sin(this.time + particle.z * 0.01) * 0.2;
            particle.y += Math.cos(this.time + particle.z * 0.01) * 0.2;
            
            // Wrap around edges
            if (particle.x < 0) particle.x = width;
            if (particle.x > width) particle.x = 0;
            if (particle.y < 0) particle.y = height;
            if (particle.y > height) particle.y = 0;
            if (particle.z < 0) particle.z = 100;
            if (particle.z > 100) particle.z = 0;
            
            // Pulsing opacity
            particle.opacity = 0.3 + Math.sin(this.time * 2 + particle.z * 0.1) * 0.3;
        });
    }

    drawParticles() {
        this.particles.forEach(particle => {
            const scale = 1 - particle.z / 100;
            const size = particle.size * scale;
            const opacity = particle.opacity * scale;
            
            // Create particle glow effect
            const gradient = this.ctx2d.createRadialGradient(
                particle.x, particle.y, 0,
                particle.x, particle.y, size * 3
            );
            gradient.addColorStop(0, particle.color + opacity + ')');
            gradient.addColorStop(0.7, particle.color + (opacity * 0.3) + ')');
            gradient.addColorStop(1, particle.color + '0)');
            
            this.ctx2d.fillStyle = gradient;
            this.ctx2d.beginPath();
            this.ctx2d.arc(particle.x, particle.y, size * 3, 0, Math.PI * 2);
            this.ctx2d.fill();
            
            // Draw core particle
            this.ctx2d.fillStyle = particle.color + opacity + ')';
            this.ctx2d.beginPath();
            this.ctx2d.arc(particle.x, particle.y, size, 0, Math.PI * 2);
            this.ctx2d.fill();
        });
    }

    drawConnections() {
        const maxDistance = 100;
        
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const p1 = this.particles[i];
                const p2 = this.particles[j];
                
                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < maxDistance) {
                    const opacity = (1 - distance / maxDistance) * 0.3;
                    
                    this.ctx2d.strokeStyle = `rgba(0, 255, 255, ${opacity})`;
                    this.ctx2d.lineWidth = 0.5;
                    this.ctx2d.beginPath();
                    this.ctx2d.moveTo(p1.x, p1.y);
                    this.ctx2d.lineTo(p2.x, p2.y);
                    this.ctx2d.stroke();
                }
            }
        }
    }

    drawHolographicGrid() {
        const gridSize = 50;
        const opacity = 0.1 + Math.sin(this.time) * 0.05;
        
        this.ctx2d.strokeStyle = `rgba(0, 255, 255, ${opacity})`;
        this.ctx2d.lineWidth = 0.5;
        
        // Vertical lines
        for (let x = 0; x < this.canvas2d.width; x += gridSize) {
            this.ctx2d.beginPath();
            this.ctx2d.moveTo(x, 0);
            this.ctx2d.lineTo(x, this.canvas2d.height);
            this.ctx2d.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.canvas2d.height; y += gridSize) {
            this.ctx2d.beginPath();
            this.ctx2d.moveTo(0, y);
            this.ctx2d.lineTo(this.canvas2d.width, y);
            this.ctx2d.stroke();
        }
    }

    // Public API
    isWebGLSupported() {
        return this.webglSupported;
    }

    isFallbackActive() {
        return this.fallbackActive;
    }

    forceEnableFallback() {
        if (!this.fallbackActive) {
            console.log('Force enabling WebGL fallback');
            this.initializeFallback();
        }
    }

    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        if (this.canvas2d && this.canvas2d.parentElement) {
            this.canvas2d.parentElement.removeChild(this.canvas2d);
        }
        
        this.particles = [];
        this.canvas2d = null;
        this.ctx2d = null;
    }

    // Integration with existing Three.js scene
    attachToScene(scene) {
        if (this.fallbackActive) {
            console.log('WebGL fallback active, Three.js scene disabled');
            
            // Hide Three.js elements
            const webglContainer = document.getElementById('webgl-container');
            if (webglContainer) {
                const threeCanvas = webglContainer.querySelector('canvas:not(.fallback-canvas)');
                if (threeCanvas) {
                    threeCanvas.style.display = 'none';
                }
            }
        }
    }

    // Handle media card interactions in fallback mode
    handleMediaCardClick(cardData) {
        if (this.fallbackActive) {
            // Simple 2D animation for media card selection
            this.animateCardSelection(cardData);
        }
    }

    animateCardSelection(cardData) {
        // Create a simple pulse effect for selected card
        const pulse = document.createElement('div');
        pulse.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(0,255,255,0.6) 0%, transparent 70%);
            border-radius: 50%;
            transform: translate(-50%, -50%) scale(0);
            animation: pulseEffect 0.8s ease-out;
            pointer-events: none;
            z-index: 9999;
        `;
        
        document.body.appendChild(pulse);
        
        setTimeout(() => {
            pulse.remove();
        }, 800);
        
        // Add pulse animation if not exists
        if (!document.querySelector('#pulseEffectStyle')) {
            const style = document.createElement('style');
            style.id = 'pulseEffectStyle';
            style.textContent = `
                @keyframes pulseEffect {
                    0% { transform: translate(-50%, -50%) scale(0); opacity: 1; }
                    100% { transform: translate(-50%, -50%) scale(3); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }
    }
}

// Initialize WebGL fallback manager
document.addEventListener('DOMContentLoaded', () => {
    window.webglFallback = new WebGLFallbackManager();
    
    // Integrate with existing dashboard
    if (window.dashboard) {
        window.webglFallback.attachToScene(window.dashboard.scene);
    }
    
    console.log('WebGL Fallback Manager initialized');
});

// Export for use in other modules
window.WebGLFallbackManager = WebGLFallbackManager;