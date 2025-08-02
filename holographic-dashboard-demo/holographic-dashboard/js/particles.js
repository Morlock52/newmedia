// Particle System for Holographic Effects

class ParticleSystem {
    constructor(scene, config) {
        this.scene = scene;
        this.config = config;
        this.particles = null;
        this.particleGeometry = null;
        this.particleMaterial = null;
        this.time = 0;
        
        this.init();
    }

    init() {
        const { count, size, sizeVariation, color, opacity, blending, spread } = this.config;
        
        // Create geometry
        this.particleGeometry = new THREE.BufferGeometry();
        
        // Generate particle positions
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        const sizes = new Float32Array(count);
        const velocities = new Float32Array(count * 3);
        const lifetimes = new Float32Array(count);
        
        const particleColor = new THREE.Color(color);
        
        for (let i = 0; i < count; i++) {
            const i3 = i * 3;
            
            // Random position within spread
            positions[i3] = (Math.random() - 0.5) * spread;
            positions[i3 + 1] = Math.random() * spread;
            positions[i3 + 2] = (Math.random() - 0.5) * spread;
            
            // Random color variation
            const colorVariation = 0.8 + Math.random() * 0.4;
            colors[i3] = particleColor.r * colorVariation;
            colors[i3 + 1] = particleColor.g * colorVariation;
            colors[i3 + 2] = particleColor.b * colorVariation;
            
            // Random size
            sizes[i] = size + (Math.random() - 0.5) * sizeVariation;
            
            // Random velocity
            velocities[i3] = (Math.random() - 0.5) * 0.1;
            velocities[i3 + 1] = Math.random() * 0.2;
            velocities[i3 + 2] = (Math.random() - 0.5) * 0.1;
            
            // Random lifetime
            lifetimes[i] = Math.random();
        }
        
        this.particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        this.particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        this.particleGeometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
        this.particleGeometry.setAttribute('lifetime', new THREE.BufferAttribute(lifetimes, 1));
        
        // Create material
        this.particleMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                opacity: { value: opacity },
                pixelRatio: { value: window.devicePixelRatio },
                fadeDistance: { value: spread * 0.8 }
            },
            vertexShader: `
                attribute float size;
                attribute vec3 velocity;
                attribute float lifetime;
                
                uniform float time;
                uniform float pixelRatio;
                
                varying vec3 vColor;
                varying float vLifetime;
                varying float vDistance;
                
                void main() {
                    vColor = color;
                    vLifetime = lifetime;
                    
                    vec3 pos = position;
                    
                    // Animate position based on velocity and time
                    float animTime = mod(time + lifetime * 10.0, 10.0);
                    pos += velocity * animTime;
                    
                    // Wrap around
                    pos.y = mod(pos.y, 50.0);
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    vDistance = -mvPosition.z;
                    
                    gl_PointSize = size * pixelRatio * (200.0 / vDistance);
                    gl_Position = projectionMatrix * mvPosition;
                }
            `,
            fragmentShader: `
                uniform float opacity;
                uniform float fadeDistance;
                
                varying vec3 vColor;
                varying float vLifetime;
                varying float vDistance;
                
                void main() {
                    // Create circular particle
                    vec2 center = gl_PointCoord - vec2(0.5);
                    float dist = length(center);
                    if (dist > 0.5) discard;
                    
                    // Soft edges
                    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
                    
                    // Fade based on distance
                    float fade = 1.0 - smoothstep(fadeDistance * 0.5, fadeDistance, vDistance);
                    
                    // Pulsing effect based on lifetime
                    float pulse = 0.5 + sin(vLifetime * 6.28318) * 0.5;
                    
                    alpha *= opacity * fade * pulse;
                    
                    gl_FragColor = vec4(vColor, alpha);
                }
            `,
            transparent: true,
            blending: blending,
            depthWrite: false,
            vertexColors: true
        });
        
        // Create particle system
        this.particles = new THREE.Points(this.particleGeometry, this.particleMaterial);
        this.scene.add(this.particles);
    }

    update(deltaTime) {
        this.time += deltaTime;
        this.particleMaterial.uniforms.time.value = this.time;
        
        // Rotate particle system slowly
        this.particles.rotation.y += deltaTime * 0.05;
    }

    setOpacity(opacity) {
        this.particleMaterial.uniforms.opacity.value = opacity;
    }

    dispose() {
        this.scene.remove(this.particles);
        this.particleGeometry.dispose();
        this.particleMaterial.dispose();
    }
}

// Data Stream Particles - represents data flow
class DataStreamParticles {
    constructor(scene) {
        this.scene = scene;
        this.streams = [];
        this.maxStreams = 10;
        
        this.init();
    }

    init() {
        // Create particle texture
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        
        const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
        gradient.addColorStop(0, 'rgba(0, 255, 255, 1)');
        gradient.addColorStop(0.5, 'rgba(0, 255, 255, 0.5)');
        gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 32, 32);
        
        this.particleTexture = new THREE.CanvasTexture(canvas);
    }

    createStream(startPos, endPos, color = 0x00FFFF, particleCount = 20) {
        const geometry = new THREE.BufferGeometry();
        
        const positions = new Float32Array(particleCount * 3);
        const delays = new Float32Array(particleCount);
        
        for (let i = 0; i < particleCount; i++) {
            const i3 = i * 3;
            positions[i3] = startPos.x;
            positions[i3 + 1] = startPos.y;
            positions[i3 + 2] = startPos.z;
            
            delays[i] = i / particleCount;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('delay', new THREE.BufferAttribute(delays, 1));
        
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                startPos: { value: startPos },
                endPos: { value: endPos },
                color: { value: new THREE.Color(color) },
                particleTexture: { value: this.particleTexture },
                duration: { value: 2.0 }
            },
            vertexShader: `
                attribute float delay;
                
                uniform float time;
                uniform vec3 startPos;
                uniform vec3 endPos;
                uniform float duration;
                
                varying float vAlpha;
                
                void main() {
                    float progress = mod(time + delay, duration) / duration;
                    
                    // Ease in-out
                    progress = smoothstep(0.0, 1.0, progress);
                    
                    // Interpolate position
                    vec3 pos = mix(startPos, endPos, progress);
                    
                    // Add some wave motion
                    pos.x += sin(progress * 3.14159 * 2.0) * 2.0;
                    pos.y += sin(progress * 3.14159 * 4.0) * 1.0;
                    
                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    
                    gl_PointSize = 20.0 * (300.0 / -mvPosition.z);
                    gl_Position = projectionMatrix * mvPosition;
                    
                    // Fade in and out
                    vAlpha = sin(progress * 3.14159);
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                uniform sampler2D particleTexture;
                
                varying float vAlpha;
                
                void main() {
                    vec4 texColor = texture2D(particleTexture, gl_PointCoord);
                    gl_FragColor = vec4(color, texColor.a * vAlpha);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
        
        const points = new THREE.Points(geometry, material);
        
        const stream = {
            points,
            material,
            startTime: Date.now(),
            duration: 2000
        };
        
        this.streams.push(stream);
        this.scene.add(points);
        
        // Remove old streams
        if (this.streams.length > this.maxStreams) {
            const oldStream = this.streams.shift();
            this.scene.remove(oldStream.points);
            oldStream.points.geometry.dispose();
            oldStream.material.dispose();
        }
        
        return stream;
    }

    update(deltaTime) {
        const currentTime = Date.now();
        
        // Update all streams
        this.streams.forEach(stream => {
            stream.material.uniforms.time.value = (currentTime - stream.startTime) / 1000;
        });
        
        // Remove finished streams
        this.streams = this.streams.filter(stream => {
            if (currentTime - stream.startTime > stream.duration * 2) {
                this.scene.remove(stream.points);
                stream.points.geometry.dispose();
                stream.material.dispose();
                return false;
            }
            return true;
        });
    }

    dispose() {
        this.streams.forEach(stream => {
            this.scene.remove(stream.points);
            stream.points.geometry.dispose();
            stream.material.dispose();
        });
        this.streams = [];
        this.particleTexture.dispose();
    }
}

// Export for use in other modules
window.ParticleSystem = ParticleSystem;
window.DataStreamParticles = DataStreamParticles;