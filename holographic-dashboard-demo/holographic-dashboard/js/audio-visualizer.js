// Audio Visualizer for Holographic Dashboard

class AudioVisualizer {
    constructor(scene) {
        this.scene = scene;
        this.analyser = null;
        this.dataArray = null;
        this.visualizer = null;
        this.enabled = false;
        this.audioContext = null;
        this.source = null;
        
        this.bars = [];
        this.barGroup = null;
        
        this.init();
    }

    async init() {
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Request microphone access for demo
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.source = this.audioContext.createMediaStreamSource(stream);
            
            // Create analyser
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = CONFIG.audioVisualizer.fftSize;
            this.analyser.smoothingTimeConstant = CONFIG.audioVisualizer.smoothingTimeConstant;
            
            // Connect source to analyser
            this.source.connect(this.analyser);
            
            // Create data array
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            
            // Create visualizer
            this.createVisualizer();
            
            console.log('Audio visualizer initialized');
        } catch (error) {
            console.warn('Audio visualizer initialization failed:', error);
            // Create demo visualizer without real audio
            this.createDemoVisualizer();
        }
    }

    createVisualizer() {
        const { barCount, barWidth, barSpacing, maxHeight } = CONFIG.audioVisualizer;
        
        this.barGroup = new THREE.Group();
        
        // Create bars
        const totalWidth = (barWidth + barSpacing) * barCount - barSpacing;
        const startX = -totalWidth / 2;
        
        for (let i = 0; i < barCount; i++) {
            // Create bar geometry
            const geometry = new THREE.BoxGeometry(barWidth, 1, barWidth);
            
            // Create gradient material
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    time: { value: 0 },
                    height: { value: 0 },
                    baseColor: { value: new THREE.Color(CONFIG.audioVisualizer.colorStart) },
                    topColor: { value: new THREE.Color(CONFIG.audioVisualizer.colorEnd) }
                },
                vertexShader: `
                    uniform float height;
                    
                    varying vec3 vPosition;
                    varying float vHeight;
                    
                    void main() {
                        vPosition = position;
                        vHeight = height;
                        
                        vec3 pos = position;
                        pos.y *= height;
                        
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform vec3 baseColor;
                    uniform vec3 topColor;
                    uniform float time;
                    
                    varying vec3 vPosition;
                    varying float vHeight;
                    
                    void main() {
                        float gradient = (vPosition.y + 0.5) / vHeight;
                        vec3 color = mix(baseColor, topColor, gradient);
                        
                        // Add pulsing glow
                        float glow = 0.5 + sin(time * 5.0 + vPosition.y) * 0.5;
                        color += glow * 0.2;
                        
                        gl_FragColor = vec4(color, 0.8);
                    }
                `,
                transparent: true,
                blending: THREE.AdditiveBlending
            });
            
            const bar = new THREE.Mesh(geometry, material);
            bar.position.x = startX + i * (barWidth + barSpacing);
            bar.position.y = 0;
            
            // Add glow plane behind bar
            const glowGeometry = new THREE.PlaneGeometry(barWidth * 2, maxHeight);
            const glowMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    glowColor: { value: new THREE.Color(CONFIG.audioVisualizer.colorStart) },
                    intensity: { value: 0 }
                },
                vertexShader: `
                    varying vec2 vUv;
                    
                    void main() {
                        vUv = uv;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform vec3 glowColor;
                    uniform float intensity;
                    
                    varying vec2 vUv;
                    
                    void main() {
                        float dist = distance(vUv, vec2(0.5, 0.0));
                        float glow = 1.0 - smoothstep(0.0, 0.5, dist);
                        
                        gl_FragColor = vec4(glowColor, glow * intensity * 0.5);
                    }
                `,
                transparent: true,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            });
            
            const glowPlane = new THREE.Mesh(glowGeometry, glowMaterial);
            glowPlane.position.z = -1;
            bar.add(glowPlane);
            
            this.barGroup.add(bar);
            this.bars.push({
                mesh: bar,
                material: material,
                glowMaterial: glowMaterial,
                targetHeight: 0,
                currentHeight: 0,
                velocity: 0
            });
        }
        
        // Position the visualizer
        this.barGroup.position.y = -5;
        this.barGroup.position.z = -30;
        
        // Add reflection
        this.createReflection();
        
        this.scene.add(this.barGroup);
    }

    createDemoVisualizer() {
        // Create visualizer without real audio input
        this.createVisualizer();
        
        // Simulate audio data
        this.dataArray = new Uint8Array(CONFIG.audioVisualizer.barCount);
        this.demoMode = true;
        this.demoTime = 0;
    }

    createReflection() {
        // Create a mirrored copy for reflection effect
        const reflectionGroup = new THREE.Group();
        
        this.bars.forEach((barData, index) => {
            const geometry = barData.mesh.geometry.clone();
            const material = barData.material.clone();
            
            material.uniforms.baseColor.value = new THREE.Color(CONFIG.audioVisualizer.colorStart).multiplyScalar(0.3);
            material.uniforms.topColor.value = new THREE.Color(CONFIG.audioVisualizer.colorEnd).multiplyScalar(0.3);
            material.transparent = true;
            material.opacity = 0.3;
            
            const reflection = new THREE.Mesh(geometry, material);
            reflection.position.copy(barData.mesh.position);
            reflection.position.y = -2;
            reflection.scale.y = -1;
            
            reflectionGroup.add(reflection);
            
            // Store reference
            barData.reflection = reflection;
            barData.reflectionMaterial = material;
        });
        
        this.barGroup.add(reflectionGroup);
    }

    update(deltaTime) {
        if (!this.enabled || !this.bars.length) return;
        
        const time = performance.now() * 0.001;
        
        // Get audio data
        if (this.demoMode) {
            // Generate demo data
            this.demoTime += deltaTime;
            for (let i = 0; i < this.dataArray.length; i++) {
                const freq = i / this.dataArray.length;
                const bass = Math.sin(this.demoTime * 2) * 0.5 + 0.5;
                const mid = Math.sin(this.demoTime * 3 + Math.PI / 3) * 0.5 + 0.5;
                const treble = Math.sin(this.demoTime * 5 + Math.PI / 2) * 0.5 + 0.5;
                
                let value = 0;
                if (freq < 0.3) value = bass;
                else if (freq < 0.7) value = mid;
                else value = treble;
                
                // Add some randomness
                value += Math.random() * 0.2;
                value *= 255;
                
                this.dataArray[i] = Math.min(255, value);
            }
        } else if (this.analyser) {
            this.analyser.getByteFrequencyData(this.dataArray);
        }
        
        // Update bars
        const maxHeight = CONFIG.audioVisualizer.maxHeight;
        const step = Math.floor(this.dataArray.length / this.bars.length);
        
        this.bars.forEach((bar, index) => {
            // Get frequency data for this bar
            let sum = 0;
            const start = index * step;
            const end = start + step;
            
            for (let i = start; i < end; i++) {
                sum += this.dataArray[i] || 0;
            }
            
            const average = sum / step;
            const normalizedHeight = (average / 255) * maxHeight;
            
            // Smooth the height changes
            bar.targetHeight = normalizedHeight;
            bar.velocity += (bar.targetHeight - bar.currentHeight) * 0.3;
            bar.velocity *= 0.8; // Damping
            bar.currentHeight += bar.velocity;
            
            // Ensure minimum height
            bar.currentHeight = Math.max(0.5, bar.currentHeight);
            
            // Update bar
            bar.mesh.scale.y = bar.currentHeight;
            bar.mesh.position.y = bar.currentHeight / 2;
            
            // Update shader uniforms
            bar.material.uniforms.height.value = bar.currentHeight;
            bar.material.uniforms.time.value = time;
            
            // Update glow intensity based on height
            const glowIntensity = bar.currentHeight / maxHeight;
            bar.glowMaterial.uniforms.intensity.value = glowIntensity;
            
            // Update reflection
            if (bar.reflection) {
                bar.reflection.scale.y = -bar.currentHeight;
                bar.reflection.position.y = -bar.currentHeight / 2 - 2;
                bar.reflectionMaterial.uniforms.height.value = bar.currentHeight;
                bar.reflectionMaterial.uniforms.time.value = time;
            }
            
            // Color variation based on frequency
            const hue = index / this.bars.length;
            const color = new THREE.Color().setHSL(hue * 0.3 + 0.5, 1, 0.5);
            bar.material.uniforms.topColor.value.lerp(color, 0.1);
        });
        
        // Rotate the entire visualizer slightly
        this.barGroup.rotation.y = Math.sin(time * 0.2) * 0.1;
    }

    setEnabled(enabled) {
        this.enabled = enabled;
        if (this.barGroup) {
            this.barGroup.visible = enabled;
        }
        
        // Resume/suspend audio context
        if (this.audioContext) {
            if (enabled && this.audioContext.state === 'suspended') {
                this.audioContext.resume();
            } else if (!enabled && this.audioContext.state === 'running') {
                this.audioContext.suspend();
            }
        }
    }

    // Connect to an audio element
    connectToAudioElement(audioElement) {
        if (!this.audioContext) return;
        
        try {
            // Disconnect previous source if exists
            if (this.source) {
                this.source.disconnect();
            }
            
            // Create source from audio element
            this.source = this.audioContext.createMediaElementSource(audioElement);
            this.source.connect(this.analyser);
            this.source.connect(this.audioContext.destination);
            
            this.demoMode = false;
            console.log('Connected to audio element');
        } catch (error) {
            console.error('Failed to connect to audio element:', error);
        }
    }

    dispose() {
        // Clean up audio context
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        // Remove from scene
        if (this.barGroup) {
            this.scene.remove(this.barGroup);
            
            // Dispose geometries and materials
            this.bars.forEach(bar => {
                bar.mesh.geometry.dispose();
                bar.material.dispose();
                bar.glowMaterial.dispose();
                
                if (bar.reflection) {
                    bar.reflection.geometry.dispose();
                    bar.reflectionMaterial.dispose();
                }
            });
        }
    }
}

// Export for use in other modules
window.AudioVisualizer = AudioVisualizer;