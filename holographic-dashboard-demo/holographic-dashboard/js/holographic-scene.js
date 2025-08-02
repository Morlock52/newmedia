// Main Holographic Scene Manager

class HolographicScene {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.composer = null;
        this.controls = null;
        this.clock = new THREE.Clock();
        
        this.particleSystem = null;
        this.dataStreams = null;
        this.gridFloor = null;
        this.ambientParticles = null;
        
        this.mousePosition = new THREE.Vector2();
        this.raycaster = new THREE.Raycaster();
        
        this.init();
        this.animate();
    }

    init() {
        // Initialize scene
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.Fog(CONFIG.scene.fogColor, CONFIG.scene.fogNear, CONFIG.scene.fogFar);

        // Initialize camera
        const aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(CONFIG.camera.fov, aspect, CONFIG.camera.near, CONFIG.camera.far);
        this.camera.position.set(
            CONFIG.camera.position.x,
            CONFIG.camera.position.y,
            CONFIG.camera.position.z
        );
        this.camera.lookAt(CONFIG.camera.lookAt.x, CONFIG.camera.lookAt.y, CONFIG.camera.lookAt.z);

        // Initialize renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: CONFIG.performance.antialias,
            alpha: false
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(CONFIG.performance.pixelRatio);
        this.renderer.setClearColor(CONFIG.scene.backgroundColor);
        this.renderer.shadowMap.enabled = CONFIG.performance.shadowsEnabled;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Initialize controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.rotateSpeed = CONFIG.camera.rotateSpeed;
        this.controls.zoomSpeed = CONFIG.camera.moveSpeed;
        this.controls.minDistance = 20;
        this.controls.maxDistance = 200;
        this.controls.maxPolarAngle = Math.PI * 0.495;

        // Setup post-processing
        this.setupPostProcessing();

        // Setup scene elements
        this.setupLighting();
        this.createGridFloor();
        this.createHolographicEnvironment();
        this.setupParticleSystems();

        // Event listeners
        window.addEventListener('resize', this.onResize.bind(this));
        window.addEventListener('mousemove', this.onMouseMove.bind(this));
    }

    setupPostProcessing() {
        this.composer = new THREE.EffectComposer(this.renderer);
        
        // Render pass
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        // Bloom pass
        if (CONFIG.postProcessing.bloom.enabled) {
            const bloomPass = new THREE.UnrealBloomPass(
                new THREE.Vector2(window.innerWidth, window.innerHeight),
                CONFIG.postProcessing.bloom.strength,
                CONFIG.postProcessing.bloom.radius,
                CONFIG.postProcessing.bloom.threshold
            );
            this.composer.addPass(bloomPass);
        }

        // Custom shader pass for additional effects
        const customShader = {
            uniforms: {
                tDiffuse: { value: null },
                time: { value: 0 },
                scanlineIntensity: { value: 0.05 },
                chromaticOffset: { value: CONFIG.postProcessing.chromatic.enabled ? CONFIG.postProcessing.chromatic.offset : 0 },
                vignetteEnabled: { value: CONFIG.postProcessing.vignette.enabled ? 1.0 : 0.0 },
                vignetteDarkness: { value: CONFIG.postProcessing.vignette.darkness },
                vignetteOffset: { value: CONFIG.postProcessing.vignette.offset }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float time;
                uniform float scanlineIntensity;
                uniform float chromaticOffset;
                uniform float vignetteEnabled;
                uniform float vignetteDarkness;
                uniform float vignetteOffset;
                
                varying vec2 vUv;
                
                ${Shaders.screenEffects.scanlines}
                ${Shaders.screenEffects.vignette}
                ${Shaders.screenEffects.chromatic}
                ${Shaders.screenEffects.noise}
                
                void main() {
                    vec3 color = chromaticAberration(tDiffuse, vUv, chromaticOffset);
                    
                    // Apply scanlines
                    color = scanlines(vUv, color, time);
                    
                    // Apply vignette
                    if (vignetteEnabled > 0.5) {
                        color = vignette(vUv, color, vignetteDarkness, vignetteOffset);
                    }
                    
                    // Apply film grain
                    color = filmGrain(vUv, color, time, 0.03);
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `
        };

        const customPass = new THREE.ShaderPass(customShader);
        customPass.renderToScreen = true;
        this.composer.addPass(customPass);

        this.customPass = customPass;
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(
            CONFIG.scene.ambientLightColor,
            CONFIG.scene.ambientLightIntensity
        );
        this.scene.add(ambientLight);

        // Point lights for holographic glow
        const colors = [0x00FFFF, 0xFF00FF, 0xFFFF00];
        const positions = [
            new THREE.Vector3(-30, 20, -30),
            new THREE.Vector3(30, 20, -30),
            new THREE.Vector3(0, 20, 30)
        ];

        positions.forEach((pos, i) => {
            const light = new THREE.PointLight(colors[i], 0.5, 100);
            light.position.copy(pos);
            light.castShadow = true;
            light.shadow.mapSize.width = 1024;
            light.shadow.mapSize.height = 1024;
            this.scene.add(light);

            // Add light helper in debug mode
            if (CONFIG.debug.enabled) {
                const helper = new THREE.PointLightHelper(light, 5);
                this.scene.add(helper);
            }
        });

        // Directional light for shadows
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight.position.set(10, 50, 10);
        dirLight.castShadow = true;
        dirLight.shadow.camera.left = -50;
        dirLight.shadow.camera.right = 50;
        dirLight.shadow.camera.top = 50;
        dirLight.shadow.camera.bottom = -50;
        dirLight.shadow.mapSize.width = 2048;
        dirLight.shadow.mapSize.height = 2048;
        this.scene.add(dirLight);
    }

    createGridFloor() {
        const gridSize = 200;
        const divisions = 40;

        // Create custom grid shader
        const gridGeometry = new THREE.PlaneGeometry(gridSize, gridSize, divisions, divisions);
        const gridMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                gridColor: { value: new THREE.Color(0x00FFFF) },
                gridSize: { value: 5 },
                lineWidth: { value: 0.02 },
                fadeDistance: { value: 100 }
            },
            vertexShader: Shaders.gridFloor.vertexShader,
            fragmentShader: Shaders.gridFloor.fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });

        this.gridFloor = new THREE.Mesh(gridGeometry, gridMaterial);
        this.gridFloor.rotation.x = -Math.PI / 2;
        this.gridFloor.position.y = -10;
        this.scene.add(this.gridFloor);
    }

    createHolographicEnvironment() {
        // Create floating holographic panels
        const panelGeometry = new THREE.PlaneGeometry(20, 15);
        const panelMaterial = Shaders.createMaterial('holographic', {
            color1: { value: new THREE.Color(0x00FFFF) },
            color2: { value: new THREE.Color(0xFF00FF) },
            color3: { value: new THREE.Color(0xFFFF00) },
            scanlineSpeed: { value: CONFIG.holographic.scanlineSpeed },
            glowIntensity: { value: CONFIG.holographic.glowIntensity },
            hologramAlpha: { value: 0.3 }
        });

        // Create multiple floating panels
        const panelPositions = [
            { x: -40, y: 10, z: -20, ry: 0.3 },
            { x: 40, y: 10, z: -20, ry: -0.3 },
            { x: 0, y: 10, z: -40, ry: 0 }
        ];

        panelPositions.forEach(pos => {
            const panel = new THREE.Mesh(panelGeometry, panelMaterial.clone());
            panel.position.set(pos.x, pos.y, pos.z);
            panel.rotation.y = pos.ry;
            this.scene.add(panel);
        });

        // Create holographic sphere in center
        const sphereGeometry = new THREE.SphereGeometry(8, 32, 32);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x00FFFF,
            emissive: 0x00FFFF,
            emissiveIntensity: 0.2,
            transparent: true,
            opacity: 0.3,
            wireframe: true
        });

        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.y = 10;
        this.scene.add(sphere);

        // Animate sphere
        this.animatedObjects = this.animatedObjects || [];
        this.animatedObjects.push({
            mesh: sphere,
            update: (time) => {
                sphere.rotation.y = time * 0.1;
                sphere.rotation.x = Math.sin(time * 0.2) * 0.1;
            }
        });
    }

    setupParticleSystems() {
        // Main particle system
        this.particleSystem = new ParticleSystem(this.scene, CONFIG.particles);

        // Data stream particles
        this.dataStreams = new DataStreamParticles(this.scene);

        // Create initial data streams
        setInterval(() => {
            if (Math.random() > 0.5) {
                const start = new THREE.Vector3(
                    (Math.random() - 0.5) * 80,
                    Math.random() * 30,
                    (Math.random() - 0.5) * 80
                );
                const end = new THREE.Vector3(
                    (Math.random() - 0.5) * 80,
                    Math.random() * 30,
                    (Math.random() - 0.5) * 80
                );
                
                const colors = [0x00FFFF, 0xFF00FF, 0xFFFF00, 0x0FF1CE];
                const color = colors[Math.floor(Math.random() * colors.length)];
                
                this.dataStreams.createStream(start, end, color);
            }
        }, 2000);
    }

    onResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
        this.composer.setSize(width, height);
    }

    onMouseMove(event) {
        this.mousePosition.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mousePosition.y = -(event.clientY / window.innerHeight) * 2 + 1;
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));

        const deltaTime = this.clock.getDelta();
        const elapsedTime = this.clock.getElapsedTime();

        // Update controls
        this.controls.update();

        // Update particle systems
        if (this.particleSystem) {
            this.particleSystem.update(deltaTime);
        }

        if (this.dataStreams) {
            this.dataStreams.update(deltaTime);
        }

        // Update animated objects
        if (this.animatedObjects) {
            this.animatedObjects.forEach(obj => {
                if (obj.update) obj.update(elapsedTime);
            });
        }

        // Update shader uniforms
        if (this.gridFloor) {
            this.gridFloor.material.uniforms.time.value = elapsedTime;
        }

        if (this.customPass) {
            this.customPass.uniforms.time.value = elapsedTime;
        }

        // Update materials with time uniform
        this.scene.traverse((child) => {
            if (child.material && child.material.uniforms && child.material.uniforms.time) {
                child.material.uniforms.time.value = elapsedTime;
            }
        });

        // Render
        this.composer.render();

        // Performance monitoring
        if (CONFIG.debug.logPerformance) {
            const info = this.renderer.info;
            console.log('Render info:', info);
        }
    }

    // Public methods
    setEffectsEnabled(enabled) {
        this.composer.passes.forEach((pass, index) => {
            if (index > 0) { // Skip render pass
                pass.enabled = enabled;
            }
        });
    }

    setParticlesEnabled(enabled) {
        if (this.particleSystem) {
            this.particleSystem.particles.visible = enabled;
        }
    }

    dispose() {
        // Clean up resources
        if (this.particleSystem) {
            this.particleSystem.dispose();
        }

        if (this.dataStreams) {
            this.dataStreams.dispose();
        }

        this.scene.traverse((child) => {
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });

        this.renderer.dispose();
        this.controls.dispose();
    }
}

// Export for use in other modules
window.HolographicScene = HolographicScene;