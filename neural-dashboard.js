// Neural Dashboard - Advanced Interactions and Visualizations

class NeuralDashboard {
    constructor() {
        this.init();
        this.setupEventListeners();
        this.startAnimations();
    }

    init() {
        // Initialize particle field
        this.initParticles();
        
        // Initialize 3D media sphere
        this.initMediaSphere();
        
        // Initialize neural network visualization
        this.initNeuralNetwork();
        
        // Initialize quantum field
        this.initQuantumField();
        
        // Initialize AI avatar
        this.initAIAvatar();
        
        // Initialize collaboration space
        this.initCollaborationSpace();
        
        // Initialize spatial audio
        this.initSpatialAudio();
        
        // Initialize haptic feedback
        this.initHaptics();
    }

    initParticles() {
        particlesJS('particle-field', {
            particles: {
                number: { value: 80, density: { enable: true, value_area: 800 } },
                color: { value: ['#00ffff', '#ff00ff', '#ffff00'] },
                shape: { type: 'circle' },
                opacity: { value: 0.5, random: true, anim: { enable: true, speed: 1, opacity_min: 0.1 } },
                size: { value: 3, random: true, anim: { enable: true, speed: 2, size_min: 0.1 } },
                line_linked: {
                    enable: true,
                    distance: 150,
                    color: '#00ffff',
                    opacity: 0.4,
                    width: 1
                },
                move: {
                    enable: true,
                    speed: 2,
                    direction: 'none',
                    random: true,
                    straight: false,
                    out_mode: 'bounce',
                    attract: { enable: true, rotateX: 600, rotateY: 1200 }
                }
            },
            interactivity: {
                detect_on: 'canvas',
                events: {
                    onhover: { enable: true, mode: 'repulse' },
                    onclick: { enable: true, mode: 'push' },
                    resize: true
                },
                modes: {
                    repulse: { distance: 100, duration: 0.4 },
                    push: { particles_nb: 4 }
                }
            },
            retina_detect: true
        });
    }

    initMediaSphere() {
        const container = document.getElementById('media-sphere-renderer');
        
        // Three.js scene setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        
        this.renderer.setSize(container.offsetWidth, container.offsetHeight);
        container.appendChild(this.renderer.domElement);
        
        // Create media spheres
        const sphereGeometry = new THREE.SphereGeometry(2, 32, 32);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x00ffff,
            emissive: 0x00ffff,
            emissiveIntensity: 0.5,
            wireframe: true
        });
        
        this.mediaSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.scene.add(this.mediaSphere);
        
        // Add inner spheres for media previews
        for (let i = 0; i < 6; i++) {
            const innerSphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.5, 16, 16),
                new THREE.MeshPhongMaterial({
                    color: new THREE.Color().setHSL(i / 6, 1, 0.5),
                    emissive: new THREE.Color().setHSL(i / 6, 1, 0.5),
                    emissiveIntensity: 0.3
                })
            );
            
            const angle = (i / 6) * Math.PI * 2;
            innerSphere.position.x = Math.cos(angle) * 3;
            innerSphere.position.z = Math.sin(angle) * 3;
            innerSphere.position.y = Math.sin(angle) * 0.5;
            
            this.mediaSphere.add(innerSphere);
        }
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        const pointLight = new THREE.PointLight(0x00ffff, 1, 100);
        pointLight.position.set(5, 5, 5);
        this.scene.add(pointLight);
        
        this.camera.position.z = 8;
        
        // Animation loop
        this.animateMediaSphere();
    }

    animateMediaSphere() {
        requestAnimationFrame(() => this.animateMediaSphere());
        
        this.mediaSphere.rotation.x += 0.005;
        this.mediaSphere.rotation.y += 0.01;
        
        // Animate inner spheres
        this.mediaSphere.children.forEach((child, index) => {
            child.rotation.x += 0.02;
            child.rotation.y += 0.03;
            child.position.y = Math.sin(Date.now() * 0.001 + index) * 0.5;
        });
        
        this.renderer.render(this.scene, this.camera);
    }

    initNeuralNetwork() {
        const canvas = document.getElementById('neural-canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        // Neural network nodes
        this.neurons = [];
        const layers = [5, 8, 6, 4];
        
        layers.forEach((count, layerIndex) => {
            for (let i = 0; i < count; i++) {
                this.neurons.push({
                    x: (layerIndex + 1) * (canvas.width / (layers.length + 1)),
                    y: (i + 1) * (canvas.height / (count + 1)),
                    layer: layerIndex,
                    activation: Math.random()
                });
            }
        });
        
        // Animate neural network
        this.animateNeuralNetwork(ctx);
    }

    animateNeuralNetwork(ctx) {
        requestAnimationFrame(() => this.animateNeuralNetwork(ctx));
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        // Update neuron activations
        this.neurons.forEach(neuron => {
            neuron.activation = (neuron.activation + (Math.random() - 0.5) * 0.1) % 1;
            if (neuron.activation < 0) neuron.activation = 1 + neuron.activation;
        });
        
        // Draw connections
        this.neurons.forEach((neuron, i) => {
            this.neurons.forEach((otherNeuron, j) => {
                if (neuron.layer === otherNeuron.layer - 1) {
                    ctx.beginPath();
                    ctx.moveTo(neuron.x, neuron.y);
                    ctx.lineTo(otherNeuron.x, otherNeuron.y);
                    ctx.strokeStyle = `rgba(0, 255, 255, ${neuron.activation * otherNeuron.activation * 0.5})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            });
        });
        
        // Draw neurons
        this.neurons.forEach(neuron => {
            ctx.beginPath();
            ctx.arc(neuron.x, neuron.y, 5 + neuron.activation * 5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 255, 255, ${neuron.activation})`;
            ctx.fill();
            ctx.strokeStyle = '#00ffff';
            ctx.stroke();
        });
    }

    initQuantumField() {
        const canvas = document.getElementById('quantum-canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        // Quantum particles
        this.quantumParticles = [];
        for (let i = 0; i < 50; i++) {
            this.quantumParticles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                entangled: null,
                color: `hsl(${Math.random() * 360}, 100%, 50%)`
            });
        }
        
        this.animateQuantumField(ctx);
    }

    animateQuantumField(ctx) {
        requestAnimationFrame(() => this.animateQuantumField(ctx));
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        // Update and draw quantum particles
        this.quantumParticles.forEach((particle, i) => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Boundary bounce
            if (particle.x < 0 || particle.x > ctx.canvas.width) particle.vx *= -1;
            if (particle.y < 0 || particle.y > ctx.canvas.height) particle.vy *= -1;
            
            // Draw entanglement lines
            if (particle.entangled) {
                ctx.beginPath();
                ctx.moveTo(particle.x, particle.y);
                ctx.lineTo(particle.entangled.x, particle.entangled.y);
                ctx.strokeStyle = 'rgba(255, 0, 255, 0.3)';
                ctx.stroke();
            }
            
            // Draw particle
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = particle.color;
            ctx.fill();
            
            // Quantum blur effect
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, 10, 0, Math.PI * 2);
            const gradient = ctx.createRadialGradient(particle.x, particle.y, 0, particle.x, particle.y, 10);
            gradient.addColorStop(0, particle.color);
            gradient.addColorStop(1, 'transparent');
            ctx.fillStyle = gradient;
            ctx.fill();
        });
    }

    initAIAvatar() {
        const canvas = document.getElementById('ai-face');
        const ctx = canvas.getContext('2d');
        
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        this.animateAIAvatar(ctx);
    }

    animateAIAvatar(ctx) {
        requestAnimationFrame(() => this.animateAIAvatar(ctx));
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        const centerX = ctx.canvas.width / 2;
        const centerY = ctx.canvas.height / 2;
        const time = Date.now() * 0.001;
        
        // Draw AI face circles
        for (let i = 0; i < 5; i++) {
            ctx.beginPath();
            ctx.arc(
                centerX + Math.sin(time + i) * 20,
                centerY + Math.cos(time + i) * 20,
                30 + Math.sin(time * 2 + i) * 10,
                0,
                Math.PI * 2
            );
            ctx.strokeStyle = `hsla(${180 + i * 20}, 100%, 50%, ${0.5 - i * 0.1})`;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        // Draw eyes
        const eyeOffset = 30;
        [1, -1].forEach(side => {
            ctx.beginPath();
            ctx.arc(centerX + eyeOffset * side, centerY - 10, 5 + Math.sin(time * 3) * 2, 0, Math.PI * 2);
            ctx.fillStyle = '#00ffff';
            ctx.fill();
        });
    }

    initCollaborationSpace() {
        const container = document.getElementById('collab-3d-space');
        
        // Create a simple 3D collaboration visualization
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        
        renderer.setSize(container.offsetWidth, container.offsetHeight);
        container.appendChild(renderer.domElement);
        
        // Create collaboration nodes
        const nodeGeometry = new THREE.BoxGeometry(1, 1, 1);
        const nodes = [];
        
        for (let i = 0; i < 5; i++) {
            const node = new THREE.Mesh(
                nodeGeometry,
                new THREE.MeshPhongMaterial({
                    color: new THREE.Color().setHSL(i / 5, 1, 0.5),
                    emissive: new THREE.Color().setHSL(i / 5, 1, 0.5),
                    emissiveIntensity: 0.3
                })
            );
            
            node.position.x = Math.cos(i / 5 * Math.PI * 2) * 3;
            node.position.z = Math.sin(i / 5 * Math.PI * 2) * 3;
            
            scene.add(node);
            nodes.push(node);
        }
        
        const light = new THREE.PointLight(0xffffff, 1, 100);
        light.position.set(0, 5, 0);
        scene.add(light);
        
        camera.position.y = 5;
        camera.position.z = 8;
        camera.lookAt(0, 0, 0);
        
        // Animation
        const animateCollab = () => {
            requestAnimationFrame(animateCollab);
            
            nodes.forEach((node, i) => {
                node.rotation.x += 0.01;
                node.rotation.y += 0.01;
                node.position.y = Math.sin(Date.now() * 0.001 + i) * 0.5;
            });
            
            renderer.render(scene, camera);
        };
        
        animateCollab();
    }

    initSpatialAudio() {
        // Create audio context for spatial audio
        if (window.AudioContext || window.webkitAudioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create oscillator for ambient sounds
            this.oscillator = this.audioContext.createOscillator();
            this.gainNode = this.audioContext.createGain();
            
            this.oscillator.connect(this.gainNode);
            this.gainNode.connect(this.audioContext.destination);
            
            this.oscillator.frequency.value = 440;
            this.gainNode.gain.value = 0.05;
            
            // Modulate frequency for ambient effect
            setInterval(() => {
                this.oscillator.frequency.value = 440 + Math.sin(Date.now() * 0.001) * 100;
            }, 100);
        }
    }

    initHaptics() {
        // Check for haptic feedback support
        if ('vibrate' in navigator) {
            document.querySelectorAll('.gesture-btn, .quantum-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    navigator.vibrate([50, 30, 50]);
                });
            });
        }
    }

    setupEventListeners() {
        // Navigation orbs
        document.querySelectorAll('.nav-orb').forEach(orb => {
            orb.addEventListener('click', (e) => {
                const section = e.currentTarget.dataset.section;
                this.navigateToSection(section);
                
                // Haptic feedback
                if ('vibrate' in navigator) {
                    navigator.vibrate(100);
                }
            });
        });

        // Gesture controls
        document.querySelectorAll('.gesture-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const gesture = e.currentTarget.dataset.gesture;
                this.handleGesture(gesture);
            });
        });

        // Quantum controls
        document.querySelectorAll('.quantum-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (e.target.textContent === 'Entangle') {
                    this.entangleParticles();
                } else if (e.target.textContent === 'Collapse') {
                    this.collapseWaveFunction();
                }
            });
        });

        // AI input
        const aiInput = document.querySelector('.ai-input');
        aiInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processAICommand(e.target.value);
                e.target.value = '';
            }
        });

        // XR toggle
        document.getElementById('xr-toggle').addEventListener('click', () => {
            this.toggleXRMode();
        });

        // Eye tracking simulation
        document.addEventListener('mousemove', (e) => {
            this.simulateEyeTracking(e.clientX, e.clientY);
        });
    }

    navigateToSection(section) {
        console.log(`Navigating to ${section}`);
        // Add navigation logic
    }

    handleGesture(gesture) {
        switch(gesture) {
            case 'rotate':
                gsap.to(this.mediaSphere.rotation, { y: '+=6.28', duration: 2 });
                break;
            case 'expand':
                gsap.to(this.mediaSphere.scale, { x: 1.5, y: 1.5, z: 1.5, duration: 1, yoyo: true, repeat: 1 });
                break;
            case 'neural':
                this.activateNeuralMode();
                break;
        }
    }

    activateNeuralMode() {
        document.body.style.filter = 'hue-rotate(180deg)';
        setTimeout(() => {
            document.body.style.filter = 'none';
        }, 2000);
    }

    entangleParticles() {
        // Randomly entangle quantum particles
        for (let i = 0; i < this.quantumParticles.length; i += 2) {
            if (i + 1 < this.quantumParticles.length) {
                this.quantumParticles[i].entangled = this.quantumParticles[i + 1];
                this.quantumParticles[i + 1].entangled = this.quantumParticles[i];
            }
        }
    }

    collapseWaveFunction() {
        // Clear all entanglements
        this.quantumParticles.forEach(particle => {
            particle.entangled = null;
            particle.vx = (Math.random() - 0.5) * 4;
            particle.vy = (Math.random() - 0.5) * 4;
        });
    }

    processAICommand(command) {
        const output = document.getElementById('ai-output');
        output.innerHTML = `<p>Processing: "${command}"</p>`;
        
        // Simulate AI processing
        setTimeout(() => {
            const responses = [
                'Neural pathways recalibrated for optimal performance.',
                'Quantum entanglement established across all nodes.',
                'Media streams synchronized at 98.7% efficiency.',
                'Initiating holographic projection protocols.',
                'Spatial computing matrix aligned.'
            ];
            
            output.innerHTML = `<p>${responses[Math.floor(Math.random() * responses.length)]}</p>`;
        }, 1000);
    }

    toggleXRMode() {
        if ('xr' in navigator) {
            // Check for WebXR support
            navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
                if (supported) {
                    console.log('VR is supported');
                    // Initialize XR session
                } else {
                    alert('XR not supported on this device');
                }
            });
        } else {
            alert('WebXR not available');
        }
    }

    simulateEyeTracking(x, y) {
        // Update UI elements based on "eye position"
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        
        const deltaX = (x - centerX) / centerX;
        const deltaY = (y - centerY) / centerY;
        
        // Move quantum overlay slightly
        const overlay = document.getElementById('quantum-overlay');
        overlay.style.transform = `translate(${deltaX * 10}px, ${deltaY * 10}px)`;
    }

    startAnimations() {
        // Start all background animations
        this.animateBackgroundEffects();
    }

    animateBackgroundEffects() {
        // Animate status orbs
        document.querySelectorAll('.status-orb').forEach((orb, index) => {
            setInterval(() => {
                orb.classList.toggle('active');
            }, 1000 + index * 333);
        });

        // Animate metrics
        setInterval(() => {
            document.querySelectorAll('.value').forEach(value => {
                if (value.textContent.includes('%')) {
                    value.textContent = Math.floor(Math.random() * 30 + 70) + '%';
                }
            });
        }, 3000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new NeuralDashboard();
    
    // Start audio context on user interaction
    document.addEventListener('click', () => {
        if (dashboard.audioContext && dashboard.audioContext.state === 'suspended') {
            dashboard.audioContext.resume();
            dashboard.oscillator.start();
        }
    }, { once: true });
});

// Handle window resize
window.addEventListener('resize', () => {
    // Update canvas sizes
    const canvases = ['neural-canvas', 'quantum-canvas', 'ai-face'];
    canvases.forEach(id => {
        const canvas = document.getElementById(id);
        if (canvas) {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        }
    });
});

// Performance optimization
const perfObserver = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
        console.log(`Performance: ${entry.name} - ${entry.duration}ms`);
    }
});
perfObserver.observe({ entryTypes: ['measure'] });