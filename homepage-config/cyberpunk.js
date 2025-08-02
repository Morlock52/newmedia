// Cyberpunk Enhanced JavaScript for Homepage Dashboard
// Advanced animations, effects, and interactions

class CyberpunkDashboard {
    constructor() {
        this.init();
        this.setupAnimations();
        this.setupInteractions();
        this.startMatrixRain();
        this.setupTypewriter();
        this.setupAudioVisualizer();
    }

    init() {
        console.log('🚀 Cyberpunk Dashboard initialized');
        document.body.classList.add('cyberpunk-ready');
        this.createParticleSystem();
        this.addGlitchEffects();
    }

    // Matrix Rain Effect
    startMatrixRain() {
        const canvas = document.createElement('canvas');
        canvas.id = 'matrix-rain';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.zIndex = '-10';
        canvas.style.opacity = '0.1';
        document.body.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        const matrix = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789@#$%^&*()*&^%+-/~{[|`]}";
        const matrixArray = matrix.split("");

        const fontSize = 10;
        const columns = canvas.width / fontSize;

        const drops = [];
        for (let x = 0; x < columns; x++) {
            drops[x] = 1;
        }

        const draw = () => {
            ctx.fillStyle = 'rgba(5, 8, 17, 0.04)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.fillStyle = '#00ff9f';
            ctx.font = fontSize + 'px Share Tech Mono';

            for (let i = 0; i < drops.length; i++) {
                const text = matrixArray[Math.floor(Math.random() * matrixArray.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);

                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i]++;
            }
        };

        setInterval(draw, 35);

        // Resize handler
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    }

    // Particle System
    createParticleSystem() {
        const particleContainer = document.createElement('div');
        particleContainer.id = 'particle-system';
        particleContainer.style.position = 'fixed';
        particleContainer.style.top = '0';
        particleContainer.style.left = '0';
        particleContainer.style.width = '100%';
        particleContainer.style.height = '100%';
        particleContainer.style.pointerEvents = 'none';
        particleContainer.style.zIndex = '-5';
        document.body.appendChild(particleContainer);

        for (let i = 0; i < 50; i++) {
            this.createParticle(particleContainer);
        }
    }

    createParticle(container) {
        const particle = document.createElement('div');
        particle.className = 'cyber-particle';
        particle.style.position = 'absolute';
        particle.style.width = Math.random() * 4 + 1 + 'px';
        particle.style.height = particle.style.width;
        particle.style.background = ['#00ff9f', '#ff0080', '#00d4ff', '#bd00ff'][Math.floor(Math.random() * 4)];
        particle.style.borderRadius = '50%';
        particle.style.boxShadow = `0 0 10px ${particle.style.background}`;
        
        // Random starting position
        particle.style.left = Math.random() * window.innerWidth + 'px';
        particle.style.top = Math.random() * window.innerHeight + 'px';

        // Animation
        const duration = Math.random() * 20 + 10;
        particle.style.animation = `float ${duration}s linear infinite`;

        container.appendChild(particle);

        // Remove and recreate after animation
        setTimeout(() => {
            if (particle.parentNode) {
                particle.parentNode.removeChild(particle);
                this.createParticle(container);
            }
        }, duration * 1000);
    }

    // Add glitch effects to headers
    addGlitchEffects() {
        const headers = document.querySelectorAll('h1, h2, h3');
        headers.forEach(header => {
            header.setAttribute('data-text', header.textContent);
            header.classList.add('glitch');
            
            // Random glitch trigger
            setInterval(() => {
                if (Math.random() > 0.95) {
                    header.style.animation = 'none';
                    setTimeout(() => {
                        header.style.animation = 'glitch 0.3s ease-out';
                    }, 10);
                }
            }, 2000);
        });
    }

    // Typewriter effect for text elements
    setupTypewriter() {
        const typewriterElements = document.querySelectorAll('[data-typewriter]');
        typewriterElements.forEach((element, index) => {
            const text = element.textContent;
            element.textContent = '';
            element.style.borderRight = '2px solid #00ff9f';
            
            let i = 0;
            const typeInterval = setInterval(() => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                } else {
                    clearInterval(typeInterval);
                    setTimeout(() => {
                        element.style.borderRight = 'none';
                    }, 1000);
                }
            }, 50 + Math.random() * 50);
        });
    }

    // Enhanced interactions
    setupInteractions() {
        // Service card hover effects
        document.addEventListener('mouseover', (e) => {
            if (e.target.closest('.service, .card, .widget')) {
                const card = e.target.closest('.service, .card, .widget');
                this.addHoverEffect(card);
            }
        });

        // Button interactions
        document.addEventListener('click', (e) => {
            if (e.target.matches('button, .button, .btn')) {
                this.createClickRipple(e);
                this.playClickSound();
            }
        });

        // Scroll effects
        window.addEventListener('scroll', this.handleScroll.bind(this));
    }

    addHoverEffect(element) {
        // Create scan line effect
        const scanLine = document.createElement('div');
        scanLine.style.position = 'absolute';
        scanLine.style.top = '0';
        scanLine.style.left = '0';
        scanLine.style.width = '100%';
        scanLine.style.height = '2px';
        scanLine.style.background = 'linear-gradient(90deg, transparent, #00ff9f, transparent)';
        scanLine.style.animation = 'scan 0.5s ease-out';
        scanLine.style.zIndex = '10';
        
        element.style.position = 'relative';
        element.appendChild(scanLine);
        
        setTimeout(() => {
            if (scanLine.parentNode) {
                scanLine.parentNode.removeChild(scanLine);
            }
        }, 500);
    }

    createClickRipple(event) {
        const ripple = document.createElement('div');
        const rect = event.target.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        
        ripple.style.position = 'absolute';
        ripple.style.width = size + 'px';
        ripple.style.height = size + 'px';
        ripple.style.borderRadius = '50%';
        ripple.style.background = 'radial-gradient(circle, rgba(0,255,159,0.6) 0%, transparent 70%)';
        ripple.style.left = (event.clientX - rect.left - size/2) + 'px';
        ripple.style.top = (event.clientY - rect.top - size/2) + 'px';
        ripple.style.pointerEvents = 'none';
        ripple.style.transform = 'scale(0)';
        ripple.style.animation = 'ripple 0.6s ease-out';
        ripple.style.zIndex = '1000';
        
        event.target.style.position = 'relative';
        event.target.appendChild(ripple);
        
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 600);
    }

    playClickSound() {
        // Create audio context for cyberpunk sounds
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.frequency.setValueAtTime(800, this.audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(400, this.audioContext.currentTime + 0.1);
        
        gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);
        
        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + 0.1);
    }

    // Audio visualizer for system sounds
    setupAudioVisualizer() {
        const visualizer = document.createElement('div');
        visualizer.id = 'audio-visualizer';
        visualizer.style.position = 'fixed';
        visualizer.style.bottom = '20px';
        visualizer.style.right = '20px';
        visualizer.style.width = '100px';
        visualizer.style.height = '30px';
        visualizer.style.display = 'flex';
        visualizer.style.alignItems = 'end';
        visualizer.style.gap = '2px';
        visualizer.style.zIndex = '1000';
        document.body.appendChild(visualizer);

        for (let i = 0; i < 10; i++) {
            const bar = document.createElement('div');
            bar.style.width = '8px';
            bar.style.background = '#00ff9f';
            bar.style.boxShadow = '0 0 10px #00ff9f';
            bar.style.animation = `audioBar ${Math.random() * 1 + 0.5}s ease-in-out infinite alternate`;
            bar.style.animationDelay = Math.random() * 0.5 + 's';
            visualizer.appendChild(bar);
        }
    }

    handleScroll() {
        const scrolled = window.pageYOffset;
        const parallax = document.querySelector('.parallax-bg');
        if (parallax) {
            parallax.style.transform = `translateY(${scrolled * 0.5}px)`;
        }

        // Reveal animations
        const reveals = document.querySelectorAll('[data-reveal]');
        reveals.forEach(element => {
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                element.classList.add('revealed');
            }
        });
    }

    // Animation presets
    setupAnimations() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes float {
                0% { transform: translateY(0px) rotate(0deg); opacity: 1; }
                50% { transform: translateY(-20px) rotate(180deg); opacity: 0.5; }
                100% { transform: translateY(-40px) rotate(360deg); opacity: 0; }
            }
            
            @keyframes ripple {
                to { transform: scale(2); opacity: 0; }
            }
            
            @keyframes audioBar {
                0% { height: 5px; }
                100% { height: 25px; }
            }
            
            @keyframes scan {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            [data-reveal] {
                opacity: 0;
                transform: translateY(50px);
                transition: all 0.8s ease-out;
            }
            
            [data-reveal].revealed {
                opacity: 1;
                transform: translateY(0);
            }
            
            .cyber-particle {
                animation: float linear infinite;
            }
        `;
        document.head.appendChild(style);
    }

    // System monitoring display
    addSystemMonitor() {
        const monitor = document.createElement('div');
        monitor.id = 'system-monitor';
        monitor.style.position = 'fixed';
        monitor.style.top = '20px';
        monitor.style.left = '20px';
        monitor.style.background = 'rgba(10, 14, 39, 0.9)';
        monitor.style.border = '1px solid #00ff9f';
        monitor.style.borderRadius = '5px';
        monitor.style.padding = '10px';
        monitor.style.fontFamily = 'Share Tech Mono, monospace';
        monitor.style.fontSize = '12px';
        monitor.style.color = '#00ff9f';
        monitor.style.zIndex = '1000';
        monitor.innerHTML = `
            <div>SYS: ONLINE</div>
            <div>MEM: <span id="mem-usage">--</span>%</div>
            <div>CPU: <span id="cpu-usage">--</span>%</div>
            <div>NET: <span id="net-status">CONN</span></div>
        `;
        document.body.appendChild(monitor);

        // Simulate system stats
        setInterval(() => {
            document.getElementById('mem-usage').textContent = Math.floor(Math.random() * 20 + 60);
            document.getElementById('cpu-usage').textContent = Math.floor(Math.random() * 30 + 20);
        }, 2000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new CyberpunkDashboard();
});

// Additional utility functions
function addCyberpunkEffect(selector, effect = 'glow') {
    const elements = document.querySelectorAll(selector);
    elements.forEach(element => {
        element.classList.add(`cyber-${effect}`);
    });
}

function createHolographicText(text, container) {
    const hologram = document.createElement('div');
    hologram.className = 'holographic';
    hologram.textContent = text;
    hologram.style.fontSize = '2rem';
    hologram.style.fontWeight = 'bold';
    hologram.style.textAlign = 'center';
    container.appendChild(hologram);
}

// Service status checker with cyberpunk styling
function updateServiceStatus(serviceName, isOnline) {
    const serviceCard = document.querySelector(`[data-service="${serviceName}"]`);
    if (serviceCard) {
        const indicator = serviceCard.querySelector('.status-indicator');
        if (indicator) {
            indicator.className = `status-indicator ${isOnline ? 'status-online' : 'status-offline'}`;
        }
        
        serviceCard.classList.toggle('service-online', isOnline);
        serviceCard.classList.toggle('service-offline', !isOnline);
    }
}

// Export for global use
window.CyberpunkDashboard = CyberpunkDashboard;
window.addCyberpunkEffect = addCyberpunkEffect;
window.createHolographicText = createHolographicText;
window.updateServiceStatus = updateServiceStatus;