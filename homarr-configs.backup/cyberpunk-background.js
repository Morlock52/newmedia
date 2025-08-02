// Cyberpunk Dynamic Background Animation for Homarr
// Creates animated mesh gradients, particles, and glitch effects

class CyberpunkBackground {
  constructor() {
    this.canvas = null;
    this.ctx = null;
    this.particles = [];
    this.meshPoints = [];
    this.glitchTimer = 0;
    this.animationId = null;
    
    this.colors = {
      neonCyan: '#00ffff',
      neonMagenta: '#ff00ff',
      neonPink: '#ff0080',
      neonBlue: '#00aaff',
      neonGreen: '#00ff88',
      neonYellow: '#ffaa00'
    };
    
    this.init();
  }
  
  init() {
    this.createCanvas();
    this.createParticles();
    this.createMeshPoints();
    this.startAnimation();
    this.addEventListeners();
  }
  
  createCanvas() {
    // Remove existing canvas if any
    const existingCanvas = document.getElementById('cyberpunk-bg');
    if (existingCanvas) {
      existingCanvas.remove();
    }
    
    this.canvas = document.createElement('canvas');
    this.canvas.id = 'cyberpunk-bg';
    this.canvas.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: -10;
      opacity: 0.6;
      pointer-events: none;
    `;
    
    document.body.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.resize();
  }
  
  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }
  
  createParticles() {
    this.particles = [];
    const particleCount = Math.floor((this.canvas?.width || window.innerWidth) / 20);
    
    for (let i = 0; i < particleCount; i++) {
      this.particles.push({
        x: Math.random() * (this.canvas?.width || window.innerWidth),
        y: Math.random() * (this.canvas?.height || window.innerHeight),
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 1,
        color: Object.values(this.colors)[Math.floor(Math.random() * Object.values(this.colors).length)],
        opacity: Math.random() * 0.8 + 0.2,
        pulse: Math.random() * Math.PI * 2
      });
    }
  }
  
  createMeshPoints() {
    this.meshPoints = [];
    const gridSize = 100;
    const cols = Math.ceil((this.canvas?.width || window.innerWidth) / gridSize);
    const rows = Math.ceil((this.canvas?.height || window.innerHeight) / gridSize);
    
    for (let x = 0; x < cols; x++) {
      for (let y = 0; y < rows; y++) {
        this.meshPoints.push({
          x: x * gridSize + Math.random() * 50 - 25,
          y: y * gridSize + Math.random() * 50 - 25,
          originX: x * gridSize,
          originY: y * gridSize,
          offset: Math.random() * Math.PI * 2,
          amplitude: Math.random() * 30 + 10
        });
      }
    }
  }
  
  updateParticles(time) {
    this.particles.forEach(particle => {
      particle.x += particle.vx;
      particle.y += particle.vy;
      particle.pulse += 0.02;
      
      // Wrap around screen
      if (particle.x < 0) particle.x = this.canvas.width;
      if (particle.x > this.canvas.width) particle.x = 0;
      if (particle.y < 0) particle.y = this.canvas.height;
      if (particle.y > this.canvas.height) particle.y = 0;
      
      // Pulse effect
      particle.opacity = (Math.sin(particle.pulse) + 1) * 0.3 + 0.2;
    });
  }
  
  updateMeshPoints(time) {
    this.meshPoints.forEach(point => {
      point.x = point.originX + Math.sin(time * 0.001 + point.offset) * point.amplitude;
      point.y = point.originY + Math.cos(time * 0.0015 + point.offset) * point.amplitude;
    });
  }
  
  drawParticles() {
    this.particles.forEach(particle => {
      this.ctx.save();
      this.ctx.globalAlpha = particle.opacity;
      this.ctx.fillStyle = particle.color;
      this.ctx.shadowColor = particle.color;
      this.ctx.shadowBlur = 10;
      
      this.ctx.beginPath();
      this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
      this.ctx.fill();
      
      this.ctx.restore();
    });
  }
  
  drawConnections() {
    const maxDistance = 150;
    
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const dx = this.particles[i].x - this.particles[j].x;
        const dy = this.particles[i].y - this.particles[j].y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < maxDistance) {
          const opacity = (1 - distance / maxDistance) * 0.3;
          
          this.ctx.save();
          this.ctx.globalAlpha = opacity;
          this.ctx.strokeStyle = this.colors.neonCyan;
          this.ctx.lineWidth = 1;
          this.ctx.shadowColor = this.colors.neonCyan;
          this.ctx.shadowBlur = 5;
          
          this.ctx.beginPath();
          this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
          this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
          this.ctx.stroke();
          
          this.ctx.restore();
        }
      }
    }
  }
  
  drawMeshGradient(time) {
    const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
    
    // Animated gradient stops
    const hue1 = (time * 0.05) % 360;
    const hue2 = (time * 0.03 + 120) % 360;
    const hue3 = (time * 0.04 + 240) % 360;
    
    gradient.addColorStop(0, `hsla(${hue1}, 100%, 50%, 0.1)`);
    gradient.addColorStop(0.5, `hsla(${hue2}, 100%, 50%, 0.05)`);
    gradient.addColorStop(1, `hsla(${hue3}, 100%, 50%, 0.1)`);
    
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }
  
  drawGlitchEffect(time) {
    if (Math.random() < 0.05) { // 5% chance per frame
      this.glitchTimer = time + 200; // Glitch for 200ms
    }
    
    if (time < this.glitchTimer) {
      const intensity = (this.glitchTimer - time) / 200;
      
      // Create glitch bars
      for (let i = 0; i < 5; i++) {
        const y = Math.random() * this.canvas.height;
        const height = Math.random() * 20 + 5;
        const offset = (Math.random() - 0.5) * 20 * intensity;
        
        this.ctx.save();
        this.ctx.globalCompositeOperation = 'screen';
        this.ctx.fillStyle = Math.random() > 0.5 ? this.colors.neonMagenta : this.colors.neonCyan;
        this.ctx.globalAlpha = intensity * 0.5;
        this.ctx.fillRect(0, y, this.canvas.width, height);
        this.ctx.restore();
      }
      
      // RGB shift effect
      this.ctx.save();
      this.ctx.globalCompositeOperation = 'multiply';
      this.ctx.filter = `hue-rotate(${Math.random() * 360}deg)`;
      this.ctx.drawImage(this.canvas, offset, 0);
      this.ctx.restore();
    }
  }
  
  drawScanlines() {
    this.ctx.save();
    this.ctx.globalAlpha = 0.1;
    this.ctx.fillStyle = this.colors.neonCyan;
    
    for (let y = 0; y < this.canvas.height; y += 4) {
      this.ctx.fillRect(0, y, this.canvas.width, 1);
    }
    
    this.ctx.restore();
  }
  
  animate(time = 0) {
    // Clear canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Draw layers
    this.drawMeshGradient(time);
    this.updateMeshPoints(time);
    this.updateParticles(time);
    this.drawParticles();
    this.drawConnections();
    this.drawGlitchEffect(time);
    this.drawScanlines();
    
    this.animationId = requestAnimationFrame((t) => this.animate(t));
  }
  
  startAnimation() {
    this.animationId = requestAnimationFrame((t) => this.animate(t));
  }
  
  stopAnimation() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
  }
  
  addEventListeners() {
    window.addEventListener('resize', () => {
      this.resize();
      this.createParticles();
      this.createMeshPoints();
    });
    
    // Mouse interaction
    document.addEventListener('mousemove', (e) => {
      const mouseInfluence = 50;
      
      this.particles.forEach(particle => {
        const dx = e.clientX - particle.x;
        const dy = e.clientY - particle.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < mouseInfluence) {
          const force = (mouseInfluence - distance) / mouseInfluence;
          particle.vx += (dx / distance) * force * 0.1;
          particle.vy += (dy / distance) * force * 0.1;
        }
      });
    });
  }
  
  destroy() {
    this.stopAnimation();
    if (this.canvas) {
      this.canvas.remove();
    }
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  // Wait a bit for Homarr to load
  setTimeout(() => {
    window.cyberpunkBg = new CyberpunkBackground();
  }, 1000);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (window.cyberpunkBg) {
    window.cyberpunkBg.destroy();
  }
});

// Additional CSS injection for enhanced effects
const additionalStyles = `
  /* Enhanced glassmorphism */
  .card, [class*="Card"], [class*="Paper"] {
    background: rgba(0, 0, 0, 0.4) !important;
    backdrop-filter: blur(20px) saturate(180%) brightness(120%) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    box-shadow: 
      0 8px 32px rgba(0, 255, 255, 0.15),
      inset 0 1px 0 rgba(255, 255, 255, 0.1),
      0 0 0 1px rgba(0, 255, 255, 0.1) !important;
  }
  
  /* Hover glow effect */
  .card:hover, [class*="Card"]:hover, [class*="Paper"]:hover {
    box-shadow: 
      0 8px 32px rgba(0, 255, 255, 0.3),
      0 0 80px rgba(255, 0, 255, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.2),
      0 0 0 1px rgba(0, 255, 255, 0.3) !important;
    transform: translateY(-4px) scale(1.02) !important;
  }
  
  /* Text glow */
  h1, h2, h3, h4, h5, h6 {
    text-shadow: 
      0 0 10px currentColor,
      0 0 20px currentColor,
      0 0 30px currentColor !important;
  }
  
  /* Icon glow */
  img[src*="icon"], [class*="icon"] {
    filter: 
      drop-shadow(0 0 10px rgba(0, 255, 255, 0.5))
      drop-shadow(0 0 20px rgba(0, 255, 255, 0.3)) !important;
    transition: all 0.3s ease !important;
  }
  
  /* Button cyberpunk style */
  button, [class*="Button"] {
    position: relative !important;
    overflow: hidden !important;
  }
  
  button::after, [class*="Button"]::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: radial-gradient(circle, rgba(0, 255, 255, 0.4) 0%, transparent 70%);
    transition: all 0.6s ease;
    transform: translate(-50%, -50%);
  }
  
  button:hover::after, [class*="Button"]:hover::after {
    width: 300px;
    height: 300px;
  }
`;

// Inject additional styles
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);