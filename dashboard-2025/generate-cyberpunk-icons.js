const fs = require('fs');
const path = require('path');
const { createCanvas } = require('canvas');

function generateCyberpunkIcon(size) {
  const canvas = createCanvas(size, size);
  const ctx = canvas.getContext('2d');

  // Dark background
  ctx.fillStyle = '#0a0e1b';
  ctx.fillRect(0, 0, size, size);

  // Create cyberpunk grid pattern
  ctx.strokeStyle = '#00ffff20';
  ctx.lineWidth = 1;
  
  // Grid lines
  const gridSize = size / 8;
  for (let i = 0; i <= size; i += gridSize) {
    // Vertical lines
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, size);
    ctx.stroke();
    
    // Horizontal lines
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(size, i);
    ctx.stroke();
  }

  // Central hexagon
  const centerX = size / 2;
  const centerY = size / 2;
  const radius = size * 0.3;
  
  // Hexagon path
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI * 2 / 6) * i - Math.PI / 2;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.closePath();

  // Gradient fill for hexagon
  const gradient = ctx.createLinearGradient(0, 0, size, size);
  gradient.addColorStop(0, '#ff00ff');
  gradient.addColorStop(0.5, '#00ffff');
  gradient.addColorStop(1, '#9d00ff');
  
  ctx.fillStyle = gradient;
  ctx.fill();
  
  // Hexagon border with glow
  ctx.strokeStyle = '#00ffff';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#00ffff';
  ctx.shadowBlur = 20;
  ctx.stroke();

  // Inner circuit pattern
  ctx.shadowBlur = 0;
  const innerRadius = radius * 0.5;
  
  // Circuit nodes
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI * 2 / 6) * i;
    const x = centerX + Math.cos(angle) * innerRadius;
    const y = centerY + Math.sin(angle) * innerRadius;
    
    // Node circle
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#00ff88';
    ctx.fill();
    
    // Connection lines
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#00ff8840';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Center core
  ctx.beginPath();
  ctx.arc(centerX, centerY, 8, 0, Math.PI * 2);
  const coreGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 8);
  coreGradient.addColorStop(0, '#ffffff');
  coreGradient.addColorStop(0.5, '#00ffff');
  coreGradient.addColorStop(1, '#0099ff');
  ctx.fillStyle = coreGradient;
  ctx.fill();

  // Add "M" letter in cyberpunk style
  ctx.font = `bold ${size * 0.2}px 'Arial'`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#0a0e1b';
  ctx.fillText('M', centerX, centerY);

  // Corner accents
  const cornerSize = size * 0.1;
  ctx.strokeStyle = '#ff00ff';
  ctx.lineWidth = 2;
  
  // Top-left corner
  ctx.beginPath();
  ctx.moveTo(0, cornerSize);
  ctx.lineTo(0, 0);
  ctx.lineTo(cornerSize, 0);
  ctx.stroke();
  
  // Top-right corner
  ctx.beginPath();
  ctx.moveTo(size - cornerSize, 0);
  ctx.lineTo(size, 0);
  ctx.lineTo(size, cornerSize);
  ctx.stroke();
  
  // Bottom-left corner
  ctx.beginPath();
  ctx.moveTo(0, size - cornerSize);
  ctx.lineTo(0, size);
  ctx.lineTo(cornerSize, size);
  ctx.stroke();
  
  // Bottom-right corner
  ctx.beginPath();
  ctx.moveTo(size - cornerSize, size);
  ctx.lineTo(size, size);
  ctx.lineTo(size, size - cornerSize);
  ctx.stroke();

  return canvas.toBuffer('image/png');
}

// Ensure icons directory exists
const iconsDir = path.join(__dirname, 'public', 'icons');
if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

// Generate icons for all required sizes
const sizes = [72, 96, 128, 144, 152, 192, 384, 512];

console.log('🎨 Generating cyberpunk icons...');

sizes.forEach(size => {
  const buffer = generateCyberpunkIcon(size);
  const filename = `icon-${size}x${size}.png`;
  const filepath = path.join(iconsDir, filename);
  
  fs.writeFileSync(filepath, buffer);
  console.log(`✨ Generated ${filename}`);
});

// Also generate favicon
const faviconBuffer = generateCyberpunkIcon(32);
fs.writeFileSync(path.join(__dirname, 'public', 'favicon.png'), faviconBuffer);
console.log('✨ Generated favicon.png');

// Generate apple-touch-icon
const appleIconBuffer = generateCyberpunkIcon(180);
fs.writeFileSync(path.join(__dirname, 'public', 'apple-touch-icon.png'), appleIconBuffer);
console.log('✨ Generated apple-touch-icon.png');

console.log('🚀 All cyberpunk icons generated successfully!');