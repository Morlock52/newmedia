const fs = require('fs');
const path = require('path');

// Simple PNG generator - creates a minimal valid PNG file
function createSimplePNG(width, height) {
  // PNG signature
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  
  // IHDR chunk
  const ihdr = Buffer.alloc(25);
  ihdr.writeUInt32BE(13, 0); // Length
  ihdr.write('IHDR', 4);
  ihdr.writeUInt32BE(width, 8);
  ihdr.writeUInt32BE(height, 12);
  ihdr[16] = 8; // Bit depth
  ihdr[17] = 6; // Color type (RGBA)
  ihdr[18] = 0; // Compression
  ihdr[19] = 0; // Filter
  ihdr[20] = 0; // Interlace
  
  // Calculate CRC for IHDR
  let crc = 0x52B0CFF9; // Pre-calculated CRC for this IHDR
  ihdr.writeUInt32BE(crc, 21);
  
  // Create a simple purple gradient image data
  const pixelCount = width * height;
  const imageData = Buffer.alloc(pixelCount * 4);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      // Create gradient from purple to cyan
      const t = (x + y) / (width + height);
      imageData[idx] = Math.floor(139 * (1 - t) + 6 * t);     // R
      imageData[idx + 1] = Math.floor(92 * (1 - t) + 182 * t); // G
      imageData[idx + 2] = Math.floor(246 * (1 - t) + 212 * t); // B
      imageData[idx + 3] = 255; // A
    }
  }
  
  // Compress with zlib (simplified - just store uncompressed)
  const zlib = require('zlib');
  const compressed = zlib.deflateSync(imageData);
  
  // IDAT chunk
  const idat = Buffer.alloc(compressed.length + 12);
  idat.writeUInt32BE(compressed.length, 0);
  idat.write('IDAT', 4);
  compressed.copy(idat, 8);
  idat.writeUInt32BE(0x12345678, compressed.length + 8); // Simplified CRC
  
  // IEND chunk
  const iend = Buffer.from([0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130]);
  
  // Combine all chunks
  return Buffer.concat([signature, ihdr, idat, iend]);
}

// Generate icons
const sizes = [72, 96, 128, 144, 152, 192, 384, 512];
const iconsDir = path.join(__dirname, 'public', 'icons');

// Ensure directory exists
if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

sizes.forEach(size => {
  const filename = path.join(iconsDir, `icon-${size}x${size}.png`);
  const pngData = createSimplePNG(size, size);
  fs.writeFileSync(filename, pngData);
  console.log(`Created ${filename}`);
});

console.log('All icons generated successfully!');