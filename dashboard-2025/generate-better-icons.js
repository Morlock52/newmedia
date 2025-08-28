const fs = require('fs');
const path = require('path');

// Create a simple valid PNG using Canvas-like approach
function createValidPNG(size) {
  // PNG file signature
  const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
  
  // Helper to calculate CRC32
  const crc32Table = [];
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    crc32Table[i] = c;
  }
  
  function crc32(data) {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < data.length; i++) {
      crc = crc32Table[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }
  
  // Create IHDR chunk
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(size, 0);  // width
  ihdrData.writeUInt32BE(size, 4);  // height
  ihdrData[8] = 8;   // bit depth
  ihdrData[9] = 6;   // color type (RGBA)
  ihdrData[10] = 0;  // compression
  ihdrData[11] = 0;  // filter
  ihdrData[12] = 0;  // interlace
  
  const ihdrType = Buffer.from('IHDR');
  const ihdrCRC = crc32(Buffer.concat([ihdrType, ihdrData]));
  
  const ihdr = Buffer.alloc(25);
  ihdr.writeUInt32BE(13, 0);
  ihdr.write('IHDR', 4);
  ihdrData.copy(ihdr, 8);
  ihdr.writeUInt32BE(ihdrCRC, 21);
  
  // Create image data (gradient)
  const imageDataSize = size * size * 4;
  const imageData = Buffer.alloc(imageDataSize);
  
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 4;
      
      // Create a purple-to-cyan gradient
      const t = (x + y) / (size * 2);
      
      imageData[idx] = Math.floor(139 * (1 - t) + 6 * t);     // R
      imageData[idx + 1] = Math.floor(92 * (1 - t) + 182 * t); // G  
      imageData[idx + 2] = Math.floor(246 * (1 - t) + 212 * t); // B
      imageData[idx + 3] = 255; // A
    }
  }
  
  // Compress the image data with zlib
  const zlib = require('zlib');
  
  // Add filter bytes (0 = no filter) for each row
  const filteredData = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    filteredData[y * (size * 4 + 1)] = 0; // filter type
    imageData.copy(filteredData, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }
  
  const compressedData = zlib.deflateSync(filteredData, { level: 9 });
  
  // Create IDAT chunk
  const idatType = Buffer.from('IDAT');
  const idatCRC = crc32(Buffer.concat([idatType, compressedData]));
  
  const idat = Buffer.alloc(compressedData.length + 12);
  idat.writeUInt32BE(compressedData.length, 0);
  idat.write('IDAT', 4);
  compressedData.copy(idat, 8);
  idat.writeUInt32BE(idatCRC, compressedData.length + 8);
  
  // Create IEND chunk
  const iend = Buffer.from([0, 0, 0, 0, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]);
  
  // Combine all chunks
  return Buffer.concat([PNG_SIGNATURE, ihdr, idat, iend]);
}

// Generate icons
const sizes = [72, 96, 128, 144, 152, 192, 384, 512];
const outputDir = path.join(__dirname, 'public', 'icons');

console.log('Generating PNG icons...');

sizes.forEach(size => {
  const pngData = createValidPNG(size);
  const outputPath = path.join(outputDir, `icon-${size}x${size}.png`);
  
  fs.writeFileSync(outputPath, pngData);
  console.log(`Created ${outputPath} (${pngData.length} bytes)`);
});

console.log('All icons generated successfully!');