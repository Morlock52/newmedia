#!/usr/bin/env node

// Quick test to verify the server can be imported and initialized
import('./index.js').then(() => {
  console.log('✅ Server imports correctly');
  process.exit(0);
}).catch((error) => {
  console.error('❌ Server import failed:', error);
  process.exit(1);
});