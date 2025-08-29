#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

function listFiles() {
  try {
    const out = execSync('rg -l "console\\.(log|warn|error|info|debug)\\(" api --hidden', { encoding: 'utf8' });
    return out.split('\n').filter(Boolean);
  } catch (e) {
    return [];
  }
}

const files = listFiles();
console.log('Files to patch:', files.length);

for (const file of files) {
  let content = fs.readFileSync(file, 'utf8');
  if (!/\blogger\b/.test(content)) {
    const rel = path.relative(path.dirname(file), path.join(process.cwd(), 'middleware', 'logger.js')).replace(/\\/g, '/');
    const requireLine = `const logger = require('${rel.startsWith('.') ? rel : './'+rel}');\n`;
    if (content.startsWith('#!')) {
      const idx = content.indexOf('\n');
      content = content.slice(0, idx+1) + requireLine + content.slice(idx+1);
    } else {
      content = requireLine + content;
    }
  }
  content = content.replace(/console\.log\s*\(/g, 'logger.info(')
                   .replace(/console\.info\s*\(/g, 'logger.info(')
                   .replace(/console\.warn\s*\(/g, 'logger.warn(')
                   .replace(/console\.error\s*\(/g, 'logger.error(')
                   .replace(/console\.debug\s*\(/g, 'logger.debug(');
  fs.writeFileSync(file, content);
  console.log('Patched', file);
}

console.log('Done.');
