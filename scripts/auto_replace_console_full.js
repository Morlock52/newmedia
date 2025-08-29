#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Build ripgrep command with exclusions
const rgCmd = [
  'rg -l "console\\.(log|warn|error|info|debug)\\(" --hidden',
  "-g '!**/src/client/**'",
  "-g '!**/public/**'",
  "-g '!**/*.html'",
  "-g '!**/dashboard-2025/**'",
  "-g '!**/*.tsx'",
  "-g '!**/*.jsx'",
  "-g '!**/*.md'",
  "-g '!**/node_modules/**'",
  "-g '!**/.git/**'"
].join(' ');

let out = '';
try { out = execSync(rgCmd, { encoding: 'utf8' }); } catch (e) { out = e.stdout || ''; }
const files = out.split('\n').filter(Boolean);
console.log('Files to patch:', files.length);

for (const file of files) {
  let content = fs.readFileSync(file, 'utf8');
  const isESM = /(^|\n)\s*import\s+/.test(content) || /(^|\n)\s*export\s+/.test(content);
  if (!/\blogger\b/.test(content)) {
    const rel = path.relative(path.dirname(file), path.join(process.cwd(), 'middleware', 'logger.js')).replace(/\\/g, '/');
    if (isESM) {
      const importLine = `import { createRequire } from 'module';\nconst require = createRequire(import.meta.url);\nconst logger = require('${rel.startsWith('.') ? rel : './'+rel}');\n`;
      if (content.startsWith('#!')) {
        const idx = content.indexOf('\n');
        content = content.slice(0, idx+1) + importLine + content.slice(idx+1);
      } else {
        content = importLine + content;
      }
    } else {
      const requireLine = `const logger = require('${rel.startsWith('.') ? rel : './'+rel}');\n`;
      if (content.startsWith('#!')) {
        const idx = content.indexOf('\n');
        content = content.slice(0, idx+1) + requireLine + content.slice(idx+1);
      } else {
        content = requireLine + content;
      }
    }
  }
  content = content.replace(/console\.log\s*\(/g, 'logger.info(')
                   .replace(/console\.info\s*\(/g, 'logger.info(')
                   .replace(/console\.warn\s*\(/g, 'logger.warn(')
                   .replace(/console\.error\s*\(/g, 'logger.error(')
                   .replace(/console\.debug\s*\(/g, 'logger.debug(');
  fs.writeFileSync(file, content, 'utf8');
  console.log('Patched', file);
}
console.log('Done.');
