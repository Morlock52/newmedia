const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Files to process: all JS files under api and other server dirs
const targets = execSync("rg -l \"console\\.(log|warn|error|info|debug)\\(" api --hidden || true", { encoding: 'utf8' })
  .split('\n').filter(Boolean);

console.log('Found files:', targets.length);

for (const file of targets) {
  let content = fs.readFileSync(file, 'utf8');
  // Skip if already has logger require
  if (!/\blogger\b/.test(content)) {
    // compute relative path to middleware/logger.js
    const rel = path.relative(path.dirname(file), path.join(process.cwd(), 'middleware', 'logger.js')).replace(/\\/g, '/');
    const requireLine = `const logger = require('${rel.startsWith('.')?rel:'./'+rel}');\n`;
    // Insert require after any shebang or 'use strict' or first line
    if (content.startsWith('#!')) {
      const idx = content.indexOf('\n');
      content = content.slice(0, idx+1) + requireLine + content.slice(idx+1);
    } else {
      content = requireLine + content;
    }
  }

  // Replace console.* occurrences with logger.* (preserve spacing)
  content = content.replace(/console\.log\s*\(/g, 'logger.info(')
                   .replace(/console\.info\s*\(/g, 'logger.info(')
                   .replace(/console\.warn\s*\(/g, 'logger.warn(')
                   .replace(/console\.error\s*\(/g, 'logger.error(')
                   .replace(/console\.debug\s*\(/g, 'logger.debug(');

  fs.writeFileSync(file, content);
  console.log('Patched', file);
}
