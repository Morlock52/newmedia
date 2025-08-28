#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

function listHtmlFiles(root) {
  const results = [];
  function walk(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const e of entries) {
      const skipPrefixes = ['node_modules', '.git', '.backup', 'backup', 'plex-config', '.cache'];
      if (skipPrefixes.some(p => e.name.startsWith(p))) continue;
      const full = path.join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.isFile() && e.name.toLowerCase().endsWith('.html')) results.push(full);
    }
  }
  walk(root);
  return results;
}

function extractAttributes(html, tag, attr) {
  const re = new RegExp(`<${tag}[^>]*${attr}=["']([^"']+)["'][^>]*>`, 'gi');
  const out = [];
  let m;
  while ((m = re.exec(html)) !== null) out.push(m[1]);
  return out;
}

function extractOnclickFunctions(html) {
  const re = /onclick\s*=\s*"([^"]+)"/gi;
  const funcs = [];
  let m;
  while ((m = re.exec(html)) !== null) {
    const call = m[1];
    const name = (call.match(/^[a-zA-Z_$][\w$]*/)||[])[0];
    if (name) funcs.push(name);
  }
  return [...new Set(funcs)];
}

function findFunctionDefinition(source, name) {
  const patterns = [
    new RegExp(`function\\s+${name}\\s*\\(`),
    new RegExp(`${name}\\s*=\\s*\\(`),
    new RegExp(`${name}\\s*:\\s*function\\s*\\(`)
  ];
  return patterns.some(r => r.test(source));
}

function readLocalScripts(htmlPath, html) {
  const dir = path.dirname(htmlPath);
  const srcs = extractAttributes(html, 'script', 'src')
    .filter(src => src && !/^https?:/i.test(src));
  const contents = [];
  for (const src of srcs) {
    const p = src.startsWith('/')
      ? path.resolve(process.cwd(), '.' + src)
      : path.resolve(dir, src);
    if (fs.existsSync(p)) {
      try { contents.push(fs.readFileSync(p, 'utf8')); } catch {}
    }
  }
  return contents.join('\n');
}

function checkFileExists(htmlPath, ref) {
  if (!ref) return true;
  if (/^https?:/i.test(ref)) return true;
  if (/^data:/i.test(ref)) return true;
  if (/^\/\//.test(ref)) return true; // protocol-relative external
  if (ref.startsWith('mailto:') || ref.startsWith('tel:')) return true;
  if (ref.startsWith('#')) return true;
  const cleanRef = ref.split('#')[0];
  const root = process.cwd();
  const dir = path.dirname(htmlPath);
  const target = cleanRef.startsWith('/')
    ? path.resolve(root, '.' + cleanRef)
    : path.resolve(dir, cleanRef);
  return fs.existsSync(target);
}

function analyzeHtml(htmlPath) {
  const html = fs.readFileSync(htmlPath, 'utf8');
  const issues = [];
  const warnings = [];

  // Assets
  const scriptSrc = extractAttributes(html, 'script', 'src');
  const linkHref = extractAttributes(html, 'link', 'href');
  const imgSrc = extractAttributes(html, 'img', 'src');
  const aHref = extractAttributes(html, 'a', 'href');

  const allRefs = [
    ...scriptSrc.map(x=>({type:'script',ref:x})),
    ...linkHref.map(x=>({type:'link',ref:x})),
    ...imgSrc.map(x=>({type:'img',ref:x})),
  ];

  for (const {type, ref} of allRefs) {
    if (!checkFileExists(htmlPath, ref)) {
      issues.push({kind:'missing-asset', type, ref});
    }
  }

  // Check anchor links point to real files or anchors
  for (const ref of aHref) {
    if (!checkFileExists(htmlPath, ref)) {
      // ignore external absolute URLs starting with //
      if (/^\/\//.test(ref) || /^https?:/i.test(ref)) continue;
      warnings.push({kind:'broken-link', ref});
    }
  }

  // onclick function references
  const onclickFuncs = extractOnclickFunctions(html);
  const scriptsCombined = readLocalScripts(htmlPath, html) + '\n' + html;
  const builtins = new Set(['alert','confirm','window','location','open','console','history','document','navigator','event','this']);
  const missingFuncs = onclickFuncs.filter(fn => !builtins.has(fn) && !findFunctionDefinition(scriptsCombined, fn));
  for (const fn of missingFuncs) issues.push({kind:'missing-function', fn});

  // Basic structure checks
  if (!/<title[^>]*>[^<]*<\/title>/i.test(html)) warnings.push({kind:'missing-title'});
  if (!/<meta[^>]*viewport/i.test(html)) warnings.push({kind:'missing-viewport'});

  return { htmlPath, issues, warnings, counts: { scriptSrc: scriptSrc.length, linkHref: linkHref.length, imgSrc: imgSrc.length, aHref: aHref.length } };
}

function main() {
  const root = process.cwd();
  const files = listHtmlFiles(root);
  const results = files.map(analyzeHtml);

  let totalIssues = 0, totalWarnings = 0;
  for (const r of results) {
    if (r.issues.length || r.warnings.length) {
      console.log(`\n== ${r.htmlPath} ==`);
      for (const i of r.issues) {
        totalIssues++;
        if (i.kind==='missing-asset') console.log(`❌ Missing ${i.type}: ${i.ref}`);
        if (i.kind==='missing-function') console.log(`❌ Onclick function not found: ${i.fn}`);
      }
      for (const w of r.warnings) {
        totalWarnings++;
        if (w.kind==='broken-link') console.log(`⚠️ Broken link: ${w.ref}`);
        if (w.kind==='missing-title') console.log(`⚠️ Missing <title>`);
        if (w.kind==='missing-viewport') console.log(`⚠️ Missing <meta viewport>`);
      }
    }
  }

  console.log(`\nSummary: ${results.length} HTML files scanned. Issues: ${totalIssues}, Warnings: ${totalWarnings}`);
  if (totalIssues>0) process.exitCode = 1;
}

if (require.main === module) main();
