async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.text();
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
}

// Minimal client-side Markdown renderer for our docs subset
function renderMarkdown(md) {
  // Normalize line endings
  md = md.replace(/\r\n?/g, '\n');
  // Code blocks ```
  md = md.replace(/```([\s\S]*?)```/g, (m, p1) => `<pre><code>${escapeHtml(p1)}</code></pre>`);
  // Images ![alt](path)
  md = md.replace(/!\[([^\]]*)\]\(([^\)]+)\)/g, (m, alt, src) => {
    // Rewrite relative image paths to /docs/images
    let s = src.trim();
    if (!/^https?:/i.test(s)) {
      // Allow docs/images/ and images/
      if (s.startsWith('docs/images/')) s = '/' + s;
      else if (s.startsWith('images/')) s = '/docs/' + s;
      else s = s; // leave as-is
    }
    return `<img alt="${escapeHtml(alt)}" src="${s}">`;
  });
  // Links [text](url)
  md = md.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, (m, text, href) => {
    const h = href.startsWith('http') ? href : href;
    return `<a href="${escapeHtml(h)}" target="_blank" rel="noopener">${escapeHtml(text)}</a>`;
  });
  // Headers
  md = md.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>')
         .replace(/^##\s+(.+)$/gm, '<h2>$1</h2>')
         .replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
  // Bulleted lists
  md = md.replace(/^(?:-\s+.+\n?)+/gm, block => {
    const items = block.trim().split(/\n/).map(l => l.replace(/^[-*]\s+/, '').trim());
    return '<ul>' + items.map(i => `<li>${i}</li>`).join('') + '</ul>';
  });
  // Code spans `code`
  md = md.replace(/`([^`]+)`/g, (m, p1) => `<code>${escapeHtml(p1)}</code>`);
  // Paragraphs (naive): wrap lines not already HTML
  md = md.split('\n\n').map(chunk => {
    if (/^\s*</.test(chunk)) return chunk;
    return '<p>' + chunk.replace(/\n/g, '<br>') + '</p>';
  }).join('\n');
  return md;
}

function slugify(name) { return name.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/(^-|-$)/g,''); }

async function loadDocs() {
  const list = await fetchJSON('/api/docs/list');
  const select = document.getElementById('docSelect');
  const docList = document.getElementById('docList');
  select.innerHTML = '';
  docList.innerHTML = '';
  (list.docs || []).forEach((d, idx) => {
    const opt = document.createElement('option');
    opt.value = d.path;
    opt.textContent = d.name;
    select.appendChild(opt);
    const div = document.createElement('div');
    div.className = 'doc-item';
    div.dataset.path = d.path;
    div.innerHTML = `<div class="doc-title">${d.name}</div>`;
    div.onclick = () => { select.value = d.path; renderDoc(d.path); highlight(d.path); };
    docList.appendChild(div);
  });
  if (list.docs && list.docs.length) {
    select.onchange = () => { renderDoc(select.value); highlight(select.value); };
    renderDoc(select.value || list.docs[0].path);
    highlight(select.value || list.docs[0].path);
  } else {
    document.getElementById('doc').innerHTML = '<p>No documentation found.</p>';
  }
}

function highlight(path) {
  document.querySelectorAll('.doc-item').forEach(el => {
    el.classList.toggle('active', el.dataset.path === path);
  });
}

async function renderDoc(path) {
  try {
    const md = await fetchText('/api/docs/content?path=' + encodeURIComponent(path));
    const html = renderMarkdown(md);
    document.getElementById('doc').innerHTML = html;
    document.title = `Docs • ${path}`;
  } catch (e) {
    document.getElementById('doc').innerHTML = '<p>Failed to load document.</p>';
  }
}

function printDoc() { window.print(); }

document.addEventListener('DOMContentLoaded', loadDocs);

