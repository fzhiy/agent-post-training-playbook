#!/usr/bin/env node
// Build a FULLY STATIC site into docs/ — markdown + math + code highlighting are
// all pre-rendered at BUILD time (marked + KaTeX + highlight.js). KaTeX & hljs
// CSS/fonts are vendored under docs/vendor/. No runtime CDN → works in mainland
// China and offline.
//   npm install        # marked + katex + highlight.js (build-only)
//   node build.js

const fs = require('fs');
const path = require('path');
const _m = require('marked');
const marked = _m.marked || _m;
const katex = require('katex');
const _h = require('highlight.js');
const hljs = _h.default || _h;

const ROOT = __dirname;
const OUT = path.join(ROOT, 'docs');
const tpl = fs.readFileSync(path.join(ROOT, '_template.html'), 'utf8');
fs.mkdirSync(OUT, { recursive: true });

function titleOf(md, fallback) {
  const m = md.match(/^#\s+(.+)$/m);
  return m ? m[1].trim() : fallback;
}

// 1) Pre-render math with KaTeX (static; no runtime MathJax).
function renderWithMath(md) {
  const slots = [];
  const tex = (s, display) => {
    let html;
    try { html = katex.renderToString(s.trim(), { displayMode: display, throwOnError: false }); }
    catch (e) { html = '<code>' + s.trim().replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</code>'; }
    slots.push(html);
    return '@@KMATH' + (slots.length - 1) + '@@';
  };
  md = md
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, s) => tex(s, true))
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, s) => tex(s, true))
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, s) => tex(s, false))
    .replace(/\$([^\$\n]+?)\$/g, (_, s) => tex(s, false));
  let html = marked.parse(md, { async: false });
  return html.replace(/@@KMATH(\d+)@@/g, (_, i) => slots[+i]);
}

// 2) Pre-highlight fenced code (static; no runtime highlight.js).
function unescapeHtml(s) {
  return s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
}
function highlightCode(html) {
  return html.replace(/<pre><code(?: class="language-([\w+#.-]+)")?>([\s\S]*?)<\/code><\/pre>/g, (m, lang, code) => {
    const raw = unescapeHtml(code);
    let value;
    try {
      value = (lang && hljs.getLanguage(lang)) ? hljs.highlight(raw, { language: lang }).value : hljs.highlightAuto(raw).value;
    } catch (e) { return m; }
    return '<pre><code class="hljs' + (lang ? ' language-' + lang : '') + '">' + value + '</code></pre>';
  });
}

// 3) Add heading ids + a per-page Table of Contents (skipped for short pages).
function addTocAndIds(html) {
  const toc = [];
  let i = 0;
  html = html.replace(/<h([23])>([\s\S]*?)<\/h\1>/g, (m, lvl, inner) => {
    const id = 'sec-' + (i++);
    toc.push({ lvl: +lvl, id, text: inner.replace(/<[^>]+>/g, '').trim() });
    return '<h' + lvl + ' id="' + id + '">' + inner + '</h' + lvl + '>';
  });
  if (toc.length < 3) return html;
  const items = toc.map((t) => '<li class="lv' + t.lvl + '"><a href="#' + t.id + '">' + t.text + '</a></li>').join('');
  const nav = '<details class="toc" open><summary>目录 / Contents (' + toc.length + ')</summary><ul>' + items + '</ul></details>';
  return /<\/h1>/.test(html) ? html.replace('</h1>', '</h1>' + nav) : nav + html;
}

function renderDoc(md, title, outFile) {
  const content = addTocAndIds(highlightCode(renderWithMath(md)));
  const html = tpl.replace('{{TITLE}}', () => title).replace('{{CONTENT}}', () => content);
  fs.writeFileSync(path.join(OUT, outFile), html);
}

const items = [];
for (const f of fs.readdirSync(path.join(ROOT, 'cheatsheets')).filter((f) => f.endsWith('.md')).sort()) {
  const slug = f.replace(/\.md$/, '');
  const md = fs.readFileSync(path.join(ROOT, 'cheatsheets', f), 'utf8');
  const out = 'cheatsheet-' + slug + '.html';
  renderDoc(md, titleOf(md, slug), out);
  items.push({ section: 'Cheatsheets 题解', title: titleOf(md, slug), href: out });
}
for (const d of fs.readdirSync(path.join(ROOT, 'drills')).sort()) {
  const rp = path.join(ROOT, 'drills', d, 'README.md');
  if (!fs.existsSync(rp)) continue;
  const md = fs.readFileSync(rp, 'utf8');
  const out = 'drill-' + d + '.html';
  renderDoc(md, titleOf(md, d), out);
  items.push({ section: 'Drills 手撕', title: titleOf(md, d), href: out });
}

// Index hub — pure HTML + vanilla JS search (no CDN).
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'); }
function buildIndex(items) {
  const groups = {};
  for (const it of items) (groups[it.section] = groups[it.section] || []).push(it);
  const sections = Object.keys(groups).map((sec) => {
    const cards = groups[sec].map((it) =>
      `<a class="card" data-t="${esc((it.title + ' ' + it.href).toLowerCase())}" href="${esc(it.href)}">${esc(it.title)}</a>`
    ).join('');
    return `<section><h2>${esc(sec)} <small>${groups[sec].length}</small></h2><div class="grid">${cards}</div></section>`;
  }).join('\n');
  return `<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Post-Training Playbook</title>
<style>
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;color:#222;background:#fafafa}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #e6e6e6;padding:.9rem 1rem;z-index:5}
h1{margin:.1rem 0;font-size:1.15rem}
.sub{color:#888;font-size:.8rem;margin:.2rem 0 .6rem}
#q{width:100%;max-width:980px;padding:.6rem .8rem;font-size:1rem;border:1px solid #ccc;border-radius:10px;background:transparent;color:inherit}
main{max-width:980px;margin:0 auto;padding:1rem}
h2{font-size:1rem;margin:1.4rem 0 .6rem;color:#0a7}
h2 small{color:#999;font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.6rem}
.card{display:block;padding:.7rem .8rem;border:1px solid #e6e6e6;border-radius:10px;background:#fff;text-decoration:none;color:inherit;font-size:.9rem;line-height:1.35}
.card:hover{border-color:#0a7}
footer{max-width:980px;margin:1rem auto 3rem;padding:0 1rem;color:#999;font-size:.78rem}
.hidden{display:none}
@media(prefers-color-scheme:dark){body{color:#ddd;background:#16181c}header{background:#1b1e23;border-color:#2c2f36}.card{background:#22262c;border-color:#2c2f36}}
</style></head>
<body>
<header>
<h1>Agent Post-Training Playbook</h1>
<div class="sub">${items.length} 篇 · Agent 后训练:RL/持续学习/自我改进 · 公式/代码静态渲染(零外部 CDN,国内直连)· 输入关键词过滤</div>
<input id="q" type="search" placeholder="搜索主题… / filter topics…" autocomplete="off" autofocus>
</header>
<main>${sections}</main>
<footer>AI 辅助整理的学习笔记,WIP,欢迎 issue/PR 纠错。由 <code>node build.js</code> 生成(marked + KaTeX + highlight.js 构建时渲染)。</footer>
<script>
const q=document.getElementById('q');
q.addEventListener('input',function(){
  const v=q.value.trim().toLowerCase();
  document.querySelectorAll('.card').forEach(function(c){c.classList.toggle('hidden', v.length>0 && c.dataset.t.indexOf(v)<0);});
  document.querySelectorAll('main section').forEach(function(s){
    const any=Array.prototype.some.call(s.querySelectorAll('.card'),function(c){return !c.classList.contains('hidden');});
    s.classList.toggle('hidden', !any);
  });
});
</script>
</body></html>`;
}

fs.writeFileSync(path.join(OUT, 'index.html'), buildIndex(items));
console.log('Built ' + items.length + ' static pages into docs/ (markdown + math + code pre-rendered, zero runtime CDN)');
