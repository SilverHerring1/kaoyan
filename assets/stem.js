import { escapeHtml, showToast } from './app.js';

const KIND_META = {
  formula: { label: '核心公式', badgeClass: 'stem-kind-formula-chip' },
  conclusion: { label: '重要结论', badgeClass: 'stem-kind-conclusion-chip' },
  example: { label: '经典例题', badgeClass: 'stem-kind-example-chip' },
  exercise: { label: '课后习题', badgeClass: 'stem-kind-exercise-chip' },
  summary: { label: '题型总结', badgeClass: 'stem-kind-summary-chip' },
};

const DEFAULT_KINDS = ['formula', 'conclusion', 'example', 'summary'];

const STAGE_META = {
  foundation: { label: '基础30讲', badgeClass: 'stem-badge-foundation' },
  intensive: { label: '强化18讲', badgeClass: 'stem-badge-intensive' },
  both: { label: '基础+强化', badgeClass: 'stem-badge-both' },
};

const KATEX_OPTIONS = {
  delimiters: [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false },
    { left: '\\(', right: '\\)', display: false },
    { left: '\\[', right: '\\]', display: true },
  ],
  throwOnError: false,
  strict: 'ignore',
};

function renderMathSafe(rootEl, retries = 25) {
  if (!rootEl) return;
  if (typeof window.renderMathInElement === 'function') {
    try {
      window.renderMathInElement(rootEl, KATEX_OPTIONS);
      // Make wide display equations keyboard-scrollable
      for (const disp of rootEl.querySelectorAll('.katex-display')) {
        if (!disp.hasAttribute('tabindex')) {
          disp.setAttribute('tabindex', '0');
          disp.setAttribute('role', 'region');
          disp.setAttribute('aria-label', '数学公式');
        }
      }
    } catch (err) {
      console.warn('KaTeX 渲染异常：', err);
    }
    return;
  }
  if (retries > 0) {
    setTimeout(() => renderMathSafe(rootEl, retries - 1), 120);
  }
}

/**
 * Builds a chapter-local index mapping textbook citations (例 1.12, 强化例 1.8,
 * 主例 2-6, 辅例 2-19, 习题 2-21, A2-5, B2-23, 表 2-1, 式(2-62), item IDs)
 * to the exact card object in the current chapter. Zero dead links!
 */
function buildChapterRefIndex(items) {
  const byId = new Map();
  const mathEx = new Map();
  const ctrlEx = new Map();
  const exercises = new Map();
  const tables = new Map();
  const equations = new Map();

  const setIfEmpty = (map, key, val) => {
    if (key && !map.has(key)) map.set(key, val);
  };

  for (const item of items || []) {
    if (!item?.id) continue;
    byId.set(item.id, item);

    const title = String(item.title || '');
    const itemRef = String(item.source?.itemRef || '');
    const crossRef = String(item.source?.crossRef || '');
    const book = String(item.source?.book || '');
    const combined = `${title} ${itemRef}`;

    // 1. Math I examples: 例 1.12 / 强化例 1.8
    if (item.kind === 'example' || /例\s*\d+\.\d+/.test(combined)) {
      const stageBucket = item.stage === 'intensive' ? 'i' : 'f';
      const reMath = /(强化)?例\s*(\d+\.\d+)/g;
      let m;
      while ((m = reMath.exec(combined)) !== null) {
        const num = m[2];
        const bucket = m[1] ? 'i' : stageBucket;
        setIfEmpty(mathEx, `${bucket}:${num}`, item);
        setIfEmpty(mathEx, `any:${num}`, item);
      }
    }

    // 2. Control main-text & study-guide examples: 例 2-1a / 例 2-21~2-23
    const isMain = /自动控制原理/.test(book) || /主教材/.test(title);
    const isGuide = /学习辅导/.test(book) || /学习辅导/.test(title);
    if (item.kind === 'example') {
      // Expand ranges like 例2-21~2-23 or 例2-21~23
      const reRange = /例\s*(\d+)-(\d+)\s*[~～]\s*(?:\1-)?(\d+)/g;
      let mr;
      while ((mr = reRange.exec(combined)) !== null) {
        const ch = mr[1];
        const start = Number(mr[2]);
        const end = Number(mr[3]);
        if (Number.isFinite(start) && Number.isFinite(end) && end >= start && end - start <= 25) {
          for (let n = start; n <= end; n += 1) {
            const code = `${ch}-${n}`;
            if (isMain) setIfEmpty(ctrlEx, `main:${code}`, item);
            if (isGuide) setIfEmpty(ctrlEx, `guide:${code}`, item);
            setIfEmpty(ctrlEx, `any:${code}`, item);
          }
        }
      }
      // Single example refs like 例 2-1a, 例 2-6
      const reCtrl = /例\s*(\d+-\d+[a-z]?)/gi;
      let mc;
      while ((mc = reCtrl.exec(combined)) !== null) {
        const code = mc[1].toLowerCase();
        if (isMain) setIfEmpty(ctrlEx, `main:${code}`, item);
        if (isGuide) setIfEmpty(ctrlEx, `guide:${code}`, item);
        setIfEmpty(ctrlEx, `any:${code}`, item);
      }
    }

    // 3. Control exercises: 习题 2-5 / A2-5 / B2-23
    if (item.kind === 'exercise') {
      const exText = `${title} ${itemRef} ${crossRef}`;
      const reExer = /习题\s*(\d+-\d+)/g;
      let me;
      while ((me = reExer.exec(exText)) !== null) {
        setIfEmpty(exercises, `ex:${me[1]}`, item);
      }
      const reCode = /\b([AB]\d+-\d+)\b/g;
      let mk;
      while ((mk = reCode.exec(exText)) !== null) {
        setIfEmpty(exercises, `code:${mk[1]}`, item);
      }
    }

    // 4. Tables: 表 2-1, 表 2-4, 辅导表 2-2
    if (Array.isArray(item.tables)) {
      for (const tbl of item.tables) {
        const cap = String(tbl.caption || '');
        const mt = cap.match(/表\s*(\d+-\d+)/);
        if (mt) {
          const isGuideTbl = /学习辅导|辅导表/.test(cap);
          if (isGuideTbl) setIfEmpty(tables, `guide:${mt[1]}`, item);
          else setIfEmpty(tables, `main:${mt[1]}`, item);
          setIfEmpty(tables, `any:${mt[1]}`, item);
        }
      }
    }
    const reTblRef = /(辅导)?表\s*(\d+-\d+)/g;
    let mtr;
    while ((mtr = reTblRef.exec(itemRef)) !== null) {
      if (mtr[1] || isGuide) setIfEmpty(tables, `guide:${mtr[2]}`, item);
      else setIfEmpty(tables, `main:${mtr[2]}`, item);
      setIfEmpty(tables, `any:${mtr[2]}`, item);
    }

    // 5. Equations (prioritize formula cards first, then others): 式(2-41)~(2-62) or 式(1-3)
    const registerEq = (code) => {
      if (item.kind === 'formula') {
        equations.set(`eq:${code}`, item);
      } else {
        setIfEmpty(equations, `eq:${code}`, item);
      }
    };
    const reEqRange = /式\s*[（(]?\s*(\d+)-(\d+)\s*[）)]?\s*[~～]\s*[（(]?\s*(?:\1-)?(\d+)\s*[）)]?/g;
    let mer;
    while ((mer = reEqRange.exec(itemRef)) !== null) {
      const ch = mer[1];
      const start = Number(mer[2]);
      const end = Number(mer[3]);
      if (Number.isFinite(start) && Number.isFinite(end) && end >= start && end - start <= 60) {
        for (let n = start; n <= end; n += 1) registerEq(`${ch}-${n}`);
      }
    }
    const reEqSingle = /式\s*[（(]?\s*(\d+-\d+'?)\s*[）)]?/g;
    let mes;
    while ((mes = reEqSingle.exec(itemRef)) !== null) {
      registerEq(mes[1]);
    }
  }

  return { byId, mathEx, ctrlEx, exercises, tables, equations };
}

function resolveRefToken(token, refIndex, currentItem) {
  if (!refIndex || !token) return null;
  const clean = token.trim();
  let target = null;

  // Direct card ID (e.g. ctrl-ch02-e-34)
  if (refIndex.byId.has(clean)) {
    target = refIndex.byId.get(clean);
  }
  // Math I example: (基础|强化)?例 1.12
  else if (/^(?:基础\s*|强化\s*)?例\s*\d+\.\d+$/.test(clean)) {
    const m = clean.match(/^(基础|强化)?\s*例\s*(\d+\.\d+)$/);
    if (m) {
      const prefix = m[1];
      const num = m[2];
      if (prefix === '强化') target = refIndex.mathEx.get(`i:${num}`);
      else if (prefix === '基础') target = refIndex.mathEx.get(`f:${num}`);
      else {
        const curStage = currentItem?.stage === 'intensive' ? 'i' : 'f';
        target = refIndex.mathEx.get(`f:${num}`) || refIndex.mathEx.get(`${curStage}:${num}`) || refIndex.mathEx.get(`any:${num}`);
      }
    }
  }
  // Control example: (主教材|主|学习辅导|辅导|辅)?例 2-6 / 2-1a
  else if (/^(?:主教材\s*|学习辅导\s*|辅导\s*|主|辅)?例\s*\d+-\d+[a-z]?$/i.test(clean)) {
    const m = clean.match(/^(主教材|学习辅导|辅导|主|辅)?\s*例\s*(\d+-\d+[a-z]?)$/i);
    if (m) {
      const prefix = m[1] || '';
      const code = m[2].toLowerCase();
      if (prefix === '主教材' || prefix === '主') {
        target = refIndex.ctrlEx.get(`main:${code}`);
      } else if (prefix === '学习辅导' || prefix === '辅导' || prefix === '辅') {
        target = refIndex.ctrlEx.get(`guide:${code}`);
      } else {
        const curIsGuide = /学习辅导/.test(String(currentItem?.source?.book || '')) || /学习辅导/.test(String(currentItem?.title || ''));
        const curIsMain = /自动控制原理/.test(String(currentItem?.source?.book || '')) || /主教材/.test(String(currentItem?.title || ''));
        if (curIsGuide) {
          target = refIndex.ctrlEx.get(`guide:${code}`) || refIndex.ctrlEx.get(`main:${code}`);
        } else if (curIsMain) {
          target = refIndex.ctrlEx.get(`main:${code}`) || refIndex.ctrlEx.get(`guide:${code}`);
        } else {
          target = refIndex.ctrlEx.get(`any:${code}`);
        }
      }
    }
  }
  // Exercise: 习题 2-5
  else if (/^(?:课后)?习题\s*\d+-\d+$/.test(clean)) {
    const m = clean.match(/习题\s*(\d+-\d+)$/);
    if (m) target = refIndex.exercises.get(`ex:${m[1]}`);
  }
  // Exercise code: A2-5 / B2-23
  else if (/^[AB]\d+-\d+$/.test(clean)) {
    target = refIndex.exercises.get(`code:${clean}`);
  }
  // Table: (主教材|辅导)?表 2-1
  else if (/^(?:主教材\s*|辅导\s*)?表\s*\d+-\d+$/.test(clean)) {
    const m = clean.match(/^(主教材|辅导)?\s*表\s*(\d+-\d+)$/);
    if (m) {
      if (m[1] === '辅导') target = refIndex.tables.get(`guide:${m[2]}`);
      else if (m[1] === '主教材') target = refIndex.tables.get(`main:${m[2]}`);
      else target = refIndex.tables.get(`main:${m[2]}`) || refIndex.tables.get(`any:${m[2]}`);
    }
  }
  // Equation: 式(2-62) / 式 1-3
  else if (/^式\s*[（(]?\s*\d+-\d+'?\s*[）)]?$/.test(clean)) {
    const m = clean.match(/式\s*[（(]?\s*(\d+-\d+'?)\s*[）)]?$/);
    if (m) target = refIndex.equations.get(`eq:${m[1]}`) || refIndex.equations.get(`eq:${m[1].replace(/'$/, '')}`);
  }

  if (target && currentItem && target.id === currentItem.id) return null;
  return target || null;
}

const MATH_SPLIT_RE = /(\$\$[\s\S]*?\$\$|\$(?!\s)[^$\n]+?\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\])/g;
const REF_TOKEN_RE = /`((?:ctrl|m1)-[a-z0-9-]+)`|(?:主教材\s*|学习辅导\s*|辅导\s*|主|辅|基础\s*|强化\s*)?例\s*\d+(?:\.\d+|-\d+[a-z]?)(?!\d)|(?:课后)?习题\s*\d+-\d+|\b[AB]\d+-\d+\b|(?:主教材\s*|辅导\s*)?表\s*\d+-\d+|式\s*[（(]?\s*\d+-\d+'?\s*[）)]?/gi;

function linkRefsInProseHtml(html, refCtx) {
  if (!refCtx?.refIndex) return html;
  // Split by HTML tags so attributes/tags are never altered
  return html.split(/(<[^>]+>)/g).map((seg, idx) => {
    if (idx % 2 === 1) return seg;
    return seg.replace(REF_TOKEN_RE, (full, backtickId) => {
      const lookup = backtickId || full;
      const target = resolveRefToken(lookup, refCtx.refIndex, refCtx.currentItem);
      if (!target) return full;
      const plainTitle = String(target.title || '').replace(/\$/g, '');
      const kindLabel = KIND_META[target.kind]?.label || '关联卡片';
      const label = backtickId ? `查看 ${plainTitle.slice(0, 18)}${plainTitle.length > 18 ? '…' : ''}` : full;
      return `<button type="button" class="stem-ref-link" data-ref-id="${escapeHtml(target.id)}" title="侧边浮窗预览（${escapeHtml(kindLabel)}）：${escapeHtml(plainTitle)}">${label}</button>`;
    });
  }).join('');
}

/**
 * Escapes HTML and converts lightweight **bold** emphasis + newlines,
 * leaving LaTeX math delimiters ($...$, $$...$$) intact for KaTeX.
 * When refCtx is supplied, turns verified citations in prose into side-drawer triggers.
 */
function renderRichText(raw, refCtx = null) {
  const str = String(raw ?? '');
  const mathSlots = [];
  const shielded = str.replace(MATH_SPLIT_RE, (m) => {
    const idx = mathSlots.length;
    mathSlots.push(escapeHtml(m));
    return `\uE000M${idx}\uE001`;
  });
  let html = escapeHtml(shielded)
    .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
  if (refCtx?.refIndex) {
    html = linkRefsInProseHtml(html, refCtx);
  }
  return html.replace(/\uE000M(\d+)\uE001/g, (_, i) => mathSlots[Number(i)] ?? '');
}

/**
 * Renders compact relatedRefs strings in summary cards (e.g. "主例2-1~2-7, 2-10, 习题2-21, B2-23")
 * into clickable reference chips where each token resolves to a real card in the chapter.
 */
function renderRelatedRefs(rawRefs, refCtx) {
  const text = String(rawRefs || '').trim();
  if (!text) return '';
  if (!refCtx?.refIndex) return escapeHtml(text);

  let activePrefix = '';
  const parts = text.split(/([,，、]\s*)/);
  return parts.map((part, idx) => {
    if (idx % 2 === 1) return escapeHtml(part);
    const tok = part.trim();
    if (!tok) return '';
    // Update carried prefix like 主例 / 辅例 / 强化例 / 基础例 / 习题 / 例 / 式
    const pm = tok.match(/^(主教材\s*|学习辅导\s*|辅导\s*|主|辅|强化\s*|基础\s*)?(例|习题|表|式)/);
    if (pm) {
      activePrefix = (pm[1] || '') + pm[2];
    } else if (/^[AB]\d+-\d+/.test(tok)) {
      activePrefix = '';
    }
    // Extract first concrete target from a token or range (e.g. "主例2-1~2-7" -> "主例2-1", "2-15" -> activePrefix + "2-15")
    let candidate = tok.split(/[~～]/)[0].trim();
    if (/^\d+(?:\.\d+|-\d+[a-z]?)$/i.test(candidate) && activePrefix) {
      candidate = `${activePrefix}${candidate}`;
    }
    const target = resolveRefToken(candidate, refCtx.refIndex, refCtx.currentItem);
    if (!target) return escapeHtml(tok);
    const plainTitle = String(target.title || '').replace(/\$/g, '');
    const kindLabel = KIND_META[target.kind]?.label || '关联卡片';
    return `<button type="button" class="stem-ref-link stem-ref-chip" data-ref-id="${escapeHtml(target.id)}" title="侧边浮窗预览（${escapeHtml(kindLabel)}）：${escapeHtml(plainTitle)}">${escapeHtml(tok)}</button>`;
  }).join('');
}

function scopeDrawerSvgs(svgStr) {
  let out = String(svgStr || '');
  const markerIds = [];
  out.replace(/<marker\s+[^>]*id=(['"])([^'"]+)\1/gi, (_, _q, id) => {
    markerIds.push(id);
    return _;
  });
  for (const id of markerIds) {
    const safeId = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out
      .replace(new RegExp(`id=(['"])${safeId}\\1`, 'g'), `id=$1drw_${id}$1`)
      .replace(new RegExp(`url\\(#${safeId}\\)`, 'g'), `url(#drw_${id})`);
  }
  return out;
}

// Runtime probe for discrete `display` transition support (modern-web-guidance: animate-to-from-top-layer)
let supportsDisplayTransition;
function canTransitionDisplay() {
  if (supportsDisplayTransition !== undefined) return supportsDisplayTransition;
  if (!window.CSS?.supports?.('transition-behavior', 'allow-discrete') || !document.body) {
    return false;
  }
  const probe = document.createElement('div');
  probe.style.cssText = 'transition: display 1s allow-discrete !important; display: block;';
  document.body.appendChild(probe);
  getComputedStyle(probe).display;
  probe.style.display = 'none';
  supportsDisplayTransition = getComputedStyle(probe).display === 'block';
  probe.remove();
  return supportsDisplayTransition;
}

function supportsTopLayerExit() {
  return Boolean(window.CSS?.supports?.('overlay', 'auto')) && canTransitionDisplay();
}

async function closeTopLayerDialog(dialog) {
  if (!dialog || !dialog.open) return;
  if (!supportsTopLayerExit()) {
    dialog.setAttribute('data-closing', '');
    const animations = dialog.getAnimations ? dialog.getAnimations({ subtree: true }) : [];
    if (animations.length > 0) {
      await Promise.race([
        Promise.allSettled(animations.map(a => a.finished)),
        new Promise(r => setTimeout(r, 450)),
      ]);
    }
    if (!dialog.hasAttribute('data-closing')) return;
    dialog.removeAttribute('data-closing');
  }
  if (dialog.open) dialog.close();
}

// Shared singleton side-preview drawer across all STEM views
let sharedRefDrawer = null;
let activeDrawerOwner = null;

function getSharedRefDrawer() {
  if (sharedRefDrawer && document.body.contains(sharedRefDrawer)) return sharedRefDrawer;
  const dialog = document.createElement('dialog');
  dialog.id = 'stemRefDrawer';
  dialog.className = 'stem-ref-drawer no-print';
  dialog.setAttribute('closedby', 'any');
  dialog.setAttribute('aria-labelledby', 'stemRefDrawerTitle');
  dialog.innerHTML = `
    <div class="stem-ref-drawer-shell">
      <header class="stem-ref-drawer-header">
        <div class="stem-ref-drawer-head-left">
          <button type="button" class="btn btn-subtle stem-ref-back-btn" data-drawer-action="back" hidden>← 返回上张</button>
          <div class="stem-ref-drawer-titles">
            <span class="stem-ref-drawer-kicker" id="stemRefDrawerKicker">侧边浮窗预览 · 原阅读位置已保留</span>
            <h2 class="stem-ref-drawer-title" id="stemRefDrawerTitle">关联卡片预览</h2>
          </div>
        </div>
        <div class="stem-ref-drawer-head-actions">
          <button type="button" class="btn btn-subtle stem-ref-locate-btn" data-drawer-action="locate">去原分区定位 ↗</button>
          <button type="button" class="icon-btn stem-ref-close-btn" data-drawer-action="close" aria-label="关闭侧边预览浮窗">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
            </svg>
          </button>
        </div>
      </header>
      <div class="stem-ref-drawer-body" id="stemRefDrawerBody"></div>
    </div>
  `;
  document.body.appendChild(dialog);

  // Route native cancel (Esc / closedby light dismiss) through top-layer exit helper
  dialog.addEventListener('cancel', (event) => {
    event.preventDefault();
    closeTopLayerDialog(dialog);
  });

  // Mandatory light-dismiss fallback for browsers without closedBy (Safari)
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    dialog.addEventListener('click', (event) => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const inside = (
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width
      );
      if (!inside) closeTopLayerDialog(dialog);
    });
  }

  dialog.addEventListener('close', () => {
    dialog.removeAttribute('data-closing');
    if (activeDrawerOwner?.onDrawerClosed) activeDrawerOwner.onDrawerClosed();
  });

  dialog.addEventListener('click', (event) => {
    if (!activeDrawerOwner) return;
    const dBtn = event.target.closest('[data-drawer-action]');
    if (dBtn) {
      const act = dBtn.dataset.drawerAction;
      if (act === 'close') closeTopLayerDialog(dialog);
      else if (act === 'back') activeDrawerOwner.drawerBack();
      else if (act === 'locate') activeDrawerOwner.drawerLocate();
      return;
    }
    const refBtn = event.target.closest('[data-ref-id]');
    if (refBtn) {
      activeDrawerOwner.openRefDrawer(refBtn.dataset.refId, { push: true });
      return;
    }
    const actionBtn = event.target.closest('[data-action]');
    if (actionBtn) {
      activeDrawerOwner.handleCardAction(actionBtn.dataset.action, actionBtn.dataset.id, { fromDrawer: true });
    }
  });

  sharedRefDrawer = dialog;
  return dialog;
}

function extractSvgTitle(svgStr, fallback) {
  const m = String(svgStr || '').match(/<title>([\s\S]*?)<\/title>/i);
  if (!m) return fallback;
  return m[1]
    .trim()
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&amp;/g, '&');
}

function renderFigureBlock(item, { inDrawer = false } = {}) {
  const rawSvg = String(item.figureSvg || '').trim();
  if (!rawSvg || !rawSvg.startsWith('<svg')) return '';
  const svg = inDrawer ? scopeDrawerSvgs(rawSvg) : rawSvg;
  const caption = extractSvgTitle(rawSvg, `${item.source?.itemRef || '示意图'}`);
  return `
    <figure class="stem-figure">
      <div class="stem-svg-stage">${svg}</div>
      <figcaption class="stem-figcaption">${escapeHtml(caption)}</figcaption>
    </figure>
  `;
}

function renderTablesBlock(item, refCtx = null) {
  const tables = Array.isArray(item.tables) ? item.tables : [];
  if (!tables.length) return '';
  const idPrefix = refCtx?.inDrawer ? 'drw-tbl-cap-' : 'tbl-cap-';
  return tables.map((tbl, idx) => {
    const capId = `${idPrefix}${String(item.id || 'item').replace(/[^a-zA-Z0-9_-]/g, '')}-${idx}`;
    const headers = Array.isArray(tbl.headers) ? tbl.headers : [];
    const rows = Array.isArray(tbl.rows) ? tbl.rows : [];
    const useRowHeader = tbl.firstColHeader !== false;
    return `
      <div tabindex="0" role="region" aria-labelledby="${capId}" class="stem-table-wrapper">
        <table class="stem-data-table">
          <caption id="${capId}">${renderRichText(tbl.caption || '对照表')}</caption>
          <thead>
            <tr>
              ${headers.map(h => `<th scope="col">${renderRichText(h)}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${rows.map(row => `
              <tr>
                ${row.map((cell, ci) =>
                  (ci === 0 && useRowHeader)
                    ? `<th scope="row">${renderRichText(cell, refCtx)}</th>`
                    : `<td>${renderRichText(cell, refCtx)}</td>`
                ).join('')}
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `;
  }).join('');
}

function buildSearchHaystack(item) {
  const parts = [
    item.id,
    item.stage ? STAGE_META[item.stage]?.label : '',
    item.section,
    item.title,
    item.prompt,
    item.formulaLatex,
    item.summary,
    item.overview,
    item.conditions,
    item.pitfalls,
    item.problem,
    item.finalAnswer,
    item.takeaway,
    item.source?.book,
    item.source?.printedPage,
    item.source?.itemRef,
    item.source?.crossRef,
  ];
  if (Array.isArray(item.symbols)) {
    for (const s of item.symbols) {
      parts.push(s.sym, s.desc);
    }
  }
  if (Array.isArray(item.steps)) {
    for (const st of item.steps) {
      parts.push(st.stepTitle, st.detail);
    }
  }
  if (Array.isArray(item.tables)) {
    for (const tbl of item.tables) {
      parts.push(tbl.caption, ...(tbl.headers || []));
      for (const row of tbl.rows || []) parts.push(...row);
    }
  }
  if (Array.isArray(item.patterns)) {
    for (const p of item.patterns) {
      parts.push(p.typeName, p.triggers, p.method, p.pitfall, p.relatedRefs);
    }
  }
  return parts.filter(Boolean).join(' ').toLowerCase();
}

export function initStemSubjectView(store, { subjectId, containerId }) {
  const root = document.getElementById(containerId);
  const state = {
    loaded: false,
    loading: false,
    error: null,
    manifest: null,
    chapterCache: new Map(),
    activeGroupId: null,
    activeChapterId: null,
    kindFilter: 'formula', // formula | conclusion | example | exercise | summary
    stageFilter: 'foundation', // foundation | intensive (no 'all')
    searchQuery: '',
    maskMode: false,
    starOnly: false,
    hideMastered: false,
    flaggedOnly: false,
    coreOnly: false,
    revealedIds: new Set(),
    openExampleIds: new Set(),
    drawerStack: [],
    returnBookmark: null,
  };

  async function loadManifest() {
    if (state.manifest) return state.manifest;
    const res = await fetch(`./data/${subjectId}/manifest.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.manifest = await res.json();
    const firstReady = (state.manifest.chapters || []).find(ch => ch.ready);
    const initialCh = firstReady || state.manifest.chapters?.[0] || null;
    state.activeChapterId = initialCh?.id || null;
    if (Array.isArray(state.manifest.groups) && state.manifest.groups.length) {
      state.activeGroupId = initialCh?.group || state.manifest.groups[0].id;
    }
    return state.manifest;
  }

  async function loadChapter(chapterMeta) {
    if (!chapterMeta || !chapterMeta.ready) return null;
    if (state.chapterCache.has(chapterMeta.id)) {
      return state.chapterCache.get(chapterMeta.id);
    }
    const res = await fetch(`./data/${subjectId}/${chapterMeta.file}`);
    if (!res.ok) throw new Error(`加载章节 ${chapterMeta.id} 失败（HTTP ${res.status}）`);
    const data = await res.json();
    // Pre-build search index & citation cross-reference index per chapter
    for (const item of data.items || []) {
      item._haystack = buildSearchHaystack(item);
    }
    data._refIndex = buildChapterRefIndex(data.items || []);
    state.chapterCache.set(chapterMeta.id, data);
    return data;
  }

  function getActiveChapterMeta() {
    return (state.manifest?.chapters || []).find(c => c.id === state.activeChapterId) || null;
  }

  function getSubjectKinds() {
    const kinds = state.manifest?.kinds;
    if (Array.isArray(kinds) && kinds.length) {
      return kinds.filter(k => Boolean(KIND_META[k]));
    }
    return DEFAULT_KINDS;
  }

  function hasStageSplit(kindItems) {
    return kindItems.some(i => i.stage === 'foundation' || i.stage === 'intensive');
  }

  function ensureValidFilters(chapterData) {
    const kinds = getSubjectKinds();
    if (!kinds.includes(state.kindFilter)) {
      state.kindFilter = kinds[0] || 'formula';
    }
    const kindItems = (chapterData?.items || []).filter(i => i.kind === state.kindFilter);
    if (hasStageSplit(kindItems)) {
      const matchCur = kindItems.some(i =>
        state.stageFilter === 'foundation'
          ? (i.stage === 'foundation' || i.stage === 'both')
          : (i.stage === 'intensive' || i.stage === 'both')
      );
      if (!matchCur) {
        state.stageFilter = state.stageFilter === 'foundation' ? 'intensive' : 'foundation';
      }
    }
  }

  function getFilteredItems(chapterData) {
    const items = chapterData?.items || [];
    const q = state.searchQuery.trim().toLowerCase();
    const kindItems = items.filter(i => i.kind === state.kindFilter);
    const applyStage = hasStageSplit(kindItems);
    return kindItems.filter(item => {
      if (applyStage && item.stage) {
        if (state.stageFilter === 'foundation' && item.stage !== 'foundation' && item.stage !== 'both') return false;
        if (state.stageFilter === 'intensive' && item.stage !== 'intensive' && item.stage !== 'both') return false;
      }
      if (state.coreOnly && item.examWeight === 'extended') return false;
      if (state.starOnly && !store.stemStarred.has(item.id)) return false;
      if (state.hideMastered && store.stemMastered.has(item.id)) return false;
      if (state.flaggedOnly && !store.stemFlagged.has(item.id)) return false;
      if (q && !item._haystack.includes(q)) return false;
      return true;
    });
  }

  function renderVerificationBadge(item) {
    const v = item.verification;
    if (!v || !v.verified) return '';
    if (v.method === 'sympy') {
      return `<span class="stem-badge stem-badge-sympy">✓ SymPy 已验算</span>`;
    }
    return `<span class="stem-badge stem-badge-checked">✓ 教材原文核对</span>`;
  }

  function renderCardHeader(item) {
    const meta = KIND_META[item.kind] || { label: item.kind, badgeClass: '' };
    const isStarred = store.stemStarred.has(item.id);
    const isMastered = store.stemMastered.has(item.id);
    const isFlagged = store.stemFlagged.has(item.id);
    const weightBadge = item.examWeight === 'extended'
      ? `<span class="stem-badge stem-badge-ext">教材拓展</span>`
      : `<span class="stem-badge stem-badge-core">★ 考纲重点</span>`;
    const stageMeta = item.stage ? STAGE_META[item.stage] : null;
    const stageBadge = stageMeta
      ? `<span class="stem-badge ${stageMeta.badgeClass}">${stageMeta.label}</span>`
      : '';
    const plainTitle = String(item.title || '').replace(/\$/g, '');

    return `
      <div class="stem-card-top">
        <div class="stem-card-meta">
          <span class="stem-badge ${meta.badgeClass}">${meta.label}</span>
          ${stageBadge}
          <span class="stem-section-tag">${escapeHtml(item.section)}</span>
          ${weightBadge}
          ${renderVerificationBadge(item)}
        </div>
        <div class="stem-card-actions no-print">
          <button
            type="button"
            class="icon-btn ${isStarred ? 'is-starred' : ''}"
            data-action="star"
            data-id="${escapeHtml(item.id)}"
            aria-pressed="${isStarred}"
            aria-label="${isStarred ? '取消重点标记' : '标记为重点'}：${escapeHtml(plainTitle)}"
          >
            <svg class="icon" viewBox="0 0 24 24" fill="${isStarred ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/>
            </svg>
          </button>
          <button
            type="button"
            class="icon-btn ${isMastered ? 'is-mastered' : ''}"
            data-action="master"
            data-id="${escapeHtml(item.id)}"
            aria-pressed="${isMastered}"
            aria-label="${isMastered ? '取消已掌握标记' : '标记为已掌握'}：${escapeHtml(plainTitle)}"
          >
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
            </svg>
          </button>
          <button
            type="button"
            class="icon-btn flag-btn ${isFlagged ? 'is-flagged' : ''}"
            data-action="flag"
            data-id="${escapeHtml(item.id)}"
            aria-pressed="${isFlagged}"
            aria-label="${isFlagged ? '取消存疑待核标记' : '标记存疑待核对原书'}：${escapeHtml(plainTitle)}"
          >
            <svg class="icon" viewBox="0 0 24 24" fill="${isFlagged ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1zM4 22v-7"/>
            </svg>
          </button>
        </div>
      </div>
      <h3 class="stem-card-title">${renderRichText(item.title)}</h3>
    `;
  }

  function renderCitationBar(item) {
    const src = item.source || {};
    const crossHtml = src.crossRef
      ? `<span class="stem-cite-sep" aria-hidden="true">｜</span><span class="stem-cite-cross">对照：${escapeHtml(src.crossRef)}</span>`
      : '';
    const testTag = item.verification?.testId
      ? `<code class="stem-verify-code">${escapeHtml(item.verification.testId)}</code>`
      : '';
    return `
      <footer class="stem-citation-bar">
        <div class="stem-cite-main">
          <span class="stem-cite-label">出处</span>
          <cite>${escapeHtml(src.book || '')}</cite>
          <span class="stem-cite-page">${escapeHtml(src.printedPage || '')}</span>
          <span class="stem-cite-ref">${escapeHtml(src.itemRef || '')}</span>
          ${crossHtml}
        </div>
        ${testTag}
      </footer>
    `;
  }

  function renderFormulaBody(item, { refCtx = null, inDrawer = false } = {}) {
    const masked = !inDrawer && state.maskMode && !state.revealedIds.has(item.id);
    const symbolsHtml = Array.isArray(item.symbols) && item.symbols.length
      ? `
        <dl class="stem-symbols-list">
          ${item.symbols.map(s => `
            <div class="stem-symbol-row">
              <dt>${renderRichText(s.sym)}</dt>
              <dd>${renderRichText(s.desc, refCtx)}</dd>
            </div>
          `).join('')}
        </dl>
      `
      : '';

    return `
      ${item.prompt ? `<p class="stem-prompt">${renderRichText(item.prompt, refCtx)}</p>` : ''}
      <div class="stem-formula-box ${masked ? 'is-masked' : ''}">
        <div class="math-scroll ${masked ? 'is-blurred' : ''}">${escapeHtml(item.formulaLatex)}</div>
        ${masked ? `
          <div class="stem-mask-overlay no-print">
            <button type="button" class="btn btn-primary stem-reveal-btn" data-action="reveal" data-id="${escapeHtml(item.id)}">翻开核对公式</button>
          </div>
        ` : (!inDrawer && state.maskMode ? `
          <div class="stem-rehide-row no-print">
            <button type="button" class="btn-link-muted" data-action="reveal" data-id="${escapeHtml(item.id)}">重新遮罩</button>
          </div>
        ` : '')}
      </div>
      ${renderFigureBlock(item, { inDrawer })}
      ${renderTablesBlock(item, refCtx)}
      ${symbolsHtml}
      <div class="stem-callouts">
        ${item.conditions ? `
          <div class="stem-callout callout-cond">
            <span class="callout-tag">适用前提</span>
            <p>${renderRichText(item.conditions, refCtx)}</p>
          </div>
        ` : ''}
        ${item.pitfalls ? `
          <div class="stem-callout callout-pitfall">
            <span class="callout-tag">易错红线 / 勘误</span>
            <p>${renderRichText(item.pitfalls, refCtx)}</p>
          </div>
        ` : ''}
      </div>
    `;
  }

  function renderConclusionBody(item, { refCtx = null, inDrawer = false } = {}) {
    const masked = !inDrawer && state.maskMode && !state.revealedIds.has(item.id);
    return `
      <div class="stem-conclusion-box ${masked ? 'is-masked' : ''}">
        <div class="stem-conclusion-text ${masked ? 'is-blurred' : ''}">${renderRichText(item.summary, refCtx)}</div>
        ${masked ? `
          <div class="stem-mask-overlay no-print">
            <button type="button" class="btn btn-primary stem-reveal-btn" data-action="reveal" data-id="${escapeHtml(item.id)}">翻开核对结论</button>
          </div>
        ` : (!inDrawer && state.maskMode ? `
          <div class="stem-rehide-row no-print">
            <button type="button" class="btn-link-muted" data-action="reveal" data-id="${escapeHtml(item.id)}">重新遮罩</button>
          </div>
        ` : '')}
      </div>
      ${renderFigureBlock(item, { inDrawer })}
      ${renderTablesBlock(item, refCtx)}
      <div class="stem-callouts">
        ${item.conditions ? `
          <div class="stem-callout callout-cond">
            <span class="callout-tag">适用范围</span>
            <p>${renderRichText(item.conditions, refCtx)}</p>
          </div>
        ` : ''}
        ${item.pitfalls ? `
          <div class="stem-callout callout-pitfall">
            <span class="callout-tag">高频坑点</span>
            <p>${renderRichText(item.pitfalls, refCtx)}</p>
          </div>
        ` : ''}
      </div>
    `;
  }

  function renderWorkedProblemBody(item, { isExercise = false, refCtx = null, inDrawer = false } = {}) {
    const isOpen = inDrawer || state.openExampleIds.has(item.id);
    const steps = item.steps || [];
    const problemLabel = isExercise ? '课后习题题干' : '题目条件';
    const summaryLabel = isExercise
      ? `展开课后习题详解与最终答案（共 ${steps.length} 步）`
      : `展开分步详解与最终答案（共 ${steps.length} 步）`;
    return `
      <div class="stem-problem-box ${isExercise ? 'stem-exercise-problem-box' : ''}">
        <span class="stem-problem-label">${problemLabel}</span>
        <div class="stem-problem-text">${renderRichText(item.problem, refCtx)}</div>
      </div>
      ${renderFigureBlock(item, { inDrawer })}
      ${renderTablesBlock(item, refCtx)}
      <details class="stem-solution-details ${isExercise ? 'stem-exercise-details' : ''}" data-example-id="${escapeHtml(item.id)}" ${isOpen ? 'open' : ''}>
        <summary class="stem-solution-summary">
          <span class="stem-summary-label">${summaryLabel}</span>
          <span class="disclosure-chevron" aria-hidden="true"></span>
        </summary>
        <div class="stem-solution-body">
          <ol class="stem-steps-list">
            ${steps.map(st => `
              <li class="stem-step-item">
                <div class="stem-step-title">${renderRichText(String(st.stepTitle || '').replace(/^\d+\.\s*/, ''), refCtx)}</div>
                <div class="stem-step-detail">${renderRichText(st.detail, refCtx)}</div>
              </li>
            `).join('')}
          </ol>
          <div class="stem-final-answer">
            <span class="callout-tag">最终答案</span>
            <div>${renderRichText(item.finalAnswer, refCtx)}</div>
          </div>
          ${item.takeaway ? `
            <div class="stem-callout callout-pitfall stem-takeaway-box">
              <span class="callout-tag">考场提炼</span>
              <p>${renderRichText(item.takeaway, refCtx)}</p>
            </div>
          ` : ''}
        </div>
      </details>
    `;
  }

  function renderSummaryBody(item, { refCtx = null, inDrawer = false } = {}) {
    const overviewText = item.overview || item.summary || '';
    const patterns = Array.isArray(item.patterns) ? item.patterns : [];
    return `
      ${overviewText ? `
        <div class="stem-summary-overview">
          <div class="stem-summary-overview-text">${renderRichText(overviewText, refCtx)}</div>
        </div>
      ` : ''}
      ${renderFigureBlock(item, { inDrawer })}
      ${renderTablesBlock(item, refCtx)}
      ${patterns.length ? `
        <div class="stem-pattern-list">
          ${patterns.map((p, idx) => `
            <div class="stem-pattern-card">
              <div class="stem-pattern-head">
                <span class="stem-pattern-index">题型 ${idx + 1}</span>
                <h4 class="stem-pattern-title">${renderRichText(p.typeName)}</h4>
                ${p.relatedRefs ? `<span class="stem-pattern-refs">对应：${renderRelatedRefs(p.relatedRefs, refCtx)}</span>` : ''}
              </div>
              <div class="stem-pattern-grid">
                <div class="stem-pattern-cell">
                  <span class="callout-tag tone-indigo">识别信号</span>
                  <p>${renderRichText(p.triggers, refCtx)}</p>
                </div>
                <div class="stem-pattern-cell">
                  <span class="callout-tag tone-emerald">标准破题步骤</span>
                  <p>${renderRichText(p.method, refCtx)}</p>
                </div>
                ${p.pitfall ? `
                  <div class="stem-pattern-cell stem-pattern-pitfall">
                    <span class="callout-tag tone-amber">避坑要点</span>
                    <p>${renderRichText(p.pitfall, refCtx)}</p>
                  </div>
                ` : ''}
              </div>
            </div>
          `).join('')}
        </div>
      ` : ''}
      ${item.takeaway ? `
        <div class="stem-callout callout-pitfall">
          <span class="callout-tag">考场总口诀</span>
          <p>${renderRichText(item.takeaway, refCtx)}</p>
        </div>
      ` : ''}
    `;
  }

  function renderItemCard(item, { refIndex = null, inDrawer = false } = {}) {
    const isMastered = store.stemMastered.has(item.id);
    const isFlagged = store.stemFlagged.has(item.id);
    const refCtx = refIndex ? { refIndex, currentItem: item, inDrawer } : null;
    let bodyHtml = '';
    if (item.kind === 'formula') bodyHtml = renderFormulaBody(item, { refCtx, inDrawer });
    else if (item.kind === 'conclusion') bodyHtml = renderConclusionBody(item, { refCtx, inDrawer });
    else if (item.kind === 'example') bodyHtml = renderWorkedProblemBody(item, { isExercise: false, refCtx, inDrawer });
    else if (item.kind === 'exercise') bodyHtml = renderWorkedProblemBody(item, { isExercise: true, refCtx, inDrawer });
    else if (item.kind === 'summary') bodyHtml = renderSummaryBody(item, { refCtx, inDrawer });

    return `
      <article class="card stem-card ${inDrawer ? 'stem-card-in-drawer' : ''} stem-kind-${escapeHtml(item.kind)} ${isMastered ? 'is-mastered' : ''} ${isFlagged ? 'is-flagged' : ''}" data-item-id="${escapeHtml(item.id)}">
        ${renderCardHeader(item)}
        <div class="stem-card-body">${bodyHtml}</div>
        ${renderCitationBar(item)}
      </article>
    `;
  }

  function renderRoadmapPanel(chMeta) {
    const manifest = state.manifest;
    const readyChapters = (manifest.chapters || []).filter(c => c.ready);
    return `
      <section class="card stem-roadmap-card">
        <div class="stem-roadmap-top">
          <div>
            <div class="stem-card-meta">
              <span class="stem-badge stem-badge-core">${escapeHtml(chMeta.syllabusCode)}</span>
              <span class="stem-badge stem-badge-pending">待逐页录入</span>
              <span class="stem-section-tag">${escapeHtml(chMeta.pages)}</span>
            </div>
            <h2 class="stem-chapter-heading">${escapeHtml(chMeta.num)} · ${escapeHtml(chMeta.title)}</h2>
          </div>
          ${readyChapters.length ? `
            <button type="button" class="btn btn-primary" data-jump-chapter="${escapeHtml(readyChapters[0].id)}">
              先看已上线试点章（${escapeHtml(readyChapters[0].num)}）
            </button>
          ` : ''}
        </div>

        <div class="stem-callout callout-cond">
          <span class="callout-tag">2027 考纲 / 单元要求</span>
          <p>${escapeHtml(chMeta.syllabusReq)}</p>
        </div>

        <div class="stem-roadmap-meta">
          <div>
            <h3>核心教材来源</h3>
            <ul>
              ${(manifest.primaryBooks || []).map(b => `<li>${escapeHtml(b)}</li>`).join('')}
            </ul>
          </div>
          <div>
            <h3>辅助对照资料</h3>
            <ul>
              ${(manifest.referenceBooks || []).map(b => `<li>${escapeHtml(b)}</li>`).join('')}
            </ul>
          </div>
        </div>

        <p class="stem-roadmap-note">
          为坚持<strong>零幻觉 · 全页码溯源 · SymPy 符号验算</strong>原则，未逐页核对原书的章节绝不自动生成占位公式。待你验收当前试点章节的排版、公式与例题风格后，即可按计划分批补齐全部单元。
        </p>
      </section>
    `;
  }

  function renderEmptyListState(allItems) {
    const kindMeta = KIND_META[state.kindFilter] || { label: state.kindFilter };
    const totalInKind = allItems.filter(i => i.kind === state.kindFilter).length;
    if (totalInKind === 0) {
      return `
        <section class="card empty-state">
          <p class="empty-title">本章「${escapeHtml(kindMeta.label)}」尚未补齐</p>
          <p class="empty-desc">正严格对照原书逐页核读、SymPy 验算并按章补全中，可先切换上方其他分区浏览。</p>
        </section>
      `;
    }
    const q = state.searchQuery.trim().toLowerCase();
    let otherHitsHtml = '';
    if (q) {
      const kinds = getSubjectKinds();
      const otherHits = kinds
        .filter(k => k !== state.kindFilter)
        .map(k => ({
          key: k,
          label: KIND_META[k].label,
          cnt: allItems.filter(i => i.kind === k && i._haystack.includes(q)).length,
        }))
        .filter(x => x.cnt > 0);
      if (otherHits.length) {
        otherHitsHtml = `
          <div class="stem-empty-jumps">
            <span>其他分区有匹配结果：</span>
            ${otherHits.map(x => `
              <button type="button" class="module-pill" data-kind-filter="${escapeHtml(x.key)}">
                <span>${escapeHtml(x.label)}</span>
                <span class="count-badge">${x.cnt}</span>
              </button>
            `).join('')}
          </div>
        `;
      }
    }
    return `
      <section class="card empty-state">
        <p class="empty-title">当前「${escapeHtml(kindMeta.label)}」分区没有找到匹配内容</p>
        <p class="empty-desc">可以清空搜索词、切换复习阶段，或关闭「重点 / 隐藏已掌握 / 存疑待核 / 仅看考纲重点」筛选。</p>
        ${otherHitsHtml}
      </section>
    `;
  }

  function renderChapterContent(chMeta, chapterData) {
    if (!chMeta) return '';
    if (!chMeta.ready || !chapterData) {
      return renderRoadmapPanel(chMeta);
    }

    ensureValidFilters(chapterData);

    const allItems = chapterData.items || [];
    const kinds = getSubjectKinds();
    const kindItems = allItems.filter(i => i.kind === state.kindFilter);
    const filtered = getFilteredItems(chapterData);

    const kindCounts = {};
    for (const k of kinds) {
      kindCounts[k] = allItems.filter(i => i.kind === k).length;
    }

    const showStageBar = hasStageSplit(kindItems);
    const stageCounts = showStageBar ? {
      foundation: kindItems.filter(i => i.stage === 'foundation' || i.stage === 'both').length,
      intensive: kindItems.filter(i => i.stage === 'intensive' || i.stage === 'both').length,
    } : null;

    const canMask = state.kindFilter === 'formula' || state.kindFilter === 'conclusion';
    const extendedCount = allItems.filter(i => i.examWeight === 'extended').length;
    const starCount = allItems.filter(i => store.stemStarred.has(i.id)).length;
    const masteredCount = allItems.filter(i => store.stemMastered.has(i.id)).length;
    const flaggedCount = allItems.filter(i => store.stemFlagged.has(i.id)).length;
    const pct = allItems.length ? Math.round((masteredCount / allItems.length) * 100) : 0;

    const req = chapterData.basicRequirements || {};
    const searchId = `stemSearch-${subjectId}`;
    const syllabusShortLabel = state.manifest?.syllabusShortLabel
      || (subjectId === 'control' ? '2027 浙大 845 大纲要求' : '2027 考研数学一大纲要求');
    const searchPlaceholder = state.manifest?.searchPlaceholder
      || '搜索公式名、关键词、例题号（如 例4-8 / 劳斯 / 主导极点 / P.95）...';

    return `
      <!-- 章节大纲与教材要求 -->
      <section class="card stem-chapter-hero">
        <div class="stem-chapter-hero-head">
          <div>
            <div class="stem-card-meta">
              <span class="stem-badge stem-badge-core">${escapeHtml(chMeta.syllabusCode)}</span>
              <span class="stem-badge stem-badge-sympy">已核对上线 · ${allItems.length} 条</span>
              <span class="stem-section-tag">${escapeHtml(chMeta.pages)}</span>
            </div>
            <h2 class="stem-chapter-heading">${escapeHtml(chapterData.title || `${chMeta.num} ${chMeta.title}`)}</h2>
            <p class="stem-source-line">${escapeHtml(chapterData.sourceMeta || '')}</p>
          </div>
        </div>

        <details class="stem-req-details">
          <summary class="disclosure-summary stem-req-summary">
            <span class="disclosure-title">
              <span class="badge-chip badge-sky">考纲对照</span>
              2027 考纲要求 &amp; 教材「掌握 / 理解 / 了解」分级说明
            </span>
            <span class="disclosure-chevron" aria-hidden="true"></span>
          </summary>
          <div class="stem-req-grid">
            <div class="stem-req-cell">
              <div class="stem-req-label tone-rose">${escapeHtml(syllabusShortLabel)}</div>
              <p>${escapeHtml(chMeta.syllabusReq)}</p>
            </div>
            ${req.master ? `
              <div class="stem-req-cell">
                <div class="stem-req-label tone-indigo">教材要求 · 掌握</div>
                <p>${escapeHtml(req.master)}</p>
              </div>
            ` : ''}
            ${req.understand ? `
              <div class="stem-req-cell">
                <div class="stem-req-label tone-emerald">教材要求 · 理解</div>
                <p>${escapeHtml(req.understand)}</p>
              </div>
            ` : ''}
            ${req.familiar ? `
              <div class="stem-req-cell">
                <div class="stem-req-label tone-amber">教材要求 · 了解</div>
                <p>${escapeHtml(req.familiar)}</p>
              </div>
            ` : ''}
          </div>
        </details>
      </section>

      <!-- 搜索与分区切换控制栏 -->
      <section class="card control-panel no-print">
        <div class="module-filter-bar stem-kind-tabs" role="group" aria-label="按单元内容分区切换">
          ${kinds.map(key => {
            const meta = KIND_META[key];
            return `
              <button type="button" class="module-pill stem-kind-tab" data-kind-filter="${key}" aria-pressed="${state.kindFilter === key}">
                <span>${meta.label}</span>
                <span class="count-badge">${kindCounts[key] || 0}</span>
              </button>
            `;
          }).join('')}
        </div>

        ${showStageBar ? `
          <div class="module-filter-bar stem-stage-filter-bar" role="group" aria-label="按复习阶段切换">
            ${[
              { key: 'foundation', label: '基础 30 讲', count: stageCounts.foundation },
              { key: 'intensive', label: '强化 18 讲', count: stageCounts.intensive },
            ].map(s => `
              <button type="button" class="module-pill stem-stage-pill" data-stage-filter="${s.key}" aria-pressed="${state.stageFilter === s.key}">
                <span>${s.label}</span>
                <span class="count-badge">${s.count}</span>
              </button>
            `).join('')}
          </div>
        ` : ''}

        <search class="control-top-row">
          <div class="search-field">
            <label for="${searchId}" class="visually-hidden">搜索本分区公式、结论、例题或原书页码</label>
            <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
            </svg>
            <input id="${searchId}" class="stem-search-input" type="search" value="${escapeHtml(state.searchQuery)}" placeholder="${escapeHtml(searchPlaceholder)}">
            <button type="button" class="icon-btn clear-search-btn" data-action="clear-search" ${state.searchQuery ? '' : 'hidden'} aria-label="清空搜索关键词">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>

          <div class="toolbar-controls">
            ${canMask ? `
              <button type="button" class="pill-toggle" data-toggle="maskMode" aria-pressed="${state.maskMode}">
                <span>默写遮罩：${state.maskMode ? '开' : '关'}</span>
              </button>
            ` : ''}
            <button type="button" class="pill-toggle pill-amber" data-toggle="starOnly" aria-pressed="${state.starOnly}">
              <span>重点</span>
              <span class="count-badge">${starCount}</span>
            </button>
            <button type="button" class="pill-toggle pill-emerald" data-toggle="hideMastered" aria-pressed="${state.hideMastered}">
              <span>隐藏已掌握</span>
              <span class="count-badge">${masteredCount}</span>
            </button>
            <button type="button" class="pill-toggle pill-rose" data-toggle="flaggedOnly" aria-pressed="${state.flaggedOnly}">
              <span>存疑待核</span>
              <span class="count-badge">${flaggedCount}</span>
            </button>
            ${extendedCount > 0 ? `
              <button type="button" class="pill-toggle pill-indigo" data-toggle="coreOnly" aria-pressed="${state.coreOnly}">
                <span>仅看考纲重点</span>
              </button>
            ` : ''}
          </div>
        </search>

        <div class="stats-footer">
          <div class="stats-left">
            <span>当前分区 <strong>${filtered.length}</strong> 条</span>
            <span class="sep" aria-hidden="true">·</span>
            <span>本章已掌握 <strong>${pct}% (${masteredCount}/${allItems.length})</strong></span>
            <span class="module-tip">每条均附原书页码出处 · 例题/习题详解默认折叠可先自做</span>
          </div>
          <div class="progress-track" role="progressbar" aria-label="本章掌握进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}">
            <div class="progress-fill" style="--pct: ${pct}%;"></div>
          </div>
        </div>
      </section>

      ${canMask && state.maskMode ? `
        <div class="selftest-banner no-print">
          <span><strong>公式/结论默写遮罩已开启</strong>：根据题示默写公式或口述结论，点击卡片「翻开核对」逐条检查。</span>
          <button type="button" class="btn btn-subtle" data-action="reveal-all">全部翻开 / 重新遮罩</button>
        </div>
      ` : ''}

      ${state.returnBookmark && state.returnBookmark.chapterId === chMeta.id ? `
        <div class="stem-return-banner no-print" role="status">
          <span>已定位到目标卡片 · 随时可一键回到刚才看到的位置（${escapeHtml(KIND_META[state.returnBookmark.kindFilter]?.label || '')}）</span>
          <div class="stem-return-actions">
            <button type="button" class="btn btn-primary" data-action="return-bookmark">← 回到原位置</button>
            <button type="button" class="btn btn-subtle" data-action="dismiss-bookmark">留在当前位置</button>
          </div>
        </div>
      ` : ''}

      <div class="stem-card-list">
        ${filtered.length
          ? filtered.map(it => renderItemCard(it, { refIndex: chapterData._refIndex })).join('')
          : renderEmptyListState(allItems)}
      </div>
    `;
  }

  function render() {
    if (!root) return;
    if (state.loading) {
      root.innerHTML = `<section class="card empty-state"><p class="empty-title">正在加载章节资料与公式索引...</p></section>`;
      return;
    }
    if (state.error) {
      root.innerHTML = `
        <section class="card empty-state">
          <p class="empty-title">加载学科数据失败</p>
          <p class="empty-desc">${escapeHtml(state.error)}</p>
        </section>
      `;
      return;
    }
    if (!state.manifest) return;

    const manifest = state.manifest;
    const chMeta = getActiveChapterMeta();
    const chapterData = chMeta?.ready ? state.chapterCache.get(chMeta.id) : null;
    const readyCount = (manifest.chapters || []).filter(c => c.ready).length;
    const totalChapters = (manifest.chapters || []).length;

    const groups = Array.isArray(manifest.groups) && manifest.groups.length ? manifest.groups : null;
    const visibleChapters = groups && state.activeGroupId
      ? (manifest.chapters || []).filter(ch => ch.group === state.activeGroupId)
      : (manifest.chapters || []);

    root.innerHTML = `
      <div class="panel-body">
        <!-- 学科顶栏：来源声明 + 章节导航 -->
        <section class="card stem-subject-banner">
          <div class="stem-subject-header">
            <div>
              <div class="stem-subject-kicker">${escapeHtml(manifest.syllabusRef)}</div>
              <div class="stem-subject-books">主教材：${escapeHtml((manifest.primaryBooks || []).join(' ｜ '))}</div>
            </div>
            <div class="stem-subject-stat">
              <span class="stem-badge stem-badge-sympy">已上线 ${readyCount} / ${totalChapters} 单元</span>
            </div>
          </div>

          ${groups ? `
            <div class="module-filter-bar stem-group-bar no-print" role="group" aria-label="子学科分组切换">
              ${groups.map(g => {
                const grpChapters = (manifest.chapters || []).filter(c => c.group === g.id);
                const grpReady = grpChapters.filter(c => c.ready).length;
                return `
                  <button type="button" class="module-pill" data-group-id="${escapeHtml(g.id)}" aria-pressed="${state.activeGroupId === g.id}">
                    <span>${escapeHtml(g.label)}</span>
                    <span class="count-badge">${grpReady}/${grpChapters.length}</span>
                  </button>
                `;
              }).join('')}
            </div>
          ` : ''}

          <div class="stem-chapter-strip no-print" role="group" aria-label="章节与单元切换">
            ${visibleChapters.map(ch => {
              const active = ch.id === state.activeChapterId;
              return `
                <button
                  type="button"
                  class="stem-chapter-pill ${ch.ready ? 'is-ready' : 'is-upcoming'}"
                  data-chapter-id="${escapeHtml(ch.id)}"
                  aria-pressed="${active}"
                >
                  <span class="stem-ch-num">${escapeHtml(ch.num)}</span>
                  <span class="stem-ch-title">${escapeHtml(ch.title)}</span>
                  <span class="stem-ch-status">${ch.ready ? '已上线' : '待录入'}</span>
                </button>
              `;
            }).join('')}
          </div>
        </section>

        ${renderChapterContent(chMeta, chapterData)}
      </div>
    `;

    // Re-bind search input live listener (preserve cursor & focus if typing)
    const searchInput = root.querySelector('.stem-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        const val = e.target.value;
        const pos = e.target.selectionStart;
        state.searchQuery = val;
        renderCardsOnly();
        const clearBtn = root.querySelector('[data-action="clear-search"]');
        if (clearBtn) clearBtn.hidden = !val;
        if (document.activeElement !== searchInput) {
          searchInput.focus();
          if (pos != null) searchInput.setSelectionRange(pos, pos);
        }
      });
    }

    // Track <details> open/close state on example cards
    for (const det of root.querySelectorAll('.stem-solution-details')) {
      det.addEventListener('toggle', () => {
        const eid = det.dataset.exampleId;
        if (!eid) return;
        if (det.open) state.openExampleIds.add(eid);
        else state.openExampleIds.delete(eid);
      });
    }

    renderMathSafe(root);
  }

  function renderCardsOnly() {
    const chMeta = getActiveChapterMeta();
    if (!chMeta?.ready) return;
    const chapterData = state.chapterCache.get(chMeta.id);
    if (!chapterData) return;

    const listEl = root.querySelector('.stem-card-list');
    if (!listEl) {
      render();
      return;
    }
    const filtered = getFilteredItems(chapterData);
    listEl.innerHTML = filtered.length
      ? filtered.map(it => renderItemCard(it, { refIndex: chapterData._refIndex })).join('')
      : renderEmptyListState(chapterData.items || []);

    // Update count display without blowing away search input focus
    const countStrong = root.querySelector('.stats-left strong');
    if (countStrong) countStrong.textContent = String(filtered.length);

    for (const det of listEl.querySelectorAll('.stem-solution-details')) {
      det.addEventListener('toggle', () => {
        const eid = det.dataset.exampleId;
        if (!eid) return;
        if (det.open) state.openExampleIds.add(eid);
        else state.openExampleIds.delete(eid);
      });
    }

    renderMathSafe(listEl);
  }

  async function selectChapter(chapterId) {
    if (!state.manifest) return;
    const chMeta = (state.manifest.chapters || []).find(c => c.id === chapterId);
    if (!chMeta) return;
    state.activeChapterId = chapterId;
    if (chMeta.group) state.activeGroupId = chMeta.group;
    state.kindFilter = getSubjectKinds()[0] || 'formula';
    state.stageFilter = 'foundation';
    state.searchQuery = '';
    state.revealedIds.clear();
    state.returnBookmark = null;
    state.drawerStack = [];
    if (sharedRefDrawer?.open && activeDrawerOwner === drawerController) {
      closeTopLayerDialog(sharedRefDrawer);
    }

    if (chMeta.ready && !state.chapterCache.has(chMeta.id)) {
      state.loading = true;
      render();
      try {
        await loadChapter(chMeta);
        state.error = null;
      } catch (err) {
        state.error = err.message;
      } finally {
        state.loading = false;
      }
    }
    render();
    updateDocumentTitle();
  }

  function updateDocumentTitle() {
    if (store.activeSubject !== subjectId || !state.manifest) return;
    const chMeta = getActiveChapterMeta();
    const prefix = chMeta ? `${chMeta.num} ${chMeta.title} · ` : '';
    document.title = `${prefix}${state.manifest.subjectName} | 考研三科冲刺工作台`;
  }

  async function activate() {
    if (!state.loaded) {
      state.loading = true;
      render();
      try {
        await loadManifest();
        const chMeta = getActiveChapterMeta();
        if (chMeta?.ready) {
          await loadChapter(chMeta);
        }
        state.loaded = true;
        state.error = null;
      } catch (err) {
        state.error = err.message;
      } finally {
        state.loading = false;
      }
    }
    render();
    updateDocumentTitle();
  }

  function getActiveChapterData() {
    const chMeta = getActiveChapterMeta();
    return chMeta?.ready ? state.chapterCache.get(chMeta.id) : null;
  }

  function renderDrawerCurrent() {
    const dialog = getSharedRefDrawer();
    const chapterData = getActiveChapterData();
    const currentId = state.drawerStack[state.drawerStack.length - 1];
    const target = currentId ? chapterData?._refIndex?.byId.get(currentId) : null;
    if (!target) {
      closeTopLayerDialog(dialog);
      return;
    }
    const kindLabel = KIND_META[target.kind]?.label || target.kind;
    const plainTitle = String(target.title || '').replace(/\$/g, '');
    const titleEl = dialog.querySelector('#stemRefDrawerTitle');
    const kickerEl = dialog.querySelector('#stemRefDrawerKicker');
    const backBtn = dialog.querySelector('[data-drawer-action="back"]');
    const bodyEl = dialog.querySelector('#stemRefDrawerBody');

    if (kickerEl) {
      const depthInfo = state.drawerStack.length > 1 ? `（第 ${state.drawerStack.length} 层预览）` : '';
      kickerEl.textContent = `侧边浮窗预览 · ${kindLabel}${depthInfo} · 原阅读位置已保留`;
    }
    if (titleEl) {
      titleEl.innerHTML = renderRichText(target.title || '');
      renderMathSafe(titleEl);
    }
    if (backBtn) backBtn.hidden = state.drawerStack.length <= 1;
    if (bodyEl) {
      bodyEl.innerHTML = renderItemCard(target, { refIndex: chapterData._refIndex, inDrawer: true });
      bodyEl.scrollTop = 0;
      renderMathSafe(bodyEl);
    }
  }

  function openRefDrawer(itemId, { push = false } = {}) {
    const chapterData = getActiveChapterData();
    const target = chapterData?._refIndex?.byId.get(itemId);
    if (!target) {
      showToast('本章暂未找到对应的关联卡片');
      return;
    }
    const dialog = getSharedRefDrawer();
    activeDrawerOwner = drawerController;
    if (push && dialog.open && state.drawerStack.length) {
      if (state.drawerStack[state.drawerStack.length - 1] !== itemId) {
        state.drawerStack.push(itemId);
      }
    } else {
      state.drawerStack = [itemId];
    }
    renderDrawerCurrent();
    if (!dialog.open) {
      dialog.showModal();
    }
  }

  function drawerBack() {
    if (state.drawerStack.length > 1) {
      state.drawerStack.pop();
      renderDrawerCurrent();
    }
  }

  function drawerLocate() {
    const chapterData = getActiveChapterData();
    const currentId = state.drawerStack[state.drawerStack.length - 1];
    const target = currentId ? chapterData?._refIndex?.byId.get(currentId) : null;
    if (!target) return;

    // Save exact scroll & filter bookmark before jumping
    state.returnBookmark = {
      chapterId: state.activeChapterId,
      kindFilter: state.kindFilter,
      stageFilter: state.stageFilter,
      scrollY: window.scrollY,
    };

    state.kindFilter = target.kind;
    if (target.stage === 'foundation' || target.stage === 'intensive') {
      state.stageFilter = target.stage;
    }
    // Ensure target is not hidden by active narrow filters
    if (state.searchQuery) state.searchQuery = '';
    if (state.starOnly && !store.stemStarred.has(target.id)) state.starOnly = false;
    if (state.flaggedOnly && !store.stemFlagged.has(target.id)) state.flaggedOnly = false;
    if (state.hideMastered && store.stemMastered.has(target.id)) state.hideMastered = false;
    if (state.coreOnly && target.examWeight === 'extended') state.coreOnly = false;
    if (target.kind === 'example' || target.kind === 'exercise') {
      state.openExampleIds.add(target.id);
    }

    const dialog = getSharedRefDrawer();
    if (dialog.open) dialog.close();
    render();

    requestAnimationFrame(() => {
      const cardEl = root?.querySelector(`.stem-card[data-item-id="${CSS.escape(target.id)}"]`);
      if (cardEl) {
        cardEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        cardEl.classList.remove('is-jump-highlight');
        void cardEl.offsetWidth;
        cardEl.classList.add('is-jump-highlight');
      }
    });
  }

  function handleCardAction(action, itemId, { fromDrawer = false } = {}) {
    if (action === 'clear-search') {
      state.searchQuery = '';
      render();
      root.querySelector('.stem-search-input')?.focus();
      return;
    }

    if (action === 'return-bookmark' && state.returnBookmark) {
      const bm = state.returnBookmark;
      state.returnBookmark = null;
      state.kindFilter = bm.kindFilter;
      state.stageFilter = bm.stageFilter;
      render();
      requestAnimationFrame(() => {
        window.scrollTo({ top: bm.scrollY, behavior: 'smooth' });
      });
      return;
    }

    if (action === 'dismiss-bookmark') {
      state.returnBookmark = null;
      render();
      return;
    }

    if (action === 'reveal' && itemId) {
      if (state.revealedIds.has(itemId)) state.revealedIds.delete(itemId);
      else state.revealedIds.add(itemId);
      render();
      return;
    }

    if (action === 'reveal-all') {
      const chData = getActiveChapterData();
      const maskable = (chData?.items || []).filter(i => i.kind === 'formula' || i.kind === 'conclusion');
      const allRevealed = maskable.length > 0 && maskable.every(i => state.revealedIds.has(i.id));
      if (allRevealed) {
        state.revealedIds.clear();
      } else {
        for (const i of maskable) state.revealedIds.add(i.id);
      }
      render();
      return;
    }

    if (action === 'star' && itemId) {
      if (store.stemStarred.has(itemId)) {
        store.stemStarred.delete(itemId);
        showToast('已取消重点标记');
      } else {
        store.stemStarred.add(itemId);
        showToast('已标记为重点');
      }
      store.notify('stem-star');
      if (fromDrawer && sharedRefDrawer?.open) renderDrawerCurrent();
      return;
    }

    if (action === 'master' && itemId) {
      if (store.stemMastered.has(itemId)) {
        store.stemMastered.delete(itemId);
        showToast('已取消掌握标记');
      } else {
        store.stemMastered.add(itemId);
        showToast('已标记为掌握');
      }
      store.notify('stem-master');
      if (fromDrawer && sharedRefDrawer?.open) renderDrawerCurrent();
      return;
    }

    if (action === 'flag' && itemId) {
      if (store.stemFlagged.has(itemId)) {
        store.stemFlagged.delete(itemId);
        showToast('已取消存疑标记');
      } else {
        store.stemFlagged.add(itemId);
        showToast('已标记为「存疑待核」，可用顶部筛选集中对照原书');
      }
      store.notify('stem-flag');
      if (fromDrawer && sharedRefDrawer?.open) renderDrawerCurrent();
    }
  }

  const drawerController = {
    openRefDrawer,
    drawerBack,
    drawerLocate,
    handleCardAction,
    onDrawerClosed() {
      state.drawerStack = [];
    },
  };

  // Event delegation on root
  root?.addEventListener('click', (e) => {
    const refBtn = e.target.closest('[data-ref-id]');
    if (refBtn) {
      openRefDrawer(refBtn.dataset.refId, { push: false });
      return;
    }

    const grpBtn = e.target.closest('[data-group-id]');
    if (grpBtn && state.manifest) {
      const gid = grpBtn.dataset.groupId;
      state.activeGroupId = gid;
      const grpChapters = (state.manifest.chapters || []).filter(c => c.group === gid);
      const nextCh = grpChapters.find(c => c.ready) || grpChapters[0];
      if (nextCh && nextCh.id !== state.activeChapterId) {
        selectChapter(nextCh.id);
      } else {
        render();
      }
      return;
    }

    const chBtn = e.target.closest('[data-chapter-id]');
    if (chBtn) {
      selectChapter(chBtn.dataset.chapterId);
      return;
    }

    const jumpBtn = e.target.closest('[data-jump-chapter]');
    if (jumpBtn) {
      selectChapter(jumpBtn.dataset.jumpChapter);
      return;
    }

    const kindBtn = e.target.closest('[data-kind-filter]');
    if (kindBtn) {
      state.kindFilter = kindBtn.dataset.kindFilter;
      state.returnBookmark = null;
      render();
      const activeTab = root.querySelector(`[data-kind-filter="${CSS.escape(state.kindFilter)}"]`);
      if (activeTab && typeof activeTab.scrollIntoView === 'function') {
        try {
          activeTab.scrollIntoView({ block: 'nearest', inline: 'nearest', container: 'nearest' });
        } catch {
          activeTab.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
      }
      return;
    }

    const stageBtn = e.target.closest('[data-stage-filter]');
    if (stageBtn) {
      state.stageFilter = stageBtn.dataset.stageFilter;
      render();
      return;
    }

    const toggleBtn = e.target.closest('[data-toggle]');
    if (toggleBtn) {
      const key = toggleBtn.dataset.toggle;
      state[key] = !state[key];
      if (key === 'maskMode' && !state.maskMode) {
        state.revealedIds.clear();
      }
      render();
      return;
    }

    const actionBtn = e.target.closest('[data-action]');
    if (!actionBtn) return;
    handleCardAction(actionBtn.dataset.action, actionBtn.dataset.id, { fromDrawer: false });
  });

  return {
    activate,
    render() {
      if (state.loaded && store.activeSubject === subjectId) {
        render();
        if (sharedRefDrawer?.open && activeDrawerOwner === drawerController) {
          renderDrawerCurrent();
        }
      }
    },
    getPrintSummary() {
      const chMeta = getActiveChapterMeta();
      if (!state.manifest) return '';
      return `导出科目：${state.manifest.subjectName}${chMeta ? ` · ${chMeta.num} ${chMeta.title}` : ''}`;
    },
  };
}
