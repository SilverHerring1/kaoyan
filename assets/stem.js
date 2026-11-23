import { escapeHtml, showToast } from './app.js';

const KIND_META = {
  formula: { label: '核心公式', badgeClass: 'stem-kind-formula-chip' },
  conclusion: { label: '结论与判据', badgeClass: 'stem-kind-conclusion-chip' },
  example: { label: '经典例题', badgeClass: 'stem-kind-example-chip' },
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
 * Escapes HTML and converts lightweight **bold** emphasis + newlines,
 * leaving LaTeX math delimiters ($...$, $$...$$) intact for KaTeX.
 */
function renderRichText(raw) {
  const safe = escapeHtml(raw ?? '');
  return safe
    .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function extractSvgTitle(svgStr, fallback) {
  const m = String(svgStr || '').match(/<title>([\s\S]*?)<\/title>/i);
  return m ? m[1].trim() : fallback;
}

function renderFigureBlock(item) {
  const svg = String(item.figureSvg || '').trim();
  if (!svg || !svg.startsWith('<svg')) return '';
  const caption = extractSvgTitle(svg, `${item.source?.itemRef || '示意图'}`);
  return `
    <figure class="stem-figure">
      <div class="stem-svg-stage">${svg}</div>
      <figcaption class="stem-figcaption">${escapeHtml(caption)}</figcaption>
    </figure>
  `;
}

function buildSearchHaystack(item) {
  const parts = [
    item.id,
    item.section,
    item.title,
    item.prompt,
    item.formulaLatex,
    item.summary,
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
    activeChapterId: null,
    kindFilter: 'all', // all | formula | conclusion | example
    searchQuery: '',
    maskMode: false,
    starOnly: false,
    hideMastered: false,
    flaggedOnly: false,
    coreOnly: false,
    revealedIds: new Set(),
    openExampleIds: new Set(),
  };

  async function loadManifest() {
    if (state.manifest) return state.manifest;
    const res = await fetch(`./data/${subjectId}/manifest.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.manifest = await res.json();
    const firstReady = (state.manifest.chapters || []).find(ch => ch.ready);
    state.activeChapterId = firstReady ? firstReady.id : (state.manifest.chapters?.[0]?.id || null);
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
    // Pre-build search index per item
    for (const item of data.items || []) {
      item._haystack = buildSearchHaystack(item);
    }
    state.chapterCache.set(chapterMeta.id, data);
    return data;
  }

  function getActiveChapterMeta() {
    return (state.manifest?.chapters || []).find(c => c.id === state.activeChapterId) || null;
  }

  function getFilteredItems(chapterData) {
    const items = chapterData?.items || [];
    const q = state.searchQuery.trim().toLowerCase();
    return items.filter(item => {
      if (state.kindFilter !== 'all' && item.kind !== state.kindFilter) return false;
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
      ? `<span class="stem-badge stem-badge-ext">拓展了解</span>`
      : `<span class="stem-badge stem-badge-core">大纲核心</span>`;
    const plainTitle = String(item.title || '').replace(/\$/g, '');

    return `
      <div class="stem-card-top">
        <div class="stem-card-meta">
          <span class="stem-badge ${meta.badgeClass}">${meta.label}</span>
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

  function renderFormulaBody(item) {
    const masked = state.maskMode && !state.revealedIds.has(item.id);
    const symbolsHtml = Array.isArray(item.symbols) && item.symbols.length
      ? `
        <dl class="stem-symbols-list">
          ${item.symbols.map(s => `
            <div class="stem-symbol-row">
              <dt>${renderRichText(s.sym)}</dt>
              <dd>${renderRichText(s.desc)}</dd>
            </div>
          `).join('')}
        </dl>
      `
      : '';

    return `
      ${item.prompt ? `<p class="stem-prompt">${renderRichText(item.prompt)}</p>` : ''}
      <div class="stem-formula-box ${masked ? 'is-masked' : ''}">
        <div class="math-scroll ${masked ? 'is-blurred' : ''}">${escapeHtml(item.formulaLatex)}</div>
        ${masked ? `
          <div class="stem-mask-overlay no-print">
            <button type="button" class="btn btn-primary stem-reveal-btn" data-action="reveal" data-id="${escapeHtml(item.id)}">翻开核对公式</button>
          </div>
        ` : (state.maskMode ? `
          <div class="stem-rehide-row no-print">
            <button type="button" class="btn-link-muted" data-action="reveal" data-id="${escapeHtml(item.id)}">重新遮罩</button>
          </div>
        ` : '')}
      </div>
      ${renderFigureBlock(item)}
      ${symbolsHtml}
      <div class="stem-callouts">
        ${item.conditions ? `
          <div class="stem-callout callout-cond">
            <span class="callout-tag">适用前提</span>
            <p>${renderRichText(item.conditions)}</p>
          </div>
        ` : ''}
        ${item.pitfalls ? `
          <div class="stem-callout callout-pitfall">
            <span class="callout-tag">易错红线 / 勘误</span>
            <p>${renderRichText(item.pitfalls)}</p>
          </div>
        ` : ''}
      </div>
    `;
  }

  function renderConclusionBody(item) {
    const masked = state.maskMode && !state.revealedIds.has(item.id);
    return `
      <div class="stem-conclusion-box ${masked ? 'is-masked' : ''}">
        <div class="stem-conclusion-text ${masked ? 'is-blurred' : ''}">${renderRichText(item.summary)}</div>
        ${masked ? `
          <div class="stem-mask-overlay no-print">
            <button type="button" class="btn btn-primary stem-reveal-btn" data-action="reveal" data-id="${escapeHtml(item.id)}">翻开核对结论</button>
          </div>
        ` : (state.maskMode ? `
          <div class="stem-rehide-row no-print">
            <button type="button" class="btn-link-muted" data-action="reveal" data-id="${escapeHtml(item.id)}">重新遮罩</button>
          </div>
        ` : '')}
      </div>
      ${renderFigureBlock(item)}
      <div class="stem-callouts">
        ${item.conditions ? `
          <div class="stem-callout callout-cond">
            <span class="callout-tag">适用范围</span>
            <p>${renderRichText(item.conditions)}</p>
          </div>
        ` : ''}
        ${item.pitfalls ? `
          <div class="stem-callout callout-pitfall">
            <span class="callout-tag">高频坑点</span>
            <p>${renderRichText(item.pitfalls)}</p>
          </div>
        ` : ''}
      </div>
    `;
  }

  function renderExampleBody(item) {
    const isOpen = state.openExampleIds.has(item.id);
    const steps = item.steps || [];
    return `
      <div class="stem-problem-box">
        <span class="stem-problem-label">题目条件</span>
        <div class="stem-problem-text">${renderRichText(item.problem)}</div>
      </div>
      ${renderFigureBlock(item)}
      <details class="stem-solution-details" data-example-id="${escapeHtml(item.id)}" ${isOpen ? 'open' : ''}>
        <summary class="stem-solution-summary">
          <span class="stem-summary-label">展开分步详解与最终答案（共 ${steps.length} 步）</span>
          <span class="disclosure-chevron" aria-hidden="true"></span>
        </summary>
        <div class="stem-solution-body">
          <ol class="stem-steps-list">
            ${steps.map(st => `
              <li class="stem-step-item">
                <div class="stem-step-title">${renderRichText(String(st.stepTitle || '').replace(/^\d+\.\s*/, ''))}</div>
                <div class="stem-step-detail">${renderRichText(st.detail)}</div>
              </li>
            `).join('')}
          </ol>
          <div class="stem-final-answer">
            <span class="callout-tag">最终答案</span>
            <div>${renderRichText(item.finalAnswer)}</div>
          </div>
          ${item.takeaway ? `
            <div class="stem-callout callout-pitfall stem-takeaway-box">
              <span class="callout-tag">考场提炼</span>
              <p>${renderRichText(item.takeaway)}</p>
            </div>
          ` : ''}
        </div>
      </details>
    `;
  }

  function renderItemCard(item) {
    const isMastered = store.stemMastered.has(item.id);
    const isFlagged = store.stemFlagged.has(item.id);
    let bodyHtml = '';
    if (item.kind === 'formula') bodyHtml = renderFormulaBody(item);
    else if (item.kind === 'conclusion') bodyHtml = renderConclusionBody(item);
    else if (item.kind === 'example') bodyHtml = renderExampleBody(item);

    return `
      <article class="card stem-card stem-kind-${escapeHtml(item.kind)} ${isMastered ? 'is-mastered' : ''} ${isFlagged ? 'is-flagged' : ''}" data-item-id="${escapeHtml(item.id)}">
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

  function renderChapterContent(chMeta, chapterData) {
    if (!chMeta) return '';
    if (!chMeta.ready || !chapterData) {
      return renderRoadmapPanel(chMeta);
    }

    const allItems = chapterData.items || [];
    const filtered = getFilteredItems(chapterData);

    const kindCounts = {
      all: allItems.length,
      formula: allItems.filter(i => i.kind === 'formula').length,
      conclusion: allItems.filter(i => i.kind === 'conclusion').length,
      example: allItems.filter(i => i.kind === 'example').length,
    };
    const extendedCount = allItems.filter(i => i.examWeight === 'extended').length;
    const starCount = allItems.filter(i => store.stemStarred.has(i.id)).length;
    const masteredCount = allItems.filter(i => store.stemMastered.has(i.id)).length;
    const flaggedCount = allItems.filter(i => store.stemFlagged.has(i.id)).length;
    const pct = allItems.length ? Math.round((masteredCount / allItems.length) * 100) : 0;

    const req = chapterData.basicRequirements || {};
    const searchId = `stemSearch-${subjectId}`;

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
              <div class="stem-req-label tone-rose">2027 浙大 845 大纲要求</div>
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

      <!-- 搜索与筛选控制栏 -->
      <section class="card control-panel no-print">
        <search class="control-top-row">
          <div class="search-field">
            <label for="${searchId}" class="visually-hidden">搜索本章公式、结论、例题或原书页码</label>
            <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
            </svg>
            <input id="${searchId}" class="stem-search-input" type="search" value="${escapeHtml(state.searchQuery)}" placeholder="搜索公式名、关键词、例题号（如 例4-8 / 劳斯 / 主导极点 / P.95）...">
            <button type="button" class="icon-btn clear-search-btn" data-action="clear-search" ${state.searchQuery ? '' : 'hidden'} aria-label="清空搜索关键词">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
              </svg>
            </button>
          </div>

          <div class="toolbar-controls">
            <button type="button" class="pill-toggle" data-toggle="maskMode" aria-pressed="${state.maskMode}">
              <span>默写遮罩：${state.maskMode ? '开' : '关'}</span>
            </button>
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
                <span>仅看大纲核心</span>
              </button>
            ` : ''}
          </div>
        </search>

        <div class="module-filter-bar" role="group" aria-label="按内容类型筛选">
          ${[
            { key: 'all', label: '全部内容', count: kindCounts.all },
            { key: 'formula', label: '核心公式', count: kindCounts.formula },
            { key: 'conclusion', label: '结论与判据', count: kindCounts.conclusion },
            { key: 'example', label: '经典例题', count: kindCounts.example },
          ].map(k => `
            <button type="button" class="module-pill" data-kind-filter="${k.key}" aria-pressed="${state.kindFilter === k.key}">
              <span>${k.label}</span>
              <span class="count-badge">${k.count}</span>
            </button>
          `).join('')}
        </div>

        <div class="stats-footer">
          <div class="stats-left">
            <span>当前显示 <strong>${filtered.length}</strong> 条</span>
            <span class="sep" aria-hidden="true">·</span>
            <span>本章已掌握 <strong>${pct}% (${masteredCount}/${allItems.length})</strong></span>
            <span class="module-tip">每条均附原书页码出处 · 例题详解默认折叠可先自做</span>
          </div>
          <div class="progress-track" role="progressbar" aria-label="本章掌握进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}">
            <div class="progress-fill" style="--pct: ${pct}%;"></div>
          </div>
        </div>
      </section>

      ${state.maskMode ? `
        <div class="selftest-banner no-print">
          <span><strong>公式/结论默写遮罩已开启</strong>：根据题示默写公式或口述结论，点击卡片「翻开核对」逐条检查。</span>
          <button type="button" class="btn btn-subtle" data-action="reveal-all">全部翻开 / 重新遮罩</button>
        </div>
      ` : ''}

      <div class="stem-card-list">
        ${filtered.length
          ? filtered.map(renderItemCard).join('')
          : `
            <section class="card empty-state">
              <p class="empty-title">没有找到匹配的知识点或例题</p>
              <p class="empty-desc">可以清空搜索词，或关闭「重点 / 隐藏已掌握 / 存疑待核 / 仅看大纲核心」筛选。</p>
            </section>
          `}
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

          <div class="stem-chapter-strip no-print" role="group" aria-label="章节与单元切换">
            ${(manifest.chapters || []).map(ch => {
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
      ? filtered.map(renderItemCard).join('')
      : `
        <section class="card empty-state">
          <p class="empty-title">没有找到匹配的知识点或例题</p>
          <p class="empty-desc">可以清空搜索词，或关闭「重点 / 隐藏已掌握 / 存疑待核 / 仅看大纲核心」筛选。</p>
        </section>
      `;

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
    state.kindFilter = 'all';
    state.searchQuery = '';
    state.revealedIds.clear();

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

  // Event delegation on root
  root?.addEventListener('click', (e) => {
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
    const action = actionBtn.dataset.action;
    const itemId = actionBtn.dataset.id;

    if (action === 'clear-search') {
      state.searchQuery = '';
      render();
      root.querySelector('.stem-search-input')?.focus();
      return;
    }

    if (action === 'reveal' && itemId) {
      if (state.revealedIds.has(itemId)) state.revealedIds.delete(itemId);
      else state.revealedIds.add(itemId);
      render();
      return;
    }

    if (action === 'reveal-all') {
      const chMeta = getActiveChapterMeta();
      const chData = chMeta ? state.chapterCache.get(chMeta.id) : null;
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
    }
  });

  return {
    activate,
    render() {
      if (state.loaded && store.activeSubject === subjectId) {
        render();
      }
    },
    getPrintSummary() {
      const chMeta = getActiveChapterMeta();
      if (!state.manifest) return '';
      return `导出科目：${state.manifest.subjectName}${chMeta ? ` · ${chMeta.num} ${chMeta.title}` : ''}`;
    },
  };
}
