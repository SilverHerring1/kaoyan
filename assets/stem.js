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

function renderTablesBlock(item) {
  const tables = Array.isArray(item.tables) ? item.tables : [];
  if (!tables.length) return '';
  return tables.map((tbl, idx) => {
    const capId = `tbl-cap-${String(item.id || 'item').replace(/[^a-zA-Z0-9_-]/g, '')}-${idx}`;
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
                    ? `<th scope="row">${renderRichText(cell)}</th>`
                    : `<td>${renderRichText(cell)}</td>`
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
      ${renderTablesBlock(item)}
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
      ${renderTablesBlock(item)}
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

  function renderWorkedProblemBody(item, { isExercise = false } = {}) {
    const isOpen = state.openExampleIds.has(item.id);
    const steps = item.steps || [];
    const problemLabel = isExercise ? '课后习题题干' : '题目条件';
    const summaryLabel = isExercise
      ? `展开课后习题详解与最终答案（共 ${steps.length} 步）`
      : `展开分步详解与最终答案（共 ${steps.length} 步）`;
    return `
      <div class="stem-problem-box ${isExercise ? 'stem-exercise-problem-box' : ''}">
        <span class="stem-problem-label">${problemLabel}</span>
        <div class="stem-problem-text">${renderRichText(item.problem)}</div>
      </div>
      ${renderFigureBlock(item)}
      ${renderTablesBlock(item)}
      <details class="stem-solution-details ${isExercise ? 'stem-exercise-details' : ''}" data-example-id="${escapeHtml(item.id)}" ${isOpen ? 'open' : ''}>
        <summary class="stem-solution-summary">
          <span class="stem-summary-label">${summaryLabel}</span>
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

  function renderSummaryBody(item) {
    const overviewText = item.overview || item.summary || '';
    const patterns = Array.isArray(item.patterns) ? item.patterns : [];
    return `
      ${overviewText ? `
        <div class="stem-summary-overview">
          <div class="stem-summary-overview-text">${renderRichText(overviewText)}</div>
        </div>
      ` : ''}
      ${renderFigureBlock(item)}
      ${renderTablesBlock(item)}
      ${patterns.length ? `
        <div class="stem-pattern-list">
          ${patterns.map((p, idx) => `
            <div class="stem-pattern-card">
              <div class="stem-pattern-head">
                <span class="stem-pattern-index">题型 ${idx + 1}</span>
                <h4 class="stem-pattern-title">${renderRichText(p.typeName)}</h4>
                ${p.relatedRefs ? `<span class="stem-pattern-refs">对应：${escapeHtml(p.relatedRefs)}</span>` : ''}
              </div>
              <div class="stem-pattern-grid">
                <div class="stem-pattern-cell">
                  <span class="callout-tag tone-indigo">识别信号</span>
                  <p>${renderRichText(p.triggers)}</p>
                </div>
                <div class="stem-pattern-cell">
                  <span class="callout-tag tone-emerald">标准破题步骤</span>
                  <p>${renderRichText(p.method)}</p>
                </div>
                ${p.pitfall ? `
                  <div class="stem-pattern-cell stem-pattern-pitfall">
                    <span class="callout-tag tone-amber">避坑要点</span>
                    <p>${renderRichText(p.pitfall)}</p>
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
          <p>${renderRichText(item.takeaway)}</p>
        </div>
      ` : ''}
    `;
  }

  function renderItemCard(item) {
    const isMastered = store.stemMastered.has(item.id);
    const isFlagged = store.stemFlagged.has(item.id);
    let bodyHtml = '';
    if (item.kind === 'formula') bodyHtml = renderFormulaBody(item);
    else if (item.kind === 'conclusion') bodyHtml = renderConclusionBody(item);
    else if (item.kind === 'example') bodyHtml = renderWorkedProblemBody(item, { isExercise: false });
    else if (item.kind === 'exercise') bodyHtml = renderWorkedProblemBody(item, { isExercise: true });
    else if (item.kind === 'summary') bodyHtml = renderSummaryBody(item);

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

      <div class="stem-card-list">
        ${filtered.length
          ? filtered.map(renderItemCard).join('')
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
      ? filtered.map(renderItemCard).join('')
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
