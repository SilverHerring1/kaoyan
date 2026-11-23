import { escapeHtml, showToast, speakText, speakSequence } from './app.js';

function fisherYates(arr) {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function initVocabView(store, callbacks) {
  const modules = store.vocabData.modules.filter(m => m.mode === 'quiz');
  const items = store.vocabData.items.filter(i => i.mode === 'quiz');
  const quizIdSet = new Set(items.map(i => i.id));
  const moduleMap = new Map(modules.map(m => [m.id, m]));

  const state = {
    filter: modules[0]?.id || 'idioms',
    search: '',
    sort: 'default',
    starOnly: false,
    hideMastered: false,
    hasQuizOnly: false,
    selfTest: false,
    revealAll: false,
    revealedIds: new Set(),
    shuffleOrder: new Map(),
    resetConfirmPending: false,
  };

  const searchInput = document.getElementById('searchInput');
  const clearSearchBtn = document.getElementById('clearSearchBtn');
  const sortSelect = document.getElementById('sortSelect');
  const reshuffleBtn = document.getElementById('reshuffleBtn');
  const selfTestBtn = document.getElementById('selfTestBtn');
  const selfTestLabel = document.getElementById('selfTestLabel');
  const selfTestBanner = document.getElementById('selfTestBanner');
  const revealAllBtn = document.getElementById('revealAllBtn');
  const starOnlyBtn = document.getElementById('starOnlyBtn');
  const hideMasteredBtn = document.getElementById('hideMasteredBtn');
  const hideMasteredLabel = document.getElementById('hideMasteredLabel');
  const hasQuizOnlyBtn = document.getElementById('hasQuizOnlyBtn');
  const resetProgressBtn = document.getElementById('resetProgressBtn');
  const moduleFilterBar = document.getElementById('moduleFilterBar');
  const vocabList = document.getElementById('vocabList');
  const vocabEmpty = document.getElementById('vocabEmpty');
  const countDisplay = document.getElementById('countDisplay');
  const progressText = document.getElementById('progressText');
  const progressBar = document.getElementById('progressBar');
  const progressTrack = document.getElementById('progressTrack');
  const moduleTipText = document.getElementById('moduleTipText');
  const printSummary = document.getElementById('printSummary');

  function regenerateShuffle() {
    const shuffled = fisherYates(items.map(i => i.id));
    state.shuffleOrder = new Map(shuffled.map((id, idx) => [id, idx]));
  }
  regenerateShuffle();

  function renderModulePills() {
    const counts = {};
    for (const item of items) {
      counts[item.type] = (counts[item.type] || 0) + 1;
    }

    const pills = modules.map(m => ({ id: m.id, label: m.label, count: counts[m.id] || 0 }));

    moduleFilterBar.innerHTML = pills
      .map(
        p => `
        <button type="button" class="module-pill" data-module="${escapeHtml(p.id)}" aria-pressed="${state.filter === p.id}">
          <span>${escapeHtml(p.label)}</span>
          <span class="count-badge">${p.count}</span>
        </button>
      `
      )
      .join('');
  }

  function updateStats() {
    let starredCnt = 0;
    let masteredCnt = 0;
    for (const id of store.starred) if (quizIdSet.has(id)) starredCnt += 1;
    for (const id of store.mastered) if (quizIdSet.has(id)) masteredCnt += 1;

    document.getElementById('starCountBadge').textContent = starredCnt;
    document.getElementById('masteredCountBadge').textContent = masteredCnt;
    document.getElementById('hasQuizCountBadge').textContent = store.questionsByVocabId.size;

    const pct = items.length ? Math.round((masteredCnt / items.length) * 100) : 0;
    progressText.textContent = `${pct}% (${masteredCnt}/${items.length})`;
    progressBar.style.setProperty('--pct', `${pct}%`);
    progressTrack.setAttribute('aria-valuenow', String(pct));
  }

  function render() {
    const q = state.search.trim().toLowerCase();

    let filtered = items.filter(item => {
      if (state.filter !== 'all' && item.type !== state.filter) return false;
      if (state.starOnly && !store.starred.has(item.id)) return false;
      if (state.hideMastered && store.mastered.has(item.id)) return false;
      if (state.hasQuizOnly && !store.questionsByVocabId.has(item.id)) return false;
      if (q) {
        const hit =
          item.word.toLowerCase().includes(q) ||
          item.meaning.toLowerCase().includes(q) ||
          item.notes.toLowerCase().includes(q) ||
          item.typeLabel.toLowerCase().includes(q);
        if (!hit) return false;
      }
      return true;
    });

    if (state.sort === 'alpha') {
      filtered.sort((a, b) => a.word.localeCompare(b.word));
    } else if (state.sort === 'starred') {
      filtered.sort((a, b) => {
        const sa = store.starred.has(a.id) ? 0 : 1;
        const sb = store.starred.has(b.id) ? 0 : 1;
        return sa - sb || a.order - b.order;
      });
    } else if (state.sort === 'shuffle') {
      filtered.sort((a, b) => (state.shuffleOrder.get(a.id) ?? 0) - (state.shuffleOrder.get(b.id) ?? 0));
    } else {
      filtered.sort((a, b) => a.order - b.order);
    }

    countDisplay.textContent = filtered.length;
    const modMeta = moduleMap.get(state.filter);
    moduleTipText.textContent = modMeta ? `· ${modMeta.tip}` : '';

    const flags = [
      state.starOnly ? '仅看重点' : '',
      state.hideMastered ? '已隐藏已掌握' : '',
      state.hasQuizOnly ? '仅已配外刊题' : '',
      q ? `搜索“${state.search}”` : '',
    ]
      .filter(Boolean)
      .join(' · ');
    printSummary.textContent = `导出范围：${modMeta ? modMeta.label : '全部词条'}${flags ? `（${flags}）` : ''} · 共 ${filtered.length} 条`;

    if (filtered.length === 0) {
      vocabList.innerHTML = '';
      vocabEmpty.hidden = false;
      return;
    }
    vocabEmpty.hidden = true;

    const frag = document.createDocumentFragment();
    for (const item of filtered) {
      const isStarred = store.starred.has(item.id);
      const isMastered = store.mastered.has(item.id);
      const hasQuiz = store.questionsByVocabId.has(item.id);
      const isRevealed = !state.selfTest || state.revealAll || state.revealedIds.has(item.id);
      const blurMeaning = !isRevealed ? 'is-blurred' : '';
      const blurSubLabel = !isRevealed && item.blurSubLabelInSelfTest ? 'is-blurred' : '';
      const isMulti = item.words.length > 1;

      const row = document.createElement('article');
      row.className = `vocab-row${isMastered ? ' is-mastered' : ''}`;
      row.dataset.id = item.id;

      const wordChipsHtml = isMulti
        ? `<div class="word-chips no-print" aria-label="逐词点读发音">
            ${item.words
              .map(
                w => `<button type="button" class="speak-word-chip" data-action="speak-one" data-term="${escapeHtml(w)}" lang="en" aria-label="朗读 ${escapeHtml(w)}">
                  <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M15.536 8.464a5 5 0 010 7.072M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/></svg>
                  <span>${escapeHtml(w)}</span>
                </button>`
              )
              .join('')}
           </div>`
        : '';

      row.innerHTML = `
        <div class="word-col">
          <div class="word-head-row">
            <span class="word-title" lang="en">${escapeHtml(item.word)}</span>
          </div>
          ${wordChipsHtml}
          <div class="row-actions no-print">
            <button type="button" class="icon-btn" data-action="speak-all" aria-label="${isMulti ? `顺序连读 ${escapeHtml(item.word)}` : `朗读 ${escapeHtml(item.word)}`}">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/>
              </svg>
            </button>
            <button type="button" class="icon-btn${isStarred ? ' is-starred' : ''}" data-action="star" aria-pressed="${isStarred}" aria-label="${isStarred ? '取消重点标记' : '标为重点'} ${escapeHtml(item.word)}">
              <svg class="icon" viewBox="0 0 24 24" fill="${isStarred ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/>
              </svg>
            </button>
            <button type="button" class="icon-btn${isMastered ? ' is-mastered' : ''}" data-action="master" aria-pressed="${isMastered}" aria-label="${isMastered ? '取消已掌握' : '标为已掌握'} ${escapeHtml(item.word)}">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
              </svg>
            </button>
            ${
              hasQuiz
                ? `<button type="button" class="quiz-jump-btn" data-action="jump-quiz">外刊刷题</button>`
                : ''
            }
          </div>
        </div>

        <div class="meta-col">
          <span class="badge-chip mod-${escapeHtml(item.type)}">${escapeHtml(item.typeLabel)}</span>
          <span class="phonetic-text ${blurSubLabel}">${escapeHtml(item.phonetic)}</span>
        </div>

        <div class="meaning-col">
          <div class="meaning-text ${blurMeaning}">${escapeHtml(item.meaning)}</div>
          ${
            state.selfTest
              ? `<button type="button" class="btn btn-subtle reveal-row-btn no-print" data-action="toggle-reveal" aria-expanded="${isRevealed}">${isRevealed ? '重新遮住' : '翻开核对'}</button>`
              : ''
          }
        </div>

        <div class="notes-col">
          <div class="notes-text ${blurMeaning}">${escapeHtml(item.notes)}</div>
        </div>
      `;

      frag.appendChild(row);
    }

    vocabList.innerHTML = '';
    vocabList.appendChild(frag);
  }

  // 单一事件委托（替代 1600+ 独立监听器）
  vocabList.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    const row = btn.closest('.vocab-row');
    if (!row) return;
    const id = row.dataset.id;
    const item = items.find(i => i.id === id);
    if (!item) return;

    const action = btn.dataset.action;
    if (action === 'speak-one') {
      speakText(btn.dataset.term || item.word);
    } else if (action === 'speak-all') {
      if (item.words.length > 1) speakSequence(item.words);
      else speakText(item.words[0] || item.word);
    } else if (action === 'star') {
      if (store.starred.has(id)) store.starred.delete(id);
      else store.starred.add(id);
      store.notify('star');
    } else if (action === 'master') {
      if (store.mastered.has(id)) store.mastered.delete(id);
      else store.mastered.add(id);
      store.notify('master');
    } else if (action === 'toggle-reveal') {
      if (state.revealAll) {
        // Exit revealAll and keep all other currently revealed except this one
        state.revealAll = false;
        state.revealedIds = new Set(items.map(x => x.id));
        state.revealedIds.delete(id);
      } else if (state.revealedIds.has(id)) {
        state.revealedIds.delete(id);
      } else {
        state.revealedIds.add(id);
      }
      render();
    } else if (action === 'jump-quiz') {
      callbacks.jumpToQuiz(id);
    }
  });

  // 搜索与筛选监听
  let searchTimer = null;
  searchInput.addEventListener('input', (e) => {
    state.search = e.target.value;
    clearSearchBtn.hidden = !state.search;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(render, 80);
  });

  clearSearchBtn.addEventListener('click', () => {
    searchInput.value = '';
    state.search = '';
    clearSearchBtn.hidden = true;
    render();
    searchInput.focus();
  });

  moduleFilterBar.addEventListener('click', (e) => {
    const pill = e.target.closest('.module-pill');
    if (!pill) return;
    state.filter = pill.dataset.module;
    renderModulePills();
    render();
  });

  sortSelect.addEventListener('change', (e) => {
    state.sort = e.target.value;
    reshuffleBtn.hidden = state.sort !== 'shuffle';
    if (state.sort === 'shuffle') regenerateShuffle();
    render();
  });

  reshuffleBtn.addEventListener('click', () => {
    regenerateShuffle();
    render();
    showToast('已重新随机打乱词序');
  });

  selfTestBtn.addEventListener('click', () => {
    state.selfTest = !state.selfTest;
    state.revealAll = false;
    state.revealedIds.clear();
    selfTestBtn.setAttribute('aria-pressed', String(state.selfTest));
    selfTestLabel.textContent = state.selfTest ? '遮罩自测：开' : '遮罩自测：关';
    selfTestBanner.hidden = !state.selfTest;
    render();
    showToast(state.selfTest ? '已开启遮罩自测模式' : '已关闭遮罩自测模式');
  });

  revealAllBtn.addEventListener('click', () => {
    state.revealAll = !state.revealAll;
    if (!state.revealAll) state.revealedIds.clear();
    render();
  });

  starOnlyBtn.addEventListener('click', () => {
    state.starOnly = !state.starOnly;
    starOnlyBtn.setAttribute('aria-pressed', String(state.starOnly));
    render();
  });

  hideMasteredBtn.addEventListener('click', () => {
    state.hideMastered = !state.hideMastered;
    hideMasteredBtn.setAttribute('aria-pressed', String(state.hideMastered));
    hideMasteredLabel.textContent = state.hideMastered ? '已隐藏掌握项' : '隐藏已掌握';
    render();
  });

  hasQuizOnlyBtn.addEventListener('click', () => {
    state.hasQuizOnly = !state.hasQuizOnly;
    hasQuizOnlyBtn.setAttribute('aria-pressed', String(state.hasQuizOnly));
    render();
  });

  let resetTimer = null;
  resetProgressBtn.addEventListener('click', () => {
    if (!state.resetConfirmPending) {
      state.resetConfirmPending = true;
      resetProgressBtn.textContent = '再点一次确认清空';
      resetProgressBtn.classList.add('btn-danger-confirm');
      clearTimeout(resetTimer);
      resetTimer = setTimeout(() => {
        state.resetConfirmPending = false;
        resetProgressBtn.textContent = '重置词库标记';
        resetProgressBtn.classList.remove('btn-danger-confirm');
      }, 3500);
    } else {
      clearTimeout(resetTimer);
      state.resetConfirmPending = false;
      resetProgressBtn.textContent = '重置词库标记';
      resetProgressBtn.classList.remove('btn-danger-confirm');
      store.starred.clear();
      store.mastered.clear();
      store.notify('reset-vocab');
      showToast('已清空所有重点与已掌握标记');
    }
  });

  renderModulePills();

  return {
    render,
    updateStats,
    searchWord(word) {
      const qLower = String(word || '').trim().toLowerCase();
      const matched = items.find(
        i =>
          i.word.toLowerCase().includes(qLower) ||
          i.meaning.toLowerCase().includes(qLower) ||
          i.notes.toLowerCase().includes(qLower)
      );
      if (matched) state.filter = matched.type;
      state.starOnly = false;
      state.hideMastered = false;
      state.hasQuizOnly = false;
      starOnlyBtn.setAttribute('aria-pressed', 'false');
      hideMasteredBtn.setAttribute('aria-pressed', 'false');
      hasQuizOnlyBtn.setAttribute('aria-pressed', 'false');
      searchInput.value = word;
      state.search = word;
      clearSearchBtn.hidden = false;
      renderModulePills();
      render();
      searchInput.focus();
    },
  };
}
