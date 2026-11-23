import { escapeHtml, showToast, speakText, speakSequence } from './app.js';

export function initMemorizeView(store) {
  const memModules = store.vocabData.modules.filter(m => m.mode === 'memorize');
  const memItems = store.vocabData.items.filter(i => i.mode === 'memorize');
  const memIdSet = new Set(memItems.map(i => i.id));
  const modMap = new Map(memModules.map(m => [m.id, m]));

  const state = {
    module: memModules[0]?.id || 'attitude-correct',
    group: 'all',
    search: '',
    selfTest: false,
    starOnly: false,
    hideMastered: false,
    revealedIds: new Set(),
  };

  const moduleBar = document.getElementById('memModuleBar');
  const groupBar = document.getElementById('memGroupBar');
  const searchInput = document.getElementById('memSearchInput');
  const selfTestBtn = document.getElementById('memSelfTestBtn');
  const selfTestLabel = document.getElementById('memSelfTestLabel');
  const starOnlyBtn = document.getElementById('memStarOnlyBtn');
  const hideMasteredBtn = document.getElementById('memHideMasteredBtn');
  const hideMasteredLabel = document.getElementById('memHideMasteredLabel');
  const starBadge = document.getElementById('memStarBadge');
  const masteredBadge = document.getElementById('memMasteredBadge');
  const countDisplay = document.getElementById('memCountDisplay');
  const progressText = document.getElementById('memProgressText');
  const progressBar = document.getElementById('memProgressBar');
  const progressTrack = document.getElementById('memProgressTrack');
  const moduleTip = document.getElementById('memModuleTip');
  const attitudeCaveat = document.getElementById('attitudeCaveatBanner');
  const container = document.getElementById('memCardContainer');

  function renderModuleBar() {
    const counts = {};
    for (const item of memItems) {
      counts[item.type] = (counts[item.type] || 0) + 1;
    }

    const tabs = [
      ...memModules.map(m => ({ id: m.id, label: m.label, count: counts[m.id] || 0 })),
      { id: 'all', label: '全部速背', count: memItems.length },
    ];

    moduleBar.innerHTML = tabs
      .map(
        t => `
          <button type="button" class="module-pill" data-mem-mod="${escapeHtml(t.id)}" aria-pressed="${state.module === t.id}">
            <span>${escapeHtml(t.label)}</span>
            <span class="count-badge">${t.count}</span>
          </button>
        `
      )
      .join('');
  }

  function getModulePool() {
    if (state.module === 'all') return memItems;
    return memItems.filter(i => i.type === state.module);
  }

  function renderGroupBar() {
    const pool = getModulePool();
    const groups = [];
    const seen = new Set();
    for (const item of pool) {
      const g = item.group || item.typeLabel;
      if (!seen.has(g)) {
        seen.add(g);
        groups.push(g);
      }
    }

    if (groups.length <= 1) {
      groupBar.innerHTML = '';
      groupBar.hidden = true;
      return;
    }

    groupBar.hidden = false;
    const pills = [
      { id: 'all', label: '全部分组' },
      ...groups.map(g => ({ id: g, label: g })),
    ];

    groupBar.innerHTML = pills
      .map(
        p => `
          <button type="button" class="subgroup-pill" data-mem-group="${escapeHtml(p.id)}" aria-pressed="${state.group === p.id}">
            ${escapeHtml(p.label)}
          </button>
        `
      )
      .join('');
  }

  function updateStats() {
    let memStarred = 0;
    let memMastered = 0;
    for (const id of store.starred) if (memIdSet.has(id)) memStarred += 1;
    for (const id of store.mastered) if (memIdSet.has(id)) memMastered += 1;

    starBadge.textContent = memStarred;
    masteredBadge.textContent = memMastered;

    const pct = memItems.length ? Math.round((memMastered / memItems.length) * 100) : 0;
    progressText.textContent = `${pct}% (${memMastered}/${memItems.length})`;
    progressBar.style.setProperty('--pct', `${pct}%`);
    progressTrack.setAttribute('aria-valuenow', String(pct));
  }

  function render() {
    updateStats();
    renderModuleBar();
    renderGroupBar();

    const showAttitudeTip =
      state.module === 'attitude-correct' ||
      state.module === 'attitude-wrong' ||
      state.module === 'all';
    attitudeCaveat.hidden = !showAttitudeTip;

    const modMeta = modMap.get(state.module);
    moduleTip.textContent = modMeta ? `· ${modMeta.tip}` : '· 覆盖态度正反词、学术名词与英一作文 7 大主题';

    const q = state.search.trim().toLowerCase();
    const filtered = getModulePool().filter(item => {
      const grp = item.group || item.typeLabel;
      if (state.group !== 'all' && grp !== state.group) return false;
      if (state.starOnly && !store.starred.has(item.id)) return false;
      if (state.hideMastered && store.mastered.has(item.id)) return false;
      if (q) {
        const hit =
          item.word.toLowerCase().includes(q) ||
          item.meaning.toLowerCase().includes(q) ||
          item.notes.toLowerCase().includes(q) ||
          grp.toLowerCase().includes(q);
        if (!hit) return false;
      }
      return true;
    });

    countDisplay.textContent = filtered.length;

    if (filtered.length === 0) {
      container.innerHTML = `
        <section class="card empty-state">
          <p class="empty-title">没有找到匹配的速背词条</p>
          <p class="empty-desc">尝试清空搜索关键词或切回“全部分组”。</p>
        </section>
      `;
      return;
    }

    // Group items visually by subgroup
    const grouped = new Map();
    for (const item of filtered) {
      const key = item.group || item.typeLabel;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(item);
    }

    const sectionsHtml = [...grouped.entries()]
      .map(([groupName, groupItems]) => {
        const cardsHtml = groupItems
          .map(item => {
            const isStarred = store.starred.has(item.id);
            const isMastered = store.mastered.has(item.id);
            const isRevealed = !state.selfTest || state.revealedIds.has(item.id);
            const blurCls = isRevealed ? '' : 'is-blurred';
            const isWriting = item.type === 'writing-themes';
            const isMulti = item.words.length > 1;

            return `
              <article class="card mem-card${isMastered ? ' is-mastered' : ''}" data-id="${escapeHtml(item.id)}">
                <div class="mem-card-top">
                  <div class="mem-card-head">
                    <span class="badge-chip mod-${escapeHtml(item.type)}">${escapeHtml(item.typeLabel)}</span>
                    ${
                      item.phonetic && !isWriting
                        ? `<span class="phonetic-text">${escapeHtml(item.phonetic)}</span>`
                        : ''
                    }
                  </div>
                  <div class="row-actions no-print">
                    <button type="button" class="icon-btn" data-mem-action="speak" aria-label="朗读 ${escapeHtml(item.word)}">
                      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/>
                      </svg>
                    </button>
                    ${
                      isWriting
                        ? `<button type="button" class="icon-btn" data-mem-action="copy" aria-label="复制写作表达与例句">
                            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                              <path stroke-linecap="round" stroke-linejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                            </svg>
                           </button>`
                        : ''
                    }
                    <button type="button" class="icon-btn${isStarred ? ' is-starred' : ''}" data-mem-action="star" aria-pressed="${isStarred}" aria-label="${isStarred ? '取消重点' : '标为重点'} ${escapeHtml(item.word)}">
                      <svg class="icon" viewBox="0 0 24 24" fill="${isStarred ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/>
                      </svg>
                    </button>
                    <button type="button" class="icon-btn${isMastered ? ' is-mastered' : ''}" data-mem-action="master" aria-pressed="${isMastered}" aria-label="${isMastered ? '取消已背会' : '标为已背会'} ${escapeHtml(item.word)}">
                      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
                      </svg>
                    </button>
                  </div>
                </div>

                <div class="mem-word-title" lang="en">${escapeHtml(item.word)}</div>
                <div class="mem-meaning ${blurCls}">${escapeHtml(item.meaning)}</div>
                <div class="mem-notes ${blurCls}">${escapeHtml(item.notes)}</div>
                ${
                  state.selfTest
                    ? `<button type="button" class="btn btn-subtle reveal-row-btn no-print" data-mem-action="reveal" aria-expanded="${isRevealed}">${isRevealed ? '重新遮住' : '翻开核对'}</button>`
                    : ''
                }
                ${
                  isMulti && !isWriting
                    ? `<div class="word-chips no-print">${item.words
                        .map(
                          w => `<button type="button" class="speak-word-chip" data-mem-action="speak-one" data-term="${escapeHtml(w)}" lang="en">${escapeHtml(w)}</button>`
                        )
                        .join('')}</div>`
                    : ''
                }
              </article>
            `;
          })
          .join('');

        return `
          <section class="mem-group-section">
            <h3 class="mem-group-heading">
              <span>${escapeHtml(groupName)}</span>
              <span class="count-badge">${groupItems.length}</span>
            </h3>
            <div class="mem-card-grid">${cardsHtml}</div>
          </section>
        `;
      })
      .join('');

    container.innerHTML = sectionsHtml;
  }

  // 事件绑定
  moduleBar.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-mem-mod]');
    if (!btn) return;
    state.module = btn.dataset.memMod;
    state.group = 'all';
    render();
  });

  groupBar.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-mem-group]');
    if (!btn) return;
    state.group = btn.dataset.memGroup;
    render();
  });

  let timer = null;
  searchInput.addEventListener('input', (e) => {
    state.search = e.target.value;
    clearTimeout(timer);
    timer = setTimeout(render, 80);
  });

  selfTestBtn.addEventListener('click', () => {
    state.selfTest = !state.selfTest;
    state.revealedIds.clear();
    selfTestBtn.setAttribute('aria-pressed', String(state.selfTest));
    selfTestLabel.textContent = state.selfTest ? '遮罩默背：开' : '遮罩默背：关';
    render();
    showToast(state.selfTest ? '已开启速背遮罩模式' : '已关闭速背遮罩模式');
  });

  starOnlyBtn.addEventListener('click', () => {
    state.starOnly = !state.starOnly;
    starOnlyBtn.setAttribute('aria-pressed', String(state.starOnly));
    render();
  });

  hideMasteredBtn.addEventListener('click', () => {
    state.hideMastered = !state.hideMastered;
    hideMasteredBtn.setAttribute('aria-pressed', String(state.hideMastered));
    hideMasteredLabel.textContent = state.hideMastered ? '已隐藏已背会' : '隐藏已背会';
    render();
  });

  container.addEventListener('click', async (e) => {
    const btn = e.target.closest('button[data-mem-action]');
    if (!btn) return;
    const card = btn.closest('.mem-card');
    if (!card) return;
    const id = card.dataset.id;
    const item = memItems.find(x => x.id === id);
    if (!item) return;

    const action = btn.dataset.memAction;
    if (action === 'speak') {
      if (item.words.length > 1) speakSequence(item.words);
      else speakText(item.word);
    } else if (action === 'speak-one') {
      speakText(btn.dataset.term || item.word);
    } else if (action === 'star') {
      if (store.starred.has(id)) store.starred.delete(id);
      else store.starred.add(id);
      store.notify('star');
    } else if (action === 'master') {
      if (store.mastered.has(id)) store.mastered.delete(id);
      else store.mastered.add(id);
      store.notify('master');
    } else if (action === 'reveal') {
      if (state.revealedIds.has(id)) state.revealedIds.delete(id);
      else state.revealedIds.add(id);
      render();
    } else if (action === 'copy') {
      const text = `${item.word}（${item.meaning}）\n${item.notes}`;
      try {
        await navigator.clipboard.writeText(text);
        showToast('已复制写作表达与范例句');
      } catch {
        showToast('复制失败，请手动选中复制');
      }
    }
  });

  return { render, updateStats };
}
