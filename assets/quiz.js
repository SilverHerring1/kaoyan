import { escapeHtml, showToast, speakText } from './app.js';

const LETTERS = ['A', 'B', 'C', 'D'];
const CORE_ROLES = new Set(['subject', 'verb', 'object', 'complement']);

const LEGEND_ITEMS = [
  { role: 'subject', name: '主语 S', bg: 'var(--syn-subject-bg)', ink: 'var(--syn-subject-ink)' },
  { role: 'verb', name: '谓语 V', bg: 'var(--syn-verb-bg)', ink: 'var(--syn-verb-ink)' },
  { role: 'object', name: '宾语 O', bg: 'var(--syn-object-bg)', ink: 'var(--syn-object-ink)' },
  { role: 'complement', name: '表/补语 C', bg: 'var(--syn-complement-bg)', ink: 'var(--syn-complement-ink)' },
  { role: 'attributive', name: '定语', bg: 'var(--syn-attributive-bg)', ink: 'var(--syn-attributive-ink)' },
  { role: 'adverbial', name: '状语', bg: 'var(--syn-adverbial-bg)', ink: 'var(--syn-adverbial-ink)' },
  { role: 'clause', name: '从句', bg: 'var(--syn-clause-bg)', ink: 'var(--syn-clause-ink)' },
  { role: 'conjunction', name: '逻辑连接', bg: 'var(--syn-conjunction-bg)', ink: 'var(--syn-conjunction-ink)' },
];

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function initQuizView(store, callbacks) {
  const allQuestions = store.questionData.questions || [];
  const modules = (store.vocabData.modules || []).filter(m => m.mode === 'quiz');
  const quizVocabTotal = (store.vocabData.items || []).filter(i => i.mode === 'quiz').length;
  const moduleLabelMap = new Map(modules.map(m => [m.id, m.label]));
  const vocabById = new Map(store.vocabData.items.map(v => [v.id, v]));

  const state = {
    moduleFilter: 'all',
    modeFilter: 'all',
    orderIds: allQuestions.map(q => q.id),
    currentIndex: 0,
    resetConfirmPending: false,
    clearMistakesPending: false,
  };

  const moduleSelect = document.getElementById('quizModuleSelect');
  const modeSelect = document.getElementById('quizModeSelect');
  const shuffleBtn = document.getElementById('quizShuffleBtn');
  const resetSessionBtn = document.getElementById('quizResetSessionBtn');
  const coverageText = document.getElementById('quizCoverageText');
  const positionText = document.getElementById('quizPositionText');
  const accuracyText = document.getElementById('quizAccuracyText');
  const quizStage = document.getElementById('quizStage');
  const mistakesList = document.getElementById('mistakesList');
  const practiceMistakesBtn = document.getElementById('practiceMistakesBtn');
  const clearMistakesBtn = document.getElementById('clearMistakesBtn');

  // 填充模块下拉框
  const qByMod = new Map();
  for (const q of allQuestions) {
    qByMod.set(q.module, (qByMod.get(q.module) || 0) + 1);
  }
  for (const mod of modules) {
    const opt = document.createElement('option');
    opt.value = mod.id;
    opt.textContent = `${mod.label} (${qByMod.get(mod.id) || 0} 题)`;
    moduleSelect.appendChild(opt);
  }

  function getFilteredQuestions() {
    const byId = new Map(allQuestions.map(q => [q.id, q]));
    const ordered = state.orderIds.map(id => byId.get(id)).filter(Boolean);
    return ordered.filter(q => {
      if (state.moduleFilter !== 'all' && q.module !== state.moduleFilter) return false;
      if (state.modeFilter === 'unanswered' && q.id in store.answers) return false;
      if (state.modeFilter === 'wrong' && !store.mistakes.has(q.id)) return false;
      return true;
    });
  }

  function renderPromptSentence(q, answered, isCorrect) {
    if (q.questionType === 'cloze') {
      const fillText = answered ? q.blankAnswer || q.options[q.answer].text : '______';
      const blankClass = answered ? (isCorrect ? 'cloze-blank is-correct' : 'cloze-blank is-wrong') : 'cloze-blank';
      const parts = q.prompt.split('______');
      if (parts.length === 2) {
        return `${escapeHtml(parts[0])}<span class="${blankClass}">${escapeHtml(fillText)}</span>${escapeHtml(parts[1])}`;
      }
      return escapeHtml(q.prompt);
    }

    // Context question (highlight phrase inside original sentence)
    const sent = q.source.sentence;
    if (q.highlight) {
      const idx = sent.toLowerCase().indexOf(q.highlight.toLowerCase());
      if (idx >= 0) {
        const before = sent.slice(0, idx);
        const match = sent.slice(idx, idx + q.highlight.length);
        const after = sent.slice(idx + q.highlight.length);
        return `${escapeHtml(before)}<mark class="context-highlight">${escapeHtml(match)}</mark>${escapeHtml(after)}`;
      }
    }
    return escapeHtml(sent);
  }

  function renderSyntaxBox(q) {
    const mode = store.syntaxMode || 'full';
    const segsHtml = q.syntax.segments
      .map(seg => {
        const isCore = CORE_ROLES.has(seg.role);
        return `
          <span class="syn-seg syn-role-${escapeHtml(seg.role)}${isCore ? ' syn-core' : ' syn-modifier'}">
            <span class="syn-seg-text" lang="en">${escapeHtml(seg.text)}</span>
            <span class="syn-seg-tag">${escapeHtml(seg.label)}</span>
          </span>
        `;
      })
      .join('');

    const legendHtml = LEGEND_ITEMS.map(
      item => `<span class="legend-dot" style="background:${item.bg};color:${item.ink};">${item.name}</span>`
    ).join('');

    return `
      <section class="syntax-box" aria-label="长难句成分划分">
        <div class="syntax-header">
          <h3>长难句成分划分</h3>
          <div class="segmented-control" role="group" aria-label="成分显示模式">
            <button type="button" class="seg-btn" data-syntax-mode="full" aria-pressed="${mode === 'full'}">全成分标注</button>
            <button type="button" class="seg-btn" data-syntax-mode="backbone" aria-pressed="${mode === 'backbone'}">提取主干</button>
            <button type="button" class="seg-btn" data-syntax-mode="plain" aria-pressed="${mode === 'plain'}">原句浏览</button>
          </div>
        </div>
        <div class="syntax-legend" aria-hidden="true">${legendHtml}</div>
        <div class="syntax-sentence" data-mode="${escapeHtml(mode)}">${segsHtml}</div>
        <div class="syntax-backbone-note"><strong>主干提炼：</strong>${escapeHtml(q.syntax.backbone)}</div>
      </section>
    `;
  }

  function renderTranslationBox(q) {
    const draftVal = store.drafts[q.id] || '';
    return `
      <section class="translation-box" aria-label="英一翻译对照练习">
        <label for="transDraft-${escapeHtml(q.id)}">翻译自练（选填，先试译再点开核对，自动保存）</label>
        <textarea id="transDraft-${escapeHtml(q.id)}" data-draft-qid="${escapeHtml(q.id)}" placeholder="试试按上面的句子主干，把这句外刊原句翻译成顺畅的考研中文...">${escapeHtml(draftVal)}</textarea>
        <details class="translation-details">
          <summary>展开参考译文（结构直译 + 考场意译）</summary>
          <div class="translation-compare">
            <div class="trans-col">
              <h4>结构直译（对应英文语序）</h4>
              <p>${escapeHtml(q.translation.literal)}</p>
            </div>
            <div class="trans-col">
              <h4>考场意译（英一采分表达）</h4>
              <p>${escapeHtml(q.translation.polished)}</p>
            </div>
            ${
              q.translation.keyPoint
                ? `<div class="trans-tip"><strong>翻译采分点：</strong>${escapeHtml(q.translation.keyPoint)}</div>`
                : ''
            }
          </div>
        </details>
      </section>
    `;
  }

  function render() {
    const list = getFilteredQuestions();
    coverageText.textContent = `${store.questionsByVocabId.size} / ${quizVocabTotal}`;

    // 统计全部已答正确率
    const answeredIds = Object.keys(store.answers);
    if (answeredIds.length === 0) {
      accuracyText.textContent = '--';
    } else {
      const qMap = new Map(allQuestions.map(q => [q.id, q]));
      let right = 0;
      for (const id of answeredIds) {
        const q = qMap.get(id);
        if (q && store.answers[id] === q.answer) right += 1;
      }
      accuracyText.textContent = `${Math.round((right / answeredIds.length) * 100)}% (${right}/${answeredIds.length})`;
    }

    if (list.length === 0) {
      positionText.textContent = '0 / 0';
      quizStage.innerHTML = `
        <section class="card empty-state">
          <p class="empty-title">当前筛选条件下暂无题目</p>
          <p class="empty-desc">可以切换回“全部模块 / 全部题目”，或运行 <code>scripts/fetch_periodicals.py --guardian</code> 继续扩充题库。</p>
        </section>
      `;
      return;
    }

    if (state.currentIndex >= list.length) state.currentIndex = list.length - 1;
    if (state.currentIndex < 0) state.currentIndex = 0;

    const q = list[state.currentIndex];
    positionText.textContent = `${state.currentIndex + 1} / ${list.length}`;

    const selected = q.id in store.answers ? store.answers[q.id] : null;
    const answered = selected !== null;
    const isCorrect = answered && selected === q.answer;
    const vocab = vocabById.get(q.vocabId);
    const isStarred = store.starred.has(q.vocabId);
    const isMastered = store.mastered.has(q.vocabId);

    const optionsHtml = q.options
      .map((opt, idx) => {
        let stateClass = '';
        let statusTag = '';
        if (answered) {
          if (idx === q.answer) {
            stateClass = ' state-correct';
            statusTag = '<span class="option-status-tag">正确</span>';
          } else if (idx === selected) {
            stateClass = ' state-wrong';
            statusTag = '<span class="option-status-tag">你的选择</span>';
          }
        }
        return `
          <button type="button" class="option-btn${stateClass}" data-option-idx="${idx}" ${answered ? 'disabled' : ''}>
            <span class="option-letter">${LETTERS[idx]}</span>
            <span lang="en">${escapeHtml(opt.text)}</span>
            ${statusTag}
          </button>
        `;
      })
      .join('');

    const explainHtml = answered
      ? `
        <div class="answer-analysis">
          <div class="verdict-bar ${isCorrect ? 'is-correct' : 'is-wrong'}">
            <span>${isCorrect ? '回答正确！' : `答错了，正确答案是 ${LETTERS[q.answer]}（${escapeHtml(q.options[q.answer].text)}），已自动加入错题本`}</span>
            <div class="verdict-actions">
              <button type="button" class="btn btn-subtle" data-quiz-action="retry">重做本题</button>
              <button type="button" class="btn btn-subtle" data-quiz-action="toggle-star">${isStarred ? '已标重点' : '标为重点词'}</button>
              <button type="button" class="btn btn-subtle" data-quiz-action="toggle-master">${isMastered ? '已掌握' : '标为已掌握'}</button>
              ${vocab ? `<button type="button" class="btn btn-subtle" data-quiz-action="view-vocab">去词库看释义</button>` : ''}
            </div>
          </div>

          <ul class="explain-list">
            ${q.options
              .map(
                (opt, idx) => `
                  <li class="explain-item${idx === q.answer ? ' is-answer' : ''}">
                    <strong>${LETTERS[idx]}. ${escapeHtml(opt.text)}</strong>：${escapeHtml(opt.explain)}
                  </li>
                `
              )
              .join('')}
          </ul>

          ${renderSyntaxBox(q)}
          ${renderTranslationBox(q)}
        </div>
      `
      : '';

    const dotsHtml = list
      .map((item, idx) => {
        const hasAns = item.id in store.answers;
        const ok = hasAns && store.answers[item.id] === item.answer;
        const cls = [
          'quiz-dot',
          idx === state.currentIndex ? 'is-current' : '',
          hasAns ? (ok ? 'is-right' : 'is-wrong') : '',
        ]
          .filter(Boolean)
          .join(' ');
        return `<button type="button" class="${cls}" data-jump-idx="${idx}" aria-label="跳到第 ${idx + 1} 题">${idx + 1}</button>`;
      })
      .join('');

    quizStage.innerHTML = `
      <article class="card quiz-card" data-qid="${escapeHtml(q.id)}">
        <div class="quiz-card-top">
          <div class="quiz-tags">
            <span class="badge-chip mod-${escapeHtml(q.module)}">${escapeHtml(moduleLabelMap.get(q.module) || q.module)}</span>
            <span class="badge-chip badge-sky">${escapeHtml(q.examFocus || '外刊精练')}</span>
            ${answered && vocab ? `<span class="quiz-target-word" lang="en">目标词：${escapeHtml(vocab.word)}</span>` : ''}
          </div>
          <div class="quiz-top-tools">
            <button type="button" class="btn btn-subtle" data-quiz-action="speak-sentence" aria-label="朗读外刊原句">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z"/>
              </svg>
              <span>朗读原句</span>
            </button>
          </div>
        </div>

        <p class="question-instruction">${escapeHtml(q.instruction || '根据外刊语境，选出最恰当的一项：')}</p>

        <figure class="quote-figure">
          <blockquote cite="${escapeHtml(q.source.url)}" lang="en">
            ${renderPromptSentence(q, answered, isCorrect)}
          </blockquote>
          <figcaption class="quote-attribution">
            <span>出处：</span>
            <cite>${escapeHtml(q.source.publication)}</cite>
            <span>· ${escapeHtml(q.source.date)} ·</span>
            <a href="${escapeHtml(q.source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(q.source.title)}</a>
          </figcaption>
        </figure>

        <div class="options-grid" role="group" aria-label="选项">
          ${optionsHtml}
        </div>

        ${explainHtml}

        <div class="quiz-nav-footer">
          <button type="button" class="btn btn-subtle" data-quiz-action="prev" ${state.currentIndex === 0 ? 'disabled' : ''}>上一题</button>
          <div class="quiz-dots" aria-label="题号导航">${dotsHtml}</div>
          <button type="button" class="btn btn-primary" data-quiz-action="next" ${state.currentIndex >= list.length - 1 ? 'disabled' : ''}>下一题</button>
        </div>
      </article>
    `;
  }

  function selectOption(q, idx) {
    if (q.id in store.answers) return;
    store.answers[q.id] = idx;
    if (idx !== q.answer) {
      store.mistakes.add(q.id);
    }
    store.notify('answer');
  }

  // 事件委托：作答、成分切换、切题、星标/掌握
  quizStage.addEventListener('click', (e) => {
    const list = getFilteredQuestions();
    const q = list[state.currentIndex];
    if (!q) return;

    const optBtn = e.target.closest('button[data-option-idx]');
    if (optBtn) {
      selectOption(q, Number(optBtn.dataset.optionIdx));
      return;
    }

    const synBtn = e.target.closest('button[data-syntax-mode]');
    if (synBtn) {
      store.syntaxMode = synBtn.dataset.syntaxMode;
      store.notify('syntax-mode');
      return;
    }

    const dotBtn = e.target.closest('button[data-jump-idx]');
    if (dotBtn) {
      state.currentIndex = Number(dotBtn.dataset.jumpIdx);
      render();
      return;
    }

    const actBtn = e.target.closest('button[data-quiz-action]');
    if (!actBtn) return;
    const action = actBtn.dataset.quizAction;

    if (action === 'speak-sentence') {
      speakText(q.source.sentence, 0.92);
    } else if (action === 'prev' && state.currentIndex > 0) {
      state.currentIndex -= 1;
      render();
    } else if (action === 'next' && state.currentIndex < list.length - 1) {
      state.currentIndex += 1;
      render();
    } else if (action === 'retry') {
      delete store.answers[q.id];
      store.notify('retry');
    } else if (action === 'toggle-star') {
      if (store.starred.has(q.vocabId)) store.starred.delete(q.vocabId);
      else store.starred.add(q.vocabId);
      store.notify('star');
    } else if (action === 'toggle-master') {
      if (store.mastered.has(q.vocabId)) store.mastered.delete(q.vocabId);
      else store.mastered.add(q.vocabId);
      store.notify('master');
    } else if (action === 'view-vocab') {
      const vocab = vocabById.get(q.vocabId);
      if (vocab) callbacks.jumpToVocab(vocab.word);
    }
  });

  // 翻译试译草稿自动保存
  quizStage.addEventListener('input', (e) => {
    const ta = e.target.closest('textarea[data-draft-qid]');
    if (!ta) return;
    store.drafts[ta.dataset.draftQid] = ta.value;
    store.save();
  });

  // 刷题键盘快捷键（仅在刷题面板激活且焦点不在输入控件时生效）
  document.addEventListener('keydown', (e) => {
    if (store.activePanel !== 'panel-quiz') return;
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    const list = getFilteredQuestions();
    const q = list[state.currentIndex];
    if (!q) return;

    if (['1', '2', '3', '4'].includes(e.key) && !(q.id in store.answers)) {
      selectOption(q, Number(e.key) - 1);
    } else if (e.key === 'ArrowRight' && state.currentIndex < list.length - 1) {
      state.currentIndex += 1;
      render();
    } else if (e.key === 'ArrowLeft' && state.currentIndex > 0) {
      state.currentIndex -= 1;
      render();
    } else if (e.key.toLowerCase() === 's' && q.id in store.answers) {
      const order = ['full', 'backbone', 'plain'];
      store.syntaxMode = order[(order.indexOf(store.syntaxMode) + 1) % order.length];
      store.notify('syntax-mode');
    }
  });

  // 筛选控制
  moduleSelect.addEventListener('change', (e) => {
    state.moduleFilter = e.target.value;
    state.currentIndex = 0;
    render();
  });

  modeSelect.addEventListener('change', (e) => {
    state.modeFilter = e.target.value;
    state.currentIndex = 0;
    render();
  });

  shuffleBtn.addEventListener('click', () => {
    state.orderIds = shuffleArray(state.orderIds);
    state.currentIndex = 0;
    render();
    showToast('已随机打乱题序');
  });

  let resetTimer = null;
  resetSessionBtn.addEventListener('click', () => {
    if (!state.resetConfirmPending) {
      state.resetConfirmPending = true;
      resetSessionBtn.textContent = '再点一次清空作答';
      resetSessionBtn.classList.add('btn-danger-confirm');
      clearTimeout(resetTimer);
      resetTimer = setTimeout(() => {
        state.resetConfirmPending = false;
        resetSessionBtn.textContent = '清空本轮作答';
        resetSessionBtn.classList.remove('btn-danger-confirm');
      }, 3500);
    } else {
      clearTimeout(resetTimer);
      state.resetConfirmPending = false;
      resetSessionBtn.textContent = '清空本轮作答';
      resetSessionBtn.classList.remove('btn-danger-confirm');
      store.answers = {};
      store.notify('reset-answers');
      showToast('已清空本轮作答记录（错题本保留）');
    }
  });

  // ==================== 错题本视图 ====================
  function renderMistakes() {
    const qMap = new Map(allQuestions.map(q => [q.id, q]));
    const items = [...store.mistakes].map(id => qMap.get(id)).filter(Boolean);

    practiceMistakesBtn.disabled = items.length === 0;

    if (items.length === 0) {
      mistakesList.innerHTML = `
        <section class="card empty-state">
          <p class="empty-title">错题本目前为空</p>
          <p class="empty-desc">去“外刊刷题 &amp; 长难句翻译”刷题，答错的题目会自动收录到这里方便考前复盘。</p>
        </section>
      `;
      return;
    }

    mistakesList.innerHTML = items
      .map(q => {
        const vocab = vocabById.get(q.vocabId);
        return `
          <article class="card mistake-card" data-qid="${escapeHtml(q.id)}">
            <div class="mistake-top">
              <div class="quiz-tags">
                <span class="badge-chip mod-${escapeHtml(q.module)}">${escapeHtml(moduleLabelMap.get(q.module) || q.module)}</span>
                <strong lang="en">${escapeHtml(vocab ? vocab.word : q.vocabId)}</strong>
                <span class="badge-chip badge-sky">正解：${LETTERS[q.answer]}. ${escapeHtml(q.options[q.answer].text)}</span>
              </div>
              <div class="mistakes-actions">
                <button type="button" class="btn btn-primary" data-mistake-action="practice-one">重练此题</button>
                <button type="button" class="btn btn-subtle" data-mistake-action="remove-one">移出错题本</button>
              </div>
            </div>
            <p class="mistake-sentence" lang="en">${escapeHtml(q.source.sentence)}</p>
            <div class="mistake-meta">
              <strong>主干：</strong>${escapeHtml(q.syntax.backbone)}<br>
              <strong>译文：</strong>${escapeHtml(q.translation.polished)}
            </div>
          </article>
        `;
      })
      .join('');
  }

  mistakesList.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-mistake-action]');
    if (!btn) return;
    const card = btn.closest('.mistake-card');
    if (!card) return;
    const qid = card.dataset.qid;
    const action = btn.dataset.mistakeAction;

    if (action === 'remove-one') {
      store.mistakes.delete(qid);
      store.notify('remove-mistake');
      showToast('已移出错题本');
    } else if (action === 'practice-one') {
      delete store.answers[qid];
      store.notify('retry-mistake');
      callbacks.switchTab('panel-quiz');
      focusQuestionById(qid);
    }
  });

  practiceMistakesBtn.addEventListener('click', () => {
    if (store.mistakes.size === 0) return;
    for (const qid of store.mistakes) {
      delete store.answers[qid];
    }
    state.moduleFilter = 'all';
    state.modeFilter = 'wrong';
    moduleSelect.value = 'all';
    modeSelect.value = 'wrong';
    state.currentIndex = 0;
    store.notify('practice-mistakes');
    callbacks.switchTab('panel-quiz');
  });

  let clearMistakesTimer = null;
  clearMistakesBtn.addEventListener('click', () => {
    if (store.mistakes.size === 0) {
      showToast('错题本已经是空的');
      return;
    }
    if (!state.clearMistakesPending) {
      state.clearMistakesPending = true;
      clearMistakesBtn.textContent = '再点一次确认清空';
      clearMistakesBtn.classList.add('btn-danger-confirm');
      clearTimeout(clearMistakesTimer);
      clearMistakesTimer = setTimeout(() => {
        state.clearMistakesPending = false;
        clearMistakesBtn.textContent = '清空错题本';
        clearMistakesBtn.classList.remove('btn-danger-confirm');
      }, 3500);
    } else {
      clearTimeout(clearMistakesTimer);
      state.clearMistakesPending = false;
      clearMistakesBtn.textContent = '清空错题本';
      clearMistakesBtn.classList.remove('btn-danger-confirm');
      store.mistakes.clear();
      store.notify('clear-mistakes');
      showToast('已清空错题本');
    }
  });

  function focusQuestionById(qid) {
    state.moduleFilter = 'all';
    state.modeFilter = 'all';
    moduleSelect.value = 'all';
    modeSelect.value = 'all';
    const list = getFilteredQuestions();
    const idx = list.findIndex(item => item.id === qid);
    if (idx >= 0) {
      state.currentIndex = idx;
    }
    render();
  }

  return {
    render,
    renderMistakes,
    focusQuestionById,
  };
}
