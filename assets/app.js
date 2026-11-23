import { initVocabView } from './vocab.js';
import { initMemorizeView } from './memorize.js';
import { initQuizView } from './quiz.js';

const STORAGE_KEY = 'kaoyan_english_sprint_v3';

export function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function createStore() {
  const store = {
    vocabData: null,
    questionData: null,
    questionsByVocabId: new Map(),
    activePanel: 'panel-vocab',
    starred: new Set(),
    mastered: new Set(),
    answers: {},
    mistakes: new Set(),
    drafts: {},
    syntaxMode: 'full',
    listeners: new Set(),

    subscribe(fn) {
      this.listeners.add(fn);
      return () => this.listeners.delete(fn);
    },

    notify(reason = 'update') {
      this.save();
      for (const fn of this.listeners) {
        fn(reason);
      }
    },

    load() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed.starred)) this.starred = new Set(parsed.starred);
        if (Array.isArray(parsed.mastered)) this.mastered = new Set(parsed.mastered);
        if (parsed.answers && typeof parsed.answers === 'object') this.answers = parsed.answers;
        if (Array.isArray(parsed.mistakes)) this.mistakes = new Set(parsed.mistakes);
        if (parsed.drafts && typeof parsed.drafts === 'object') this.drafts = parsed.drafts;
        if (['full', 'backbone', 'plain'].includes(parsed.syntaxMode)) {
          this.syntaxMode = parsed.syntaxMode;
        }
      } catch (err) {
        console.warn('读取本地进度失败，将使用内存状态。', err);
      }
    },

    save() {
      try {
        const payload = {
          version: 3,
          updatedAt: new Date().toISOString(),
          starred: [...this.starred],
          mastered: [...this.mastered],
          answers: this.answers,
          mistakes: [...this.mistakes],
          drafts: this.drafts,
          syntaxMode: this.syntaxMode,
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      } catch (err) {
        console.warn('保存本地进度失败。', err);
      }
    },

    exportPayload() {
      return {
        app: 'kaoyan-english-sprint',
        version: 3,
        exportedAt: new Date().toISOString(),
        starred: [...this.starred],
        mastered: [...this.mastered],
        answers: this.answers,
        mistakes: [...this.mistakes],
        drafts: this.drafts,
      };
    },

    importPayload(parsed) {
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('JSON 格式无效');
      }
      if (Array.isArray(parsed.starred)) this.starred = new Set(parsed.starred);
      if (Array.isArray(parsed.mastered)) this.mastered = new Set(parsed.mastered);
      if (parsed.answers && typeof parsed.answers === 'object') this.answers = parsed.answers;
      if (Array.isArray(parsed.mistakes)) this.mistakes = new Set(parsed.mistakes);
      if (parsed.drafts && typeof parsed.drafts === 'object') this.drafts = parsed.drafts;
      this.notify('import');
    },
  };

  store.load();
  return store;
}

let toastTimer = null;
export function showToast(msg) {
  const el = document.getElementById('toast');
  if (!el) return;
  clearTimeout(toastTimer);
  el.textContent = msg;
  el.hidden = false;
  toastTimer = setTimeout(() => {
    el.hidden = true;
  }, 2400);
}

let speakQueueCancelToken = 0;

export function speakText(rawText, rate = 0.94) {
  try {
    if (!('speechSynthesis' in window)) {
      showToast('当前浏览器不支持语音朗读');
      return;
    }
    speakQueueCancelToken += 1;
    window.speechSynthesis.cancel();
    const clean = String(rawText).replace(/\(.*?\)/g, '').trim();
    if (!clean) return;
    const utter = new SpeechSynthesisUtterance(clean);
    utter.lang = 'en-US';
    utter.rate = rate;
    window.speechSynthesis.speak(utter);
  } catch {
    showToast('语音播报暂不可用');
  }
}

export function speakSequence(words) {
  try {
    if (!('speechSynthesis' in window)) {
      showToast('当前浏览器不支持语音朗读');
      return;
    }
    const list = (words || []).map(w => String(w).trim()).filter(Boolean);
    if (!list.length) return;
    speakQueueCancelToken += 1;
    const token = speakQueueCancelToken;
    window.speechSynthesis.cancel();

    let idx = 0;
    const next = () => {
      if (token !== speakQueueCancelToken || idx >= list.length) return;
      const utter = new SpeechSynthesisUtterance(list[idx]);
      utter.lang = 'en-US';
      utter.rate = 0.92;
      utter.onend = () => {
        idx += 1;
        setTimeout(next, 260);
      };
      window.speechSynthesis.speak(utter);
    };
    next();
  } catch {
    showToast('语音播报暂不可用');
  }
}

function initTabs(store, onTabChange) {
  const tablist = document.querySelector('.primary-tablist');
  const tabs = [...document.querySelectorAll('.primary-tab')];
  const panels = new Map(tabs.map(t => [t.dataset.panel, document.getElementById(t.dataset.panel)]));

  const titleMap = {
    'panel-vocab': '核心刷题词库 | 考研英语一冲刺工作台',
    'panel-memorize': '纯记忆速背区 | 考研英语一冲刺工作台',
    'panel-quiz': '外刊刷题与长难句翻译 | 考研英语一冲刺工作台',
    'panel-mistakes': '错题本 | 考研英语一冲刺工作台',
  };

  function activate(panelId, moveFocus = false) {
    store.activePanel = panelId;
    for (const tab of tabs) {
      const selected = tab.dataset.panel === panelId;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      const panel = panels.get(tab.dataset.panel);
      if (panel) panel.hidden = !selected;
      if (selected && moveFocus) {
        tab.focus();
      }
    }
    document.title = titleMap[panelId] || '考研英语一冲刺工作台';
    if (onTabChange) onTabChange(panelId);
  }

  tablist.addEventListener('click', (e) => {
    const tab = e.target.closest('.primary-tab');
    if (!tab) return;
    activate(tab.dataset.panel, false);
  });

  tablist.addEventListener('keydown', (e) => {
    const idx = tabs.findIndex(t => t === document.activeElement);
    if (idx === -1) return;
    let nextIdx = -1;
    if (e.key === 'ArrowRight') nextIdx = (idx + 1) % tabs.length;
    else if (e.key === 'ArrowLeft') nextIdx = (idx - 1 + tabs.length) % tabs.length;
    else if (e.key === 'Home') nextIdx = 0;
    else if (e.key === 'End') nextIdx = tabs.length - 1;
    if (nextIdx >= 0) {
      e.preventDefault();
      activate(tabs[nextIdx].dataset.panel, true);
    }
  });

  return { activate };
}

function initBackupDialog(store) {
  const dialog = document.getElementById('backupDialog');
  const openBtn = document.getElementById('openBackupBtn');
  const exportBtn = document.getElementById('exportProgressBtn');
  const importInput = document.getElementById('importProgressInput');

  openBtn.addEventListener('click', () => {
    dialog.showModal();
  });

  dialog.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close();
  });

  exportBtn.addEventListener('click', () => {
    const data = JSON.stringify(store.exportPayload(), null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const dateStr = new Date().toISOString().slice(0, 10);
    a.href = url;
    a.download = `kaoyan-vocab-progress-${dateStr}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('进度 JSON 已导出');
  });

  importInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      store.importPayload(parsed);
      showToast('进度已成功恢复');
      dialog.close();
    } catch (err) {
      showToast(`导入失败：${err.message || '文件格式不正确'}`);
    } finally {
      importInput.value = '';
    }
  });
}

function initBackToTop() {
  const btn = document.getElementById('backToTopBtn');
  window.addEventListener(
    'scroll',
    () => {
      btn.hidden = window.scrollY < 420;
    },
    { passive: true }
  );
  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

async function bootstrap() {
  const store = createStore();

  document.getElementById('printBtn').addEventListener('click', () => window.print());
  initBackupDialog(store);
  initBackToTop();

  try {
    const [vocabRes, qRes] = await Promise.all([
      fetch('./data/vocab.json'),
      fetch('./data/questions.json'),
    ]);
    if (!vocabRes.ok || !qRes.ok) {
      throw new Error(`HTTP ${vocabRes.status} / ${qRes.status}`);
    }
    store.vocabData = await vocabRes.json();
    store.questionData = await qRes.json();
  } catch (err) {
    const main = document.getElementById('main-content');
    main.innerHTML = `
      <section class="card empty-state">
        <p class="empty-title">无法加载词库或题库 JSON 数据</p>
        <p class="empty-desc">如果你是直接双击本地 HTML 文件（<code>file://</code>）打开，浏览器会拦截本地 <code>fetch</code> 请求。请在项目根目录运行 <code>python -m http.server 8000</code> 后访问 <code>http://localhost:8000</code>，或部署到 GitHub Pages 使用。</p>
        <p class="empty-desc">错误信息：${escapeHtml(err.message)}</p>
      </section>
    `;
    return;
  }

  for (const q of store.questionData.questions || []) {
    if (!store.questionsByVocabId.has(q.vocabId)) {
      store.questionsByVocabId.set(q.vocabId, []);
    }
    store.questionsByVocabId.get(q.vocabId).push(q);
  }

  let quizApi = null;
  let vocabApi = null;
  let memApi = null;

  const tabs = initTabs(store, (panelId) => {
    if (panelId === 'panel-vocab' && vocabApi) vocabApi.render();
    if (panelId === 'panel-memorize' && memApi) memApi.render();
    if (panelId === 'panel-quiz' && quizApi) quizApi.render();
    if (panelId === 'panel-mistakes' && quizApi) quizApi.renderMistakes();
  });

  vocabApi = initVocabView(store, {
    jumpToQuiz(vocabId) {
      const qs = store.questionsByVocabId.get(vocabId);
      if (!qs || !qs.length) {
        showToast('该词条暂未收录外刊题，运行采集脚本后可补充');
        return;
      }
      tabs.activate('panel-quiz', false);
      quizApi.focusQuestionById(qs[0].id);
    },
  });

  memApi = initMemorizeView(store);

  quizApi = initQuizView(store, {
    switchTab(panelId) {
      tabs.activate(panelId, false);
    },
    jumpToVocab(word) {
      tabs.activate('panel-vocab', false);
      vocabApi.searchWord(word);
    },
  });

  function updateHeaderCounts() {
    const quizVocabCount = store.vocabData.items.filter(i => i.mode === 'quiz').length;
    const memVocabCount = store.vocabData.items.filter(i => i.mode === 'memorize').length;
    document.getElementById('tabVocabCount').textContent = quizVocabCount;
    document.getElementById('tabMemorizeCount').textContent = memVocabCount;
    document.getElementById('tabQuizCount').textContent = store.questionData.questions.length;
    document.getElementById('tabMistakeCount').textContent = store.mistakes.size;
  }

  store.subscribe(() => {
    updateHeaderCounts();
    vocabApi.updateStats();
    memApi.updateStats();
    if (store.activePanel === 'panel-vocab') vocabApi.render();
    if (store.activePanel === 'panel-memorize') memApi.render();
    if (store.activePanel === 'panel-quiz') quizApi.render();
    if (store.activePanel === 'panel-mistakes') quizApi.renderMistakes();
  });

  updateHeaderCounts();
  vocabApi.updateStats();
  vocabApi.render();
  memApi.render();
  quizApi.render();
  quizApi.renderMistakes();
}

window.addEventListener('DOMContentLoaded', bootstrap);
