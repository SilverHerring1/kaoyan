import { initVocabView } from './vocab.js';
import { initMemorizeView } from './memorize.js';
import { initQuizView } from './quiz.js';
import { initStemSubjectView } from './stem.js';

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
    activeSubject: 'english',
    activePanel: 'panel-vocab',
    starred: new Set(),
    mastered: new Set(),
    answers: {},
    mistakes: new Set(),
    drafts: {},
    syntaxMode: 'full',
    stemStarred: new Set(),
    stemMastered: new Set(),
    stemFlagged: new Set(),
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
        if (Array.isArray(parsed.stemStarred)) this.stemStarred = new Set(parsed.stemStarred);
        if (Array.isArray(parsed.stemMastered)) this.stemMastered = new Set(parsed.stemMastered);
        if (Array.isArray(parsed.stemFlagged)) this.stemFlagged = new Set(parsed.stemFlagged);
      } catch (err) {
        console.warn('读取本地进度失败，将使用内存状态。', err);
      }
    },

    save() {
      try {
        const payload = {
          version: 4,
          updatedAt: new Date().toISOString(),
          starred: [...this.starred],
          mastered: [...this.mastered],
          answers: this.answers,
          mistakes: [...this.mistakes],
          drafts: this.drafts,
          syntaxMode: this.syntaxMode,
          stemStarred: [...this.stemStarred],
          stemMastered: [...this.stemMastered],
          stemFlagged: [...this.stemFlagged],
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      } catch (err) {
        console.warn('保存本地进度失败。', err);
      }
    },

    exportPayload() {
      return {
        app: 'kaoyan-sprint-studio',
        version: 4,
        exportedAt: new Date().toISOString(),
        starred: [...this.starred],
        mastered: [...this.mastered],
        answers: this.answers,
        mistakes: [...this.mistakes],
        drafts: this.drafts,
        stemStarred: [...this.stemStarred],
        stemMastered: [...this.stemMastered],
        stemFlagged: [...this.stemFlagged],
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
      if (Array.isArray(parsed.stemStarred)) this.stemStarred = new Set(parsed.stemStarred);
      if (Array.isArray(parsed.stemMastered)) this.stemMastered = new Set(parsed.stemMastered);
      if (Array.isArray(parsed.stemFlagged)) this.stemFlagged = new Set(parsed.stemFlagged);
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
    'panel-vocab': '核心刷题词库 · 英语一 | 研习台 · Kaoyan Studio',
    'panel-memorize': '纯记忆速背区 · 英语一 | 研习台 · Kaoyan Studio',
    'panel-quiz': '外刊刷题与长难句翻译 · 英语一 | 研习台 · Kaoyan Studio',
    'panel-mistakes': '错题本 · 英语一 | 研习台 · Kaoyan Studio',
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
    if (store.activeSubject === 'english') {
      document.title = titleMap[panelId] || '英语一 | 研习台 · Kaoyan Studio';
    }
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

function initSubjectNav(store, onSubjectChange) {
  const VALID = new Set(['english', 'control', 'math1']);
  const links = [...document.querySelectorAll('.subject-link[data-subject]')];
  const englishTabWrap = document.getElementById('englishTabBarWrap');
  const containers = {
    english: document.getElementById('subject-english'),
    control: document.getElementById('subject-control'),
    math1: document.getElementById('subject-math1'),
  };
  const subtitleEl = document.getElementById('headerSubtitle');
  const subtitleMap = {
    english: '英语一：7 大可刷题核心词模块 · 4 大纯记忆速背模块 · 同源外刊单选与长难句翻译',
    control: '845 自动控制原理（浙大 2027 考纲）：孙优贤《自动控制原理》+《学习辅导》逐页溯源 · SymPy 验算',
    math1: '考研数学一：27 张宇六书（基础30讲 + 高数18讲/线代9讲/概率9讲）+ 27 没咋了线代强化讲义',
  };

  function resolveFromHash() {
    const h = window.location.hash.replace(/^#/, '').trim().toLowerCase();
    return VALID.has(h) ? h : 'english';
  }

  function activate(subjectId, updateHash = false) {
    const target = VALID.has(subjectId) ? subjectId : 'english';
    store.activeSubject = target;

    for (const link of links) {
      const active = link.dataset.subject === target;
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }

    if (englishTabWrap) englishTabWrap.hidden = target !== 'english';
    for (const [id, el] of Object.entries(containers)) {
      if (el) el.hidden = id !== target;
    }
    if (subtitleEl && subtitleMap[target]) {
      subtitleEl.textContent = subtitleMap[target];
    }

    if (updateHash && window.location.hash !== `#${target}`) {
      history.replaceState(null, '', `#${target}`);
    }

    if (onSubjectChange) onSubjectChange(target);
  }

  for (const link of links) {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      activate(link.dataset.subject, true);
    });
  }

  window.addEventListener('hashchange', () => {
    activate(resolveFromHash(), false);
  });

  return { activate, resolveFromHash };
}

async function bootstrap() {
  const store = createStore();

  let controlApi = null;
  let mathApi = null;

  document.getElementById('printBtn').addEventListener('click', () => {
    const summaryEl = document.getElementById('printSummary');
    if (summaryEl) {
      if (store.activeSubject === 'control' && controlApi) {
        summaryEl.textContent = controlApi.getPrintSummary() || '导出科目：845 自动控制原理';
      } else if (store.activeSubject === 'math1' && mathApi) {
        summaryEl.textContent = mathApi.getPrintSummary() || '导出科目：考研数学一';
      }
    }
    window.print();
  });
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

  controlApi = initStemSubjectView(store, {
    subjectId: 'control',
    containerId: 'subject-control',
  });

  mathApi = initStemSubjectView(store, {
    subjectId: 'math1',
    containerId: 'subject-math1',
  });

  const subjectNav = initSubjectNav(store, (subjectId) => {
    if (subjectId === 'english') {
      tabs.activate(store.activePanel, false);
    } else if (subjectId === 'control') {
      controlApi.activate();
    } else if (subjectId === 'math1') {
      mathApi.activate();
    }
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
    if (store.activeSubject === 'english') {
      if (store.activePanel === 'panel-vocab') vocabApi.render();
      if (store.activePanel === 'panel-memorize') memApi.render();
      if (store.activePanel === 'panel-quiz') quizApi.render();
      if (store.activePanel === 'panel-mistakes') quizApi.renderMistakes();
    } else if (store.activeSubject === 'control') {
      controlApi.render();
    } else if (store.activeSubject === 'math1') {
      mathApi.render();
    }
  });

  updateHeaderCounts();
  vocabApi.updateStats();
  vocabApi.render();
  memApi.render();
  quizApi.render();
  quizApi.renderMistakes();

  // Activate initial subject from URL hash (defaults to #english)
  subjectNav.activate(subjectNav.resolveFromHash(), false);
}

window.addEventListener('DOMContentLoaded', bootstrap);
