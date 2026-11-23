# 考研英语一冲刺工作台（核心刷题词库 · 纯记忆速背区 · 同源外刊刷题与翻译）

面向考研英语一冲刺阶段的纯静态 Web 学习工具，零后端、零构建依赖，可直接通过 **GitHub Desktop + GitHub Pages** 一键部署上线。

## 四大视图与 11 个精编模块（共 625 条词目，覆盖 1000+ 高频表达）

### 1. 核心刷题词库（7 大可刷题模块 · 470 条）
全部支持**逐词点读 + 顺序连读**、**防剧透遮罩自测**、**随机洗牌**与**直达外刊原句单选**：
- **高频非字面词组（65）** + 小品词空间隐喻速查（UP / OUT / OFF / DOWN / OVER / THROUGH / BACK）
- **熟词生义（183）**
- **易混形近词（62）**
- **完形近义词辨析（40 组）**：4 个选项初看意思相近、实则细微语义/情感色彩/适用对象不同的完形高频陷阱
- **高频固定搭配与介词（45）**：完形必考介词搭配与英一翻译断句核心
- **逻辑词与完形（40）**：含 `all but` / `for all` / `anything but` / `but for` 等高危翻译/完形信号词
- **阅读同义替换（35 组）**：原文高阶词 → 题干选项高频改写簇

### 2. 纯记忆速背区（4 大速背模块 · 155 条，不强行出单选）
专攻只需高效记忆与写作替换的内容，支持**极性/主题分组、遮罩默背、逐词发音、一键复制写作例句**：
- **态度题-核心词（28）** & **态度题-必错词（26）**（按极性/排除家族分组，并标注「作者态度 vs 第三方态度」避坑铁律）
- **英一抽象学术名词（45）**（机制与趋势 / 逻辑与证据 / 社会与治理 / 价值与困境）
- **英一作文主题词（56）**（科技与 AI 伦理 / 教育与青年成长 / 文化传承与交流 / 环保与可持续发展 / 社会治理与公共信任 / 心理健康与个人品质 / 图画图表论证表达）

### 3. 外刊刷题 & 长难句翻译
- 全部例句摘自真实公开考研英一同源外刊（*The Guardian*、*The Conversation*、*Aeon* 等），仅引用单句并附刊名、文章标题、日期与原文链接。
- **三档长难句成分划分器**：一键切换「全成分色块标注 / 提取主干（高亮 S-V-O-C、淡化修饰）/ 原句浏览」，附主干提炼公式。
- **翻译自练与对照**：自动保存试译草稿，原生折叠展开「结构直译 + 考场意译 + 采分点」。
- **快捷键**：`1`–`4` 选答案 · `←` / `→` 切题 · `S` 切换成分模式。

### 4. 错题本 & 进度备份
- 答错自动收录，支持一键进入错题专项重练。
- 支持顶部「备份/恢复」导出与导入学习进度 JSON（包含重点词、已掌握、作答记录、错题本与翻译草稿）。

---

## 部署到 GitHub Pages（配合 GitHub Desktop）

1. 用 **GitHub Desktop** 打开或新建仓库后，将本项目文件放入仓库根目录（`.env` 和 `data/raw/` 已被 `.gitignore` 忽略，绝不会被提交）。
2. 提交并 Push 到 `main` 分支。
3. 进入 GitHub 仓库 **Settings → Pages**：
   - Source 选择 **GitHub Actions**（自动使用 `.github/workflows/pages.yml`）；或选择 **Deploy from a branch** → `main` / `/ (root)` 即可。

## 本地预览

```powershell
python -m http.server 8000
```

浏览器访问 `http://localhost:8000`（请勿直接双击 `index.html`，否则浏览器会拦截本地 JSON `fetch`）。

## 持续扩充同源外刊题库（方案 B 流水线）

所有脚本仅用 Python 3 标准库，无需 `pip install`：

1. **配置 The Guardian 免费 Developer Key**（非商业学习用途，限额 1 次/秒、500 次/天）：
   - 注册地址：[The Guardian Open Platform](https://open-platform.theguardian.com/)
   - 复制 `.env.example` 为 `.env`，填入 `GUARDIAN_API_KEY`。
2. **抓取真实外刊候选长难句**（仅针对 7 大可刷题模块，自动节流与断点续传）：
   ```powershell
   # 抓公开 RSS（无需 Key）
   python scripts/fetch_periodicals.py --rss

   # 按模块用 Guardian 历史库补抓
   python scripts/fetch_periodicals.py --guardian --module collocation
   python scripts/fetch_periodicals.py --guardian --module nuance
   ```
3. **精编题目并跑硬校验**：
   ```powershell
   python scripts/validate_data.py
   ```
   严格校验词库引用有效、不含纯记忆模块词条、4 选项无重复、挖空还原逐字一致、成分切片拼接零漏字、出处链接完整。
