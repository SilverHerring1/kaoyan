#!/usr/bin/env python3
"""
Extract the 411 base Kaoyan vocab entries from the original HTML and merge
the expanded quiz & memorize-only modules into structured data/vocab.json.

Module layout:
  Quiz-mode modules (mode = "quiz", eligible for real-periodical multiple-choice):
    1. idioms       (65)  - 高频非字面词组
    2. polysemy     (183) - 熟词生义
    3. confusing    (62)  - 易混形近词
    4. nuance       (40)  - 完形近义词辨析 [NEW]
    5. collocation  (45)  - 高频固定搭配与介词 [NEW]
    6. logic        (40)  - 逻辑词与完形 [25 + 15]
    7. paraphrase   (35)  - 阅读同义替换 [22 + 13]
  Memorize-only modules (mode = "memorize", dedicated high-speed recall studio):
    8. attitude-correct (28) - 态度题-核心词（按极性分组）
    9. attitude-wrong   (26) - 态度题-必错词（按排除家族分组）
   10. academic-nouns   (45) - 英一高频抽象学术名词 [NEW]
   11. writing-themes   (56) - 英一作文主题词与句型 [NEW]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from extra_vocab_data import (
    ACADEMIC_NOUN_ITEMS,
    ATTITUDE_CORRECT_GROUPS,
    ATTITUDE_WRONG_GROUPS,
    COLLOCATION_ITEMS,
    LOGIC_EXTRA_ITEMS,
    NUANCE_ITEMS,
    PARAPHRASE_EXTRA_ITEMS,
    WRITING_THEME_ITEMS,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_HTML = Path(r"d:\Downloads\考研英语冲刺高频核心词库（升级版）.html")
OUTPUT_JSON = ROOT / "data" / "vocab.json"

EXPLICIT_WORD_EXPANSIONS: dict[str, list[str]] = {
    "draw on / upon": ["draw on", "draw upon"],
    "give in / away / off / out": ["give in", "give away", "give off", "give out"],
    "turn down / up / to": ["turn down", "turn up", "turn to"],
    "provided (that) / providing": ["provided that", "providing"],
    "vulnerable / susceptible / prone / subject to": [
        "vulnerable to",
        "susceptible to",
        "prone to",
        "subject to",
    ],
    "be contingent on / upon": ["contingent on", "contingent upon"],
    "correspond to / with": ["correspond to", "correspond with"],
    "embark on / upon": ["embark on", "embark upon"],
    "cater to / for": ["cater to", "cater for"],
    "fall prey / victim to": ["fall prey to", "fall victim to"],
    "wrestle / grapple with": ["wrestle with", "grapple with"],
    "hinge / bank / count on": ["bank on", "count on", "hinge on"],
    "yield / succumb to": ["succumb to", "yield to"],
    "be attributed to / attribute A to B": ["attributed to", "attribute"],
    "reconcile A with B": ["reconcile"],
    "deter sb from": ["deter"],
    "deprive sb of sth": ["deprive"],
    "inflict sth on sb": ["inflict"],
    "entitle sb to sth": ["entitle"],
}

PHONETIC_FIXES: dict[str, str] = {
    "dubious / doubtful": "/ˈduːbiəs/ · /ˈdaʊtfl/",
    "approving / favorable": "/əˈpruːvɪŋ/ · /ˈfeɪvərəbl/",
    "admiring / appreciative": "/ədˈmaɪərɪŋ/ · /əˈpriːʃətɪv/",
    "indulgent / permissive": "/ɪnˈdʌldʒənt/ · /pərˈmɪsɪv/",
    "uninterested / aloof": "/ʌnˈɪntrəstɪd/ · /əˈluːf/",
    "perplexed / bewildered": "/pərˈplekst/ · /bɪˈwɪldərd/",
    "evasive / ambiguous": "/ɪˈveɪsɪv/ · /æmˈbɪɡjuəs/",
    "desperate / hopeless": "/ˈdespərət/ · /ˈhoʊpləs/",
}

MODULES = [
    # Quiz-mode modules
    {
        "id": "idioms",
        "mode": "quiz",
        "label": "高频非字面词组",
        "shortLabel": "非字面词组",
        "tip": "直击动词短语与介词习语盲区，可结合上方小品词空间逻辑速查。",
    },
    {
        "id": "polysemy",
        "mode": "quiz",
        "label": "熟词生义",
        "shortLabel": "熟词生义",
        "tip": "考研阅读与翻译最高频失分点：忘掉中学熟义，死磕外刊学术生义。",
    },
    {
        "id": "confusing",
        "mode": "quiz",
        "label": "易混形近词",
        "shortLabel": "易混形近词",
        "tip": "完形与阅读高频拼写陷阱，支持逐词点读对比发音。",
    },
    {
        "id": "nuance",
        "mode": "quiz",
        "label": "完形近义词辨析",
        "shortLabel": "近义辨析",
        "tip": "四个选项初看意思相近，实则感情色彩、程度或适用对象不同——考研英一完形核心杀手。",
    },
    {
        "id": "collocation",
        "mode": "quiz",
        "label": "高频固定搭配与介词",
        "shortLabel": "固定搭配",
        "tip": "完形每年必考 3-5 题固定介词搭配，且直接决定英一长难句翻译断句精度。",
    },
    {
        "id": "logic",
        "mode": "quiz",
        "label": "逻辑词与完形",
        "shortLabel": "逻辑词与完形",
        "tip": "完形、新题型与翻译核心逻辑信号词（含 all but / for all 等高危陷阱）。",
    },
    {
        "id": "paraphrase",
        "mode": "quiz",
        "label": "阅读同义替换",
        "shortLabel": "同义替换",
        "tip": "原文高阶词 → 题干/选项高频改写簇，提升定位与选项比对速度。",
    },
    # Memorize-only modules
    {
        "id": "attitude-correct",
        "mode": "memorize",
        "label": "态度题-核心词",
        "shortLabel": "态度核心词",
        "tip": "阅读作者态度题高频正确项，按极性分组速背（无需硬做单句选词题）。",
    },
    {
        "id": "attitude-wrong",
        "mode": "memorize",
        "label": "态度题-必错词",
        "shortLabel": "态度必错词",
        "tip": "问「作者本人态度」时大概率排除；若题干问文中第三方群体态度，仍须回原文核实。",
    },
    {
        "id": "academic-nouns",
        "mode": "memorize",
        "label": "英一抽象学术名词",
        "shortLabel": "抽象学术词",
        "tip": "阅读主旨、翻译与大作文高频骨架名词，掌握精准中文对应即可大幅提速。",
    },
    {
        "id": "writing-themes",
        "mode": "memorize",
        "label": "英一作文主题词",
        "shortLabel": "作文主题词",
        "tip": "按历年考研英一大作文 7 大高频主题归类，每条配考场可直接套用的高分例句。",
    },
]

MODULE_MAP = {m["id"]: m for m in MODULES}


def expand_words(raw_word: str) -> list[str]:
    if raw_word in EXPLICIT_WORD_EXPANSIONS:
        return EXPLICIT_WORD_EXPANSIONS[raw_word]
    parts = [p.strip() for p in raw_word.split("/") if p.strip()]
    cleaned = [re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", p)).strip() for p in parts]
    # Strip leading "be " for search matching when needed while keeping readable display
    return [c for c in cleaned if c]


def build_search_queries(item: dict) -> list[str]:
    queries: list[str] = []
    words = item["words"]
    notes = item["notes"]
    t = item["type"]

    if t == "polysemy":
        collocations = re.findall(r"[A-Za-z][A-Za-z'-]+(?:\s+[A-Za-z][A-Za-z'-]+){1,4}", notes)
        base = words[0].lower()
        for col in collocations:
            col_clean = col.strip()
            if base[:4] in col_clean.lower() and len(col_clean.split()) <= 4:
                if col_clean not in queries:
                    queries.append(col_clean)
            if len(queries) >= 2:
                break
        if base not in [q.lower() for q in queries]:
            queries.append(words[0])
    elif t == "collocation":
        for w in words[:2]:
            clean = re.sub(r"^be\s+", "", w, flags=re.I).strip()
            if clean and clean not in queries:
                queries.append(clean)
    else:
        for w in words[:4]:
            if w not in queries:
                queries.append(w)

    return queries


def make_entry(
    order: int,
    word: str,
    phonetic: str,
    item_type: str,
    meaning: str,
    notes: str,
    group: str = "",
) -> dict:
    mod = MODULE_MAP[item_type]
    words = expand_words(word)
    blur_sub = item_type in ("logic", "paraphrase", "nuance", "collocation")
    entry = {
        "id": f"{item_type}::{word}",
        "order": order,
        "word": word,
        "words": words,
        "phonetic": phonetic,
        "type": item_type,
        "mode": mod["mode"],
        "typeLabel": mod["label"],
        "meaning": meaning,
        "notes": notes,
        "blurSubLabelInSelfTest": blur_sub,
    }
    if group:
        entry["group"] = group
    entry["searchQueries"] = build_search_queries(entry)
    return entry


def extract(source_path: Path = DEFAULT_SOURCE_HTML, output_path: Path = OUTPUT_JSON) -> list[dict]:
    html = source_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'\{\s*word:\s*"([^"]+)",\s*phonetic:\s*"([^"]+)",\s*type:\s*"([^"]+)",'
        r'\s*typeLabel:\s*"([^"]+)",\s*meaning:\s*"([^"]+)",\s*notes:\s*"([^"]+)"\s*\}'
    )
    matches = pattern.findall(html)
    if not matches:
        raise RuntimeError(f"No vocab entries found in {source_path}")

    by_type: dict[str, list[dict]] = {m["id"]: [] for m in MODULES}

    for word, phonetic, item_type, _type_label, meaning, notes in matches:
        if item_type == "attitude-correct" and word == "compromised / compromising":
            word = "compromising"
            phonetic = "/ˈkɑːmprəmaɪzɪŋ/"
            meaning = "折中的，妥协的（态度选项义）"
            notes = (
                "【校订辨析】考研态度题选项中偶将 compromising 作“折中妥协的”考查（若文中态度鲜明多为干扰项）；"
                "注意日常外刊里 compromised 实为“受损的/受牵连的”，compromising 亦常指“有损名誉的”，切勿混用。"
            )
        elif word in PHONETIC_FIXES:
            phonetic = PHONETIC_FIXES[word]

        grp = ""
        if item_type == "attitude-correct":
            grp = ATTITUDE_CORRECT_GROUPS.get(word, "其他核心态度")
        elif item_type == "attitude-wrong":
            grp = ATTITUDE_WRONG_GROUPS.get(word, "其他干扰态度")

        by_type[item_type].append(
            {
                "word": word,
                "phonetic": phonetic,
                "meaning": meaning,
                "notes": notes,
                "group": grp,
            }
        )

    # Append new & expanded modules
    by_type["nuance"].extend(NUANCE_ITEMS)
    by_type["collocation"].extend(COLLOCATION_ITEMS)
    by_type["logic"].extend(LOGIC_EXTRA_ITEMS)
    by_type["paraphrase"].extend(PARAPHRASE_EXTRA_ITEMS)
    by_type["academic-nouns"].extend(ACADEMIC_NOUN_ITEMS)
    by_type["writing-themes"].extend(WRITING_THEME_ITEMS)

    # Flatten in canonical MODULES order
    items: list[dict] = []
    seen_ids: set[str] = set()
    order = 0
    for mod in MODULES:
        mid = mod["id"]
        for raw in by_type[mid]:
            entry = make_entry(
                order=order,
                word=raw["word"],
                phonetic=raw["phonetic"],
                item_type=mid,
                meaning=raw["meaning"],
                notes=raw["notes"],
                group=raw.get("group", ""),
            )
            if entry["id"] in seen_ids:
                raise RuntimeError(f"Duplicate entry id detected: {entry['id']}")
            seen_ids.add(entry["id"])
            items.append(entry)
            order += 1

    quiz_count = sum(1 for x in items if x["mode"] == "quiz")
    mem_count = sum(1 for x in items if x["mode"] == "memorize")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": "3.0",
        "totalCount": len(items),
        "quizVocabCount": quiz_count,
        "memorizeVocabCount": mem_count,
        "modules": MODULES,
        "items": items,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {len(items)} entries (quiz={quiz_count}, memorize={mem_count}) -> {output_path}")
    for mod in MODULES:
        cnt = sum(1 for x in items if x["type"] == mod["id"])
        print(f"  [{mod['mode']:8s}] {mod['id']:18s}: {cnt:3d}")
    return items


if __name__ == "__main__":
    extract()
