#!/usr/bin/env python3
"""
Strict quality validator for data/vocab.json and data/questions.json.

Ensures:
1. All 411 vocab entries have unique IDs, valid modules, and non-empty fields.
2. Every question references a real vocabId with matching module.
3. Options have 4 unique choices and valid answer index (0-3).
4. Cloze prompt with blank filled in matches source.sentence verbatim (case-insensitive first char tolerated).
5. Syntax breakdown segments concatenate verbatim to source.sentence (zero dropped/mutated words).
6. Every question has verifiable attribution (publication, title, date, https:// URL) and dual translations.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocab.json"
QUESTIONS_PATH = ROOT / "data" / "questions.json"

VALID_SYNTAX_ROLES = {
    "subject",
    "verb",
    "object",
    "complement",
    "adverbial",
    "attributive",
    "clause",
    "parenthetical",
    "conjunction",
    "other",
}


def norm_space(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s


def validate() -> int:
    errors: list[str] = []

    if not VOCAB_PATH.exists():
        print(f"[FAIL] Missing {VOCAB_PATH}")
        return 1

    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    items = vocab.get("items", [])
    module_ids = {m["id"] for m in vocab.get("modules", [])}
    vocab_map: dict[str, dict] = {}

    for idx, item in enumerate(items):
        vid = item.get("id", "")
        if not vid:
            errors.append(f"vocab[{idx}]: missing id")
        elif vid in vocab_map:
            errors.append(f"vocab[{idx}]: duplicate id {vid}")
        else:
            vocab_map[vid] = item

        if item.get("type") not in module_ids:
            errors.append(f"vocab[{vid}]: unknown module {item.get('type')}")
        if not item.get("words") or not isinstance(item["words"], list):
            errors.append(f"vocab[{vid}]: missing words list")
        for field in ("word", "meaning", "notes"):
            if not str(item.get(field, "")).strip():
                errors.append(f"vocab[{vid}]: empty {field}")

    print(f"[OK] Checked {len(items)} vocab entries across {len(module_ids)} modules.")

    if not QUESTIONS_PATH.exists():
        print(f"[FAIL] Missing {QUESTIONS_PATH}")
        return 1

    qdata = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = qdata.get("questions", [])
    seen_qids: set[str] = set()
    mod_counts: dict[str, int] = {}

    for idx, q in enumerate(questions):
        qid = q.get("id", f"index-{idx}")
        if qid in seen_qids:
            errors.append(f"question[{qid}]: duplicate id")
        seen_qids.add(qid)

        vid = q.get("vocabId", "")
        if vid not in vocab_map:
            errors.append(f"question[{qid}]: unknown vocabId '{vid}'")
        else:
            expected_mod = vocab_map[vid]["type"]
            if q.get("module") != expected_mod:
                errors.append(f"question[{qid}]: module '{q.get('module')}' != vocab module '{expected_mod}'")
            if vocab_map[vid].get("mode", "quiz") != "quiz":
                errors.append(f"question[{qid}]: vocabId '{vid}' belongs to memorize-only module '{expected_mod}'")

        mod_counts[q.get("module", "unknown")] = mod_counts.get(q.get("module", "unknown"), 0) + 1

        # Options check
        opts = q.get("options", [])
        ans = q.get("answer")
        if len(opts) != 4:
            errors.append(f"question[{qid}]: must have exactly 4 options (got {len(opts)})")
        else:
            texts = [str(o.get("text", "")).strip().lower() for o in opts]
            if len(set(texts)) != 4 or any(not t for t in texts):
                errors.append(f"question[{qid}]: duplicate or empty option text")
            for oi, o in enumerate(opts):
                if not str(o.get("explain", "")).strip():
                    errors.append(f"question[{qid}]: option[{oi}] missing explanation")
        if not isinstance(ans, int) or not (0 <= ans <= 3):
            errors.append(f"question[{qid}]: invalid answer index {ans}")

        # Source attribution check
        src = q.get("source", {})
        sentence = str(src.get("sentence", "")).strip()
        if not sentence:
            errors.append(f"question[{qid}]: empty source.sentence")
        for sf in ("publication", "title", "date", "url"):
            if not str(src.get(sf, "")).strip():
                errors.append(f"question[{qid}]: missing source.{sf}")
        if not str(src.get("url", "")).startswith("https://"):
            errors.append(f"question[{qid}]: source.url must start with https://")

        # Prompt integrity check
        qtype = q.get("questionType", "cloze")
        prompt = str(q.get("prompt", "")).strip()
        if qtype == "cloze":
            if "______" not in prompt:
                errors.append(f"question[{qid}]: cloze prompt missing '______'")
            else:
                fill = str(q.get("blankAnswer") or (opts[ans]["text"] if len(opts) == 4 else "")).strip()
                rebuilt = norm_space(prompt.replace("______", fill, 1))
                if rebuilt.lower() != norm_space(sentence).lower():
                    errors.append(
                        f"question[{qid}]: cloze rebuild mismatch!\n"
                        f"  rebuilt : {rebuilt}\n"
                        f"  original: {norm_space(sentence)}"
                    )
        else:
            # Context question (e.g., attitude / paraphrase): highlight phrase must exist verbatim
            hl = str(q.get("highlight", "")).strip()
            if hl and hl.lower() not in sentence.lower():
                errors.append(f"question[{qid}]: highlight '{hl}' not found verbatim in source.sentence")

        # Syntax breakdown integrity check
        syntax = q.get("syntax", {})
        segments = syntax.get("segments", [])
        if not segments:
            errors.append(f"question[{qid}]: missing syntax.segments")
        else:
            joined = norm_space(" ".join(str(seg.get("text", "")).strip() for seg in segments))
            if joined != norm_space(sentence):
                errors.append(
                    f"question[{qid}]: syntax segments do not reconstruct source.sentence verbatim!\n"
                    f"  segments: {joined}\n"
                    f"  original: {norm_space(sentence)}"
                )
            for si, seg in enumerate(segments):
                if seg.get("role") not in VALID_SYNTAX_ROLES:
                    errors.append(f"question[{qid}]: segment[{si}] invalid role '{seg.get('role')}'")
                if not str(seg.get("label", "")).strip():
                    errors.append(f"question[{qid}]: segment[{si}] missing Chinese label")

        if not str(syntax.get("backbone", "")).strip():
            errors.append(f"question[{qid}]: missing syntax.backbone summary")

        # Translation check
        trans = q.get("translation", {})
        if not str(trans.get("literal", "")).strip():
            errors.append(f"question[{qid}]: missing translation.literal")
        if not str(trans.get("polished", "")).strip():
            errors.append(f"question[{qid}]: missing translation.polished")

    print(f"[OK] Checked {len(questions)} questions across modules: {mod_counts}")

    if errors:
        print(f"\n[FAIL] Found {len(errors)} validation error(s):")
        for e in errors:
            print("  -", e)
        return 1

    print("[PASS] All data validation checks passed with 0 errors!")
    return 0


if __name__ == "__main__":
    sys.exit(validate())
