#!/usr/bin/env python3
"""
Strict quality validator for STEM subject datasets (data/control/ and data/math1/).

Enforces:
1. Every manifest chapter file marked ready=true exists and parses as valid JSON.
2. Every item has unique id, valid kind (formula | conclusion | example), and non-empty title.
3. Strict citation rule: source.book, source.printedPage, and source.itemRef MUST all be non-empty.
4. Formula items MUST specify `formulaLatex` and `conditions` (applicable preconditions).
5. Example items MUST specify `problem`, non-empty `steps` array, and `finalAnswer`.
6. LaTeX delimiter sanity check: balanced `$` / `$$` pairs and balanced `{}` braces inside math spans.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBJECTS = ["control", "math1"]
VALID_KINDS = {"formula", "conclusion", "example", "exercise", "summary"}
VALID_WEIGHTS = {"core", "extended"}
VALID_STAGES = {"foundation", "intensive", "both"}
DOUBLE_BS_CMD_RE = re.compile(
    r"(?<!\\)\\\\(boldsymbol|mathrm|mathcal|mathbf|cdot|frac|sqrt|infty|alpha|beta|gamma|omega|zeta|sigma|theta|Delta|Sigma|sum|int|lim|det|left|right|text|dot|ddot|bar|tilde|hat|vec|quad|qquad|le|ge|ne|approx|to|Rightarrow|Leftrightarrow|implies|partial|nabla|times|pm|mp|cos|sin|tan|ln|log|exp|max|min|arg|arctan|arcsin|arccos)\b"
)


def check_latex_delimiters(text: str, ctx: str, errors: list[str]) -> None:
    if not text:
        return
    # Remove escaped dollar signs
    cleaned = text.replace(r"\$", "")
    # Count $$ first
    dd_count = cleaned.count("$$")
    if dd_count % 2 != 0:
        errors.append(f"{ctx}: unbalanced '$$' display math delimiters")
        return
    # Extract $$...$$ blocks and check brace balance & unescaped & inside \text{}
    for idx, block in enumerate(re.findall(r"\$\$(.*?)\$\$", cleaned, flags=re.S)):
        if block.replace(r"\{", "").replace(r"\}", "").count("{") != block.replace(r"\{", "").replace(r"\}", "").count("}"):
            errors.append(f"{ctx}: unbalanced braces in display math block #{idx + 1}")
        if re.search(r"\\text\{[^{}]*?(?<!\\)&[^{}]*?\}", block):
            errors.append(f"{ctx}: unescaped '&' inside \\text{{...}} in display math block #{idx + 1} (breaks KaTeX aligned)")
    # Remove $$...$$ and check single $
    rem = re.sub(r"\$\$.*?\$\$", "", cleaned, flags=re.S)
    if rem.count("$") % 2 != 0:
        errors.append(f"{ctx}: unbalanced inline '$' delimiters")
        return
    for idx, block in enumerate(re.findall(r"\$([^$]+)\$", rem)):
        b = block.replace(r"\{", "").replace(r"\}", "")
        if b.count("{") != b.count("}"):
            errors.append(f"{ctx}: unbalanced braces in inline math #{idx + 1}: {block[:40]}")


def validate_subject(subj: str) -> tuple[int, int, list[str]]:
    errors: list[str] = []
    subj_dir = ROOT / "data" / subj
    manifest_path = subj_dir / "manifest.json"
    if not manifest_path.exists():
        return 0, 0, [f"[{subj}] missing manifest.json"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chapters = manifest.get("chapters", [])
    seen_ids: set[str] = set()
    total_items = 0
    ready_chapters = 0

    for ch in chapters:
        cid = ch.get("id", "")
        if not cid:
            errors.append(f"[{subj}] chapter missing id")
            continue
        if not ch.get("ready", False):
            continue
        ready_chapters += 1
        ch_file = subj_dir / ch.get("file", f"{cid}.json")
        if not ch_file.exists():
            errors.append(f"[{subj}:{cid}] marked ready=true but missing file {ch_file.name}")
            continue

        data = json.loads(ch_file.read_text(encoding="utf-8"))
        items = data.get("items", [])
        if not items:
            errors.append(f"[{subj}:{cid}] empty items list")
        for idx, item in enumerate(items):
            iid = item.get("id", f"{cid}-idx{idx}")
            ctx = f"[{subj}:{cid}:{iid}]"
            if iid in seen_ids:
                errors.append(f"{ctx}: duplicate id")
            seen_ids.add(iid)
            total_items += 1

            kind = item.get("kind")
            if kind not in VALID_KINDS:
                errors.append(f"{ctx}: invalid kind '{kind}'")
            if item.get("examWeight", "core") not in VALID_WEIGHTS:
                errors.append(f"{ctx}: invalid examWeight '{item.get('examWeight')}'")
            if "stage" in item and item.get("stage") not in VALID_STAGES:
                errors.append(f"{ctx}: invalid stage '{item.get('stage')}'")
            for req in ("title", "section"):
                if not str(item.get(req, "")).strip():
                    errors.append(f"{ctx}: missing {req}")
            check_latex_delimiters(item.get("title", ""), f"{ctx}.title", errors)

            # Check for accidental backtick-dollar mixing or unescaped LaTeX control chars (\f, \b, \t, \r)
            def scan_bad_chars(val, field_path):
                if isinstance(val, str):
                    if "`$" in val or "$`" in val:
                        errors.append(f"{ctx}.{field_path}: suspicious backtick adjacent to '$' math delimiter")
                    for ch_code, ch_name in [("\x0c", "\\f (e.g. \\frac)"), ("\x08", "\\b (e.g. \\beta/\\boldsymbol)"), ("\x09", "\\t (e.g. \\tau/\\text)"), ("\x0d", "\\r (e.g. \\right)")]:
                        if ch_code in val:
                            errors.append(f"{ctx}.{field_path}: unescaped JSON control char {ch_name}")
                    m_dbs = DOUBLE_BS_CMD_RE.search(val)
                    if m_dbs:
                        errors.append(f"{ctx}.{field_path}: accidental double-backslash LaTeX command '{m_dbs.group(0)}' (causes KaTeX newline + literal command name)")
                elif isinstance(val, list):
                    for i, elem in enumerate(val):
                        scan_bad_chars(elem, f"{field_path}[{i}]")
                elif isinstance(val, dict):
                    for dk, dv in val.items():
                        scan_bad_chars(dv, f"{field_path}.{dk}")

            for k, v in item.items():
                scan_bad_chars(v, k)

            # Strict citation enforcement
            src = item.get("source", {})
            for sf in ("book", "printedPage", "itemRef"):
                if not str(src.get(sf, "")).strip():
                    errors.append(f"{ctx}: missing strict source.{sf}")

            # Optional structured tables validation (allowed on any kind)
            tables = item.get("tables")
            if tables is not None:
                if not isinstance(tables, list):
                    errors.append(f"{ctx}: tables must be a list")
                else:
                    for ti, tbl in enumerate(tables):
                        if not isinstance(tbl, dict):
                            errors.append(f"{ctx}.tables[{ti}]: must be an object")
                            continue
                        tcap = str(tbl.get("caption", "")).strip()
                        if not tcap:
                            errors.append(f"{ctx}.tables[{ti}]: missing caption")
                        check_latex_delimiters(tcap, f"{ctx}.tables[{ti}].caption", errors)
                        headers = tbl.get("headers", [])
                        rows = tbl.get("rows", [])
                        if not isinstance(headers, list) or len(headers) == 0:
                            errors.append(f"{ctx}.tables[{ti}]: missing headers")
                        if not isinstance(rows, list) or len(rows) == 0:
                            errors.append(f"{ctx}.tables[{ti}]: missing rows")
                        for hi, hcell in enumerate(headers):
                            check_latex_delimiters(str(hcell), f"{ctx}.tables[{ti}].headers[{hi}]", errors)
                        for ri, row in enumerate(rows):
                            if not isinstance(row, list) or (headers and len(row) != len(headers)):
                                errors.append(f"{ctx}.tables[{ti}].rows[{ri}]: column count mismatch")
                                continue
                            for ci, cell in enumerate(row):
                                check_latex_delimiters(str(cell), f"{ctx}.tables[{ti}].rows[{ri}][{ci}]", errors)

            # Kind-specific checks
            if kind == "formula":
                if not str(item.get("formulaLatex", "")).strip():
                    errors.append(f"{ctx}: formula missing formulaLatex")
                if not str(item.get("conditions", "")).strip():
                    errors.append(f"{ctx}: formula missing conditions (applicable preconditions)")
                check_latex_delimiters(item.get("prompt", ""), f"{ctx}.prompt", errors)
                check_latex_delimiters(item.get("formulaLatex", ""), f"{ctx}.formulaLatex", errors)
                check_latex_delimiters(item.get("conditions", ""), f"{ctx}.conditions", errors)
                check_latex_delimiters(item.get("pitfalls", ""), f"{ctx}.pitfalls", errors)
                for si, sym in enumerate(item.get("symbols", []) or []):
                    check_latex_delimiters(sym.get("sym", ""), f"{ctx}.symbols[{si}].sym", errors)
                    check_latex_delimiters(sym.get("desc", ""), f"{ctx}.symbols[{si}].desc", errors)
            elif kind == "conclusion":
                if not str(item.get("summary", "")).strip():
                    errors.append(f"{ctx}: conclusion missing summary")
                if not str(item.get("conditions", "")).strip():
                    errors.append(f"{ctx}: conclusion missing conditions")
                check_latex_delimiters(item.get("summary", ""), f"{ctx}.summary", errors)
                check_latex_delimiters(item.get("conditions", ""), f"{ctx}.conditions", errors)
                check_latex_delimiters(item.get("pitfalls", ""), f"{ctx}.pitfalls", errors)
            elif kind in ("example", "exercise"):
                if not str(item.get("problem", "")).strip():
                    errors.append(f"{ctx}: {kind} missing problem")
                steps = item.get("steps", [])
                if not isinstance(steps, list) or len(steps) == 0:
                    errors.append(f"{ctx}: {kind} missing steps")
                if not str(item.get("finalAnswer", "")).strip():
                    errors.append(f"{ctx}: {kind} missing finalAnswer")
                check_latex_delimiters(item.get("problem", ""), f"{ctx}.problem", errors)
                check_latex_delimiters(item.get("finalAnswer", ""), f"{ctx}.finalAnswer", errors)
                check_latex_delimiters(item.get("takeaway", ""), f"{ctx}.takeaway", errors)
                for si, st in enumerate(steps):
                    check_latex_delimiters(st.get("stepTitle", ""), f"{ctx}.steps[{si}].stepTitle", errors)
                    check_latex_delimiters(st.get("detail", ""), f"{ctx}.steps[{si}].detail", errors)
            elif kind == "summary":
                ov = str(item.get("overview", "") or item.get("summary", "")).strip()
                if not ov:
                    errors.append(f"{ctx}: summary missing overview")
                check_latex_delimiters(ov, f"{ctx}.overview", errors)
                check_latex_delimiters(item.get("takeaway", ""), f"{ctx}.takeaway", errors)
                patterns = item.get("patterns", [])
                if patterns is not None:
                    if not isinstance(patterns, list):
                        errors.append(f"{ctx}: summary patterns must be a list")
                    else:
                        for pi, pat in enumerate(patterns):
                            for pf in ("typeName", "triggers", "method"):
                                if not str(pat.get(pf, "")).strip():
                                    errors.append(f"{ctx}.patterns[{pi}]: missing {pf}")
                                check_latex_delimiters(str(pat.get(pf, "")), f"{ctx}.patterns[{pi}].{pf}", errors)
                            check_latex_delimiters(str(pat.get("pitfall", "")), f"{ctx}.patterns[{pi}].pitfall", errors)
                            check_latex_delimiters(str(pat.get("relatedRefs", "")), f"{ctx}.patterns[{pi}].relatedRefs", errors)

    return ready_chapters, total_items, errors


def main() -> int:
    all_errors: list[str] = []
    for subj in SUBJECTS:
        ch_cnt, item_cnt, errs = validate_subject(subj)
        print(f"[STEM:{subj}] ready chapters={ch_cnt}, items={item_cnt}, errors={len(errs)}")
        all_errors.extend(errs)

    if all_errors:
        print(f"\n[FAIL] Found {len(all_errors)} STEM validation error(s):")
        for e in all_errors:
            print("  -", e)
        return 1
    print("[PASS] All STEM datasets passed strict citation & LaTeX validation with 0 errors!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
