#!/usr/bin/env python3
"""
Fetch real, verifiable sentences from Kaoyan English I homologous open periodicals.

Supports two complementary legal sources (NO paywall/anti-bot bypassing):
  1. --rss       Open RSS/Atom feeds + open article pages (zero API key required):
                 The Conversation, The Guardian RSS, Aeon, BBC, Nature, NPR, Smithsonian.
  2. --guardian  The Guardian Open Platform Content API (free Developer Tier key in .env,
                 searches 1999-present full text, throttled to <= 1 req/sec, checkpointed).

Outputs local candidate pool to data/raw/candidates.json (gitignored) for AI/human curation
into data/questions.json (single-sentence attributed excerpts).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "vocab.json"
RAW_DIR = ROOT / "data" / "raw"
CANDIDATES_PATH = RAW_DIR / "candidates.json"
CORPUS_CACHE_PATH = RAW_DIR / "rss_corpus.json"
ENV_PATH = ROOT / ".env"

USER_AGENT = "KaoyanEnglishSprintBot/1.0 (Personal Non-Commercial Study Tool)"

RSS_FEEDS = [
    ("The Conversation", "https://theconversation.com/us/articles.atom", True),
    ("The Conversation UK", "https://theconversation.com/uk/articles.atom", True),
    ("The Guardian", "https://www.theguardian.com/uk/commentisfree/rss", True),
    ("The Guardian", "https://www.theguardian.com/science/rss", True),
    ("The Guardian", "https://www.theguardian.com/technology/rss", True),
    ("The Guardian", "https://www.theguardian.com/business/rss", True),
    ("Aeon", "https://aeon.co/feed.rss", True),
    ("BBC News", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", False),
    ("BBC News", "https://feeds.bbci.co.uk/news/business/rss.xml", False),
    ("Smithsonian Magazine", "https://www.smithsonianmag.com/rss/latest_articles/", False),
    ("Nature News", "https://www.nature.com/nature.rss", False),
]

GUARDIAN_SECTIONS = (
    "commentisfree|science|technology|business|environment|society|education|culture|world"
)

ABBREV_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S|U\.K|No|Fig|Jan|Feb|Mar|Apr|Aug|Sep|Oct|Nov|Dec)\."
)

CLAUSE_MARKERS = re.compile(
    r"\b(?:which|that|who|whom|whose|although|though|whereas|while|because|since|unless|whether|however|nevertheless|therefore|despite)\b",
    re.I,
)


def load_dotenv(env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def http_get_text(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def strip_html(raw: str) -> str:
    # Remove script/style blocks first
    cleaned = re.sub(r"<(script|style|noscript|figure|aside)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_paragraphs_from_html(page_html: str) -> str:
    # Prefer <article> or <main> if present to reduce boilerplate
    m = re.search(r"<article[^>]*>(.*?)</article>", page_html, flags=re.S | re.I)
    scope = m.group(1) if m else page_html
    ps = re.findall(r"<p[^>]*>(.*?)</p>", scope, flags=re.S | re.I)
    good = []
    for p in ps:
        txt = strip_html(p)
        if len(txt.split()) >= 12 and not txt.lower().startswith(("subscribe", "sign up", "read more", "photograph:")):
            good.append(txt)
    return " ".join(good)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    # Normalize smart quotes and whitespace
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    # Temporarily shield abbreviations and decimals
    protected = ABBREV_RE.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
    raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)
    out = []
    for s in raw_sents:
        s = s.replace("<DOT>", ".").strip()
        s = re.sub(r"^Photograph:\s*\S+(?:\s+\S+){0,4}\s+(?=[A-Z])", "", s)
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        if not s:
            continue
        words = s.split()
        if not (15 <= len(words) <= 48):
            continue
        if not re.match(r'^[A-Z"]', s):
            continue
        if not s.endswith((".", "?", '"')):
            continue
        if "http://" in s or "https://" in s or "@" in s:
            continue
        # Reject sentences with unmatched double quotes
        if s.count('"') % 2 != 0:
            continue
        out.append(s)
    return out


def score_sentence(sentence: str, matched_term: str) -> float:
    words = sentence.split()
    wc = len(words)
    # Sweet spot for Kaoyan English I: 22-36 words
    length_score = max(0.0, 20.0 - abs(wc - 28) * 0.9)
    clause_hits = len(CLAUSE_MARKERS.findall(sentence))
    punct_bonus = (2.0 if "," in sentence else 0.0) + (3.0 if ";" in sentence or "—" in sentence or " - " in sentence else 0.0)
    # Penalize sentences starting with quotes or heavy digits
    quote_penalty = -4.0 if sentence.startswith('"') else 0.0
    digit_penalty = -3.0 if len(re.findall(r"\d+", sentence)) >= 3 else 0.0
    return round(length_score + min(clause_hits, 4) * 3.5 + punct_bonus + quote_penalty + digit_penalty, 2)


def compile_term_regex(term: str) -> re.Pattern:
    clean_term = re.sub(r"^be\s+", "", term.strip(), flags=re.I).strip()
    escaped = re.escape(clean_term)
    # Allow simple inflection on single words or first verb of phrasal verb
    parts = clean_term.split()
    if len(parts) == 1:
        base = parts[0]
        if len(base) >= 4:
            # Match exact word or simple plural/past/ing
            stem = base[:-1] if base.endswith("e") else base
            pat = rf"\b(?:{re.escape(base)}|{re.escape(stem)}(?:s|es|ed|d|ing|er|ly)?)\b"
        else:
            pat = rf"\b{escaped}\b"
    else:
        first = parts[0]
        rest = r"\s+".join(re.escape(p) for p in parts[1:])
        stem = first[:-1] if first.endswith("e") and len(first) >= 4 else first
        first_pat = rf"(?:{re.escape(first)}|{re.escape(stem)}(?:s|es|ed|d|ing)?)"
        pat = rf"\b{first_pat}\s+{rest}\b"
    return re.compile(pat, re.I)


def parse_feed(xml_text: str, pub_name: str, max_items: int = 20) -> list[dict]:
    articles: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return articles

    # RSS <item>
    for item in root.findall(".//item")[:max_items]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        desc = strip_html(item.findtext("description") or "")
        if title and link.startswith("http"):
            articles.append(
                {
                    "publication": pub_name,
                    "title": html.unescape(title),
                    "url": link,
                    "date": pub_date[:16] if pub_date else "",
                    "summary": desc,
                }
            )

    # Atom <entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall(".//atom:entry", ns)[:max_items]:
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link_el = entry.find("atom:link[@rel='alternate']", ns)
        if link_el is None:
            link_el = entry.find("atom:link", ns)
        link = (link_el.attrib.get("href", "") if link_el is not None else "").strip()
        updated = (
            entry.findtext("atom:published", default="", namespaces=ns)
            or entry.findtext("atom:updated", default="", namespaces=ns)
        )[:10]
        content = strip_html(
            entry.findtext("atom:content", default="", namespaces=ns)
            or entry.findtext("atom:summary", default="", namespaces=ns)
        )
        if title and link.startswith("http"):
            articles.append(
                {
                    "publication": pub_name,
                    "title": html.unescape(title),
                    "url": link,
                    "date": updated,
                    "summary": content,
                }
            )
    return articles


def fetch_rss_corpus(max_per_feed: int = 12, fetch_full_pages: bool = True) -> list[dict]:
    """Download recent open periodical articles and extract clean sentences."""
    corpus: list[dict] = []
    seen_urls: set[str] = set()

    for pub_name, feed_url, allow_full in RSS_FEEDS:
        print(f"[RSS] Fetching {pub_name}: {feed_url}")
        try:
            xml_text = http_get_text(feed_url, timeout=10)
        except Exception as exc:
            print(f"  [WARN] Feed failed: {exc}")
            continue

        entries = parse_feed(xml_text, pub_name, max_items=max_per_feed)
        print(f"  Found {len(entries)} items")
        for art in entries:
            if art["url"] in seen_urls:
                continue
            seen_urls.add(art["url"])
            body = art["summary"]
            if fetch_full_pages and allow_full and len(body.split()) < 250:
                try:
                    page_html = http_get_text(art["url"], timeout=10)
                    extracted = extract_paragraphs_from_html(page_html)
                    if len(extracted) > len(body):
                        body = extracted
                    time.sleep(0.35)
                except Exception:
                    pass
            sents = split_sentences(body)
            if sents:
                corpus.append(
                    {
                        "publication": art["publication"],
                        "title": art["title"],
                        "url": art["url"],
                        "date": art["date"],
                        "sentences": sents,
                    }
                )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_CACHE_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    total_sents = sum(len(a["sentences"]) for a in corpus)
    print(f"[RSS] Saved {len(corpus)} articles ({total_sents} quality sentences) -> {CORPUS_CACHE_PATH}")
    return corpus


def match_vocab_against_corpus(vocab_items: list[dict], corpus: list[dict], candidates: dict) -> int:
    new_hits = 0
    for item in vocab_items:
        vid = item["id"]
        bucket = candidates.setdefault(vid, [])
        existing_sents = {c["sentence"] for c in bucket}
        if len(bucket) >= 5:
            continue

        terms = item.get("words", [item["word"]])
        compiled = [(t, compile_term_regex(t)) for t in terms]
        found_local = []

        for art in corpus:
            for sent in art["sentences"]:
                if sent in existing_sents:
                    continue
                for term, regex in compiled:
                    if regex.search(sent):
                        found_local.append(
                            {
                                "sentence": sent,
                                "matchedTerm": term,
                                "publication": art["publication"],
                                "title": art["title"],
                                "date": art["date"],
                                "url": art["url"],
                                "wordCount": len(sent.split()),
                                "score": score_sentence(sent, term),
                                "via": "rss",
                            }
                        )
                        existing_sents.add(sent)
                        break

        if found_local:
            found_local.sort(key=lambda x: x["score"], reverse=True)
            bucket.extend(found_local[:5])
            bucket.sort(key=lambda x: x["score"], reverse=True)
            candidates[vid] = bucket[:5]
            new_hits += 1
    return new_hits


def fetch_guardian_for_vocab(
    vocab_items: list[dict],
    candidates: dict,
    api_key: str,
    max_requests: int = 450,
    delay: float = 1.15,
) -> int:
    """Query The Guardian Open Platform API for missing/under-covered vocab items."""
    requests_made = 0
    updated = 0

    for item in vocab_items:
        vid = item["id"]
        bucket = candidates.setdefault(vid, [])
        if len(bucket) >= 3:
            continue
        if requests_made >= max_requests:
            print(f"[Guardian] Reached --max-requests={max_requests}. Re-run tomorrow to resume.")
            break

        queries = item.get("searchQueries") or item.get("words") or [item["word"]]
        primary_query = queries[0]
        if " " in primary_query:
            q_param = f'"{primary_query}"'
        else:
            q_param = primary_query

        params = urllib.parse.urlencode(
            {
                "q": q_param,
                "section": GUARDIAN_SECTIONS,
                "page-size": 5,
                "show-fields": "bodyText",
                "order-by": "relevance",
                "api-key": api_key,
            }
        )
        url = f"https://content.guardianapis.com/search?{params}"
        requests_made += 1
        print(f"[Guardian {requests_made}/{max_requests}] {vid} (q={q_param})")

        try:
            raw_json = http_get_text(url, timeout=15)
            data = json.loads(raw_json)
            results = data.get("response", {}).get("results", [])
        except urllib.error.HTTPError as err:
            print(f"  [ERROR] HTTP {err.code}: {err.reason}")
            if err.code in (401, 403, 429):
                print("  Stopping Guardian fetch due to auth/rate limit error.")
                break
            time.sleep(delay)
            continue
        except Exception as exc:
            print(f"  [WARN] Request failed: {exc}")
            time.sleep(delay)
            continue

        terms = item.get("words", [item["word"]])
        compiled = [(t, compile_term_regex(t)) for t in terms]
        existing_sents = {c["sentence"] for c in bucket}
        found_local = []

        for art in results:
            body = art.get("fields", {}).get("bodyText", "")
            title = art.get("webTitle", "")
            web_url = art.get("webUrl", "")
            pub_date = (art.get("webPublicationDate") or "")[:10]
            for sent in split_sentences(body):
                if sent in existing_sents:
                    continue
                for term, regex in compiled:
                    if regex.search(sent):
                        found_local.append(
                            {
                                "sentence": sent,
                                "matchedTerm": term,
                                "publication": "The Guardian",
                                "title": title,
                                "date": pub_date,
                                "url": web_url,
                                "wordCount": len(sent.split()),
                                "score": score_sentence(sent, term),
                                "via": "guardian-api",
                            }
                        )
                        existing_sents.add(sent)
                        break

        if found_local:
            found_local.sort(key=lambda x: x["score"], reverse=True)
            bucket.extend(found_local[:5])
            bucket.sort(key=lambda x: x["score"], reverse=True)
            candidates[vid] = bucket[:5]
            updated += 1
            print(f"  -> +{len(found_local[:5])} candidate sentences")
        else:
            print("  -> 0 matching sentences in top 5 articles")

        # Save checkpoint every 10 calls
        if requests_made % 10 == 0:
            CANDIDATES_PATH.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

        time.sleep(delay)

    return updated


def print_summary(vocab_items: list[dict], candidates: dict) -> None:
    by_mod: dict[str, list[int]] = {}
    for item in vocab_items:
        m = item["type"]
        by_mod.setdefault(m, [0, 0])
        by_mod[m][1] += 1
        if candidates.get(item["id"]):
            by_mod[m][0] += 1

    print("\n=== Candidate Coverage Summary ===")
    total_hit = 0
    for mod, (hit, total) in by_mod.items():
        total_hit += hit
        print(f"  {mod:18s}: {hit:3d} / {total:3d} ({hit / total * 100:5.1f}%)")
    print(f"  {'TOTAL':18s}: {total_hit:3d} / {len(vocab_items):3d} ({total_hit / len(vocab_items) * 100:5.1f}%)")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch Kaoyan English I periodical example candidates.")
    parser.add_argument("--rss", action="store_true", help="Fetch open RSS feeds and match vocab entries.")
    parser.add_argument("--match-cache", action="store_true", help="Match vocab entries against cached rss_corpus.json without re-downloading RSS.")
    parser.add_argument("--rss-max-per-feed", type=int, default=10, help="Max articles per RSS feed.")
    parser.add_argument("--guardian", action="store_true", help="Query Guardian Open Platform API using GUARDIAN_API_KEY.")
    parser.add_argument("--module", type=str, default="", help="Only target a specific module id (e.g. idioms, nuance, collocation).")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of vocab items processed.")
    parser.add_argument("--max-requests", type=int, default=450, help="Max Guardian API requests per run (daily limit=500).")
    args = parser.parse_args()

    if not args.rss and not args.guardian and not args.match_cache:
        parser.print_help()
        print("\nTip: run with --rss, --match-cache, or --guardian (requires GUARDIAN_API_KEY in .env).")
        return 1

    vocab_data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    # Only collect periodical quiz candidates for quiz-mode modules (skip memorize-only modules)
    vocab_items = [x for x in vocab_data["items"] if x.get("mode", "quiz") == "quiz"]
    if args.module:
        vocab_items = [x for x in vocab_items if x["type"] == args.module]
    if args.limit > 0:
        vocab_items = vocab_items[: args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CANDIDATES_PATH.exists():
        candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    else:
        candidates = {}

    if args.match_cache and CORPUS_CACHE_PATH.exists():
        corpus = json.loads(CORPUS_CACHE_PATH.read_text(encoding="utf-8"))
        hits = match_vocab_against_corpus(vocab_items, corpus, candidates)
        print(f"[Cache] Matched {hits} quiz vocab entries from cached RSS corpus.")

    if args.rss:
        corpus = fetch_rss_corpus(max_per_feed=args.rss_max_per_feed, fetch_full_pages=True)
        hits = match_vocab_against_corpus(vocab_items, corpus, candidates)
        print(f"[RSS] Matched {hits} vocab entries with fresh real periodical sentences.")

    if args.guardian:
        api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()
        if not api_key or api_key == "your-guardian-developer-key-here":
            print("[ERROR] GUARDIAN_API_KEY not configured. Copy .env.example to .env and add your free Developer Key.")
            return 2
        fetch_guardian_for_vocab(vocab_items, candidates, api_key=api_key, max_requests=args.max_requests)

    CANDIDATES_PATH.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(vocab_items, candidates)
    print(f"Candidates saved to {CANDIDATES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
