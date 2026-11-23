#!/usr/bin/env python3
r"""
Render exact printed pages from scanned textbooks in D:\Documents\antigravity-workspace\documents\
into gitignored tmp/pdf_pages/ for zero-hallucination visual inspection and citation verification.

Supports:
  --book sun_main   : 自动控制原理 (孙优贤，王慧，2011第1版)  [printedPage + 9 = pdfPage]
  --book sun_guide  : 自动控制原理学习辅导 (孙优贤，王慧，2017第1版) [printedPage + 9 = pdfPage]
  --book hu_main    : 自动控制原理（第七版）(胡寿松)
  --book liubao     : 现代控制理论（第3版）(刘豹，唐万生)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz
from PIL import Image

DOCS_DIR = Path(r"D:\Documents\antigravity-workspace\documents")
OUT_DIR = Path(__file__).resolve().parent.parent / "tmp" / "pdf_pages"

BOOK_CONFIG = {
    "sun_main": {
        "glob": "自动控制原理 (孙优贤*",
        "offset": 9,
        "label": "孙优贤《自动控制原理》(2011第1版)",
    },
    "sun_guide": {
        "glob": "自动控制原理学习辅导*",
        "offset": 9,
        "label": "孙优贤《自动控制原理学习辅导》(2017第1版)",
    },
    "hu_main": {
        "glob": "自动控制原理（第七版）*",
        "offset": 0,
        "label": "胡寿松《自动控制原理》(第七版)",
    },
    "liubao": {
        "glob": "现代控制理论（第3版） (刘豹*",
        "offset": 0,
        "label": "刘豹《现代控制理论》(第3版)",
    },
}


def render_pages(book_key: str, start_printed: int, end_printed: int, dpi: int = 160, pair: bool = False) -> list[Path]:
    cfg = BOOK_CONFIG[book_key]
    matches = list(DOCS_DIR.glob(cfg["glob"]))
    if not matches:
        raise FileNotFoundError(f"Book not found matching {cfg['glob']}")
    pdf_path = matches[0]
    offset = cfg["offset"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    saved: list[Path] = []
    imgs: list[tuple[int, Image.Image]] = []

    for p_printed in range(start_printed, end_printed + 1):
        pdf_idx = p_printed + offset - 1
        if pdf_idx < 0 or pdf_idx >= len(doc):
            continue
        pix = doc[pdf_idx].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if pair:
            imgs.append((p_printed, img))
        else:
            out = OUT_DIR / f"{book_key}_P{p_printed:03d}.png"
            img.save(out)
            saved.append(out)

    if pair and imgs:
        for i in range(0, len(imgs), 2):
            chunk = imgs[i : i + 2]
            if len(chunk) == 1:
                out = OUT_DIR / f"{book_key}_P{chunk[0][0]:03d}.png"
                chunk[0][1].save(out)
                saved.append(out)
            else:
                (p1, im1), (p2, im2) = chunk
                sheet = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)), (255, 255, 255))
                sheet.paste(im1, (0, 0))
                sheet.paste(im2, (im1.width, 0))
                out = OUT_DIR / f"{book_key}_P{p1:03d}-P{p2:03d}.png"
                sheet.save(out)
                saved.append(out)

    doc.close()
    for s in saved:
        print(f"Rendered: {s}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Render exact printed pages from scanned textbooks.")
    parser.add_argument("--book", required=True, choices=BOOK_CONFIG.keys())
    parser.add_argument("--start", required=True, type=int, help="Start printed page number")
    parser.add_argument("--end", required=True, type=int, help="End printed page number (inclusive)")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--pair", action="store_true", help="Stitch 2 consecutive pages side-by-side")
    args = parser.parse_args()
    render_pages(args.book, args.start, args.end, dpi=args.dpi, pair=args.pair)


if __name__ == "__main__":
    main()
