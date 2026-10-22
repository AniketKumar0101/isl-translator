#!/usr/bin/env python3
"""Unzip INCLUDE zips and populate data/raw/<clean-label>/ with .MOV files.

Clean-label rule: strip the 'NN. ' numeric prefix, lowercase.
'you (plural)' -> 'you all'. Copies (not moves) so zips stay intact.
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DL = ROOT / "data" / "raw_dl"
EXTRACTED = RAW_DL / "extracted"
RAW = ROOT / "data" / "raw"

ZIPS = [
    "Greetings_1of2.zip",
    "Greetings_2of2.zip",
    "Pronouns_1of2.zip",
    "Pronouns_2of2.zip",
]

WANT = {
    'alright', 'good afternoon', 'good evening', 'good morning', 'good night',
    'he', 'hello', 'how are you', 'i', 'it', 'pleased', 'she', 'thank you',
    'they', 'we', 'you', 'you all',
}


def clean_label(dirname: str) -> str:
    name = re.sub(r"^\d+\.\s*", "", dirname).lower().strip()
    if name == "you (plural)":
        name = "you all"
    return name


def main():
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    # 1. unzip
    for z in ZIPS:
        zp = RAW_DL / z
        marker = EXTRACTED / (z + ".extracted_ok")
        if not zp.exists():
            print(f"MISSING zip, skipping: {z}")
            continue
        if marker.exists():
            print(f"already extracted, skipping: {z}")
            continue
        print(f"unzipping {z} ...", flush=True)
        try:
            with zipfile.ZipFile(zp) as zh:
                zh.extractall(EXTRACTED)
        except zipfile.BadZipFile:
            print(f"  BAD/INCOMPLETE zip, skipping for now: {z}")
            continue
        marker.touch()
        print(f"  done: {z}")

    # 2. populate data/raw/<clean-label>/
    counts = {}
    for sign_dir in sorted(EXTRACTED.rglob("*")):
        if not sign_dir.is_dir():
            continue
        movs = sorted(sign_dir.glob("*.MOV")) + sorted(sign_dir.glob("*.mov"))
        if not movs:
            continue
        label = clean_label(sign_dir.name)
        if label not in WANT:
            print(f"  skip unwanted sign dir: {sign_dir.relative_to(EXTRACTED)}")
            continue
        dest = RAW / label
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for m in movs:
            target = dest / m.name
            if not target.exists():
                shutil.copy2(m, target)
                n += 1
        counts[label] = counts.get(label, 0) + n
        print(f"  {label}: +{n} videos")

    print("\nFinal per-label counts in data/raw/:")
    total = 0
    for label in sorted(counts):
        actual = len(list((RAW / label).glob("*.MOV"))) + len(list((RAW / label).glob("*.mov")))
        print(f"  {label:15s} {actual}")
        total += actual
    print(f"TOTAL: {total} videos, {len(counts)} labels")
    if set(counts) != WANT:
        print(f"NOTE: label set incomplete (more zips pending). missing={WANT - set(counts)} extra={set(counts) - WANT}")


if __name__ == "__main__":
    main()
