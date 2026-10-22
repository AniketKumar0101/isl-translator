"""Unzip INCLUDE zips and organise videos into data/raw/<clean-label>/.

In-zip layout:  Greetings/48. Hello/MVI_0029.MOV
Output layout:  data/raw/hello/MVI_0029.MOV

clean-label = strip leading "NN. " numeric prefix, lowercase,
              "you (plural)" -> "you all".
Moves (not copies) videos; zips are kept. Verifies final counts.
"""
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DL = ROOT / "data" / "raw_dl"
RAW = ROOT / "data" / "raw"
EXTRACTED = RAW_DL / "extracted"

ZIPS = ["Greetings_1of2.zip", "Greetings_2of2.zip",
        "Pronouns_1of2.zip", "Pronouns_2of2.zip"]


def clean_label(folder_name: str) -> str:
    name = re.sub(r"^\d+\.\s*", "", folder_name).strip().lower()
    return "you all" if name == "you (plural)" else name


def main():
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    for z in ZIPS:
        zp = RAW_DL / z
        if not zp.exists():
            raise SystemExit(f"missing zip: {zp}")
        print(f"unzipping {z} ...", flush=True)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(EXTRACTED)
        print(f"  done: {z}", flush=True)

    moved, collisions, skipped = 0, [], 0
    for sign_dir in sorted(EXTRACTED.rglob("*")):
        if not sign_dir.is_dir():
            continue
        if not re.match(r"^\d+\.\s*", sign_dir.name):
            continue
        label = clean_label(sign_dir.name)
        dest_dir = RAW / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        for vid in sorted(sign_dir.iterdir()):
            if not vid.is_file() or vid.suffix.lower() not in {".mov", ".mp4", ".avi"}:
                skipped += 1
                continue
            dest = dest_dir / vid.name
            if dest.exists():
                collisions.append(str(dest))
                continue
            shutil.move(str(vid), str(dest))
            moved += 1

    # remove now-empty extracted tree
    shutil.rmtree(EXTRACTED)
    print(f"moved {moved} videos, skipped {skipped} non-video files")
    if collisions:
        print(f"COLLISIONS ({len(collisions)}):")
        for c in collisions:
            print("  ", c)
    print("\nper-label counts in data/raw:")
    total = 0
    for d in sorted(RAW.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*")))
            total += n
            print(f"  {d.name}: {n}")
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()
