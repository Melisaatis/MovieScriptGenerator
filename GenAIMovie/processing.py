# scripts_to_jsonl.py
import re
import json
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import random

SCRIPTS_ROOT = Path("scripts")
OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)
TRAIN_PATH = OUT_DIR / "train.jsonl"
VALID_PATH = OUT_DIR / "valid.jsonl"

# heuristics
SCENE_REGEX = re.compile(r'^(INT\.|EXT\.).*', re.IGNORECASE)
CHAR_REGEX = re.compile(r'^[A-Z][A-Z0-9 \-()\.]{1,60}$')  # simple uppercase line heuristic
PAREN_REGEX = re.compile(r'^\(.*\)$')
DIALOGUE_REGEX = re.compile(r'^[^\s].*')  # not empty, not indented check (loose)

def read_script(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # normalize CRLF and split into lines
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    # filter out many empty lines
    lines = [ln for ln in lines if ln != ""]
    return lines

def chunk_scenes(lines: List[str]) -> List[List[str]]:
    """Split by scene headers heuristically."""
    scenes = []
    cur = []
    for ln in lines:
        if SCENE_REGEX.match(ln) and cur:
            scenes.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        scenes.append(cur)
    return scenes

def extract_pairs_from_scene(scene_lines: List[str]) -> List[Tuple[str, str]]:
    """
    Return list of (prompt, completion) pairs from a scene.
    We'll look for pattern: ... ACTION ... CHAR_NAME \n DIALOGUE...
    """
    pairs = []
    i = 0
    last_action = []
    scene_header = ""
    if scene_lines and SCENE_REGEX.match(scene_lines[0]):
        scene_header = scene_lines[0]
        start_idx = 1
    else:
        start_idx = 0

    i = start_idx
    while i < len(scene_lines):
        ln = scene_lines[i]

        # scene header
        if SCENE_REGEX.match(ln):
            scene_header = ln
            last_action = []
            i += 1
            continue

        # character line?
        if CHAR_REGEX.match(ln) and not PAREN_REGEX.match(ln):
            char = ln.strip()
            # gather subsequent dialogue lines until next uppercase char or blank
            j = i + 1
            dialogue_lines = []
            # skip parentheses lines immediately after character
            while j < len(scene_lines):
                cand = scene_lines[j]
                if PAREN_REGEX.match(cand):
                    j += 1
                    continue
                # stop when next uppercase "CHAR" like line encountered
                if CHAR_REGEX.match(cand) and len(cand.split()) < 6:
                    break
                # treat as dialogue/action lines otherwise
                if cand.strip():
                    dialogue_lines.append(cand.strip())
                j += 1

            if dialogue_lines:
                prompt = f"SCENE: {scene_header}\nACTION: {' '.join(last_action[-3:])}\nCHAR: {char}\nDIALOG:"
                completion = " ".join(dialogue_lines).strip()
                # add end token if you like; we'll keep raw text
                pairs.append((prompt, completion))

            i = j
            continue

        # otherwise treat as action / description
        if ln.strip():
            last_action.append(ln.strip())
            # also create "continue scene" sample occasionally
            if random.random() < 0.05 and len(last_action) > 0:
                prompt = f"SCENE: {scene_header}\nACTION: {' '.join(last_action)}\nCONTINUE_ACTION:"
                completion = ""  # placeholder for model to continue action; skip for now
        i += 1

    return pairs

def main():
    all_pairs = []
    for path in tqdm(sorted(SCRIPTS_ROOT.rglob("*.txt")), desc="scripts"):
        try:
            lines = read_script(path)
            scenes = chunk_scenes(lines)
            for scene in scenes:
                pairs = extract_pairs_from_scene(scene)
                for p, c in pairs:
                    all_pairs.append({"prompt": p, "completion": " " + c})  # leading space helps tokenization
        except Exception as e:
            print("ERR", path, e)

    # shuffle and split by title-level to avoid leaking (we do naive split)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_valid = max(200, int(0.02 * n))  # 2% or 200 min
    valid = all_pairs[:n_valid]
    train = all_pairs[n_valid:]

    print(f"Total pairs: {n} -> train {len(train)}, valid {len(valid)}")
    with TRAIN_PATH.open("w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with VALID_PATH.open("w", encoding="utf-8") as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
