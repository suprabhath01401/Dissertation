"""Download the 3 benchmark datasets that have an automated fetcher to data/eval/.

This deliberately covers only 3 of the 5 datasets in SYSTEM_EXPLANATION.txt
section 7b — LoCoMo-Plus (direct GitHub download) and CaseHOLD/CUAD
(HuggingFace). ContractNLI and TimeQA are NOT downloaded by this module or
by any other automated feature in this codebase — there is no "download"
API endpoint or UI button anywhere in the app. Those two files were fetched
once, manually, and committed to data/eval/ directly; see the docstrings of
`_load_contractnli`/`_load_timeqa` in benchmarks.py for exactly how and
where from. Do not add automated downloaders for them here.

The earlier LexRAG/ChronoQA/LegalBench/LexGLUE datasets (a superseded
7-dataset suite) have been removed from this module entirely.

Sources
-------
Locomo-Plus https://github.com/xjtuleeyf/Locomo-Plus  (direct download)
CaseHOLD    coastalcph/lex_glue[case_hold]  (HuggingFace)
CUAD        chenghao/cuad_qa  (HuggingFace)

Usage
-----
  python backend/evaluation/download_datasets.py
  python backend/evaluation/download_datasets.py --datasets casehold cuad
  python backend/evaluation/download_datasets.py --force   # re-download even if present
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent.parent / "data" / "eval"

# ---------------------------------------------------------------------------
# Direct-download manifests
# ---------------------------------------------------------------------------

DIRECT_DOWNLOADS: dict[str, dict[str, str]] = {
    "locomoplus": {
        "locomoplus_locomo10.json": (
            "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo10.json"
        ),
        "locomoplus_plus.json": (
            "https://raw.githubusercontent.com/xjtuleeyf/Locomo-Plus/main/data/locomo_plus.json"
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path) -> bool:
    """Stream-download url → dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "legal-rag-eval/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        size_kb = dest.stat().st_size // 1024
        print(f"  ✓ {dest.name} ({size_kb} KB)")
        return True
    except Exception as exc:
        print(f"  ✗ {dest.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Per-source downloaders
# ---------------------------------------------------------------------------

def download_direct(name: str, force: bool = False) -> None:
    """Download every file listed for `name` in DIRECT_DOWNLOADS, skipping existing ones unless forced."""
    files = DIRECT_DOWNLOADS.get(name, {})
    if not files:
        print(f"[{name}] No direct-download entries defined.")
        return
    print(f"\n[{name.upper()}]")
    for filename, url in files.items():
        dest = EVAL_DIR / filename
        if dest.exists() and not force:
            print(f"  Skipping {filename} (exists; use --force to re-download)")
        else:
            _download_file(url, dest)


def _hf_load(hf_id: str, config: str | None, split: str, limit: int | None = None) -> list[dict]:
    """Load a HuggingFace dataset split (optionally sliced/limited) and return it as plain dicts."""
    from datasets import load_dataset as hf_load
    # HF split slicing uses a "split[:limit]" string, not a separate kwarg — build it here
    # so callers can pass a plain int limit instead of learning the slicing syntax.
    actual_split = f"{split}[:{limit}]" if limit else split
    if config:
        ds = hf_load(hf_id, config, split=actual_split)
    else:
        ds = hf_load(hf_id, split=actual_split)
    return [dict(ex) for ex in ds]


def download_casehold(force: bool = False) -> None:
    """Download CaseHOLD via LexGLUE (case_hold config) — avoids deprecated dataset scripts."""
    print("\n[CASEHOLD]")
    dest = EVAL_DIR / "casehold.json"
    if dest.exists() and not force:
        print(f"  Skipping casehold.json (exists; use --force to re-download)")
        return
    try:
        raw = _hf_load("coastalcph/lex_glue", "case_hold", "test", limit=1000)
        samples = [
            {
                "context":    ex.get("context", ""),
                "endings":    ex.get("endings", []),
                "label":      int(ex.get("label", 0)),
                "_benchmark": "casehold",
            }
            for ex in raw
            if ex.get("context")
        ]
        dest.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
        print(f"  ✓ casehold.json ({len(samples)} samples)")
    except Exception as exc:
        print(f"  ✗ CaseHOLD: {exc}")


def download_cuad(force: bool = False) -> None:
    """Download CUAD QA pairs from HuggingFace, keeping only answerable questions, to cuad.json."""
    print("\n[CUAD]")
    dest = EVAL_DIR / "cuad.json"
    if dest.exists() and not force:
        print(f"  Skipping cuad.json (exists; use --force to re-download)")
        return
    try:
        rows = _hf_load("chenghao/cuad_qa", None, "test", limit=1000)
        samples: list[dict] = []
        for ex in rows[:1000]:
            answers = ex.get("answers", {})
            texts = answers.get("text", []) if isinstance(answers, dict) else []
            # CUAD includes unanswerable questions (empty answer list); skip them since
            # this eval needs a ground-truth span to score against.
            if not texts:
                continue
            samples.append({
                "question":    ex.get("question", ""),
                "reference":   texts[0],
                "all_answers": texts,
                "context":     ex.get("context", "")[:2000],
                "_benchmark":  "cuad",
            })
        dest.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
        print(f"  ✓ cuad.json ({len(samples)} answerable samples)")
    except Exception as exc:
        print(f"  ✗ CUAD: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_HF_DOWNLOADERS: dict[str, callable] = {
    "casehold": download_casehold,
    "cuad":     download_cuad,
}

ALL_SOURCES: set[str] = set(DIRECT_DOWNLOADS) | set(ALL_HF_DOWNLOADERS)


def main() -> None:
    """Parse CLI args and run the requested dataset downloaders."""
    parser = argparse.ArgumentParser(description="Download benchmark datasets to data/eval/")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(ALL_SOURCES) + ["all"],
        default=["all"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    want: set[str] = set(args.datasets)
    if "all" in want:
        want = ALL_SOURCES

    if "locomoplus" in want:
        download_direct("locomoplus", force=args.force)

    for name, fn in ALL_HF_DOWNLOADERS.items():
        if name in want:
            fn(force=args.force)

    print(f"\n[Done] Datasets written to {EVAL_DIR}")


if __name__ == "__main__":
    main()
