from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"

INPUT_PATH = DATA_DIR / "publisher_mapping.csv"
OUT_LOOKUP = DATA_DIR / "publisher_to_class.csv"
OUT_SUMMARY = DATA_DIR / "publisher_mapping_summary.csv"

def parse_publishers_cell(value) -> list[str]:
    """Turn "['Valve', 'EA']" (string) into ['Valve', 'EA'] (list)."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # fall back to treating the whole thing as one publisher name
        return [s]

    if isinstance(parsed, str):
        return [parsed.strip()]
    if isinstance(parsed, (list, tuple, set)):
        return [str(p).strip() for p in parsed if str(p).strip()]
    return [str(parsed).strip()]


def read_mapping(path: Path) -> pd.DataFrame:
    """
    Read the CSV robustly. Handles:
      - quoted fields      -> Indie,"['A','B']"
      - unquoted fields    -> Indie,['A']
      - no header row
      - comments / empty lines
      - BOM and non-ASCII names
    """
    raw = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

    if not lines:
        return pd.DataFrame(columns=["publisherClass", "publishers"])

    header = lines[0].lower()
    has_header = ("publisherclass" in header) and ("publishers" in header)
    body = lines[1:] if has_header else lines

    records: list[dict[str, str]] = []
    for ln in body:
        if "," not in ln:
            records.append({"publisherClass": ln.strip(), "publishers": "[]"})
            continue

        cls, rest = ln.split(",", 1)
        cls = cls.strip()
        rest = rest.strip()

        # strip outer CSV quotes if present
        if len(rest) >= 2 and rest[0] == '"' and rest[-1] == '"':
            rest = rest[1:-1]

        records.append({"publisherClass": cls, "publishers": rest})

    return pd.DataFrame(records)

def main() -> int:
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input file not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    print(f"Reading {INPUT_PATH}")
    df = read_mapping(INPUT_PATH)
    print(f"Raw rows read: {len(df)}")

    # parse the list column
    df["publishers_list"] = df["publishers"].apply(parse_publishers_cell)

    # explode one row per (class, publisher)
    long_df = (
        df.explode("publishers_list")
          .rename(columns={"publishers_list": "publisher"})
          [["publisherClass", "publisher"]]
          .dropna()
    )
    long_df["publisherClass"] = long_df["publisherClass"].astype(str).str.strip()
    long_df["publisher"] = long_df["publisher"].astype(str).str.strip()
    long_df = long_df[(long_df["publisherClass"] != "") & (long_df["publisher"] != "")]
    long_df = long_df.drop_duplicates().reset_index(drop=True)

    print(f"Expanded to {len(long_df)} (class, publisher) pairs")

    # detect publishers assigned to more than one class
    classes_for: dict[str, set[str]] = defaultdict(set)
    for _, r in long_df.iterrows():
        classes_for[r["publisher"]].add(r["publisherClass"])

    conflicts = {p: sorted(c) for p, c in classes_for.items() if len(c) > 1}
    if conflicts:
        print(f"[WARN] {len(conflicts)} publisher(s) appear in multiple classes:")
        for p, cs in list(conflicts.items())[:15]:
            print(f"        {p} -> {cs}")
        if len(conflicts) > 15:
            print(f"        ... and {len(conflicts) - 15} more")

    # publisher -> class lookup (resolve conflicts deterministically)
    priority = {"AAA": 0, "AA": 1, "Indie": 2, "Hobbyist": 3}
    lookup_rows = []
    for p, cs in classes_for.items():
        chosen = sorted(cs, key=lambda c: (priority.get(c, 99), c))[0]
        lookup_rows.append({"publisher": p, "publisherClass": chosen})
    lookup_df = (
        pd.DataFrame(lookup_rows)
          .sort_values("publisher")
          .reset_index(drop=True)
    )

    summary_df = (
        long_df.groupby("publisherClass")["publisher"]
              .nunique()
              .reset_index()
              .rename(columns={"publisher": "n_publishers"})
              .sort_values("n_publishers", ascending=False)
              .reset_index(drop=True)
    )

    # write outputs
    lookup_df.to_csv(OUT_LOOKUP, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print(f"Wrote {OUT_LOOKUP}  ({len(lookup_df)} rows)")
    print(f"Wrote {OUT_SUMMARY} ({len(summary_df)} rows)")
    print("\nSummary by class:")
    print(summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())