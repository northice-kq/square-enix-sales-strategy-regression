from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd



SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"

VGCHARTZ_IN = DATA_DIR / "VGCHARTZ_DATA_v1.csv"
LOOKUP_IN = DATA_DIR / "publisher_to_class.csv"
VGCHARTZ_OUT = DATA_DIR / "VGCHARTZ_DATA_v1.1.csv"
UNMATCHED_OUT = DATA_DIR / "unmatched_publishers.csv"

PUBLISHER_ALIASES = {
    #  EA 
    "ea sports": "electronic arts",
    "ea sports big": "electronic arts",
    "ea games": "electronic arts",
    "ea": "electronic arts",
    "ea sports freestyle": "electronic arts",
    #sony
    "sony computer entertainment": "sony interactive entertainment",
    "sony computer entertainment america": "sony interactive entertainment",
    "sony computer entertainment europe": "sony interactive entertainment",
    "sony computer entertainment inc.": "sony interactive entertainment",
    "sony online entertainment": "sony interactive entertainment",
    "scea": "sony interactive entertainment",
    "scee": "sony interactive entertainment",
    "sony pictures interactive": "sony interactive entertainment",

    #  Microsoft 
    "microsoft game studios": "xbox game studios",
    "microsoft studios": "xbox game studios",

    #  Namco / Bandai
    "namco": "bandai namco",
    "namco bandai": "bandai namco",
    "namco bandai games": "bandai namco",
    "namco bandai games america": "bandai namco",
    "bandai": "bandai namco",
    "bandai namco games": "bandai namco",

    #  Koei / Tecmo 
    "koei": "koei tecmo",
    "tecmo": "koei tecmo",
    "tecmo koei": "koei tecmo",
    "tecmo koei games": "koei tecmo",
    "koei tecmo games": "koei tecmo",

    #  Take-Two / 2K 
    "2k sports": "2k",
    "2k games": "2k",
    "2k play": "2k",
    "rockstar": "rockstar games",


    "buena vista": "disney",
    "buena vista interactive": "disney",
    "level 5": "level-5",
    "level-5": "level-5",

    #  D3 
    "d3 publisher": "d3publisher",
    "d3publisher": "d3publisher",

    #  Disney 
    "disney interactive studios": "disney",
    "disney interactive": "disney",
    "buena vista games": "disney",

    #  Warner 
    "warner bros. interactive entertainment": "warner bros. interactive entertainment",
    "warner bros. interactive": "warner bros. interactive entertainment",
    "warner bros interactive entertainment": "warner bros. interactive entertainment",
    "warner bros interactive": "warner bros. interactive entertainment",
    "wb games": "warner bros. interactive entertainment",

    #  Square Enix family 
    "eidos interactive": "square enix",
    "eidos": "square enix",
    "square enix europe": "square enix",
    "taito": "square enix",

    #  Atari 
    "atari inc.": "atari",
    "atari interactive": "atari",
    "infogrames": "atari",

    #  Ubisoft 
    "ubisoft entertainment": "ubisoft",
    "ubisoft montreal": "ubisoft",

    #  Sega 
    "sega of america": "sega",
    "sega europe": "sega",
    "sega sammy": "sega",

    #  Activision 
    "activision publishing": "activision",
    "activision value": "activision",

    #  Bethesda 
    "zenimax media": "bethesda softworks",

    #  Konami 
    "konami digital entertainment": "konami",

    #  Capcom 
    "capcom usa": "capcom",
    "capcom entertainment": "capcom",
}


# Fallback class for publishers NOT in the Steam lookup at all
FALLBACK_CLASSES = {
    #major 
    "nintendo": "AAA",
    "sony interactive entertainment": "AAA",
    "bandai namco": "AAA",
    "koei tecmo": "AAA",
    "electronic arts": "AAA",
    "xbox game studios": "AAA",
    "square enix": "AAA",
    "2k": "AAA",
    "d3publisher": "AA",
    "disney": "AA",
    "warner bros. interactive entertainment": "AAA",
    "konami": "AAA",
    "capcom": "AAA",
    "ubisoft": "AAA",
    "sega": "AAA",
    "activision": "AAA",
    "atari": "AA",
    "rockstar games": "AAA",
    "bethesda softworks": "AAA",
"take-two interactive": "AAA",
    "take 2 interactive": "AAA",
    "vu games": "AAA",
    "vivendi universal": "AAA",
    "mtv games": "AA",
    "global star software": "AA",
    "hudson soft": "AA",
    "hudson entertainment": "AA",
    "nippon ichi software": "AA",
    "nippon ichi": "AA",
    "snk playmore": "AA",
    "snk": "AA",
    "telltale games": "AA",
    "marvelous interactive": "AA",
    "marvelous entertainment": "AA",
    "kadokawa shoten": "AA",
    "southpeak interactive": "AA",
    "southpeak": "AA",
    "dreamcatcher interactive": "AA",
    "dreamcatcher": "AA",
    "microsoft": "AAA",
    "level-5": "AA",
    "level 5": "AA",
    "kalypso media": "AA",
    "banpresto": "AA",
    "takara tomy": "AA",

    #established 
    "majesco": "AA",
    "majesco entertainment": "AA",
    "atlus": "AA",
    "atlus usa": "AA",
    "midway games": "AA",
    "midway": "AA",
    "nis america": "AA",
    "crave entertainment": "AA",
    "acclaim entertainment": "AA",
    "acclaim": "AA",
    "destineer": "AA",
    "5pb": "AA",
    "5pb.": "AA",
    "5pb. / mages.": "AA",
    "idea factory": "AA",
    "xseed games": "AA",
    "aksys games": "AA",
    "natsume": "AA",
    "southpeak games": "AA",
    "ignition entertainment": "AA",
    "codemasters": "AA",
    "sierra entertainment": "AA",
    "sierra": "AA",
    "vivendi universal games": "AAA",
    "vivendi games": "AAA",
    "thq": "AA",
    "eidos": "AA",
    "gathering of developers": "AA",
    "gotham games": "AA",
    "interplay": "AA",
    "interplay entertainment": "AA",
    "lucasarts": "AA",
    "lucasarts entertainment": "AA",
    "microsoft game studios": "AAA",
    "bethesda": "AAA",
    "zenimax": "AAA",
    "focus home interactive": "AA",
    "deep silver": "AA",
    "dmg entertainment": "Indie",
    "unknown": "Indie",
}

def normalise(s: pd.Series) -> pd.Series:
    """Strip whitespace and lower-case for matching."""
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
    )


# - main
def main() -> int:
    if not VGCHARTZ_IN.exists():
        print(f"[ERROR] Not found: {VGCHARTZ_IN}", file=sys.stderr)
        return 1
    if not LOOKUP_IN.exists():
        print(f"[ERROR] Not found: {LOOKUP_IN}", file=sys.stderr)
        print("        Run scripts/parse_publisher_mapping.py first.", file=sys.stderr)
        return 1

    print(f"[INFO] Reading {VGCHARTZ_IN}")
    games = pd.read_csv(VGCHARTZ_IN, low_memory=False)

    if "publisher" not in games.columns:
        print(f"[ERROR] 'publisher' column not found.", file=sys.stderr)
        print(f"        Columns: {list(games.columns)}", file=sys.stderr)
        return 1

    print(f"[INFO] Reading {LOOKUP_IN}")
    lookup = pd.read_csv(LOOKUP_IN)

    if "publisher" not in lookup.columns or "publisherClass" not in lookup.columns:
        print(f"[ERROR] Lookup must have columns 'publisher' and 'publisherClass'.",
              file=sys.stderr)
        print(f"        Found: {list(lookup.columns)}", file=sys.stderr)
        return 1
    lookup = lookup.dropna(subset=["publisher", "publisherClass"])

    # build normalised keys on both sides
    games["_pub_key"] = normalise(games["publisher"])

    # apply aliases so VGChartz names map to the Steam lookup names
    games["_pub_key"] = games["_pub_key"].replace(PUBLISHER_ALIASES)

    lookup["_pub_key"] = normalise(lookup["publisher"])

    # collapse lookup to one row per normalised key (in case of dupes)
    lookup_clean = (
        lookup[["_pub_key", "publisherClass"]]
        .drop_duplicates(subset=["_pub_key"])
    )

    # merge
    merged = games.merge(
        lookup_clean,
        how="left",
        on="_pub_key",
    )
    # use manual info
    still_missing = merged["publisherClass"].isna()
    merged.loc[still_missing, "publisherClass"] = (
        merged.loc[still_missing, "_pub_key"].map(FALLBACK_CLASSES)
    )
    # Hobbyist just have 11 rows
    merged["publisherClass"] = merged["publisherClass"].replace({"Hobbyist": "Indie"})

    # - matching report 
    n_total = len(merged)
    n_matched = merged["publisherClass"].notna().sum()
    n_unmatched = n_total - n_matched

    print(f"[INFO] Rows in VGChartz:          {n_total}")
    print(f"[INFO] Rows matched to a class:   {n_matched} "
          f"({n_matched / n_total:.1%})")
    print(f"[INFO] Rows without a class:      {n_unmatched} "
          f"({n_unmatched / n_total:.1%})")

    # unique publishers
    unique_pubs = games[["_pub_key", "publisher"]].drop_duplicates("_pub_key")
    matched_pubs = unique_pubs["_pub_key"].isin(lookup_clean["_pub_key"]).sum()
    print(f"[INFO] Unique publishers matched: {matched_pubs} / {len(unique_pubs)} "
          f"({matched_pubs / len(unique_pubs):.1%})")

    # - top unmatched publishers --
    unmatched = (
        merged[merged["publisherClass"].isna()]
        [["publisher"]]
        .drop_duplicates()
    )
    # how many games does each unmatched publisher have?
    unmatched_counts = (
        merged[merged["publisherClass"].isna()]
        .groupby("publisher")
        .size()
        .reset_index(name="n_games")
        .sort_values("n_games", ascending=False)
        .reset_index(drop=True)
    )


    print(f"\n[INFO] Top 20 unmatched publishers by number of games:")
    print(unmatched_counts.head(20).to_string(index=False))


    # unfilled become unknown
    merged["publisherClass"] = merged["publisherClass"].fillna("Unknown")


    # write outputs
    merged = merged.drop(columns=["_pub_key"])
    merged.to_csv(VGCHARTZ_OUT, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Wrote {VGCHARTZ_OUT}   ({len(merged)} rows)")

    unmatched_counts.to_csv(UNMATCHED_OUT, index=False, encoding="utf-8-sig")
    print(f"[INFO] Wrote {UNMATCHED_OUT}  ({len(unmatched_counts)} rows)")

    # quick class distribution in the merged file
    print("\n[INFO] publisherClass distribution (rows):")
    print(
        merged["publisherClass"]
        .value_counts(dropna=False)
        .to_string()
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())