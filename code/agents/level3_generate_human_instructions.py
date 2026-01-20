# generate_level3_green_prompts.py
# Read every *.geojson under ../../data, randomly assign a target ratio percent (20..50),
# and write a sibling JSON file "<stem>_prompt.json" containing ONLY {"prompt": "..."}.

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, List


# ----------------------------
# Prompt template bank (B-style: cautious outward adjustment; no cover/merge/delete)
# - Keep semantics fixed, vary wording/structure.
# - NOTE: You asked to keep "8m spacing" as a soft preference in prompt text only.
# ----------------------------

TEMPLATE_1 = (
    "Expand the existing green polygons so that the overall green area increases by about {RATIO}%.\n"
    "Only adjust areas that are already green, using cautious outward movements that do not intrude into any non-green "
    "regions. All non-green polygons must remain unchanged, and no features should be merged, removed, or deleted.\n"
    "If possible, keep a separation of roughly 8 meters between distinct green areas, but this spacing is optional and "
    "should not override the primary constraints. Ensure all geometry remains valid before exporting in WGS84 (EPSG:4326)."
)

TEMPLATE_2 = (
    "The goal is to achieve an overall increase of approximately {RATIO}% in green area relative to the current green "
    "coverage.\n"
    "Gently extend the boundaries of existing green polygons while preserving the exact shape and extent of all non-green "
    "features. Do not allow green areas to overlap, merge with each other, or cover non-green parcels.\n"
    "Where feasible, try to maintain around 8 meters of spacing between separate green areas. Validate geometry before "
    "exporting in WGS84 (EPSG:4326)."
)

TEMPLATE_3 = (
    "Gradually grow the current green areas until the total green coverage is increased by about {RATIO}%.\n"
    "Each outward adjustment must be small and careful, applied only to polygons that are already green. Non-green areas "
    "must be left exactly as they are, and no polygons may be deleted, merged, or absorbed.\n"
    "If it can be done without violating these constraints, aim for roughly 8 meters between separate green areas. Run "
    "planar validity checks before exporting in WGS84 (EPSG:4326)."
)

TEMPLATE_4 = (
    "Adjust the boundaries of existing green polygons in a conservative manner to obtain roughly {RATIO}% additional green "
    "area overall.\n"
    "Extensions should originate from current green regions only and must not interfere with any non-green parcels. Preserve "
    "all original features—avoid any deletion, merging, or coverage of other areas.\n"
    "Maintaining about 8 meters of spacing between green regions is preferred when feasible. Ensure geometric validity and "
    "export in WGS84 (EPSG:4326)."
)

TEMPLATE_5 = (
    "Increase green-space coverage by approximately {RATIO}% through careful outward adjustments of existing green polygons.\n"
    "Respect surrounding land uses: keep all non-green polygons unchanged and prevent any overlap, merging, or removal of "
    "features. Modify each green polygon independently and avoid distorting the original shapes.\n"
    "If possible, preserve around 8 meters of separation between distinct green areas. Validate the geometry before exporting "
    "in WGS84 (EPSG:4326)."
)

TEMPLATE_6 = (
    "Raise the total green footprint by about {RATIO}% while strictly preserving the current non-green layout.\n"
    "Only existing green polygons may be extended, using cautious outward edits that never cover non-green parcels and never "
    "cause overlaps, merges, deletions, or removals.\n"
    "Keeping roughly 8 meters between separate green areas is a nice-to-have if it does not conflict with the main rules. "
    "Validate geometry and export in WGS84 (EPSG:4326)."
)

TEMPLATE_7 = (
    "Without altering or encroaching on any non-green areas, extend the existing green polygons to reach an overall green-area "
    "increase of approximately {RATIO}%.\n"
    "Proceed in small, controlled steps; do not merge green polygons together and do not delete or remove any features. Leave "
    "all non-green polygons fully intact.\n"
    "If feasible, maintain around 8 meters of spacing between separate green regions. Perform planar validity checks before "
    "exporting in WGS84 (EPSG:4326)."
)

TEMPLATE_8 = (
    "Increase the total green area by approximately {RATIO}% via cautious outward adjustment of existing green polygons.\n"
    "Do not affect non-green areas, and avoid any merging, deletion, removal, or overlap of features. Expand green polygons "
    "independently with minimal geometric distortion.\n"
    "An approximate 8-meter separation between green areas may be preserved when feasible. Ensure validity before exporting in "
    "WGS84 (EPSG:4326)."
)

TEMPLATE_9 = (
    "Enlarge the current green parcels to add about {RATIO}% more green area overall.\n"
    "Only push outward from already-green boundaries, and do not let the green areas spill into any non-green polygons. Keep "
    "every non-green feature unchanged, and do not merge, delete, or remove any polygons.\n"
    "If the layout permits, keep green patches about 8 meters apart. Check geometry validity and export in WGS84 (EPSG:4326)."
)

TEMPLATE_10 = (
    "Bring the total green coverage up by roughly {RATIO}% using careful boundary adjustments on existing green polygons.\n"
    "All edits must stay within the green class: do not cover non-green parcels, do not combine green polygons into one, and do "
    "not delete or remove features.\n"
    "Try to keep around an 8 m gap between separate green areas when feasible. Validate geometry before exporting in WGS84 (EPSG:4326)."
)

PROMPT_TEMPLATES: List[str] = [
    TEMPLATE_1, TEMPLATE_2, TEMPLATE_3, TEMPLATE_4, TEMPLATE_5,
    TEMPLATE_6, TEMPLATE_7, TEMPLATE_8, TEMPLATE_9, TEMPLATE_10,
]


def iter_geojson_files(root: Path) -> Iterator[Path]:
    yield from root.rglob("*.geojson")


def make_prompt(ratio_percent: int, rng: random.Random) -> str:
    tpl = rng.choice(PROMPT_TEMPLATES)
    return tpl.format(RATIO=int(ratio_percent)).strip()


def write_prompt_json(prompt_path: Path, prompt_text: str) -> None:
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt_text}, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data", help="Root folder containing *.geojson")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (reproducible prompts)")
    parser.add_argument("--ratio_min", type=int, default=20, help="Min target increase percent (int, inclusive)")
    parser.add_argument("--ratio_max", type=int, default=50, help="Max target increase percent (int, inclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of files processed")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    rng = random.Random(int(args.seed))

    files = sorted(iter_geojson_files(root))
    if args.limit is not None:
        files = files[: int(args.limit)]

    print(f"[INFO] Scanning: {root}")
    print(f"[INFO] Found {len(files)} geojson files to process")

    for i, p in enumerate(files, start=1):
        ratio = rng.randint(int(args.ratio_min), int(args.ratio_max))  # integer in [min, max]
        prompt_text = make_prompt(ratio, rng)

        out_path = p.with_name(f"{p.stem}_prompt.json")
        write_prompt_json(out_path, prompt_text)

        # show a short preview
        preview = prompt_text.replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:140] + "..."
        print(f"[{i:05d}/{len(files):05d}] {p.name} -> {out_path.name} | ratio={ratio}% | {preview}")

    print("[DONE] Prompt JSON files written (each contains ONLY the key: prompt).")


if __name__ == "__main__":
    main()
