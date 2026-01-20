




# compute_ree_from_prompt_label_and_after.py
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from pyproj import CRS, Transformer

# -----------------------------
# Parsing helpers
# -----------------------------
DIGITS_RE = re.compile(r"(\d{5,})")  # id 通常很长；5+ 只是为了少误匹配
DIST_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(m|meter|meters|metre|metres)\b", re.IGNORECASE)
DIR_RE = re.compile(
    r"\b(north|south|east|west|northeast|northwest|southeast|southwest|ne|nw|se|sw|n|s|e|w)\b",
    re.IGNORECASE,
)

DIR_ALIASES = {
    "north": "north", "n": "north",
    "south": "south", "s": "south",
    "east": "east", "e": "east",
    "west": "west", "w": "west",
    "northeast": "northeast", "north-east": "northeast", "ne": "northeast",
    "northwest": "northwest", "north-west": "northwest", "nw": "northwest",
    "southeast": "southeast", "south-east": "southeast", "se": "southeast",
    "southwest": "southwest", "south-west": "southwest", "sw": "southwest",
}


def normalize_digits_only(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = DIGITS_RE.search(s)
    return m.group(1) if m else None


def parse_prompt_for_id_and_magnitude(prompt_text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """
    从 prompt 文本里尽量抽取：
    - target_id digits-only
    - magnitude m (float meters)  —— 用于 REE 的分母
    - direction (可选，仅作为记录/debug)
    """
    if not prompt_text:
        return None, None, None

    # target_id：优先找 "ID:" 附近，否则退化为找第一个长数字串
    pid = None
    m = re.search(r"\bID\s*[:=]?\s*(node/|way/|relation/)?(\d{5,})\b", prompt_text, re.IGNORECASE)
    if m:
        pid = m.group(2)
    else:
        m2 = DIGITS_RE.search(prompt_text)
        if m2:
            pid = m2.group(1)

    # magnitude (meters): 取 prompt 里第一个出现的 “xx m/meters”
    magnitude_m = None
    dm = DIST_RE.search(prompt_text)
    if dm:
        magnitude_m = float(dm.group(1))

    # direction: 只做记录，不参与 REE（REE 由 label/edit 的几何距离定义）
    direction = None
    dr = DIR_RE.search(prompt_text)
    if dr:
        direction = DIR_ALIASES.get(dr.group(1).lower())

    return pid, magnitude_m, direction


# -----------------------------
# Geo helpers (planar distance)
# -----------------------------
def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def planar_distance_m(p1_lonlat: Tuple[float, float], p2_lonlat: Tuple[float, float]) -> float:
    lon1, lat1 = p1_lonlat
    lon2, lat2 = p2_lonlat
    wgs84 = CRS.from_epsg(4326)
    crs_planar = utm_crs_for_lonlat(lon1, lat1)
    tf = Transformer.from_crs(wgs84, crs_planar, always_xy=True)
    x1, y1 = tf.transform(lon1, lat1)
    x2, y2 = tf.transform(lon2, lat2)
    return float(math.hypot(x2 - x1, y2 - y1))


# -----------------------------
# File/path helpers
# -----------------------------
def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)

# -----------------------------
# GeoJSON feature lookup (Point only)
# -----------------------------
def get_point_lonlat_by_pid(fc: Dict[str, Any], pid_digits: str) -> Tuple[float, float]:
    feats = fc.get("features", [])
    for feat in feats:
        props = feat.get("properties") or {}
        pid = normalize_digits_only(props.get("id"))
        if pid != pid_digits:
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            raise ValueError(f"Matched feature id={pid_digits} but geometry is not Point: {geom.get('type')}")
        coords = geom.get("coordinates")
        if not (isinstance(coords, (list, tuple)) and len(coords) >= 2):
            raise ValueError(f"Invalid Point coordinates for id={pid_digits}: {coords}")
        lon, lat = float(coords[0]), float(coords[1])
        return lon, lat
    raise KeyError(f"Feature not found by properties.id digits-only = {pid_digits}")


# -----------------------------
# Main per-sample computation
# -----------------------------
def compute_for_prompt_file(prompt_path: Path, model_id: str, use_baseline: bool) -> Dict[str, Any]:
    prompt_obj = load_json(prompt_path)
    prompt_text = prompt_obj.get("prompt", "")

    pid, magnitude_m, direction = parse_prompt_for_id_and_magnitude(prompt_text)

    # 文件名约定：
    #   prompt:  xxxxx_prompt_level_1.json
    #   label:   xxxxx_label_1.geojson        (同级目录)
    #   after:   <same dir>/<sanitize(model_id)>/xxxxx_after.geojson
    stem = prompt_path.name.replace("_prompt_level_1.json", "")
    bbox_dir = prompt_path.parent

    label_path = bbox_dir / f"{stem}_label_1.geojson"
    if use_baseline:
        after_dir = bbox_dir / "baseline" / sanitize_model_id(model_id)
    else:
        after_dir = bbox_dir / sanitize_model_id(model_id)
    after_path = after_dir / f"{stem}_after.geojson"

    rec: Dict[str, Any] = {
        "prompt_path": str(prompt_path),
        "stem": stem,
        "model_id": model_id,
        "model_dir": str(after_dir),
        "target_id": pid,
        "magnitude_m": magnitude_m,
        "direction": direction,
        "label_geojson": str(label_path),
        "after_geojson": str(after_path),
        "label_lon": None,
        "label_lat": None,
        "after_lon": None,
        "after_lat": None,
        "d_edit_label_m": None,
        "REE": None,
        "ok": False,
        "error": None,
    }

    if not pid:
        rec["error"] = "Failed to parse target_id from prompt"
        return rec
    if magnitude_m is None or magnitude_m <= 0:
        rec["error"] = f"Failed to parse valid magnitude m (meters) from prompt; got {magnitude_m}"
        return rec
    if not label_path.exists():
        rec["error"] = f"Missing label geojson: {label_path}"
        return rec
    if not after_path.exists():
        rec["error"] = f"Missing after geojson: {after_path}"
        return rec

    try:
        label_fc = load_json(label_path)
        after_fc = load_json(after_path)

        p_label = get_point_lonlat_by_pid(label_fc, pid)
        p_after = get_point_lonlat_by_pid(after_fc, pid)

        rec["label_lon"], rec["label_lat"] = p_label
        rec["after_lon"], rec["after_lat"] = p_after

        d_m = planar_distance_m(p_after, p_label)  # d(g_edit, g_label)
        rec["d_edit_label_m"] = d_m

        rec["REE"] = float(d_m) / float(magnitude_m)
        rec["ok"] = True
        return rec
    except Exception as e:
        rec["error"] = repr(e)
        return rec


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../../data_1_2")
    ap.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument(
        "--use_baseline",
        action="store_true",
        help="If set, read edited results from baseline/<model_id>/ instead of <model_id>/",
    )
    ap.add_argument("--out_csv", type=str, default="ree_level1_edit_vs_label.csv")
    args = ap.parse_args()

    root = Path(args.data_root)
    prompt_files = list(root.rglob("*_prompt_level_1.json"))
    print(f"[Info] Found {len(prompt_files)} prompt files under: {root}")

    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(prompt_files, 1):
        rec = compute_for_prompt_file(p, args.model_id, args.use_baseline)
        rows.append(rec)
        # ===== per-sample print =====
        # print(f"[SAMPLE] {rec['prompt_path']}")
        # if rec.get("ok"):
        #     print(f"  REE = {rec['REE']:.6f}")
        #     print(f"  target_m = {rec['magnitude_m']} (meters)")
        # else:
        #     print(f"  ERROR: {rec.get('error')}")
        # print("-" * 60)
        #
        # if i % 50 == 0:
        #     print(f"[Progress] {i}/{len(prompt_files)}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt_path",
        "stem",
        "model_id",
        "model_dir",
        "target_id",
        "magnitude_m",
        "direction",
        "label_geojson",
        "after_geojson",
        "label_lon",
        "label_lat",
        "after_lon",
        "after_lat",
        "d_edit_label_m",
        "REE",
        "ok",
        "error",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ok_rees = [r["REE"] for r in rows if r.get("ok") and r.get("REE") is not None]

    ok_n = len(ok_rees)
    total_n = len(rows)

    mean_ree = None
    std_ree = None

    if ok_n > 0:
        mean_ree = sum(ok_rees) / ok_n
        var_ree = sum((x - mean_ree) ** 2 for x in ok_rees) / ok_n
        std_ree = math.sqrt(var_ree)

        # 如果你想用 sample standard deviation（除以 N-1），用下面三行替换上面三行：
        # mean_ree = sum(ok_rees) / ok_n
        # var_ree = sum((x - mean_ree) ** 2 for x in ok_rees) / (ok_n - 1)
        # std_ree = math.sqrt(var_ree)

    print(f"{args.model_id} [Done] Saved CSV: {out_path}")
    print(f"[Summary] ok={ok_n} / total={total_n}")

    if mean_ree is not None:
        print(f"[Summary] mean_REE (ok only) = {mean_ree:.6f}")
        print(f"[Summary] std_REE  (ok only) = {std_ree:.6f}")
    else:
        print("[Summary] mean_REE (ok only) = N/A")
        print("[Summary] std_REE  (ok only) = N/A")


if __name__ == "__main__":

    model_ids = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ]
    FORCE_USE_BASELINE = True      # True / False / None
    # FORCE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # 或 None
    for model_id in model_ids:
        FORCE_MODEL_ID = model_id

        try:
            import sys

            if FORCE_USE_BASELINE is True and "--use_baseline" not in sys.argv:
                sys.argv.append("--use_baseline")

            if FORCE_MODEL_ID is not None:
                for i, v in enumerate(sys.argv):
                    if v == "--model_id":
                        del sys.argv[i:i+2]
                        break
                sys.argv.extend(["--model_id", FORCE_MODEL_ID])

            main()
        except Exception as e:
            print(e)









