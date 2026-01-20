from __future__ import annotations
import json
import os
import random
import re
from typing import Dict, Any, Iterator, Tuple, Optional, List, Literal, Iterable

import math
from shapely.ops import unary_union as shp_unary_union

from pathlib import Path
import json, math, gzip

_GEO_TYPES_FACE = {"Polygon", "MultiPolygon"}
_GEO_TYPES_ALL = _GEO_TYPES_FACE | {"Point", "MultiPoint", "LineString", "MultiLineString", "GeometryCollection"}

def _open_auto(path: Path):
    """支持 .gz 与普通文件；自动处理 utf-8 与 BOM。"""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")  # gzip 文本模式
    # 用 utf-8-sig 以吞掉 BOM
    return open(path, "r", encoding="utf-8-sig")

def _has_nan_inf(coords: Iterable) -> bool:
    """递归检测坐标里是否出现 NaN/Inf。"""
    for v in coords:
        if isinstance(v, (list, tuple)):
            if _has_nan_inf(v):
                return True
        else:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return True
    return False

def pair_paths_from_before(before_path: str | Path) -> Dict[str, Path]:
    """
    给一个 BEFORE 路径，返回:
    {
      "before": <.../location.geojson>,
      "label":  <.../location_label.geojson>  # 若不存在则不返回该键
    }
    """
    p = Path(before_path).resolve()
    parent = p.parent
    stem = p.stem  # e.g. "location"
    label = parent / f"{stem}_label.geojson"
    out: Dict[str, Path] = {"before": p}
    if label.exists():
        out["label"] = label
    return out

def find_any_pair(base_dir: str | Path) -> Optional[Dict[str, Path]]:
    """
    在数据集中挑一个有 label 的 before/label 对（用于 smoke test）
    """
    for _city, _rel, p in iter_geojson_paths(base_dir):
        pair = pair_paths_from_before(p)
        if "label" in pair:
            return pair
    return None

BBOX_DIR_RE = re.compile(r"^lat_[\-\d_]+_lon_[\-\d_]+_\d+$")


def strip_trailing_number(folder: str) -> str:
    """
    去掉末尾 '_数字'（例如 'lat_40_422634_lon_-80_019506_342' → 'lat_40_422634_lon_-80_019506'）
    """
    return re.sub(r'_\d+$', '', folder)

def iter_geojson_paths(root: str) -> Iterator[Tuple[str, str, str]]:
    for city in sorted(os.listdir(root)):
        city_path = os.path.join(root, city)
        if not os.path.isdir(city_path):
            continue
        for bbox_dir in sorted(os.listdir(city_path)):
            if not BBOX_DIR_RE.match(bbox_dir):
                continue
            bbox_path = os.path.join(city_path, bbox_dir)
            if not os.path.isdir(bbox_path):
                continue

            clean_base = strip_trailing_number(bbox_dir)
            candidate = os.path.join(bbox_path, f"{clean_base}.geojson")
            if os.path.exists(candidate) and not candidate.endswith("_label.geojson"):
                yield city, bbox_dir, candidate

def load_fc(
    path: str | Path,
    *,
    strict: bool = False,
    only_polys: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    读取 GeoJSON FeatureCollection，带更稳健的校验与兼容：
    - 支持 PathLike / .gz / BOM
    - 严格模式校验每个 feature/geometry 的结构与坐标异常
    - 可选只保留 Polygon/MultiPolygon（仅影响 features，下游可自己决定）
    返回 None 表示读取或校验失败（同时打印可读告警）。
    """
    p = Path(path)
    try:
        with _open_auto(p) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] read fail {p}: {e}")
        return None

    if not isinstance(data, dict):
        print(f"[WARN] not a JSON object: {p}")
        return None
    if data.get("type") != "FeatureCollection":
        print(f"[WARN] not FeatureCollection: {p}")
        return None

    feats = data.get("features")
    if not isinstance(feats, list):
        print(f"[WARN] 'features' is not a list: {p}")
        return None

    # 轻量规范化 & 基础健壮性
    normalized_feats = []
    for idx, ft in enumerate(feats):
        if not isinstance(ft, dict):
            if strict:
                print(f"[WARN] feature[{idx}] not an object, drop")
                continue
            else:
                print(f"[INFO] feature[{idx}] not an object, drop"); continue

        geom = ft.get("geometry")
        props = ft.get("properties")
        if props is None or not isinstance(props, dict):
            props = {}
        ft["properties"] = props  # 规范化

        if geom is None:
            if strict:
                print(f"[WARN] feature[{idx}] missing geometry, drop")
                continue
            else:
                # 某些管线会允许空几何，直接跳过
                continue

        if not isinstance(geom, dict):
            print(f"[WARN] feature[{idx}] geometry not an object, drop"); continue

        gtype = geom.get("type")
        if gtype not in _GEO_TYPES_ALL:
            print(f"[WARN] feature[{idx}] unsupported geometry type={gtype}, drop"); continue

        # 坐标基本结构与 NaN/Inf 检查（strict）
        if strict:
            coords = geom.get("coordinates")
            if gtype != "GeometryCollection":
                if coords is None:
                    print(f"[WARN] feature[{idx}] missing coordinates, drop"); continue
                if _has_nan_inf(coords):
                    print(f"[WARN] feature[{idx}] coordinates contain NaN/Inf, drop"); continue

        # 仅保留面要素（可选）
        if only_polys and gtype not in _GEO_TYPES_FACE:
            continue

        normalized_feats.append(ft)

    # 如果 only_polys 导致全被过滤，允许返回空 FC（与原逻辑一致）
    data["features"] = normalized_feats

    # 可选：严格模式下给出数量告警
    if strict and len(normalized_feats) == 0:
        print(f"[WARN] no valid features remain after validation: {p}")

    return data

def write_fc(out_path: str, fc: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

GREEN_TAGS = {
    "landuse": {
        "grass", "flowerbed", "forest", "meadow", "recreation_ground",
        "greenery", "village_green", "allotments", "orchard", "vineyard",
        "plant_nursery", "greenhouse_horticulture", "tree_pit", 'green'
    },
    "leisure": {
        "garden", "park", "nature_reserve", "common", "commons",
        "parklet", "recreation_ground", "disc_golf_course"
    },
    "natural": {
        "wood", "scrub", "grassland", "shrubbery",
        "heath", "grass", "tree_group", "wetland"
    },
    "plant": {"tree"}
}

def is_green_feature(feature):
    props = feature.get("properties") or {}
    tags = props.get("tags") or {}
    if not isinstance(tags, dict) or not tags:
        return False
    for k, v in tags.items():
        if k in GREEN_TAGS:
            vv = str(v).strip().lower()
            if vv in GREEN_TAGS[k]:
                return True
    return False



def to_shapely(geom):
    from shapely.geometry import shape
    return shape(geom) if geom else None

def from_shapely(geom):
    from shapely.geometry import mapping
    return mapping(geom)

def pick_utm_epsg(lon: float, lat: float) -> int:
    # WGS84 UTM zone
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    north = lat >= 0
    return (32600 if north else 32700) + zone


def ensure_valid(sgeom):
    try:
        if not sgeom.is_valid:
            sgeom = sgeom.buffer(0)
    except Exception:
        pass
    return sgeom

def geom_area_m2(geom: Dict[str, Any]) -> float:
    gtype = (geom or {}).get("type")
    if gtype not in ("Polygon", "MultiPolygon"):
        return 0.0

    def _collect(g):
        if g["type"] == "Polygon":
            coords = []
            for ring in g.get("coordinates", []):
                coords.extend(ring)
            return coords
        coords = []
        for poly in g.get("coordinates", []):
            for ring in poly:
                coords.extend(ring)
        return coords

    coords = _collect(geom)
    if not coords:
        return 0.0

    mean_lat = sum(y for _, y in coords) / len(coords)
    cos_lat = math.cos(math.radians(mean_lat))
    kx, ky = 111320.0 * cos_lat, 110540.0

    def _ring_area(ring):
        n = len(ring)
        if n < 3:
            return 0.0
        s = 0.0
        for i in range(n):
            j = (i + 1) % n
            x1, y1 = ring[i][0] * kx, ring[i][1] * ky
            x2, y2 = ring[j][0] * kx, ring[j][1] * ky
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    def _poly_area(coords_poly):
        if not coords_poly:
            return 0.0
        area = _ring_area(coords_poly[0])
        for hole in coords_poly[1:]:
            area -= _ring_area(hole)
        return max(area, 0.0)

    if gtype == "Polygon":
        return _poly_area(geom.get("coordinates", []))
    return sum(_poly_area(p) for p in geom.get("coordinates", []))


def calculate_total_area(feats: List[Dict]) -> float:
    return sum(geom_area_m2(ft.get("geometry", {})) for ft in feats)

def geom_centroid_lonlat(geom: Dict[str, Any]) -> Tuple[float, float]:
    gtype = (geom or {}).get("type")
    coords = []
    if gtype == "Polygon":
        for ring in geom.get("coordinates", [])[:1]:
            coords.extend(ring[:10])
    elif gtype == "MultiPolygon":
        for poly in geom.get("coordinates", [])[:1]:
            for ring in poly[:1]:
                coords.extend(ring[:10])
    if not coords:
        return (0.0, 0.0)
    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def calculate_centroid(feats: List[Dict]) -> Tuple[float, float]:
    accx = accy = n = 0
    for ft in feats:
        g = ft.get("geometry", {})
        if (g or {}).get("type") in ("Polygon", "MultiPolygon"):
            x, y = geom_centroid_lonlat(g)
            accx += x; accy += y; n += 1
    return (accx / n, accy / n) if n else (0.0, 0.0)

def patch_feature_directive(ft: Dict, ratio: float):
    """为 feature 添加 Level-2 的 properties directive"""
    props = dict(ft.get("properties") or {})
    props["_directive"] = f"reshape-geom|greenify|no-overlap|target+={int(round(ratio * 100))}%"
    ft["properties"] = props


def get_project_funcs_internal(lon0: float, lat0: float) -> Tuple[Optional[Any], Optional[Any]]:
    try:
        import pyproj
    except Exception:
        return None, None
    zone = math.floor((lon0 + 180) / 6) + 1
    utm_proj = f'+proj=utm +zone={zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    try:
        wgs84 = pyproj.CRS("EPSG:4326")
        utm = pyproj.CRS.from_proj4(utm_proj)
        fwd = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
        inv = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform
    except Exception:
        return None, None
    return fwd, inv


def create_transformer_funcs(lon0: float, lat0: float):
    try:
        from shapely.ops import transform as shp_transform
    except Exception:
        return None, None
    fwd, inv = get_project_funcs_internal(lon0, lat0)
    if fwd is None or inv is None:
        return None, None
    def to_local(sgeom):
        return shp_transform(lambda x, y, z=None: fwd(x, y), sgeom)
    def to_wgs84(sgeom):
        return shp_transform(lambda x, y, z=None: inv(x, y), sgeom)
    return to_local, to_wgs84

def process_single_expansion(
    ft: dict, to_local, to_wgs84, forbidden_m,
    remaining_need: float, tol: float,
    r_min: float, r_max: float, pick_radius, gap_clean_m: float = 1.0
):
    s0 = to_shapely(ft.get("geometry", {}))
    if s0 is None or s0.is_empty:
        return ft.get("geometry"), 0.0
    s0 = ensure_valid(s0)
    s0_m = to_local(s0)
    if s0_m.is_empty:
        return ft.get("geometry"), 0.0

    r = pick_radius()
    buf_m = ensure_valid(s0_m.buffer(r))
    if forbidden_m is not None and (not forbidden_m.is_empty):
        buf_m = ensure_valid(buf_m.difference(forbidden_m))
        if buf_m.is_empty:
            return ft.get("geometry"), 0.0

    s_new_m = ensure_valid(s0_m.union(buf_m))
    s_new_m = ensure_valid(s_new_m.buffer(-gap_clean_m)).buffer(+gap_clean_m)
    s_new = ensure_valid(to_wgs84(s_new_m))

    added = max(0.0, geom_area_m2(from_shapely(s_new)) - geom_area_m2(from_shapely(s0)))
    if added > remaining_need * (1.0 + tol):
        r2 = max(r_min, r * 0.5)
        buf_m2 = ensure_valid(s0_m.buffer(r2))
        if forbidden_m is not None and (not forbidden_m.is_empty):
            buf_m2 = ensure_valid(buf_m2.difference(forbidden_m))
        if not buf_m2.is_empty:
            s_new_m2 = ensure_valid(s0_m.union(buf_m2))
            s_new_m2 = ensure_valid(s_new_m2.buffer(-gap_clean_m)).buffer(+gap_clean_m)
            s_new2 = ensure_valid(to_wgs84(s_new_m2))
            added2 = max(0.0, geom_area_m2(from_shapely(s_new2)) - geom_area_m2(from_shapely(s0)))
            if 0 < added2 <= added:
                return from_shapely(s_new2), added2

    return from_shapely(s_new), added


def cluster_and_union_green_features(
    feats: List[Dict[str, Any]],
    centroid: Tuple[float, float],
    dist_m: float
) -> List[Dict[str, Any]]:
    try:
        from shapely.ops import unary_union
    except Exception:
        return feats
    to_local, to_wgs84 = create_transformer_funcs(centroid[0], centroid[1])
    if to_local is None or to_wgs84 is None:
        return feats

    greens, others = [], []
    for ft in feats:
        if is_green_feature(ft):
            g = ft.get("geometry", {})
            if (g or {}).get("type") not in ("Polygon", "MultiPolygon"):
                continue
            s = ensure_valid(to_shapely(g))
            if s and (not s.is_empty):
                greens.append({"ft": ft, "geom_m": to_local(s)})
        else:
            others.append(ft)

    if len(greens) <= 1:
        return feats

    n = len(greens)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    buffers = [ensure_valid(item["geom_m"].buffer(dist_m / 2.0)) for item in greens]
    for i in range(n):
        bi = buffers[i]
        for j in range(i + 1, n):
            if bi.intersects(buffers[j]):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    new_green_features = []
    for root, idxs in clusters.items():
        if len(idxs) == 1:
            new_green_features.append(greens[idxs[0]]["ft"])
        else:
            merged_m = ensure_valid(unary_union([greens[i]["geom_m"] for i in idxs]))
            merged = to_wgs84(merged_m)
            orig = greens[idxs[0]]["ft"]
            new_green_features.append({
                "type": "Feature",
                "geometry": from_shapely(merged),
                "properties": {
                    **(orig.get("properties") or {}),
                    "_merged_count": len(idxs),
                }
            })
    return others + new_green_features

def get_sorted_green_features_from_list(feats: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    items = []
    for i, ft in enumerate(feats):
        if is_green_feature(ft):
            a = geom_area_m2(ft.get("geometry", {}))
            if a > 0:
                items.append((i, a))
    return sorted(items, key=lambda x: x[1])  # 升序：小块优先长


def build_dynamic_forbidden_m(
    feats: list, idx_self: int, to_local, static_forbidden_m, gap_between_features_m: float,
):
    try:
        from shapely.ops import unary_union as shp_unary_union
    except Exception:
        return static_forbidden_m
    bufs = []
    for j, ftj in enumerate(feats):
        if j == idx_self:
            continue
        s = to_shapely(ftj.get("geometry") or {})
        if s is None or s.is_empty:
            continue
        sm = to_local(ensure_valid(s))
        if sm.is_empty:
            continue
        bufs.append(ensure_valid(sm.buffer(gap_between_features_m)))
    dyn = shp_unary_union(bufs) if bufs else None
    if static_forbidden_m is None:
        return ensure_valid(dyn) if dyn else None
    if dyn is None or dyn.is_empty:
        return static_forbidden_m
    return ensure_valid(shp_unary_union([static_forbidden_m, dyn]))

def build_forbidden_union_m(
    feats, to_local,
    include_buildings: bool = False,
    include_non_green_polygons: bool = False,
    way_setback_m: float = 2.0,
    b_setback_m: float = 4.0,
    small_gap_m: float = 1.5
):
    """
    构造“硬禁入”：
      - include_buildings=True  时，把建筑面 buffer(b_setback_m) 作为禁入
      - include_non_green_polygons=True 时，把其它非绿地面 buffer(small_gap_m) 作为禁入
      - 始终：把道路等线状 buffer(way_setback_m) 作为禁入（若存在）
    Level-3 扩张阶段推荐传 (False, False, way_setback_m) —— 允许覆盖面要素，只避线状
    """
    try:
        from shapely.ops import unary_union as shp_unary_union
    except Exception:
        return None

    buildings_m, others_m, ways_m = [], [], []

    for ft in feats:
        props = ft.get("properties") or {}
        tags = props.get("tags") or {}
        g = ft.get("geometry") or {}
        t = g.get("type")

        is_building = isinstance(tags, dict) and ("building" in tags and str(tags["building"]).lower() not in ("no", ""))

        if t in ("Polygon", "MultiPolygon"):
            s = to_shapely(g)
            if s is None or s.is_empty:
                continue
            sm = to_local(ensure_valid(s))
            if include_buildings and is_building:
                buildings_m.append(sm)
            elif include_non_green_polygons and (not is_green_feature(ft)):
                others_m.append(sm)

        elif t in ("LineString", "MultiLineString"):
            s = to_shapely(g)
            if s is None or s.is_empty:
                continue
            # 任何标注 highway 的线都视为受保护线状
            if isinstance(tags, dict) and str(tags.get("highway", "")).lower():
                ways_m.append(to_local(ensure_valid(s)))

    parts = []
    if buildings_m:
        parts.append(shp_unary_union([ensure_valid(s.buffer(b_setback_m)) for s in buildings_m]))
    if others_m:
        parts.append(shp_unary_union([ensure_valid(s.buffer(small_gap_m)) for s in others_m]))
    if ways_m and way_setback_m > 0:
        parts.append(shp_unary_union([ensure_valid(s.buffer(way_setback_m)) for s in ways_m]))

    if not parts:
        return None
    u = shp_unary_union([p for p in parts if p and not p.is_empty])
    return ensure_valid(u) if u else None


def enforce_min_gap_between_features(
    fc: dict, gap_m: float, to_local, to_wgs84, priority: str = "largest",
) -> dict:
    try:
        from shapely.ops import unary_union as shp_unary_union
    except Exception:
        return fc
    feats = list(fc.get("features", []))
    if not feats:
        return fc

    s_list = []
    for ft in feats:
        s = to_shapely(ft.get("geometry", {}))
        s_list.append(None if (s is None or s.is_empty) else to_local(ensure_valid(s)))

    order = list(range(len(feats)))
    if priority == "largest":
        areas = [(i, (s_list[i].area if s_list[i] is not None else 0.0)) for i in order]
        order = [i for i, _ in sorted(areas, key=lambda t: -t[1])]

    for i in order:
        si = s_list[i]
        if si is None or si.is_empty:
            continue
        others_buf = []
        for j, sj in enumerate(s_list):
            if j == i or sj is None or sj.is_empty:
                continue
            others_buf.append(ensure_valid(sj.buffer(gap_m * 0.5)))
        if not others_buf:
            continue
        forbidden = ensure_valid(shp_unary_union(others_buf))
        si_new = ensure_valid(si.difference(forbidden))
        s_list[i] = si_new if not si_new.is_empty else si

    for i, s_m in enumerate(s_list):
        if s_m is None:
            continue
        feats[i]["geometry"] = from_shapely(ensure_valid(to_wgs84(s_m)))
    fc["features"] = feats
    return fc


def remove_fully_covered_features(
    feats: List[dict],
    covering_geom_geojson: dict,
    skip_index: Optional[int] = None,
    cover_types: Tuple[str, ...] = ("Polygon", "MultiPolygon"),
    keep_green: bool = False,
) -> Tuple[List[dict], int]:
    """
    删除“几何被 covering_geom 完全覆盖”的要素：
      - skip_index：当前扩张的那块，绝不删除
      - cover_types：仅处理这些类型（默认面要素）
      - keep_green=False：被覆盖就删（包括小绿地）；True：只删非绿地
    """
    cover_s = to_shapely(covering_geom_geojson)
    cover_s = ensure_valid(cover_s)
    if cover_s is None or cover_s.is_empty:
        return feats, 0

    kept, removed = [], 0
    for i, ft in enumerate(feats):
        if i == skip_index:  # 保留当前扩张对象
            kept.append(ft)
            continue

        g = ft.get("geometry") or {}
        t = g.get("type")
        if not t or t not in cover_types:
            kept.append(ft); continue

        if keep_green and is_green_feature(ft):
            kept.append(ft); continue

        s = to_shapely(g)
        if s is None or s.is_empty:
            kept.append(ft); continue

        s = ensure_valid(s)
        # covers：边界贴合也视为覆盖
        if cover_s.covers(s):
            removed += 1
            # 丢弃
        else:
            kept.append(ft)

    return kept, removed

def random_green_kv(rnd: random.Random) -> Tuple[str, str]:
    k = rnd.choice(list(GREEN_TAGS.keys()))
    v = rnd.choice(sorted(list(GREEN_TAGS[k])))
    return k, v




def _union_area_m2_of_subset(feats, pred):
    """
    统一在局部米制投影下计算某子集的 union 面积（m²）。
    pred(ft) 用于筛选子集；仅对 Polygon/MultiPolygon 生效。
    """
    try:
        from shapely.ops import unary_union
    except Exception:
        # 退化兜底：回到逐面近似面积求和（比 union 稍偏大，但避免 0）
        return sum(
            geom_area_m2((ft.get("geometry") or {}))
            for ft in feats
            if pred(ft) and ((ft.get("geometry") or {}).get("type") in ("Polygon","MultiPolygon"))
        )

    # 选子集几何
    polys = []
    for ft in feats:
        if not pred(ft):
            continue
        gj = (ft.get("geometry") or {})
        if gj.get("type") not in ("Polygon","MultiPolygon"):
            continue
        s = ensure_valid(to_shapely(gj))
        if s and (not s.is_empty):
            polys.append(s)
    if not polys:
        return 0.0

    # 构造局部投影
    lon0, lat0 = calculate_centroid(feats)
    to_local, _ = create_transformer_funcs(lon0, lat0)
    if to_local is None:
        # 退化兜底：逐面近似面积求和（同上）
        return sum(geom_area_m2(from_shapely(p)) for p in polys)

    # 投影→union→面积（米²）
    polys_m = [to_local(p) for p in polys]
    u_m = ensure_valid(unary_union(polys_m))
    if u_m.is_empty:
        return 0.0
    return float(u_m.area)



def patch_feature_to_green_level1(feature: dict, ratio: float, seed: int) -> dict:
    """
    Level-1 修改器：
    - 只改 properties.tags，不动几何/坐标/feature 数量和顺序
    """
    rnd = random.Random(seed)
    props = dict(feature.get("properties") or {})

    tags = {}

    k, v = random_green_kv(rnd)
    tags[k] = v

    props["_directive"] = f"patch-tags|greenify|geom-locked|ratio={int(round(ratio * 100))}%"

    props["tags"] = tags
    feature["properties"] = props
    return feature


def _to_local_polys(feats: List[Dict[str, Any]], to_local):
    """将 Polygon/MultiPolygon 转为本地米制 shapely 多边形列表（已 ensure_valid）。"""
    import shapely
    local_polys = []
    for ft in feats:
        g = ft.get("geometry", {}) or {}
        t = g.get("type")
        if t not in ("Polygon", "MultiPolygon"):
            continue
        s = to_shapely(g)
        if s is None:
            continue
        s = ensure_valid(s)
        if s.is_empty:
            continue
        try:
            s_local = to_local(s)
        except Exception:
            # 投影失败就跳过这个要素
            continue
        if s_local is None or s_local.is_empty:
            continue
        # 统一转为 MultiPolygon 迭代
        if s_local.geom_type == "Polygon":
            local_polys.append(s_local)
        elif s_local.geom_type == "MultiPolygon":
            local_polys.extend([p for p in s_local.geoms if not p.is_empty])
        else:
            # 非面不要
            continue
    return local_polys

def _count_components(polys) -> int:
    """按拓扑连通（接触/相交都算连通）统计连通分量数。"""
    from shapely.ops import unary_union
    if not polys:
        return 0
    try:
        merged = ensure_valid(unary_union(polys))
    except Exception:
        # 退化处理：无 union 就把每个多边形当一个分量
        return len(polys)
    if merged.is_empty:
        return 0
    if merged.geom_type == "Polygon":
        return 1
    if merged.geom_type == "MultiPolygon":
        return len(merged.geoms)
    # 少见情况（比如出现线/点残留），保守处理
    return 1

def _bin_density(value: float, thresholds=(0.001, 0.01)) -> str:
    """简单密度分箱占位（你后面可以换成真实道路/建筑长度或覆盖率计算）。"""
    # 目前用总面积比例当占位，等你有 road/building 面积/长度就替换 value 入参
    if value <= thresholds[0]:
        return "low"
    if value <= thresholds[1]:
        return "mid"
    return "high"

def summarize_fc(fc: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成一个“短摘要”，仅含米制统计与计数，不包含坐标。
    面积统计采用几何 union 口径，避免叠层/重复导致的双算。
    """
    assert isinstance(fc, dict), "summarize_fc: fc must be a dict"
    feats: List[Dict[str, Any]] = list(fc.get("features", []))

    # 1) 面积统计（union 口径）
    total_editable_m2 = _union_area_m2_of_subset(
        feats,
        pred=lambda ft: True  # 内部仅会对 Polygon/MultiPolygon 计入
    )

    green_feats = [ft for ft in feats if is_green_feature(ft)]
    green_m2 = _union_area_m2_of_subset(
        green_feats,
        pred=lambda ft: True
    )

    # 估算 bbox 尺度（米）+ 计算连通分量
    lon0, lat0 = calculate_centroid(feats)
    to_local, _to_wgs84 = create_transformer_funcs(lon0, lat0)

    if to_local is None:
        # 投影缺失兜底
        bbox_m = [0.0, 0.0]
        n_green_components = 0
    else:
        # 2.1 全部面要素 -> local polys
        local_polys_all = _to_local_polys(feats, to_local)
        if local_polys_all:
            minx = min(p.bounds[0] for p in local_polys_all)
            miny = min(p.bounds[1] for p in local_polys_all)
            maxx = max(p.bounds[2] for p in local_polys_all)
            maxy = max(p.bounds[3] for p in local_polys_all)
            bbox_m = [float(maxx - minx), float(maxy - miny)]
        else:
            bbox_m = [0.0, 0.0]

        # 2.2 仅绿地 -> 计算连通分量
        local_polys_green = _to_local_polys(green_feats, to_local)
        if local_polys_green:
            n_green_components = _count_components(local_polys_green)
        else:
            n_green_components = 0

    # 3) 基础计数
    n_faces_total = sum(
        1 for ft in feats
        if (ft.get("geometry", {}) or {}).get("type") in ("Polygon", "MultiPolygon")
    )
    n_green = sum(
        1 for ft in green_feats
        if (ft.get("geometry", {}) or {}).get("type") in ("Polygon", "MultiPolygon")
    )

    # 4) 占位密度分箱（如后续有道路/建筑的实际密度，可替换 _bin_density 的入参）
    road_density_bin = _bin_density(0.0)
    building_density_bin = _bin_density(0.0)

    # 5) 返回摘要
    return {
        "total_editable_m2": float(total_editable_m2),
        "green_m2": float(green_m2),
        "n_faces_total": int(n_faces_total),
        "n_green": int(n_green),
        "n_green_components": int(n_green_components),
        "bbox_m": [float(bbox_m[0]), float(bbox_m[1])],
        "road_density_bin": road_density_bin,
        "building_density_bin": building_density_bin,
    }

def summarize_pair(before_fc: Dict[str, Any], after_fc: Dict[str, Any]) -> Dict[str, Any]:
    """
    汇总 BEFORE / AFTER 的摘要、差分，以及 AFTER 的 metadata（如存在）。
    """
    s_before = summarize_fc(before_fc)
    s_after  = summarize_fc(after_fc)

    delta_green_m2 = float(s_after["green_m2"] - s_before["green_m2"])
    connectivity_delta = int(s_after["n_green_components"] - s_before["n_green_components"])

    # 从 AFTER 元数据读取意图与统计（如果有）
    meta = (after_fc.get("metadata") or {})
    intent = meta.get("_edit_intent")
    stats  = meta.get("_edit_stats") or {}

    # purge 判断优先读 intent / stats，否则用启发式（连通度显著下降 + 要素减少时猜测发生了吞并）
    purge_happened = False
    if intent and isinstance(intent, str) and "expand-then-purge" in intent:
        purge_happened = True
    elif stats.get("purge", None) is True:
        purge_happened = True
    else:
        # 启发式：AFTER 面要素总数 < BEFORE，且绿地面积显著上升
        before_faces = s_before["n_faces_total"]
        after_faces  = s_after["n_faces_total"]
        if after_faces < before_faces and delta_green_m2 > 0:
            purge_happened = True

    # 如果 metadata 没写 ratio/tol 等，后面 infer 时会再估
    after_meta = None
    if intent or stats:
        after_meta = {"_edit_intent": intent, "_edit_stats": stats}

    # purged_count/area 这两个只有有显式 trace 时才能准算；这里先给 0，供上游显示
    diff = {
        "delta_green_m2": delta_green_m2,
        "connectivity_delta": connectivity_delta,
        "purge_happened": purge_happened,
        "purged_count": int(stats.get("purged_count", 0) or 0),
        "purged_area_m2": float(stats.get("purged_area_m2", 0.0) or 0.0),
    }

    return {
        "before": s_before,
        "after":  s_after,
        "diff":   diff,
        "after_meta": after_meta,
    }
