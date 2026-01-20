# utils/level3_tools.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
import math
import copy

# ----------------------------
# 1) Green tag classifier
# ----------------------------
GREEN_TAGS = {
    "landuse": {
        "grass", "flowerbed", "forest", "meadow", "recreation_ground",
        "greenery", "village_green", "allotments", "orchard", "vineyard",
        "plant_nursery", "greenhouse_horticulture", "tree_pit", "green",
    },
    "leisure": {
        "garden", "park", "nature_reserve", "common", "commons",
        "parklet", "recreation_ground", "disc_golf_course",
    },
    "natural": {
        "wood", "scrub", "grassland", "shrubbery",
        "heath", "grass", "tree_group", "wetland",
    },
    "plant": {"tree"},
}

def is_green_feature(feature: Dict[str, Any]) -> bool:
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


# ----------------------------
# 2) GeoJSON <-> Shapely
# ----------------------------
def to_shapely(geojson_geom: Dict[str, Any]):
    """
    GeoJSON geometry dict -> shapely geometry
    """
    from shapely.geometry import shape
    if not geojson_geom or not isinstance(geojson_geom, dict):
        return shape({"type": "GeometryCollection", "geometries": []})
    return shape(geojson_geom)

def from_shapely(shp) -> Dict[str, Any]:
    """
    shapely geometry -> GeoJSON geometry dict
    """
    from shapely.geometry import mapping
    if shp is None:
        return {"type": "GeometryCollection", "geometries": []}
    return mapping(shp)


# ----------------------------
# 3) Validity fixing
# ----------------------------
def ensure_valid(shp):
    """
    尽最大努力把几何修到可用：
    - shapely.make_valid（若可用）
    - buffer(0) 兜底
    """
    if shp is None:
        return shp
    try:
        if getattr(shp, "is_empty", False):
            return shp
    except Exception:
        pass

    # shapely 2.x: make_valid
    try:
        from shapely.validation import make_valid
        fixed = make_valid(shp)
        # make_valid 可能返回 GeometryCollection；buffer(0) 再归一
        try:
            fixed2 = fixed.buffer(0)
            return fixed2
        except Exception:
            return fixed
    except Exception:
        pass

    # shapely 1.8/2.x 通用兜底
    try:
        return shp.buffer(0)
    except Exception:
        return shp


# ----------------------------
# 4) Robust centroid (lon, lat)
# ----------------------------
def calculate_centroid(features: Iterable[Dict[str, Any]]) -> Tuple[float, float]:
    """
    输入：Feature 列表（或 fc["features"]）
    输出：一个合理的 (lon, lat) 作为投影中心点

    策略（稳健）：
    - 先收集所有坐标点（对 Point/Line/Polygon 取代表点）
    - 用 bbox 中心作为 fallback
    """
    # shapely 代表点更稳定
    xs: List[float] = []
    ys: List[float] = []

    for ft in features or []:
        if not isinstance(ft, dict):
            continue
        geom = ft.get("geometry") or {}
        if not isinstance(geom, dict) or "type" not in geom:
            continue
        try:
            shp = ensure_valid(to_shapely(geom))
        except Exception:
            continue
        if shp is None or getattr(shp, "is_empty", False):
            continue
        try:
            # representative_point 对面更稳（一定在内部）
            p = shp.representative_point()
            xs.append(float(p.x))
            ys.append(float(p.y))
        except Exception:
            try:
                c = shp.centroid
                xs.append(float(c.x))
                ys.append(float(c.y))
            except Exception:
                pass

    if xs and ys:
        # 用均值足够；也可以改成中位数更抗离群
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    raise ValueError("no geometry found to compute centroid")


# ----------------------------
# 5) Local projection (UTM)
# ----------------------------
def _utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """
    根据 lon/lat 自动推 UTM EPSG：
    - 北半球：326xx
    - 南半球：327xx
    """
    # zone: 1..60
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone

def create_transformer_funcs(
    lon0: float,
    lat0: float,
) -> Tuple[Optional[Callable], Optional[Callable]]:
    """
    返回：
      to_local(shapely_geom)  : EPSG:4326 -> UTM（米）
      to_wgs84(shapely_geom) : UTM -> EPSG:4326
    """
    try:
        from shapely.ops import transform as shp_transform
        from pyproj import Transformer, CRS
    except Exception:
        return None, None

    # 输入 lon0/lat0 可能是 0,0（fallback），仍然能选个 UTM，但意义不大
    utm_epsg = _utm_epsg_from_lonlat(lon0, lat0)

    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(utm_epsg)

    fwd = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    inv = Transformer.from_crs(crs_utm, crs_wgs84, always_xy=True)

    def to_local(shp):
        return shp_transform(fwd.transform, shp)

    def to_wgs84(shp):
        return shp_transform(inv.transform, shp)

    return to_local, to_wgs84
# ----------------------------
# GeoState IO compatibility
# ----------------------------
def _get_fc(state) -> Dict[str, Any]:
    if not hasattr(state, "fc") or not isinstance(state.fc, dict):
        raise AttributeError("GeoState must have .fc (FeatureCollection dict)")
    if state.fc.get("type") != "FeatureCollection":
        raise ValueError("state.fc must be a GeoJSON FeatureCollection")
    return state.fc


def _set_fc(state, fc: Dict[str, Any]) -> None:
    state.fc = fc

def _cache(state) -> Dict[str, Any]:
    """
    统一把中间结果写到 state.cache；若没有则写 state.scratch；
    都没有就挂到 state.cache 上（动态属性）。
    """
    if hasattr(state, "cache") and isinstance(state.cache, dict):
        return state.cache
    if hasattr(state, "scratch") and isinstance(state.scratch, dict):
        return state.scratch
    # 动态创建
    state.cache = {}
    return state.cache

def _ensure_path(d: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
    cur = d
    for k in path:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur

def _green_cache(state) -> Dict[str, Any]:
    c = _cache(state)
    return _ensure_path(c, ["green"])

def _proj_cache(state) -> Dict[str, Any]:
    c = _cache(state)
    return _ensure_path(c, ["projection"])

def _forbidden_cache(state) -> Dict[str, Any]:
    c = _cache(state)
    return _ensure_path(c, ["forbidden"])

def _checks_cache(state) -> Dict[str, Any]:
    c = _cache(state)
    return _ensure_path(c, ["checks"])


# ----------------------------
# Helpers
# ----------------------------
def _iter_features(fc: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats = fc.get("features") or []
    return [f for f in feats if isinstance(f, dict)]

def _feature_id(ft: Dict[str, Any]) -> Optional[str]:
    fid = ft.get("id")
    if fid is None:
        fid = (ft.get("properties") or {}).get("id")
    if fid is None:
        return None
    return str(fid)

def _geom_type(ft: Dict[str, Any]) -> str:
    return (ft.get("geometry") or {}).get("type") or ""

def _is_polygonish(ft: Dict[str, Any]) -> bool:
    return _geom_type(ft) in ("Polygon", "MultiPolygon")

def _get_feature_by_id(fc: Dict[str, Any], fid: str) -> Optional[Tuple[int, Dict[str, Any]]]:
    feats = _iter_features(fc)
    for i, ft in enumerate(feats):
        if _feature_id(ft) == str(fid):
            return i, ft
    return None

def _area_m2_local(shp, to_local) -> float:
    try:
        g = to_local(ensure_valid(shp))
        return float(getattr(g, "area", 0.0))
    except Exception:
        return 0.0

def _safe_union(geoms: List[Any]):
    # shapely unary_union 更快，但这里避免强依赖写法差异
    if not geoms:
        return None
    out = geoms[0]
    for g in geoms[1:]:
        out = ensure_valid(out.union(g))
    return ensure_valid(out)

def _difference(a, b):
    if a is None:
        return None
    if b is None:
        return a
    if getattr(b, "is_empty", False):
        return a
    try:
        return ensure_valid(a.difference(b))
    except Exception:
        return a

def _intersects(a, b) -> bool:
    if a is None or b is None:
        return False
    if getattr(a, "is_empty", False) or getattr(b, "is_empty", False):
        return False
    try:
        return bool(a.intersects(b))
    except Exception:
        return False

def _intersection_area(a, b) -> float:
    try:
        inter = a.intersection(b)
        inter = ensure_valid(inter)
        return float(getattr(inter, "area", 0.0))
    except Exception:
        return 0.0


# ============================================================
# Level-3 Tools (Executor allowlist candidates)
# All functions: fn(state, **kwargs) but kwargs optional/ignored.
# ============================================================

def l3_collect_green_ids(state, **kwargs) -> Dict[str, Any]:
    """
    识别所有“绿地”面要素，写入 cache.green.green_ids
    默认只收 Polygon/MultiPolygon。
    """
    fc = _get_fc(state)
    feats = _iter_features(fc)

    green_ids: List[str] = []
    for ft in feats:
        if not _is_polygonish(ft):
            continue
        if is_green_feature(ft):
            fid = _feature_id(ft)
            if fid is not None:
                green_ids.append(fid)

    gc = _green_cache(state)
    gc["green_ids"] = green_ids
    if len(green_ids) == 0:
        raise RuntimeError(
            "l3_collect_green_ids: no green polygon ids found. "
            "This dataset is expected to contain at least one green Polygon/MultiPolygon."
        )

    return {"ok": True, "green_count": len(green_ids)}

def l3_prepare_local_projection(state, **kwargs) -> Dict[str, Any]:
    """
    计算中心点并构建投影函数（to_local/to_wgs84 handle），写入 cache.projection
    """
    fc = _get_fc(state)
    feats = _iter_features(fc)
    lon0, lat0 = calculate_centroid(feats)  # 你现有 utils 已实现
    to_local, to_wgs84 = create_transformer_funcs(lon0, lat0)

    pc = _proj_cache(state)
    pc["lon0"] = lon0
    pc["lat0"] = lat0
    pc["to_local"] = to_local
    pc["to_wgs84"] = to_wgs84
    pc["epsg"] = None  # 如果你内部能推 EPSG 可写这里
    return {"ok": bool(to_local and to_wgs84), "lon0": lon0, "lat0": lat0}

def l3_compute_green_area_stats(state, **kwargs) -> Dict[str, Any]:
    """
    计算每个绿地的面积（m²）+ 总绿地面积，写入 cache.green
    依赖 cache.green.green_ids 和 cache.projection.to_local
    """
    fc = _get_fc(state)
    gc = _green_cache(state)
    pc = _proj_cache(state)
    green_ids = list(gc.get("green_ids") or [])

    to_local = pc.get("to_local")
    if to_local is None:
        raise ValueError("to_local is required")

    area_by_id: Dict[str, float] = {}
    total = 0.0

    for gid in green_ids:
        got = _get_feature_by_id(fc, gid)
        if not got:
            area_by_id[gid] = 0.0
            continue
        _, ft = got
        shp = ensure_valid(to_shapely(ft.get("geometry") or {}))
        a = _area_m2_local(shp, to_local)
        area_by_id[gid] = a
        total += a

    gc["green_area_by_id_m2"] = area_by_id
    gc["green_area_total_m2"] = total
    return {"ok": True, "green_total_m2": total, "n_green": len(green_ids)}

def l3_set_green_area_target(state, **kwargs) -> Dict[str, Any]:
    ratio = kwargs.get("ratio", None)
    if ratio is None:
        raise ValueError("ratio is required for l3_set_green_area_target")
    # if ratio is None:
    #     ratio = 0.30
    # ratio = float(ratio)

    gc = _green_cache(state)
    A0 = float(gc.get("green_area_total_m2") or 0.0)
    target_total = A0 * (1.0 + ratio)
    need = max(0.0, target_total - A0)

    gc["target_ratio"] = ratio
    gc["target_total_m2"] = target_total
    gc["need_increase_m2"] = need
    gc["achieved_increase_m2"] = float(gc.get("achieved_increase_m2") or 0.0)
    return {"ok": True, "A0": A0, "target_total": target_total, "need_increase": need}

def l3_build_forbidden_union_m(state, **kwargs) -> Dict[str, Any]:
    """
    构建 forbidden_union（米制投影下），用于 hard constraint: 不与周围 overlap
    默认：所有“非绿地”Polygon/MultiPolygon 都是 forbidden（禁止绿地扩张覆盖）
    可选 kwargs:
      - include_non_green_polygons: bool (default True)
      - include_green_others: bool (default False)  # 若 True 则绿地之间也禁入（相当于不允许互相合并/接触）
      - polygon_setback_m: float (default 0.0)      # 给 forbidden 退让/膨胀
    结果写入 cache.forbidden.forbidden_union_m
    """
    include_non_green_polygons = bool(kwargs.get("include_non_green_polygons", True))
    include_green_others = bool(kwargs.get("include_green_others", False))
    polygon_setback_m = float(kwargs.get("polygon_setback_m", 0.0))

    fc = _get_fc(state)
    feats = _iter_features(fc)

    gc = _green_cache(state)
    green_ids = set(gc.get("green_ids") or [])

    pc = _proj_cache(state)
    to_local = pc.get("to_local")
    if to_local is None:
        raise ValueError("to_local is required")

    forbidden_parts = []
    for ft in feats:
        if not _is_polygonish(ft):
            continue
        fid = _feature_id(ft)
        shp = ensure_valid(to_shapely(ft.get("geometry") or {}))
        shp_m = ensure_valid(to_local(shp))

        is_green = (fid is not None and fid in green_ids and is_green_feature(ft))
        if is_green:
            if include_green_others:
                forbidden_parts.append(shp_m)
        else:
            if include_non_green_polygons:
                forbidden_parts.append(shp_m)

    forbidden = _safe_union(forbidden_parts) if forbidden_parts else None
    if forbidden is not None and polygon_setback_m != 0.0:
        try:
            forbidden = ensure_valid(forbidden.buffer(polygon_setback_m))
        except Exception:
            pass

    fb = _forbidden_cache(state)
    fb["forbidden_union_m"] = forbidden
    fb["polygon_setback_m"] = polygon_setback_m
    fb["include_non_green_polygons"] = include_non_green_polygons
    fb["include_green_others"] = include_green_others
    return {"ok": True, "has_forbidden": forbidden is not None}

def l3_plan_budget_allocation(state, **kwargs) -> Dict[str, Any]:
    """
    分配扩张预算：决定扩张哪些绿地、顺序与每步 step_budget_m2
    默认 policy: multi_greedy_largest_first

    写入 cache.green.allocation_plan = [{"id":..., "step_budget_m2":...}, ...]
    """
    import os
    import random
    import time

    policy = str(kwargs.get("policy", "multi_greedy_largest_first"))
    max_steps = int(kwargs.get("max_steps", 60))
    min_step_area_m2 = float(kwargs.get("min_step_area_m2", 5.0))

    # 随机幅度：默认 ±20%
    jitter_frac = float(kwargs.get("jitter_frac", 0.20))
    jitter_frac = max(0.0, min(jitter_frac, 0.90))

    # 是否按面积加权随机选 id（True 更合理）
    weighted = bool(kwargs.get("weighted", True))

    gc = _green_cache(state)
    if "green_ids" not in gc:
        raise RuntimeError(
            "l3_plan_budget_allocation: missing cache.green.green_ids. "
            "You must call l3_collect_green_ids() first."
        )
    if "green_area_by_id_m2" not in gc or "green_area_total_m2" not in gc:
        raise RuntimeError(
            "l3_plan_budget_allocation: missing green area stats in cache. "
            "You must call l3_compute_green_area_stats() before allocation."
        )
    if "need_increase_m2" not in gc:
        raise RuntimeError(
            "l3_plan_budget_allocation: missing cache.green.need_increase_m2. "
            "You must call l3_set_green_area_target(ratio=...) before allocation."
        )

    green_ids = list(gc.get("green_ids") or [])
    if not green_ids:
        raise RuntimeError("l3_plan_budget_allocation: green_ids is empty (strict failure).")

    need_raw = gc.get("need_increase_m2", None)
    if need_raw is None:
        raise RuntimeError("l3_plan_budget_allocation: need_increase_m2 is None (strict failure).")
    need = float(need_raw)
    if need <= 0.0:
        raise RuntimeError(
            f"l3_plan_budget_allocation: need_increase_m2 <= 0 (need={need}). "
            "Strict mode treats this as failure (check ratio/area stats/call order)."
        )

    area_by_id = dict(gc.get("green_area_by_id_m2") or {})
    if not area_by_id:
        raise RuntimeError(
            "l3_plan_budget_allocation: green_area_by_id_m2 empty (strict failure). "
            "Did you call l3_compute_green_area_stats()?"
        )

    # ====== 1) 本次调用的随机 seed（每次运行都不同）======
    # 若你未来想可复现：export L3_GLOBAL_SEED=123
    env_seed = os.environ.get("L3_GLOBAL_SEED", "").strip()
    if env_seed:
        try:
            seed = int(env_seed)
        except Exception:
            # 环境变量不是整数，就 hash 一下（仍可复现）
            seed = abs(hash(env_seed)) % (2 ** 31)
    else:
        # 每次调用不同：时间 + 随机源
        seed = int(time.time_ns() ^ random.getrandbits(64)) & 0xFFFFFFFF

    rng = random.Random(seed)

    # ====== 2) 决定每步预算的 base ======
    n_steps = max(1, min(max_steps, int(math.ceil(need / max(min_step_area_m2, 1e-6)))))
    base_budget = max(min_step_area_m2, need / n_steps)

    # ====== 3) 构造抽样池（可按 policy 先排序，但随机抽取仍会打散）======
    pairs = [(gid, float(area_by_id.get(gid, 0.0))) for gid in green_ids]

    # policy 只影响“候选顺序/权重构造”，不再决定固定轮询
    if policy == "multi_greedy_largest_first":
        pairs.sort(key=lambda x: x[1], reverse=True)
    elif policy == "multi_greedy_smallest_first":
        pairs.sort(key=lambda x: x[1])
    else:
        pairs.sort(key=lambda x: x[1], reverse=True)

    ids = [p[0] for p in pairs]
    if weighted:
        weights = [max(p[1], 1e-6) for p in pairs]  # 避免 0 权重
    else:
        weights = None

    # ====== 4) 生成 allocation_plan：随机选 id + 随机预算 ======
    plan: List[Dict[str, Any]] = []
    remaining = need

    for step_i in range(max_steps):
        if remaining <= 1e-6:
            break

        # 4.1 随机选一个目标绿地
        if weights is None:
            gid = rng.choice(ids)
        else:
            gid = rng.choices(ids, weights=weights, k=1)[0]

        # 4.2 随机预算：在 base_budget 上 jitter
        mult = 1.0 + rng.uniform(-jitter_frac, jitter_frac)
        b = base_budget * mult
        b = max(min_step_area_m2, b)

        # 4.3 最后收尾：别超过 remaining
        b = min(b, remaining)

        plan.append({"id": gid, "step_budget_m2": float(b)})
        remaining -= b

    if not plan:
        raise RuntimeError(
            "l3_plan_budget_allocation: allocation_plan ended up empty in strict mode (unexpected)."
        )

    # ====== 5) 写 cache（给 trace/debug 用）======
    gc["allocation_plan"] = plan
    gc["allocation_policy"] = policy
    gc["allocation_seed"] = seed
    gc["allocation_weighted"] = weighted
    gc["allocation_jitter_frac"] = jitter_frac
    gc["allocation_base_budget_m2"] = float(base_budget)

    return {"ok": True, "n_steps": len(plan), "policy": policy, "seed": seed}

def l3_expand_one_green_polygon_step(state, **kwargs) -> Dict[str, Any]:
    """
    扩张单步：从 allocation_plan 里按序取当前步，尝试把某个 green polygon 扩张一小步。

    约束：不 overlap forbidden_union_m
    不合并：通过 forbidden_union 的构造（把非绿地都禁入），以及可选把其它绿地也禁入实现。

    需要的 cache:
      - cache.green.allocation_plan
      - cache.forbidden.forbidden_union_m
      - cache.projection.to_local / to_wgs84

    关键：这里不要求 executor 传 id；默认从 plan 中 pop 一步。
    """
    buffer_min = float(kwargs.get("buffer_min_m", 1.0))
    buffer_max = float(kwargs.get("buffer_max_m", 20.0))
    trials = int(kwargs.get("trials", 10))

    gc = _green_cache(state)
    fb = _forbidden_cache(state)
    pc = _proj_cache(state)

    plan = list(gc.get("allocation_plan") or [])
    if not plan:
        return {"ok": False, "status": "NO_PLAN"}

    step = plan.pop(0)  # 消费一步
    gc["allocation_plan"] = plan

    target_id = str(step["id"])
    step_budget = float(step.get("step_budget_m2") or 0.0)
    if step_budget <= 0:
        return {"ok": False, "status": "BAD_BUDGET", "id": target_id}

    to_local = pc.get("to_local")
    to_wgs84 = pc.get("to_wgs84")
    if to_local is None:
        raise ValueError("to_local and to_wgs84 are required")

    forbidden = fb.get("forbidden_union_m", None)

    fc = _get_fc(state)
    got = _get_feature_by_id(fc, target_id)
    if not got:
        _append_attempt(state, target_id, step_budget, 0.0, None, "NOT_FOUND")
        return {"ok": False, "status": "NOT_FOUND", "id": target_id}

    idx, ft = got
    if not _is_polygonish(ft) or not is_green_feature(ft):
        _append_attempt(state, target_id, step_budget, 0.0, None, "NOT_GREEN_POLY")
        return {"ok": False, "status": "NOT_GREEN_POLY", "id": target_id}

    old_shp = ensure_valid(to_shapely(ft.get("geometry") or {}))
    old_m = ensure_valid(to_local(old_shp))
    old_area = float(getattr(old_m, "area", 0.0))

    # 半径采样：围绕一个 “能达到 step_budget 的近似半径” 做搜索
    # step_budget ≈ perimeter * r（粗略），用 sqrt(step/pi) 给个尺度
    r_guess = math.sqrt(max(step_budget, 1.0) / math.pi)
    r_guess = max(buffer_min, min(r_guess, buffer_max))

    radii = []
    for t in range(trials):
        # 在 [0.6, 1.4] * guess 里抖动
        jitter = 0.6 + (0.8 * (t / max(trials - 1, 1)))
        r = r_guess * jitter
        r = max(buffer_min, min(r, buffer_max))
        radii.append(r)

    best = None  # (added, r, new_m)
    for r in radii:
        try:
            cand = ensure_valid(old_m.buffer(r))
        except Exception:
            continue

        # hard no-overlap：把 forbidden 从候选里扣掉（并确保不缩回原形）
        if forbidden is not None and not getattr(forbidden, "is_empty", False):
            cand2 = _difference(cand, forbidden)
        else:
            cand2 = cand

        if cand2 is None or getattr(cand2, "is_empty", False):
            continue

        # 确保包含原几何（不 shrink）
        try:
            cand2 = ensure_valid(cand2.union(old_m))
        except Exception:
            pass

        new_area = float(getattr(cand2, "area", 0.0))
        added = max(0.0, new_area - old_area)

        if added <= 0:
            continue

        # 选择 added 最接近 step_budget（但不强制 <=）
        score = abs(added - step_budget)
        if best is None or score < best[0]:
            best = (score, added, r, cand2)

    if best is None:
        _append_attempt(state, target_id, step_budget, 0.0, None, "NO_FEASIBLE")
        return {"ok": True, "status": "NO_FEASIBLE", "id": target_id, "added_m2": 0.0}

    _, added, r_best, new_m = best
    # 写回 GeoJSON（回到 WGS84）
    new_wgs = ensure_valid(to_wgs84(new_m))
    new_gj = from_shapely(new_wgs)
    ft2 = copy.deepcopy(ft)
    ft2["geometry"] = new_gj

    # 写回 FC
    feats = _iter_features(fc)
    feats[idx] = ft2
    fc2 = dict(fc)
    fc2["features"] = feats
    _set_fc(state, fc2)

    # 更新 achieved
    gc["achieved_increase_m2"] = float(gc.get("achieved_increase_m2") or 0.0) + float(added)

    _append_attempt(state, target_id, step_budget, float(added), float(r_best), "OK")
    return {"ok": True, "status": "OK", "id": target_id, "added_m2": float(added), "buffer_m": float(r_best)}

def _append_attempt(state, gid: str, budget: float, added: float, buffer_m: Optional[float], status: str):
    gc = _green_cache(state)
    log = gc.get("attempt_log")
    if not isinstance(log, list):
        log = []
    log.append({
        "id": str(gid),
        "step_budget_m2": float(budget),
        "added_m2": float(added),
        "buffer_m": (None if buffer_m is None else float(buffer_m)),
        "status": status
    })
    gc["attempt_log"] = log

def l3_execute_expansion_loop(state, **kwargs) -> Dict[str, Any]:
    """
    批量执行扩张：不断调用 l3_expand_one_green_polygon_step，直到达标或 plan 用尽。
    """
    gc = _green_cache(state)
    need = float(gc.get("need_increase_m2") or 0.0)
    if need <= 0:
        return {"ok": True, "status": "ALREADY_REACHED"}

    max_total_steps = int(kwargs.get("max_total_steps", 120))
    tol = float(kwargs.get("tol", 0.02))

    steps = 0
    while steps < max_total_steps:
        achieved = float(gc.get("achieved_increase_m2") or 0.0)
        if achieved >= need * (1.0 - tol):
            break
        plan = gc.get("allocation_plan") or []
        if not plan:
            break
        l3_expand_one_green_polygon_step(state)
        steps += 1

    achieved = float(gc.get("achieved_increase_m2") or 0.0)
    return {"ok": True, "steps": steps, "achieved_increase_m2": achieved, "need_increase_m2": need}

def l3_refresh_green_area_stats(state, **kwargs) -> Dict[str, Any]:
    """
    重新计算绿地面积统计（after），直接复用 compute
    """
    return l3_compute_green_area_stats(state)

# ============================================================
# Validator-oriented helpers (NOT for executor allowlist)
# ============================================================

def v_snapshot_push(state, **kwargs) -> Dict[str, Any]:
    """
    Validator/loop 用：保存当前 FC 快照到 cache.snapshots 栈
    """
    c = _cache(state)
    stack = c.get("snapshots")
    if not isinstance(stack, list):
        stack = []
    fc = _get_fc(state)
    stack.append(copy.deepcopy(fc))
    c["snapshots"] = stack
    return {"ok": True, "depth": len(stack)}

def v_snapshot_pop(state, **kwargs) -> Dict[str, Any]:
    """
    Validator/loop 用：回滚到上一个快照
    """
    c = _cache(state)
    stack = c.get("snapshots")
    if not isinstance(stack, list) or not stack:
        return {"ok": False, "status": "EMPTY_STACK"}
    fc_prev = stack.pop()
    c["snapshots"] = stack
    _set_fc(state, fc_prev)
    return {"ok": True, "depth": len(stack)}

def v_check_no_overlap_with_forbidden(state, **kwargs) -> Dict[str, Any]:
    """
    Validator 用：检查绿地（当前几何）是否与 forbidden_union_m 有重叠。
    结果写入 cache.checks.overlap_forbidden
    """
    fc = _get_fc(state)
    gc = _green_cache(state)
    fb = _forbidden_cache(state)
    pc = _proj_cache(state)
    to_local = pc.get("to_local")

    if to_local is None:
        l3_prepare_local_projection(state)
        pc = _proj_cache(state)
        to_local = pc.get("to_local")

    forbidden = fb.get("forbidden_union_m", None)
    green_ids = list(gc.get("green_ids") or [])

    violations = []
    passed = True

    if forbidden is None or getattr(forbidden, "is_empty", False):
        passed = True
    else:
        for gid in green_ids:
            got = _get_feature_by_id(fc, gid)
            if not got:
                continue
            _, ft = got
            if not _is_polygonish(ft) or not is_green_feature(ft):
                continue
            shp = ensure_valid(to_shapely(ft.get("geometry") or {}))
            g_m = ensure_valid(to_local(shp))
            if _intersects(g_m, forbidden):
                oa = _intersection_area(g_m, forbidden)
                if oa > 0:
                    passed = False
                    violations.append({"id": gid, "overlap_area_m2": float(oa)})

    ck = _checks_cache(state)
    ck["overlap_forbidden"] = {"passed": passed, "violations": violations}
    return {"ok": True, "passed": passed, "n_violations": len(violations)}

def v_check_target_reached(state, **kwargs) -> Dict[str, Any]:
    """
    Validator 用：检查是否达到目标总绿地面积（含 tol）。
    依赖 cache.green.green_area_total_m2 / target_total_m2
    """
    tol = float(kwargs.get("tol", 0.02))
    gc = _green_cache(state)
    cur = float(gc.get("green_area_total_m2") or 0.0)
    target = float(gc.get("target_total_m2") or 0.0)
    passed = (cur >= target * (1.0 - tol)) if target > 0 else True

    ck = _checks_cache(state)
    ck["target_reached"] = {"passed": passed, "current_total_m2": cur, "target_total_m2": target, "tol": tol}
    return {"ok": True, "passed": passed, "cur": cur, "target": target, "tol": tol}
