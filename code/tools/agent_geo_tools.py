from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Callable
from pathlib import Path
import re

# ==== A. 几何 / 投影 / 面积 / 质心 ==========================================

from .utils import (
    geom_area_m2,
    calculate_total_area,
    geom_centroid_lonlat,
    calculate_centroid,
    pick_utm_epsg,
    get_project_funcs_internal,
    create_transformer_funcs,
    ensure_valid,
    to_shapely,
    from_shapely,
)


# ==== B. 绿地识别 & 属性 patch ==============================================

from .utils import (
    GREEN_TAGS,
    is_green_feature,
    random_green_kv,
    patch_feature_to_green_level1,
    patch_feature_directive,
)

# ==== C. 扩张 / 禁入 / 留缝 / 聚类 / 排序 ===================================

from .utils import (
    cluster_and_union_green_features,
    get_sorted_green_features_from_list,
    build_forbidden_union_m,
    build_dynamic_forbidden_m,
    process_single_expansion,
    enforce_min_gap_between_features,
    remove_fully_covered_features,
)

def dedupe_identical_polygons(fc: dict) -> dict:
    """
    删除几何完全相同(拓扑等价)的重复 Polygon/MultiPolygon。
    保留优先级：green > non-green；若均同类则保留先出现的。
    仅比较 Polygon/MultiPolygon；其它类型直接原样收集。
    """
    try:
        import shapely
        from shapely.geometry import shape as _shape
    except Exception:
        # 没有 shapely 就直接返回
        return fc

    feats = list(fc.get("features", []))
    out = []
    seen = {}  # key -> index_in_out

    def _geom_key(gj):
        try:
            s = ensure_valid(_shape(gj)).buffer(0)
            # 用规范化 WKB 作键，避免坐标顺序差异
            return s.wkb
        except Exception:
            return None

    for ft in feats:
        gj = (ft.get("geometry") or {})
        gtype = gj.get("type", "")
        if gtype not in ("Polygon", "MultiPolygon"):
            out.append(ft)
            continue

        key = _geom_key(gj)
        if key is None:
            out.append(ft)
            continue

        is_green = bool(is_green_feature(ft))

        if key not in seen:
            seen[key] = len(out)
            out.append(ft)
        else:
            # 冲突：已有相同几何
            kept_idx = seen[key]
            kept = out[kept_idx]
            kept_green = bool(is_green_feature(kept))
            # 绿地优先
            if is_green and not kept_green:
                out[kept_idx] = ft
            # 若两者同为绿地或同为非绿地，则保留原来的，不动

    fc2 = {"type": "FeatureCollection", "features": out}
    # 维持 metadata
    if "metadata" in fc:
        fc2["metadata"] = fc["metadata"]
    return fc2

# ==== D. 摘要 & 连通性 & pair 对比 =========================================

from .utils import (
    summarize_fc,
    summarize_pair,
    _union_area_m2_of_subset,
    _to_local_polys,
    _count_components,
    _bin_density,
)

# ==== E. L1/L2/L3 候选筛选 & 贪心选面 =============
def select_l1_editable_candidates(fc: Dict[str, Any]) -> List[Tuple[int, float]]:
    """
    Level-1 用的候选集合选择：

    - 仅考虑 geometry.type in {Polygon, MultiPolygon}
    - tags 为空 或 is_green_feature(ft) 为 False 的要素
    - 返回列表 [(feature_index, area_m2), ...]
    """
    feats = list(fc.get("features", []))
    candidates: List[Tuple[int, float]] = []

    for i, ft in enumerate(feats):
        geom = ft.get("geometry", {}) or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        tags = ft.get("properties", {}).get("tags", {})
        tags_is_empty = (not isinstance(tags, dict)) or (len(tags) == 0)

        if tags_is_empty or (not is_green_feature(ft)):
            area = geom_area_m2(geom)
            if area > 0:
                candidates.append((i, area))

    return candidates


def greedy_select_by_area(
    candidates: List[Tuple[int, float]],
    target_area: float,
    tol: float,
) -> Tuple[List[int], float]:
    """
    通用贪心选面函数（用于 Level-1 / Level-2 seeding）：

    输入：
        candidates: [(idx, area_m2), ...]（未排序 & 已排序都能接受）
        target_area: float，目标面积
        tol: float，容差，如 0.05

    返回：
        selected_indices: List[int]
        achieved_area: float
    """
    if not candidates:
        return [], 0.0

    # 按面积从大到小排序（内部不依赖外部排序）
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    selected: List[int] = []
    achieved = 0.0

    lower = target_area * (1.0 - tol)
    upper = target_area * (1.0 + tol)

    # --- Step 1: 主贪心：一直选直到达到 lower ---
    for idx, area in candidates:
        if achieved >= lower:
            break
        selected.append(idx)
        achieved += area

    # --- Step 2: 如果仍然未达标，尝试补一点但不超过 upper ---
    if achieved < lower:
        remaining = candidates[len(selected):]
        for idx, area in remaining:
            # 不能超过 upper
            if achieved + area <= upper:
                selected.append(idx)
                achieved += area
            if achieved >= lower:
                break

    # --- Step 3: 返回结果（若仍未达到 lower，也就是“尽力而为”） ---
    return selected, achieved


def score_l2_edge_candidates(
    fc: Dict[str, Any],
    to_local: Callable,
) -> List[Tuple[int, float, float]]:
    """
    L2 无绿地场景用的“边缘候选评分”：

    - 在局地坐标系下：
        1) 计算 bbox（所有 Polygon/MultiPolygon）
        2) 对每个 Polygon/MultiPolygon:
            - 计算质心到 bbox 边界的最短距离 dist_edge
            - 计算面积 area_m2
    - 返回排序好的列表: [(idx, dist_edge, area_m2), ...]
      排序规则：dist_edge 升序，其次 area_m2 降序（靠边 + 面积大优先）
    """
    from shapely.geometry import box

    feats = list(fc.get("features", []))

    # 1) 收集所有面要素的 bounds（local 米制）
    all_bounds: List[Tuple[float, float, float, float]] = []
    for ft in feats:
        geom = (ft.get("geometry") or {})
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        s = to_local(ensure_valid(to_shapely(geom)))
        if s.is_empty:
            continue
        all_bounds.append(s.bounds)

    if not all_bounds:
        return []

    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    bbox = box(minx, miny, maxx, maxy)

    # 2) 对每个 Polygon/MultiPolygon 计算 “离 bbox 边界的距离 + 面积”
    scored: List[Tuple[int, float, float]] = []
    for i, ft in enumerate(feats):
        geom = (ft.get("geometry") or {})
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        s = to_local(ensure_valid(to_shapely(geom)))
        if s.is_empty:
            continue

        c = s.centroid
        dist_edge = c.distance(bbox.boundary)  # 越小越靠边缘
        area_m2 = float(s.area)               # 米制面积

        scored.append((i, dist_edge, area_m2))

    # 3) 排序：dist_edge 升序，其次 area_m2 降序（靠边 + 大块优先）
    scored.sort(key=lambda x: (x[1], -x[2]))

    return scored


def select_seed_polygons_for_l2(
    fc: Dict[str, Any],
    ratio: float,
    tol: float,
    to_local: Callable,
) -> Tuple[List[int], float]:
    """
    L2 Case B（无绿地）的一站式选址器：

    - 先按“靠 bbox 边缘 + 面积大”给所有面要素打分
      （score_l2_edge_candidates）
    - 再用 greedy_select_by_area 在 [target*(1-tol), target*(1+tol)]
      区间内贪心选若干块
    - 返回：
        chosen_indices: 被当做“初始绿地”的面索引（在 fc.features 里的 idx）
        achieved_area_m2: 这些面在局地投影下的面积和（米²）
    """
    feats = list(fc.get("features", []))
    if not feats:
        return [], 0.0

    # ① 用你原来的 total_area * ratio 逻辑算目标面积
    total_area = calculate_total_area(feats)  # 用的还是 utils 里的那个
    if total_area <= 0.0:
        return [], 0.0
    target_area = total_area * ratio

    # ② 对所有 Polygon/MultiPolygon 做“边缘评分”
    #    scored: [(idx, dist_edge, area_m2_local), ...]
    scored = score_l2_edge_candidates(fc, to_local)
    if not scored:
        return [], 0.0

    # ③ 把 (idx, dist, area) 转成 greedy_select_by_area 需要的格式
    #    candidates: [(idx, area_m2_local), ...]
    candidates: List[Tuple[int, float]] = [
        (idx, area_m2) for (idx, _dist_edge, area_m2) in scored
    ]

    # ④ 用统一的贪心逻辑在 [target*(1-tol), target*(1+tol)] 里尽量贴近目标
    selected_indices, achieved_area = greedy_select_by_area(
        candidates=candidates,
        target_area=target_area,
        tol=tol,
    )

    # ⑤ 兜底：如果完全没选上，但 scored 不空，就至少选最靠边的那一块
    if not selected_indices and scored:
        first_idx, _dist_edge, area_m2 = scored[0]
        selected_indices = [first_idx]
        achieved_area = float(area_m2)

    return selected_indices, float(achieved_area)


# ==== F. 评估指标（基于 summarize_fc / summarize_pair 的包装，可选）=======

def compute_area_compliance(
    before_fc: Dict[str, Any],
    after_fc: Dict[str, Any],
    target_ratio: float,
    tol: float,
) -> Dict[str, Any]:
    """
    计算 Area Compliance 指标：
    Apred_green - Abefore_green ∈ [Atarget(1-tol), Atarget(1+tol)]

    这里：
      - Atarget = target_ratio * A_total_editable
      - A_total_editable 用 BEFORE 的 total_editable_m2
    """
    # 1) 摘要统计
    s_before = summarize_fc(before_fc)
    s_after = summarize_fc(after_fc)

    total_editable_m2 = float(s_before.get("total_editable_m2", 0.0))
    green_before = float(s_before.get("green_m2", 0.0))
    green_after = float(s_after.get("green_m2", 0.0))

    # 2) 如果没有可编辑面积，退化为 trivially pass（啥也做不了）
    if total_editable_m2 <= 0.0:
        return {
            "metric": "Area_Compliance",
            "pass": True,
            "delta_green_m2": 0.0,
            "target_area_m2": 0.0,
            "abs_diff_pct": 0.0,
        }

    # 3) 目标面积 & 绿地增量
    target_area = target_ratio * total_editable_m2
    delta_green = green_after - green_before

    lower = target_area * (1.0 - tol)
    upper = target_area * (1.0 + tol)

    pass_flag = (delta_green >= lower) and (delta_green <= upper)

    eps = 1e-9
    abs_diff_pct = abs(delta_green - target_area) / max(target_area, eps)

    return {
        "metric": "Area_Compliance",
        "pass": bool(pass_flag),
        "delta_green_m2": float(delta_green),
        "target_area_m2": float(target_area),
        "abs_diff_pct": float(abs_diff_pct),
    }


def compute_connectivity_improvement(
    before_fc: Dict[str, Any],
    after_fc: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Level-3 用的 Connectivity Improvement 指标：

    improvement = 1 - N_after / max(N_before, 1)

    其中 N_before / N_after 来自 summarize_fc(... ) 里的 n_green_components。
    """
    s_before = summarize_fc(before_fc)
    s_after = summarize_fc(after_fc)

    n_before = int(s_before.get("n_green_components", 0) or 0)
    n_after = int(s_after.get("n_green_components", 0) or 0)

    if n_before > 0:
        improvement = 1.0 - (n_after / float(n_before))
    else:
        # 没有初始绿地：
        # - 若 after 仍然没有绿地：认为没有变化 -> 0
        # - 若 after 出现连通绿地块：可以视作“从 0 到有” -> 1.0
        improvement = 1.0 if n_after > 0 else 0.0

    return {
        "metric": "Connectivity_Improvement",
        "improvement": float(improvement),
        "n_before": n_before,
        "n_after": n_after,
    }


def compute_validity_rate(fc: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算几何合法率 Validity_Rate：

    - 遍历所有 Polygon/MultiPolygon:
        - 尝试 to_shapely + ensure_valid
        - 统计“成功且非空”的个数 versus “总面要素数”
    """
    feats = list(fc.get("features", [])) or []

    total_faces = 0
    valid_count = 0

    for ft in feats:
        geom = ft.get("geometry") or {}
        gtype = geom.get("type")

        # 只统计面要素
        if gtype not in ("Polygon", "MultiPolygon"):
            continue

        total_faces += 1

        try:
            s = to_shapely(geom)
        except Exception:
            # 解析失败直接视为无效
            continue

        if s is None:
            continue

        s2 = ensure_valid(s)
        if s2 is None or s2.is_empty:
            continue

        valid_count += 1

    rate = (valid_count / total_faces) if total_faces > 0 else 0.0

    return {
        "metric": "Validity_Rate",
        "valid_count": int(valid_count),
        "total_faces": int(total_faces),
        "rate": float(rate),
    }


# ==== G. plan.json & Level 辅助 (给 TaskPlanner / GeoExecutor 用) ===========

def infer_level(plan: Dict[str, Any]) -> str:
    """
    从 plan.json 推断 Level 类型:
        - 首选使用 plan["level"]（如 "L1"/"L2"/"L3" 或 1/2/3）
        - 若缺失，可从 _edit_intent / edit_intent 中推断
        - 默认回退到 "L1"
    """
    lvl = plan.get("level")

    # 1) 直接解析 level 字段
    if isinstance(lvl, str):
        s = lvl.strip().upper()
        if s in ("L1", "L2", "L3"):
            return s
        if s in ("1", "2", "3"):
            return "L" + s
    elif isinstance(lvl, int):
        if lvl in (1, 2, 3):
            return f"L{lvl}"

    # 2) 从 intent 里猜
    intent = plan.get("edit_intent") or plan.get("_edit_intent")
    if isinstance(intent, str):
        low = intent.lower()
        if "level3" in low or "l3" in low:
            return "L3"
        if "level2" in low or "l2" in low:
            return "L2"
        if "level1" in low or "l1" in low:
            return "L1"

    # 3) 默认
    return "L1"


def extract_targets(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 plan.json 中抽取执行需要的核心参数，统一成一个 flat dict。

    方便 GeoExecutor 直接使用。
    """
    level = infer_level(plan)

    # 一些基础字段
    ratio = float(plan.get("ratio", 0.3))
    tol = float(plan.get("tolerance", plan.get("tol", 0.05)))

    cluster_dist_m = float(plan.get("cluster_dist_m", 8.0))
    gap_m = float(plan.get("gap_m", 2.0))
    buffer_m_range_raw = plan.get("buffer_m_range", [4.0, 20.0])
    if isinstance(buffer_m_range_raw, (list, tuple)) and len(buffer_m_range_raw) >= 2:
        buffer_m_range = (float(buffer_m_range_raw[0]), float(buffer_m_range_raw[1]))
    else:
        buffer_m_range = (4.0, 20.0)

    max_iter = int(plan.get("max_iter", 10))
    purge_enabled = bool(plan.get("purge_enabled", False))
    seed = int(plan.get("seed", 42))

    # setbacks
    setbacks_raw = plan.get("setbacks", {}) or {}
    setbacks = {
        "building": float(setbacks_raw.get("building", 2.0)),
        "way": float(setbacks_raw.get("way", 2.0)),
        "small_gap": float(setbacks_raw.get("small_gap", 0.0)),
    }

    # CRS / 其他参数
    params = plan.get("params", {}) or {}
    crs_planar = params.get("crs_planar", "auto_local_utm")

    # 路径信息（给 executor/aggregator 用）
    paths_raw = plan.get("paths", {}) or {}
    paths = {
        "before": paths_raw.get("before"),
        "prompt": paths_raw.get("prompt"),
        "label": paths_raw.get("label"),  # 预留，可能没有
    }

    targets: Dict[str, Any] = {
        "level": level,
        "ratio": ratio,
        "tolerance": tol,
        "cluster_dist_m": cluster_dist_m,
        "gap_m": gap_m,
        "buffer_m_range": buffer_m_range,
        "max_iter": max_iter,
        "purge_enabled": purge_enabled,
        "seed": seed,
        "setbacks": setbacks,
        "crs_planar": crs_planar,
        "paths": paths,
    }
    return targets


def check_preconditions(fc: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行前检查一些前置条件，并给出是否需要 fallback 的建议，例如:

    - total_area <= 0 → 不执行高阶几何，建议回退到 L1 属性 patch
    - level == "L3" 但没有任何绿地 → 可建议回退到 L2
    - 依赖的投影/库不可用 → 回退到 L1
    """
    level = targets.get("level", "L1")
    feats = list(fc.get("features", []))

    # 1) 总面积检查
    total_area = calculate_total_area(feats)
    if total_area <= 0.0:
        return {
            "ok": False,
            "fallback_level": "L1",
            "reason": "total editable area <= 0, fallback to Level-1 tag patch.",
        }

    # 2) L3 需要已有绿地，否则没法做 “connectivity-aware 扩张”
    has_green = any(is_green_feature(ft) for ft in feats)
    if level == "L3" and not has_green:
        return {
            "ok": False,
            "fallback_level": "L2",
            "reason": "Level-3 requires existing green space but none found; fallback to Level-2.",
        }

    # 3) 几何依赖库 (shapely/pyproj) 是否可用
    if level in ("L2", "L3"):
        try:
            import shapely  # noqa: F401
            import pyproj   # noqa: F401
        except Exception:
            return {
                "ok": False,
                "fallback_level": "L1",
                "reason": "shapely/pyproj not available; fallback to Level-1.",
            }

    # 一切正常
    return {
        "ok": True,
        "fallback_level": None,
        "reason": None,
    }


def decide_recipe(
    level: str,
    has_green: bool,
    purge_enabled: bool,
) -> str:
    """
    给 GeoExecutor 用的决策器，用于选择具体的“执行配方”：

    例如:
        - level == "L1" -> "L1_PATCH"
        - level == "L2" and has_green -> "L2_EXPAND"
        - level == "L2" and not has_green -> "L2_SEED"
        - level == "L3" and purge_enabled -> "L3_EXPAND_THEN_PURGE"
        - level == "L3" and not purge_enabled -> "L3_EXPAND_ONLY" (若你未来加)

    返回字符串 recipe_id，便于在 geo_executor.py 里做路由。
    """
    if level == "L1":
        return "L1_PATCH"
    if level == "L2":
        return "L2_EXPAND" if has_green else "L2_SEED"
    if level == "L3":
        return "L3_EXPAND_THEN_PURGE" if purge_enabled else "L3_EXPAND_ONLY"
    return "UNKNOWN"


def shapely_available() -> bool:
    try:
        import shapely  # noqa: F401
        return True
    except Exception:
        return False

def repair_polygon_geojson(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Deterministic "repair" used for ValidityRate:
    - If shapely exists: shape -> (make_valid if available else buffer(0)) -> mapping
    - Else: return original geometry dict
    """
    if not isinstance(g, dict):
        return None
    if g.get("type") not in ("Polygon", "MultiPolygon"):
        return None

    if not shapely_available():
        return g

    try:
        from shapely.geometry import shape, mapping
        geom = shape(g)
        if geom.is_empty:
            return None

        # Prefer make_valid when available
        try:
            from shapely.validation import make_valid  # shapely >= 2.0
            geom2 = make_valid(geom)
        except Exception:
            geom2 = geom.buffer(0)

        if geom2 is None or geom2.is_empty:
            return None

        # Keep only polygonal parts
        if geom2.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
            return None

        if geom2.geom_type == "GeometryCollection":
            polys = [p for p in geom2.geoms if p.geom_type in ("Polygon", "MultiPolygon") and (not p.is_empty)]
            if not polys:
                return None
            from shapely.ops import unary_union
            geom2 = unary_union(polys)

        if geom2.is_empty:
            return None

        return mapping(geom2)
    except Exception:
        return g


def count_green_connected_components(fc: Dict[str, Any]) -> Optional[int]:
    """
    Connected components of union of green polygons.
    - If shapely exists: unary_union over green polygons -> count Polygon=1 / MultiPolygon=len(parts)
    - Else: fallback = number of green polygon features (upper bound; weak but deterministic)
    """
    feats = list((fc or {}).get("features", []) or [])

    if not shapely_available():
        # fallback: count green polygon features
        c = 0
        for ft in feats:
            try:
                if not is_green_feature(ft):
                    continue
                g = (ft or {}).get("geometry", {}) or {}
                if g.get("type") in ("Polygon", "MultiPolygon"):
                    c += 1
            except Exception:
                continue
        return int(c)

    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union

        geoms = []
        for ft in feats:
            try:
                if not is_green_feature(ft):
                    continue
                g = (ft or {}).get("geometry", {}) or {}
                if g.get("type") not in ("Polygon", "MultiPolygon"):
                    continue
                gg = repair_polygon_geojson(g) or g
                geom = shape(gg)
                if geom.is_empty:
                    continue
                # light fix
                if not geom.is_valid:
                    try:
                        geom = geom.buffer(0)
                    except Exception:
                        pass
                if geom.is_empty:
                    continue
                geoms.append(geom)
            except Exception:
                continue

        if not geoms:
            return 0

        u = unary_union(geoms)
        if u.is_empty:
            return 0
        if u.geom_type == "Polygon":
            return 1
        if u.geom_type == "MultiPolygon":
            return int(len(u.geoms))
        if u.geom_type == "GeometryCollection":
            polys = [p for p in u.geoms if p.geom_type == "Polygon" and (not p.is_empty)]
            return int(len(polys))

        # unexpected
        return 0
    except Exception:
        return None


def find_label_geojson(lat_dir: Path) -> Optional[Path]:
    """
    Best-effort locate procedural LABEL geojson under the same lat_dir.
    Adjust candidate names to your dataset conventions.
    """
    lat_dir = Path(lat_dir)

    p = lat_dir / f"{lat_dir.name}_label.geojson"
    if p.exists():
        return p



    # (D) Fallback: any *_label.geojson in this folder
    hits = sorted(lat_dir.glob("*_label.geojson"))
    if hits:
        return hits[0]

    return None


def sanitize_model_id(model_id: str) -> str:
    s = (model_id or "").strip()
    s = s.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    return s


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def total_polygon_area_m2(fc: Dict[str, Any]) -> float:
    feats = list((fc or {}).get("features", []) or [])
    s = 0.0
    for ft in feats:
        g = (ft or {}).get("geometry", {}) or {}
        if g.get("type") in ("Polygon", "MultiPolygon"):
            try:
                s += float(geom_area_m2(g) or 0.0)
            except Exception:
                pass
    return float(s)


def green_area_m2(fc: Dict[str, Any]) -> float:
    feats = list((fc or {}).get("features", []) or [])
    s = 0.0
    for ft in feats:
        try:
            if is_green_feature(ft):
                g = (ft or {}).get("geometry", {}) or {}
                if g.get("type") in ("Polygon", "MultiPolygon"):
                    s += float(geom_area_m2(g) or 0.0)
        except Exception:
            continue
    return float(s)






reporter_USER_TEMPLATE = """PROMPT TEXT:
{prompt}

GEOJSON SUMMARY (BEFORE):
{geo_summary}

Infer the task level (L1/L2/L3) and all parameters strictly following the classification rules provided in the system prompt.

Return ONLY the JSON object — no extra text.
"""

reporter_SYSTEM_PROMPT = """You are TaskPlanner, an expert geospatial analyst. Your job is to infer the original task level (L1, L2, or L3) from a natural-language editing instruction (PROMPT TEXT) and the initial geospatial context (GEOJSON SUMMARY).

CRITICAL CLASSIFICATION RULES — USE THESE FIRST (they are deterministic fingerprints):

- L1 (pure tag-only / geometry-locked):
  - **Immediately classify as L1 if**: Contains phrases like: "re-label", "mark as green", "flip semantics", "without changing any geometry", "keep vertex and feature counts unchanged", "do not split, merge, expand, or delete".
  - **Immediate L1 classification** if NO **geometric modification verbs** are found, such as: expand, extend, merge, union, dilate, grow, cover, consume.
  - **Immediate L1 classification** if NO **deletion verbs** are found, such as: remove, delete, purge, erase, clean up covered.
  - Focus is **exclusively** on **semantic changes or selection order**.

- L2 (gentle/conservative extension, no deletions):
  - **Immediately classify as L2 if**: Contains geometric verbs like: "extend", "expand", "buffer", "setback", "outward adjustments", "preserve non-green", "avoid covering", "reject if it would consume", "build forbidden zones", "gentle", "modest".
  - **Immediate L2 classification** if it involves **carefully adjusted expansions** or spatial constraints that do not involve deletions or major coverage.
  - **DO NOT classify as L2 if**: Deletion words like remove, delete, purge, erase, or "fully covered → remove" are present.
  - Emphasizes **preservation**, **spacing**, and **rejection of over-aggressive steps**.

- L3 (aggressive consolidation with cleanup):
  - **Immediately classify as L3 if**: Contains verbs like: "merge", "simplify", "remove", "purge", "clean", "consume", "delete", "cover", "erase".
  - **Immediate L3 classification** if **deletion verbs** appear in the prompt, such as: "remove", "delete", "purge", "clean up", "erase", "cover non-green areas", "remove fully covered features".
  - **Immediate L3 classification** if the task explicitly mentions **covering** or **deleting** areas, or **simplifying** layouts.

Decision priority:
1. **Scan for forbidden words first**:
   - **If any deletion word appears** → **L3** (Immediate).
   - **If geometric extension words appear** but no deletion words → **L2** (Immediate).
   - **If only re-labeling** and **explicit "no geometry change"** → **L1** (Immediate).
2. **If conflict**, stronger verbs (remove/cover) override weaker ones.
3. **GEOJSON SUMMARY** can provide supporting evidence (e.g., large delta area + reduced feature count strongly suggests L3), but **language rules above take precedence**.

Now infer all other parameters based on the prompt's described strategy (verbalized numbers, buffers, iterations, etc.). Use reasonable defaults when not mentioned:
- ratio: extract from phrases like "about X%" → X/100
- gap_m / setbacks: extract from "~X m spacing", "buffer ~X m", "setback ~X m"
- buffer_m_range: usually [4, 20] or similar range mentioned
- max_iter: often 8–10 if loops mentioned
- purge_enabled: True only for L3
- stochastic_strength: low (0.1–0.3) if "fixed seed" or "deterministic" emphasized; higher if variation encouraged
- seed: usually 42 if mentioned

Return ONLY a valid JSON object with exactly this structure (no explanations, no markdown, no trailing text):
{
  "level": "L1" | "L2" | "L3",
  "ratio": float between 0 and 1,
  "tolerance": float between 0 and 0.2,
  "cluster_dist_m": float > 0,
  "gap_m": float > 0,
  "setbacks": {"building": float >=0, "way": float >=0, "small_gap": float >=0},
  "buffer_m_range": [float >0, float >= first],
  "max_iter": int between 1 and 20,
  "purge_enabled": true | false,
  "stochastic_strength": float between 0.0 and 0.6,
  "seed": int
}
"""