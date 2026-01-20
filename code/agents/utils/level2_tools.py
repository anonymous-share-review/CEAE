from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

from shapely.geometry import shape, mapping, LineString
from pyproj import CRS, Transformer


def _normalize_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return s


def _get_properties(feat: Dict[str, Any]) -> Dict[str, Any]:
    return feat.get("properties") or {}


def _get_pid(feat: Dict[str, Any]) -> Optional[str]:
    """
    Canonical feature handle: ONLY feat["properties"]["id"] (string)
    """
    props = _get_properties(feat)
    return _normalize_id(props.get("id", None))


def _utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def _to_planar_transformer(crs_from: CRS, crs_to: CRS) -> Transformer:
    return Transformer.from_crs(crs_from, crs_to, always_xy=True)


def _vec_from_bearing(bearing_deg: float) -> Tuple[float, float]:
    rad = math.radians(bearing_deg)
    # dx, dy in planar coordinates; 0°=North
    return math.sin(rad), math.cos(rad)


def _bearing_from_vec(dx: float, dy: float) -> float:
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0


def _length_linestring_planar(coords_lonlat: List[Tuple[float, float]]) -> Tuple[float, int]:
    """
    Compute linestring length in meters using local UTM.
    Returns (length_m, planar_epsg).
    """
    if len(coords_lonlat) < 2:
        return 0.0, 4326

    lon0, lat0 = coords_lonlat[0]
    wgs84 = CRS.from_epsg(4326)
    crs_planar = _utm_crs_for_lonlat(lon0, lat0)
    tf = _to_planar_transformer(wgs84, crs_planar)

    coords_p = [tf.transform(lon, lat) for lon, lat in coords_lonlat]
    ls_p = LineString(coords_p)
    return float(ls_p.length), int(crs_planar.to_epsg())


def _move_lonlat_planar(
    origin_lonlat: Tuple[float, float],
    bearing_deg: float,
    distance_m: float,
    planar_epsg: int,
) -> Tuple[float, float]:
    """
    Move a lon/lat point by distance along bearing (0°=North) in planar CRS.
    """
    lon, lat = origin_lonlat
    wgs84 = CRS.from_epsg(4326)
    crs_planar = CRS.from_epsg(int(planar_epsg))
    tf_fwd = _to_planar_transformer(wgs84, crs_planar)
    tf_inv = _to_planar_transformer(crs_planar, wgs84)

    x, y = tf_fwd.transform(lon, lat)
    ux, uy = _vec_from_bearing(float(bearing_deg))
    nx = x + float(distance_m) * ux
    ny = y + float(distance_m) * uy
    nlon, nlat = tf_inv.transform(nx, ny)
    return float(nlon), float(nlat)


@dataclass
class GeoState:
    """
    IMPORTANT:
    - feature_handle is ALWAYS str(feature.properties.id)
    - We build index ONLY from feature.properties.id
    - scratch stores intermediate values; NOT written into GeoJSON.
    """
    fc: Dict[str, Any]
    scratch: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.fc.get("type") == "FeatureCollection", "State must be a GeoJSON FeatureCollection"
        self._index: Dict[str, Dict[str, Any]] = {}
        for feat in self.fc.get("features", []):
            pid = _get_pid(feat)
            if pid is None:
                continue
            self._index[pid] = feat

    def get_feature(self, feature_id: Any) -> Optional[Dict[str, Any]]:
        hid = _normalize_id(feature_id)
        if hid is None:
            return None
        return self._index.get(hid)

    def get_shapely(self, feature_id: Any):
        hid = _normalize_id(feature_id)
        if hid is None:
            raise KeyError("Feature id is null/empty")
        feat = self.get_feature(hid)
        if feat is None:
            raise KeyError(f"Feature not found by properties.id: {hid}")
        return shape(feat["geometry"])

    def update_geometry(self, feature_id: Any, new_geom) -> None:
        hid = _normalize_id(feature_id)
        if hid is None:
            raise KeyError("Feature id is null/empty")
        feat = self.get_feature(hid)
        if feat is None:
            raise KeyError(f"Feature not found by properties.id: {hid}")
        feat["geometry"] = mapping(new_geom)



# --------------------------
# 1) get_feature_by_id (optional)
# --------------------------
def get_feature_by_id(state: GeoState, id: str) -> Dict[str, Any]:
    hid = _normalize_id(id)
    if hid is None:
        raise KeyError("Feature id is null/empty")
    feat = state.get_feature(hid)
    if feat is None:
        raise KeyError(f"Feature not found by properties.id: {hid}")
    return {"feature_handle": hid}


# --------------------------
# 2) get_feature_geometry
# --------------------------
def get_feature_geometry(state: GeoState, feature: str) -> Dict[str, Any]:
    hid = _normalize_id(feature)
    if hid is None:
        raise KeyError("Feature handle is null/empty")
    geom = state.get_shapely(hid)
    return {"geometry": mapping(geom)}


def get_linestring_stats(state: GeoState, line: str) -> Dict[str, Any]:
    l_id = _normalize_id(line)
    if l_id is None:
        raise ValueError("line handle is null/empty")

    geom = state.get_shapely(l_id)
    if geom.geom_type != "LineString":
        raise ValueError("line must be a LineString")

    coords = list(geom.coords)  # lon/lat
    length_m, planar_epsg = _length_linestring_planar(coords)

    # cache working copy into scratch
    state.scratch["l2.line_id"] = l_id
    state.scratch["l2.orig_coords"] = [(float(x), float(y)) for x, y in coords]
    state.scratch["l2.work_coords"] = [(float(x), float(y)) for x, y in coords]
    state.scratch["l2.planar_epsg"] = int(planar_epsg)
    state.scratch["l2.length_m"] = float(length_m)

    return {
        "line": l_id,
        "num_points": int(len(coords)),
        "length_m": float(length_m),
        "planar_epsg": int(planar_epsg),
    }


# --------------------------
# 4) select_linestring_endpoint
# --------------------------
def select_linestring_endpoint(state: GeoState, line: str, endpoint: str) -> Dict[str, Any]:
    l_id = _normalize_id(line)
    if l_id is None:
        raise ValueError("line handle is null/empty")

    # prefer scratch; fall back to reading geometry
    coords = state.scratch.get("l2.work_coords")
    if not coords or state.scratch.get("l2.line_id") != l_id:
        geom = state.get_shapely(l_id)
        if geom.geom_type != "LineString":
            raise ValueError("line must be a LineString")
        coords = [(float(x), float(y)) for x, y in list(geom.coords)]
        state.scratch["l2.line_id"] = l_id
        state.scratch["l2.work_coords"] = coords

    ep = str(endpoint).lower().strip()
    if ep not in ("head", "tail"):
        raise ValueError("endpoint must be 'head' or 'tail'")

    state.scratch["l2.endpoint"] = ep

    if len(coords) < 2:
        return {"line": l_id, "endpoint": ep, "endpoint_lonlat": None, "prev_lonlat": None}

    if ep == "tail":
        end_pt = coords[-1]
        prev_pt = coords[-2]
    else:
        end_pt = coords[0]
        prev_pt = coords[1]

    # store for later
    state.scratch["l2.endpoint_lonlat"] = end_pt
    state.scratch["l2.prev_lonlat"] = prev_pt

    return {
        "line": l_id,
        "endpoint": ep,
        "endpoint_lonlat": [end_pt[0], end_pt[1]],
        "prev_lonlat": [prev_pt[0], prev_pt[1]],
    }


# --------------------------
# 5) compute_length_change_m
# --------------------------
def set_length_change_m(state: GeoState, delta_m: float) -> Dict[str, Any]:
    if delta_m is None:
        raise ValueError("delta_m is null")

    try:
        dm = float(delta_m)  # 关键：把 "12" -> 12.0
    except Exception:
        raise ValueError(f"delta_m not numeric: {repr(delta_m)}")


    if dm <= 0:
        raise ValueError(f"delta_m must be > 0, got {dm}")

    state.scratch["l2.change_m"] = float(delta_m)
    state.scratch["l2.change_source"] = "plan"

    return {
        "change_m": float(delta_m),
        "source": "plan"
    }

def compute_length_change_m(
    state: GeoState,
    rule: str = "min(random, 0.5*len)",
    max_m: float = 20.0,
    min_m: float = 5.0,
) -> Dict[str, Any]:
    """
    Default matches your label logic:
    - random_target_m ~ Uniform(min_m, max_m)
    - change_m = min(random_target_m, 0.5 * length_m)
    """
    length_m = state.scratch.get("l2.length_m")
    if length_m is None:
        raise ValueError("missing l2.length_m in scratch; call get_linestring_stats first")

    r = str(rule).strip().lower()
    max_m = float(max_m)
    min_m = float(min_m)
    if min_m < 0 or max_m <= 0 or min_m > max_m:
        raise ValueError("invalid min_m/max_m")

    if r == "min(random, 0.5*len)":
        random_target_m = random.uniform(min_m, max_m)
        half_length = 0.5 * float(length_m)
        change_m = float(min(random_target_m, half_length))
        rule_used = r
    else:
        # fallback: just random in [min_m, max_m] but still clamp by 0.5*len
        random_target_m = random.uniform(min_m, max_m)
        half_length = 0.5 * float(length_m)
        change_m = float(min(random_target_m, half_length))
        rule_used = f"fallback({r})"

    state.scratch["l2.change_m"] = change_m
    state.scratch["l2.change_rule"] = rule_used

    ratio = 0.0 if float(length_m) <= 0 else float(change_m) / float(length_m)
    state.scratch["l2.change_ratio"] = ratio

    return {
        "change_m": float(change_m),
        "length_m": float(length_m),
        "ratio": float(ratio),
        "rule_used": rule_used,
        "random_target_m": float(random_target_m),
    }


# --------------------------
# 6) apply_linestring_extend / apply_linestring_shorten (scratch only)
# --------------------------
def apply_linestring_extend(state: GeoState) -> Dict[str, Any]:
    """
    Extend at selected endpoint by l2.change_m.
    Scratch-only: updates l2.work_coords but DOES NOT write to fc.
    """
    coords = state.scratch.get("l2.work_coords")
    ep = state.scratch.get("l2.endpoint")
    change_m = state.scratch.get("l2.change_m")
    planar_epsg = state.scratch.get("l2.planar_epsg")

    if not coords or ep not in ("head", "tail") or change_m is None or planar_epsg is None:
        raise ValueError("missing scratch: call get_linestring_stats, select_linestring_endpoint, compute_length_change_m first")

    coords = [(float(x), float(y)) for x, y in coords]
    if len(coords) < 2:
        return {"ok": False, "reason": "LINE_TOO_SHORT"}

    if ep == "tail":
        p_prev, p_end = coords[-2], coords[-1]
        bearing = _bearing_from_vec(p_end[0] - p_prev[0], p_end[1] - p_prev[1])  # not accurate in lonlat
        # better: compute bearing in planar using local UTM derived from planar_epsg and those points
        bearing = _bearing_from_vec(*_lonlat_to_planar_vec(p_prev, p_end, planar_epsg))
        new_end = _move_lonlat_planar(p_end, bearing, float(change_m), int(planar_epsg))
        coords.append(new_end)
    else:
        p_end, p_next = coords[0], coords[1]
        bearing = _bearing_from_vec(*_lonlat_to_planar_vec(p_next, p_end, planar_epsg))
        new_head = _move_lonlat_planar(p_end, bearing, float(change_m), int(planar_epsg))
        coords = [new_head] + coords

    state.scratch["l2.work_coords"] = coords
    state.scratch["l2.action"] = "extend"

    return {
        "action": "extend",
        "endpoint": ep,
        "change_m": float(change_m),
        "new_num_points": int(len(coords)),
    }


def apply_linestring_shorten(state: GeoState) -> Dict[str, Any]:
    """
    Shorten at selected endpoint by l2.change_m.
    Scratch-only: updates l2.work_coords but DOES NOT write to fc.
    """
    coords = state.scratch.get("l2.work_coords")
    ep = state.scratch.get("l2.endpoint")
    change_m = state.scratch.get("l2.change_m")
    planar_epsg = state.scratch.get("l2.planar_epsg")

    if not coords or ep not in ("head", "tail") or change_m is None or planar_epsg is None:
        raise ValueError("missing scratch: call get_linestring_stats, select_linestring_endpoint, compute_length_change_m first")

    coords = [(float(x), float(y)) for x, y in coords]
    if len(coords) < 2:
        return {"ok": False, "reason": "LINE_TOO_SHORT"}

    remaining = float(change_m)

    if ep == "tail":
        # remove from end backwards
        while len(coords) > 1 and remaining > 0:
            p_prev, p_end = coords[-2], coords[-1]
            seg_len = _segment_len_m_planar(p_prev, p_end, int(planar_epsg))
            if remaining >= seg_len:
                coords.pop()
                remaining -= seg_len
            else:
                # move endpoint towards p_prev by (seg_len - remaining)
                bearing_back = _bearing_from_vec(*_lonlat_to_planar_vec(p_end, p_prev, int(planar_epsg)))
                new_end = _move_lonlat_planar(p_prev, bearing_back, (seg_len - remaining), int(planar_epsg))
                coords[-1] = new_end
                remaining = 0.0
    else:
        # remove from head forwards
        while len(coords) > 1 and remaining > 0:
            p_head, p_next = coords[0], coords[1]
            seg_len = _segment_len_m_planar(p_head, p_next, int(planar_epsg))
            if remaining >= seg_len:
                coords.pop(0)
                remaining -= seg_len
            else:
                bearing_fwd = _bearing_from_vec(*_lonlat_to_planar_vec(p_head, p_next, int(planar_epsg)))
                new_head = _move_lonlat_planar(p_head, bearing_fwd, remaining, int(planar_epsg))
                coords[0] = new_head
                remaining = 0.0

    state.scratch["l2.work_coords"] = coords
    state.scratch["l2.action"] = "shorten"

    return {
        "action": "shorten",
        "endpoint": ep,
        "change_m": float(change_m),
        "remaining_unapplied_m": float(remaining),
        "new_num_points": int(len(coords)),
    }


# --------------------------
# 7) commit_linestring_geometry (write back fc)
# --------------------------
def commit_linestring_geometry(state: GeoState, line: str) -> Dict[str, Any]:
    """
    The ONLY step that mutates the final GeoJSON (state.fc).
    Writes l2.work_coords into the feature geometry.
    """
    l_id = _normalize_id(line)
    if l_id is None:
        raise ValueError("line handle is null/empty")

    coords = state.scratch.get("l2.work_coords")
    if not coords or state.scratch.get("l2.line_id") != l_id:
        raise ValueError("missing l2.work_coords in scratch for this line; call previous steps first")

    if len(coords) < 2:
        raise ValueError("cannot commit: LineString needs >= 2 points")

    ls = LineString(coords)
    state.update_geometry(l_id, ls)

    # optional: recompute stats after commit (still no extra fields)
    new_len, _epsg = _length_linestring_planar([(float(x), float(y)) for x, y in coords])

    return {
        "updated_handle": l_id,
        "action": state.scratch.get("l2.action"),
        "endpoint": state.scratch.get("l2.endpoint"),
        "change_m": state.scratch.get("l2.change_m"),
        "new_num_points": len(coords),
        "new_length_m": float(new_len),
    }


# ============================================================
# Internal planar helpers for accurate segment/bearing
# ============================================================
def _segment_len_m_planar(p1: Tuple[float, float], p2: Tuple[float, float], planar_epsg: int) -> float:
    wgs84 = CRS.from_epsg(4326)
    crs_planar = CRS.from_epsg(int(planar_epsg))
    tf = _to_planar_transformer(wgs84, crs_planar)
    x1, y1 = tf.transform(p1[0], p1[1])
    x2, y2 = tf.transform(p2[0], p2[1])
    return float(math.hypot(x2 - x1, y2 - y1))


def _lonlat_to_planar_vec(p_from: Tuple[float, float], p_to: Tuple[float, float], planar_epsg: int) -> Tuple[float, float]:
    """
    Return planar dx,dy from p_from to p_to in meters (using planar_epsg).
    """
    wgs84 = CRS.from_epsg(4326)
    crs_planar = CRS.from_epsg(int(planar_epsg))
    tf = _to_planar_transformer(wgs84, crs_planar)
    x1, y1 = tf.transform(p_from[0], p_from[1])
    x2, y2 = tf.transform(p_to[0], p_to[1])
    return float(x2 - x1), float(y2 - y1)