# level1_tools.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import shape, mapping, Point, LineString
from shapely.validation import explain_validity

from pyproj import CRS, Transformer


# -------------------------
# Helpers: ID / tags / CRS / bearing
# -------------------------
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
    Get canonical feature handle from a GeoJSON Feature.
    We ONLY trust feat["properties"]["id"] as the ID.
    """
    props = _get_properties(feat)
    return _normalize_id(props.get("id", None))


def _get_tags(feat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your data stores OSM tags under properties["tags"].
    """
    props = _get_properties(feat)
    tags = props.get("tags")
    return tags if isinstance(tags, dict) else {}


def _utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _to_planar_transformer(crs_from: CRS, crs_to: CRS) -> Transformer:
    return Transformer.from_crs(crs_from, crs_to, always_xy=True)


def _bearing_from_vec(dx: float, dy: float) -> float:
    ang = math.degrees(math.atan2(dx, dy))  # atan2(x, y) gives 0 at north
    return (ang + 360.0) % 360.0


def _vec_from_bearing(bearing_deg: float) -> Tuple[float, float]:
    rad = math.radians(bearing_deg)
    return math.sin(rad), math.cos(rad)  # dx, dy


def _direction_to_bearing(direction: str) -> Optional[float]:
    d = str(direction).upper().strip()
    mapping_dir = {
        "NORTH": 0.0,
        "NORTHEAST": 45.0,
        "EAST": 90.0,
        "SOUTHEAST": 135.0,
        "SOUTH": 180.0,
        "SOUTHWEST": 225.0,
        "WEST": 270.0,
        "NORTHWEST": 315.0,
    }
    return mapping_dir.get(d)


def _unit(vx: float, vy: float) -> Tuple[float, float]:
    n = math.hypot(vx, vy)
    if n == 0:
        return 0.0, 0.0
    return vx / n, vy / n


def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def _cross_z(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _angle_deg_between(ax: float, ay: float, bx: float, by: float) -> float:
    au = _unit(ax, ay)
    bu = _unit(bx, by)
    denom = max(1e-12, math.hypot(*au) * math.hypot(*bu))
    c = max(-1.0, min(1.0, _dot(au[0], au[1], bu[0], bu[1]) / denom))
    return math.degrees(math.acos(c))


# -------------------------
# State container
# -------------------------
@dataclass
class GeoState:
    """
    Minimal state wrapper for a GeoJSON FeatureCollection.

    IMPORTANT:
    - feature_handle is ALWAYS str(feature.properties.id)
    - We build index ONLY from feature.properties.id
    """
    fc: Dict[str, Any]

    def __post_init__(self):
        assert self.fc.get("type") == "FeatureCollection", "State must be a GeoJSON FeatureCollection"
        self._index: Dict[str, Dict[str, Any]] = {}

        for feat in self.fc.get("features", []):
            pid = _get_pid(feat)
            if pid is None:
                continue
            # if duplicates exist, last one wins (or raise if you prefer)
            self._index[pid] = feat

    def get_feature(self, feature_id: Any) -> Optional[Dict[str, Any]]:
        hid = _normalize_id(feature_id)
        if hid is None:
            return None
        return self._index.get(hid)

    def update_geometry(self, feature_id: Any, new_geom) -> None:
        hid = _normalize_id(feature_id)
        if hid is None:
            raise KeyError("Feature id is null/empty")
        feat = self.get_feature(hid)
        if feat is None:
            raise KeyError(f"Feature not found by properties.id: {hid}")
        feat["geometry"] = mapping(new_geom)

    def get_shapely(self, feature_id: Any):
        hid = _normalize_id(feature_id)
        if hid is None:
            raise KeyError("Feature id is null/empty")
        feat = self.get_feature(hid)
        if feat is None:
            raise KeyError(f"Feature not found by properties.id: {hid}")
        return shape(feat["geometry"])


# -------------------------
# Tools: 定位/读取（全部按 properties.id）
# -------------------------
def get_feature_by_id(state: GeoState, id: str) -> Dict[str, Any]:
    hid = _normalize_id(id)
    if hid is None:
        raise KeyError("Feature id is null/empty")
    feat = state.get_feature(hid)
    if feat is None:
        raise KeyError(f"Feature not found by properties.id: {hid}")
    return {"feature_handle": hid}


def get_feature_geometry(state: GeoState, feature: str) -> Dict[str, Any]:
    hid = _normalize_id(feature)
    if hid is None:
        raise KeyError("Feature handle is null/empty")
    geom = state.get_shapely(hid)
    return {"geometry": mapping(geom)}


def get_feature_properties(state: GeoState, feature: str) -> Dict[str, Any]:
    hid = _normalize_id(feature)
    if hid is None:
        raise KeyError("Feature handle is null/empty")
    feat = state.get_feature(hid)
    if feat is None:
        raise KeyError(f"Feature not found by properties.id: {hid}")
    return {"properties": _get_properties(feat)}


# -------------------------
# Tools: 道路参考（nearest + project + tangent）
# -------------------------
def find_nearest_feature(
    state: GeoState,
    from_feature: str,
    filter: Optional[Dict[str, Any]] = None,
    max_dist_m: float = 200.0,
) -> Dict[str, Any]:
    """
    filter example:
      {
        "geom_type": "LineString",
        "tags": {"highway": "*"}   # wildcard, matched against properties["tags"]
      }
    """
    from_id = _normalize_id(from_feature)
    if from_id is None:
        raise ValueError("from_feature is null/empty")

    from_geom = state.get_shapely(from_id)
    if not isinstance(from_geom, Point):
        raise ValueError("from_feature must be a Point for L1 tasks")

    lon, lat = from_geom.x, from_geom.y
    wgs84 = CRS.from_epsg(4326)
    crs_planar = _utm_crs_for_lonlat(lon, lat)
    tf = _to_planar_transformer(wgs84, crs_planar)

    px, py = tf.transform(lon, lat)
    p_planar = Point(px, py)

    want_geom_type = (filter or {}).get("geom_type")
    want_tags = (filter or {}).get("tags", {}) if isinstance((filter or {}).get("tags", {}), dict) else {}

    best_id = None
    best_dist = float("inf")

    for feat in state.fc.get("features", []):
        fid = _get_pid(feat)  # <<< properties.id
        if fid is None:
            continue
        if fid == from_id:
            continue

        geom = shape(feat["geometry"])
        if want_geom_type and geom.geom_type != want_geom_type:
            continue

        if want_tags:
            tags = _get_tags(feat)  # <<< properties.tags
            ok = True
            for k, v in want_tags.items():
                if k not in tags:
                    ok = False
                    break
                if v != "*" and tags.get(k) != v:
                    ok = False
                    break
            if not ok:
                continue

        if geom.geom_type != "LineString":
            continue

        coords = list(geom.coords)
        coords_p = [tf.transform(x, y) for x, y in coords]
        line_p = LineString(coords_p)

        dist = p_planar.distance(line_p)
        if dist < best_dist:
            best_dist = dist
            best_id = fid  # already normalized string

    if best_id is None or best_dist > float(max_dist_m):
        return {"nearest_handle": None, "distance_m": None}

    return {"nearest_handle": best_id, "distance_m": float(best_dist), "planar_epsg": crs_planar.to_epsg()}


def project_point_to_line(state: GeoState, point: str, line: str) -> Dict[str, Any]:
    p_id = _normalize_id(point)
    l_id = _normalize_id(line)
    if p_id is None or l_id is None:
        raise ValueError("point/line handle is null/empty")

    p = state.get_shapely(p_id)
    l = state.get_shapely(l_id)
    if not isinstance(p, Point) or l.geom_type != "LineString":
        raise ValueError("Expect point=Point and line=LineString")

    lon, lat = p.x, p.y
    wgs84 = CRS.from_epsg(4326)
    crs_planar = _utm_crs_for_lonlat(lon, lat)
    tf = _to_planar_transformer(wgs84, crs_planar)

    px, py = tf.transform(lon, lat)
    p_planar = Point(px, py)

    coords_p = [tf.transform(x, y) for x, y in list(l.coords)]
    line_p = LineString(coords_p)

    s = float(line_p.project(p_planar))
    proj = line_p.interpolate(s)
    dist = float(p_planar.distance(proj))
    line_len = float(line_p.length)

    t_norm = 0.0 if line_len == 0 else s / line_len

    return {
        "planar_epsg": crs_planar.to_epsg(),
        "s_along_m": s,
        "t_norm": t_norm,
        "proj_point_xy": [float(proj.x), float(proj.y)],
        "distance_m": dist,
        "line_length_m": line_len,
    }


def get_line_tangent_at(
    state: GeoState,
    line: str,
    t_norm: float,
    planar_epsg: Optional[int] = None,
) -> Dict[str, Any]:
    l_id = _normalize_id(line)
    if l_id is None:
        raise ValueError("line handle is null/empty")

    l = state.get_shapely(l_id)
    if l.geom_type != "LineString":
        raise ValueError("line must be a LineString")

    coords = list(l.coords)
    if len(coords) < 2:
        return {"unit_tangent": [0.0, 0.0], "bearing_deg": None}

    wgs84 = CRS.from_epsg(4326)
    if planar_epsg is None:
        lon0, lat0 = coords[0]
        crs_planar = _utm_crs_for_lonlat(lon0, lat0)
    else:
        crs_planar = CRS.from_epsg(planar_epsg)

    tf = _to_planar_transformer(wgs84, crs_planar)
    coords_p = [tf.transform(x, y) for x, y in coords]
    line_p = LineString(coords_p)

    s = max(0.0, min(1.0, float(t_norm))) * float(line_p.length)
    eps = 1.0
    s1 = max(0.0, s - eps)
    s2 = min(float(line_p.length), s + eps)
    p1 = line_p.interpolate(s1)
    p2 = line_p.interpolate(s2)

    dx = float(p2.x - p1.x)
    dy = float(p2.y - p1.y)
    ux, uy = _unit(dx, dy)
    bearing = _bearing_from_vec(ux, uy) if (ux, uy) != (0.0, 0.0) else None

    return {"planar_epsg": crs_planar.to_epsg(), "unit_tangent": [ux, uy], "bearing_deg": bearing}


# -------------------------
# Tools: 移动
# -------------------------
def move_point_by_distance_direction(state: GeoState, point: str, direction: str, distance_m: float) -> Dict[str, Any]:
    p_id = _normalize_id(point)
    if p_id is None:
        raise ValueError("point handle is null/empty")
    bearing = _direction_to_bearing(direction)
    if bearing is None:
        raise ValueError(f"Unsupported direction: {direction}")
    return move_point_along_bearing(state, point=p_id, bearing_deg=bearing, distance_m=distance_m)


def move_point_along_bearing(state: GeoState, point: str, bearing_deg: float, distance_m: float) -> Dict[str, Any]:
    p_id = _normalize_id(point)
    if p_id is None:
        raise ValueError("point handle is null/empty")

    p = state.get_shapely(p_id)
    if not isinstance(p, Point):
        raise ValueError("point must be a Point")

    lon, lat = p.x, p.y
    wgs84 = CRS.from_epsg(4326)
    crs_planar = _utm_crs_for_lonlat(lon, lat)
    tf_fwd = _to_planar_transformer(wgs84, crs_planar)
    tf_inv = _to_planar_transformer(crs_planar, wgs84)

    x, y = tf_fwd.transform(lon, lat)
    ux, uy = _vec_from_bearing(float(bearing_deg))
    nx = x + float(distance_m) * ux
    ny = y + float(distance_m) * uy
    nlon, nlat = tf_inv.transform(nx, ny)

    state.update_geometry(p_id, Point(nlon, nlat))
    return {"updated_handle": p_id, "new_lonlat": [float(nlon), float(nlat)], "planar_epsg": crs_planar.to_epsg()}


def move_point_in_local_frame(
    state: GeoState,
    point: str,
    ref_line: str,
    axis: str,  # "TANGENT" or "NORMAL"
    signed_distance_m: float,
) -> Dict[str, Any]:
    p_id = _normalize_id(point)
    l_id = _normalize_id(ref_line)
    if p_id is None or l_id is None:
        raise ValueError("point/ref_line handle is null/empty")

    proj = project_point_to_line(state, point=p_id, line=l_id)
    tan = get_line_tangent_at(state, line=l_id, t_norm=proj["t_norm"], planar_epsg=proj["planar_epsg"])
    ux, uy = tan["unit_tangent"]
    if (ux, uy) == (0.0, 0.0):
        raise ValueError("Cannot compute tangent direction")

    ax = axis.upper().strip()
    if ax == "TANGENT":
        mvx, mvy = ux, uy
    elif ax == "NORMAL":
        mvx, mvy = -uy, ux
    else:
        raise ValueError("axis must be TANGENT or NORMAL")

    p = state.get_shapely(p_id)
    lon, lat = p.x, p.y
    wgs84 = CRS.from_epsg(4326)
    crs_planar = CRS.from_epsg(proj["planar_epsg"])
    tf_fwd = _to_planar_transformer(wgs84, crs_planar)
    tf_inv = _to_planar_transformer(crs_planar, wgs84)
    x, y = tf_fwd.transform(lon, lat)

    nx = x + float(signed_distance_m) * mvx
    ny = y + float(signed_distance_m) * mvy
    nlon, nlat = tf_inv.transform(nx, ny)

    state.update_geometry(p_id, Point(nlon, nlat))
    return {"updated_handle": p_id, "axis": ax, "planar_epsg": crs_planar.to_epsg(), "new_lonlat": [float(nlon), float(nlat)]}


# -------------------------
# Tools: 约束/修复
# -------------------------
def compute_current_offset_to_line(state: GeoState, point: str, line: str) -> Dict[str, Any]:
    p_id = _normalize_id(point)
    l_id = _normalize_id(line)
    if p_id is None or l_id is None:
        raise ValueError("point/line handle is null/empty")

    p = state.get_shapely(p_id)
    l = state.get_shapely(l_id)
    if not isinstance(p, Point) or l.geom_type != "LineString":
        raise ValueError("Expect point=Point and line=LineString")

    proj = project_point_to_line(state, point=p_id, line=l_id)
    tan = get_line_tangent_at(state, line=l_id, t_norm=proj["t_norm"], planar_epsg=proj["planar_epsg"])
    ux, uy = tan["unit_tangent"]

    lon, lat = p.x, p.y
    wgs84 = CRS.from_epsg(4326)
    crs_planar = CRS.from_epsg(proj["planar_epsg"])
    tf = _to_planar_transformer(wgs84, crs_planar)
    px, py = tf.transform(lon, lat)

    qx, qy = proj["proj_point_xy"]
    vx, vy = px - qx, py - qy

    cz = _cross_z(ux, uy, vx, vy)
    side = "LEFT" if cz > 0 else "RIGHT"
    offset = math.hypot(vx, vy)

    return {"planar_epsg": crs_planar.to_epsg(), "offset_m": float(offset), "side": side, "t_norm": proj["t_norm"]}


def snap_point_to_nearest_line_offset(
    state: GeoState,
    point: str,
    line: str,
    offset_m: float,
    side: Optional[str] = None,
) -> Dict[str, Any]:
    p_id = _normalize_id(point)
    l_id = _normalize_id(line)
    if p_id is None or l_id is None:
        raise ValueError("point/line handle is null/empty")

    cur = compute_current_offset_to_line(state, point=p_id, line=l_id)
    use_side = side.upper() if side else cur["side"]

    proj = project_point_to_line(state, point=p_id, line=l_id)
    tan = get_line_tangent_at(state, line=l_id, t_norm=proj["t_norm"], planar_epsg=proj["planar_epsg"])
    ux, uy = tan["unit_tangent"]

    if use_side == "LEFT":
        nx, ny = -uy, ux
    else:
        nx, ny = uy, -ux

    qx, qy = proj["proj_point_xy"]
    tx = qx + float(offset_m) * nx
    ty = qy + float(offset_m) * ny

    wgs84 = CRS.from_epsg(4326)
    crs_planar = CRS.from_epsg(proj["planar_epsg"])
    tf_inv = _to_planar_transformer(crs_planar, wgs84)
    nlon, nlat = tf_inv.transform(tx, ty)
    state.update_geometry(p_id, Point(nlon, nlat))

    return {"updated_handle": p_id, "new_lonlat": [float(nlon), float(nlat)], "offset_m": float(offset_m), "side": use_side}


def apply_move_with_constraints(state: GeoState, point: str, move: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    p_id = _normalize_id(point)
    if p_id is None:
        raise ValueError("point handle is null/empty")

    road = _normalize_id(constraints.get("road_handle"))
    if road is None and constraints.get("parallel_to_road", False):
        res = find_nearest_feature(
            state,
            from_feature=p_id,
            filter={"geom_type": "LineString", "tags": {"highway": "*"}},
            max_dist_m=float(constraints.get("max_road_dist_m", 100.0)),
        )
        road = res.get("nearest_handle")

    if constraints.get("parallel_to_road", False):
        if road is None:
            raise ValueError("parallel_to_road requested but no road found/provided")

        dist = float(move.get("distance_m", 0.0))
        direction = move.get("direction")
        bear_pref = _direction_to_bearing(direction) if direction else None

        proj = project_point_to_line(state, point=p_id, line=road)
        tan = get_line_tangent_at(state, line=road, t_norm=proj["t_norm"], planar_epsg=proj["planar_epsg"])
        ux, uy = tan["unit_tangent"]
        if (ux, uy) == (0.0, 0.0):
            raise ValueError("Cannot compute road tangent")

        if bear_pref is not None:
            b1 = _bearing_from_vec(ux, uy)
            b2 = _bearing_from_vec(-ux, -uy)

            def angdiff(a, b):
                d = abs(a - b) % 360.0
                return min(d, 360.0 - d)

            use_sign = +1.0 if angdiff(b1, bear_pref) <= angdiff(b2, bear_pref) else -1.0
        else:
            use_sign = +1.0

        return move_point_in_local_frame(
            state,
            point=p_id,
            ref_line=road,
            axis="TANGENT",
            signed_distance_m=use_sign * dist,
        )

    # fallback: unconstrained move
    if "bearing_deg" in move:
        return move_point_along_bearing(
            state,
            point=p_id,
            bearing_deg=float(move["bearing_deg"]),
            distance_m=float(move.get("distance_m", 0.0)),
        )
    if "direction" in move:
        return move_point_by_distance_direction(
            state,
            point=p_id,
            direction=str(move["direction"]),
            distance_m=float(move.get("distance_m", 0.0)),
        )

    raise ValueError("Unsupported move spec")


# -------------------------
# Tools: 校验
# -------------------------
def check_geometry_validity(state: GeoState, feature: str) -> Dict[str, Any]:
    hid = _normalize_id(feature)
    if hid is None:
        raise KeyError("Feature handle is null/empty")
    geom = state.get_shapely(hid)
    ok = bool(geom.is_valid)
    return {"valid": ok, "reason": None if ok else explain_validity(geom)}


def check_parallelism_point_to_road(
    state: GeoState,
    point_before_lonlat: List[float],
    point_after_lonlat: List[float],
    road: str,
    angle_threshold_deg: float = 15.0,
) -> Dict[str, Any]:
    road_id = _normalize_id(road)
    if road_id is None:
        return {"pass": False, "angle_deg": None, "reason": "ROAD_ID_NULL"}

    pb = Point(point_before_lonlat[0], point_before_lonlat[1])
    pa = Point(point_after_lonlat[0], point_after_lonlat[1])

    if pa.x == pb.x and pa.y == pb.y:
        return {"pass": False, "angle_deg": None, "reason": "ZERO_DISPLACEMENT"}

    lon, lat = pb.x, pb.y
    wgs84 = CRS.from_epsg(4326)
    crs_planar = _utm_crs_for_lonlat(lon, lat)
    tf = _to_planar_transformer(wgs84, crs_planar)

    pbx, pby = tf.transform(pb.x, pb.y)
    pax, pay = tf.transform(pa.x, pa.y)
    mvx, mvy = pax - pbx, pay - pby

    tmp = _tmp_point_handle_from_lonlat(state, pb.x, pb.y)
    proj = project_point_to_line(state, point=tmp, line=road_id)
    tan = get_line_tangent_at(state, line=road_id, t_norm=proj["t_norm"], planar_epsg=proj["planar_epsg"])
    ux, uy = tan["unit_tangent"]
    if (ux, uy) == (0.0, 0.0):
        return {"pass": False, "angle_deg": None, "reason": "NO_TANGENT"}

    ang1 = _angle_deg_between(mvx, mvy, ux, uy)
    ang2 = _angle_deg_between(mvx, mvy, -ux, -uy)
    ang = min(ang1, ang2)

    return {
        "pass": bool(ang <= float(angle_threshold_deg)),
        "angle_deg": float(ang),
        "threshold_deg": float(angle_threshold_deg),
        "planar_epsg": crs_planar.to_epsg(),
    }


def _tmp_point_handle_from_lonlat(state: GeoState, lon: float, lat: float) -> str:
    """
    Internal helper: temporary point feature handle used ONLY for computations.
    Still follows the SAME rule: handle == str(feature.properties.id).
    """
    tmp_id = "__tmp_point__"  # string is fine; index key is always str(properties.id)

    feat = {
        "type": "Feature",
        "properties": {"id": tmp_id, "_tmp": True},
        "geometry": mapping(Point(lon, lat)),
    }

    # Insert/update in index only (do not pollute fc["features"])
    state._index[tmp_id] = feat
    return tmp_id