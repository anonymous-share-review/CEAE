from typing import Dict, Any, List, Tuple
import geopandas as gpd
import json
import pandas as pd
from playwright.sync_api import sync_playwright
import tempfile
import math
from shapely.geometry import box
from pathlib import Path
import folium
import os
import requests, time, random
base_path = '../../data_1_2'


OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter"
]


def robust_overpass_request(query: str, mirrors=OVERPASS_MIRRORS,
                            max_retries=5, base_sleep=2.0, timeout=200):
    mirrors = list(mirrors)
    random.shuffle(mirrors)
    last_err = None

    for attempt in range(1, max_retries + 1):
        for url in mirrors:
            try:
                resp = requests.post(url, data=query.encode("utf-8"),timeout=timeout)
                if 200 <= resp.status_code < 300:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = Exception(f"{resp.status_code} {resp.reason} at {url}")
                else:
                    resp.raise_for_status()
            except requests.RequestException as e:
                last_err = e
        sleep_s = base_sleep * (2 ** (attempt - 1))
        time.sleep(sleep_s)

    raise RuntimeError(f"Overpass request failed after {max_retries} attempts: {last_err}")
def bbox_from_center(lat: float, lon: float, side_km: float = 1.0):
    """
    根据中心点与方框边长（km）计算 bbox: (south, west, north, east)
    近似计算，适用于小范围（~几公里）的区域。
    """
    half_km = side_km / 2.0
    # 1°纬度 ≈ 111.32 km
    dlat = half_km / 111.32
    # 1°经度 ≈ 111.32 * cos(lat) km
    lat_rad = math.radians(lat)
    dlon = half_km / (111.32 * math.cos(lat_rad))

    south = lat - dlat
    north = lat + dlat
    west  = lon - dlon
    east  = lon + dlon
    return south, west, north, east

def _utm_epsg_from_latlon(lat: float, lon: float) -> int:
    """根据经纬度推断对应的 UTM EPSG 代号（北半球 326xx / 南半球 327xx）"""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone

def drop_overlaps(
    geojson_obj: Dict[str, Any],
    overlap_threshold: float = 0.9,     # ← 重合率阈值（默认 50%）
    overlap_metric: str = "min",        # ← "min"=交/小者；"union"=IoU；"a"=交/面A；"b"=交/面B
) -> (Dict[str, Any], Dict[str, int]):
    """
    仅当两面之间的重合率 ≥ overlap_threshold 时，才把它们归到同一簇，
    每簇仅保留面积最大的一个；其它（未达到阈值的相交）双方都保留。
    返回: (去重后的_geojson, 统计信息)
    """

    valid_features = [f for f in geojson_obj.get("features", []) if f.get("geometry")]
    if not valid_features:
        return {"type": "FeatureCollection", "features": []}, {"groups": 0, "dropped": 0, "kept": 0}

    # 1) 构造成 GeoDataFrame（WGS84）
    gdf = gpd.GeoDataFrame.from_features(geojson_obj.get("features", []), crs="EPSG:4326")

    # 只针对面要素；其它几何（若存在）直接保留
    poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf_poly = gdf[poly_mask].copy()
    gdf_rest = gdf[~poly_mask].copy()

    if gdf_poly.empty:
        return geojson_obj, {"groups": 0, "dropped": 0, "kept": len(gdf)}

    # 2) 投影到当地 UTM，用米制计算面积/交并更可靠
    c_lon = float(gdf_poly.unary_union.centroid.x)
    c_lat = float(gdf_poly.unary_union.centroid.y)
    epsg = _utm_epsg_from_latlon(c_lat, c_lon)
    gdf_poly_m = gdf_poly.to_crs(epsg)

    # 3) 用空间索引做候选配对 → 计算“重合率” → 达阈值才连边（并查集成簇）
    sidx = gdf_poly_m.sindex
    n = len(gdf_poly_m)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    geoms = gdf_poly_m.geometry.reset_index(drop=True)

    for i, gi in enumerate(geoms):
        # 候选：先 bbox/intersects 粗筛
        cand = list(sidx.query(gi, predicate="intersects"))
        for j in cand:
            if j <= i:
                continue
            gj = geoms.iloc[j]

            if gi.equals(gj) or gi.contains(gj) or gj.contains(gi) or gi.covers(gj) or gj.covers(gi):
                union(i, j)
                continue

            inter = gi.intersection(gj).area
            if inter <= 0:
                continue

            # 选择重合率度量
            if overlap_metric == "union":
                denom = gi.area + gj.area - inter  # = union.area
            elif overlap_metric == "a":
                denom = gi.area
            elif overlap_metric == "b":
                denom = gj.area
            else:  # "min"（默认：交/小者）
                denom = min(gi.area, gj.area)

            if denom <= 0:
                continue

            ratio = inter / denom
            if ratio >= overlap_threshold:
                union(i, j)

    # 4) 连通分量分簇；仅当簇大小>1时才执行“只留最大”
    from collections import defaultdict
    groups = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)

    keep_local_idx: List[int] = []
    dropped = 0
    for comp in groups.values():
        if len(comp) == 1:
            keep_local_idx.extend(comp)
        else:
            areas = geoms.iloc[comp].area
            max_pos = areas.argmax()
            keep_local_idx.append(comp[max_pos])
            dropped += (len(comp) - 1)

    # 5) 还原到原始 CRS 的行索引并合回非面要素
    kept_poly = gdf_poly.iloc[keep_local_idx]
    gdf_out = pd.concat([kept_poly, gdf_rest], ignore_index=True)

    # 6) 输出 GeoJSON
    out_geojson = {
        "type": "FeatureCollection",
        "features": json.loads(gdf_out.to_json())["features"]
    }

    # 仅统计“真正发生合并的簇”数量
    merged_groups = sum(1 for comp in groups.values() if len(comp) > 1)
    return out_geojson, {"groups": merged_groups, "dropped": dropped, "kept": len(gdf_out)}


def save_png_html(geojson_obj, lat=38.9590, lon=-95.3256, zoom=20, file_name=None, out_dir=None):
    # —— 1) 生成 HTML：高清瓦片 + 细线 + 红色 bbox + 中心点 + fit 到 1.5km 正方形
    m = folium.Map(location=[lat, lon], zoom_start=zoom, width=1600, height=1600)
    folium.TileLayer("OpenStreetMap", detect_retina=True).add_to(m)

    def thin_style(_):
        return {"color": "#2563EB", "weight": 1, "opacity": 0.9, "fillOpacity": 0.2}

    folium.GeoJson(geojson_obj, name="preview", style_function=thin_style).add_to(m)


    # 1.5 km × 1.5 km 的 bbox（严格正方形）
    south, west, north, east = bbox_from_center(lat, lon, side_km=1.0)

    # 红框 + 中心点
    folium.Rectangle(bounds=[(south, west), (north, east)],
                     color="#FF2D2D", weight=2, fill=False).add_to(m)
    folium.CircleMarker([lat, lon], radius=4, color="#FF2D2D",
                        fill=True, fill_opacity=1).add_to(m)

    # 让 folium 页面默认就贴合到 1.5km bbox
    m.fit_bounds([[south, west], [north, east]])

    html_out_path = os.path.join(out_dir, f'{file_name}.html')
    m.save(html_out_path)
    print(f"[OK] HTML file has been saved: {html_out_path}")

    # —— 2) 截 PNG：正方形容器、零 padding、invalidateSize 后再 fitBounds
    out_png = os.path.join(out_dir, f'{file_name}.png')
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1600, "height": 1600, "deviceScaleFactor": 2}  # 高分辨率
        )
        page.goto(Path(html_out_path).resolve().as_uri(), wait_until="load")

        # 让地图容器严格正方形并去掉页面留白
        page.evaluate("""
            (() => {
              const el = document.querySelector('.folium-map');
              el.style.margin='0'; el.style.padding='0';
              el.style.width='1600px'; el.style.height='1600px';
              document.body.style.margin='0';
              document.documentElement.style.margin='0';
            })();
            """)

        # 让 Leaflet 重新计算尺寸
        page.evaluate("""
            (() => {
              const key = Object.keys(window).find(x => /^map_/.test(x) && window[x]?.invalidateSize);
              if (key) window[key].invalidateSize(true);
            })();
            """)

        # 严格贴合 1.5km bbox（零 padding），并画红框/中心点以防止样式被覆盖
        page.evaluate(f"""
            (() => {{
              const key = Object.keys(window).find(x => /^map_/.test(x) && window[x]?.fitBounds);
              const map = window[key];
              map.fitBounds([[{south},{west}], [{north},{east}]], {{padding:[0,0]}});
              L.rectangle([[{south},{west}], [{north},{east}]], {{color:'#FF2D2D', weight:2, fill:false}}).addTo(map);
              L.circleMarker([{lat},{lon}], {{radius:4, color:'#FF2D2D', fill:true, fillOpacity:1}}).addTo(map);
            }})();
            """)

        # 等至少有一块瓦片加载，避免空白
        try:
            page.wait_for_selector(".leaflet-tile-loaded", timeout=5000)
        except:
            pass

        # 只截地图元素（不含外部留白）
        page.locator(".folium-map").screenshot(path=out_png)
        browser.close()

    print(f"[OK] PNG file has been saved: {out_png}")
    return out_png


def apply_feature_filters(geojson_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    过滤 OSM→GeoJSON 的 features，保留：
      - Polygon / MultiPolygon
      - building=*
      - 绿地/公园类：leisure in {park, garden, recreation_ground, pitch, playground},
                    natural in {wood, forest, grassland, scrub, heath, wetland, water},
                    landuse in {grass, meadow, forest, recreation_ground, cemetery}
    丢弃：
      - LineString / MultiLineString
      - node
      - 边界/国界类 relation
      - 桥梁、route、power、highway、waterway、area:highway
      - 地铁/车站/地下设施（railway, public_transport, building=train_station/station/transportation, location=underground, layer<0）
    同时移除 properties.nodes
    """

    # —— 可配置白名单（绿地/公园）—— #
    LEISURE_KEEP = {"park", "garden", "recreation_ground", "pitch", "playground"}
    NATURAL_KEEP = {"wood", "forest", "grassland", "scrub", "heath", "wetland", "water"}
    LANDUSE_KEEP = {"grass", "meadow", "forest", "recreation_ground", "cemetery"}

    def is_green(tags: dict) -> bool:
        if tags.get("leisure") in LEISURE_KEEP:
            return True
        if tags.get("natural") in NATURAL_KEEP:
            return True
        if tags.get("landuse") in LANDUSE_KEEP:
            return True
        return False

    def skip_feature(feat: Dict[str, Any]) -> bool:
        props = feat.get("properties", {}) or {}
        tags  = props.get("tags", {}) or {}
        geom  = feat.get("geometry", {}) or {}
        gtype = geom.get("type", "")

        # ① 线几何一律丢
        if gtype in ("LineString", "MultiLineString"):
            return True

        # ② OSM 类型过滤
        ptype = (props.get("type") or "").lower()
        if ptype == "node":
            return True

        if ptype == "relation":
            # 明确不要的 relation：边界/国界等
            if "border_type" in tags or "boundary" in tags:
                return True
            # multipolygon：只保留 building 或 绿地白名单（其余丢）
            if tags.get("type") == "multipolygon":
                # if not (("building" in tags) or is_green(tags)):
                return True
        if tags.get("water") in ("river", "stream", "canal", "lake", "pond", "reservoir"):
            return True
        # ③ 明确黑名单的“语义类”标签
        if tags.get("man_made") == "bridge":
            return True
        if tags.get("historic") == "neighbourhood" or tags.get("place") == "neighbourhood":
            return True
        if tags.get("area") == "yes" and len(tags) == 1:
            return True
        if tags.get("boundary") == "administrative":
            return True

        if not tags or len(tags) == 0:
            return True

        # ④ 线路/电力/道路/水系/面积化道路等（注意：natural=water 会通过 is_green 保留为面；这里丢的是 waterway=河道线）
        if "power" in tags:
            return True
        if "route" in tags:
            return True
        if "highway" in tags:
            return True
        if "waterway" in tags:
            return True
        if "area:highway" in tags:
            return True
        if "barrier" in tags:
            return True
        if "bridge:support" in tags:
            return True

        # ⑤ 交通/地下设施：直接丢
        bval = (tags.get("building") or "").lower()
        if bval in {"train_station", "station", "transportation"}:
            return True
        if "railway" in tags or "public_transport" in tags:
            return True
        if (tags.get("location") or "").lower() == "underground":
            return True
        layer_val = tags.get("layer")
        if layer_val is not None:
            try:
                if int(layer_val) < 0:
                    return True
            except ValueError:
                pass

        # ⑥ landuse：以前是全丢；现在通过 is_green() 选择性保留
        #   不在 LANDUSE_KEEP 的 landuse 不作强制丢弃（交由其他规则决定）
        #   如需更严格，可在此丢掉其它 landuse：
        #   if "landuse" in tags and tags["landuse"] not in LANDUSE_KEEP: return True

        return False

    features_in = geojson_obj.get("features", []) or []
    filtered_features = []
    dropped = 0

    for feat in features_in:
        if skip_feature(feat):
            dropped += 1
            continue

        # 移除 properties.nodes（若存在）
        props = feat.get("properties", {})
        if isinstance(props, dict) and "nodes" in props:
            props = dict(props)
            props.pop("nodes", None)
            feat = dict(feat)
            feat["properties"] = props

        filtered_features.append(feat)

    out = {"type": "FeatureCollection", "features": filtered_features}
    stats = {"in": len(features_in), "kept": len(filtered_features), "dropped": dropped}
    return out, stats

def generate_grid_bboxes(lat: float, lon: float, side_km: float = 1.0, grid_size: int = 5):
    """
    以 (lat, lon) 为中心，生成 grid_size x grid_size 的 bbox 网格。
    每个 bbox 大小 side_km x side_km (km)。
    返回: list of dicts
    """
    # 单格偏移（中心到中心的距离 = side_km）
    dlat = side_km / 111.32
    dlon = side_km / (111.32 * math.cos(math.radians(lat)))

    bboxes = []
    half = grid_size // 2

    for i in range(-half, half + 1):   # 行 (lat 方向)
        for j in range(-half, half + 1):  # 列 (lon 方向)
            c_lat = lat + i * dlat
            c_lon = lon + j * dlon
            south, west, north, east = bbox_from_center(c_lat, c_lon, side_km)
            bboxes.append({
                "center": (c_lat, c_lon),
                "bbox": (south, west, north, east)
            })
    return bboxes