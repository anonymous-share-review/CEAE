import glob
import os
from typing import Dict, Any, Tuple
import requests
import osm2geojson
import os
import sys
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_collection.collection_tools import drop_overlaps, OVERPASS_MIRRORS, bbox_from_center, save_png_html, base_path, robust_overpass_request
import json

def apply_feature_filters(geojson_obj: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    过滤 OSM→GeoJSON 的 features，保留：
      - Point 类型（包括 node 类型）
      - Polygon 类型（不涉及绿地处理）
      - LineString 和 MultiLineString（过滤无关元素）
    丢弃：
      - Node、Relation（边界类等）
      - 明确不需要的标签，如 waterway、power、highway 等
      - 重叠的元素
    """
    EXCLUDED_TAGS = {"waterway", "power", "natural"}

    def skip_feature(feat: Dict[str, Any]) -> bool:
        props = feat.get("properties", {}) or {}
        tags = props.get("tags", {}) or {}
        geom = feat.get("geometry", {}) or {}
        gtype = geom.get("type", "")

        # ① 过滤无关的 Node 类型元素
        if gtype == "Point":
            # 如果是 node 类型且没有其他有意义的标签，丢弃
            if "node" in tags and not tags:
                return True

        # ② Relation 类型的元素丢弃，特别是边界类
        if "relation" in tags and ("boundary" in tags or "border_type" in tags):
            return True

        # ③ 无关的水系、电力、高速公路等丢弃
        if any(tag in tags for tag in EXCLUDED_TAGS):
            return True

        # ④ LineString / MultiLineString 过滤与交通相关的元素
        if gtype in ("LineString", "MultiLineString") and any(tag in tags for tag in EXCLUDED_TAGS):
            return True

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

def fetch_osm_bbox_geojson(lat: float, lon: float, side_km: float = 1.0, base_path: str = None):

    lat_str = f"{lat:.6f}".replace(".", "_")
    lon_str = f"{lon:.6f}".replace(".", "_")

    location = f'lat_{lat_str}_lon_{lon_str}'
    file_name = f'lat_{lat_str}_lon_{lon_str}'


    south, west, north, east = bbox_from_center(lat, lon, side_km)
    query = f"""
    [out:json][timeout:180];
    (
      node({south},{west},{north},{east});
      way({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    data = robust_overpass_request(query)  # 会自动换镜像 + 重试 + 退避

    # 转 GeoJSON
    try:
        shapes_with_props = osm2geojson.json2geojson(data)
        shapes_with_props, stats0 = apply_feature_filters(shapes_with_props)
        shapes_with_props, stats = drop_overlaps(shapes_with_props)


        if stats['kept'] > 100:
            print(f"[overlap] groups={stats['groups']}, dropped={stats['dropped']}, kept={stats['kept']}")
            location  = location + '_' + str(stats['kept'])
            out_dir = os.path.join(base_path, location)
            os.makedirs(out_dir, exist_ok=True)

            geojson_output_path = os.path.join(out_dir, f'{file_name}.geojson')

            with open(geojson_output_path, "w", encoding="utf-8") as f:
                json.dump(shapes_with_props, f, ensure_ascii=False, indent=2)

            save_png_html(shapes_with_props, lat, lon, file_name=file_name, out_dir=out_dir)

            return {
                "bbox": {...},
                "elements_count": len(data.get("elements", [])),
                "outfile": geojson_output_path,
                "saved": True
            }

        else:
            print(f"[skip] kept={stats['kept']} (<=100), skip saving for {location}")
            return {
                "bbox": {"south": south, "west": west, "north": north, "east": east},
                "elements_count": len(data.get("elements", [])),
                "outfile": None,
                "saved": False
            }
    except Exception as e:
        return {
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "elements_count": len(data.get("elements", [])),
            "outfile": None,
            "saved": False
        }

if __name__ == "__main__":

    side_km = 1
    saved_count = 0

    all_centers = []
    for city_dir in os.listdir(base_path):
        folder = os.path.join(base_path, city_dir)
        if not os.path.isdir(folder):
            continue
        json_files = glob.glob(os.path.join(folder, "*_bboxes.json"))
        if not json_files:
            continue  # 如果没找到，就跳过
        json_path = json_files[0]  # 取第一个匹配到的

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取 center 坐标
        for item in data:
            center = item["center"]

            info = fetch_osm_bbox_geojson(center["lat"], center["lon"], side_km, folder)
            if info["saved"]:
                saved_count += 1
                print(f"✅ Saved {saved_count} GeoJSON files")