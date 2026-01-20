import os
import sys
import json
import random
import math
import argparse
import copy
from typing import List, Dict, Any, Tuple
from pathlib import Path

# --- 路径设置 ---
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from tools.utils import iter_geojson_paths

# --- 常量定义 ---
R_EARTH = 6371000

BIKE_ROAD_TAGS = [
    'highway=cycleway', 'highway=bicycle', 'bicycle=designated',
    'route=bicycle', 'segregated=yes'
]

FACILITY_TAGS = {
    'bus_stop': ['highway=bus_stop', 'public_transport=platform', 'amenity=bus_station'],
    'bench': ['amenity=bench', 'leisure=park_bench', 'amenity=seat'],
    'toilets': ['amenity=toilets', 'amenity=waste_basket', 'amenity=public_toilet'],
    'waste_basket': ['amenity=waste_basket', 'amenity=trash_can', 'amenity=bin']
}


# --- 数学与核心计算函数 ---

def get_distance_m(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点(lon, lat)之间的米级距离"""
    lon1, lat1 = p1
    lon2, lat2 = p2
    x = (lon2 - lon1) * (math.pi / 180) * R_EARTH * math.cos(math.radians((lat1 + lat2) / 2))
    y = (lat2 - lat1) * (math.pi / 180) * R_EARTH
    return math.hypot(x, y)


def get_bearing(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算从 p1 指向 p2 的方位角 (0-360)"""
    lon1, lat1 = math.radians(p1[0]), math.radians(p1[1])
    lon2, lat2 = math.radians(p2[0]), math.radians(p2[1])

    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def bearing_to_cardinal(degrees):
    degrees = degrees % 360
    directions = ["North", "Northeast", "East", "Southeast",
                  "South", "Southwest", "West", "Northwest"]
    index = round(degrees / 45) % 8
    return directions[index]


def get_linestring_length(coords: List[List[float]]) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        total += get_distance_m(coords[i], coords[i + 1])
    return total


def calculate_distance_and_bearing_to_line(point: Tuple[float, float], line_coords: List[Tuple[float, float]]) -> Tuple[
    float, float]:
    """计算点到折线的最短距离及Bearing"""

    def get_xy_from_latlon(lon, lat, origin_lon, origin_lat):
        x = (lon - origin_lon) * (math.pi / 180) * R_EARTH * math.cos(origin_lat * math.pi / 180)
        y = (lat - origin_lat) * (math.pi / 180) * R_EARTH
        return x, y

    def point_to_segment_dist(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        return math.hypot(px - nearest_x, py - nearest_y)

    p_lon, p_lat = point
    min_dist = float('inf')
    best_road_dx = 0.0
    best_road_dy = 0.0

    for i in range(len(line_coords) - 1):
        lon1, lat1 = line_coords[i]
        lon2, lat2 = line_coords[i + 1]
        px, py = get_xy_from_latlon(p_lon, p_lat, lon1, lat1)
        x2, y2 = get_xy_from_latlon(lon2, lat2, lon1, lat1)
        dist = point_to_segment_dist(px, py, 0, 0, x2, y2)
        if dist < min_dist:
            min_dist = dist
            best_road_dx = x2
            best_road_dy = y2

    math_angle = math.atan2(best_road_dy, best_road_dx)
    road_bearing = (90 - math.degrees(math_angle)) % 360
    return min_dist, road_bearing


def move_point_planar(start_coords: Tuple[float, float], distance_m: float, bearing_deg: float) -> List[float]:
    """移动点"""
    lon1, lat1 = start_coords
    theta_rad = math.radians(90 - bearing_deg)
    dx = distance_m * math.cos(theta_rad)
    dy = distance_m * math.sin(theta_rad)
    d_lat = dy / (math.pi / 180 * R_EARTH)
    d_lon = dx / (math.pi / 180 * R_EARTH * math.cos(math.radians(lat1)))
    return [lon1 + d_lon, lat1 + d_lat]


def check_for_facilities(tags: Dict[str, str]) -> str:
    if not tags: return None
    for facility, tag_list in FACILITY_TAGS.items():
        for item in tag_list:
            k, v = item.split("=")
            if tags.get(k) == v:
                return facility
    return None


def is_bike_road(tags: Dict[str, str]) -> bool:
    if not tags: return False
    for tag_item in BIKE_ROAD_TAGS:
        k, v = tag_item.split("=")
        if tags.get(k) == v:
            return True
    return False


def find_nearest_road(point_coords: tuple, point_type: str, geojson_data: dict) -> Dict[str, Any]:
    min_distance = float('inf')
    nearest_road = None

    preferred_config = {
        'bus_stop': (['highway=primary', 'highway=secondary', 'highway=tertiary', 'highway=residential'], True),
        'bench': (['highway=footway', 'highway=cycleway', 'highway=residential', 'highway=park'], False),
        'toilets': (['highway=footway', 'highway=service', 'highway=residential'], False),
        'waste_basket': (['highway=primary', 'highway=residential', 'highway=footway'], True)
    }
    defaults = (['highway=residential', 'highway=unclassified'], False)
    preferred_road_types, need_high_capacity = preferred_config.get(point_type, defaults)

    for feature in geojson_data.get('features', []):
        geometry = feature.get('geometry', {})
        properties = feature.get('properties', {})
        tags = properties.get('tags', {})

        if geometry.get('type') == 'LineString' and tags:
            road_type_matches = any(
                f"highway={v}" in type_str for type_str in preferred_road_types for k, v in tags.items() if
                k == 'highway')
            if need_high_capacity and (int(tags.get('lanes', 1)) > 1 or tags.get('oneway') == 'yes'):
                road_type_matches = True

            if road_type_matches:
                line_coords = geometry.get('coordinates', [])
                distance, bearing = calculate_distance_and_bearing_to_line(point_coords, line_coords)
                if distance < min_distance:
                    min_distance = distance
                    nearest_road = {
                        "id": properties.get('id'),
                        "name": properties.get('name', 'N/A'),
                        "distance": min_distance,
                        "bearing": bearing,
                        "coordinates": line_coords
                    }
    return nearest_road


def modify_road_geometry(coords: List[List[float]], action: str, end_point: str, meters: float) -> List[List[float]]:
    """道路几何修改：延长或缩短"""
    new_coords = copy.deepcopy(coords)

    if end_point == 'head':
        new_coords.reverse()

    if action == 'extend':
        p_last = new_coords[-1]
        p_prev = new_coords[-2]
        bearing = get_bearing(p_prev, p_last)
        new_pt = move_point_planar(p_last, meters, bearing)
        new_coords.append(new_pt)

    elif action == 'shorten':
        remaining = meters
        while len(new_coords) > 1 and remaining > 0:
            p_last = new_coords[-1]
            p_prev = new_coords[-2]
            seg_len = get_distance_m(p_prev, p_last)

            if remaining >= seg_len:
                new_coords.pop()
                remaining -= seg_len
            else:
                bearing = get_bearing(p_prev, p_last)
                keep_dist = seg_len - remaining
                new_pt = move_point_planar(p_prev, keep_dist, bearing)
                new_coords[-1] = new_pt
                remaining = 0

    if end_point == 'head':
        new_coords.reverse()

    return new_coords


# --- 业务逻辑处理函数 ---

def process_level_1(geojson_path: Path):
    """Level 1 逻辑"""
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {geojson_path}: {e}")
        return

    facilities = []
    features = data.get('features', [])
    for idx, feature in enumerate(features):
        geom = feature.get('geometry', {})
        tags = feature.get('properties', {}).get('tags', {})
        if geom.get('type') == 'Point':
            f_type = check_for_facilities(tags)
            if f_type:
                facilities.append((idx, f_type, geom['coordinates']))

    if not facilities:
        return

    target_idx, target_type, target_coords = random.choice(facilities)
    road_info = find_nearest_road(target_coords, target_type, data)

    if not road_info:
        return

    move_distance_m = random.uniform(10, 30)
    road_bearing = road_info['bearing']
    is_forward = random.choice([True, False])
    move_bearing = road_bearing if is_forward else (road_bearing + 180) % 360
    new_coords = move_point_planar(target_coords, move_distance_m, move_bearing)

    data['features'][target_idx]['geometry']['coordinates'] = new_coords
    if 'properties' not in data['features'][target_idx]:
        data['features'][target_idx]['properties'] = {}
    data['features'][target_idx]['properties']['_is_edited'] = True

    cardinal_dir = bearing_to_cardinal(move_bearing)
    intent_str = f"level1|point-shift|parallel-to-road|range=10-30m|mode=random-select"
    stats = {
        "target_feature_id": data['features'][target_idx]['properties'].get('id', 'unknown'),
        "target_type": target_type,
        "nearest_road_id": road_info['id'],
        "original_coords": target_coords,
        "new_coords": new_coords,
        "move_distance_m": round(move_distance_m, 2),
        "move_bearing_deg": round(move_bearing, 2),
        "cardinal_direction": cardinal_dir,
        "relative_direction": "forward" if is_forward else "backward",
        "dist_to_road_m": round(road_info['distance'], 2)
    }

    data['metadata'] = {
        "_edit_intent": intent_str,
        "_edit_stats": stats,
        "_generated_prompt": f"Identify the {target_type}. Move it {int(move_distance_m)} meters {cardinal_dir} along the nearest road."
    }

    original_stem = geojson_path.stem
    output_name = f"{original_stem}_label_1.geojson"
    output_path = geojson_path.parent / output_name

    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=2, ensure_ascii=False)
        print(f"[LEVEL 1 SUCCESS] 生成: {output_name}")
    except Exception as e:
        print(f"[ERROR] 保存失败: {e}")


def process_level_2(geojson_path: Path):
    """
    Level 2 逻辑：道路几何修改，增加双重长度限制
    """
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 {geojson_path}: {e}")
        return

    bike_roads = []
    features = data.get('features', [])

    # 1. 筛选自行车道
    for idx, feature in enumerate(features):
        geom = feature.get('geometry', {})
        tags = feature.get('properties', {}).get('tags', {})

        if geom.get('type') == 'LineString' and is_bike_road(tags):
            coords = geom.get('coordinates', [])
            if len(coords) >= 2:
                length = get_linestring_length(coords)
                # 【改动】调低过滤阈值，让短路也能进入测试逻辑
                if length > 5:
                    bike_roads.append((idx, feature, length))

    if not bike_roads:
        # print(f"[SKIP] {geojson_path.name} (Level 2): 未找到自行车道")
        return

    # 2. 随机选择
    target_idx, target_feature, road_length = random.choice(bike_roads)
    original_coords = target_feature['geometry']['coordinates']
    road_id = target_feature['properties'].get('id', 'unknown')

    action_type = random.choice(['extend', 'shorten'])
    end_point = random.choice(['head', 'tail'])

    # --- 【核心修改：双重限制逻辑】 ---

    # A. 随机一个目标值 (比如 5 到 20米)
    random_target_m = random.uniform(5, 20)

    # B. 计算自身长度的一半
    half_length = road_length * 0.5

    # C. 决策：谁小用谁 (Min(20m, 0.5*Length))
    # 这样既保证了绝对值不超过20米，也保证了短路不会变得面目全非
    if random_target_m > half_length:
        change_meters = half_length
    else:
        change_meters = random_target_m

    # 计算实际比例用于记录
    ratio = change_meters / road_length if road_length > 0 else 0

    # 3. 执行修改
    new_coords = modify_road_geometry(original_coords, action_type, end_point, change_meters)

    data['features'][target_idx]['geometry']['coordinates'] = new_coords
    if 'properties' not in data['features'][target_idx]:
        data['features'][target_idx]['properties'] = {}
    data['features'][target_idx]['properties']['_is_edited'] = True

    # 4. 生成 Metadata
    new_length = get_linestring_length(new_coords)

    intent_str = f"level2|road-mod|{action_type}|{end_point}|ratio={ratio:.2f}"

    stats = {
        "target_feature_id": road_id,
        "original_length_m": round(road_length, 2),
        "change_meters": round(change_meters, 2),
        "action": action_type,
        "end_point": end_point,
        "new_length_m": round(new_length, 2),
        "point_count_diff": len(new_coords) - len(original_coords)
    }

    prompt_verb = "Extend" if action_type == 'extend' else "Shorten"
    if end_point == 'head':
        end_desc = "starting point (head)"
    else:
        end_desc = "ending point (tail)"

    prompt_desc = f"{prompt_verb} the cycling path (ID: {road_id}) at its {end_desc} by approximately {int(change_meters)} meters."

    data['metadata'] = {
        "_edit_intent": intent_str,
        "_edit_stats": stats,
        "_generated_prompt": prompt_desc
    }

    # 5. 保存
    original_stem = geojson_path.stem
    output_name = f"{original_stem}_label_2.geojson"
    output_path = geojson_path.parent / output_name

    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=2, ensure_ascii=False)
        print(f"[LEVEL 2 SUCCESS] 生成: {output_path}")
        print(f"   -> ID: {road_id} (原长 {int(road_length)}m)")
        print(f"   -> 动作: {action_type.upper()} {end_point.upper()} | 改变: {change_meters:.1f}m")
    except Exception as e:
        print(f"[ERROR] 保存失败: {e}")


# --- 主程序入口 ---

def main(args):
    root_directory = Path(args.dir).resolve()

    print(f"开始处理目录: {root_directory}")
    print(f"当前模式: {args.level.upper()}")

    if not root_directory.exists():
        print(f"错误: 目录 {root_directory} 不存在")
        return

    count = 0
    # 遍历文件并分发任务
    for city, bbox, geojson_path in iter_geojson_paths(root_directory):
        geojson_path = Path(geojson_path)

        # 跳过已经是标签文件的 (防止重复处理)
        if "_label_" in geojson_path.name:
            continue

        if args.level == 'level1':
            process_level_1(geojson_path)
        elif args.level == 'level2':
            process_level_2(geojson_path)

        count += 1

    if count == 0:
        print("未找到任何 GeoJSON 文件。")
    else:
        print(f"\n处理完成，共扫描 {count} 个文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoJSON Label Generator")
    parser.add_argument('--level', type=str, choices=['level1', 'level2'])
    parser.add_argument('--dir', type=str, default="../../data_1_2")
    args = parser.parse_args()

    args.level = 'level2'

    main(args)