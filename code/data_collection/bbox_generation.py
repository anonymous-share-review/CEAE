import json
import math
import os.path
import sys
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_collection.collection_tools import generate_grid_bboxes, base_path, bbox_from_center
import folium
from shapely.geometry import box



cities = [
    {"city":"Seattle","state":"WA","lat":47.740947,"lon":-122.332100},          # 上15
    {"city":"San Diego","state":"CA","lat":32.733666,"lon":-117.139746},        # 上2 右2
    {"city":"Baton Rouge","state":"LA","lat":30.451500,"lon":-91.155838},       # 右3
    {"city":"New Orleans","state":"LA","lat":29.951100,"lon":-90.092235},       # 左2
    {"city":"Miami","state":"FL","lat":25.779666,"lon":-80.221723},             # 上2 左3
    {"city":"Tampa","state":"FL","lat":27.977549,"lon":-82.457200},             # 上3
    {"city":"Virginia Beach","state":"VA","lat":36.825951,"lon":-76.011679},    # 下3 左3
    {"city":"Norfolk","state":"VA","lat":36.805884,"lon":-76.229770},           # 下5 右5
    {"city":"Richmond","state":"VA","lat":37.558666,"lon":-77.436000},          # 上2
    {"city":"Washington","state":"DC","lat":38.916183,"lon":-77.025356},        # 上1 右1
    {"city":"Annapolis","state":"MD","lat":39.005349,"lon":-76.573089},         # 上3 左7
    {"city":"Baltimore","state":"MD","lat":39.290400,"lon":-76.658628},
    {"city":"Philadelphia","state":"PA","lat":39.952600,"lon":-75.235511},
    {"city":"Chicago","state":"IL","lat":41.878100,"lon":-87.653930},
    {"city":"New York","state":"NY","lat":40.712800,"lon":-73.923041},
    {"city":"New Haven","state":"CT","lat":41.326266,"lon":-72.927900},
    {"city":"Providence","state":"RI","lat":41.824000,"lon":-71.436909},
    {"city":"Boston","state":"MA","lat":42.324168,"lon":-71.083214},
    {"city":"Burlington","state":"VT","lat":44.466917,"lon":-73.186921},
    {"city":"Buffalo","state":"NY","lat":42.904366,"lon":-78.853880},
    {"city":"Hartford","state":"CT","lat":41.765800,"lon":-72.709531},
    {"city":"Detroit","state":"MI","lat":42.358349,"lon":-83.070103},
    {"city":"Madison","state":"WI","lat":43.028184,"lon":-89.462687},
    {"city":"Minneapolis","state":"MN","lat":44.941868,"lon":-93.265000},
    {"level":"mega","city":"Dallas","state":"TX","lat":32.7767,"lon":-96.7970},
    {"level":"mega","city":"Houston","state":"TX","lat":29.7604,"lon":-95.3698},
    {"level":"mega","city":"Atlanta","state":"GA","lat":33.7490,"lon":-84.3880},
    {"level":"large","city":"Phoenix","state":"AZ","lat":33.4484,"lon":-112.0740},
    {"level":"large","city":"San Francisco","state":"CA","lat":37.7749,"lon":-122.4194},
    {"level":"large","city":"Denver","state":"CO","lat":39.7392,"lon":-104.9903},
    {"level":"large","city":"St. Louis","state":"MO","lat":38.6270,"lon":-90.2224},
    {"level":"medium","city":"Omaha","state":"NE","lat":41.2565,"lon":-95.9345},
    {"level":"medium","city":"Tulsa","state":"OK","lat":36.1540,"lon":-95.9928},
    {"level":"medium","city":"Albuquerque","state":"NM","lat":35.0844,"lon":-106.6504},
    {"level":"medium","city":"Fresno","state":"CA","lat":36.7378,"lon":-119.7871},
    {"level":"medium","city":"Tucson","state":"AZ","lat":32.2226,"lon":-110.9747},
    {"level":"medium","city":"Knoxville","state":"TN","lat":35.9606,"lon":-83.9207},
    {"level":"medium","city":"Des Moines","state":"IA","lat":41.5868,"lon":-93.6250},
    {"level":"medium","city":"Boise","state":"ID","lat":43.6150,"lon":-116.2023},
    {"level":"medium","city":"Spokane","state":"WA","lat":47.6588,"lon":-117.4260},
    {"level":"small","city":"Fort Collins","state":"CO","lat":40.5853,"lon":-105.0844},
    {"level":"small","city":"Huntsville","state":"AL","lat":34.7304,"lon":-86.5861},
    {"level":"small","city":"Eugene","state":"OR","lat":44.0521,"lon":-123.0868},
    {"level":"small","city":"Tallahassee","state":"FL","lat":30.4383,"lon":-84.2807},
    {"level":"small","city":"Santa Rosa","state":"CA","lat":38.4405,"lon":-122.7144},
    {"level":"small","city":"Sioux Falls","state":"SD","lat":43.5446,"lon":-96.7311},
    {"level":"small","city":"Overland Park","state":"KS","lat":38.9822,"lon":-94.6708},
    {"level":"small","city":"Tempe","state":"AZ","lat":33.4255,"lon":-111.9400},
    {"level":"small","city":"Chattanooga","state":"TN","lat":35.0456,"lon":-85.3097},
    {"level":"small","city":"Little Rock","state":"AR","lat":34.7465,"lon":-92.2896},
    {"level":"town","city":"Bozeman","state":"MT","lat":45.6770,"lon":-111.0429},
    {"level":"town","city":"Flagstaff","state":"AZ","lat":35.1983,"lon":-111.6513},
    {"level":"town","city":"Santa Fe","state":"NM","lat":35.6870,"lon":-105.9378},
    {"level":"town","city":"Napa","state":"CA","lat":38.2975,"lon":-122.2869},
    {"level":"town","city":"Jackson","state":"WY","lat":43.4799,"lon":-110.7624},
    {"level":"town","city":"Missoula","state":"MT","lat":46.8721,"lon":-113.9940},
    {"level":"town","city":"Asheville","state":"NC","lat":35.5951,"lon":-82.5515},
    {"level":"extra","city":"Austin","state":"TX","lat":30.2672,"lon":-97.7431},
    {"level":"extra","city":"San Antonio","state":"TX","lat":29.4241,"lon":-98.4936},
    {"level":"extra","city":"San Jose","state":"CA","lat":37.3382,"lon":-121.8863},
    {"level":"extra","city":"Sacramento","state":"CA","lat":38.5816,"lon":-121.4944},
    {"level":"extra","city":"Portland","state":"OR","lat":45.5152,"lon":-122.6784},
    {"level":"extra","city":"Las Vegas","state":"NV","lat":36.1699,"lon":-115.1398},
    {"level":"extra","city":"Orlando","state":"FL","lat":28.5383,"lon":-81.3792},
    {"level":"extra","city":"Charlotte","state":"NC","lat":35.2271,"lon":-80.8431},
    {"level":"extra","city":"Raleigh","state":"NC","lat":35.7796,"lon":-78.6382},
    {"level":"extra","city":"Nashville","state":"TN","lat":36.1447,"lon":-86.8036},
    {"level":"extra","city":"Indianapolis","state":"IN","lat":39.7684,"lon":-86.1581},
    {"level":"extra","city":"Columbus","state":"OH","lat":39.9612,"lon":-82.9988},
    {"level":"extra","city":"Cincinnati","state":"OH","lat":39.1031,"lon":-84.5120},
    {"level":"extra","city":"Cleveland","state":"OH","lat":41.4813,"lon":-81.6704},
    {"level":"extra","city":"Pittsburgh","state":"PA","lat":40.4406,"lon":-79.9959},
    {"level":"extra","city":"Kansas City","state":"MO","lat":39.0997,"lon":-94.5786},
    {"level":"extra","city":"Wichita","state":"KS","lat":37.6872,"lon":-97.3301},
    {"level":"extra","city":"Newark","state":"NJ","lat":40.7357,"lon":-74.1724},
    {"level":"extra","city":"Salt Lake City","state":"UT","lat":40.7608,"lon":-111.8910},
    {"level":"extra","city":"Oklahoma City","state":"OK","lat":35.4676,"lon":-97.5164},
    {"level": "extra", "city": "Lawrence", "state": "KS", "lat": 38.9717, "lon": -95.2353},
    {"level": "extra", "city": "Los Angeles", "state": "CA", "lat": 34.0522, "lon": -118.2437},
    {"level": "extra", "city": "Columbia", "state": "SC", "lat": 34.0007, "lon": -81.0348},
    {"level": "extra", "city": "Lexington", "state": "KY", "lat": 38.0406, "lon": -84.5037},
    {"level": "extra", "city": "Charleston", "state": "SC", "lat": 32.7765, "lon": -79.9311},
    {"level": "extra", "city": "Columbus", "state": "GA", "lat": 32.4600, "lon": -84.9877},
    {"level": "extra", "city": "Springfield", "state": "MO", "lat": 37.2089, "lon": -93.2923},
    {"level":"extra","city":"Colorado Springs","state":"CO","lat":38.8339,"lon":-104.8214},
    {"level":"extra","city":"Aurora","state":"CO","lat":39.7294,"lon":-104.8319},
    {"level":"extra","city":"Pueblo","state":"CO","lat":38.2544,"lon":-104.6091},
    {"level":"extra","city":"Greeley","state":"CO","lat":40.4233,"lon":-104.7091},

    {"level":"extra","city":"Cheyenne","state":"WY","lat":41.1400,"lon":-104.8202},
    {"level":"extra","city":"Casper","state":"WY","lat":42.8501,"lon":-106.3252},
    {"level":"extra","city":"Billings","state":"MT","lat":45.7833,"lon":-108.5007},
    {"level":"extra","city":"Great Falls","state":"MT","lat":47.5002,"lon":-111.3008},
    {"level":"extra","city":"Rapid City","state":"SD","lat":44.0805,"lon":-103.2310},
    {"level":"extra","city":"Bismarck","state":"ND","lat":46.8083,"lon":-100.7837},
    {"level":"extra","city":"Fargo","state":"ND","lat":46.8772,"lon":-96.7898},

    {"level":"extra","city":"Lincoln","state":"NE","lat":40.8136,"lon":-96.7026},
    {"level":"extra","city":"Topeka","state":"KS","lat":39.0473,"lon":-95.6752},
    {"level":"extra","city":"Manhattan","state":"KS","lat":39.1836,"lon":-96.5717},
    {"level":"extra","city":"Salina","state":"KS","lat":38.8403,"lon":-97.6114},
    {"level":"extra","city":"Hutchinson","state":"KS","lat":38.0608,"lon":-97.9298},

    {"level":"extra","city":"Columbia","state":"MO","lat":38.9517,"lon":-92.3341},
    {"level":"extra","city":"Jefferson City","state":"MO","lat":38.5767,"lon":-92.1735},
    {"level":"extra","city":"Joplin","state":"MO","lat":37.0842,"lon":-94.5133},
    {"level":"extra","city":"St. Joseph","state":"MO","lat":39.7675,"lon":-94.8467},

    {"level":"extra","city":"Springfield","state":"IL","lat":39.7817,"lon":-89.6501},
    {"level":"extra","city":"Peoria","state":"IL","lat":40.6936,"lon":-89.5889},
    {"level":"extra","city":"Champaign","state":"IL","lat":40.1164,"lon":-88.2434},
    {"level":"extra","city":"Bloomington","state":"IL","lat":40.4842,"lon":-88.9937},
    {"level":"extra","city":"Rockford","state":"IL","lat":42.2711,"lon":-89.0937},
    {"level":"extra","city":"Decatur","state":"IL","lat":39.8403,"lon":-88.9548},

    {"level":"extra","city":"Fort Wayne","state":"IN","lat":41.0793,"lon":-85.1394},
    {"level":"extra","city":"South Bend","state":"IN","lat":41.6764,"lon":-86.2520},
    {"level":"extra","city":"Lafayette","state":"IN","lat":40.4167,"lon":-86.8753},
    {"level":"extra","city":"Muncie","state":"IN","lat":40.1934,"lon":-85.3864},

    {"level":"extra","city":"Dayton","state":"OH","lat":39.7589,"lon":-84.1916},
    {"level":"extra","city":"Akron","state":"OH","lat":41.0814,"lon":-81.5190},
    {"level":"extra","city":"Canton","state":"OH","lat":40.7989,"lon":-81.3784},
    {"level":"extra","city":"Youngstown","state":"OH","lat":41.0998,"lon":-80.6495},

    {"level":"extra","city":"Lansing","state":"MI","lat":42.7325,"lon":-84.5555},
    {"level":"extra","city":"Ann Arbor","state":"MI","lat":42.2808,"lon":-83.7430},
    {"level":"extra","city":"Kalamazoo","state":"MI","lat":42.2917,"lon":-85.5872},
    {"level":"extra","city":"Grand Rapids","state":"MI","lat":42.9634,"lon":-85.6681},
    {"level":"extra","city":"Flint","state":"MI","lat":43.0125,"lon":-83.6875},
    {"level":"extra","city":"Saginaw","state":"MI","lat":43.4195,"lon":-83.9508},

    {"level":"extra","city":"Syracuse","state":"NY","lat":43.0481,"lon":-76.1474},
    {"level":"extra","city":"Albany","state":"NY","lat":42.6526,"lon":-73.7562},
    {"level":"extra","city":"Binghamton","state":"NY","lat":42.0987,"lon":-75.9179},
    {"level":"extra","city":"Utica","state":"NY","lat":43.1009,"lon":-75.2327},

    {"level":"extra","city":"Allentown","state":"PA","lat":40.6023,"lon":-75.4714},
    {"level":"extra","city":"Harrisburg","state":"PA","lat":40.2732,"lon":-76.8867},
    {"level":"extra","city":"Lancaster","state":"PA","lat":40.0379,"lon":-76.3055},
    {"level":"extra","city":"York","state":"PA","lat":39.9626,"lon":-76.7277},
    {"level":"extra","city":"Scranton","state":"PA","lat":41.4089,"lon":-75.6624},
    {"level":"extra","city":"State College","state":"PA","lat":40.7934,"lon":-77.8600},
    {"level":"extra","city":"Reading","state":"PA","lat":40.3356,"lon":-75.9269},

    {"level":"extra","city":"Roanoke","state":"VA","lat":37.2709,"lon":-79.9414},
    {"level":"extra","city":"Charlottesville","state":"VA","lat":38.0293,"lon":-78.4767},
    {"level":"extra","city":"Lynchburg","state":"VA","lat":37.4138,"lon":-79.1423},
    {"level":"extra","city":"Blacksburg","state":"VA","lat":37.2296,"lon":-80.4139},

    {"level":"extra","city":"Greensboro","state":"NC","lat":36.0726,"lon":-79.7920},
    {"level":"extra","city":"Winston-Salem","state":"NC","lat":36.0999,"lon":-80.2442},
    {"level":"extra","city":"Durham","state":"NC","lat":35.9940,"lon":-78.8986},
    {"level":"extra","city":"High Point","state":"NC","lat":35.9557,"lon":-80.0053},
    {"level":"extra","city":"Greenville","state":"SC","lat":34.8526,"lon":-82.3940},
    {"level":"extra","city":"Spartanburg","state":"SC","lat":34.9496,"lon":-81.9320},

    {"level":"extra","city":"Augusta","state":"GA","lat":33.4735,"lon":-82.0105},
    {"level":"extra","city":"Athens","state":"GA","lat":33.9519,"lon":-83.3576},
    {"level":"extra","city":"Macon","state":"GA","lat":32.8407,"lon":-83.6324},
    {"level":"extra","city":"Warner Robins","state":"GA","lat":32.6130,"lon":-83.6242},

    {"level":"extra","city":"Birmingham","state":"AL","lat":33.5186,"lon":-86.8104},
    {"level":"extra","city":"Montgomery","state":"AL","lat":32.3668,"lon":-86.3000},
    {"level":"extra","city":"Tuscaloosa","state":"AL","lat":33.2098,"lon":-87.5692},
    {"level":"extra","city":"Dothan","state":"AL","lat":31.2232,"lon":-85.3905},
    {"level":"extra","city":"Jackson","state":"MS","lat":32.2988,"lon":-90.1848},
    {"level":"extra","city":"Hattiesburg","state":"MS","lat":31.3271,"lon":-89.2903},

    {"level":"extra","city":"Shreveport","state":"LA","lat":32.5252,"lon":-93.7502},
    {"level":"extra","city":"Lafayette","state":"LA","lat":30.2241,"lon":-92.0198},
    {"level":"extra","city":"Alexandria","state":"LA","lat":31.3113,"lon":-92.4451},

    {"level":"extra","city":"Fayetteville","state":"AR","lat":36.0626,"lon":-94.1574},
    {"level":"extra","city":"Fort Smith","state":"AR","lat":35.3859,"lon":-94.3985},
    {"level":"extra","city":"Jonesboro","state":"AR","lat":35.8423,"lon":-90.7043},

    {"level":"extra","city":"Norman","state":"OK","lat":35.2210,"lon":-97.4395},
    {"level":"extra","city":"Edmond","state":"OK","lat":35.6528,"lon":-97.4781},
    {"level":"extra","city":"Lawton","state":"OK","lat":34.6036,"lon":-98.3959},
    {"level":"extra","city":"Stillwater","state":"OK","lat":36.1156,"lon":-97.0584},

    {"level":"extra","city":"Fort Worth","state":"TX","lat":32.7555,"lon":-97.3308},
    {"level":"extra","city":"Arlington","state":"TX","lat":32.7357,"lon":-97.1081},
    {"level":"extra","city":"Plano","state":"TX","lat":33.0198,"lon":-96.6989},
    {"level":"extra","city":"Irving","state":"TX","lat":32.8140,"lon":-96.9489},
    {"level":"extra","city":"Garland","state":"TX","lat":32.9126,"lon":-96.6389},
    {"level":"extra","city":"Frisco","state":"TX","lat":33.1507,"lon":-96.8236},
    {"level":"extra","city":"McKinney","state":"TX","lat":33.1972,"lon":-96.6398},
    {"level":"extra","city":"Lubbock","state":"TX","lat":33.5779,"lon":-101.8552},
    {"level":"extra","city":"Amarillo","state":"TX","lat":35.2219,"lon":-101.8313},
    {"level":"extra","city":"Midland","state":"TX","lat":31.9973,"lon":-102.0779},
    {"level":"extra","city":"Odessa","state":"TX","lat":31.8457,"lon":-102.3676},
    {"level":"extra","city":"Waco","state":"TX","lat":31.5493,"lon":-97.1467},
    {"level":"extra","city":"Killeen","state":"TX","lat":31.1171,"lon":-97.7278},
    {"level":"extra","city":"College Station","state":"TX","lat":30.6279,"lon":-96.3344},
    {"level":"extra","city":"San Angelo","state":"TX","lat":31.4638,"lon":-100.4370},
    {"level":"extra","city":"Abilene","state":"TX","lat":32.4487,"lon":-99.7331},
    {"level":"extra","city":"Round Rock","state":"TX","lat":30.5083,"lon":-97.6789},
    {"level":"extra","city":"Temple","state":"TX","lat":31.0982,"lon":-97.3428},
    {"level":"extra","city":"Laredo","state":"TX","lat":27.5306,"lon":-99.4803},

    {"level":"extra","city":"Las Cruces","state":"NM","lat":32.3199,"lon":-106.7637},
    {"level":"extra","city":"Rio Rancho","state":"NM","lat":35.2328,"lon":-106.6630},
    {"level":"extra","city":"Roswell","state":"NM","lat":33.3943,"lon":-104.5230},

    {"level":"extra","city":"Mesa","state":"AZ","lat":33.4152,"lon":-111.8315},
    {"level":"extra","city":"Chandler","state":"AZ","lat":33.3062,"lon":-111.8413},
    {"level":"extra","city":"Gilbert","state":"AZ","lat":33.3528,"lon":-111.7890},
    {"level":"extra","city":"Glendale","state":"AZ","lat":33.5387,"lon":-112.1860},
    {"level":"extra","city":"Peoria","state":"AZ","lat":33.5806,"lon":-112.2374},

    {"level":"extra","city":"Provo","state":"UT","lat":40.2338,"lon":-111.6585},
    {"level":"extra","city":"Ogden","state":"UT","lat":41.2230,"lon":-111.9738},
    {"level":"extra","city":"Sandy","state":"UT","lat":40.5649,"lon":-111.8388},
    {"level":"extra","city":"West Jordan","state":"UT","lat":40.6097,"lon":-111.9391},
    {"level":"extra","city":"Lehi","state":"UT","lat":40.3916,"lon":-111.8508},
    {"level":"extra","city":"Layton","state":"UT","lat":41.0602,"lon":-111.9711},

    {"level":"extra","city":"Reno","state":"NV","lat":39.5296,"lon":-119.8138},
    {"level":"extra","city":"Sparks","state":"NV","lat":39.5349,"lon":-119.7527},

    {"level":"extra","city":"Idaho Falls","state":"ID","lat":43.4917,"lon":-112.0333},
    {"level":"extra","city":"Pocatello","state":"ID","lat":42.8616,"lon":-112.4447},
    {"level":"extra","city":"Twin Falls","state":"ID","lat":42.5629,"lon":-114.4605},
    {"level":"extra","city":"Nampa","state":"ID","lat":43.5407,"lon":-116.5635},
    {"level":"extra","city":"Meridian","state":"ID","lat":43.6121,"lon":-116.3915},
    {"level":"extra","city":"Caldwell","state":"ID","lat":43.6629,"lon":-116.6874},

    {"level":"extra","city":"Bend","state":"OR","lat":44.0582,"lon":-121.3153},
    {"level":"extra","city":"Salem","state":"OR","lat":44.9429,"lon":-123.0351},
    {"level":"extra","city":"Corvallis","state":"OR","lat":44.5646,"lon":-123.2620},

    {"level":"extra","city":"Yakima","state":"WA","lat":46.6021,"lon":-120.5059},
    {"level":"extra","city":"Kennewick","state":"WA","lat":46.2112,"lon":-119.1372},
    {"level":"extra","city":"Pasco","state":"WA","lat":46.2396,"lon":-119.1006},
    {"level":"extra","city":"Richland","state":"WA","lat":46.2857,"lon":-119.2845},
    {"level":"extra","city":"Wenatchee","state":"WA","lat":47.4235,"lon":-120.3103},

    {"level":"extra","city":"Bakersfield","state":"CA","lat":35.3733,"lon":-119.0187},
    {"level":"extra","city":"Riverside","state":"CA","lat":33.9806,"lon":-117.3755},
    {"level":"extra","city":"San Bernardino","state":"CA","lat":34.1083,"lon":-117.2898},
    {"level":"extra","city":"Modesto","state":"CA","lat":37.6391,"lon":-120.9969},
    {"level":"extra","city":"Visalia","state":"CA","lat":36.3302,"lon":-119.2921},
    {"level":"extra","city":"Victorville","state":"CA","lat":34.5361,"lon":-117.2912},
    {"level":"extra","city":"Ontario","state":"CA","lat":34.0633,"lon":-117.6509},
    {"level":"extra","city":"Moreno Valley","state":"CA","lat":33.9425,"lon":-117.2297},
    {"level":"extra","city":"Roseville","state":"CA","lat":38.7521,"lon":-121.2880},
    {"level":"extra","city":"Elk Grove","state":"CA","lat":38.4088,"lon":-121.3716},
    {"level":"extra","city":"Folsom","state":"CA","lat":38.6779,"lon":-121.1761},

    {"level":"extra","city":"Worcester","state":"MA","lat":42.2626,"lon":-71.8023},
    {"level":"extra","city":"Springfield","state":"MA","lat":42.1015,"lon":-72.5898},
    {"level":"extra","city":"Lowell","state":"MA","lat":42.6334,"lon":-71.3162},
    {"level":"extra","city":"Manchester","state":"NH","lat":42.9956,"lon":-71.4548},
    {"level":"extra","city":"Concord","state":"NH","lat":43.2081,"lon":-71.5376},
    {"level":"extra","city":"Montpelier","state":"VT","lat":44.2601,"lon":-72.5754},
    {"level":"extra","city":"Rutland","state":"VT","lat":43.6106,"lon":-72.9726}
]


def visualize_bboxes(bboxes, center_lat, center_lon, zoom_start=14):

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    for idx, b in enumerate(bboxes):
        south, west, north, east = b["bbox"]
        rect = [
            [south, west],
            [south, east],
            [north, east],
            [north, west],
            [south, west],
        ]
        folium.Polygon(
            locations=rect,
            color="red",
            weight=2,
            fill=False,
            tooltip=f"BBox {idx+1}"
        ).add_to(m)

    folium.Marker([center_lat, center_lon], popup="中心点", icon=folium.Icon(color="blue")).add_to(m)

    return m

grid_size = 5
side_km = 1

all_bboxes = []

seen = set()
unique_cities = []

for info in cities:
    city_name = info["city"]
    if city_name not in seen:
        unique_cities.append(info)
        seen.add(city_name)


for info in unique_cities:
    city_name = info["city"]
    state = info.get("state", "")
    lat, lon = info["lat"], info["lon"]
    print(f"Now processing city: {city_name}, {state} (lat={lat}, lon={lon})")

    bboxes = generate_grid_bboxes(lat, lon, side_km=side_km, grid_size=grid_size)

    output_path = os.path.join(base_path,city_name)
    os.makedirs(output_path, exist_ok=True)

    m = visualize_bboxes(bboxes, lat, lon, zoom_start=13)
    html_path = os.path.join(output_path, f"{city_name}_bboxes.html")
    m.save(html_path)

    bboxes_with_index = []
    half = 5 // 2
    idx = 0
    for i in range(-half, half + 1):  # row
        for j in range(-half, half + 1):  # col
            b = bboxes[idx]
            bboxes_with_index.append({
                "city": city_name,
                "state": state,
                "row": i + half + 1,  # 行号 (1-based)
                "col": j + half + 1,  # 列号 (1-based)
                "center": {"lat": b["center"][0], "lon": b["center"][1]},
                "bbox": {
                    "south": b["bbox"][0],
                    "west": b["bbox"][1],
                    "north": b["bbox"][2],
                    "east": b["bbox"][3]
                }
            })
            idx += 1
            all_bboxes.append(bboxes_with_index[-1])  # 加到总集合

    json_path = os.path.join(output_path, f"{city_name}_bboxes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bboxes_with_index, f, indent=2, ensure_ascii=False)

    print(f"✅ Save {city_name}:")
    print(f"   HTML: {html_path}")
    print(f"   JSON: {json_path}")

if all_bboxes:
        # 用美国中心点 roughly 定位
        m_all = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        for entry in all_bboxes:
            bbox = entry["bbox"]
            city = entry["city"]
            state = entry["state"]
            row, col = entry["row"], entry["col"]

            # 画矩形
            folium.Rectangle(
                bounds=[(bbox["south"], bbox["west"]), (bbox["north"], bbox["east"])],
                color="red", weight=1, fill=False,
                popup=f"{city}, {state} (row={row}, col={col})"
            ).add_to(m_all)

        html_all = os.path.join(base_path, "all_cities_bboxes.html")
        m_all.save(html_all)
        print(f"🌎 Global overview file saved: {html_all}")

print("Number of unique cities:", len(unique_cities))
