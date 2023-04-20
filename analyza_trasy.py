# import nezbytných knihoven
import networkx as nx
import osmnx as ox
from pyproj import Transformer
import rasterio
import numpy as np
from PIL import Image
from owslib.wms import WebMapService
from rasterio.warp import calculate_default_transform, reproject, Resampling
import bisect
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import folium
from folium.features import DivIcon
import streamlit as st
from streamlit_folium import st_folium
import os
import pickle
from shapely.geometry import Point, LineString
import pyproj
from shapely.ops import transform

### Absoluní cesta ke složce skriptu
def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

### Absoluní cesta k souboru
def get_data_path(path):
    #if path list return list of paths
    if isinstance(path, list):
        return [os.path.join(get_script_dir(),"cache", p) for p in path]
    else:
        return os.path.join(get_script_dir(),"cache", path)
    
### Ověření, zda existuje soubor
def file_exists(path):
    return os.path.isfile(get_data_path(path))

### Funkce pro aktualizaci textu statusu průběhu
def update_status(status, text):
    status.text(text)

### Funkce pro ukončení výpisu statusu průběhu
def stop_status(status):
    status.empty()

### Stažení OSM dat
def download_osm_graph(center_coordinates, area_half_side_length, network_type="all"):
    return ox.graph_from_point(center_coordinates, dist=area_half_side_length, network_type=network_type)

### Stažení výškových dat DMR5G
def get_bbox(center_coordinates, area_half_side_length):
    x1, x2, y1, y2 = ox.utils_geo.bbox_from_point(center_coordinates, dist=area_half_side_length, project_utm=False, return_crs=False)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5514")
    x1, y1 = transformer.transform(x1, y1)
    x2, y2 = transformer.transform(x2, y2)
    return [x2, y2, x1, y1]

def download_elevation_data(bbox, output_file, user_agent):
    wms = WebMapService("https://ags.cuzk.cz/arcgis2/services/dmr5g/ImageServer/WMSServer", headers={"User-agent": user_agent})
    img = wms.getmap(
        layers=["dmr5g:None"],
        size=[4000, 4000],
        srs="EPSG:5514",
        bbox=bbox,
        transparent=True,
        bgcolor="0xFFFFFF",
        format="image/tiff",
        headers={"User-agent": user_agent})
    return Image.open(img)

def preprocess_elevation_data(image, min_elevation, max_elevation):
    image_data = np.array(image)
    image_data[image_data > max_elevation] = None
    image_data[image_data <= min_elevation] = None
    return image_data

def save_geotiff(image, image_data, bbox, file_path):
    metadata = {'driver': 'GTiff',
                'dtype': image_data.dtype,
                'nodata': None,
                'width': image.width,
                'height': image.height,
                'count': 1,
                'crs': rasterio.crs.CRS.from_epsg(5514),
                'transform': rasterio.transform.from_bounds(*bbox, image.width, image.height)}

    with rasterio.open(get_data_path(file_path), 'w', **metadata) as dst:
        dst.write(image_data, indexes=1)

def reproject_geotiff(src_path, dst_path, dst_crs='EPSG:4326'):
    with rasterio.open(get_data_path(src_path)) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(get_data_path(dst_path), 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
                
### Přiřazení výškových dat k cestní síti
def add_elevation_to_graph(graph, raster_path):
    return ox.elevation.add_node_elevations_raster(graph, get_data_path(raster_path))

def remove_nodes_by_elevation(graph, min_elevation):
    nan_nodes = [n for n, data in graph.nodes(data=True) if (np.isnan(data['elevation']) or data['elevation'] <= min_elevation)]
    graph.remove_nodes_from(nan_nodes)
    graph.remove_edges_from(list(graph.edges(nan_nodes)))
    return graph

def verify_node_elevation(graph):
    assert not np.isnan(np.array(graph.nodes(data="elevation"))[:, 1]).any()

def add_edge_grades(graph, add_absolute=True):
    return ox.elevation.add_edge_grades(graph, add_absolute=add_absolute)

### Výpočet rychlosti  v závislosti na sklonu a rychlosti v metrech za minutu

# Funkce pro výpočet rychlosti pohybu na základě sklonu (viz https://www.researchgate.net/figure/Walking-speeds-slope-in-percent_fig3_282335699)
# reduce: 1 - základní rychlost, <1 - rychlost snížená na dané procento, >1 - rychlost zvýšená na dané procento
def getSpeedFromSlope(x, reduce=1):
    if x == 0:
        speed = 3.78
    else:
        boundaries = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] #percent
        values = [0.61,1.28,1.38,1.49,1.63,1.79,1.99,2.24,3.03,6.16,4.30,4.00,2.99,1.98,1.49,1.19,0.99,0.85,0.75,0.66,0.60,0.54,0.24] #naismith km/h

        speed = values[bisect.bisect_left(boundaries, x)]
    return speed * reduce

# Rychlost v metrech za minutu
def meters_per_minute(travel_speed):
    return travel_speed * 1000 / 60  # km per hour to m per minute

### Výpočet času pro cestu mezi uzly - Přidá do grafu čas pohybu mezi uzly
def add_travel_time_to_edges(graph, speed_reduce=0.72):
    for _, _, _, data in graph.edges(data=True, keys=True):
        grade = data.get("grade", 0)
        speed = getSpeedFromSlope(grade, speed_reduce)
        data["time"] = float(data["length"] / meters_per_minute(speed))
    return graph

### Převod souřadnic z GPS na tuple
def parse_coordinates(coordinates_str):
    # Split the string into latitude and longitude strings
    lat_str, long_str = coordinates_str.split(', ')

    # Extract the numeric values and convert to float
    latitude = float(lat_str[:-1]) if lat_str[-1] in ['N', 'S'] else None
    longitude = float(long_str[:-1]) if long_str[-1] in ['E', 'W'] else None

    # Determine the direction (north/south for latitude, east/west for longitude)
    if lat_str[-1] == 'S':
        latitude = -latitude
    if long_str[-1] == 'W':
        longitude = -longitude

    # Create a tuple of coordinates
    coordinates = (latitude, longitude)
    
    return coordinates

# Funkce pro nalezení nejbližšího bodu na linii
def nearest_point_on_line(point, line):
    return line.interpolate(line.project(point))

def length_in_meters(geometry):
    """
    Převede délku liniové geometrie z WGS84 (EPSG:4326) na S-JTSK (EPSG:5514) a vrátí délku v metrech.

    :param geometry: Liniová geometrie ve formátu WGS84 (EPSG:4326)
    :return: Délka geometrie v jednotkách S-JTSK (metrech)
    """
    if not isinstance(geometry, LineString):
        raise ValueError("Input geometry must be a LineString.")

    # Projekce pro konverzi mezi souřadnicovými systémy
    project_WGS84_to_SJTSK = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True).transform

    # Převod geometrie na S-JTSK
    geometry_sjtsk = transform(project_WGS84_to_SJTSK, geometry)

    # Výpočet délky v metrech
    length_m = geometry_sjtsk.length

    return length_m

# Funkce pro nalezení nejbližšího uzlu v toleranci
def find_nearby_node(G, point, tolerance=100):
    # Najděte uzly v dosahu tolerance
    nearby_nodes = ox.distance.nearest_nodes(G, X=[point[1]], Y=[point[0]], return_dist=True)
    for node, dist in zip(nearby_nodes[0], nearby_nodes[1]):
        if dist <= tolerance:
            return node  # Pokud existuje uzel v toleranci, vraťte jeho ID
    return None  # Pokud není žádný uzel v toleranci, vraťte None



# Funkce pro přidání nového uzlu na nejbližší hranu
def add_node_on_nearest_edge(G, point, search_nearby=True):

    # Ověří, zda se poblíž vytvářeného uzlu nenachází jiný vhodný uzel v toleranci, pokud ano, vrátí graf bez změn
    if search_nearby:
        nearby_node = find_nearby_node(G, point)
        if nearby_node is not None:
            return G
    
    # Vyhledejte nejbližší hranu
    nearest_edge = ox.distance.nearest_edges(G, X=[point[1]], Y=[point[0]])[0]
    u, v, _ = nearest_edge
    
    # Vytvořte Point objekt ze zadaných souřadnic
    point_geom = Point(point[::-1])
    
    # Získejte geometrii nejbližší hrany pomocí get_route_edge_attributes a vypočítejte nejbližší bod na ní
    edge_geom = G[u][v][0]['geometry']
    nearest_point = nearest_point_on_line(point_geom, edge_geom)
    
    # Rozdělte původní geometrii hrany na dvě části pomocí nejbližšího bodu
    coords_list = list(edge_geom.coords)
    coords_np = np.array(coords_list)
    nearest_point_np = np.array(nearest_point.coords[0])
    nearest_point_index = np.argmin(np.linalg.norm(coords_np - nearest_point_np, axis=1))
    first_part = LineString(coords_list[:nearest_point_index + 1] + [nearest_point.coords[0]])
    second_part = LineString([nearest_point.coords[0]] + coords_list[nearest_point_index:])
    
    # Přidejte nový uzel do grafu
    new_node_id = max(G.nodes) + 1
    G.add_node(new_node_id, x=nearest_point.x, y=nearest_point.y, osmid=new_node_id)
    
    # Přidejte nové hrany spojující nový uzel se starými uzly
    edge_data = G[u][v][0].copy()

    G.add_edge(u, new_node_id, **edge_data)
    G.add_edge(new_node_id, v, **edge_data)
    
    # Nastavte geometrii nových hran
    G[u][new_node_id][0]['geometry'] = first_part
    G[new_node_id][v][0]['geometry'] = second_part

    # Aktualizujte délku nových hran
    G[u][new_node_id][0]['length'] = length_in_meters(first_part)
    G[new_node_id][v][0]['length'] = length_in_meters(second_part)

    ### Přidání hran v opačném směru
    # Reverse the coordinates to create new LineStrings
    first_part_reversed = LineString(first_part.coords[::-1])
    second_part_reversed = LineString(second_part.coords[::-1])

    G.add_edge(new_node_id, u, **edge_data)
    G.add_edge(v, new_node_id, **edge_data)
    
    # Nastavte geometrii nových hran
    G[new_node_id][u][0]['geometry'] = first_part_reversed
    G[v][new_node_id][0]['geometry'] = second_part_reversed

    # Aktualizujte délku nových hran
    G[u][new_node_id][0]['length'] = length_in_meters(first_part_reversed)
    G[new_node_id][v][0]['length'] = length_in_meters(second_part_reversed)
    
    # Odstraňte původní hranu
    G.remove_edge(u, v)
    G.remove_edge(v, u)

    return G



### Převod souřadnic na nejbližší node
def find_nearest_nodes(seznam_gps, G):
    node_ids = []
    for gps in seznam_gps:
        # Převod souřadnic na tuple
        gps_tuple = parse_coordinates(gps)
        # Vyhledání nejbližšího node pomocí funkce ox.get_nearest_node
        node_id = ox.distance.nearest_nodes(G, gps_tuple[1], gps_tuple[0])
        # Přidání node id do seznamu
        node_ids.append(node_id)
    return node_ids

### Zobrazení vybraných uzlů pomocí matplotlib
def plot_selected_nodes(G, node_ids):
    fig, ax = ox.plot_graph(G, node_color='k', node_size=5, edge_color='gray', edge_linewidth=0.5, show=False)
    nc = ['r' if node in node_ids else 'w' for node in G.nodes()]
    ns = [25 if node in node_ids else 5 for node in G.nodes()]
    coords = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()]
    x, y = zip(*coords)
    ax.scatter(x, y, c=nc, s=ns, alpha=1)
    plt.show()

### Výpočet nejkratších cest mezi uzly pomocí Floyd-Warshall algoritmu
def compute_shortest_paths(graph, weight="time"):
    return nx.floyd_warshall_numpy(graph, weight=weight)

### Vytvoření všech možných kombinací uzlů (permutace)
def get_node_combinations(node_indexes):
    return list(itertools.permutations(node_indexes))

### Doplnění kombinací jednotlivých cest o kombinaci různých startovních bodů a jeden cílový bod
def create_combinations_with_start_end(combinations, gps_start_list, G, end_node):
    start_node_ids = find_nearest_nodes(gps_start_list, G)
    start_node_indexes = [list(G.nodes()).index(node_id) for node_id in start_node_ids]
    end_node_index = list(G.nodes()).index(end_node)

    new_combinations = []
    for start_node_index in start_node_indexes:
        for combination in combinations:
            new_combinations.append((start_node_index,) + combination + (end_node_index,))

    return new_combinations

### Alternativní řešení, kdy každý node v území může být startovní
def create_alternate_combinations(combinations, G, end_node):
    end_node_index = list(G.nodes()).index(end_node)
    alt_combinations = []

    for i in range(len(G.nodes())):
        for combination in combinations:
            alt_combinations.append((i,) + combination + (end_node_index,))

    return alt_combinations

### Výpočet délky cesty pro každou kombinaci bodů
def calculate_shortest_combinations(combinations, shortest_paths):
    paths = []
    for combo in combinations:
        path_length = 0
        for i in range(len(combo) - 1):
            path_length += shortest_paths[combo[i]][combo[i+1]]
        paths.append((combo, path_length))

    shortest_combinations = sorted(paths, key=lambda x: x[1])

    return shortest_combinations

### Výběr nodu z grafu podle indexu
def get_node_by_index(G, nodeIndex):
    return list(G.nodes())[nodeIndex]

### Výběr nodů z grafu podle indexů
def get_nodes_from_indexes(G, shortest_path_nodes):
    return [get_node_by_index(G, node_index) for node_index in shortest_path_nodes]

### Vytvoření cest z vybraných nodů
def create_routes(G, mynodes):
    routes = []
    total_length = 0
    parts = []
    old ='''
    for i in range(len(mynodes) - 1):
        source = mynodes[i]
        target = mynodes[i+1]
        if source != target:
            route = ox.shortest_path(G, source, target, weight="time")
        
            # Calculate the path length and add it to the total length
            path_length = nx.path_weight(G, route, "length")
            total_length += path_length

            routes.append(route)

    return [routes, total_length]
    '''
    #create output with route-parts with their length and times, total length, all routes

    for i in range(len(mynodes) - 1):
        source = mynodes[i]
        target = mynodes[i+1]
        if source != target:
            route = ox.shortest_path(G, source, target, weight="time")
            time = nx.path_weight(G, route, "time")
            # Calculate the path length and add it to the total length
            path_length = nx.path_weight(G, route, "length")
            total_length += path_length

            routes.append(route)
            parts.append({"route":route, "length":path_length, "time":time})

    return [routes, total_length, parts]

### Zobrazení cest pomocí matplotlib
def plot_graph_routes(G, routes, node_size=1, figsize=(10, 10)):
    # Create a color map with a unique color for each route
    route_colors = cm.viridis(np.linspace(0, 1, len(routes)))

    # Plot the graph with the routes
    fig, ax = ox.plot_graph_routes(G, routes, route_colors=route_colors, node_size=node_size, figsize=figsize)

### Vytvoření mapy pomocí folium a Mapy.cz
def create_map(G, mynodes, routes, center_coordinates):
    def create_svg_marker(i, is_start):
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg id="Vrstva_2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16.4 19.5">
    <defs>
        <style>.cls-1{{fill:#fff;}}.cls-2{{font-family:ArialMT, Arial;font-size:9px;}}.cls-2,.cls-3{{fill:#d82931;}}</style>
    </defs>
    <g id="Vrstva_1-2">
        <g>
            <path class="cls-3" d="m8.3,0c4.5,0,8.1,3.6,8.1,8.1,0,1.5-.4,2.9-1.1,4.1-1.2,1.9-2.1,3.2-7.1,7.3C3.2,15.3,2.2,14.2,1,12.2c-.6-1.2-1-2.6-1-4.1C.1,3.6,3.7,0,8.3,0Z"/>
            <circle class="cls-1" cx="8.2" cy="8.2" r="6.34"/>
            <text class="cls-2" transform="translate(8 11.5)" text-anchor="middle">
                <tspan x="0" y="0">{i+1}</tspan>
            </text>
        </g>
    </g>
</svg>'''

    # Create a Folium map object and set the initial view
    m = folium.Map(location=[center_coordinates[0], center_coordinates[1]], zoom_start=14, tiles='https://tiles.windy.com/v1/maptiles/outdoor/256/{z}/{x}/{y}/?lang=en', name="Mapy.cz", attr="Mapy.cz", zoom_control=False)
    
    for i, mynode_id in enumerate(mynodes):
        node = G.nodes(data=True)[mynode_id]
        is_start = (i == 0) and (len(mynodes) - 1 != len(routes))
        icon_anchor = (0, 48) if is_start else (18, 48)
        folium.Marker(location=[node['y'], node['x']], 
                      icon=DivIcon(icon_size=(36, 42.8), icon_anchor=icon_anchor, html=create_svg_marker(i, is_start))
                     ).add_to(m)

    for route in routes:
        m = ox.plot_route_folium(G, route, route_map=m, **{'color': '#ff0000'}, opacity=0.8, fit_bounds=False)
        
    # Fit and display the map
    m.fit_bounds(m.get_bounds(), padding=[50, 50])
    return m





# Main code
if __name__ == '__main__':
    ### @@@ streamlit expander s checkboxem "Vynutit úplnou aktualizaci dat"
    if 'key' not in st.session_state:
        st.session_state['max_results'] = 3000

    with st.expander("Nastavení aplikace"):
        force_update = st.button("Vynutit úplnou aktualizaci dat")
        num_returned_paths = st.slider("Maximální počet vrácených tras", 0, st.session_state['max_results'], 40,10)
        max_combinations_placeholder = st.empty()

    ### @@@ streamlit stav
    status = st.empty()

    ### Stažení OSM dat
    center_coordinates = (49.40566394099164, 12.80545193036346)  # Rudolfova pila
    area_half_side_length = 5500  # meters; half of the square area side length

    #pickle save the shortest_combinations to a file if file exists and not bool forced to recalculate, load the file
    if not (file_exists('shortest_combinations.pickle')) or not (file_exists('g_elev.graphml')) or force_update:

        update_status(status,"Stahuji OSM data") ### @@@ aktualizace streamlit stavu
        G = download_osm_graph(center_coordinates, area_half_side_length)

        #coordinates_str = "49.3930278N, 12.8011208E"

        # Převeďte souřadnice na standardní formát
        #point = parse_coordinates(coordinates_str)

        # Vytvoří nový uzel na hraně, která je nejbližší k bodu
        #G = add_node_on_nearest_edge(G, point, search_nearby=False)

        ### Stažení výškových dat DMR5G
        bbox = get_bbox(center_coordinates, area_half_side_length)
        user_agent = "Mozilla/5.0 (Linux; Android 8.0.0; SM-G960F Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.84 Mobile Safari/537.36"

        update_status(status, "Stahuji výšková data DMR5G") ### @@@ aktualizace streamlit stavu
        image = download_elevation_data(bbox, 'elevation.tiff', user_agent)

        update_status(status, "Upravuji výšková data DMR5G") ### @@@ aktualizace streamlit stavu
        image_data = preprocess_elevation_data(image, min_elevation=0, max_elevation=1603)

        update_status(status, "Ukládám výšková data DMR5G") ### @@@ aktualizace streamlit stavu
        save_geotiff(image, image_data, bbox, 'elevation.tiff')

        update_status(status, "Převádím výšková data z S-JTSK do WGS-84") ### @@@ aktualizace streamlit stavu
        reproject_geotiff('elevation.tiff', 'elevation_4326.tiff', dst_crs='EPSG:4326')

        ### Přiřazení výškových dat k cestní síti
        raster_path = ['elevation_4326.tiff']

        update_status(status, "Přiřazení výškových dat k cestní síti") ### @@@ aktualizace streamlit stavu
        G_elev = add_elevation_to_graph(G, raster_path)

        update_status(status, "Odstranění uzlů s chybnými výškami") ### @@@ aktualizace streamlit stavu
        G_elev = remove_nodes_by_elevation(G_elev, min_elevation=10)
        verify_node_elevation(G_elev)

        update_status(status, "Výpočet sklonu na trase") ### @@@ aktualizace streamlit stavu

        G_elev = add_edge_grades(G_elev, add_absolute=True)

        ### Výpočet času pro cestu mezi uzly
        update_status(status, "Výpočet času pro cestu mezi uzly") ### @@@ aktualizace streamlit stavu
        G_elev = add_travel_time_to_edges(G_elev, speed_reduce=0.72)

        
        # Save graph to a file
        ox.io.save_graphml(G_elev, filepath=get_data_path('g_elev.graphml'))

        ### Převod souřadnic na nejbližší node
        gps_seznam = ["49.3926261N, 12.8029267E",
        "49.3856700N, 12.8102089E",
        "49.3801017N, 12.8202792E",
        "49.3694169N, 12.8150381E",
        "49.3770761N, 12.8012128E",
        "49.3840933N, 12.7845400E",
        "49.4017264N, 12.8011631E",
        "49.3987294N, 12.7941933E"
        ]

        update_status(status, "Převod souřadnic na nejbližší node") ### @@@ aktualizace streamlit stavu
        node_ids = find_nearest_nodes(gps_seznam, G_elev)

        ### Zobrazení vybraných uzlů pomocí matplotlib - deaktivováno
        # plot_selected_nodes(G_elev, node_ids)

        ### Výpočet nejkratších cest mezi uzly pomocí Floyd-Warshall algoritmu

        update_status(status, "Výpočet nejkratších cest mezi uzly pomocí Floyd-Warshall algoritmu") ### @@@ aktualizace streamlit stavu
        shortest_paths = compute_shortest_paths(G_elev, weight="time")

        ### Výpočet délky cesty pro každou kombinaci bodů

        ### Vytvoření seznamu indexů uzlů
        node_indexes = [list(G_elev.nodes()).index(node_id) for node_id in node_ids]

        ### Vytvoření všech možných kombinací uzlů (permutace)
        update_status(status, "Vytvoření všech možných kombinací uzlů (permutace)") ### @@@ aktualizace streamlit stavu
        
        combinations = get_node_combinations(node_indexes)

        ### Doplnění kombinací jednotlivých cest o kombinaci různých startovních bodů a jeden cílový bod
        libovolny_start = False

        if not libovolny_start:
            update_status(status, "Doplnění kombinací jednotlivých cest o kombinaci různých startovních bodů a jeden cílový bod") ### @@@ aktualizace streamlit stavu

            gps_seznam_start = [
                "49.4080286N, 12.7872861E",
                "49.4040633N, 12.7899467E",
                "49.3995669N, 12.7933800E",
                "49.3930314N, 12.8016628E",
                "49.3967183N, 12.7908908E",
                "49.3807117N, 12.8222619E",
                "49.3786722N, 12.8105892E",
            ]

            end_node = ox.distance.nearest_nodes(G_elev, center_coordinates[1], center_coordinates[0])
            combinations = create_combinations_with_start_end(combinations, gps_seznam_start, G_elev, end_node)

        ### Alternativní řešení, kdy každý node v území může být startovní
        if libovolny_start:
            update_status(status, "Doplnění kombinací jednotlivých cest o kombinaci všech možností startovních bodů a jeden cílový bod") ### @@@ aktualizace streamlit stavu
            
            end_node = ox.distance.nearest_nodes(G_elev, center_coordinates[1], center_coordinates[0])
            combinations = create_alternate_combinations(combinations, G_elev, end_node)
            
        update_status(status, "Výpočet délky cesty pro každou kombinaci bodů") ### @@@ aktualizace streamlit stavu
        shortest_combinations = calculate_shortest_combinations(combinations, shortest_paths)
        # Save the shortest_combinations to a pickle file
        with open(get_data_path('shortest_combinations.pickle'), 'wb') as handle:
            pickle.dump(shortest_combinations, handle, protocol=pickle.HIGHEST_PROTOCOL)
        st.session_state.force_update_shortest_combinations = False
    else:
        update_status(status, "Nahrávám uložená OSM data s výškovými daty")
        edge_dtypes = {
            "time": float,
            "grade_abs": float,
            "grade": float,
            "length": float
        }

        node_dtypes = {
            "elevation": float
        }

        G_elev = ox.io.load_graphml(get_data_path('g_elev.graphml'), edge_dtypes=edge_dtypes, node_dtypes=node_dtypes)

        update_status(status, "Nahrávám uložené kombinace nejkratších tras")
        with open(get_data_path('shortest_combinations.pickle'), 'rb') as handle:
            shortest_combinations = pickle.load(handle)
    
    max_combinations_placeholder.write(f"Maximální teoretický počet tras je **{len(shortest_combinations)}**.")

    shortest_combinations = shortest_combinations[:num_returned_paths]

    ### Přidání výběru trasy
    max_combinations = len(shortest_combinations)

    slider = st.slider('Vyber trasu podle pořadí', 1, max_combinations, 1)

    ### Výběr nejkratší cesty
    shortest_path_nodes = shortest_combinations[slider - 1][0]

    col1, col2 = st.columns(2)

    with col1:
        #st.write(f"Délka této trasy je **{int(shortest_combinations[slider - 1][1])}** minut.")
        st.metric("Délka pochodu", f"{int(shortest_combinations[slider - 1][1])} min", label_visibility="visible")

    ### Výběr nodů z grafu podle indexů
    mynodes = get_nodes_from_indexes(G_elev, shortest_path_nodes)

    ### Vytvoření cest z vybraných nodů
    update_status(status, "Vytvoření cest z vybraných nodů") ### @@@ aktualizace streamlit stavu
    routes, length, parts = create_routes(G_elev, mynodes)
    #st.write(parts) ###################################################parts
    
    with col2:
        #st.write(f"Délka této trasy je **{round(length/1000,2)}** km.")
        st.metric("Délka trasy", f"{round(length/1000,2)} km", label_visibility="visible")
    
    ### Zobrazení cest pomocí matplotlib - deaktivováno
    # plot_graph_routes(G_elev, routes)

    ### Zobrazení cest pomocí folium a mapy.cz
    update_status(status, "Vytváření mapy pomocí Folium a Mapy.cz") ### @@@ aktualizace streamlit stavu
    m = create_map(G_elev, mynodes, routes, center_coordinates)

    stop_status(status) ### @@@ smazání streamlit stavu
    st_folium(m, height=600, width=None, returned_objects=[])
    