import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image, ImageFilter, ImageCms
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import shutil
from skimage import color as colorcv
from datetime import datetime
from scipy.spatial import distance_matrix

import warnings
warnings.filterwarnings("ignore")

def hextoint(hexcode):
    return (int(hexcode[:2], 16), int(hexcode[2:4], 16), int(hexcode[4:], 16))

def get_colours():
    headers = {
        'User-Agent': "Mozilla/5.0"
    }

    r = requests.get("https://www.bricklink.com/catalogColors.asp", headers=headers)
    tree = BeautifulSoup(r.text, "lxml")
    html_table = tree.select("table")[3].select("table")[0]
    colour_table = pd.read_html(str(html_table), header=0)[0]
    colour_table = colour_table.drop(["Unnamed: 1", "Unnamed: 2"] , axis=1)
    rgb_table = pd.DataFrame([hextoint(td.attrs["bgcolor"]) for td in html_table.select("td[bgcolor]")], 
                             columns=["r", "g", "b"])
    colour_table = colour_table.merge(rgb_table, left_index=True, right_index=True)
    current_colours = colour_table[(colour_table["Color Timeline"].str.contains(str(datetime.now().year)))]
    current_colours = current_colours.reset_index(drop=True)
    current_colours[["l", "a", "b2"]] = pd.DataFrame(
        colorcv.rgb2lab(current_colours[["r", "g", "b"]].values.reshape(1, -1, 3)/255)[0])

    return current_colours

def find_rectangles(G, named_colors, maxim_size=12, maxim_width=6, critical_colors=set([])):
    assert maxim_size % 2 == 0 or maxim_size < 4, "Even numbers for max_size only!"
    bad_dims = set([5, 7, 9, 11, 13])
    all_rectangles = []
    
    # all connected subgraphs
    for c in nx.connected_components(G):
        gr = nx.Graph(G.subgraph(c))
        max_size = maxim_size
        max_width = maxim_width
        while list(gr.nodes()): # as long as there are nodes
            node = min(gr.nodes(), key=lambda x: [int(y) for y in x.split("|")]) # get min x,y
            if named_colors[node] in critical_colors:
                max_size = 6
                max_width = 6
            rectangle = []
            count_south = 0

            # vertical edge (expand southwards) until bottom line of subgraph
            # or max size
            while node in gr and count_south <= max_size - 1:
                rectangle.append(node)
                x,y = node.split("|")
                node = "{}|{}".format(int(x)+1, y)
                # widths > 4 are: 6, 8, 10, 12, 14
                if count_south > 2:
                    if node in gr and count_south + 1 <= max_size - 1:
                        rectangle.append(node)
                        node = "{}|{}".format(int(x)+2, y)
                        count_south += 1
                count_south += 1
            
            # 5 and 3 are stupid numbers for LEGO plates, 
            # 3 can only occur in one dimension,
            # 5 can never occur anywhere
            if count_south in bad_dims:
                rectangle.pop(-1)
            if count_south == 3:
                bloody_3 = True
            else:
                bloody_3 = False
                
            final_rectangle = rectangle
            
            border_found = False
            count_north = 0
            
            # horizontal edge (expand eastwards)
            while (not border_found):
                right_border = []
                for rect_node in rectangle:
                    x,y = rect_node.split("|")
                    
                    node = "{}|{}".format(x, int(y)+1)
                    # widths > 4 are: 6, 8, 10, 12  
                    if count_north > 2:
                        if node in gr and count_north + 1 <= max_width - 1:
                            right_border.append(node)
                            node = "{}|{}".format(x, int(y)+2)
                    # found right border of colour (not in subgraph)
                    # or reached max size
                    # or the first dimension is 3 and so the second dimension must be <= 3
                    # or the second would be 3 (not possible)
                    if count_north == 1:
                        node_la = "{}|{}".format(x, int(y)+2)
                    if (node not in gr) or (count_north >= max_width - 1) or (bloody_3 and count_north > 0)\
                                        or (count_north == 1 and node_la not in gr and count_south > 2):
                        gr.remove_nodes_from(final_rectangle)
                        all_rectangles.append(final_rectangle)
                        border_found = True
                        break
                    else:
                        right_border.append(node)
                
                if not border_found:
                    final_rectangle += right_border
                    rectangle = right_border
                    if count_north > 2:
                        count_north += 2
                    else:
                        count_north += 1
                    
    return all_rectangles

def get_parts(meh, named_colors):
    measures = []
    plate_colours = []

    for rect in meh:
        color = named_colors[rect[0]]
        
        key_x = lambda x: int(x.split("|")[0])
        key_y = lambda y: int(y.split("|")[1])
        
        min_x = int(min(rect, key=key_x).split("|")[0])
        min_y = int(min(rect, key=key_y).split("|")[1])
        
        max_x = int(max(rect, key=key_x).split("|")[0])
        max_y = int(max(rect, key=key_y).split("|")[1])
        
        measures.append("{} x {}".format(min(max_x-min_x+1, max_y-min_y+1), max(max_x-min_x+1, max_y-min_y+1)))
        plate_colours.append(color)
    
    return measures, plate_colours

def get_part_list():
    headers = {
        'User-Agent': "Mozilla/5.0"
    }
    r = requests.get("https://www.bricklink.com/catalogList.asp?catType=P&catString=26", headers=headers)

    tree = BeautifulSoup(r.text, "lxml")

    html_table = tree.select("#ItemEditForm")[0].select("table")[1]
    part_table = pd.read_html(str(html_table), header=0)[0]
    part_table.drop("Image", axis=1, inplace=True)
    part_table.columns = ["ID", "Description"]
    part_table["Description"] = part_table["Description"].str.split("Cat").str[0].str[6:]
    part_table = part_table[part_table["Description"].str.len() < 10]

    return part_table


def build_xml(measures, plate_colours, part_table):
    order = defaultdict(int)
    xml_string = "<INVENTORY>\n"
    item_string = """<ITEM> 
<ITEMTYPE>P</ITEMTYPE>
<ITEMID>{}</ITEMID>
<COLOR>{}</COLOR>
<MINQTY>{}</MINQTY>
</ITEM>
"""

    for plate, color in zip(measures, plate_colours):
        p_id = part_table[part_table["Description"] == plate]["ID"].values[0]
        order[str(p_id) + "|" + str(color)] += 1
    
    print("Found {} items".format(len(order)))
    for plate, num  in order.items():
        xml_string += item_string.format(*plate.split("|"), num)

    xml_string += "</INVENTORY>"

    return xml_string


def build_instructions(G, meh, pos, print_colors, w, h, ratio, name):
    i = 0
    nodes_draw = []
    edges_draw = list(G.subgraph(meh[-1]).edges())
    chassis_graph = nx.Graph()
    chassis_graph.add_nodes_from([1,2,3,4])

    path_name = "instructions_{}".format(name)

    if os.path.exists(path_name):
        shutil.rmtree(path_name)
    os.mkdir(path_name)

    while i < len(meh):
        nodes_draw.extend(meh[i])
        edges_draw.extend(G.subgraph(meh[i]).edges())
        
        if i == len(meh) - 1:
            plt.figure(figsize=(int(10*ratio),10))

            nx.draw(chassis_graph, pos={1: [w,1], 2: [w,h], 3: [-1,1], 4: [-1,h]}, node_color="red", 
                    node_shape="s", width=0.0)
            nx.draw(G, nodelist=nodes_draw, pos=pos, node_color="black", node_size=200, 
                    edgelist=edges_draw, width=6.0, node_shape="o")
            nx.draw(G, nodelist=nodes_draw, pos=pos, node_color=[print_colors[c] for c in nodes_draw], 
                    node_size=150, edgelist=edges_draw, width=5.0, node_shape="o", 
                    edge_color=[print_colors[c[0]] for c in edges_draw])

            plt.savefig("{}/{}.pdf".format(path_name, i), bbox_inches='tight')

        plt.close("all")
        i += 1

def main(args):
    assert args.maxlength <= 12 and args.maxlength > 0, "Brick length can only be between 1 and 12."
    assert args.maxwidth <= 6 and args.maxwidth > 0, "Brick width can only be between 1 and 6."
    image = Image.open(args.input).convert("RGB")
    name = args.input.split(".")[0]
    current_colours = get_colours()

    print("Using image {}".format(args.input))

    ratio = image.size[0]/image.size[1]

    max_size = args.size

    if ratio < 1:
        h = max_size
        w = int(max_size*ratio)
    else:
        w = max_size
        h = int(max_size/ratio)

    pixelated = image.filter(ImageFilter.MedianFilter(args.smooth)).resize((w,h), resample=0)
    pixelated = np.array(pixelated, dtype=np.uint8)
    if args.lab:
        pixelated = colorcv.rgb2lab(pixelated/255)
        distances = distance_matrix(pixelated.reshape(-1,3), current_colours[["l","a","b2"]].values)
        faster = current_colours.iloc[np.argmin(distances, axis=1), -3:].values.reshape(pixelated.shape)
        pixelated = Image.fromarray(np.uint8(colorcv.lab2rgb(faster)*255), mode="RGB")
    else:
        distances = distance_matrix(pixelated.reshape(-1,3), current_colours[["r","g","b"]].values)
        faster = current_colours.iloc[np.argmin(distances, axis=1), -6:-3].values.reshape(pixelated.shape)
        pixelated = Image.fromarray(np.uint8(faster), mode="RGB")
        
    final = pixelated
    final_arr = colorcv.rgb2lab(np.array(final)/255)

    preview =  "{}_preview.png".format(name)
    final.resize(image.size, resample=0).save(preview)
    print("Saved preview under {}".format(preview))

    new_p = np.zeros((final.size))
    G = nx.Graph()
    pos = {}
    colors = {}
    print_colors = {}
    named_colors = {}
    i = 0
    color_transpose = {11: np.zeros(3), 63: np.array([0.0745098 , 0.18823529, 0.36666667]),
                       88: np.array([0.588235, 0.235294, 0]), 71: np.array([0.7098 , 0.18039, 0.396])}
    critical_colors = set([])

    for x in range(final.size[1]):
        for y in range(final.size[0]):
            G.add_node(str(x)+"|"+str(y), x=x, y=y)
            colors[str(x)+"|"+str(y)] = final_arr[x,y]
            named_color = current_colours. \
                          loc[np.all(np.isclose(current_colours[["l","a","b2"]], final_arr[x,y], atol=1), 
                                     axis=1),
                             "ID"].values[0]
            named_colors[str(x)+"|"+str(y)] = named_color

            if named_color in color_transpose.keys():
                print_color = color_transpose[named_color]
            else:
                print_color = colorcv.lab2rgb(final_arr[x,y].reshape(1,1,3)).flatten()
            print_colors[str(x)+"|"+str(y)] = print_color

            pos[str(x)+"|"+str(y)] = [y,h-x]
            i += 1

            if x!=0 and np.array_equal(final_arr[x,y], final_arr[x-1,y]):
                G.add_edge(str(x)+"|"+str(y), str(x-1)+"|"+str(y))

            if y!=0 and np.array_equal(final_arr[x,y], final_arr[x,y-1]):
                G.add_edge(str(x)+"|"+str(y), str(x)+"|"+str(y-1))

    meh = find_rectangles(G, named_colors=named_colors, maxim_size=args.maxlength, 
                          maxim_width=args.maxwidth, critical_colors=critical_colors)

    part_table = get_part_list()
    measures, plate_colours = get_parts(meh, named_colors)

    xml = build_xml(measures, plate_colours, part_table)
    with open(args.output, "w") as xml_file:
        xml_file.write(xml)

    print("Saved xml with {} parts to {}".format(len(meh), args.output))

    build_instructions(G, meh, pos, print_colors, w, h, ratio, name)
    print("Saved instructions to 'instructions_{}/'".format(name))

if __name__ == '__main__':
    parser = ArgumentParser(description='Build your image with bricks.')

    parser.add_argument("-i", "--input", type=str, help="input image", required=True)

    parser.add_argument("-o", "--output", type=str, help="output xml file", required=True)

    parser.add_argument("-sm", "--smooth", type=int, 
        help="Smoothing factor for prefiltering. Increase for removing artifacts. Can only be odd. Defaults to 1.", default=1)

    parser.add_argument("-s", "--size", type=int, help="Max size for the output image in pixels/studs. Defaults to 32.", default=32)

    parser.add_argument("-ml", "--maxlength", type=int, help="Max length of an individual LEGO plate in studs. Defaults to 12.", default=12)

    parser.add_argument("-mw", "--maxwidth", type=int, help="Max width of an individual LEGO plate in studs. Defaults to 6.", default=6)

    parser.add_argument("-lab", type=int, help="If not zero, LAB is used for distances between pixels, otherwise RGB. Defaults to 1.", default=1)

    args = parser.parse_args()

    main(args)
