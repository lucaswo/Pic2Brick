import requests
from bs4 import BeautifulSoup
import pandas as pd
from PIL import Image, ImageFilter, ImageCms
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import shutil
from skimage import color as colorcv

import warnings
warnings.filterwarnings("ignore")

def hextoint(hexcode):
    return (int(hexcode[:2], 16), int(hexcode[2:4], 16), int(hexcode[4:], 16))

def my_lab_metric(x, y):
    x = x / 255
    y = y / 255

    both = np.vstack((x,y)).reshape(1,-1,3)

    x,y = colorcv.rgb2lab(both)[0]

    return np.linalg.norm(x-y)

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
    current_colours = colour_table[colour_table["Color Timeline"].str.contains("2018")]
    current_colours = current_colours[~(current_colours["Name"].str.contains("Flesh") 
                                        | current_colours["Name"].str.contains("Dark Pink")
                                        | (current_colours["Name"] == "Lavender")
                                        | current_colours["Name"].str.contains("Sand Blue")
                                        | current_colours["Name"].str.contains("Olive Green")
                                        | current_colours["Name"].str.contains("Light Yellow")
                                        | (current_colours["Name"] == "Lime"))]

    return current_colours

def fit_NN(current_colours, rgb):
    if rgb:
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric="euclidean")
    else:
        nn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric=my_lab_metric)
    nn.fit(current_colours[["r", "g", "b"]])

    return nn

def legofy(pixel, nn, colours):
    new_pixel = nn.kneighbors(pixel.reshape(1, -1), return_distance=False)[0][0]
    return tuple(colours.iloc[new_pixel, -3:])

def find_rectangles(G, max_size=8):
    assert max_size % 2 == 0 or max_size < 4, "Even numbers for max_size only!"
    bad_dims = set([5, 7, 9, 11, 13])
    all_rectangles = []
    
    # all connected subgraphs
    for gr in nx.algorithms.connected_component_subgraphs(G):
        while list(gr.nodes()): # as long as there are nodes
            node = min(gr.nodes(), key=lambda x: [int(y) for y in x.split("|")]) # get min x,y
            rectangle = []
            count = 0

            # vertical edge (expand southwards) until bottom line of subgraph
            # or max size
            while node in gr and count <= max_size - 1:
                rectangle.append(node)
                x,y = node.split("|")
                node = "{}|{}".format(int(x)+1, y)
                # widths > 4 are: 6, 8, 10, 12, 14
                if count > 2:
                    if node in gr and count + 1 <= max_size - 1:
                        rectangle.append(node)
                        node = "{}|{}".format(int(x)+2, y)
                        count += 1
                count += 1
            
            # 5 and 3 are stupid numbers for LEGO plates, 
            # 3 can only occur in one dimension,
            # 5 can never occur anywhere
            if count in bad_dims:
                rectangle.pop(-1)
            if count == 3:
                bloody_3 = True
            else:
                bloody_3 = False
                
            final_rectangle = rectangle
            
            border_found = False
            count = 0
            
            # horizontal edge (expand eastwards)
            while (not border_found):
                right_border = []
                for rect_node in rectangle:
                    x,y = rect_node.split("|")
                    
                    node = "{}|{}".format(x, int(y)+1)
                    # widths > 4 are: 6, 8, 10, 12  
                    if count > 2:
                        if node in gr and count + 1 <= max_size - 1:
                            right_border.append(node)
                            node = "{}|{}".format(x, int(y)+2)
                            count += 1
                    # found right border of colour (not in subgraph)
                    # or reached max size
                    # or the first dimension is 3 and so the second dimension must be <= 3
                    # or the second would be 3 (not possible)
                    if count == 1:
                        node_la = "{}|{}".format(x, int(y)+2)
                    if (node not in gr) or (count >= max_size - 1) or (bloody_3 and count > 0) or (count == 1 and node_la not in gr):
                        gr.remove_nodes_from(final_rectangle)
                        all_rectangles.append(final_rectangle)
                        border_found = True
                        break
                    else:
                        right_border.append(node)
                
                if not border_found:
                    final_rectangle += right_border
                    rectangle = right_border
                    count += 1
                    
    return all_rectangles

def get_parts(meh, nn, colors, current_colours):
    measures = []
    plate_colours = []

    for rect in meh:
        
        row = nn.kneighbors(np.array(colors[rect[0]]*255).reshape(1, -1), return_distance=False)[0][0]
        color = current_colours.iloc[row, 0]
        
        key_x = lambda x: int(x.split("|")[0])
        key_y = lambda y: int(y.split("|")[1])
        
        min_x = int(min(rect, key=key_x).split("|")[0])
        min_y = int(min(rect, key=key_y).split("|")[1])
        
        max_x = int(max(rect, key=key_x).split("|")[0])
        max_y = int(max(rect, key=key_y).split("|")[1])
        
        measures.append("{} x {}".format(min(max_x-min_x+1, max_y-min_y+1), max(max_x-min_x+1, max_y-min_y+1)))
        plate_colours.append(color)
    
    return(measures, plate_colours)

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
        
    for plate, num  in order.items():
        xml_string += item_string.format(*plate.split("|"), num)

    xml_string += "</INVENTORY>"

    return xml_string


def build_instructions(G, meh, pos, colors, w, h, ratio, name):
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

        plt.figure(figsize=(9*ratio,9))

        nx.draw(chassis_graph, pos={1: [w,1], 2: [w,h], 3: [-1,1], 4: [-1,h]}, node_color="red", node_size=100,
            node_shape="s")
        nx.draw(G, nodelist=nodes_draw, pos=pos, node_color=[colors[c] for c in nodes_draw], node_size=100,
            edgelist=edges_draw, width=3.0)
        plt.savefig("{}/{}.png".format(path_name,i+1))

        plt.close()
        i += 1

def main(args):
    image = Image.open(args.input)
    name = args.input.split(".")[0]
    current_colours = get_colours()
    nn = fit_NN(current_colours, args.rgb)

    w10 = int(image.size[0]/10)
    h10 = int(image.size[1]/10)
    ratio = image.size[0]/image.size[1]

    print("Using image {}".format(args.input))

    pixelated = image.filter(ImageFilter.MedianFilter(args.smooth)).resize((2*w10,2*h10))

    pixelated = np.array(pixelated)
    pixelated = np.apply_along_axis(legofy, 2, pixelated, nn, current_colours)
    pixelated = Image.fromarray(np.uint8(pixelated), mode="RGB")

    max_size = args.size

    if ratio < 1:
        h = max_size
        w = int(max_size*ratio)
    else:
        w = max_size
        h = int(max_size/ratio)
        
    final = pixelated.resize((w,h))
    final_arr = np.array(final)

    preview =  "{}_preview.png".format(name)
    final.resize(image.size).save(preview)
    print("Saved preview under {}".format(preview))

    new_p = np.zeros((final.size))
    G = nx.Graph()
    pos = {}
    colors = {}
    i = 0

    for x in range(final.size[1]):
        for y in range(final.size[0]):
            G.add_node(str(x)+"|"+str(y), x=x, y=y)
            colors[str(x)+"|"+str(y)] = final_arr[x,y]/255
            pos[str(x)+"|"+str(y)] = [y,h-x]
            i += 1
            
            if x!=0 and np.array_equal(final_arr[x,y], final_arr[x-1,y]):
                G.add_edge(str(x)+"|"+str(y), str(x-1)+"|"+str(y))
                
            if y!=0 and np.array_equal(final_arr[x,y], final_arr[x,y-1]):
                G.add_edge(str(x)+"|"+str(y), str(x)+"|"+str(y-1))

    meh = find_rectangles(G, max_size=args.maxsize)

    part_table = get_part_list()
    measures, plate_colours = get_parts(meh, nn, colors, current_colours)

    xml = build_xml(measures, plate_colours, part_table)
    with open(args.output, "w") as xml_file:
        xml_file.write(xml)

    print("Saved xml to {}".format(args.output))

    build_instructions(G, meh, pos, colors, w, h, ratio, name)
    print("Saved instructions to 'instructions_{}/'".format(name))

if __name__ == '__main__':
    parser = ArgumentParser(description='Legofy your image.')

    parser.add_argument("-i", "--input", type=str, help="input image", required=True)

    parser.add_argument("-o", "--output", type=str, help="output xml file", required=True)

    parser.add_argument("-sm", "--smooth", type=int, 
        help="Smoothing factor for prefiltering. Increase for removing artifacts. Can only be odd. Defaults to 1.", default=1)

    parser.add_argument("-s", "--size", type=int, help="Max size for the output image in pixels/studs. Defaults to 32.", default=32)

    parser.add_argument("-ms", "--maxsize", type=int, help="Max size of an individual LEGO plate in studs. Defaults to 12.", default=12)

    parser.add_argument("-rgb", type=int, help="If not zero, RGB is used for distances between pixels, otherwise LAB. Defaults to 1.", default=1)

    args = parser.parse_args()

    main(args)
