import dgl
from dgl.data.utils import save_graphs
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm

edge_size = 0
node_size = 0


def format_dgl_edges(edge_file, dgl_file):
    node_maps = pkl.load(open(dgl_file + ".nodes.dgl", 'rb'))['maps']
    edges = {}
    edges_hg = {}

    process = tqdm(total=edge_size)
    count = 0
    with open(edge_file, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            line_info = line.strip().split(",")
            source_id, dest_id, source_type, dest_type, edge_type = line_info
            source_id = node_maps[source_type][int(source_id)]
            dest_id = node_maps[dest_type][int(dest_id)]
            edges.setdefault(edge_type, {})
            edges[edge_type].setdefault('source', []).append(source_id)
            edges[edge_type].setdefault('dest', []).append(dest_id)
            edges[edge_type].setdefault('source_type', source_type)
            edges[edge_type].setdefault('dest_type', dest_type)
            count += 1
            if count % 100000 == 0:
                process.update(100000)
    process.close()

    for edge_type in edges:
        source_type = edges[edge_type]['source_type']
        dest_type = edges[edge_type]['dest_type']
        source = edges[edge_type]['source']
        dest = edges[edge_type]['dest']
        edges_hg[(source_type, edge_type, dest_type)] = list(zip(source, dest))

    hg = dgl.heterograph(edges_hg)
    print('Start saving dgl-style graph structure\n')
    save_graphs(dgl_file + ".graph.dgl", hg)
    print('Complete saving dgl-style graph structure\n')


def store_node_atts(node_file, label_file, dgl_file):
    node_maps = {}
    node_embeds = {}
    count = 0
    count2 = 0
    node_counts = node_size
    process = tqdm(total=node_counts)
    with open(node_file, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            info = line.strip().split(",")

            node_id = int(info[0])
            node_type = info[1].strip()

            # node_maps[node_type]
            # node_maps[node_type]

            node_maps.setdefault(node_type, {})
            node_id_v2 = len(node_maps[node_type])
            node_maps[node_type][node_id] = node_id_v2
            if node_type == 'item' and len(info[2]) == 0:
                node_embeds.setdefault(node_type, {})
                node_embeds[node_type][node_id_v2] = np.zeros(128, dtype=np.float32)
                count2 += 1
            elif len(info) == 3 and len(info[2]) > 0:
                node_embeds.setdefault(node_type, {})
                node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
                #
                #
                # if len(node_embeds[node_type]) == node_id_v2:
                #     print(node_id_v2, len(node_embeds[node_type]))
                #     node_emb = list(np.array(info[2].split(":"), np.float32))
                #     node_embeds[node_type].append(node_emb)
                # print(len(node_embeds[node_type]), node_id_v2, node_embeds[node_type][node_id_v2])
            count += 1
            if count % 100000 == 0:
                process.update(100000)
                # print('node size:', count)
    process.close()

    print('lack of features:', count2, count, len(node_maps['item']))
    print('node_types', node_maps.keys())
    for node_type in node_maps:
        print('node_type', node_type, len(node_maps[node_type]))
    print(len(node_embeds['item']))
    # exit()

    nodes_dict = {'maps': node_maps, 'embeds': node_embeds}
    nodes_dict['labels'] = {}

    if label_file is not None:
        labels_info = [x.strip().split(",") for x in open(label_file).readlines()]
        labels = []
        for i in range(len(labels_info)):
            x = labels_info[i]
            item_id = node_maps['item'][int(x[0])]
            label = int(x[1])
            labels.append([item_id, label])
        nodes_dict['labels']['item'] = labels
    print('Start saving dgl-style node information\n')
    pkl.dump(nodes_dict, open(dgl_file + ".nodes.dgl", 'wb'))
    print('Complete saving pkl-style node information\n')
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--node', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--storefile', type=str, default=None)
    args = parser.parse_args()
    if "session2" in args.storefile:
        edge_size = 120691444
        node_size = 10284026
    else:
        edge_size = 157814864
        node_size = 13806619

    if args.graph is not None and args.storefile is not None and args.node is not None:  # and args.label is not None:
        store_node_atts(args.node, args.label, args.storefile)
        format_dgl_edges(args.graph, args.storefile)
