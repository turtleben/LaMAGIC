import networkx as nx
import matplotlib.pyplot as plt
from networkx import path_graph


def visualize_paths(paths, filename):
    path_num = len(paths)
    plt.figure()
    for path_idx, path in enumerate(paths):
        nodes = path.split(' - ')
        size = len(nodes)

        G = path_graph(len(nodes))

        pos = [(idx, path_num - path_idx) for idx in range(size)]
        labels = {idx: nodes[idx] for idx in range(size)}
        label_pos = {idx: (idx, path_num - path_idx + .3) for idx in range(size)}

        nx.draw(G, pos=pos)
        nx.draw_networkx_labels(G, label_pos, labels)

    plt.xlim(-0.5, 3.5)
    plt.ylim(0, len(paths) + 1)
    plt.savefig(filename + ".png", dpi=300, format="png")

if __name__ == '__main__':
    visualize_paths(['VIN - inductor - VOUT', 'VIN - inductor - VOUT'])