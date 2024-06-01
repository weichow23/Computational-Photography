import numpy as np
import maxflow
from termcolor import cprint
def abswap(composite, source, composite_mask, source_mask):
    '''
    alpha-beta swap maxflow for the current composite and source
    '''
    h, w, _ = composite.shape
    graph = maxflow.Graph[int](h*w, 2*((h-1)*w+(w-1)*h))
    nodeids = graph.add_grid_nodes((h, w))
    color_diff_x = composite[:, :-1] - source[:, 1:]
    color_diff_x = np.sum(np.abs(color_diff_x), -1)
    color_diff_y = composite[:-1, :] - source[1:, :]
    color_diff_y = np.sum(np.abs(color_diff_y), -1)

    # add edges for horizontally adjacent pixels (n-links)
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])
    graph.add_grid_edges(nodeids[:, :-1], color_diff_x, structure, symmetric=True)

    # add edges for vertically adjacent pixels (n-links)
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])
    graph.add_grid_edges(nodeids[:-1, :], color_diff_y, structure, symmetric=True)

    # add terminal edges (t-links)
    # note that the alpha and beta weights are reversed
    # since a pixel has the label of the segment it is not in
    # alpha is 0 (label for composite)
    alpha_weight = source_mask.astype(np.int64) * 100000000
    # beta is 1 (label for source)
    beta_weight = composite_mask.astype(np.int64) * 100000000

    graph.add_grid_tedges(nodeids, alpha_weight, beta_weight)

    # since there are only two labels, 1 iteration is enough
    graph.maxflow()
    sgm = graph.get_grid_segments(nodeids)
    label_map = np.logical_not(sgm).astype(np.uint8)

    return label_map

