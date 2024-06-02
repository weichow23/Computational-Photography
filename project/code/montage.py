import numpy as np
import maxflow
from PIL import Image
from poisson import poisson_edit

def alpha_beta_swap(composite, source, composite_mask, source_mask):
    '''
    Perform alpha-beta swap maxflow for the current composite and source images.

    Args:
        composite (np.ndarray): The composite image.
        source (np.ndarray): The source image.
        composite_mask (np.ndarray): The mask for the composite image.
        source_mask (np.ndarray): The mask for the source image.

    Returns:
        np.ndarray: The label map after the alpha-beta swap.
    '''
    h, w, _ = composite.shape
    scale = 100000000

    # Initialize the maxflow graph with sufficient space for nodes and edges
    graph = maxflow.Graph[int](h * w, 2 * ((h - 1) * w + (w - 1) * h))
    nodeids = graph.add_grid_nodes((h, w))

    # Calculate the color differences between adjacent pixels for n-links
    color_diff_x = np.sum(np.abs(composite[:, :-1] - source[:, 1:]), axis=-1)
    color_diff_y = np.sum(np.abs(composite[:-1, :] - source[1:, :]), axis=-1)

    # Add edges for horizontally adjacent pixels
    horizontal_structure = np.array([[0, 0, 0],
                                     [0, 0, 1],
                                     [0, 0, 0]])
    graph.add_grid_edges(nodeids[:, :-1], color_diff_x, horizontal_structure, symmetric=True)

    # Add edges for vertically adjacent pixels
    vertical_structure = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 1, 0]])
    graph.add_grid_edges(nodeids[:-1, :], color_diff_y, vertical_structure, symmetric=True)

    # Add terminal edges
    # Alpha is 0 (label for composite), beta is 1 (label for source)
    alpha_weight = source_mask.astype(np.int64) * scale
    beta_weight = composite_mask.astype(np.int64) * scale
    graph.add_grid_tedges(nodeids, alpha_weight, beta_weight)

    # Perform maxflow to get the label map
    graph.maxflow()
    sgm = graph.get_grid_segments(nodeids)

    # Convert the segments to a binary label map
    label_map = np.logical_not(sgm).astype(np.uint8)

    return label_map

def create_composite(binary_map, source: np.ndarray, target: np.ndarray):
    mask = binary_map.astype(np.uint8) * 255
    return Image.fromarray(poisson_edit(source, target, mask, (0, 0)))
