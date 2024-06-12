import numpy as np
cimport numpy as np
import maxflow
from PIL import Image
from montage.poisson cimport poisson_edit

def alpha_beta_swap(np.ndarray[np.uint8_t, ndim=3] composite,
                    np.ndarray[np.uint8_t, ndim=3] source,
                    np.ndarray[np.uint8_t, ndim=2] composite_mask,
                    np.ndarray[np.uint8_t, ndim=2] source_mask):
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
    cdef int h, w
    cdef int scale = 100000000

    h, w, _ = composite.shape

    # Initialize the maxflow graph with sufficient space for nodes and edges
    cdef maxflow.Graph[int] graph = maxflow.Graph[int](h * w, 2 * ((h - 1) * w + (w - 1) * h))
    cdef np.ndarray[np.int32_t, ndim=2] nodeids = graph.add_grid_nodes((h, w))

    # Calculate the color differences between adjacent pixels for n-links
    cdef np.ndarray[np.uint8_t, ndim=2] color_diff_x = np.sum(np.abs(composite[:, :-1] - source[:, 1:]), axis=-1)
    cdef np.ndarray[np.uint8_t, ndim=2] color_diff_y = np.sum(np.abs(composite[:-1, :] - source[1:, :]), axis=-1)

    # Add edges for horizontally adjacent pixels
    cdef np.ndarray[np.int32_t, ndim=2] horizontal_structure = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    graph.add_grid_edges(nodeids[:, :-1], color_diff_x, horizontal_structure, symmetric=True)

    # Add edges for vertically adjacent pixels
    cdef np.ndarray[np.int32_t, ndim=2] vertical_structure = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    graph.add_grid_edges(nodeids[:-1, :], color_diff_y, vertical_structure, symmetric=True)

    # Add terminal edges
    # Alpha is 0 (label for composite), beta is 1 (label for source)
    cdef np.ndarray[np.int64_t, ndim=2] alpha_weight = source_mask.astype(np.int64) * scale
    cdef np.ndarray[np.int64_t, ndim=2] beta_weight = composite_mask.astype(np.int64) * scale
    graph.add_grid_tedges(nodeids, alpha_weight, beta_weight)

    # Perform maxflow to get the label map
    graph.maxflow()
    cdef np.ndarray[np.int32_t, ndim=2] sgm = graph.get_grid_segments(nodeids)

    # Convert the segments to a binary label map
    cdef np.ndarray[np.uint8_t, ndim=2] label_map = np.logical_not(sgm).astype(np.uint8)

    return label_map

def create_composite(np.ndarray[np.uint8_t, ndim=2] binary_map,
                     np.ndarray[np.uint8_t, ndim=3] source,
                     np.ndarray[np.uint8_t, ndim=3] target):
    cdef np.ndarray[np.uint8_t, ndim=2] mask = binary_map.astype(np.uint8) * 255
    return Image.fromarray(poisson_edit(source, target, mask, (0, 0)))