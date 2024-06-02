import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from os import path
from PIL import Image

def laplacian_matrix(n, m):
    """
    Generate the Poisson matrix.
    """
    # Create the main diagonal
    main_diag = 4 * sp.eye(m, format='lil')

    # Create the diagonals for the -1s within each block
    off_diag_1 = -1 * sp.eye(m, k=1, format='lil')
    off_diag_2 = -1 * sp.eye(m, k=-1, format='lil')

    # Combine the diagonals within each block
    mat_D = main_diag + off_diag_1 + off_diag_2

    # Create the block diagonal matrix for all blocks
    mat_A = sp.block_diag([mat_D] * n, format='lil')

    # Add the off-diagonal blocks
    for i in range(n - 1):
        mat_A[i * m:(i + 1) * m, (i + 1) * m:(i + 2) * m] = -1 * sp.eye(m, format='lil')
        mat_A[(i + 1) * m:(i + 2) * m, i * m:(i + 1) * m] = -1 * sp.eye(m, format='lil')

    return mat_A

def poisson_edit(source, target, mask, offset):
    """
    poisson blending function.
    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0
    x_range = x_max - x_min
    y_range = y_max - y_min

    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    source = cv2.warpAffine(source, M, (x_range, y_range))

    mask = mask[y_min:y_max, x_min:x_max]
    mask = (mask != 0).astype(np.uint8)
    mat_A = laplacian_matrix(y_range, x_range).tolil()
    laplacian = mat_A.tocsc() # for \Delta g

    # set the region outside the mask to identity
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()
        mat_b = laplacian.dot(source_flat) * 1 # alpha = 1
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))

        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        target[y_min:y_max, x_min:x_max, channel] = x

    return target


def poisson_edit_from_path(source, target, mask=None, offset=(0,0)):
    '''
    test poisson edit function
    '''
    source = np.array(Image.open(source).convert('RGB'))
    target = np.array(Image.open(target).convert('RGB'))
    if mask is None:
        # 上半部分为1，下半部分为0. 最终结果为上半部分为source的图像，下半部分为target的图像
        mask = np.ones_like(source[:, :, 0])
        mask[source.shape[0] // 2:, :] = 0
    else:
        mask = np.array(Image.open(mask).convert('L'))
    result = poisson_edit(source, target, mask, offset)
    return result

if __name__ == '__main__':
    scr_dir = 'source_imgs'
    res = poisson_edit_from_path(path.join(scr_dir, "famille1.jpg"),
                                 path.join(scr_dir, "famille2.jpg"))
    Image.fromarray(res).save('test.jpg')