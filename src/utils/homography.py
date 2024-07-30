"""
Source: https://github.com/cvg/glue-factory
"""

import math
import numpy as np


def flat2mat(H):
    return np.reshape(np.concatenate([H, np.ones_like(H[:, :1])], axis=1), [3, 3])


# Homography creation
def create_center_patch(shape, patch_shape=None):
    if patch_shape is None:
        patch_shape = shape
    width, height = shape
    pwidth, pheight = patch_shape
    left = int((width - pwidth) / 2)
    bottom = int((height - pheight) / 2)
    right = int((width + pwidth) / 2)
    top = int((height + pheight) / 2)
    return np.array([[left, bottom], [left, top], [right, top], [right, bottom]])


def check_convex(patch, min_convexity=0.05):
    """Checks if given polygon vertices [N,2] form a convex shape"""
    for i in range(patch.shape[0]):
        x1, y1 = patch[(i - 1) % patch.shape[0]]
        x2, y2 = patch[i]
        x3, y3 = patch[(i + 1) % patch.shape[0]]
        if (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1) > -min_convexity:
            return False
    return True


def sample_homography_corners(
    shape,
    patch_shape,
    difficulty=1.0,
    translation=0.4,
    n_angles=10,
    max_angle=90,
    min_convexity=0.05,
    rng=np.random,
):
    max_angle = max_angle / 180.0 * math.pi
    width, height = shape
    pwidth, pheight = width * (1 - difficulty), height * (1 - difficulty)
    min_pts1 = create_center_patch(shape, (pwidth, pheight))
    full = create_center_patch(shape)
    pts2 = create_center_patch(patch_shape)
    scale = min_pts1 - full
    found_valid = False
    cnt = -1
    while not found_valid:
        offsets = rng.uniform(0.0, 1.0, size=(4, 2)) * scale
        pts1 = full + offsets
        found_valid = check_convex(pts1 / np.array(shape), min_convexity)
        cnt += 1

    # re-center
    pts1 = pts1 - np.mean(pts1, axis=0, keepdims=True)
    pts1 = pts1 + np.mean(min_pts1, axis=0, keepdims=True)

    # Rotation
    if n_angles > 0 and difficulty > 0:
        angles = np.linspace(-max_angle * difficulty, max_angle * difficulty, n_angles)
        rng.shuffle(angles)
        rng.shuffle(angles)
        angles = np.concatenate([[0.0], angles], axis=0)

        center = np.mean(pts1, axis=0, keepdims=True)
        rot_mat = np.reshape(
            np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],
                axis=1,
            ),
            [-1, 2, 2],
        )
        rotated = (
            np.matmul(
                np.tile(np.expand_dims(pts1 - center, axis=0), [n_angles + 1, 1, 1]),
                rot_mat,
            )
            + center
        )

        for idx in range(1, n_angles):
            warped_points = rotated[idx] / np.array(shape)
            if np.all((warped_points >= 0.0) & (warped_points < 1.0)):
                pts1 = rotated[idx]
                break

    # Translation
    if translation > 0:
        min_trans = -np.min(pts1, axis=0)
        max_trans = shape - np.max(pts1, axis=0)
        trans = rng.uniform(min_trans, max_trans)[None]
        pts1 += trans * translation * difficulty

    H = compute_homography(pts1, pts2, [1.0, 1.0])
    warped = warp_points(full, H, inverse=False)
    return H, full, warped, patch_shape


def compute_homography(pts1_, pts2_, shape):
    """Compute the homography matrix from 4 point correspondences"""
    # Rescale to actual size
    shape = np.array(shape[::-1], dtype=np.float32)  # different convention [y, x]
    pts1 = pts1_ * np.expand_dims(shape, axis=0)
    pts2 = pts2_ * np.expand_dims(shape, axis=0)

    def ax(p, q):
        return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q):
        return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = np.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = np.transpose(
        np.stack([[pts2[i][j] for i in range(4) for j in range(2)]], axis=0)
    )
    homography = np.transpose(np.linalg.solve(a_mat, p_mat))
    return flat2mat(homography)


def warp_points(points, homography, inverse=True):
    """
    Warp a list of points with the INVERSE of the given homography.
    The inverse is used to be coherent with tf.contrib.image.transform
    Arguments:
        points: list of N points, shape (N, 2).
        homography: batched or not (shapes (B, 3, 3) and (3, 3) respectively).
    Returns: a Tensor of shape (N, 2) or (B, N, 2) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.
    """
    H = homography[None] if len(homography.shape) == 2 else homography

    # Get the points to the homogeneous format
    num_points = points.shape[0]
    # points = points.astype(np.float32)[:, ::-1]
    points = np.concatenate([points, np.ones([num_points, 1], dtype=np.float32)], -1)

    H_inv = np.transpose(np.linalg.inv(H) if inverse else H)
    warped_points = np.tensordot(points, H_inv, axes=[[1], [0]])

    warped_points = np.transpose(warped_points, [2, 0, 1])
    warped_points[np.abs(warped_points[:, :, 2]) < 1e-8, 2] = 1e-8
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]

    return warped_points[0] if len(homography.shape) == 2 else warped_points