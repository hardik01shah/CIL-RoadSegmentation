import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.morphology import opening

def remove_isolated_pixels(binary_map, kernel_size=4):
    """
    Removes isolated pixels using morphological opening.

    Args:
        binary_map: A numpy array representing the binary occupancy map.
        kernel_size: Size of the structuring element for opening operation.

    Returns:
        A new numpy array with isolated pixels removed.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_map = opening(binary_map, kernel)
    return opened_map

def close_small_holes(binary_map, kernel_size):
  """
  Closes small holes in the binary map using morphological closing.

  Args:
      binary_map: A numpy array representing the binary occupancy map.
      kernel_size: Size of the structuring element for closing operation.

  Returns:
      A new numpy array with small holes in walls closed.
  """
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  closed_map = cv2.morphologyEx(binary_map.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
  return closed_map.astype(np.uint8)

def remove_small_artifacts(binary_map, threshold):
    """
    Removes artifacts smaller than a threshold size from a binary map.

    Args:
        binary_map: A numpy array representing the binary occupancy map.
        threshold: Minimum size (area) of an object to be considered valid.

    Returns:
        A new numpy array with artifacts removed.
    """

    # Get statistics for each component
    output = cv2.connectedComponentsWithStats(binary_map.astype(np.uint8), connectivity=8)
    areas = output[2][:, cv2.CC_STAT_AREA] # Get the areas of all connected components
    labels = output[1] # Get the labels of all connected components

    # Create a mask to keep valid objects
    mask = np.ones(binary_map.shape, np.uint8)
    for i in range(1, len(areas)):
        if areas[i] < threshold:
            mask[labels == i] = 0

    # Apply mask to remove artifacts
    filtered_map = cv2.bitwise_and(binary_map.astype(np.uint8), mask)
    return filtered_map.astype(np.uint8)