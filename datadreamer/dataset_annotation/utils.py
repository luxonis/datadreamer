from __future__ import annotations

from typing import List

import cv2
import numpy as np
from torchvision import transforms


def apply_tta(image) -> List[transforms.Compose]:
    """Apply test-time augmentation (TTA) to the given image.

    Args:
        image: The image to be augmented.

    Returns:
        list: A list of augmented images.

    Note:
        Currently, only horizontal flip is enabled. Additional transformations like
        vertical flip and color jitter are commented out but can be enabled as needed.
    """
    tta_transforms = [
        # Original image
        # transforms.Compose([]),
        # Horizontal Flip
        transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),
        # Vertical Flip
        # transforms.Compose([transforms.RandomVerticalFlip(p=1)]),
        # Color Jitter
        # transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]),
        # Add more TTA transformations if needed
    ]

    augmented_images = [t(image) for t in tta_transforms]
    return augmented_images


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """Converts a binary mask to a polygon.

    Args:
        mask: The binary mask to be converted.

    Returns:
        List: A list of vertices of the polygon.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return []
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon
