from __future__ import annotations

from typing import List

import cv2
import numpy as np
import pycocotools.mask as mask_utils
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


def convert_binary_mask(
    mask: np.ndarray, mask_format: str = "polygon", epsilon_ratio: float = 0.001
) -> List[List[int]]:
    mask = cv2.medianBlur(mask, 5)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
    mask = (blurred > 0.5).astype(np.uint8)

    if mask_format == "polyline":
        return mask_to_polygon(mask, epsilon_ratio)
    elif mask_format == "rle":
        return mask_to_rle(mask)


def mask_to_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.array(mask, dtype=np.uint8, order="F"))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.001) -> List[List[int]]:
    """Converts a binary mask to a smoothed polygon.

    Args:
        mask: The binary mask to be converted.
        epsilon_ratio: Controls the smoothing level. Higher values mean more smoothing.

    Returns:
        List: A list of vertices of the polygon.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return []

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Apply contour approximation for smoothing
    epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
    smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    polygon = smoothed_contour.reshape(-1, 2).tolist()

    return polygon
