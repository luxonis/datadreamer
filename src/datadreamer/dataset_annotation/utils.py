from torchvision import transforms


def apply_tta(image):
    """Apply test-time augmentation (TTA) to the given image.

    Args:
        image: The image to be augmented.

    Returns:
        list: A list of augmented images, including the original and transformed versions.

    Note:
        Currently, only horizontal flip is enabled. Additional transformations like
        vertical flip and color jitter are commented out but can be enabled as needed.
    """
    tta_transforms = [
        # Original image
        transforms.Compose([]),
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
