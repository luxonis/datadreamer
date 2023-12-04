from torchvision import transforms


def apply_tta(image):
    tta_transforms = [
        # Original image
        transforms.Compose([]),
        
        # Horizontal Flip
        transforms.Compose([transforms.RandomHorizontalFlip(p=1)]),

        # Vertical Flip
        # transforms.Compose([transforms.RandomVerticalFlip(p=1)]),

        # Color Jitter
        #transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]),

        # Add more TTA transformations if needed
    ]

    augmented_images = [t(image) for t in tta_transforms]
    return augmented_images