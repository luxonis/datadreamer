from __future__ import annotations

from typing import Type, Union

import psutil
import pytest
import torch
from PIL import Image

from datadreamer.image_generation import (
    StableDiffusionImageGenerator,
    StableDiffusionLightningImageGenerator,
    StableDiffusionTurboImageGenerator,
)

# Get the total memory in GB
total_memory = psutil.virtual_memory().total / (1024**3)
# Get the total disk space in GB
total_disk_space = psutil.disk_usage("/").total / (1024**3)


def _check_image_generator(
    image_generator_class: Type[
        Union[
            StableDiffusionImageGenerator,
            StableDiffusionTurboImageGenerator,
            StableDiffusionLightningImageGenerator,
        ]
    ],
    device: str,
):
    image_generator = image_generator_class(device=device)
    # Check that the image generator is not None
    assert image_generator is not None
    # Generate images and check each of them
    for generated_images_batch in image_generator.generate_images(
        ["A photo of a cat, dog"], [["cat", "dog"]]
    ):
        generated_image = generated_images_batch[0]
        assert generated_image is not None
        assert isinstance(generated_image, Image.Image)

    images = image_generator.generate_images_batch(
        ["A photo of a cat, dog"],
        "blurry, bad quality",
    )
    assert len(images) == 1
    assert images[0] is not None
    assert isinstance(images[0], Image.Image)

    # Release the generator
    image_generator.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 25,
    reason="Test requires GPU, at least 16GB of RAM and 25GB of HDD",
)
def test_cuda_sdxl_image_generator():
    _check_image_generator(StableDiffusionImageGenerator, "cuda")


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 25,
    reason="Test requires at least 16GB of RAM and 25GB of HDD",
)
def test_cpu_sdxl_image_generator():
    _check_image_generator(StableDiffusionImageGenerator, "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 25,
    reason="Test requires GPU, at least 16GB of RAM and 25GB of HDD",
)
def test_cuda_sdxl_lightning_image_generator():
    _check_image_generator(StableDiffusionLightningImageGenerator, "cuda")


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 25,
    reason="Test requires at least 16GB of RAM and 25GB of HDD",
)
def test_cpu_sdxl_lightning_image_generator():
    _check_image_generator(StableDiffusionLightningImageGenerator, "cpu")
