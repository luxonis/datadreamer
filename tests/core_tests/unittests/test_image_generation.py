from __future__ import annotations

from typing import Type, Union

import psutil
import pytest
import requests
import torch
from PIL import Image

from datadreamer.image_generation import (
    Shuttle3DiffusionImageGenerator,
    StableDiffusionImageGenerator,
    StableDiffusionLightningImageGenerator,
    StableDiffusionTurboImageGenerator,
)
from datadreamer.image_generation.clip_image_tester import ClipImageTester

# Get the total memory in GB
total_memory = psutil.virtual_memory().total / (1024**3)
# Get the total disk space in GB
total_disk_space = psutil.disk_usage("/").total / (1024**3)


def _check_clip_image_tester(device: str):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    tester = ClipImageTester(device=device)
    # Check that the tester is not None
    assert tester is not None
    passed, probs, num_passed = tester.test_image(im, ["bus"])
    # Check that the image passed the test
    assert passed is True
    # Check that the number of objects passed is correct
    assert num_passed == 1
    # Check that the probability has correct shape
    assert probs.shape == (1, 1)
    # Check that the probability is not zero
    assert probs[0, 0] > 0
    passed_list, probs_list, num_passed_list = tester.test_images_batch([im], [["bus"]])
    # Check that the image passed the test
    assert passed_list[0] is True
    # Check that the number of objects passed is correct
    assert num_passed_list[0] == 1
    # Check that the probability has correct shape
    assert len(probs_list) == 1
    # Check that the probability is not zero
    assert probs_list[0][0] > 0
    tester.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_clip_image_tester():
    _check_clip_image_tester("cuda")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_clip_image_tester():
    _check_clip_image_tester("cpu")


def _check_image_generator(
    image_generator_class: Type[
        Union[
            StableDiffusionImageGenerator,
            StableDiffusionTurboImageGenerator,
            StableDiffusionLightningImageGenerator,
            Shuttle3DiffusionImageGenerator,
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
def test_cuda_sdxl_turbo_image_generator():
    _check_image_generator(StableDiffusionTurboImageGenerator, "cuda")


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 25,
    reason="Test requires at least 16GB of RAM and 25GB of HDD",
)
def test_cpu_sdxl_turbo_image_generator():
    _check_image_generator(StableDiffusionTurboImageGenerator, "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 25,
    reason="Test requires GPU, at least 16GB of RAM and 25GB of HDD",
)
def test_cuda_shuttle_3_image_generator():
    _check_image_generator(Shuttle3DiffusionImageGenerator, "cuda")


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 25,
    reason="Test requires at least 16GB of RAM and 25GB of HDD",
)
def test_cpu_shuttle_3_image_generator():
    _check_image_generator(Shuttle3DiffusionImageGenerator, "cpu")
