import pytest
import torch
from PIL import Image
import requests
import psutil
from typing import Union, Type
from datadreamer.image_generation.sdxl_image_generator import (
    StableDiffusionImageGenerator,
)
from datadreamer.image_generation.sdxl_turbo_image_generator import (
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
    passed, probs, num_passed = tester.test_image(im, ["bus"])
    # Check that the image passed the test
    assert passed is True
    # Check that the number of objects passed is correct
    assert num_passed == 1
    # Check that the probability has correct shape
    assert probs.shape == (1, 1)
    # Check that the probability is not zero
    assert probs[0, 0] > 0
    # Release the tester
    tester.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_cuda_clip_image_tester():
    _check_clip_image_tester("cuda")


def test_cpu_clip_image_tester():
    _check_clip_image_tester("cpu")


def _check_image_generator(
    image_generator_class: Type[
        Union[StableDiffusionImageGenerator, StableDiffusionTurboImageGenerator]
    ],
    device: str,
):
    image_generator = image_generator_class(device=device)
    # Generate images and check each of them
    for generated_image in image_generator.generate_images(
        ["A photo of a cat, dog"], [["cat", "dog"]]
    ):
        assert generated_image is not None
        assert isinstance(generated_image, Image.Image)
    # Release the generator
    image_generator.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_cuda_sdxl_image_generator():
    _check_image_generator(StableDiffusionImageGenerator, "cuda")


def test_cpu_sdxl_image_generator():
    _check_image_generator(StableDiffusionImageGenerator, "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 16,
    reason="Test requires GPU, at least 16GB of RAM and 16GB of HDD",
)
def test_cuda_sdxl_turbo_image_generator():
    _check_image_generator(StableDiffusionTurboImageGenerator, "cuda")


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 16,
    reason="Test requires at least 16GB of RAM and 16GB of HDD",
)
def test_cpu_sdxl_turbo_image_generator():
    _check_image_generator(StableDiffusionTurboImageGenerator, "cpu")
