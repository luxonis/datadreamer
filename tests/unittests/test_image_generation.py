import pytest
import torch
from PIL import Image
import requests
from datadreamer.image_generation.sdxl_image_generator import (
    StableDiffusionImageGenerator,
)
from datadreamer.image_generation.sdxl_turbo_image_generator import (
    StableDiffusionTurboImageGenerator,
)
from datadreamer.image_generation.clip_image_tester import ClipImageTester


def test_clip_image_tester():
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def test_sdxl_image_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # image_generator = StableDiffusionImageGenerator(device=device)
    image_generator = StableDiffusionImageGenerator()
    for generated_image in image_generator.generate_images(
        ["A photo of a cat, dog"], [["cat", "dog"]]
    ):
        assert generated_image is not None
        assert isinstance(generated_image, Image.Image)
    # Release the generator
    image_generator.release(empty_cuda_cache=True if device != "cpu" else False)


def test_sdxl_turbo_image_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # image_generator = StableDiffusionTurboImageGenerator(device=device)
    image_generator = StableDiffusionTurboImageGenerator()
    for generated_image in image_generator.generate_images(
        ["A photo of a cat, dog"], [["cat", "dog"]]
    ):
        assert generated_image is not None
        assert isinstance(generated_image, Image.Image)
    # Release the generator
    image_generator.release(empty_cuda_cache=True if device != "cpu" else False)


if __name__ == "__main__":
    pytest.main()
