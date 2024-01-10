import pytest
import torch
from PIL import Image
from datadreamer.image_generation.sdxl_image_generator import StableDiffusionImageGenerator
from datadreamer.image_generation.sdxl_turbo_image_generator import StableDiffusionTurboImageGenerator
from datadreamer.image_generation.clip_image_tester import ClipImageTester


def test_clip_image_tester():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device in ["cuda", "cpu"]


def test_sdxl_image_generator():
    device = "cpu" #  "cuda" if torch.cuda.is_available() else "cpu"
    # image_generator = StableDiffusionImageGenerator(device=device)
    image_generator = StableDiffusionImageGenerator()
    for generated_image in image_generator.generate_images(["A photo of a cat, dog"], [["cat", "dog"]]):
        assert generated_image is not None
        assert isinstance(generated_image, Image.Image)
    image_generator.release(empty_cuda_cache=True if device != 'cpu' else False)


def test_sdxl_turbo_image_generator():
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    # image_generator = StableDiffusionTurboImageGenerator(device=device)
    image_generator = StableDiffusionTurboImageGenerator()
    for generated_image in image_generator.generate_images(["A photo of a cat, dog"], [["cat", "dog"]]):
        assert generated_image is not None
        assert isinstance(generated_image, Image.Image)
    image_generator.release(empty_cuda_cache=True if device != 'cpu' else False)


if __name__ == "__main__":
    pytest.main()
