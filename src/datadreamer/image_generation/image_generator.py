import random
from abc import abstractmethod
from typing import List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

from datadreamer.image_generation.clip_image_tester import ClipImageTester


class ImageGenerator:
    """A class for generating images based on textual prompts, with optional CLIP model
    testing.

    Attributes:
        prompt_prefix (str): Optional prefix to add to every prompt.
        prompt_suffix (str): Optional suffix to add to every prompt, e.g., for adding details like resolution.
        negative_prompt (str): A string of negative prompts to guide the generation away from certain features.
        use_clip_image_tester (bool): Flag to use CLIP model testing for generated images.
        image_tester_patience (int): The number of attempts to generate an image that passes CLIP testing.
        seed (float): Seed for reproducibility.
        clip_image_tester (ClipImageTester): Instance of ClipImageTester if use_clip_image_tester is True.

    Methods:
        set_seed(seed): Sets the seed for random number generators.
        generate_images(prompts, prompt_objects): Generates images based on provided prompts and optional object prompts.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache. (Abstract method)
        generate_image(prompt, negative_prompt, prompt_objects): Generates a single image based on the provided prompt. (Abstract method)

    Note:
        The actual model for image generation needs to be defined in the subclass.
    """

    def __init__(
        self,
        prompt_prefix: Optional[str] = "",
        prompt_suffix: Optional[str] = ", hd, 8k, highly detailed",
        negative_prompt: Optional[
            str
        ] = "cartoon, blue skin, painting, scrispture, golden, illustration, worst quality, low quality, normal quality:2, unrealistic dream, low resolution,  static, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, bad anatomy",
        use_clip_image_tester: Optional[bool] = False,
        image_tester_patience: Optional[int] = 1,
        seed: Optional[float] = 42,
    ) -> None:
        """Initializes the ImageGenerator with the specified settings."""
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.use_clip_image_tester = use_clip_image_tester
        self.image_tester_patience = image_tester_patience
        if self.use_clip_image_tester:
            self.clip_image_tester = ClipImageTester()
        if seed is not None:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int):
        """Sets the seed for random number generators in Python and PyTorch.

        Args:
            seed (int): The seed value to set.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def generate_images(
        self,
        prompts: Union[str, List[str]],
        prompt_objects: Optional[List[List[str]]] = None,
    ):
        """Generates images based on the provided prompts and optional object prompts.

        Args:
            prompts (Union[str, List[str]]): Single prompt or a list of prompts to guide the image generation.
            prompt_objects (Optional[List[List[str]]]): Optional list of objects for each prompt for CLIP model testing.

        Yields:
            Image.Image: Generated images.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [
            self.prompt_prefix + prompt + self.prompt_suffix for prompt in prompts
        ]
        if prompt_objects is None:
            prompt_objects = [None] * len(prompts)

        for prompt, prompt_objs in tqdm(
            zip(prompts, prompt_objects), desc="Generating images", total=len(prompts)
        ):
            if self.use_clip_image_tester:
                best_prob = 0
                best_image = None
                best_num_passed = 0
                passed = False

                for _ in tqdm(range(self.image_tester_patience), desc="Testing image"):
                    image = self.generate_image(
                        prompt, self.negative_prompt, prompt_objs
                    )
                    passed, probs, num_passed = self.clip_image_tester.test_image(
                        image, prompt_objs
                    )
                    # Return the first image that passes the test
                    if passed:
                        yield image
                        break
                    mean_prob = probs.mean().item()
                    if num_passed > best_num_passed or (
                        num_passed == best_num_passed and mean_prob > best_prob
                    ):
                        best_image = image
                        best_prob = mean_prob
                        best_num_passed = num_passed
                # If no image passed the test, return the image with the highest number of objects that passed the test
                if not passed:
                    yield best_image

            else:
                yield self.generate_image(prompt, self.negative_prompt, prompt_objs)

    @abstractmethod
    def release(self, empty_cuda_cache=False) -> None:
        """Releases resources and optionally empties the CUDA cache."""
        pass

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_objects: Optional[List[str]] = None,
    ) -> Image.Image:
        """Generates a single image based on the provided prompt.

        Args:
            prompt (str): The positive prompt to guide image generation.
            negative_prompt (str): The negative prompt to avoid certain features in the image.
            prompt_objects (Optional[List[str]]): Optional list of objects to be used in CLIP model testing.

        Returns:
            Image.Image: The generated image.
        """
        pass
