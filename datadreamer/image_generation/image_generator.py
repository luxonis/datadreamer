from __future__ import annotations

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
        batch_size (int): The number of images to generate in each batch.
        seed (float): Seed for reproducibility.
        clip_image_tester (ClipImageTester): Instance of ClipImageTester if use_clip_image_tester is True.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).

    Methods:
        set_seed(seed): Sets the seed for random number generators.
        generate_images(prompts, prompt_objects): Generates images based on provided prompts and optional object prompts.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache. (Abstract method)
        generate_images_batch(prompts, negative_prompt, prompt_objects): Generates a batch of images based on the provided prompts. Abstract method)

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
        batch_size: Optional[int] = 1,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the ImageGenerator with the specified settings."""
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.use_clip_image_tester = use_clip_image_tester
        self.image_tester_patience = image_tester_patience
        self.batch_size = batch_size
        self.device = device
        if self.use_clip_image_tester:
            self.clip_image_tester = ClipImageTester(self.device)
        if seed is not None:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int) -> None:
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
    ) -> List[Image.Image]:
        """Generates images based on the provided prompts and optional object prompts.

        Args:
            prompts (Union[str, List[str]]): Single prompt or a list of prompts to guide the image generation.
            prompt_objects (Optional[List[List[str]]]): Optional list of objects for each prompt for CLIP model testing.

        Yields:
            List[Image.Image]: A batch of generated images.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [
            self.prompt_prefix + prompt + self.prompt_suffix for prompt in prompts
        ]
        if prompt_objects is None:
            prompt_objects = [None] * len(prompts)

        progress_bar = tqdm(
            desc="Generating images", total=len(prompts), dynamic_ncols=True
        )

        for i in range(0, len(prompts), self.batch_size):
            prompts_batch = prompts[i : i + self.batch_size]
            prompt_objs_batch = prompt_objects[i : i + self.batch_size]
            if self.use_clip_image_tester:
                best_prob = 0
                best_images_batch = None
                best_num_passed = 0
                passed = False

                for _ in tqdm(range(self.image_tester_patience), desc="Testing image"):
                    images_batch = self.generate_images_batch(
                        prompts_batch, self.negative_prompt, prompt_objs_batch
                    )
                    (
                        passed_list,
                        probs_list,
                        num_passed_list,
                    ) = self.clip_image_tester.test_images_batch(
                        images_batch, prompt_objs_batch
                    )
                    passed = all(passed_list)
                    mean_prob = sum(
                        torch.mean(probs).item() for probs in probs_list
                    ) / len(probs_list)
                    num_passed = sum(num_passed_list)
                    if passed:
                        yield images_batch
                        break
                    if num_passed > best_num_passed or (
                        num_passed == best_num_passed and mean_prob > best_prob
                    ):
                        best_images_batch = images_batch
                        best_prob = mean_prob
                        best_num_passed = num_passed
                # If no image passed the test, return the image with the highest number of objects that passed the test
                if not passed:
                    yield best_images_batch
            else:
                yield self.generate_images_batch(
                    prompts_batch, self.negative_prompt, prompt_objs_batch
                )

            progress_bar.update(len(prompts_batch))

    @abstractmethod
    def release(self, empty_cuda_cache=False) -> None:
        """Releases resources and optionally empties the CUDA cache."""
        pass

    @abstractmethod
    def generate_images_batch(
        self,
        prompts: List[str],
        negative_prompt: str,
        prompt_objects: Optional[List[List[str]]] = None,
    ) -> List[Image.Image]:
        """Generates a batch of images based on the provided prompts.

        Args:
            prompts (List[str]): A list of positive prompts to guide image generation.
            negative_prompt (str): The negative prompt to avoid certain features in the image.
            prompt_objects (Optional[List[List[str]]]): Optional list of objects to be used in CLIP model testing.

        Returns:
            List[Image.Image]: A list of generated images.
        """
        pass
