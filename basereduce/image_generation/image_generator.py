from abc import ABC, abstractmethod
from typing import Optional, Union, List
import enum
from PIL import Image
import torch
from tqdm import tqdm
import random


class ImageGenerator:
    def __init__(
        self,
        # model_name: GenModelName,
        prompt_prefix: Optional[str] = "",
        prompt_suffix: Optional[str] = ", hd, 8k, highly detailed",
        negative_prompt: Optional[
            str
        ] = "cartoon, blue skin, painting, scrispture, golden, illustration, worst quality, low quality, normal quality:2, unrealistic dream, low resolution,  static, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, bad anatomy",
        seed: Optional[float] = 42,
    ) -> None:
        # model_class = globals()[model_name.value]
        # self.model = model_class()
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.negative_prompt = negative_prompt
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def generate_images(
        self,
        prompts: Union[str, List[str]],
        prompt_objects: Optional[List[List[str]]] = None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = [
            self.prompt_prefix + prompt + self.prompt_suffix for prompt in prompts
        ]
        if prompt_objects is None:
            for prompt in tqdm(prompts, desc="Generating images"):
                yield self.generate_image(prompt, self.negative_prompt)
        else:
            for prompt, prompt_object in tqdm(zip(prompts, prompt_objects), desc="Generating images"):
                yield self.generate_image(prompt, self.negative_prompt, prompt_object)

    @abstractmethod
    def release(self, empty_cuda_cache=False) -> None:
        pass

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        prompt_objects: Optional[List[str]] = None,
    ) -> Image.Image:
        pass
