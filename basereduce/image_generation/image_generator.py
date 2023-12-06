from abc import ABC, abstractmethod
from typing import Optional, Union, List
import enum
from PIL import Image
import torch
from tqdm import tqdm
import random

from basereduce.image_generation.clip_image_tester import ClipImageTester



class ImageGenerator:
    def __init__(
        self,
        # model_name: GenModelName,
        prompt_prefix: Optional[str] = "",
        prompt_suffix: Optional[str] = ", hd, 8k, highly detailed",
        negative_prompt: Optional[
            str
        ] = "cartoon, blue skin, painting, scrispture, golden, illustration, worst quality, low quality, normal quality:2, unrealistic dream, low resolution,  static, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, bad anatomy",
        use_clip_image_tester: Optional[bool] = False,
        image_tester_patience: Optional[int] = 1,
        seed: Optional[float] = 42,
    ) -> None:
        # model_class = globals()[model_name.value]
        # self.model = model_class()
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
            prompt_objects = [None] * len(prompts)
        
        for prompt, prompt_objs in tqdm(
                zip(prompts, prompt_objects), desc="Generating images", total=len(prompts)
            ):
                if self.use_clip_image_tester:
                    best_prob = 0
                    best_image = None
                    best_num_passed = 0

                    for _ in tqdm(range(self.image_tester_patience), desc="Testing image"):
                        image = self.generate_image(prompt, self.negative_prompt, prompt_objs)
                        passed, probs, num_passed = self.clip_image_tester.test_image(
                            image, prompt_objs
                        )
                        # Return the first image that passes the test
                        if passed:
                            yield image
                            break
                        mean_prob = probs.mean().item()
                        if num_passed > best_num_passed or (num_passed == best_num_passed and mean_prob > best_prob):
                            best_image = image
                            best_prob = mean_prob
                            best_num_passed = num_passed
                    if passed:
                        continue
                    # If no image passed the test, return the image with the highest number of objects that passed the test
                    yield best_image
                
                else:
                    yield self.generate_image(prompt, self.negative_prompt, prompt_objs)

                

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
