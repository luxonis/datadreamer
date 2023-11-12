from abc import ABC, abstractmethod
from typing import Optional, Union
import enum

# Enum for generative model names
class GenModelName(enum.Enum):
    STABLE_DIFFUSION_XL = 'stabilityai/stable-diffusion-xl'
    # Add more models as needed

# Abstract base class for image generation
class ImageGenerator(ABC):

    def __init__(
        self,
        seed: float,
        model_name: GenModelName,
        prompt_prefix: Optional[str] = None,
        prompt_suffix: Optional[str] = None,
        negative_prompts: Optional[str] = None
    ) -> None:
        self.seed = seed
        self.model_name = model_name
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.negative_prompts = negative_prompts

    @abstractmethod
    def generate_image(self):
        pass

    @abstractmethod
    def _test_image(self):
        pass