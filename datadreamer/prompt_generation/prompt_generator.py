from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import torch


# Abstract base class for prompt generation
class PromptGenerator(ABC):
    """Abstract base class for prompt generation.

    Attributes:
        class_names (List[str]): List of class names or objects for prompt generation.
        prompts_number (int): Number of prompts to generate.
        num_objects_range (Optional[List[int]]): Range for the number of objects to include in prompts.
        seed (Optional[float]): Seed for randomization.
        device (str): Device to run the prompt generator on ('cuda' for GPU, 'cpu' for CPU).
        quantization (str): Quantization type for the prompt generator.

    Methods:
        set_seed(seed): Sets the random seed for consistent prompt generation.
        save_prompts(prompts, save_path): Saves generated prompts to a JSON file.
        generate_prompts(): Abstract method to generate prompts (must be implemented in subclasses).
        release(empty_cuda_cache): Abstract method to release resources (must be implemented in subclasses).
    """

    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        num_objects_range: Optional[List[int]] = None,
        batch_size: int = 1,
        seed: Optional[float] = None,
        device: str = "cuda",
        quantization: Optional[Literal["none", "4bit"]] = "none",
    ) -> None:
        """Initializes the PromptGenerator with class names and other settings."""
        self.class_names = class_names
        self.prompts_number = prompts_number
        self.num_objects_range = num_objects_range or [1, 3]
        self.batch_size = batch_size
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)
        self.device = device
        self.quantization = quantization if quantization is not None else "none"

    @staticmethod
    def set_seed(seed: int) -> None:
        """Sets the random seed for consistent prompt generation.

        Args:
            seed (int): The random seed.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_prompts(self, prompts: List[str], save_path: str) -> None:
        """Saves generated prompts to a JSON file.

        Args:
            prompts (List[str]): List of generated prompts.
            save_path (str): Path to save the prompts JSON file.
        """
        with open(save_path, "w") as f:
            json.dump(prompts, f)

    @abstractmethod
    def generate_prompts(self) -> List[str]:
        """Abstract method to generate prompts (must be implemented in subclasses).

        Returns:
            List[str]: A list of generated prompts.
        """
        pass

    @abstractmethod
    def release(self, empty_cuda_cache=False) -> None:
        """Abstract method to release resources (must be implemented in subclasses)."""
        pass
