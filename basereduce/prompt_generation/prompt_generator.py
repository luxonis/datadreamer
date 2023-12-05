import json
import random
import torch
from typing import List, Optional


from abc import ABC, abstractmethod


# Abstract base class for prompt generation
class PromptGenerator(ABC):
    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        num_objects_range: Optional[List[int]] = [1, 3],
        seed: Optional[float] = None,
    ) -> None:
        self.class_names = class_names
        self.prompts_number = prompts_number
        self.num_objects_range = num_objects_range
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_prompts(self, prompts: List[str], save_path: str) -> None:
        with open(save_path, "w") as f:
            json.dump(prompts, f)

    @abstractmethod
    def generate_prompts(self) -> List[str]:
        pass

    @abstractmethod
    def release(self, empty_cuda_cache=False) -> None:
        pass
