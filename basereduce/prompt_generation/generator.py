import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
from typing import List, Optional, Union


from abc import ABC, abstractmethod
import enum

# Enum for language model names
class LMName(enum.Enum):
    MISTRAL = 'mistralai/Mistral-7B-Instruct-v0.1'

# Abstract base class for prompt generation
class PromptGenerator(ABC):

    def __init__(self, class_names: List[str], prompts_number: int = 10, seed: Optional[float] = None) -> None:
        self.class_names = class_names
        self.prompts_number = prompts_number
        self.seed = seed

    @abstractmethod
    def generate_prompts(self) -> List[str]:
        pass

    @abstractmethod
    def _test_prompt(self, prompt: str) -> bool:
        pass


class LMPromptGenerator(PromptGenerator):

    def __init__(self, class_names: List[str], model_name: LMName, prompts_number: int = 10, seed: Optional[float] = None) -> None:
        super().__init__(class_names, prompts_number, seed)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._init_lang_model()

    def _init_lang_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name.value, torch_dtype=torch.float16).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name.value)
        return model, tokenizer

    def generate_prompts(self) -> List[str]:
        prompts = []
        for _ in range(self.prompts_number):
            selected_objects = random.sample(self.class_names, random.randint(1, 3))
            prompt_text = self._create_prompt_text(selected_objects)
            generated_caption = self._generate_caption(prompt_text)
            if self._test_prompt(generated_caption):
                prompts.append(generated_caption)
        return prompts

    def _create_prompt_text(self, selected_objects: List[str]) -> str:
        return f"A HD photo of {', '.join(selected_objects)}"

    def _generate_caption(self, prompt_text: str) -> str:
        encoded_input = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**encoded_input, max_new_tokens=100, do_sample=True)
        decoded_caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return decoded_caption

    def _test_prompt(self, prompt: str) -> bool:
        return True

