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
    MISTRAL = "mistralai/Mistral-7B-Instruct-v0.1"


# Abstract base class for prompt generation
class PromptGenerator(ABC):
    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        seed: Optional[float] = None,
    ) -> None:
        self.class_names = class_names
        self.prompts_number = prompts_number
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

    @abstractmethod
    def generate_prompts(self) -> List[str]:
        pass

    @abstractmethod
    def _test_prompt(self, prompt: str) -> bool:
        pass

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class LMPromptGenerator(PromptGenerator):
    def __init__(
        self,
        class_names: List[str],
        model_name: LMName,
        prompts_number: int = 10,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        super().__init__(class_names, prompts_number, seed)
        self.model_name = model_name
        self.device = device
        self.model, self.tokenizer = self._init_lang_model()

    def _init_lang_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name.value, torch_dtype=torch.float16
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name.value)
        return model, tokenizer

    def generate_prompts(self) -> List[str]:
        prompts = []
        for _ in tqdm(range(self.prompts_number)):
            selected_objects = random.sample(self.class_names, random.randint(1, 3))
            prompt_text = self._create_prompt_text(selected_objects)
            correct_prompt_generated = False
            while not correct_prompt_generated:
                generated_prompt = self._generate_prompt(prompt_text)
                if self._test_prompt(generated_prompt, selected_objects):
                    prompts.append(generated_prompt)
                    correct_prompt_generated = True
        return prompts

    def _create_prompt_text(self, selected_objects: List[str]) -> str:
        return f"[INST] Generate a short and consice caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. [/INST]"

    def _generate_prompt(self, prompt_text: str) -> str:
        encoded_input = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **encoded_input, max_new_tokens=100, do_sample=True
        )
        decoded_prompt = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        instructional_pattern = r"\[INST].*?\[/INST\]\s*"
        # Remove the instructional text to isolate the caption
        decoded_prompt = (
            re.sub(instructional_pattern, "", decoded_prompt)
            .replace('"', "")
            .replace("'", "")
        )

        return decoded_prompt

    def _test_prompt(self, prompt: str, selected_objects: List[str]) -> bool:
        return all(obj.lower() in prompt.lower() for obj in selected_objects)

    def save_prompts(self, prompts: List[str], save_path: str) -> None:
        with open(save_path, "w") as f:
            json.dump(prompts, f)
