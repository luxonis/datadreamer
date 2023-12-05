import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
from typing import List, Optional

from basereduce.prompt_generation.prompt_generator import PromptGenerator


class LMPromptGenerator(PromptGenerator):
    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        num_objects_range: Optional[List[int]] = [1, 3],
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        super().__init__(class_names, prompts_number, num_objects_range, seed)
        self.device = device
        self.model, self.tokenizer = self._init_lang_model()

    def _init_lang_model(self):
        print("Loading language model...")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        print("Done!")
        return model, tokenizer

    def generate_prompts(self) -> List[str]:
        prompts = []
        for _ in tqdm(range(self.prompts_number), desc="Generating prompts"):
            selected_objects = random.sample(
                self.class_names, random.randint(*self.num_objects_range)
            )
            prompt_text = self._create_lm_prompt_text(selected_objects)
            correct_prompt_generated = False
            while not correct_prompt_generated:
                generated_prompt = self.generate_prompt(prompt_text)
                if self._test_prompt(generated_prompt, selected_objects):
                    prompts.append((selected_objects, generated_prompt))
                    correct_prompt_generated = True
        return prompts

    def _create_lm_prompt_text(self, selected_objects: List[str]) -> str:
        return f"[INST] Generate a short and consice caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. [/INST]"

    def generate_prompt(self, prompt_text: str) -> str:
        encoded_input = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **encoded_input, max_new_tokens=70, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
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
        return all(
            obj.lower() in prompt.lower() for obj in selected_objects
        ) and prompt.startswith("A photo of")

    def release(self, empty_cuda_cache=False) -> None:
        self.model = self.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    object_names = ["aeroplane", "bicycle", "bird", "boat"]
    prompt_generator = LMPromptGenerator(class_names=object_names)
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
