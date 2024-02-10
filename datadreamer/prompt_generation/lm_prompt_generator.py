import random
import re
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from datadreamer.prompt_generation.prompt_generator import PromptGenerator


class LMPromptGenerator(PromptGenerator):
    """A language model-based prompt generator class, extending PromptGenerator.

    Attributes:
        device (str): Device to run the language model on ('cuda' for GPU, 'cpu' for CPU).
        model (AutoModelForCausalLM): The pre-trained causal language model for generating prompts.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.

    Methods:
        _init_lang_model(): Initializes the language model and tokenizer.
        generate_prompts(): Generates a list of prompts based on the class names.
        _create_lm_prompt_text(selected_objects): Creates a text prompt for the language model.
        generate_prompt(prompt_text): Generates a single prompt using the language model.
        _test_prompt(prompt, selected_objects): Tests if the generated prompt is valid.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        num_objects_range: Optional[List[int]] = None,
        seed: Optional[float] = 42,
        device: str = "cuda",
        quantization: str = "none",
    ) -> None:
        """Initializes the LMPromptGenerator with class names and other settings."""
        num_objects_range = num_objects_range or [1, 3]
        super().__init__(
            class_names, prompts_number, num_objects_range, seed, device, quantization
        )
        self.model, self.tokenizer, self.pipeline = self._init_lang_model()

    def _init_lang_model(self):
        """Initializes the language model and tokenizer for prompt generation.

        Returns:
            tuple: The initialized language model and tokenizer.
        """
        selected_device = "cpu"
        selected_dtype = "auto"
        if self.device == "cpu":
            print("Loading language model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            if self.quantization == "none":
                print("Loading FP16 language model on GPU...")
                selected_device = "cuda"
                selected_dtype = torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    torch_dtype=selected_dtype,
                    trust_remote_code=True,
                    device_map=selected_device,
                )
            else:
                print(f"Loading INT4 language model on GPU...")
                # Create the BitsAndBytesConfig object with the dynamically constructed arguments
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
                selected_device = "cuda"
                selected_dtype = torch.bfloat16

                model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    load_in_4bit=True,
                    quantization_config=bnb_config,
                    torch_dtype=selected_dtype,
                    device_map=selected_device,
                    trust_remote_code=True,
                )

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            torch_dtype=selected_dtype, 
            device_map=selected_device
        )
        print("Done!")
        return model, tokenizer, pipe

    def generate_prompts(self) -> List[str]:
        """Generates a list of text prompts based on the class names.

        Returns:
            List[str]: A list of generated prompts.
        """
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
        """Creates a language model text prompt based on selected objects.

        Args:
            selected_objects (List[str]): Objects to include in the prompt.

        Returns:
            str: A text prompt for the language model.
        """
        return f"[INST] Generate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. [/INST]"

    def generate_prompt(self, prompt_text: str) -> str:
        """Generates a single prompt using the language model.

        Args:
            prompt_text (str): The text prompt for the language model.

        Returns:
            str: The generated prompt.
        """
        sequences = self.pipeline(
            prompt_text,
            max_new_tokens=70, 
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded_prompt = sequences[0]['generated_text']
        instructional_pattern = r"\[INST].*?\[/INST\]\s*"
        # Remove the instructional text to isolate the caption
        decoded_prompt = (
            re.sub(instructional_pattern, "", decoded_prompt)
            .replace('"', "")
            .replace("'", "")
        )

        return decoded_prompt

    def _test_prompt(self, prompt: str, selected_objects: List[str]) -> bool:
        """Tests if the generated prompt is valid based on selected objects.

        Args:
            prompt (str): The generated prompt.
            selected_objects (List[str]): Objects to check in the prompt.

        Returns:
            bool: True if the prompt is valid, False otherwise.
        """
        return prompt.lower().startswith(
            "a photo of"
        )  # and all(obj.lower() in prompt.lower() for obj in selected_objects)

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache."""
        if self.quantization == "none":
            self.model = self.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage of the class
    object_names = ["aeroplane", "bicycle", "bird", "boat"]
    prompt_generator = LMPromptGenerator(
        class_names=object_names, prompts_number=2, device="cpu"
    )
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
