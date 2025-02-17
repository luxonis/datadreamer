from __future__ import annotations

import random
import re
from typing import List, Literal, Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    pipeline,
)

from datadreamer.prompt_generation.prompt_generator import PromptGenerator


class LMPromptGenerator(PromptGenerator):
    """A language model-based prompt generator class, extending PromptGenerator.

    Attributes:
        device (str): Device to run the language model on ('cuda' for GPU, 'cpu' for CPU).
        num_objects_range (List[int]): Range for number of objects in a single image.
        model (AutoModelForCausalLM): The pre-trained causal language model for generating prompts.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.
        pipeline (pipeline): The HuggingFace pipeline for generating text.

    Methods:
        _init_lang_model(): Initializes the language model and tokenizer.
        _remove_incomplete_sentence(text): Removes incomplete sentences from the generated prompt.
        _create_lm_prompt_text(selected_objects): Creates a text prompt for the language model.
        _create_lm_prompt_text_batch(selected_objects_batch): Creates a batch of text prompts for the language model.
        _postprocess_prompt(prompt): Post-processes the generated prompt.
        _test_prompt(prompt, selected_objects): Tests if the generated prompt is valid.
        generate_prompts_batch(prompt_texts_batch): Generates a batch of prompts using the language model.
        generate_prompts(): Generates a list of prompts based on the class names.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        num_objects_range: Optional[List[int]] = None,
        batch_size: int = 1,
        seed: Optional[float] = 42,
        device: str = "cuda",
        quantization: Optional[Literal["none", "4bit"]] = "none",
    ) -> None:
        """Initializes the LMPromptGenerator with class names and other settings."""
        num_objects_range = num_objects_range or [1, 3]
        super().__init__(
            class_names,
            prompts_number,
            num_objects_range,
            batch_size,
            seed,
            device,
            quantization,
        )
        self.model, self.tokenizer, self.pipeline = self._init_lang_model()

    def _init_lang_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Pipeline]:
        """Initializes the language model, tokenizer and pipeline for prompt generation.

        Returns:
            tuple: The initialized language model, tokenizer and pipeline.
        """
        selected_dtype = "auto"
        logger.info(f"Initializing Mistral-7B language model on {self.device}...")
        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            if self.quantization == "none":
                logger.info("Loading FP16 language model...")
                selected_dtype = torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    torch_dtype=selected_dtype,
                    trust_remote_code=True,
                    device_map=self.device,
                )
            else:
                logger.info("Loading INT4 language model...")
                # Create the BitsAndBytesConfig object with the dynamically constructed arguments
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                selected_dtype = torch.bfloat16

                model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    quantization_config=bnb_config,
                    torch_dtype=selected_dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                )

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer.pad_token = tokenizer.eos_token
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=selected_dtype,
            device_map=self.device,
            batch_size=self.batch_size,
        )
        return model, tokenizer, pipe

    def _remove_incomplete_sentence(self, text: str) -> str:
        """Removes incomplete sentences from the generated prompt.

        Args:
            text (str): The generated prompt text.

        Returns:
            str: The cleaned prompt text.
        """
        # Define the regex pattern to capture up to the last sentence-ending punctuation
        pattern = r"^(.*[.!?])"
        match = re.search(pattern, text)
        return match.group(0) if match else text

    def _create_lm_prompt_text(self, selected_objects: List[str]) -> str:
        """Creates a language model text prompt based on selected objects.

        Args:
            selected_objects (List[str]): Objects to include in the prompt.

        Returns:
            str: A text prompt for the language model.
        """
        return f"[INST] Generate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. [/INST]"

    def _create_lm_prompt_texts_batch(
        self, selected_objects_batch: List[List[str]]
    ) -> List[str]:
        """Creates a list of language model text prompts based on selected objects.

        Args:
            selected_objects_batch (List[List[str]]): List of objects to include in the prompts.

        Returns:
            List[str]: List of text prompts for the language model.
        """
        return [
            self._create_lm_prompt_text(selected_objects)
            for selected_objects in selected_objects_batch
        ]

    def _postprocess_prompt(self, prompt: str) -> str:
        """Post-processes the generated prompt.

        Args:
            prompt (str): The generated prompt.

        Returns:
            str: The post-processed prompt.
        """
        instructional_pattern = r"\[INST].*?\[/INST\]\s*"
        # Remove the instructional text to isolate the caption
        prompt = (
            re.sub(instructional_pattern, "", prompt).replace('"', "").replace("'", "")
        )
        prompt = self._remove_incomplete_sentence(prompt)
        return prompt

    def _test_prompt(self, prompt: str, selected_objects: List[str]) -> bool:
        """Tests if the generated prompt is valid based on selected objects.

        Args:
            prompt (str): The generated prompt.
            selected_objects (List[str]): Objects to check in the prompt.

        Returns:
            bool: True if the prompt is valid, False otherwise.
        """
        return prompt.lower().startswith("a photo of")

    def generate_prompts_batch(self, prompt_texts_batch: List[str]) -> List[str]:
        """Generates a list of prompts using the language model.

        Args:
            prompt_texts_batch (List[str]): List of text prompts for the language model.

        Returns:
            List[str]: List of generated prompts.
        """
        sequences = self.pipeline(
            prompt_texts_batch,
            max_new_tokens=70,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_prompts = [
            self._postprocess_prompt(sequence[0]["generated_text"])
            for sequence in sequences
        ]
        return decoded_prompts

    def generate_prompts(self) -> List[str]:
        """Generates a list of text prompts based on the class names.

        Returns:
            List[str]: A list of generated prompts.
        """
        prompts = []
        progress_bar = tqdm(
            desc="Generating prompts", position=0, total=self.prompts_number
        )
        while len(prompts) < self.prompts_number:
            selected_objects_batch = [
                random.sample(self.class_names, random.randint(*self.num_objects_range))
                for _ in range(self.batch_size)
            ]
            prompt_texts_batch = self._create_lm_prompt_texts_batch(
                selected_objects_batch
            )
            generated_prompts_batch = self.generate_prompts_batch(prompt_texts_batch)
            for generated_prompt, selected_objects in zip(
                generated_prompts_batch, selected_objects_batch
            ):
                if self._test_prompt(generated_prompt, selected_objects):
                    prompts.append((selected_objects, generated_prompt))
                    progress_bar.update()
                if len(prompts) == self.prompts_number:
                    break

        progress_bar.close()
        return prompts

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
