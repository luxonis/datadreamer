from __future__ import annotations

import re
from typing import List, Literal, Optional, Tuple

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    pipeline,
)

from datadreamer.prompt_generation.lm_prompt_generator import LMPromptGenerator


class Qwen2LMPromptGenerator(LMPromptGenerator):
    """A language model-based prompt generator class, extending PromptGenerator.

    Attributes:
        device (str): Device to run the language model on ('cuda' for GPU, 'cpu' for CPU).
        model (AutoModelForCausalLM): The pre-trained causal language model for generating prompts.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.
        pipeline (pipeline): The HuggingFace pipeline for generating text.

    Methods:
        _init_lang_model(): Initializes the language model and tokenizer.
        _remove_caption_sentences(text): Removes caption sentences from the generated prompt.
        _create_lm_prompt_text(selected_objects): Creates a text prompt for the language model.
        _postprocess_prompt(prompt): Post-processes the generated prompt.
        generate_prompts_batch(prompt_texts_batch): Generates a batch of prompts using the language model.
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
        super().__init__(
            class_names,
            prompts_number,
            num_objects_range,
            batch_size,
            seed,
            device,
            quantization,
        )

    def _init_lang_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Pipeline]:
        """Initializes the language model, tokenizer and pipeline for prompt generation.

        Returns:
            tuple: The initialized language model, tokenizer and pipeline.
        """
        selected_dtype = "auto"
        logger.info(
            f"Initializing Qwen2.5-1.5B-Instruct language model on {self.device}..."
        )
        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            if self.quantization == "none":
                logger.info("Loading FP16 language model...")
                selected_dtype = torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct",
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
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    quantization_config=bnb_config,
                    torch_dtype=selected_dtype,
                    device_map=self.device,
                    trust_remote_code=True,
                )

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", padding_side="left"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=selected_dtype,
            device_map=self.device,
            batch_size=self.batch_size,
        )
        return model, tokenizer, pipe

    def _remove_caption_sentences(self, text: str) -> str:
        """Removes caption sentences from the generated prompt.

        Args:
            text (str): The generated prompt text.

        Returns:
            str: The cleaned prompt text.
        """
        pattern = re.compile(r"\s*Caption reads: [^\.!?]*[\.\!?]", re.IGNORECASE)
        # Replace the matched sentences with an empty string
        cleaned_text = re.sub(pattern, "", text)
        return cleaned_text

    def _create_lm_prompt_text(self, selected_objects: List[str]) -> str:
        """Creates a language model text prompt based on selected objects.

        Args:
            selected_objects (List[str]): Objects to include in the prompt.

        Returns:
            str: A text prompt for the language model.
        """
        return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a chatbot who describes content of images!<|im_end|>\n<|im_start|>user\nGenerate a short and concise caption for an image. The caption must begin with this template: 'A photo of {', '.join(selected_objects)}'. The objects within the scene interact in a meaningful way. Complete the caption with a short scene description.<|im_end|>\n<|im_start|>assistant\n"

    def _postprocess_prompt(self, prompt: str) -> str:
        """Post-processes the generated prompt.

        Args:
            prompt (str): The generated prompt.

        Returns:
            str: The post-processed prompt.
        """
        instructional_pattern = r"<\|im_start\|>system\n.*?<\|im_end\|>\n<\|im_start\|>user\n.*?<\|im_end\|>\n<\|im_start\|>assistant\n"
        # Remove the instructional text to isolate the caption
        prompt = (
            re.sub(instructional_pattern, "", prompt).replace('"', "").replace("'", "")
        )
        prompt = self._remove_caption_sentences(
            self._remove_incomplete_sentence(prompt)
        )
        return prompt

    def generate_prompts_batch(self, prompt_texts_batch: List[str]) -> List[str]:
        """Generates a list of prompts using the language model.

        Args:
            prompt_texts_batch (List[str]): List of text prompts for the language model.

        Returns:
            List[str]: List of generated prompts.
        """
        sequences = self.pipeline(prompt_texts_batch, max_new_tokens=70)
        decoded_prompts = [
            self._postprocess_prompt(sequence[0]["generated_text"])
            for sequence in sequences
        ]

        return decoded_prompts


if __name__ == "__main__":
    # Example usage of the class
    object_names = ["aeroplane", "bicycle", "bird", "boat", "city"]
    prompt_generator = Qwen2LMPromptGenerator(
        class_names=object_names, prompts_number=5, device="cpu"
    )
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
