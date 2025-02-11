from __future__ import annotations

import re
from typing import List, Literal, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from datadreamer.prompt_generation.lm_prompt_generator import LMPromptGenerator


class TinyLlamaLMPromptGenerator(LMPromptGenerator):
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
        logger.info(f"Initializing TinyLlama-1.1B language model on {self.device}...")
        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16,
                device_map=self.device,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            trust_remote_code=True,
            padding_side="left",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if self.device == "cuda" else "auto",
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
        # Pattern to find sentences that start with "Caption reads: "
        # \s* matches any whitespace characters at the beginning of the string (including none)
        # re.IGNORECASE makes the search case-insensitive
        # [^\.!?]* matches any sequence of characters that are not a period, exclamation mark, or question mark
        # [\.\!?] matches a period, exclamation mark, or question mark, indicating the end of a sentence
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
        return f"<|system|>\nYou are a chatbot who describes content of images!</s>\n<|user|>\nGenerate a short and concise caption for an image. Follow this template: 'A photo of {', '.join(selected_objects)}', where the objects interact in a meaningful way within a scene, complete with a short scene description. The caption must be short in length and start with the words: 'A photo of '! Do not use the phrase 'Caption reads'.</s>\n<|assistant|>\n"

    def _postprocess_prompt(self, prompt: str) -> str:
        """Post-processes the generated prompt.

        Args:
            prompt (str): The generated prompt.

        Returns:
            str: The post-processed prompt.
        """
        instructional_pattern = r"<\|system\|>\n.*?\n<\|user\|>\n.*?\n<\|assistant\|>\n"
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
        sequences = self.pipeline(
            prompt_texts_batch,
            max_new_tokens=70,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_prompts = [
            self._postprocess_prompt(sequence[0]["generated_text"])
            for sequence in sequences
        ]

        return decoded_prompts


if __name__ == "__main__":
    # Example usage of the class
    object_names = ["aeroplane", "bicycle", "bird", "boat", "city"]
    prompt_generator = TinyLlamaLMPromptGenerator(
        class_names=object_names, prompts_number=5, device="cpu"
    )
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
