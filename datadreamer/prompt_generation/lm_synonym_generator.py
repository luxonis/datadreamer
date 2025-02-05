from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

from datadreamer.prompt_generation.synonym_generator import SynonymGenerator


class LMSynonymGenerator(SynonymGenerator):
    """Synonym generator that generates synonyms for a list of words using a language
    model.

    Args:
        synonyms_number (int): Number of synonyms to generate for each word.
        seed (Optional[float]): Seed for randomization.
        device (str): Device for model inference (default is "cuda").

    Methods:
        _init_lang_model(): Initializes the language model and tokenizer.
        _generate_synonyms(prompt_text): Generates synonyms based on a given prompt text.
        _extract_synonyms(text): Extracts synonyms from a text containing synonyms.
        _create_prompt_text(word): Creates a prompt text for generating synonyms for a given word.
        generate_synonyms(word): Generates synonyms for a single word and returns them in a list.
        release(empty_cuda_cache): Releases resources (no action is taken in this implementation).
    """

    def __init__(
        self,
        synonyms_number: int = 3,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the SynonymGenerator with parameters."""
        super().__init__(synonyms_number, seed, device)
        self.model, self.tokenizer, self.pipeline = self._init_lang_model()

    def _init_lang_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Pipeline]:
        """Initializes the language model, tokenizer and pipeline for prompt generation.

        Returns:
            tuple: The initialized language model, tokenizer and pipeline.
        """
        logger.info(f"Initializing Mistral-7B language model on {self.device}...")
        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            logger.info("Loading FP16 language model...")
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.1",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=self.device,
            )

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if self.device == "cuda" else "auto",
            device_map=self.device,
        )
        logger.info("Done!")
        return model, tokenizer, pipe

    def _generate_synonyms(self, prompt_text: str) -> List[str]:
        """Generates synonyms based on a given prompt text.

        Args:
            prompt_text (str): The prompt text for generating synonyms.

        Returns:
            List[str]: A list of generated synonyms.
        """
        sequences = self.pipeline(
            prompt_text,
            max_new_tokens=50,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = sequences[0]["generated_text"]

        instructional_pattern = r"\[INST].*?\[/INST\]\s*"
        # Remove the instructional text to isolate the caption
        generated_text = (
            re.sub(instructional_pattern, "", generated_text)
            .replace('"', "")
            .replace("'", "")
            .replace(".", "")
        )

        # Process the generated text to extract synonyms
        synonyms = self._extract_synonyms(generated_text)
        return synonyms

    def _extract_synonyms(self, text: str) -> List[str]:
        """Extracts synonyms from a text containing synonyms.

        Args:
            text (str): The text containing synonyms.

        Returns:
            List[str]: A list of extracted synonyms.
        """
        synonyms = [
            word.strip() for word in text.split(",")
        ]  # Split and strip each synonym
        return synonyms[: self.synonyms_number]

    def _create_prompt_text(self, word: str) -> str:
        """Creates a prompt text for generating synonyms for a given word.

        Args:
            word (str): The word for which synonyms are generated.

        Returns:
            str: The prompt text for generating synonyms.
        """
        return f"[INST] List {self.synonyms_number} most common synonyms for the word '{word}'. Write only synonyms separated by commas. [/INST]"

    def generate_synonyms(self, word: str) -> List[str]:
        """Generates synonyms for a single word and returns them in a list.

        Args:
            word (str): The word for which synonyms are generated.

        Returns:
            List[str]: A list of generated synonyms for the word.
        """
        prompt_text = self._create_prompt_text(word)
        generated_synonyms = self._generate_synonyms(prompt_text)
        return generated_synonyms

    def release(self, empty_cuda_cache=False) -> None:
        """Releases resources and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool): Whether to empty the CUDA cache (default is False).
        """
        self.model = self.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage
    generator = LMSynonymGenerator(synonyms_number=3, device="cpu")
    synonyms = generator.generate_synonyms_for_list(
        ["astronaut", "cat", "dog", "person", "horse"]
    )
    print(synonyms)
    # generator.save_synonyms(synonyms, "synonyms.json")
    # generator.release()
