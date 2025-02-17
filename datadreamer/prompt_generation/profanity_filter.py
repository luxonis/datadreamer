from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from datadreamer.utils.bad_words import BAD_WORDS_LIST


class ProfanityFilter:
    """Class for filtering bad words from texts and checking if texts are safe.

    Attributes:
        device (str): Device to run the language model on ('cuda' for GPU, 'cpu' for CPU).
        use_lm (bool): Whether to use a language model for checking text safety.
        seed (Optional[float]): Seed for randomization.
        model (AutoModelForCausalLM): The pre-trained causal language model for checking text safety.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.

    Methods:
        set_seed(seed): Sets the random seed for consistent prompt generation.
        _init_lang_model(): Initializes the language model and tokenizer.
        _contains_bad_words(texts): Checks if a list of texts contain bad words.
        _check_lm_safety(text): Checks if a text is safe using a language model.
        is_safe(classes): Checks if a list of classes is safe.
        release(empty_cuda_cache): Releases the model and optionally empties the CUDA cache.
    """

    LLM_PROMPT = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant who classifies the classes as appropriate or inappropriate. Inappropriate classes are those that directly relate to drugs, hate, racism, harassment, nudity, sexual or offensive words. Here are inappropriate examples:
- 'ass',
- 'a**',
- 'bitch',
- 'pussy',
- and 'f**k'.

Otherwise, the classes are considered appropriate. They can talk about people, characters, animals, nature, history, human conflicts, and so on. Some acceptable examples are:
- 'cat',
- 'angry barking dog',
- 'alien',
- 'dracula',
- 'war',
- 'soldier',
- 'pluto',
- 'sun',
- and 'mercury.'

Respond 'inappropriate' if the classes are unacceptable, otherwise respond with 'appropriate'."""

    def __init__(
        self,
        device: str = "cuda",
        use_lm: bool = False,
        seed: Optional[float] = 42,
    ) -> None:
        """Initializes the ProfanityFilter with parameters."""
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)
        self.device = device
        self.use_lm = use_lm
        if self.use_lm:
            self.model, self.tokenizer = self._init_lang_model()

    @staticmethod
    def set_seed(seed: int) -> None:
        """Sets the random seed for consistent prompt generation.

        Args:
            seed (int): The random seed.
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _init_lang_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initializes the language model and tokenizer for prompt generation.

        Returns:
            tuple: The initialized language model and tokenizer.
        """
        logger.info(
            f"Initializing Qwen2.5-1.5B-Instruct language model on {self.device}..."
        )
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        if self.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def _contains_bad_words(self, texts: List[str]) -> bool:
        """Checks if a list of texts contain bad words.

        Args:
            texts (List[str]): List of texts to checks against bad words list.

        Returns:
            bool: True if any of the texts contain bad words, False otherwise.
        """
        return any(text.lower() in BAD_WORDS_LIST for text in texts)

    def _check_lm_safety(self, text: str) -> bool:
        """Checks if a text is safe using a language model.

        Args:
            text (str): Text to check for bad words.

        Returns:
            bool: True if the text is safe, False otherwise.
        """
        if self.use_lm:
            messages = [
                {
                    "role": "system",
                    "content": self.LLM_PROMPT,
                },
                {"role": "user", "content": text},
            ]
            processed_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.tokenizer([processed_text], return_tensors="pt").to(
                self.model.device
            )
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    top_k=None,
                    top_p=None,
                    temperature=None,
                )
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        model_inputs.input_ids, generated_ids
                    )
                ]
                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                return "inappropriate" not in response.lower().strip()
        return True

    def is_safe(self, classes: List[str]) -> bool:
        """Checks if a list of classes is safe.

        Args:
            classes (List[str]): List of classes to check for safety.

        Returns:
            bool: True if the classes are safe, False otherwise.
        """
        logger.info(f"Profanity filter is checking classes: {classes}")
        return not self._contains_bad_words(classes) and self._check_lm_safety(
            ",".join(classes)
        )

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache."""
        if self.use_lm:
            if self.device == "cuda":
                self.model = self.model.to("cpu")
            if empty_cuda_cache:
                with torch.no_grad():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Example usage of the class
    profanity_filter = ProfanityFilter(use_lm=True, device="cpu")
    classes_1 = ["cat", "fish", "dog", "ass", "person", "soldier", "war"]
    print(f"Are classes#1 {classes_1} safe: {profanity_filter.is_safe(classes_1)}")
    classes_2 = ["cat", "fish", "dog", "person", "soldier", "war"]
    print(f"Are classes#2 {classes_2} safe: {profanity_filter.is_safe(classes_2)}")
    profanity_filter.release()
