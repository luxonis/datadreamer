from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datadreamer.utils.bad_words import BAD_WORDS_LIST

logger = logging.getLogger(__name__)


class ProfanityFilter:
    """Class for filtering bad words from texts and checking if texts are safe.

    Attributes:
        seed (Optional[float]): Seed for randomization.
        device (str): Device to run the language model on ('cuda' for GPU, 'cpu' for CPU).
        use_lm (bool): Whether to use a language model for checking text safety.
        bad_words (List[str]): List of bad words to filter from texts.
        model (AutoModelForCausalLM): The pre-trained causal language model for checking text safety.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.

    Methods:
        set_seed(seed): Sets the random seed for consistent prompt generation.
        check_bad_words(texts): Filters bad words from a list of texts.
        _init_lang_model(): Initializes the language model and tokenizer.
        _is_bad_word(text): Checks if a text contains a bad word.
        _check_lm_safety(text): Checks if a text is safe using a language model.
        is_safe(text): Checks if a text is safe (does not contain bad words).
        release(empty_cuda_cache): Releases the model and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_lm: bool = True,
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

    @staticmethod
    def check_bad_words(texts: List[str]) -> bool:
        """Filters bad words from a list of texts.

        Args:
            texts (List[str]): List of texts to filter bad words from.

        Returns:
            bool: True if any of the texts contain bad words, False otherwise.
        """
        return any(text.lower() in BAD_WORDS_LIST for text in texts)

    def _init_lang_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initializes the language model and tokenizer for prompt generation.

        Returns:
            tuple: The initialized language model and tokenizer.
        """
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map=self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def _is_bad_word(self, text: str) -> bool:
        """Checks if a text contains a bad word.

        Args:
            text (str): Text to check for bad words.

        Returns:
            bool: True if the text contains a bad word, False otherwise.
        """
        return text.lower() in BAD_WORDS_LIST

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
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that classifies the given text as Safe or Unsafe. Respond with 'Safe' if the text is benign or represents an object that is not harmful such as planets like 'pluto', 'uran' or 'mercury'. Respond with 'Unsafe' if the text contains any of the following categories: drugs, harassment, hate, nudity or sexual, offensive words, self-harm, terrorism or extremism, toxic, violence, weapons. Here are unsafe examples: 'ass', 'a**', 'bitch', 'pussy', 'f**k'.",
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
                return response.lower() == "safe"
        return True

    def is_safe(self, text: str) -> bool:
        """Checks if a text is safe (does not contain bad words).

        Args:
            text (str): Text to check for bad words.

        Returns:
            bool: True if the text is safe, False otherwise.
        """
        return not self._is_bad_word(text) and self._check_lm_safety(text)

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
    texts = ["cat", "fish", "dog", "ass"]
    print(ProfanityFilter.check_bad_words(texts))
    for text in texts:
        print(
            f"Text: '{text}' is {'safe' if profanity_filter.is_safe(text) else 'unsafe'}!"
        )
    profanity_filter.release()