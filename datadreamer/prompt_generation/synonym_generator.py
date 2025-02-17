from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from tqdm import tqdm


# Abstract base class for synonym generation
class SynonymGenerator(ABC):
    """Abstract base class for synonym generation.

    Attributes:
        synonyms_number (int): Number of synonyms to generate for each word.
        seed (Optional[float]): Seed for randomization.
        device (str): Device to run the prompt generator on ('cuda' for GPU, 'cpu' for CPU).

    Methods:
        _init_lang_model(): Initializes the language model and tokenizer.
        _generate_synonyms(prompt_text): Generates synonyms based on a given prompt text.
        _extract_synonyms(text): Extracts synonyms from a text containing synonyms.
        _create_prompt_text(word): Creates a prompt text for generating synonyms for a given word.
        generate_synonyms_for_list(words): Generates synonyms for a list of words and returns them in a dictionary.
        generate_synonyms(word): Generates synonyms for a single word and returns them in a list.
        save_synonyms(synonyms, save_path): Saves the generated synonyms to a JSON file.
        release(empty_cuda_cache): Releases resources (no action is taken in this implementation).
    """

    def __init__(
        self,
        synonyms_number: int = 3,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the SynonymGenerator with parameters."""
        self.synonyms_number = synonyms_number
        self.seed = seed
        self.device = device

    def generate_synonyms_for_list(self, words: List[str]) -> Dict:
        """Generates synonyms for a list of words and returns them in a dictionary.

        Args:
            words (List[str]): List of words for which synonyms are generated.

        Returns:
            dict: A dictionary where each word is associated with a list of its most common synonyms.
        """
        synonyms_dict = {}
        for word in tqdm(words, desc="Generating synonyms"):
            synonyms = self.generate_synonyms(word)
            synonyms_dict[word] = synonyms
        return synonyms_dict

    def save_synonyms(self, synonyms: Dict, save_path: str) -> None:
        """Saves the generated synonyms to a JSON file.

        Args:
            synonyms: The synonyms to save (typically a dictionary).
            save_path (str): The path to the JSON file where synonyms will be saved.
        """
        with open(save_path, "w") as f:
            json.dump(synonyms, f)

    @abstractmethod
    def generate_synonyms(self, word: str) -> List[str]:
        """Generates synonyms for a single word and returns them in a list.

        Args:
            word (str): The word for which synonyms are generated.

        Returns:
            List[str]: A list of generated synonyms for the word.
        """
        pass

    @abstractmethod
    def release(self, empty_cuda_cache=False) -> None:
        """Abstract method to release resources (must be implemented in subclasses)."""
        pass
