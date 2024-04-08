from __future__ import annotations

from typing import List, Optional

import nltk
from nltk.corpus import wordnet

from datadreamer.prompt_generation.synonym_generator import SynonymGenerator

# Ensure that WordNet data is downloaded
nltk.download("wordnet")


class WordNetSynonymGenerator(SynonymGenerator):
    """Synonym generator that generates synonyms for a list of words using WordNet.

    Args:
        synonyms_number (int): Number of synonyms to generate for each word.
        seed (Optional[float]): Seed for randomization.
        device (str): Device to run the prompt generator on ('cuda' for GPU, 'cpu' for CPU).

    Methods:
        generate_synonyms(word): Generates synonyms for a single word and returns them in a list.
    """

    def __init__(
        self,
        synonyms_number: int = 3,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the SynonymGenerator with parameters."""
        super().__init__(synonyms_number, seed, device)

    def generate_synonyms(self, word: str) -> List[str]:
        """Generates synonyms for a single word and returns them in a list.

        Args:
            word (str): The word for which synonyms are generated.

        Returns:
            List[str]: A list of generated synonyms for the word.
        """
        synonyms = set()
        for syn in wordnet.synsets(word, pos=wordnet.NOUN):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        return list(synonyms)[: self.synonyms_number]

    def release(self, empty_cuda_cache: bool = False) -> None:
        """Releases resources (no action is taken in this implementation)."""
        pass


if __name__ == "__main__":
    # Example usage
    generator = WordNetSynonymGenerator(synonyms_number=3)
    synonyms = generator.generate_synonyms_for_list(
        ["astronaut", "cat", "dog", "person", "horse"]
    )
    print(synonyms)
    generator.save_synonyms(synonyms, "synonyms.json")
