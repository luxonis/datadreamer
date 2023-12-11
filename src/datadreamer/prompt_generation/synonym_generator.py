import json
import re
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class SynonymGenerator:
    """Synonym generator that generates synonyms for a list of words using a language
    model.

    Args:
        synonyms_number (int): Number of synonyms to generate for each word.
        seed (Optional[float]): Seed for randomization.
        device (str): Device for model inference (default is "cuda").

    Methods:
        generate_synonyms_for_list(words): Generates synonyms for a list of words and returns them in a dictionary.
        generate_synonyms(word): Generates synonyms for a single word and returns them in a list.
        save_synonyms(synonyms, save_path): Saves the generated synonyms to a JSON file.
        release(empty_cuda_cache): Releases resources (no action is taken in this implementation).
    """

    def __init__(
        self,
        synonyms_number: int = 5,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the SynonymGenerator with parameters."""
        self.synonyms_number = synonyms_number
        self.seed = seed
        self.device = device
        self.model, self.tokenizer = self._init_lang_model()

    def _init_lang_model(self):
        """Initializes the language model and tokenizer for synonym generation."""
        print("Initializing language model for synonym generation")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        return model, tokenizer

    def generate_synonyms_for_list(self, words: List[str]) -> dict:
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
        print("Synonyms generated")
        return synonyms_dict

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

    def _create_prompt_text(self, word: str) -> str:
        """Creates a prompt text for generating synonyms for a given word.

        Args:
            word (str): The word for which synonyms are generated.

        Returns:
            str: The prompt text for generating synonyms.
        """
        return f"[INST] List {self.synonyms_number} most common synonyms for the word '{word}'. Write only synonyms separated by commas. [/INST]"

    def _generate_synonyms(self, prompt_text: str) -> List[str]:
        """Generates synonyms based on a given prompt text.

        Args:
            prompt_text (str): The prompt text for generating synonyms.

        Returns:
            List[str]: A list of generated synonyms.
        """
        encoded_input = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **encoded_input,
            max_new_tokens=50,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

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

    def save_synonyms(self, synonyms, save_path: str) -> None:
        """Saves the generated synonyms to a JSON file.

        Args:
            synonyms: The synonyms to save (typically a dictionary).
            save_path (str): The path to the JSON file where synonyms will be saved.
        """
        with open(save_path, "w") as f:
            json.dump(synonyms, f)

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
    generator = SynonymGenerator(
        "mistralai/Mistral-7B-Instruct-v0.1", synonyms_number=3
    )
    synonyms = generator.generate_synonyms_for_list(
        ["astronaut", "cat", "dog", "person", "horse"]
    )
    print(synonyms)
    # generator.save_synonyms(synonyms, "synonyms.json")
    # generator.release()
