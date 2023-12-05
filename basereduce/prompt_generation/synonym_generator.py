

from typing import List, Optional
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SynonymGenerator:
    def __init__(
        self,
        synonyms_number: int = 5,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        self.synonyms_number = synonyms_number
        self.seed = seed
        self.device = device
        self.model, self.tokenizer = self._init_lang_model()

    def _init_lang_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            torch_dtype=torch.float16
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer
    
    def generate_synonyms_for_list(self, words: List[str]) -> dict:
        synonyms_dict = {}
        for word in words:
            synonyms = self.generate_synonyms(word)
            synonyms_dict[word] = synonyms
        return synonyms_dict

    def generate_synonyms(self, word: str) -> List[str]:
        prompt_text = self._create_prompt_text(word)
        generated_synonyms = self._generate_synonyms(prompt_text)
        return generated_synonyms

    def _create_prompt_text(self, word: str) -> str:
        return f"[INST] List {self.synonyms_number} most common synonyms for the word '{word}'. Write only synonyms separated by commas. [/INST]"

    def _generate_synonyms(self, prompt_text: str) -> List[str]:
        encoded_input = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **encoded_input, max_new_tokens=50, do_sample=True, num_return_sequences=1
        )
        generated_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        print(generated_text)

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
        # Assuming the output is in the format "Synonyms: word1, word2, word3"
        #synonyms_text = text.split(':')[-1]  # Get the part after "Synonyms:"
        synonyms = [word.strip() for word in text.split(',')]  # Split and strip each synonym
        return synonyms[:self.synonyms_number]

    def save_synonyms(self, synonyms: List[str], save_path: str) -> None:
        with open(save_path, "w") as f:
            json.dump(synonyms, f)

    def release(self, empty_cuda_cache=False) -> None:
        self.model = self.model.to('cpu')
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example usage
    generator = SynonymGenerator("mistralai/Mistral-7B-Instruct-v0.1", synonyms_number=3)
    synonyms = generator.generate_synonyms_for_list(["astronaut", "cat", "dog", "person", "horse"])
    print(synonyms)
    #generator.save_synonyms(synonyms, "synonyms.json")
    #generator.release()

