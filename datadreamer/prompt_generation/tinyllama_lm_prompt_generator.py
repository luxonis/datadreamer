import re
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datadreamer.prompt_generation.lm_prompt_generator import LMPromptGenerator


class TinyLlamaLMPromptGenerator(LMPromptGenerator):
    """A language model-based prompt generator class, extending PromptGenerator.

    Attributes:
        device (str): Device to run the language model on ('cuda' for GPU, 'cpu' for CPU).
        model (AutoModelForCausalLM): The pre-trained causal language model for generating prompts.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.

    Methods:
        _init_lang_model(): Initializes the language model and tokenizer.
        generate_prompts(): Generates a list of prompts based on the class names.
        _create_lm_prompt_text(selected_objects): Creates a text prompt for the language model.
        generate_prompt(prompt_text): Generates a single prompt using the language model.
        _test_prompt(prompt, selected_objects): Tests if the generated prompt is valid.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        class_names: List[str],
        prompts_number: int = 10,
        num_objects_range: Optional[List[int]] = None,
        seed: Optional[float] = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the LMPromptGenerator with class names and other settings."""
        super().__init__(class_names, prompts_number, num_objects_range, seed, device)

    def _init_lang_model(self):
        """Initializes the language model and tokenizer for prompt generation.

        Returns:
            tuple: The initialized language model and tokenizer.
        """
        if self.device == "cpu":
            print("Loading language model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            print("Loading language model on GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True
        )
        print("Done!")
        return model.to(self.device), tokenizer

    def remove_incomplete_sentence(self, text):
        # Define the regex pattern to capture up to the last sentence-ending punctuation
        pattern = r"^(.*[.!?])"
        match = re.search(pattern, text)
        return match.group(0) if match else text

    def remove_caption_sentences(self, text):
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

    def generate_prompt(self, prompt_text: str) -> str:
        """Generates a single prompt using the language model.

        Args:
            prompt_text (str): The text prompt for the language model.

        Returns:
            str: The generated prompt.
        """
        encoded_input = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **encoded_input,
            max_new_tokens=70,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_prompt = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        instructional_pattern = r"<\|system\|>\n.*?\n<\|user\|>\n.*?\n<\|assistant\|>\n"
        # Remove the instructional text to isolate the caption
        decoded_prompt = (
            re.sub(instructional_pattern, "", decoded_prompt)
            .replace('"', "")
            .replace("'", "")
        )

        return self.remove_caption_sentences(
            self.remove_incomplete_sentence(decoded_prompt)
        )


if __name__ == "__main__":
    # Example usage of the class
    object_names = ["aeroplane", "bicycle", "bird", "boat", "city"]
    prompt_generator = TinyLlamaLMPromptGenerator(
        class_names=object_names, prompts_number=5, device="cpu"
    )
    generated_prompts = prompt_generator.generate_prompts()
    for prompt in generated_prompts:
        print(prompt)
