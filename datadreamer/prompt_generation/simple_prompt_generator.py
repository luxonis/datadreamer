from __future__ import annotations

import random
from typing import List

from datadreamer.prompt_generation.prompt_generator import PromptGenerator


class SimplePromptGenerator(PromptGenerator):
    """Prompt generator that creates simple prompts for text generation tasks.

    Args:
        class_names (List[str]): List of class names or objects for prompt generation.
        prompts_number (int): Number of prompts to generate.
        num_objects_range (Optional[List[int]]): Range for the number of objects to include in prompts.
        seed (Optional[float]): Seed for randomization.

    Methods:
        generate_prompts(): Generates a list of simple prompts.
        generate_prompt(selected_objects): Generates a single simple prompt based on selected objects.
        release(empty_cuda_cache): Releases resources (no action is taken in this implementation).
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the SimplePromptGenerator with class names and other settings."""
        super().__init__(*args, **kwargs)

    def generate_prompts(self) -> List[str]:
        """Generates a list of simple prompts.

        Returns:
            List[str]: A list of generated prompts in the form of "A photo of a {selected_objects}".
        """
        prompts = []
        for _ in range(self.prompts_number):
            selected_objects = random.sample(
                self.class_names, random.randint(*self.num_objects_range)
            )
            prompt_text = self.generate_prompt(selected_objects)
            prompts.append((selected_objects, prompt_text))
        return prompts

    def generate_prompt(self, selected_objects: List[str]) -> str:
        """Generates a single simple prompt based on selected objects.

        Args:
            selected_objects (List[str]): List of selected objects to include in the prompt.

        Returns:
            str: A simple prompt in the form of "A photo of a {selected_objects}".
        """
        return f"A photo of a {', a '.join(selected_objects)}"

    def release(self, empty_cuda_cache=False) -> None:
        """Releases resources (no action is taken in this implementation)."""
        pass


if __name__ == "__main__":
    class_names = ["dog", "cat", "bird", "tree", "car", "person", "house", "flower"]
    prompt_generator = SimplePromptGenerator(class_names, prompts_number=10)
    prompts = prompt_generator.generate_prompts()
    print(prompts)
    print("Done!")
