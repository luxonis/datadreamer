import random
from typing import List, Optional

from basereduce.prompt_generation.prompt_generator import PromptGenerator


class SimplePromptGenerator(PromptGenerator):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def generate_prompts(self) -> List[str]:
        prompts = []
        for _ in range(self.prompts_number):
            selected_objects = random.sample(
                self.class_names, random.randint(*self.num_objects_range)
            )
            prompt_text = self.generate_prompt(selected_objects)
            prompts.append((selected_objects, prompt_text))
        return prompts

    def generate_prompt(self, selected_objects: List[str]) -> str:
        return f"A photo of a {', a '.join(selected_objects)}"

    def release(self, empty_cuda_cache=False) -> None:
        pass


if __name__ == "__main__":
    class_names = ["dog", "cat", "bird", "tree", "car", "person", "house", "flower"]
    prompt_generator = SimplePromptGenerator(class_names, prompts_number=10)
    prompts = prompt_generator.generate_prompts()
    print(prompts)
    # prompt_generator.save_prompts(prompts, "simple_prompts.json")
    # prompt_generator.release()
    print("Done!")
