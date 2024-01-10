from datadreamer.prompt_generation.lm_prompt_generator import LMPromptGenerator
from datadreamer.prompt_generation.simple_prompt_generator import SimplePromptGenerator
import pytest
import torch

from datadreamer.prompt_generation.synonym_generator import SynonymGenerator


def test_simple_prompt_generator():
    class_names = ["dog", "cat", "bird", "tree", "car", "person", "house", "flower"]
    prompt_generator = SimplePromptGenerator(class_names, prompts_number=10)
    prompts = prompt_generator.generate_prompts()
    # Check that the some prompts were generated
    assert len(prompts) > 0
    # Iterate through the prompts
    for selected_objects, prompt_text in prompts:
        # Selected objects aren't empty
        assert len(selected_objects) > 0
        # The slected objects are in the range
        assert (
            prompt_generator.num_objects_range[0]
            <= len(selected_objects)
            <= prompt_generator.num_objects_range[1]
        )
        # Check the generated text
        assert prompt_text == f"A photo of a {', a '.join(selected_objects)}"


def test_lm_prompt_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_names = ["aeroplane", "bicycle", "bird", "boat"]
    prompt_generator = LMPromptGenerator(
        class_names=object_names, prompts_number=2, device=device
    )
    prompts = prompt_generator.generate_prompts()
    # Check that the some prompts were generated
    assert len(prompts) > 0
    # Iterate through the prompts
    for selected_objects, prompt_text in prompts:
        # Selected objects aren't empty
        assert len(selected_objects) > 0
        # The slected objects are in the range
        assert (
            prompt_generator.num_objects_range[0]
            <= len(selected_objects)
            <= prompt_generator.num_objects_range[1]
        )
        # Check the generated text
        assert len(prompt_text) > 0 and any(
            [x in prompt_text for x in selected_objects]
        )
    prompt_generator.release(empty_cuda_cache=True if device != "cpu" else False)


def test_synonym_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    synonyms_num = 3
    generator = SynonymGenerator(
        "mistralai/Mistral-7B-Instruct-v0.1",
        synonyms_number=synonyms_num,
        device=device,
    )
    synonyms = generator.generate_synonyms_for_list(["astronaut", "cat", "dog"])
    # Check that the some synonyms were generated
    assert len(synonyms) > 0
    # Iterate through the synonyms
    for word, synonym_list in synonyms.items():
        # Check that the word is not empty
        assert len(word) > 0
        # Check that the synonym list is not empty and has the correct number of synonyms
        assert len(synonym_list) > 0 and len(synonym_list) == synonyms_num
        # Check that the synonyms are not empty
        for synonym in synonym_list:
            assert len(synonym) > 0
    generator.release(empty_cuda_cache=True if device != "cpu" else False)


if __name__ == "__main__":
    pytest.main()
