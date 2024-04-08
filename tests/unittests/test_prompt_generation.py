from __future__ import annotations

import psutil
import pytest
import torch

from datadreamer.prompt_generation.lm_prompt_generator import LMPromptGenerator
from datadreamer.prompt_generation.lm_synonym_generator import LMSynonymGenerator
from datadreamer.prompt_generation.simple_prompt_generator import SimplePromptGenerator
from datadreamer.prompt_generation.tinyllama_lm_prompt_generator import (
    TinyLlamaLMPromptGenerator,
)
from datadreamer.prompt_generation.wordnet_synonym_generator import (
    WordNetSynonymGenerator,
)

# Get the total memory in GB
total_memory = psutil.virtual_memory().total / (1024**3)
# Get the total disk space in GB
total_disk_space = psutil.disk_usage("/").total / (1024**3)


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


def _check_lm_prompt_generator(
    device: str, prompt_generator_class=LMPromptGenerator, quantization: str = "none"
):
    object_names = ["aeroplane", "bicycle", "bird", "boat"]
    prompt_generator = prompt_generator_class(
        class_names=object_names,
        prompts_number=2,
        device=device,
        quantization=quantization,
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
        assert len(prompt_text) > 0 and prompt_text.lower().startswith("a photo of")
    prompt_generator.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM, 35GB of HDD and CUDA support",
)
def test_cuda_lm_prompt_generator():
    _check_lm_prompt_generator("cuda")


@pytest.mark.skipif(
    total_memory < 12 or not torch.cuda.is_available() or total_disk_space < 25,
    reason="Test requires at least 12GB of RAM, 25GB of HDD and CUDA support",
)
def test_cuda_4bit_lm_prompt_generator():
    _check_lm_prompt_generator("cuda", quantization="4bit")


@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 35,
    reason="Test requires at least 28GB of RAM and 35GB of HDD for running on CPU",
)
def test_cpu_lm_prompt_generator():
    _check_lm_prompt_generator("cpu")


@pytest.mark.skipif(
    total_memory < 8 or not torch.cuda.is_available() or total_disk_space < 12,
    reason="Test requires at least 8GB of RAM, 12GB of HDD and CUDA support",
)
def test_cuda_tinyllama_lm_prompt_generator():
    _check_lm_prompt_generator("cuda", TinyLlamaLMPromptGenerator)


@pytest.mark.skipif(
    total_memory < 12 or total_disk_space < 12,
    reason="Test requires at least 12GB of RAM and 12GB of HDD for running on CPU",
)
def test_cpu_tinyllama_lm_prompt_generator():
    _check_lm_prompt_generator("cpu", TinyLlamaLMPromptGenerator)


def _check_synonym_generator(device: str, synonym_generator_class=LMSynonymGenerator):
    synonyms_num = 3
    generator = synonym_generator_class(synonyms_number=synonyms_num, device=device)
    synonyms = generator.generate_synonyms_for_list(["astronaut", "cat", "dog"])
    # Check that the some synonyms were generated
    assert len(synonyms) > 0
    # Iterate through the synonyms
    for word, synonym_list in synonyms.items():
        # Check that the word is not empty
        assert len(word) > 0
        # Check that the synonym list is not empty
        assert len(synonym_list) > 0
        # Check that the synonyms are not empty
        for synonym in synonym_list:
            assert len(synonym) > 0
    generator.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM, 35GB of HDD and CUDA support",
)
def test_cuda_synonym_generator():
    _check_synonym_generator("cuda")


@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 35,
    reason="Test requires at least 28GB of RAM and 35GB of HDD for running on CPU",
)
def test_cpu_synonym_generator():
    _check_synonym_generator("cpu")


def test_cpu_wordnet_synonym_generator():
    _check_synonym_generator("cpu", WordNetSynonymGenerator)


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="Test requires CUDA support",
)
def test_cuda_wordnet_synonym_generator():
    _check_synonym_generator("cuda", WordNetSynonymGenerator)
