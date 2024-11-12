from __future__ import annotations

import os
import subprocess

import psutil
import pytest
import torch

# Get the total memory in GB
total_memory = psutil.virtual_memory().total / (1024 * 3)
# Get the total disk space in GB
total_disk_space = psutil.disk_usage("/").total / (1024**3)


def _check_detection_pipeline(cmd: str, target_folder: str):
    # Run the command
    result = subprocess.run(cmd, shell=True)
    assert result.returncode == 0, "Command failed to run"
    # Check that the target folder is a folder
    assert os.path.isdir(target_folder), "Directory not created"
    files = [
        "annotations.json",
        "generation_args.yaml",
        "prompts.json",
    ]
    # Check that all the files were created
    for file in files:
        assert os.path.isfile(os.path.join(target_folder, file)), f"{file} not created"
    # Check that an image with an unique was created
    assert (
        len(
            list(
                filter(
                    lambda x: "image_" in x and ".jpg" in x, os.listdir(target_folder)
                )
            )
        )
        > 0
    ), "Images not created"
    # Check that the "bboxes_visualization" folder was created
    assert os.path.isdir(
        os.path.join(target_folder, "bboxes_visualization")
    ), "bboxes_visualization directory not created"


# =========================================================
# DETECTION - SIMPLE LM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-simple-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-simple-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 55,
    reason="Test requires GPU, at least 16GB of RAM and 55GB of HDD",
)
def test_cuda_simple_llm_synonym_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-simple-llm-synonym-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--synonym_generator llm "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_wordnet_synonym_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-simple-wordnet-synonym-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--synonym_generator wordnet "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-simple-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-simple-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# DETECTION - LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 55,
    reason="Test requires at least 32GB of RAM and 55GB of HDD for running on CPU",
)
def test_cpu_lm_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 55,
    reason="Test requires at least 16GB of RAM, CUDA support and 55GB of HDD",
)
def test_cuda_lm_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 14 or not torch.cuda.is_available() or total_disk_space < 45,
    reason="Test requires at least 14GB of RAM, CUDA support and 45GB of HDD",
)
def test_cuda_4bit_lm_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-4bit-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--lm_quantization 4bit "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 55,
    reason="Test requires at least 32GB of RAM and 55GB of HDD for running on CPU",
)
def test_cpu_lm_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 55,
    reason="Test requires at least 16GB of RAM, CUDA support and 55GB of HDD",
)
def test_cuda_lm_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 14 or not torch.cuda.is_available() or total_disk_space < 45,
    reason="Test requires at least 14GB of RAM, CUDA support and 45GB of HDD",
)
def test_cuda_4bit_lm_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-4bit-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--lm_quantization 4bit "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# DETECTION - TinyLlama LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_tiny_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-tiny-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_tiny_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-tiny-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_tiny_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-tiny-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_tiny_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-tiny-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# DETECTION - Qwen2.5 LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_qwen2_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-qwen2-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_qwen2_sdxl_turbo_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-qwen2-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_qwen2_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-qwen2-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_qwen2_sdxl_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-qwen2-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# INSTANCE SEGMENTATION - SIMPLE LM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-simple-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-simple-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 55,
    reason="Test requires GPU, at least 16GB of RAM and 55GB of HDD",
)
def test_cuda_simple_llm_synonym_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-simple-llm-synonym-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--image_annotator clip "
        f"--use_image_tester "
        f"--synonym_generator llm "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_wordnet_synonym_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-simple-wordnet-synonym-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--synonym_generator wordnet "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-simple-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--image_annotator clip "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-simple-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--image_annotator clip "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# INSTANCE SEGMENTATION - LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 55,
    reason="Test requires at least 32GB of RAM and 55GB of HDD for running on CPU",
)
def test_cpu_lm_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 55,
    reason="Test requires at least 16GB of RAM, 55GB of HDD and CUDA support",
)
def test_cuda_lm_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 14 or not torch.cuda.is_available() or total_disk_space < 45,
    reason="Test requires at least 14GB of RAM, 45GB of HDD and CUDA support",
)
def test_cuda_4bit_lm_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-4bit-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--lm_quantization 4bit "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 55,
    reason="Test requires at least 32GB of RAM and 55GB of HDD for running on CPU",
)
def test_cpu_lm_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--image_annotator clip "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 55,
    reason="Test requires at least 16GB of RAM, CUDA support and 55GB of HDD",
)
def test_cuda_lm_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--image_annotator clip "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 14 or not torch.cuda.is_available() or total_disk_space < 45,
    reason="Test requires at least 14GB of RAM, CUDA support and 45GB of HDD",
)
def test_cuda_4bit_lm_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-4bit-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--lm_quantization 4bit "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# INSTANCE SEGMENTATION - TinyLlama LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_tiny_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-tiny-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--image_annotator clip "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_tiny_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-tiny-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_tiny_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-tiny-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_tiny_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-tiny-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# CLASSIFICATION - Qwen2.5 LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_qwen2_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-qwen2-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--image_annotator clip "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_qwen2_sdxl_turbo_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-qwen2-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_qwen2_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-qwen2-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_qwen2_sdxl_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-qwen2-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# INSTANCE SEGMENTATION - SIMPLE LM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-simple-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-simple-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 55,
    reason="Test requires GPU, at least 16GB of RAM and 55GB of HDD",
)
def test_cuda_simple_llm_synonym_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-simple-llm-synonym-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--image_annotator owlv2-slimsam "
        f"--use_image_tester "
        f"--synonym_generator llm "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_wordnet_synonym_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-simple-wordnet-synonym-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--synonym_generator wordnet "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-simple-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--image_annotator owlv2-slimsam "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-simple-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--image_annotator owlv2-slimsam "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# INSTANCE SEGMENTATION - LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 55,
    reason="Test requires at least 32GB of RAM and 55GB of HDD for running on CPU",
)
def test_cpu_lm_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 55,
    reason="Test requires at least 16GB of RAM, 55GB of HDD and CUDA support",
)
def test_cuda_lm_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 14 or not torch.cuda.is_available() or total_disk_space < 45,
    reason="Test requires at least 14GB of RAM, 45GB of HDD and CUDA support",
)
def test_cuda_4bit_lm_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-4bit-lm-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--lm_quantization 4bit "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 32 or total_disk_space < 55,
    reason="Test requires at least 32GB of RAM and 55GB of HDD for running on CPU",
)
def test_cpu_lm_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--image_annotator owlv2-slimsam "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or not torch.cuda.is_available() or total_disk_space < 55,
    reason="Test requires at least 16GB of RAM, CUDA support and 55GB of HDD",
)
def test_cuda_lm_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--image_annotator owlv2-slimsam "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 14 or not torch.cuda.is_available() or total_disk_space < 45,
    reason="Test requires at least 14GB of RAM, CUDA support and 45GB of HDD",
)
def test_cuda_4bit_lm_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-4bit-lm-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator lm "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--lm_quantization 4bit "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# INSTANCE SEGMENTATION - TinyLlama LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_tiny_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-tiny-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--image_annotator owlv2-slimsam "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_tiny_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-tiny-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_tiny_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-tiny-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_tiny_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-tiny-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator tiny "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


# =========================================================
# INSTANCE SEGMENTATION - Qwen2.5 LLM
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_qwen2_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-qwen2-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--image_annotator owlv2-slimsam "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_qwen2_sdxl_turbo_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-qwen2-sdxl-turbo/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_qwen2_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-qwen2-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_qwen2_sdxl_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-qwen2-sdxl/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--class_names alien bear cat "
        f"--prompts_number 1 "
        f"--prompt_generator qwen2 "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)
