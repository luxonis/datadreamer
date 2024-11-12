from __future__ import annotations

import os
import subprocess

import psutil
import pytest
import torch

# Get the total memory in GB
total_memory = psutil.virtual_memory().total / (1024**3)
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
        f"--class_names alien mars cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-turbo "
        f"--use_image_tester "
        f"--synonym_generator wordnet "
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
        f"--class_names alien mars cat "
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


# =========================================================
# TEST WITH CONFIG FILE
# =========================================================
@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_turbo_config_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-simple-sdxl-turbo-config/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--num_objects_range 1 2 "
        f"--config ./tests/core_tests/integration/sample_config.yaml "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_turbo_config_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-simple-sdxl-turbo-config/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--num_objects_range 1 2 "
        f"--config ./tests/core_tests/integration/sample_config.yaml "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_turbo_config_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cpu-simple-sdxl-turbo-config/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--config ./tests/core_tests/integration/sample_config.yaml "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_turbo_config_classification_pipeline():
    # Define target folder
    target_folder = "data/data-cls-cuda-simple-sdxl-turbo-config/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task classification "
        f"--save_dir {target_folder} "
        f"--num_objects_range 1 2 "
        f"--image_annotator clip "
        f"--config ./tests/core_tests/integration/sample_config.yaml "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_turbo_config_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cpu-simple-sdxl-turbo-config/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--config ./tests/core_tests/integration/sample_config.yaml "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_turbo_config_instance_segmentation_pipeline():
    # Define target folder
    target_folder = "data/data-inst-seg-cuda-simple-sdxl-turbo-config/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --task instance-segmentation "
        f"--save_dir {target_folder} "
        f"--num_objects_range 1 2 "
        f"--image_annotator owlv2-slimsam "
        f"--config ./tests/core_tests/integration/sample_config.yaml "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)
