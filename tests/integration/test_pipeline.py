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
        "generation_args.json",
        "image_0.jpg",
        "prompts.json",
    ]
    # Check that all the files were created
    for file in files:
        assert os.path.isfile(os.path.join(target_folder, file)), f"{file} not created"
    # Check that the "bboxes_visualization" folder was created
    assert os.path.isdir(
        os.path.join(target_folder, "bboxes_visualization")
    ), "bboxes_visualization directory not created"


def _check_wrong_argument_choice(cmd: str):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(cmd, shell=True)


def _check_wrong_value(cmd: str):
    with pytest.raises(ValueError):
        try:
            subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise ValueError(e.output.decode()) from e


# =========================================================
# ARGUMENTS CHECKS
# =========================================================
def test_invalid_task_value():
    # Define the cmd
    cmd = "datadreamer --task invalid_task"
    _check_wrong_argument_choice(cmd)


def test_invalid_prompts_number_type():
    # Define the cmd
    cmd = "datadreamer --prompts_number value"
    _check_wrong_argument_choice(cmd)


def test_invalid_num_objects_range_type():
    # Define the cmd
    cmd = "datadreamer --num_objects_range value"
    _check_wrong_argument_choice(cmd)


def test_invalid_conf_threshold_range_type():
    # Define the cmd
    cmd = "datadreamer --conf_threshold value"
    _check_wrong_argument_choice(cmd)


def test_invalid_image_tester_patience_type():
    # Define the cmd
    cmd = "datadreamer --image_tester_patience value"
    _check_wrong_argument_choice(cmd)


def test_invalid_seed_type():
    # Define the cmd
    cmd = "datadreamer --seed value --device cpu"
    _check_wrong_argument_choice(cmd)


def test_invalid_prompt_generator():
    # Define the cmd
    cmd = "datadreamer --prompt_generator invalide_value"
    _check_wrong_argument_choice(cmd)


def test_invalid_image_generator():
    # Define the cmd
    cmd = "datadreamer --image_generator invalide_value"
    _check_wrong_argument_choice(cmd)


def test_invalid_image_annotator():
    # Define the cmd
    cmd = "datadreamer --image_annotator invalide_value"
    _check_wrong_argument_choice(cmd)


def test_invalid_det_image_annotator():
    # Define the cmd
    cmd = "datadreamer --image_annotator clip"
    _check_wrong_argument_choice(cmd)


def test_invalid_clf_image_annotator():
    # Define the cmd
    cmd = "datadreamer --image_annotator owlv2 --task classification"
    _check_wrong_argument_choice(cmd)


def test_invalid_device():
    # Define the cmd
    cmd = "datadreamer --device invalide_value"
    _check_wrong_argument_choice(cmd)


def test_invalid_annotator_size():
    # Define the cmd
    cmd = "datadreamer --annotator_size invalide_value"
    _check_wrong_argument_choice(cmd)


def test_empty_class_names():
    # Define the cmd
    cmd = "datadreamer --class_names []"
    _check_wrong_value(cmd)


def test_empty_negative_prompt():
    # Define the cmd
    cmd = "datadreamer --negative_prompt --device cpu"
    _check_wrong_value(cmd)


def test_empty_prompt_suffix():
    # Define the cmd
    cmd = "datadreamer --prompt_suffix --device cpu"
    _check_wrong_value(cmd)


def test_invalid_class_names():
    # Define the cmd
    cmd = "datadreamer --class_names [2, -1]"
    _check_wrong_value(cmd)


def test_invalid_prompts_number():
    # Define the cmd
    cmd = "datadreamer --prompts_number -1"
    _check_wrong_value(cmd)


def test_negative_conf_threshold():
    # Define the cmd
    cmd = "datadreamer --conf_threshold -1"
    _check_wrong_value(cmd)


def test_big_conf_threshold():
    # Define the cmd
    cmd = "datadreamer --conf_threshold 10"
    _check_wrong_value(cmd)


def test_negative_annotation_iou_threshold():
    # Define the cmd
    cmd = "datadreamer --annotation_iou_threshold -1"
    _check_wrong_value(cmd)


def test_big_annotation_iou_threshold():
    # Define the cmd
    cmd = "datadreamer --annotation_iou_threshold 10"
    _check_wrong_value(cmd)


def test_invalid_image_tester_patience():
    # Define the cmd
    cmd = "datadreamer --image_tester_patience -1"
    _check_wrong_value(cmd)


def test_invalid_seed():
    # Define the cmd
    cmd = "datadreamer --seed -1 --device cpu"
    _check_wrong_value(cmd)


def test_invalid_synonym_generator():
    # Define the cmd
    cmd = "datadreamer --device cpu --synonym_generator invalid"
    _check_wrong_value(cmd)


def test_invalid_lm_quantization():
    # Define the cmd
    cmd = "datadreamer --device cude --lm_quantization invalid"
    _check_wrong_value(cmd)


def test_invalid_device_lm_quantization():
    # Define the cmd
    cmd = "datadreamer --device cpu --lm_quantization 4bit"
    _check_wrong_value(cmd)


def test_invalid_batch_size_prompt():
    # Define the cmd
    cmd = "datadreamer --batch_size_prompt -1"
    _check_wrong_value(cmd)


def test_invalid_batch_size_annotation():
    # Define the cmd
    cmd = "datadreamer --batch_size_annotation -1"
    _check_wrong_value(cmd)


def test_invalid_batch_size_image():
    # Define the cmd
    cmd = "datadreamer --batch_size_image -1"
    _check_wrong_value(cmd)


def test_invalid_num_objects_range():
    # Define the cmd
    cmd = "datadreamer --num_objects_range 1"
    _check_wrong_value(cmd)


def test_many_num_objects_range():
    # Define the cmd
    cmd = "datadreamer --num_objects_range 1 2 3"
    _check_wrong_value(cmd)


def test_desc_num_objects_range():
    # Define the cmd
    cmd = "datadreamer --num_objects_range 3 1"
    _check_wrong_value(cmd)


def test_negative_num_objects_range():
    # Define the cmd
    cmd = "datadreamer --num_objects_range -3 1"
    _check_wrong_value(cmd)


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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl "
        f"--use_image_tester "
        f"--device cuda"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    total_memory < 16 or total_disk_space < 35,
    reason="Test requires at least 16GB of RAM and 35GB of HDD",
)
def test_cpu_simple_sdxl_lightning_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cpu-simple-sdxl-lightning/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien mars cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-lightning "
        f"--use_image_tester "
        f"--device cpu"
    )
    # Check the run of the pipeline
    _check_detection_pipeline(cmd, target_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_memory < 16 or total_disk_space < 35,
    reason="Test requires GPU, at least 16GB of RAM and 35GB of HDD",
)
def test_cuda_simple_sdxl_lightning_detection_pipeline():
    # Define target folder
    target_folder = "data/data-det-cuda-simple-sdxl-lightning/"
    # Define the command to run the datadreamer
    cmd = (
        f"datadreamer --save_dir {target_folder} "
        f"--class_names alien mars cat "
        f"--prompts_number 1 "
        f"--prompt_generator simple "
        f"--num_objects_range 1 2 "
        f"--image_generator sdxl-lightning "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
# CLASSIFICATION - SIMPLE LM
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
# CLASSIFICATION - LLM
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
# CLASSIFICATION - TinyLlama LLM
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
        f"--class_names alien mars cat "
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
