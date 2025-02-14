from __future__ import annotations

import subprocess

import pytest


def _check_wrong_argument_choice(cmd: str):
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(cmd, shell=True)


def _check_wrong_value(cmd: str, expected_message: str = None):
    if expected_message:
        with pytest.raises(ValueError, match=expected_message):
            try:
                subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise ValueError(e.output.decode()) from e
    else:
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


def test_dataset_plugin_without_luxonis_format():
    cmd = (
        "datadreamer --dataset_plugin custom_plugin --dataset_format yolo --device cpu"
    )
    expected_message = (
        "--dataset_format must be 'luxonis-dataset' if --dataset_plugin is specified"
    )
    _check_wrong_value(cmd, expected_message)


def test_dataset_plugin_with_invalid_plugin():
    cmd = "datadreamer --dataset_plugin unknown_plugin --dataset_format luxonis-dataset --device cpu"
    expected_message = (
        "Dataset plugin 'unknown_plugin' is not registered in DATASETS_REGISTRY"
    )
    _check_wrong_value(cmd, expected_message)
