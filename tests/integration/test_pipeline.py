import pytest
import torch
import os
import subprocess


def test_detection_pipeline():
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    cmd = f"datadreamer --save_dir data/data-det/ --class_names alien mars cat --prompts_number 2 --prompt_generator simple --num_objects_range 1 2 --image_generator sdxl-turbo --device {device}"
    
    result = subprocess.run(cmd, shell=True)
    assert result.returncode == 0, "Command failed to run"

    assert os.path.isdir("data/data-det"), "Directory not created"
    
    files = ["annotations.json", "generation_args.json", "image_1.jpg", "image_0.jpg", "prompts.json"]
    for file in files:
        assert os.path.isfile(os.path.join("data/data_cls", file)), f"{file} not created"
    
    assert os.path.isdir("data/data_cls/bboxes_visualization"), "bboxes_visualization directory not created"


def test_classification_pipeline():
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    cmd = f"datadreamer --task classification --save_dir data/data-cls/ --class_names alien mars cat --prompts_number 2 --prompt_generator simple --num_objects_range 1 2 --image_generator sdxl-turbo --device {device}"
    
    result = subprocess.run(cmd, shell=True)
    assert result.returncode == 0, "Command failed to run"

    assert os.path.isdir("data/data-cls"), "Directory not created"
    
    files = ["annotations.json", "generation_args.json", "image_1.jpg", "image_0.jpg", "prompts.json"]
    for file in files:
        assert os.path.isfile(os.path.join("data/data_cls", file)), f"{file} not created"
    
    assert os.path.isdir("data/data_cls/bboxes_visualization"), "bboxes_visualization directory not created"


if __name__ == "__main__":
    pytest.main()
