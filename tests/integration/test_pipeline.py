import pytest
import torch


def test_detection_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = "datadreamer --save_dir data-det/ --class_names alien mars cat --prompts_number 2 --prompt_generator simple --num_objects_range 1 2 --image_generator sdxl-turbo --device cpu"
    pass

def test_classification_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = "datadreamer --task classification --save_dir data_cls/ --class_names alien mars cat --prompts_number 2 --prompt_generator simple --num_objects_range 1 2 --image_generator sdxl-turbo --device cpu"
    pass


if __name__ == "__main__":
    pytest.main()
