from __future__ import annotations

from typing import Annotated, List

from luxonis_ml.utils import LuxonisConfig
from pydantic import Field, model_validator


class Config(LuxonisConfig):
    # General arguments
    save_dir: str = "generated_dataset"
    class_names: List[str] = ["bear", "bicycle", "bird", "person"]
    prompts_number: int = 10
    task: str = "detection"
    seed: int = 42
    device: str = "cuda"
    annotate_only: bool = False
    dataset_format: str = "raw"
    split_ratios: Annotated[
        List[float], Field(default=[0.8, 0.1, 0.1], min_length=3, max_length=3)
    ] = [0.8, 0.1, 0.1]
    # Prompt generation arguments
    prompt_generator: str = "simple"
    synonym_generator: str = "none"
    num_objects_range: Annotated[
        List[int], Field(default=[1, 3], min_length=2, max_length=2)
    ] = [1, 3]
    lm_quantization: str = "none"
    batch_size_prompt: int = 64
    # Image generation arguments
    image_generator: str = "sdxl-turbo"
    prompt_prefix: str = ""
    prompt_suffix: str = ", hd, 8k, highly detailed"
    negative_prompt: str = "cartoon, blue skin, painting, scrispture, golden, illustration, worst quality, low quality, normal quality:2, unrealistic dream, low resolution, static, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, bad anatomy"
    batch_size_image: int = 1
    use_image_tester: bool = False
    image_tester_patience: int = 1
    # Annotation arguments
    image_annotator: str = "owlv2"
    conf_threshold: float = 0.15
    annotation_iou_threshold: float = 0.2
    use_tta: bool = False
    annotator_size: str = "base"
    batch_size_annotation: int = 1

    @model_validator(mode="after")
    def check_option_args(self):
        print("Checking option args")
        if self.task not in ["detection", "classification"]:
            raise ValueError("task must be either 'detection' or 'classification'")
        if self.dataset_format not in [
            "raw",
            "yolo",
            "coco",
            "luxonis-dataset",
            "cls-single",
        ]:
            raise ValueError(
                "dataset_format must be either 'raw', 'yolo', 'coco', 'luxonis-dataset', 'cls-single'"
            )
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be either 'cuda' or 'cpu'")
        if self.prompt_generator not in ["simple", "lm", "tiny"]:
            print("Prompt generator is not valid")
            raise ValueError("prompt_generator must be either 'simple', 'lm' or 'tiny'")
        if self.synonym_generator not in ["none", "llm", "wordnet"]:
            raise ValueError(
                "synonym_generator must be either 'none', 'llm' or 'wordnet'"
            )
        if self.lm_quantization not in ["none", "4bit"]:
            raise ValueError("lm_quantization must be either 'none' or '4bit'")
        if self.image_generator not in ["sdxl", "sdxl-turbo", "sdxl-lightning"]:
            raise ValueError(
                "image_generator must be either 'sdxl', 'sdxl-turbo' or 'sdxl-lightning'"
            )
        if self.image_annotator not in ["owlv2", "clip"]:
            raise ValueError("image_annotator must be either 'owlv2' or 'clip'")
        if self.annotator_size not in ["base", "large"]:
            raise ValueError("annotator_size must be either 'base' or 'large'")
        return self
