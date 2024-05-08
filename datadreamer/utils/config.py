from __future__ import annotations

from typing import Annotated, List, Literal

from luxonis_ml.utils import LuxonisConfig
from pydantic import Field


class Config(LuxonisConfig):
    # General arguments
    save_dir: str = "generated_dataset"
    class_names: List[str] = ["bear", "bicycle", "bird", "person"]
    prompts_number: int = 10
    task: Literal["detection", "classification"] = "detection"
    seed: int = 42
    device: Literal["cuda", "cpu"] = "cuda"
    annotate_only: bool = False
    dataset_format: Literal[
        "raw", "yolo", "coco", "luxonis-dataset", "cls-single"
    ] = "raw"
    split_ratios: Annotated[
        List[float], Field(default=[0.8, 0.1, 0.1], min_length=3, max_length=3)
    ] = [0.8, 0.1, 0.1]
    # Prompt generation arguments
    prompt_generator: Literal["simple", "lm", "tiny"] = "simple"
    synonym_generator: Literal["none", "llm", "wordnet"] = "none"
    num_objects_range: Annotated[
        List[int], Field(default=[1, 3], min_length=2, max_length=2)
    ] = [1, 3]
    lm_quantization: Literal["none", "4bit"] = "none"
    batch_size_prompt: int = 64
    # Image generation arguments
    image_generator: Literal["sdxl", "sdxl-turbo", "sdxl-lightning"] = "sdxl-turbo"
    prompt_prefix: str = ""
    prompt_suffix: str = ", hd, 8k, highly detailed"
    negative_prompt: str = "cartoon, blue skin, painting, scrispture, golden, illustration, worst quality, low quality, normal quality:2, unrealistic dream, low resolution, static, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, bad anatomy"
    batch_size_image: int = 1
    use_image_tester: bool = False
    image_tester_patience: int = 1
    # Annotation arguments
    image_annotator: Literal["owlv2", "clip"] = "owlv2"
    conf_threshold: float = 0.15
    annotation_iou_threshold: float = 0.2
    use_tta: bool = False
    annotator_size: Literal["base", "large"] = "base"
    batch_size_annotation: int = 1
    dataset_plugin: str = None
