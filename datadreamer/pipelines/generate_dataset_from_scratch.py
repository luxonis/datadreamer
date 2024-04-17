from __future__ import annotations

import argparse
import os
import shutil
import uuid

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from box import Box
from PIL import Image
from tqdm import tqdm

from datadreamer.dataset_annotation import CLIPAnnotator, OWLv2Annotator
from datadreamer.image_generation import (
    StableDiffusionImageGenerator,
    StableDiffusionLightningImageGenerator,
    StableDiffusionTurboImageGenerator,
)
from datadreamer.prompt_generation import (
    LMPromptGenerator,
    LMSynonymGenerator,
    SimplePromptGenerator,
    TinyLlamaLMPromptGenerator,
    WordNetSynonymGenerator,
)
from datadreamer.utils import Config, convert_dataset
from datadreamer.utils.dataset_utils import save_annotations_to_json

prompt_generators = {
    "simple": SimplePromptGenerator,
    "lm": LMPromptGenerator,
    "tiny": TinyLlamaLMPromptGenerator,
}

synonym_generators = {
    "llm": LMSynonymGenerator,
    "wordnet": WordNetSynonymGenerator,
}

image_generators = {
    "sdxl": StableDiffusionImageGenerator,
    "sdxl-turbo": StableDiffusionTurboImageGenerator,
    "sdxl-lightning": StableDiffusionLightningImageGenerator,
}

det_annotators = {"owlv2": OWLv2Annotator}
clf_annotators = {"clip": CLIPAnnotator}


def parse_args():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate and annotate images.")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save generated images and annotations",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["detection", "classification"],
        help="Task to generate data for",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        help="List of object names for prompt generation",
    )

    parser.add_argument(
        "--annotate_only",
        action="store_true",
        default=None,
        help="Only annotate the images without generating new ones, prompt and image generator will be skipped.",
    )

    parser.add_argument(
        "--prompts_number",
        type=int,
        help="Number of prompts to generate",
    )

    parser.add_argument(
        "--num_objects_range",
        type=int,
        nargs="+",
        help="Range of number of objects in a prompt",
    )

    parser.add_argument(
        "--prompt_generator",
        type=str,
        choices=["simple", "lm", "tiny"],
        help="Prompt generator to use: simple or language model",
    )
    parser.add_argument(
        "--image_generator",
        type=str,
        choices=["sdxl", "sdxl-turbo", "sdxl-lightning"],
        help="Image generator to use",
    )
    parser.add_argument(
        "--image_annotator",
        type=str,
        choices=["owlv2", "clip"],
        help="Image annotator to use",
    )

    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["raw", "yolo", "coco", "luxonis-dataset", "cls-single"],
        help="Dataset format to use",
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs="+",
        help="Train-validation-test split ratios (default: 0.8, 0.1, 0.1).",
    )

    parser.add_argument(
        "--synonym_generator",
        type=str,
        choices=["none", "llm", "wordnet"],
        help="Image annotator to use",
    )

    parser.add_argument(
        "--negative_prompt",
        type=str,
        help="Negative prompt to guide the generation away from certain features",
    )

    parser.add_argument(
        "--prompt_suffix",
        type=str,
        help="Suffix to add to every image generation prompt, e.g., for adding details like resolution",
    )

    parser.add_argument(
        "--prompt_prefix",
        type=str,
        help="Prefix to add to every image generation prompt",
    )

    parser.add_argument(
        "--conf_threshold",
        type=float,
        help="Confidence threshold for annotation",
    )

    parser.add_argument(
        "--annotation_iou_threshold",
        type=float,
        help="Intersection over Union (IoU) threshold for annotation",
    )

    parser.add_argument(
        "--use_tta",
        default=None,
        action="store_true",
        help="Whether to use test time augmentation for object detection",
    )

    parser.add_argument(
        "--use_image_tester",
        default=None,
        action="store_true",
        help="Whether to use image tester for image generation",
    )

    parser.add_argument(
        "--image_tester_patience",
        type=int,
        help="Patience for image tester",
    )

    parser.add_argument(
        "--lm_quantization",
        type=str,
        choices=["none", "4bit"],
        help="Quantization to use for Mistral language model",
    )

    parser.add_argument(
        "--annotator_size",
        type=str,
        choices=["base", "large"],
        help="Size of the annotator model to use",
    )

    parser.add_argument(
        "--batch_size_prompt",
        type=int,
        help="Batch size for prompt generation",
    )

    parser.add_argument(
        "--batch_size_annotation",
        type=int,
        help="Batch size for annotation",
    )

    parser.add_argument(
        "--batch_size_image",
        type=int,
        help="Batch size for image generation",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for image generation",
    )

    return parser.parse_args()


def check_args(args):
    # Check save_dir
    if not os.path.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir)
        except OSError as e:
            raise ValueError(f"Cannot create directory {args.save_dir}: {e}") from e

    # Check class_names
    if not args.class_names or any(
        not isinstance(name, str) for name in args.class_names
    ):
        raise ValueError("--class_names must be a non-empty list of strings")

    # Check prompts_number
    if args.prompts_number <= 0:
        raise ValueError("--prompts_number must be a positive integer")

    # Check num_objects_range
    if (
        len(args.num_objects_range) != 2
        or not all(isinstance(n, int) for n in args.num_objects_range)
        or args.num_objects_range[0] > args.num_objects_range[1]
    ):
        raise ValueError(
            "--num_objects_range must be two integers where the first is less than or equal to the second"
        )

    # Check num_objects_range[1]
    if args.num_objects_range[1] > len(args.class_names):
        raise ValueError(
            "--num_objects_range[1] must be less than or equal to the number of class names"
        )

    # Check conf_threshold
    if not 0 <= args.conf_threshold <= 1:
        raise ValueError("--conf_threshold must be between 0 and 1")

    # Check annotation_iou_threshold
    if not 0 <= args.annotation_iou_threshold <= 1:
        raise ValueError("--annotation_iou_threshold must be between 0 and 1")

    # Check image_tester_patience
    if args.image_tester_patience < 0:
        raise ValueError("--image_tester_patience must be a non-negative integer")

    # Check device availability (for 'cuda')
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please use --device cpu")

    # Check for LM quantization availability
    if args.lm_quantization != "none" and (
        args.device == "cpu"
        or not torch.cuda.is_available()
        or args.prompt_generator != "lm"
    ):
        raise ValueError(
            "LM Quantization is only available for CUDA devices and Mistral LM"
        )

    # Check batch_size_prompt
    if args.batch_size_prompt < 1:
        raise ValueError("--batch_size_prompt must be a positive integer")

    # Check batch_size_prompt
    if args.batch_size_annotation < 1:
        raise ValueError("--batch_size_annotation must be a positive integer")

    # Check batch_size_image
    if args.batch_size_image < 1:
        raise ValueError("--batch_size_image must be a positive integer")

    # Check seed
    if args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")

    # Check correct annotation and task
    if args.task == "detection" and args.image_annotator not in det_annotators:
        raise ValueError(
            "--image_annotator must be one of the available annotators for detection task"
        )

    if args.task == "classification" and args.image_annotator not in clf_annotators:
        raise ValueError(
            "--image_annotator must be one of the available annotators for classification task"
        )

    # Check coorect task and dataset_format
    if args.task == "classification" and args.dataset_format in ["coco", "yolo"]:
        raise ValueError(
            "--dataset_format must be one of the available dataset formats for classification task: raw, cls-single, luxonis-dataset"
        )

    if args.task == "detection" and args.dataset_format in ["cls-single"]:
        raise ValueError(
            "--dataset_format must be one of the available dataset formats for detection task: raw, coco, yolo, luxonis-dataset"
        )

    # Check split_ratios
    if (
        len(args.split_ratios) != 3
        or not all(0 <= ratio <= 1 for ratio in args.split_ratios)
        or sum(args.split_ratios) != 1
    ):
        raise ValueError(
            "--split_ratios must be a list of three floats that sum up to 1"
        )


def main():
    args = parse_args()
    # Get the None args without the config
    args_dict = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    config = Config.get_config(args.config, args_dict)
    args = Box(config.model_dump(exclude_none=True, by_alias=True))
    # Check arguments
    check_args(args)

    # Directories for saving images and bboxes
    save_dir = args.save_dir
    if not args.annotate_only:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    bbox_dir = os.path.join(save_dir, "bboxes_visualization")
    if os.path.exists(bbox_dir):
        shutil.rmtree(bbox_dir)
    os.makedirs(bbox_dir)

    # Save arguments
    config.save_data(os.path.join(save_dir, "generation_args.yaml"))

    generated_prompts = None
    image_paths = []

    if not args.annotate_only:
        # Prompt generation
        prompt_generator_class = prompt_generators[args.prompt_generator]
        prompt_generator = prompt_generator_class(
            class_names=args.class_names,
            prompts_number=args.prompts_number,
            num_objects_range=args.num_objects_range,
            seed=args.seed,
            device=args.device,
            quantization=args.lm_quantization,
            batch_size=args.batch_size_prompt,
        )
        generated_prompts = prompt_generator.generate_prompts()
        prompt_generator.save_prompts(
            generated_prompts, os.path.join(save_dir, "prompts.json")
        )
        prompt_generator.release(empty_cuda_cache=True)

        # Image generation
        image_generator_class = image_generators[args.image_generator]
        image_generator = image_generator_class(
            prompt_prefix=args.prompt_prefix,
            prompt_suffix=args.prompt_suffix,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            use_clip_image_tester=args.use_image_tester,
            image_tester_patience=args.image_tester_patience,
            batch_size=args.batch_size_image,
            device=args.device,
        )

        prompts = [p[1] for p in generated_prompts]
        prompt_objects = [p[0] for p in generated_prompts]

        num_generated_images = 0
        for generated_images_batch in image_generator.generate_images(
            prompts, prompt_objects
        ):
            for generated_image in generated_images_batch:
                unique_id = uuid.uuid4().hex
                unique_filename = f"image_{num_generated_images}_{unique_id}.jpg"
                image_path = os.path.join(save_dir, unique_filename)
                generated_image.save(image_path)
                image_paths.append(image_path)
                num_generated_images += 1

        image_generator.release(empty_cuda_cache=True)

    else:
        # Load image paths for annotation
        for image_path in os.listdir(save_dir):
            # Check file extension: jpg, png, jpeg
            if image_path.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", "webp")):
                image_paths.append(os.path.join(save_dir, image_path))

    # Synonym generation
    synonym_dict = None
    if args.synonym_generator != "none":
        synonym_generator_class = synonym_generators[args.synonym_generator]
        synonym_generator = synonym_generator_class(device=args.device)
        synonym_dict = synonym_generator.generate_synonyms_for_list(args.class_names)
        synonym_generator.release(empty_cuda_cache=True)
        synonym_generator.save_synonyms(
            synonym_dict, os.path.join(save_dir, "synonyms.json")
        )

    boxes_list = []
    scores_list = []
    labels_list = []

    if args.task == "classification":
        # Classification annotation
        annotator_class = clf_annotators[args.image_annotator]
        annotator = annotator_class(device=args.device, size=args.annotator_size)

        # Split image_paths into batches
        image_batches = [
            image_paths[i : i + args.batch_size_annotation]
            for i in range(0, len(image_paths), args.batch_size_annotation)
        ]

        for image_batch in tqdm(
            image_batches,
            desc="Annotating images",
            total=len(image_batches),
        ):
            images = [Image.open(image_path) for image_path in image_batch]
            batch_labels = annotator.annotate_batch(
                images,
                args.class_names,
                conf_threshold=args.conf_threshold,
                synonym_dict=synonym_dict,
            )
            labels_list.extend(batch_labels)

        save_annotations_to_json(
            image_paths=image_paths,
            labels_list=labels_list,
            class_names=args.class_names,
            save_dir=save_dir,
        )

        if args.dataset_format == "cls-single":
            convert_dataset.convert_dataset(
                args.save_dir,
                args.save_dir,
                "cls-single",
                args.split_ratios,
                copy_files=False,
                seed=args.seed,
            )
    else:
        # Annotation
        annotator_class = det_annotators[args.image_annotator]
        annotator = annotator_class(device=args.device, size=args.annotator_size)

        # Split image_paths into batches
        image_batches = [
            image_paths[i : i + args.batch_size_annotation]
            for i in range(0, len(image_paths), args.batch_size_annotation)
        ]
        for i, image_batch in tqdm(
            enumerate(image_batches),
            desc="Annotating images",
            total=len(image_batches),
        ):
            images = [Image.open(image_path) for image_path in image_batch]
            boxes_batch, scores_batch, local_labels_batch = annotator.annotate_batch(
                images,
                args.class_names,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.annotation_iou_threshold,
                use_tta=args.use_tta,
                synonym_dict=synonym_dict,
            )

            boxes_list.extend(boxes_batch)
            scores_list.extend(scores_batch)

            for j, image in enumerate(images):
                labels = []
                # Save bbox visualizations
                fig, ax = plt.subplots(1)
                ax.imshow(image)
                for box, score, label in zip(
                    boxes_batch[j], scores_batch[j], local_labels_batch[j]
                ):
                    labels.append(label)
                    x1, y1, x2, y2 = box
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    label_text = args.class_names[label]
                    plt.text(
                        x1,
                        y1,
                        f"{label_text} {score:.2f}",
                        bbox=dict(facecolor="yellow", alpha=0.5),
                    )
                    # Add prompt text as title
                if generated_prompts:
                    plt.title(generated_prompts[i * args.batch_size_annotation + j][1])
                else:
                    plt.title("Annotated image")

                labels_list.append(np.array(labels))

                plt.savefig(
                    os.path.join(
                        bbox_dir, f"bbox_{i * args.batch_size_annotation + j}.jpg"
                    )
                )
                plt.close()

        # Save annotations as JSON files
        save_annotations_to_json(
            image_paths=image_paths,
            labels_list=labels_list,
            boxes_list=boxes_list,
            class_names=args.class_names,
            save_dir=save_dir,
        )

        if args.dataset_format == "yolo":
            # Convert annotations to YOLO format
            convert_dataset.convert_dataset(
                args.save_dir,
                args.save_dir,
                "yolo",
                args.split_ratios,
                copy_files=False,
                seed=args.seed,
            )
        # Convert annotations to COCO format
        elif args.dataset_format == "coco":
            convert_dataset.convert_dataset(
                args.save_dir,
                args.save_dir,
                "coco",
                args.split_ratios,
                copy_files=False,
                seed=args.seed,
            )

    # Convert annotations to LuxonisDataset format
    if args.dataset_format == "luxonis-dataset":
        convert_dataset.convert_dataset(
            args.save_dir,
            args.save_dir,
            "luxonis-dataset",
            args.split_ratios,
            copy_files=False,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
