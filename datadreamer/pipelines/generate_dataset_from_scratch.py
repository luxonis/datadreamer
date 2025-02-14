from __future__ import annotations

import argparse
import os
import shutil
import textwrap
import uuid

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import torch
from box import Box
from luxonis_ml.data import DATASETS_REGISTRY, LOADERS_REGISTRY
from luxonis_ml.utils import setup_logging
from PIL import Image
from tqdm import tqdm

from datadreamer.dataset_annotation import (
    AIMv2Annotator,
    CLIPAnnotator,
    OWLv2Annotator,
    SAM2Annotator,
    SlimSAMAnnotator,
)
from datadreamer.image_generation import (
    Shuttle3DiffusionImageGenerator,
    StableDiffusionImageGenerator,
    StableDiffusionLightningImageGenerator,
    StableDiffusionTurboImageGenerator,
)
from datadreamer.prompt_generation import (
    LMPromptGenerator,
    LMSynonymGenerator,
    ProfanityFilter,
    Qwen2LMPromptGenerator,
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
    "qwen2": Qwen2LMPromptGenerator,
}

synonym_generators = {
    "llm": LMSynonymGenerator,
    "wordnet": WordNetSynonymGenerator,
}

image_generators = {
    "sdxl": StableDiffusionImageGenerator,
    "sdxl-turbo": StableDiffusionTurboImageGenerator,
    "sdxl-lightning": StableDiffusionLightningImageGenerator,
    "shuttle-3": Shuttle3DiffusionImageGenerator,
}

det_annotators = {"owlv2": OWLv2Annotator}
clf_annotators = {"clip": CLIPAnnotator, "aimv2": AIMv2Annotator}
inst_seg_annotators = {"owlv2-slimsam": SlimSAMAnnotator, "owlv2-sam2": SAM2Annotator}
inst_seg_detectors = {"owlv2-slimsam": OWLv2Annotator, "owlv2-sam2": OWLv2Annotator}

setup_logging()


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
        choices=["detection", "classification", "instance-segmentation"],
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
        choices=["simple", "lm", "tiny", "qwen2"],
        help="Prompt generator to use: simple, lm, tiny, or qwen2 (default).",
    )
    parser.add_argument(
        "--image_generator",
        type=str,
        choices=["sdxl", "sdxl-turbo", "sdxl-lightning", "shuttle-3"],
        help="Image generator to use",
    )
    parser.add_argument(
        "--image_annotator",
        type=str,
        choices=["owlv2", "clip", "owlv2-slimsam", "aimv2", "owlv2-sam2"],
        help="Image annotator to use",
    )

    parser.add_argument(
        "--dataset_format",
        type=str,
        choices=["raw", "yolo", "coco", "voc", "luxonis-dataset", "cls-single"],
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
        "--disable_lm_filter",
        default=None,
        action="store_true",
        help="Whether to use only bad words in profanity filter",
    )

    parser.add_argument(
        "--keep_unlabeled_images",
        default=None,
        action="store_true",
        help="Whether to keep images without any annotations",
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
        "--raw_mask_format",
        type=str,
        choices=["polyline", "rle"],
        help="Format of segmentations masks when saved in raw dataset format",
    )

    parser.add_argument(
        "--vis_anns",
        default=None,
        action="store_true",
        help="Whether to save visualizations of annotations",
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
        "--dataset_plugin",
        type=str,
        help="LuxonisDataset plugin for the luxonis-dataset format",
    )

    parser.add_argument(
        "--loader_plugin",
        type=str,
        help="Loader plugin for the LuxonisLoader",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to create if dataset_plugin or loader_plugin is used",
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
    if not args.annotate_only and args.num_objects_range[1] > len(args.class_names):
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
        or args.prompt_generator not in ["lm", "qwen2"]
    ):
        raise ValueError(
            "LM Quantization is only available for CUDA devices and Mistral/Qwen2.5 prompt generators"
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

    if (
        args.task == "instance-segmentation"
        and args.image_annotator not in inst_seg_annotators
    ):
        raise ValueError(
            "--image_annotator must be one of the available annotators for instance segmentation task"
        )

    # Check coorect task and dataset_format
    if args.task == "classification" and args.dataset_format in ["coco", "yolo", "voc"]:
        raise ValueError(
            "--dataset_format must be one of the available dataset formats for classification task: raw, cls-single, luxonis-dataset"
        )

    if args.task == "detection" and args.dataset_format in ["cls-single"]:
        raise ValueError(
            "--dataset_format must be one of the available dataset formats for detection task: raw, coco, yolo, luxonis-dataset"
        )

    if args.task == "instance-segmentation" and args.dataset_format in ["cls-single"]:
        raise ValueError(
            "--dataset_format must be one of the available dataset formats for instance segmentation task: raw, coco, yolo, luxonis-dataset"
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

    # Check if dataset_plugin is valid
    if args.dataset_plugin:
        if args.dataset_format != "luxonis-dataset":
            raise ValueError(
                "--dataset_format must be 'luxonis-dataset' if --dataset_plugin is specified"
            )
        try:
            DATASETS_REGISTRY.get(args.dataset_plugin)
        except KeyError:
            raise KeyError(
                f"Dataset plugin '{args.dataset_plugin}' is not registered in DATASETS_REGISTRY"
            ) from None


def main():
    args = parse_args()
    # Get the None args without the config
    args_dict = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    config = Config.get_config(args.config, args_dict)
    args = Box(config.model_dump(exclude_none=True, by_alias=True))
    # Check arguments
    check_args(args)

    profanity_filter = ProfanityFilter(
        seed=args.seed, device=args.device, use_lm=not args.disable_lm_filter
    )
    # Check class names for bad words
    if not profanity_filter.is_safe(args.class_names):
        raise ValueError(f"Class names: '{args.class_names}' contain bad words!")
    profanity_filter.release(empty_cuda_cache=True)

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

    def split_image_paths(image_paths, batch_size):
        return [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

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

        # Split image_paths into batches
        image_batches = split_image_paths(image_paths, args.batch_size_annotation)

    else:
        if args.loader_plugin:
            if "DATASET_ID" in os.environ:
                image_batches = LOADERS_REGISTRY.get(args.loader_plugin)(
                    view="all",
                    dataset_id=os.getenv("DATASET_ID"),
                    sync_target_directory=save_dir,
                    load_image_paths=True,
                )
            else:
                raise ValueError(
                    "DATASET_ID environment variable is not set for using the loader plugin"
                )

        else:
            # Load image paths for annotation
            for image_path in os.listdir(save_dir):
                # Check file extension: jpg, png, jpeg
                if image_path.lower().endswith(
                    (".jpg", ".png", ".jpeg", ".bmp", "webp")
                ):
                    image_paths.append(os.path.join(save_dir, image_path))
            # Split image_paths into batches
            image_batches = split_image_paths(image_paths, args.batch_size_annotation)

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

    def read_image_batch(image_batch, batch_num, batch_size):
        if type(image_batch[0]) == np.ndarray:
            images = []
            batch_image_paths = []
            for i, image in enumerate(image_batch[:-1]):
                image = Image.fromarray(image)
                unique_id = uuid.uuid4().hex
                image_path = os.path.join(
                    save_dir, f"image_{batch_num * batch_size + i}_{unique_id}.jpg"
                )
                image.save(image_path)
                images.append(image)
                batch_image_paths.append(image_path)

        else:
            images = [
                Image.open(image_path).convert("RGB") for image_path in image_batch
            ]
            batch_image_paths = image_batch
        return images, batch_image_paths

    boxes_list = []
    scores_list = []
    labels_list = []
    segment_list = []
    image_paths = []

    if args.task == "classification":
        # Classification annotation
        annotator_class = clf_annotators[args.image_annotator]
        annotator = annotator_class(device=args.device, size=args.annotator_size)

        for i, image_batch in tqdm(
            enumerate(image_batches),
            desc="Annotating images",
            total=len(image_batches),
        ):
            images, batch_image_paths = read_image_batch(
                image_batch, i, args.batch_size_annotation
            )
            image_paths.extend(batch_image_paths)

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
        # Detection annotation
        if args.task == "detection":
            annotator_class = det_annotators[args.image_annotator]
        else:
            annotator_class = inst_seg_detectors[args.image_annotator]
            inst_seg_annotator_class = inst_seg_annotators[args.image_annotator]
            inst_seg_annotator = inst_seg_annotator_class(
                device=args.device,
                size=args.annotator_size,
                mask_format=args.raw_mask_format,
            )
        annotator = annotator_class(device=args.device, size=args.annotator_size)

        for i, image_batch in tqdm(
            enumerate(image_batches),
            desc="Annotating images",
            total=len(image_batches),
        ):
            images, batch_image_paths = read_image_batch(
                image_batch, i, args.batch_size_annotation
            )
            image_paths.extend(batch_image_paths)

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
            labels_list.extend(local_labels_batch)

            if args.task == "instance-segmentation":
                masks_batch = inst_seg_annotator.annotate_batch(
                    images=images,
                    boxes_batch=boxes_batch,
                    conf_threshold=args.conf_threshold,
                )
                segment_list.extend(masks_batch)

            if args.vis_anns:
                for j, image in enumerate(images):
                    # Save bbox visualizations
                    fig, ax = plt.subplots(1)
                    ax.imshow(image)
                    for k in range(len(boxes_batch[j])):
                        box = boxes_batch[j][k]
                        score = scores_batch[j][k]
                        label = local_labels_batch[j][k]

                        if args.task == "instance-segmentation":
                            if k < len(masks_batch[j]):
                                mask = masks_batch[j][k]
                                if len(mask) > 0:  # Ensure mask is valid
                                    if isinstance(mask, dict) and "counts" in mask:
                                        binary_mask = mask_utils.decode(mask)
                                        if len(binary_mask.shape) == 3:
                                            binary_mask = binary_mask.squeeze(0)
                                        rgba_mask = np.zeros(
                                            (
                                                binary_mask.shape[0],
                                                binary_mask.shape[1],
                                                4,
                                            ),
                                            dtype=np.uint8,
                                        )
                                        rgba_mask[..., :3] = np.random.randint(
                                            0, 256, size=3
                                        )
                                        rgba_mask[..., 3] = np.where(
                                            binary_mask == 1, 128, 0
                                        )

                                        ax.imshow(rgba_mask)
                                    else:
                                        x_points, y_points = zip(*mask)
                                        ax.fill(x_points, y_points, label, alpha=0.5)

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
                        title = generated_prompts[i * args.batch_size_annotation + j][1]
                        wrapped_title = "\n".join(textwrap.wrap(title, width=50))
                        plt.title(wrapped_title)
                    else:
                        plt.title("Annotated image")

                    plt.axis("off")
                    plt.savefig(
                        os.path.join(
                            bbox_dir,
                            f"bbox_{(i * args.batch_size_annotation + j):07d}.jpg",
                        )
                    )

                    plt.close()

        # Save annotations as JSON files
        save_annotations_to_json(
            image_paths=image_paths,
            labels_list=labels_list,
            boxes_list=boxes_list,
            masks_list=segment_list if len(segment_list) > 0 else None,
            class_names=args.class_names,
            save_dir=save_dir,
        )

        if args.dataset_format in ["yolo", "coco", "voc"]:
            # Convert annotations to YOLO format
            convert_dataset.convert_dataset(
                args.save_dir,
                args.save_dir,
                "yolo",
                args.split_ratios,
                copy_files=False,
                is_instance_segmentation=args.task == "instance-segmentation",
                keep_unlabeled_images=args.keep_unlabeled_images,
                seed=args.seed,
            )

    # Convert annotations to LuxonisDataset format
    if args.dataset_format == "luxonis-dataset":
        convert_dataset.convert_dataset(
            args.save_dir,
            args.save_dir,
            "luxonis-dataset",
            args.split_ratios,
            dataset_plugin=args.dataset_plugin,
            dataset_name=args.dataset_name,
            is_instance_segmentation=args.task == "instance-segmentation",
            keep_unlabeled_images=args.keep_unlabeled_images,
            copy_files=False,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
