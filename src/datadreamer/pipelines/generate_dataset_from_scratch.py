import argparse
import json
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from datadreamer.dataset_annotation import OWLv2Annotator
from datadreamer.image_generation import (
    StableDiffusionImageGenerator,
    StableDiffusionTurboImageGenerator,
)
from datadreamer.prompt_generation import (
    LMPromptGenerator,
    SimplePromptGenerator,
    SynonymGenerator,
)

prompt_generators = {"simple": SimplePromptGenerator, "lm": LMPromptGenerator}

image_generators = {
    "sdxl": StableDiffusionImageGenerator,
    "sdxl-turbo": StableDiffusionTurboImageGenerator,
}

annotators = {"owlv2": OWLv2Annotator}


def parse_args():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Generate and annotate images.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="generated_dataset",
        help="Directory to save generated images and annotations",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="detection",
        choices=["detection", "classification"],
        help="Task to generate data for",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=["bear", "bicycle", "bird", "person"],
        help="List of object names for prompt generation",
    )

    parser.add_argument(
        "--prompts_number", type=int, default=10, help="Number of prompts to generate"
    )

    parser.add_argument(
        "--num_objects_range",
        type=int,
        nargs="+",
        default=[1, 3],
        help="Range of number of objects in a prompt",
    )

    parser.add_argument(
        "--prompt_generator",
        type=str,
        default="simple",
        choices=["simple", "lm"],
        help="Prompt generator to use: simple or language model",
    )
    parser.add_argument(
        "--image_generator",
        type=str,
        default="sdxl-turbo",
        choices=["sdxl", "sdxl-turbo"],
        help="Image generator to use",
    )
    parser.add_argument(
        "--image_annotator",
        type=str,
        default="owlv2",
        choices=["owlv2"],
        help="Image annotator to use",
    )

    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for object detection",
    )

    parser.add_argument(
        "--use_tta",
        default=False,
        action="store_true",
        help="Whether to use test time augmentation for object detection",
    )

    parser.add_argument(
        "--enhance_class_names",
        default=False,
        action="store_true",
        help="Whether to enhance class names with synonyms",
    )

    parser.add_argument(
        "--use_image_tester",
        default=False,
        action="store_true",
        help="Whether to use image tester for image generation",
    )

    parser.add_argument(
        "--image_tester_patience",
        type=int,
        default=1,
        help="Patience for image tester",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for image generation"
    )

    return parser.parse_args()


def save_det_annotations_to_json(
    image_paths,
    boxes_list,
    labels_list,
    class_names,
    save_dir,
    file_name="annotations.json",
):
    annotations = {}
    for image_path, bboxes, labels in zip(image_paths, boxes_list, labels_list):
        image_name = os.path.basename(image_path)
        annotations[image_name] = {
            "boxes": bboxes.tolist(),
            "labels": labels.tolist(),
        }
    annotations["class_names"] = class_names

    # Save to JSON file
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(annotations, f, indent=4)


def save_clf_annotations_to_json(
    image_paths, labels_list, class_names, save_dir, file_name="annotations.json"
):
    annotations = {}
    for image_path, labels in zip(image_paths, labels_list):
        image_name = os.path.basename(image_path)
        annotations[image_name] = {
            "labels": labels.tolist(),
        }
    annotations["class_names"] = class_names

    # Save to JSON file
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(annotations, f, indent=4)


def main():
    args = parse_args()

    save_dir = args.save_dir

    # Directories for saving images and bboxes
    bbox_dir = os.path.join(save_dir, "bboxes_visualization")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(bbox_dir):
        os.makedirs(bbox_dir)

    # Save arguments
    with open(os.path.join(save_dir, "generation_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Prompt generation
    prompt_generator_class = prompt_generators[args.prompt_generator]
    prompt_generator = prompt_generator_class(
        class_names=args.class_names,
        prompts_number=args.prompts_number,
        num_objects_range=args.num_objects_range,
        seed=args.seed,
    )
    generated_prompts = prompt_generator.generate_prompts()
    prompt_generator.save_prompts(
        generated_prompts, os.path.join(save_dir, "prompts.json")
    )
    prompt_generator.release(empty_cuda_cache=True)

    # Synonym generation
    synonym_dict = None
    if args.enhance_class_names:
        synonym_generator = SynonymGenerator()
        synonym_dict = synonym_generator.generate_synonyms_for_list(args.class_names)
        synonym_generator.release(empty_cuda_cache=True)
        synonym_generator.save_synonyms(
            synonym_dict, os.path.join(save_dir, "synonyms.json")
        )

    # Image generation
    image_generator_class = image_generators[args.image_generator]
    image_generator = image_generator_class(
        seed=args.seed,
        use_clip_image_tester=args.use_image_tester,
        image_tester_patience=args.image_tester_patience,
    )

    prompts = [p[1] for p in generated_prompts]
    prompt_objects = [p[0] for p in generated_prompts]

    image_paths = []
    for i, generated_image in enumerate(
        image_generator.generate_images(prompts, prompt_objects)
    ):
        image_path = os.path.join(save_dir, f"image_{i}.jpg")
        generated_image.save(image_path)
        image_paths.append(image_path)

    image_generator.release(empty_cuda_cache=True)

    if args.task == "classification":
        # Classification annotation
        labels_list = []
        for prompt_objs in prompt_objects:
            labels = []
            for obj in prompt_objs:
                labels.append(args.class_names.index(obj))
            labels_list.append(np.unique(labels))

        save_clf_annotations_to_json(
            image_paths, labels_list, args.class_names, save_dir
        )
    else:
        # Annotation
        annotator_class = annotators[args.image_annotator]
        annotator = annotator_class(device=args.device)

        boxes_list = []
        scores_list = []
        labels_list = []

        for i, image_path in tqdm(
            enumerate(image_paths),
            desc="Annotating images",
            total=len(image_paths),
        ):
            image = Image.open(image_path)
            boxes, scores, local_labels = annotator.annotate(
                image,
                args.class_names,
                conf_threshold=args.conf_threshold,
                use_tta=args.use_tta,
                synonym_dict=synonym_dict,
            )
            # Convert to numpy arrays
            boxes = (
                boxes.detach().cpu().numpy()
                if not isinstance(boxes, np.ndarray)
                else boxes
            )
            scores = (
                scores.detach().cpu().numpy()
                if not isinstance(scores, np.ndarray)
                else scores
            )
            local_labels = (
                local_labels
                if isinstance(local_labels, np.ndarray)
                else local_labels.detach().cpu().numpy()
            )

            boxes_list.append(boxes)
            scores_list.append(scores)

            labels = []
            # Save bbox visualizations
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            for box, score, label in zip(boxes, scores, local_labels):
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
                plt.title(generated_prompts[i][1])

            labels_list.append(np.array(labels))

            plt.savefig(os.path.join(bbox_dir, f"bbox_{i}.jpg"))
            plt.close()

        # Save annotations as JSON files
        save_det_annotations_to_json(
            image_paths, boxes_list, labels_list, args.class_names, save_dir
        )


if __name__ == "__main__":
    main()
