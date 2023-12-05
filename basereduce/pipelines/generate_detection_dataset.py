import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image
import numpy as np
import os
import json
import argparse

from basereduce.prompt_generation import SimplePromptGenerator
from basereduce.image_generation import StableDiffusionTurboImageGenerator
from basereduce.dataset_annotation import OWLv2Annotator

# Argument parsing
parser = argparse.ArgumentParser(description="Generate and annotate images.")
parser.add_argument(
    "--save_dir",
    type=str,
    default="generated_dataset",
    help="Directory to save generated images and annotations",
)
parser.add_argument(
    "--object_names",
    type=str,
    nargs="+",
    default=["aeroplane", "bicycle", "bird", "boat", "person"],
    help="List of object names for prompt generation",
)
parser.add_argument(
    "--prompts_number", type=int, default=10, help="Number of prompts to generate"
)

args = parser.parse_args()

save_dir = args.save_dir
object_names = args.object_names
prompts_number = args.prompts_number

# Directories for saving images and bboxes
bbox_dir = os.path.join(save_dir, "bboxes")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(bbox_dir):
    os.makedirs(bbox_dir)

# Prompt generator
prompt_generator = SimplePromptGenerator(
    class_names=object_names, prompts_number=prompts_number
)
generated_prompts = prompt_generator.generate_prompts()
print(generated_prompts)

# Image generation
image_generator = StableDiffusionTurboImageGenerator(seed=42.0)
prompts = [p[1] for p in generated_prompts]
prompt_objects = [p[0] for p in generated_prompts]
generated_images = list(image_generator.generate_images(prompts))
image_generator.release(empty_cuda_cache=True)

# Annotation
annotator = OWLv2Annotator(seed=42, device="cuda")
boxes_list = []
scores_list = []
labels_list = []

for i, (image, prompt_objs) in enumerate(zip(generated_images, prompt_objects)):
    boxes, scores, labels = annotator.annotate(
        image, prompt_objs, conf_threshold=0.2, use_tta=True
    )
    # Convert to numpy arrays
    boxes = boxes.detach().cpu().numpy() if not isinstance(boxes, np.ndarray) else boxes
    scores = (
        scores.detach().cpu().numpy() if not isinstance(scores, np.ndarray) else scores
    )
    labels = labels if isinstance(labels, np.ndarray) else labels.detach().cpu().numpy()

    boxes_list.append(boxes)
    scores_list.append(scores)
    labels_list.append(labels)

    # Save bbox visualizations
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        label_text = prompt_objs[label] if isinstance(label, np.int64) else label
        plt.text(
            x1,
            y1,
            f"{label_text} {score:.2f}",
            bbox=dict(facecolor="yellow", alpha=0.5),
        )

    plt.savefig(os.path.join(bbox_dir, f"bbox_{i}.jpg"))
    plt.close()

# Save images
for i, img in enumerate(generated_images):
    img.save(os.path.join(save_dir, f"image_{i}.jpg"))


# Function to save data to JSON
def save_to_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)


# Save annotations as JSON files
save_to_json(
    [sub.tolist() for sub in boxes_list], os.path.join(save_dir, "boxes_list.json")
)
save_to_json(
    [sub.tolist() for sub in labels_list], os.path.join(save_dir, "labels_list.json")
)
save_to_json(object_names, os.path.join(save_dir, "object_names.json"))
