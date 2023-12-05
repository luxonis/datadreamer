from PIL import Image
from basereduce.dataset_annotation import OWLv2Annotator, TaskList, Kosmos2Annotator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Initialize the OWLv2Annotator
annotator = OWLv2Annotator(
    seed=42,
    device="cuda",  # Use "cuda" for GPU or "cpu" for CPU
)

# Load your image
image = Image.open("generated_images/Figure_1.png").convert("RGB")



# Define prompts to guide the detection
#prompts = ["Eye", "Text"]

class_map = {
    "Austronaut": "astronaut",
    "Spaceman": "astronaut",
    "Cosmonaut": "astronaut",
    "Moonwalker": "astronaut",
    "Horse": "horse",
    "Stablemate": "horse",
    "Steed": "horse",
    "Moon": "moon",
}

# class_map = {
#     "Austronaut": "astronaut",
#     "Horse": "horse",
#     "Moon": "moon",
# }

prompts = list(class_map.keys())

# Perform object detection
boxes, scores, labels = annotator.annotate(image, prompts, conf_threshold=0.2, use_tta=True)

# Convert to numpy arrays
if not isinstance(boxes, np.ndarray):
    boxes = boxes.detach().cpu().numpy()
if not isinstance(scores, np.ndarray):
    scores = scores.detach().cpu().numpy()
if not isinstance(labels, np.ndarray):
    labels = labels.detach().cpu().numpy()

# Process the results
for box, score, label in zip(boxes, scores, labels):
    if isinstance(label, np.int64):
        print(f"Box: {box}, Score: {score}, Label: {class_map[prompts[label]]}")
    else:
        print(f"Box: {box}, Score: {score}, Label: {label}")


fig, ax = plt.subplots(1)
ax.imshow(image)

# Iterate over each detection
for box, score, label in zip(boxes, scores, labels):
    # Each box is (x1, y1, x2, y2)
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    rect = patches.Rectangle(
        (x1, y1), width, height, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)

    if isinstance(label, np.int64):
        label = class_map[prompts[label]]

    plt.text(
        x1,
        y1,
        f"{label} {score:.2f}",
        bbox=dict(facecolor="yellow", alpha=0.5),
    )

plt.show()
