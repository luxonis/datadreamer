from PIL import Image
from basereduce.dataset_annotation import OWLv2Annotator, ModelName, TaskList
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize the OWLv2Annotator
annotator = OWLv2Annotator(
    seed=42,
    model_name=ModelName.OWL_V2,
    task_definition=TaskList.OBJECT_DETECTION,
    device="cuda",  # Use "cuda" for GPU or "cpu" for CPU
)

# Load your image
image = Image.open("../images/dalle-logo.png").convert("RGB")


# Define prompts to guide the detection
prompts = ["Eye", "Text"]

# Perform object detection
boxes, scores, labels = annotator.annotate(image, prompts, conf_threshold=0.1)

# Convert to numpy arrays
boxes = boxes.detach().cpu().numpy()
scores = scores.detach().cpu().numpy()
labels = labels.detach().cpu().numpy()

# Process the results
for box, score, label in zip(boxes, scores, labels):
    print(f"Box: {box}, Score: {score}, Label: {prompts[label]}")


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

    plt.text(
        x1,
        y1,
        f"{prompts[label]} {score:.2f}",
        bbox=dict(facecolor="yellow", alpha=0.5),
    )

plt.show()
