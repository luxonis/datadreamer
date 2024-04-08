import json
import os

from luxonis_ml.data import LuxonisDataset
from PIL import Image


def save_annotations_to_json(
    image_paths,
    labels_list,
    boxes_list=None,
    class_names=None,
    save_dir=None,
    file_name="annotations.json",
):
    annotations = {}
    for i in range(len(image_paths)):
        # for image_path, bboxes, labels in zip(image_paths, boxes_list, labels_list):
        image_name = os.path.basename(image_paths[i])
        # image_name = os.path.basename(image_path)
        labels = labels_list[i]
        annotations[image_name] = {
            "labels": labels.tolist(),
        }
        if boxes_list is not None:
            bboxes = boxes_list[i]
            annotations[image_name]["boxes"] = bboxes.tolist()

    annotations["class_names"] = class_names

    # Save to JSON file
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(annotations, f, indent=4)

