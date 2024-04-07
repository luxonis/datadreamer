import json
import os
from PIL import Image


from luxonis_ml.data import LuxonisDataset

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
    #for image_path, bboxes, labels in zip(image_paths, boxes_list, labels_list):
        image_name = os.path.basename(image_paths[i])
        #image_name = os.path.basename(image_path)
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

def convert_to_ldf(image_paths, labels_list, boxes_list, save_dir, class_names, split_ratios):
    width, height = Image.open(image_paths[0]).size
    def dataset_generator():
        # find image paths and load COCO annotations
        
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            labels = labels_list[i]
            for label in labels:
                yield {
                    "file": image_path,
                    "class": class_names[label],
                    "type": "classification",
                    "value": True,
                }

            if boxes_list:
                boxes = boxes_list[i]
                for box in boxes:
                    x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                    yield {
                        "file": image_path,
                        "class": class_names[label],
                        "type": "box",
                        "value": (x / width, y / height, w / width, h / height),
                    }
    
    dataset_name = os.path.basename(save_dir)
    if LuxonisDataset.exists(dataset_name):
        dataset = LuxonisDataset(dataset_name)
        dataset.delete_dataset()

    dataset = LuxonisDataset(dataset_name)
    dataset.set_classes(class_names)

    dataset.add(dataset_generator)

    dataset.make_splits(split_ratios)


