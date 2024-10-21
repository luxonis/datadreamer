import json
import os


def save_annotations_to_json(
    image_paths,
    labels_list,
    boxes_list=None,
    masks_list=None,
    class_names=None,
    save_dir=None,
    file_name="annotations.json",
) -> None:
    """Saves annotations to a JSON file.

    Args:
        image_paths (list): List of image paths.
        labels_list (list): List of labels.
        boxes_list (list, optional): List of bounding boxes. Defaults to None.
        masks_list (list, optional): List of instance segmentation masks. Defaults to None.
        class_names (list, optional): List of class names. Defaults to None.
        save_dir (str, optional): Directory to save the JSON file. Defaults to None.
        file_name (str, optional): Name of the JSON file. Defaults to 'annotations.json'.

    No return value.
    """
    if save_dir is None:
        save_dir = os.getcwd()

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

        if masks_list is not None:
            masks = masks_list[i]
            annotations[image_name]["masks"] = masks

    annotations["class_names"] = class_names

    # Save to JSON file
    with open(os.path.join(save_dir, file_name), "w") as f:
        json.dump(annotations, f, indent=4)
