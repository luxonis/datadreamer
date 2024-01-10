import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


def read_annotations(annotation_path):
    """Reads annotations from a JSON file located at the specified path.

    Args:
    - annotation_path (str): The path to the JSON file containing annotations.

    Returns:
    - dict: A dictionary containing the data loaded from the JSON file.
    """
    with open(annotation_path) as f:
        data = json.load(f)
    return data


def convert_to_yolo_format(box, image_width, image_height):
    """Converts bounding box coordinates to YOLO format.

    Args:
    - box (list of float): A list containing the bounding box coordinates [x_min, y_min, x_max, y_max].
    - image_width (int): The width of the image.
    - image_height (int): The height of the image.

    Returns:
    - list of float: A list containing the bounding box in YOLO format [x_center, y_center, width, height].
    """
    x_center = (box[0] + box[2]) / 2 / image_width
    y_center = (box[1] + box[3]) / 2 / image_height
    width = (box[2] - box[0]) / image_width
    height = (box[3] - box[1]) / image_height
    return [x_center, y_center, width, height]


def process_data(data, image_dir, output_dir, split_ratio):
    """Processes the data by dividing it into training and validation sets, and saves
    the images and labels in YOLO format.

    Args:
    - data (dict): The dictionary containing image annotations.
    - image_dir (str): The directory where the source images are located.
    - output_dir (str): The base directory where the processed data will be saved.
    - split_ratio (float): The ratio to split the data into training and validation sets.

    No return value.
    """
    images = list(data.keys())
    np.random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    for dataset_type, image_set in [("train", train_images), ("val", val_images)]:
        image_output_dir = os.path.join(output_dir, dataset_type, "images")
        label_output_dir = os.path.join(output_dir, dataset_type, "labels")

        # If the output directories already exist, replace them
        if os.path.exists(image_output_dir):
            shutil.rmtree(image_output_dir)
        if os.path.exists(label_output_dir):
            shutil.rmtree(label_output_dir)

        os.makedirs(image_output_dir)
        os.makedirs(label_output_dir)

        for image_name in image_set:
            if image_name == "class_names":
                continue
            # extract image name from image path
            image_full_path = os.path.join(image_dir, image_name)
            annotation = data[image_name]
            image = Image.open(image_full_path)
            image_width, image_height = image.size

            label_file = os.path.join(
                label_output_dir, os.path.splitext(image_name)[0] + ".txt"
            )
            with open(label_file, "w") as f:
                for box, label in zip(annotation["boxes"], annotation["labels"]):
                    yolo_box = convert_to_yolo_format(box, image_width, image_height)
                    f.write(f"{label} {' '.join(map(str, yolo_box))}\n")

            shutil.copy(image_full_path, os.path.join(image_output_dir, image_name))


def create_data_yaml(root_dir, class_names):
    """Creates a YAML file for dataset configuration, specifying paths and class names.

    Args:
    - root_dir (str): The root directory where the dataset is located.
    - class_names (list of str): A list of class names.

    No return value.
    """
    yaml_content = (
        f"train: {os.path.abspath(os.path.join(root_dir, 'train'))}\n"
        f"val: {os.path.abspath(os.path.join(root_dir, 'val'))}\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}"
    )
    with open(os.path.join(root_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)


def convert(dataset_dir, output_dir, train_val_split_ratio):
    """Converts a dataset into a format suitable for training with YOLO, including
    creating training and validation splits.

    Args:
    - dataset_dir (str): The directory where the source dataset is located.
    - output_dir (str): The directory where the processed dataset should be saved.
    - train_val_split_ratio (float): The ratio to split the dataset into training and validation sets.

    No return value.
    """
    annotation_path = os.path.join(dataset_dir, "annotations.json")
    data = read_annotations(annotation_path)
    process_data(data, dataset_dir, output_dir, train_val_split_ratio)
    create_data_yaml(output_dir, data["class_names"])


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset to YOLO format with train-val split."
    )
    parser.add_argument(
        "--save_dir", type=str, help="Directory containing the images and annotations."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Train-validation split ratio (default: 0.8)",
    )

    args = parser.parse_args()

    convert(args.save_dir, args.output_dir, args.split_ratio)


if __name__ == "__main__":
    main()
