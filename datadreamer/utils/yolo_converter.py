from __future__ import annotations

import os
import shutil

from PIL import Image

from datadreamer.utils import BaseConverter


class YOLOConverter(BaseConverter):
    """Class for converting a dataset to YOLO format.

    Format:

    dataset_dir
    ├── train
    │   ├── images
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   ├── labels
    │   │   ├── 0.txt
    │   │   ├── 1.txt
    ├── val
    │   ├── images
    │   ├── labels
    ├── test
    │   ├── images
    │   ├── labels
    """

    def __init__(self, seed=42):
        super().__init__(seed)

    def convert(self, dataset_dir, output_dir, split_ratios, copy_files=True):
        """Converts a dataset into a format suitable for training with YOLO, including
        creating training and validation splits.

        Args:
        - dataset_dir (str): The directory where the source dataset is located.
        - output_dir (str): The directory where the processed dataset should be saved.
        - split_ratios (list of float): The ratios to split the data into training, validation, and test sets.
        - copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.

        No return value.
        """
        annotation_path = os.path.join(dataset_dir, "annotations.json")
        data = BaseConverter.read_annotations(annotation_path)
        self.process_data(data, dataset_dir, output_dir, split_ratios, copy_files)

    def convert_to_yolo_format(self, box, image_width, image_height):
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

    def process_data(self, data, image_dir, output_dir, split_ratios, copy_files=True):
        """Processes the data by dividing it into training and validation sets, and
        saves the images and labels in YOLO format.

        Args:
        - data (dict): The dictionary containing image annotations.
        - image_dir (str): The directory where the source images are located.
        - output_dir (str): The base directory where the processed data will be saved.
        - split_ratios (float): The ratio to split the data into training, validation, and test sets.
        - copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.


        No return value.
        """
        images = list(data.keys())
        images.remove("class_names")

        train_images, val_images, test_images = BaseConverter.make_splits(
            images, split_ratios
        )

        for dataset_type, image_set in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
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
                        yolo_box = self.convert_to_yolo_format(
                            box, image_width, image_height
                        )
                        f.write(f"{label} {' '.join(map(str, yolo_box))}\n")

                if copy_files:
                    shutil.copy(
                        image_full_path, os.path.join(image_output_dir, image_name)
                    )
                else:
                    shutil.move(
                        image_full_path, os.path.join(image_output_dir, image_name)
                    )

        self.create_data_yaml(output_dir, data["class_names"])

    def create_data_yaml(self, root_dir, class_names):
        """Creates a YAML file for dataset configuration, specifying paths and class
        names.

        Args:
        - root_dir (str): The root directory where the dataset is located.
        - class_names (list of str): A list of class names.

        No return value.
        """
        yaml_content = (
            f"train: {os.path.abspath(os.path.join(root_dir, 'train'))}\n"
            f"val: {os.path.abspath(os.path.join(root_dir, 'val'))}\n"
            f"test: {os.path.abspath(os.path.join(root_dir, 'test'))}\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}"
        )
        with open(os.path.join(root_dir, "data.yaml"), "w") as f:
            f.write(yaml_content)
