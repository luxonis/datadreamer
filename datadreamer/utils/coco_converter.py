from __future__ import annotations

import json
import os
import shutil

from PIL import Image

from datadreamer.utils.base_converter import BaseConverter


class COCOConverter(BaseConverter):
    """Class for converting a dataset to COCO format.

    Format:

    dataset_dir
    ├── train
    │   ├── data
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   ├── labels.json
    ├── validation
    │   ├── data
    │   ├── labels.json
    ├── test
    │   ├── data
    │   ├── labels.json
    """

    def __init__(self, seed=42):
        super().__init__(seed)

    def convert(self, dataset_dir, output_dir, split_ratios, copy_files=True):
        """Converts a dataset into a COCO format.

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

    def process_data(self, data, image_dir, output_dir, split_ratios, copy_files=True):
        """Processes the data by dividing it into training and validation sets, and
        saves the images and labels in COCO format.

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
            ("validation", val_images),
            ("test", test_images),
        ]:
            dataset_output_dir = os.path.join(output_dir, dataset_type)
            data_output_dir = os.path.join(dataset_output_dir, "data")

            if os.path.exists(data_output_dir):
                shutil.rmtree(data_output_dir)

            os.makedirs(data_output_dir)

            images_info = []
            annotations = []
            annotation_id = 0

            for image_name in image_set:
                image_full_path = os.path.join(image_dir, image_name)
                annotation = data[image_name]
                image = Image.open(image_full_path)
                image_width, image_height = image.size

                images_info.append(
                    {
                        "id": len(images_info) + 1,
                        "file_name": image_name,
                        "width": image_width,
                        "height": image_height,
                    }
                )

                for box, label in zip(annotation["boxes"], annotation["labels"]):
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": len(images_info),
                            "category_id": label,
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                            "segmentation": None,  # [[box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]], # bbox mask
                            "area": (box[2] - box[0]) * (box[3] - box[1]),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

                if copy_files:
                    shutil.copy(
                        image_full_path, os.path.join(data_output_dir, image_name)
                    )
                else:
                    shutil.move(
                        image_full_path, os.path.join(data_output_dir, image_name)
                    )

            self.save_labels(
                dataset_output_dir, images_info, annotations, data["class_names"]
            )

    def save_labels(self, dataset_output_dir, images_info, annotations, class_names):
        """Saves the labels to a JSON file.

        Args:
        - dataset_output_dir (str): The directory where the labels should be saved.
        - images_info (list of dict): A list of dictionaries containing image information.
        - annotations (list of dict): A list of dictionaries containing annotation information.
        - class_names (list of str): A list of class names.

        No return value.
        """

        with open(os.path.join(dataset_output_dir, "labels.json"), "w") as f:
            json.dump(
                {
                    "images": images_info,
                    "annotations": annotations,
                    "categories": [
                        {"id": i, "name": name} for i, name in enumerate(class_names)
                    ],
                },
                f,
            )
