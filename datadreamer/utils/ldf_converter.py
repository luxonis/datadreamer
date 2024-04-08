from __future__ import annotations

import os

from luxonis_ml.data import LuxonisDataset
from PIL import Image

from datadreamer.utils import BaseConverter


class LDFConverter(BaseConverter):
    def convert(self, dataset_dir, output_dir, split_ratios, copy_files=True):
        """Converts a dataset into a LuxonisDataset format.

        Args:
        - dataset_dir (str): The directory where the source dataset is located.
        - output_dir (str): The directory where the processed dataset should be saved.
        - split_ratios (list of float): The ratios to split the data into training, validation, and test sets.
        - copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.

        No return value.
        """
        annotation_path = os.path.join(dataset_dir, "annotations.json")
        data = BaseConverter.read_annotations(annotation_path)
        self.process_data(data, dataset_dir, output_dir, split_ratios)

    def process_data(self, data, dataset_dir, output_dir, split_ratios):
        class_names = data["class_names"]
        image_paths = list(data.keys())
        image_paths.remove("class_names")

        def dataset_generator():
            # find image paths and load COCO annotations

            for image_path in image_paths:
                image_full_path = os.path.join(dataset_dir, image_path)
                width, height = Image.open(image_full_path).size
                labels = data[image_path]["labels"]
                for label in labels:
                    yield {
                        "file": image_full_path,
                        "class": class_names[label],
                        "type": "classification",
                        "value": True,
                    }

                if "boxes" in data[image_path]:
                    boxes = data[image_path]["boxes"]
                    for box in boxes:
                        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                        yield {
                            "file": image_full_path,
                            "class": class_names[label],
                            "type": "box",
                            "value": (x / width, y / height, w / width, h / height),
                        }

        dataset_name = os.path.basename(output_dir)
        if LuxonisDataset.exists(dataset_name):
            dataset = LuxonisDataset(dataset_name)
            dataset.delete_dataset()

        dataset = LuxonisDataset(dataset_name)
        dataset.set_classes(class_names)

        dataset.add(dataset_generator)

        dataset.make_splits(split_ratios)
