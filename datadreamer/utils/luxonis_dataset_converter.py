from __future__ import annotations

import os

from luxonis_ml.data import DATASETS_REGISTRY, LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage
from PIL import Image

from datadreamer.utils import BaseConverter


class LuxonisDatasetConverter(BaseConverter):
    """Class for converting a dataset to LuxonisDataset format."""

    def __init__(
        self, dataset_plugin=None, dataset_name=None, dataset_id=None, seed=42
    ):
        super().__init__(seed)
        self.dataset_plugin = dataset_plugin
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id

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
                        "annotation": {
                            "class": class_names[label],
                            "type": "classification",
                            # "value": True,
                        },
                    }

                if "boxes" in data[image_path]:
                    boxes = data[image_path]["boxes"]
                    for box in boxes:
                        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                        x = max(0, x)
                        y = max(0, y)
                        yield {
                            "file": image_full_path,
                            "annotation": {
                                "class": class_names[label],
                                "type": "boundingbox",
                                "x": x / width,
                                "y": y / height,
                                "w": w / width,
                                "h": h / height,
                            },
                        }

        dataset_name = (
            os.path.basename(output_dir)
            if self.dataset_name is None
            else self.dataset_name
        )
        if LuxonisDataset.exists(dataset_name):
            dataset = LuxonisDataset(dataset_name)
            dataset.delete_dataset()

        # if dataset_plugin is set, use that
        if self.dataset_plugin:
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                print(f"Using {self.dataset_plugin} dataset")
                dataset_constructor = DATASETS_REGISTRY.get(self.dataset_plugin)
                dataset = dataset_constructor(dataset_name, self.dataset_id)
            else:
                raise ValueError(
                    "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set for using the dataset plugin."
                )
        # if LUXONISML_BUCKET and GOOGLE_APPLICATION_CREDENTIALS are set, use GCS bucket
        elif (
            "LUXONISML_BUCKET" in os.environ
            and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
        ):
            print("Using GCS bucket")
            dataset = LuxonisDataset(dataset_name, bucket_storage=BucketStorage.GCS)
        else:
            print("Using local dataset")
            dataset = LuxonisDataset(dataset_name)

        # NOTE: Not implemented in the LuxonisOnlineDataset yet
        # dataset.set_classes(class_names, task = "classification")
        # if is_detection:
        #     print("Setting task to boundingbox")
        #     dataset.set_classes(class_names, task = "boundingbox")

        dataset.add(dataset_generator())

        dataset.make_splits(split_ratios)
