from __future__ import annotations

import logging
import os
from typing import Dict, List

from luxonis_ml.data import DATASETS_REGISTRY, LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage
from PIL import Image

from datadreamer.utils import BaseConverter

logger = logging.getLogger(__name__)


class LuxonisDatasetConverter(BaseConverter):
    """Class for converting a dataset to LuxonisDataset format."""

    def __init__(
        self,
        dataset_plugin: str = None,
        dataset_name: str = None,
        seed: int = 42,
        is_instance_segmentation: bool = False,
    ):
        super().__init__(seed)
        self.is_instance_segmentation = is_instance_segmentation
        self.dataset_plugin = dataset_plugin
        self.dataset_name = dataset_name

        if self.is_instance_segmentation:
            logger.warning(
                "Instance segmentation will be treated as semantic segmentation until the support for instance segmentation is added to Luxonis-ml."
            )

    def convert(
        self,
        dataset_dir: str,
        output_dir: str,
        split_ratios: List[float],
        keep_unlabeled_images: bool = False,
        copy_files: bool = True,
    ) -> None:
        """Converts a dataset into a LuxonisDataset format.

        Args:
            dataset_dir (str): The directory where the source dataset is located.
            output_dir (str): The directory where the processed dataset should be saved.
            split_ratios (list of float): The ratios to split the data into training, validation, and test sets.
            keep_unlabeled_images (bool, optional): Whether to keep images with no annotations. Defaults to False.
            copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.

        No return value.
        """
        annotation_path = os.path.join(dataset_dir, "annotations.json")
        data = BaseConverter.read_annotations(annotation_path)
        self.process_data(
            data, dataset_dir, output_dir, split_ratios, keep_unlabeled_images
        )

    def process_data(
        self,
        data: Dict,
        dataset_dir: str,
        output_dir: str,
        split_ratios: List[float],
        keep_unlabeled_images: bool = False,
    ) -> None:
        """Processes the data into LuxonisDataset format.

        Args:
            data (dict): The data to process.
            dataset_dir (str): The directory where the source dataset is located.
            output_dir (str): The directory where the processed dataset should be saved.
            split_ratios (list of float): The ratios to split the data into training, validation, and test sets.

        No return value.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        class_names = data["class_names"]
        image_paths = list(data.keys())
        image_paths.remove("class_names")

        def dataset_generator():
            # find image paths and load COCO annotations

            for image_path in image_paths:
                image_full_path = os.path.join(dataset_dir, image_path)
                width, height = Image.open(image_full_path).size
                labels = data[image_path]["labels"]

                if len(labels) == 0 and keep_unlabeled_images:
                    logger.warning(
                        f"Image {image_path} has no annotations. Training on empty images with `luxonis-train` will result in an error."
                    )
                    yield {
                        "file": image_full_path,
                    }

                for label in labels:
                    yield {
                        "file": image_full_path,
                        "annotation": {
                            "class": class_names[label],
                            "type": "classification",
                            # "value": True,
                        },
                    }

                if "masks" in data[image_path]:  # polyline format
                    masks = data[image_path]["masks"]
                    for mask, label in zip(masks, labels):
                        poly = []
                        poly += [
                            (point[0] / width, point[1] / height) for point in mask
                        ]
                        yield {
                            "file": image_full_path,
                            "annotation": {
                                "type": "polyline",
                                "class": class_names[label],
                                "points": poly,  # masks,
                            },
                        }

                if "boxes" in data[image_path]:
                    boxes = data[image_path]["boxes"]
                    for box, label in zip(boxes, labels):
                        x, y = max(0, box[0] / width), max(0, box[1] / height)
                        w = min(box[2] / width - x, 1 - x)
                        h = min(box[3] / height - y, 1 - y)
                        yield {
                            "file": image_full_path,
                            "annotation": {
                                "class": class_names[label],
                                "type": "boundingbox",
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
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
                logger.info(f"Using {self.dataset_plugin} dataset")
                dataset_constructor = DATASETS_REGISTRY.get(self.dataset_plugin)
                dataset = dataset_constructor(dataset_name)
            else:
                raise ValueError(
                    "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set for using the dataset plugin."
                )
        # if LUXONISML_BUCKET and GOOGLE_APPLICATION_CREDENTIALS are set, use GCS bucket
        elif (
            "LUXONISML_BUCKET" in os.environ
            and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
        ):
            logger.info("Using GCS bucket")
            dataset = LuxonisDataset(dataset_name, bucket_storage=BucketStorage.GCS)
        else:
            logger.info("Using local dataset")
            dataset = LuxonisDataset(dataset_name)

        dataset.add(dataset_generator())

        if not keep_unlabeled_images:
            n_empty_images = len(
                list(filter(lambda x: len(data[x]["labels"]) == 0, image_paths))
            )
            if n_empty_images > 0:
                logger.info(
                    f"Removed {n_empty_images} empty images with no annotations from the dataset."
                )
        dataset.make_splits(tuple(split_ratios))
