from __future__ import annotations

import os
from typing import Dict, List

from loguru import logger
from luxonis_ml.data import DATASETS_REGISTRY, LuxonisDataset
from luxonis_ml.data.utils.enums import BucketStorage
from PIL import Image

from datadreamer.utils import BaseConverter


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
        if output_dir is None:
            output_dir = dataset_dir
        class_names = data["class_names"]
        image_paths = list(data.keys())
        image_paths.remove("class_names")

        def dataset_generator():
            # find image paths and load COCO annotations

            for image_path in image_paths:
                image_full_path = os.path.join(dataset_dir, image_path)
                width, height = Image.open(image_full_path).size
                image_data = data[image_path]
                labels = image_data["labels"]

                if len(labels) == 0:
                    if keep_unlabeled_images:
                        logger.warning(
                            f"Image {image_path} has no annotations. Training on empty images with `luxonis-train` will result in an error."
                        )
                        yield {
                            "file": image_full_path,
                        }
                    else:
                        continue

                has_boxes = "boxes" in image_data
                has_masks = "masks" in image_data

                for i, label in enumerate(labels):
                    annotation = {
                        "class": class_names[label],
                    }

                    if has_boxes:
                        box = image_data["boxes"][i]
                        x, y = max(0, box[0] / width), max(0, box[1] / height)
                        w = min(box[2] / width - x, 1 - x)
                        h = min(box[3] / height - y, 1 - y)
                        annotation["boundingbox"] = {"x": x, "y": y, "w": w, "h": h}

                    if has_masks:
                        mask = image_data["masks"][i]
                        if isinstance(mask, list):
                            poly = [
                                (point[0] / width, point[1] / height) for point in mask
                            ]
                            annotation["instance_segmentation"] = {
                                "points": poly,
                                "height": height,
                                "width": width,
                            }
                        else:
                            annotation["instance_segmentation"] = {
                                "counts": mask["counts"],
                                "height": mask["size"][0],
                                "width": mask["size"][1],
                            }

                    yield {
                        "file": image_full_path,
                        "annotation": annotation,
                    }

        dataset_name = (
            os.path.basename(output_dir)
            if self.dataset_name is None or self.dataset_name == ""
            else self.dataset_name
        )

        # if dataset_plugin is set, use that
        if self.dataset_plugin:
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                logger.info(f"Using {self.dataset_plugin} dataset")
                dataset_constructor = DATASETS_REGISTRY.get(self.dataset_plugin)
                dataset = dataset_constructor(
                    dataset_name, delete_local=True, delete_remote=True
                )
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
            dataset = LuxonisDataset(
                dataset_name,
                bucket_storage=BucketStorage.GCS,
                delete_local=True,
                delete_remote=True,
            )
        else:
            logger.info("Using local dataset")
            dataset = LuxonisDataset(dataset_name, delete_local=True)

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
