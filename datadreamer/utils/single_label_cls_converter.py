from __future__ import annotations

import os
import shutil
from typing import Dict, List

from loguru import logger

from datadreamer.utils import BaseConverter


class SingleLabelClsConverter(BaseConverter):
    """Class for converting a dataset for single-label classification task.

    NOTE: The number of images after conversion may be different from the number of images in the original dataset, as images with zero or more than one labels are removed.

    Format:

    dataset_dir
    ├── train
    │   ├── class_1
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   ├── class_2
    │   │   ├── image3.jpg
    │   │   ├── image4.jpg
    ├── val
    │   ├── class_1
    │   ├── class_2
    ├── test
    │   ├── class_1
    │   ├── class_2
    """

    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def convert(
        self,
        dataset_dir: str,
        output_dir: str,
        split_ratios: List[float],
        keep_unlabeled_images: bool = False,
        copy_files: bool = True,
    ) -> None:
        """Converts a dataset into a format suitable for single-label classification.

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
        self.process_data(data, dataset_dir, output_dir, split_ratios, copy_files)

    def process_data(
        self,
        data: Dict,
        image_dir: str,
        output_dir: str,
        split_ratios: List[float],
        copy_files: bool = True,
    ) -> None:
        """Processes the data by removing images with multiple labels, then dividing it
        into training and validation sets, and saves the images with single labels.

        Args:
            data (dict): The dictionary containing image annotations.
            image_dir (str): The directory where the source images are located.
            output_dir (str): The base directory where the processed data will be saved.
            split_ratios (float): The ratio to split the data into training, validation, and test sets.
            copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.

        No return value.
        """
        images = list(data.keys())
        class_names = data["class_names"]
        images.remove("class_names")

        logger.info(f"Number of images: {len(images)}")

        # Remove images with multiple labels
        single_label_images = [img for img in images if len(data[img]["labels"]) == 1]

        logger.info(f"Number of images with single label: {len(single_label_images)}")

        # Split the data into training, validation, and test sets
        train_images, val_images, test_images = BaseConverter.make_splits(
            single_label_images, split_ratios
        )

        for dataset_type, image_set in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
            if os.path.exists(os.path.join(output_dir, dataset_type)):
                shutil.rmtree(os.path.join(output_dir, dataset_type))
            for label in class_names:
                image_output_dir = os.path.join(output_dir, dataset_type, label)
                os.makedirs(image_output_dir)

            for image_name in image_set:
                annotation = data[image_name]
                label = class_names[annotation["labels"][0]]
                image_full_path = os.path.join(image_dir, image_name)
                if copy_files:
                    shutil.copy(
                        image_full_path,
                        os.path.join(output_dir, dataset_type, label, image_name),
                    )
                else:
                    shutil.move(
                        image_full_path,
                        os.path.join(output_dir, dataset_type, label, image_name),
                    )
