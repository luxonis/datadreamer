from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class BaseConverter(ABC):
    """Abstract base class for converter."""

    def __init__(self, seed=42):
        np.random.seed(seed)

    @abstractmethod
    def convert(
        self,
        dataset_dir: str,
        output_dir: str,
        split_ratios: List[float],
        keep_unlabeled_images: bool = False,
        copy_files: bool = True,
    ) -> None:
        """Converts a dataset into another format.

        Args:
            dataset_dir (str): The directory where the source dataset is located.
            output_dir (str): The directory where the processed dataset should be saved.
            split_ratios (list of float): The ratios to split the data into training, validation, and test sets.
            keep_unlabeled_images (bool, optional): Whether to keep images with no annotations. Defaults to False.
            copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.

        No return value.
        """
        pass

    @staticmethod
    def read_annotations(annotation_path: str) -> Dict:
        """Reads annotations from a JSON file located at the specified path.

        Args:
            annotation_path (str): The path to the JSON file containing annotations.

        Returns:
            dict: A dictionary containing the data loaded from the JSON file.
        """
        with open(annotation_path) as f:
            data = json.load(f)
        return data

    @staticmethod
    def make_splits(
        images: List[str], split_ratios: List[float], shuffle: bool = True
    ) -> Tuple[List, List, List]:
        """Splits the list of images into training, validation, and test sets.

        Args:
            images (list of str): A list of image paths.
            split_ratios (list of float): The ratios to split the data into training, validation, and test sets.
            shuffle (bool, optional): Whether to shuffle the list of images. Defaults to True.

        Returns:
            list of str: A list of image paths for the training set.
            list of str: A list of image paths for the validation set.
            list of str: A list of image paths for the test set.
        """
        if shuffle:
            np.random.shuffle(images)

        train_images = images[: int(len(images) * split_ratios[0])]
        val_images = images[
            int(len(images) * split_ratios[0]) : int(
                len(images) * (split_ratios[0] + split_ratios[1])
            )
        ]
        test_images = images[int(len(images) * (split_ratios[0] + split_ratios[1])) :]

        return train_images, val_images, test_images
