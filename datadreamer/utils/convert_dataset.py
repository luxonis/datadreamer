from __future__ import annotations

import argparse
from typing import List, Optional

from datadreamer.utils import (
    COCOConverter,
    LuxonisDatasetConverter,
    SingleLabelClsConverter,
    VOCConverter,
    YOLOConverter,
)


def convert_dataset(
    input_dir: str,
    output_dir: str,
    dataset_format: str,
    split_ratios: List[float],
    dataset_plugin: Optional[str] = None,
    dataset_name: Optional[str] = None,
    is_instance_segmentation: bool = False,
    keep_unlabeled_images: bool = False,
    copy_files: bool = True,
    seed: int = 42,
) -> None:
    """Converts a dataset from one format to another.

    Args:
        input_dir (str): Directory containing the images and annotations.
        output_dir (str): Directory where the processed dataset will be saved.
        dataset_format (str): Format of the dataset. Can be 'yolo', 'coco', 'voc', 'luxonis-dataset', or 'cls-single'.
        split_ratios (lis of float): List of ratios for train, val, and test splits.
        dataset_plugin (str, optional): Plugin for Luxonis dataset. Defaults to None.
        dataset_name (str, optional): Name of the Luxonis dataset. Defaults to None.
        is_instance_segmentation (bool, optional): Whether the dataset is for instance segmentation. Defaults to False.
        keep_unlabeled_images (bool, optional): Whether to keep images with no annotations. Defaults to False.
        copy_files (bool, optional): Whether to copy the files to the output directory. Defaults to True.
        seed (int, optional): Random seed. Defaults to 42.

    No return value.
    """

    if dataset_format == "yolo":
        converter = YOLOConverter(
            seed=seed, is_instance_segmentation=is_instance_segmentation
        )
    elif dataset_format == "coco":
        converter = COCOConverter(
            seed=seed, is_instance_segmentation=is_instance_segmentation
        )
    elif dataset_format == "voc":
        converter = VOCConverter(
            seed=seed, is_instance_segmentation=is_instance_segmentation
        )
    elif dataset_format == "luxonis-dataset":
        converter = LuxonisDatasetConverter(
            dataset_plugin=dataset_plugin,
            dataset_name=dataset_name,
            seed=seed,
            is_instance_segmentation=is_instance_segmentation,
        )
    elif dataset_format == "cls-single":
        converter = SingleLabelClsConverter(seed=seed)
    else:
        raise ValueError(f"Invalid dataset format: {dataset_format}")

    converter.convert(
        input_dir, output_dir, split_ratios, keep_unlabeled_images, copy_files
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw dataset to another format with train-val-test split."
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing the images and annotations."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="yolo",
        choices=["yolo", "coco", "voc", "luxonis-dataset", "cls-single"],
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs="+",
        default=[0.8, 0.1, 0.1],
        help="Train-validation-test split ratios (default: 0.8, 0.1, 0.1).",
    )
    parser.add_argument(
        "--dataset_plugin",
        type=str,
        default=None,
        help="Dataset plugin to use for luxonis-dataset format.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to create if dataset_plugin is used",
    )
    parser.add_argument(
        "--is_instance_segmentation",
        default=None,
        action="store_true",
        help="Whether the dataset is for instance segmentation.",
    )
    parser.add_argument(
        "--keep_unlabeled_images",
        default=None,
        action="store_true",
        help="Whether to keep images without any annotations",
    )
    parser.add_argument(
        "--copy_files",
        type=bool,
        default=True,
        help="Copy files to output directory, otherwise move them.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_format=args.dataset_format,
        split_ratios=args.split_ratios,
        dataset_plugin=args.dataset_plugin,
        dataset_name=args.dataset_name,
        is_instance_segmentation=args.is_instance_segmentation,
        keep_unlabeled_images=args.keep_unlabeled_images,
        copy_files=args.copy_files,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
