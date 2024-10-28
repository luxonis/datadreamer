from __future__ import annotations

import argparse

from datadreamer.utils import (
    COCOConverter,
    LuxonisDatasetConverter,
    SingleLabelClsConverter,
    YOLOConverter,
)


def convert_dataset(
    input_dir,
    output_dir,
    dataset_format,
    split_ratios,
    dataset_plugin=None,
    dataset_name=None,
    is_instance_segmentation=False,
    copy_files=True,
    seed=42,
) -> None:
    """Converts a dataset from one format to another.

    Args:
        input_dir (str): Directory containing the images and annotations.
        output_dir (str): Directory where the processed dataset will be saved.
        dataset_format (str): Format of the dataset. Can be 'yolo', 'coco', 'luxonis-dataset', or 'cls-single'.
        split_ratios (list): List of ratios for train, val, and test splits.
        dataset_plugin (str, optional): Plugin for Luxonis dataset. Defaults to None.
        dataset_name (str, optional): Name of the Luxonis dataset. Defaults to None.
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

    converter.convert(input_dir, output_dir, split_ratios, copy_files)


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
        choices=["yolo", "coco", "luxonis-dataset", "cls-single"],
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
        args.input_dir,
        args.output_dir,
        args.dataset_format,
        args.split_ratios,
        args.dataset_plugin,
        args.dataset_name,
        args.copy_files,
        args.seed,
    )


if __name__ == "__main__":
    main()
