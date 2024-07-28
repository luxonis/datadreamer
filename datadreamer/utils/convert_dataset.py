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
    dataset_id=None,
    copy_files=True,
    seed=42,
):
    if dataset_format == "yolo":
        converter = YOLOConverter(seed=seed)
    elif dataset_format == "coco":
        converter = COCOConverter(seed=seed)
    elif dataset_format == "luxonis-dataset":
        converter = LuxonisDatasetConverter(
            dataset_plugin=dataset_plugin,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            seed=seed,
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
        "--dataset_id",
        type=str,
        help="ID of the dataset to create if dataset_plugin is used",
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
        args.dataset_id,
        args.copy_files,
        args.seed,
    )


if __name__ == "__main__":
    main()
