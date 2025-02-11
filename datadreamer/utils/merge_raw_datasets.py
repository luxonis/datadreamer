from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import List

from loguru import logger


def merge_datasets(
    input_dirs: List[str], output_dir: str, copy_files: bool = True
) -> None:
    """Merges multiple raw datasets into a single dataset.

    Args:
        input_dirs (List[str]): A list of input directories containing raw datasets.
        output_dir (str): The output directory where the merged dataset will be saved.
        copy_files (bool, optional): Whether to copy the files from the input directories
            to the output directory. Defaults to True.

    No return value.
    """
    # Check if all input directories exist
    config_tasks = []
    config_classes = []
    random_seeds = []
    for input_dir in input_dirs:
        with open(os.path.join(input_dir, "generation_args.yaml")) as f:
            generation_args = json.load(f)
        config_tasks.append(generation_args["task"])
        config_classes.append(generation_args["class_names"])
        random_seeds.append(generation_args["seed"])

    # Check if all tasks are the same
    if len(set(config_tasks)) != 1:
        raise ValueError("All datasets must have the same task")
    # Check if all list of classes are the same
    if len(set(tuple(sorted(classes)) for classes in config_classes)) != 1:
        raise ValueError("All datasets must have the same list of classes")

    # Check if all datasets have different random seeds
    if len(set(random_seeds)) != len(input_dirs):
        raise ValueError("All datasets must have different random seeds")

    # Create output directory
    logger.info(f"Output directory: {output_dir}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    annotations_merged = {}
    for i, input_dir in enumerate(input_dirs):
        with open(os.path.join(input_dir, "annotations.json")) as f:
            annotations = json.load(f)
            class_names = annotations.pop("class_names")
            annotations_merged = {**annotations_merged, **annotations}

        # Copy or move generation_args.yaml files
        if copy_files:
            shutil.copy(
                os.path.join(input_dir, "generation_args.yaml"),
                os.path.join(output_dir, f"generation_args_{i}.yaml"),
            )
        else:
            shutil.move(
                os.path.join(input_dir, "generation_args.yaml"),
                os.path.join(output_dir, f"generation_args_{i}.yaml"),
            )

        # Copy or move images
        for image_path in annotations:
            if copy_files:
                shutil.copy(
                    os.path.join(input_dir, image_path),
                    os.path.join(output_dir, image_path),
                )
            else:
                shutil.move(
                    os.path.join(input_dir, image_path),
                    os.path.join(output_dir, image_path),
                )

    annotations_merged["class_names"] = class_names
    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations_merged, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Merge raw datasets")
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        help="Directories containing the images and annotations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where the merged dataset will be saved.",
    )
    parser.add_argument(
        "--copy_files",
        type=bool,
        default=True,
        help="Copy files to output directory, otherwise move them.",
    )

    args = parser.parse_args()

    merge_datasets(args.input_dirs, args.output_dir, args.copy_files)


if __name__ == "__main__":
    main()
