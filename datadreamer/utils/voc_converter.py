from __future__ import annotations

import os
import shutil
from typing import Dict, List, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from pycocotools import mask as mask_utils

from datadreamer.utils import BaseConverter


class VOCConverter(BaseConverter):
    """Class for converting a dataset to PASCAL VOC format.

    Format:

    dataset_dir
    ├── Annotations/        # XML files with image annotations (bounding boxes, labels, segmentation)
    |   ├── 0.xml
    |   ├── 1.xml
    |   ├── ...
    ├── ImageSets/
    │   ├── Main/           # Main image splits (train, val, test, etc.)
    │       ├── train.txt
    │       ├── val.txt
    │       ├── test.txt
    ├── JPEGImages/         # Images in JPG format
    |   ├── 0.jpg
    |   ├── 1.jpg
    |   ├── ...
    ├── SegmentationClass/  # Segmentation masks (with class labels)
    |   ├── 0.png
    |   ├── 1.png
    |   ├── ...
    ├── SegmentationObject/ # Segmentation masks (object-wise masks)labels.json
    |   ├── 0.png
    |   ├── 1.png
    |   ├── ...
    """

    def __init__(self, seed=42, is_instance_segmentation: bool = False):
        super().__init__(seed)
        self.is_instance_segmentation = is_instance_segmentation

    def convert(
        self,
        dataset_dir: str,
        output_dir: str,
        split_ratios: List[float],
        keep_unlabeled_images: bool = False,
        copy_files: bool = True,
    ):
        """Convert a dataset to PASCAL VOC format.

        Args:
            dataset_dir (str): The directory where the source dataset is located.
            output_dir (str): The directory where the processed dataset should be saved.
            split_ratios (list of float): The ratios to split the data into training, validation, and test sets.
            keep_unlabeled_images (bool, optional): Whether to keep images with no annotations. Defaults to False.
            copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.

        No return value.
        """
        annotation_path = os.path.join(dataset_dir, "annotations.json")
        data = self.read_annotations(annotation_path)
        self.process_data(
            data,
            dataset_dir,
            output_dir,
            split_ratios,
            keep_unlabeled_images,
            copy_files,
        )

    def process_data(
        self,
        data: Dict,
        image_dir: str,
        output_dir: str,
        split_ratios: List[float],
        keep_unlabeled_images: bool = False,
        copy_files: bool = True,
    ):
        """Process the data, create XML annotations and handle dataset splitting.

        Args:
            data (dict): The dictionary containing image annotations.
            image_dir (str): The directory where the source images are located.
            output_dir (str): The directory where the processed dataset should be saved.
            split_ratios (list of float): The ratios to split the data into training, validation, and test sets.
            keep_unlabeled_images (bool, optional): Whether to keep images with no annotations. Defaults to False.
            copy_files (bool, optional): Whether to copy the source files to the output directory, otherwise move them. Defaults to True.

        No return value.
        """
        images = list(data.keys())
        images.remove("class_names")

        empty_images = list(filter(lambda x: len(data[x]["labels"]) == 0, images))
        if keep_unlabeled_images and len(empty_images) > 0:
            logger.warning(
                f"{len(empty_images)} images with no annotations will be included in the dataset."
            )
        elif not keep_unlabeled_images and len(empty_images) > 0:
            logger.info(
                f"{len(empty_images)} images with no annotations will be excluded from the dataset."
            )
            for image in empty_images:
                images.remove(image)

        annotations_dir = os.path.join(output_dir, "Annotations")
        images_dir = os.path.join(output_dir, "JPEGImages")
        segmentation_class_dir = os.path.join(output_dir, "SegmentationClass")
        segmentation_object_dir = os.path.join(output_dir, "SegmentationObject")
        image_sets_dir = os.path.join(output_dir, "ImageSets", "Main")

        for dir_path in [
            annotations_dir,
            images_dir,
            segmentation_class_dir,
            segmentation_object_dir,
            image_sets_dir,
        ]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)

        train_images, val_images, test_images = self.make_splits(images, split_ratios)

        self.create_image_sets(image_sets_dir, train_images, val_images, test_images)

        folder_name = os.path.basename(output_dir)
        for image_name in images:
            image_full_path = os.path.join(image_dir, image_name)
            annotation = data[image_name]
            image = Image.open(image_full_path)
            image_width, image_height = image.size

            annotation_xml = self.create_xml(
                annotation,
                folder_name,
                image_name,
                image_width,
                image_height,
                data["class_names"],
            )

            xml_path = os.path.join(annotations_dir, f"{image_name.split('.')[0]}.xml")
            with open(xml_path, "wb") as xml_file:
                xml_file.write(tostring(annotation_xml))

            if copy_files:
                shutil.copy(image_full_path, os.path.join(images_dir, image_name))
            else:
                shutil.move(image_full_path, os.path.join(images_dir, image_name))

            if self.is_instance_segmentation:
                class_mask, object_mask = self.create_segmentation_masks(
                    annotation, image_name, image_width, image_height
                )
                class_mask_path = os.path.join(
                    segmentation_class_dir, f"{image_name.split('.')[0]}.png"
                )
                object_mask_path = os.path.join(
                    segmentation_object_dir, f"{image_name.split('.')[0]}.png"
                )

                Image.fromarray(class_mask).save(class_mask_path)
                Image.fromarray(object_mask).save(object_mask_path)

    def create_image_sets(
        self,
        image_sets_dir: str,
        train_images: List[str],
        val_images: List[str],
        test_images: List[str],
    ) -> None:
        """Create text files for each split (train, val, test).

        Args:
            image_sets_dir (str): The directory where the image sets will be saved.
            train_images (list of str): The list of image names for the training set.
            val_images (list of str): The list of image names for the validation set.
            test_images (list of str): The list of image names for the test set.

        No return value.
        """
        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for split_name, image_set in splits.items():
            split_file_path = os.path.join(image_sets_dir, f"{split_name}.txt")
            with open(split_file_path, "w") as f:
                for image_name in image_set:
                    f.write(f"{image_name.split('.')[0]}\n")

    def create_xml(
        self,
        annotation: Dict,
        folder_name: str,
        image_name: str,
        width: int,
        height: int,
        class_names: List[str],
    ) -> Element:
        """Generate XML annotation for a single image.

        Args:
            annotation (dict): The dictionary containing image annotations.
            folder_name (str): The name of the folder where the image is located.
            image_name (str): The name of the image file.
            width (int): The width of the image.
            height (int): The height of the image.
            class_names (list of str): The list of class names.

        Returns:
            Element: The root element of the XML annotation.
        """
        annotation_xml = Element("annotation")

        folder = SubElement(annotation_xml, "folder")
        folder.text = folder_name

        filename = SubElement(annotation_xml, "filename")
        filename.text = image_name

        path = SubElement(annotation_xml, "path")
        path.text = os.path.join(folder_name, image_name)

        size = SubElement(annotation_xml, "size")
        SubElement(size, "width").text = str(width)
        SubElement(size, "height").text = str(height)
        SubElement(size, "depth").text = "3"

        for box, label in zip(annotation["boxes"], annotation["labels"]):
            obj_xml = SubElement(annotation_xml, "object")
            SubElement(obj_xml, "name").text = class_names[label]
            SubElement(obj_xml, "pose").text = "Unspecified"
            SubElement(obj_xml, "truncated").text = "0"
            SubElement(obj_xml, "difficult").text = "0"

            bndbox = SubElement(obj_xml, "bndbox")
            SubElement(bndbox, "xmin").text = str(int(box[0]))
            SubElement(bndbox, "ymin").text = str(int(box[1]))
            SubElement(bndbox, "xmax").text = str(int(box[2]))
            SubElement(bndbox, "ymax").text = str(int(box[3]))

        return annotation_xml

    def create_segmentation_masks(
        self, annotation: Dict, image_name: str, width: int, height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create and save segmentation masks (class and object).

        Args:
            annotation (dict): The dictionary containing image annotations.
            image_name (str): The name of the image file.
            width (int): The width of the image.
            height (int): The height of the image.

        Returns:
            tuple of np.ndarray: A tuple containing the class mask and object mask.
        """

        class_mask = np.zeros((width, height), dtype=np.uint8)
        object_mask = np.zeros((width, height), dtype=np.uint8)
        for i, (mask, label) in enumerate(
            zip(annotation["masks"], annotation["labels"]), 1
        ):
            if isinstance(mask, dict):
                binary_mask = mask_utils.decode(mask)
            else:
                binary_mask = np.zeros((width, height), dtype=np.uint8)
                mask = np.array([[int(p[0]), int(p[1])] for p in mask])
                cv2.fillPoly(binary_mask, [mask], 1)

            class_mask[binary_mask == 1] = label + 1
            object_mask[binary_mask == 1] = i

        return class_mask, object_mask
