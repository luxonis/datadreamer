import json
import os
import shutil
import unittest

import numpy as np
from PIL import Image

from datadreamer.utils import (
    dataset_utils,
    merge_raw_datasets,
)


def create_sample_image(
    image_name, image_size=(100, 100), color=(255, 0, 0), save_dir="test_images"
):
    """Create and save a simple image with a solid color.

    Args:
        image_name (str): The name of the image file.
        image_size (tuple): The size of the image (width, height).
        color (tuple): The RGB color of the image.
        save_dir (str): The directory to save the images.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a blank image with the given color
    img = Image.new("RGB", image_size, color)

    # Save the image to the specified directory
    img.save(os.path.join(save_dir, image_name))


class TestSaveAnnotationsToJson(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for saving images and JSON file
        self.test_dir = "test_dir"
        self.image_dir = "test_images"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        # Create sample images
        create_sample_image("image1.jpg", save_dir=self.image_dir)
        create_sample_image("image2.jpg", save_dir=self.image_dir)

        self.file_name = "annotations.json"
        self.image_paths = [
            os.path.join(self.image_dir, "image1.jpg"),
            os.path.join(self.image_dir, "image2.jpg"),
        ]
        self.labels_list = [
            [0],  # Labels for image1
            [1],  # Labels for image2
        ]
        self.labels_list = np.array(self.labels_list)
        self.boxes_list = [
            [[10, 10, 50, 50]],  # Bounding boxes for image1
            [[20, 20, 40, 40]],  # Bounding boxes for image2
        ]
        self.boxes_list = np.array(self.boxes_list)
        self.class_names = ["class_1", "class_2"]

    def tearDown(self):
        # Clean up the test directory after each test
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        for file in os.listdir(self.image_dir):
            os.remove(os.path.join(self.image_dir, file))
        os.rmdir(self.test_dir)
        os.rmdir(self.image_dir)

    def test_save_annotations_to_json(self):
        # Test saving annotations to JSON
        dataset_utils.save_annotations_to_json(
            self.image_paths,
            self.labels_list,
            boxes_list=self.boxes_list,
            class_names=self.class_names,
            save_dir=self.test_dir,
            file_name=self.file_name,
        )

        # Load the saved JSON file and check contents
        with open(os.path.join(self.test_dir, self.file_name), "r") as f:
            annotations = json.load(f)

        # Check if annotations are correct
        self.assertEqual(len(annotations), 3)  # 2 images + class_names
        self.assertIn("image1.jpg", annotations)
        self.assertIn("image2.jpg", annotations)
        self.assertEqual(annotations["image1.jpg"]["labels"], [0])
        self.assertEqual(annotations["image2.jpg"]["labels"], [1])
        self.assertEqual(annotations["class_names"], self.class_names)


class TestMergeDatasets(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for test datasets
        self.input_dir_1 = "input_dir_1"
        self.input_dir_2 = "input_dir_2"
        self.input_dir_3 = "input_dir_3"
        self.output_dir = "output_dir"
        os.makedirs(self.input_dir_1, exist_ok=True)
        os.makedirs(self.input_dir_2, exist_ok=True)
        os.makedirs(self.input_dir_3, exist_ok=True)

        # Create generation_args.json files
        self.generation_args_1 = {
            "task": "object_detection",
            "class_names": ["class_1", "class_2"],
            "seed": 1,
        }
        self.generation_args_2 = {
            "task": "object_detection",
            "class_names": ["class_1", "class_2"],
            "seed": 2,
        }
        with open(os.path.join(self.input_dir_1, "generation_args.yaml"), "w") as f:
            json.dump(self.generation_args_1, f)
        with open(os.path.join(self.input_dir_2, "generation_args.yaml"), "w") as f:
            json.dump(self.generation_args_2, f)

        # Create annotations.json files
        self.annotations_1 = {
            "image1.jpg": {"labels": [0]},
            "image2.jpg": {"labels": [1]},
            "class_names": ["class_1", "class_2"],
        }
        self.annotations_2 = {
            "image3.jpg": {"labels": [0]},
            "image4.jpg": {"labels": [1]},
            "class_names": ["class_1", "class_2"],
        }
        with open(os.path.join(self.input_dir_1, "annotations.json"), "w") as f:
            json.dump(self.annotations_1, f)
        with open(os.path.join(self.input_dir_2, "annotations.json"), "w") as f:
            json.dump(self.annotations_2, f)

        # Create image files
        with open(os.path.join(self.input_dir_1, "image1.jpg"), "wb") as f:
            f.write(os.urandom(1024))  # Dummy image content
        with open(os.path.join(self.input_dir_1, "image2.jpg"), "wb") as f:
            f.write(os.urandom(1024))  # Dummy image content
        with open(os.path.join(self.input_dir_2, "image3.jpg"), "wb") as f:
            f.write(os.urandom(1024))  # Dummy image content
        with open(os.path.join(self.input_dir_2, "image4.jpg"), "wb") as f:
            f.write(os.urandom(1024))  # Dummy image content

    def tearDown(self):
        # Clean up the test directories after each test
        shutil.rmtree(self.input_dir_1)
        shutil.rmtree(self.input_dir_2)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_merge_datasets(self):
        # Test merging datasets
        merge_raw_datasets.merge_datasets(
            [self.input_dir_1, self.input_dir_2], self.output_dir, copy_files=True
        )

        # Check if output directory is created
        self.assertTrue(os.path.exists(self.output_dir))

        # Check if annotations.json is merged correctly
        with open(os.path.join(self.output_dir, "annotations.json"), "r") as f:
            merged_annotations = json.load(f)

        print(merged_annotations)

        self.assertEqual(len(merged_annotations), 5)  # 4 images in total + class_names
        self.assertIn("image1.jpg", merged_annotations)
        self.assertIn("image2.jpg", merged_annotations)
        self.assertIn("image3.jpg", merged_annotations)
        self.assertIn("image4.jpg", merged_annotations)
        self.assertEqual(merged_annotations["class_names"], ["class_1", "class_2"])

        # Check if images are copied correctly
        for image_name in ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]:
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, image_name)))


if __name__ == "__main__":
    unittest.main()
