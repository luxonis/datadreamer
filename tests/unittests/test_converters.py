import json
import os
import shutil
import unittest

from PIL import Image

from datadreamer.utils import (
    BaseConverter,
    COCOConverter,
    YOLOConverter,
)


class TestBaseConverter(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_dataset"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create sample annotations
        self.annotations = {
            "class_names": ["cat", "dog"],
            "0.jpg": {"boxes": [[10, 10, 50, 50]], "labels": [0]},
            "1.jpg": {"boxes": [[20, 20, 70, 70]], "labels": [1]},
        }
        with open(os.path.join(self.test_dir, "annotations.json"), "w") as f:
            json.dump(self.annotations, f)

        # Create sample images
        open(os.path.join(self.test_dir, "0.jpg"), "a").close()
        open(os.path.join(self.test_dir, "1.jpg"), "a").close()

    def tearDown(self):
        os.remove(os.path.join(self.test_dir, "annotations.json"))
        os.remove(os.path.join(self.test_dir, "0.jpg"))
        os.remove(os.path.join(self.test_dir, "1.jpg"))
        os.rmdir(self.test_dir)

    def test_read_annotations(self):
        annotation_path = os.path.join(self.test_dir, "annotations.json")
        data = BaseConverter.read_annotations(annotation_path)
        self.assertEqual(data, self.annotations)

    def test_make_splits(self):
        images = ["0.jpg", "1.jpg"]
        split_ratios = [0.5, 0.5, 0.0]
        train_images, val_images, test_images = BaseConverter.make_splits(
            images, split_ratios, shuffle=False
        )

        self.assertEqual(len(train_images), 1)
        self.assertEqual(len(val_images), 1)
        self.assertEqual(len(test_images), 0)
        self.assertTrue("0.jpg" in train_images)
        self.assertTrue("1.jpg" in val_images)


class TestCOCOConverter(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_dataset"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create sample images
        self.image_size = (100, 100)
        self.create_sample_image("0.jpg")
        self.create_sample_image("1.jpg")

        # Create sample labels
        self.labels = {
            "class_names": ["cat", "dog"],
            "0.jpg": {"boxes": [(10, 10, 50, 50)], "labels": [0]},
            "1.jpg": {"boxes": [(20, 20, 70, 70)], "labels": [1]},
        }
        with open(os.path.join(self.test_dir, "annotations.json"), "w") as f:
            json.dump(self.labels, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if hasattr(self, "output_dir") and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def create_sample_image(self, filename):
        image = Image.new("RGB", self.image_size, color="white")
        image.save(os.path.join(self.test_dir, filename))

    def test_convert(self):
        self.output_dir = "output_dir"
        split_ratios = [0.6, 0.2, 0.2]
        converter = COCOConverter()
        converter.convert(self.test_dir, self.output_dir, split_ratios, copy_files=True)

        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "train")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "validation")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test")))

        # Test whether labels.json files exist in all output directories
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "train", "labels.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "validation", "labels.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test", "labels.json"))
        )

    def test_process_data(self):
        self.output_dir = "output_dir"
        split_ratios = [0.6, 0.2, 0.2]
        converter = COCOConverter()
        converter.process_data(
            self.labels, self.test_dir, self.output_dir, split_ratios, copy_files=True
        )

        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "train")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "validation")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test")))

        # Test whether labels.json files exist in all output directories
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "train", "labels.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "validation", "labels.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "test", "labels.json"))
        )

    def test_save_labels(self):
        self.output_dir = "output_dir"
        converter = COCOConverter()
        images_info = [
            {"id": 1, "file_name": "0.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "1.jpg", "width": 100, "height": 100},
        ]
        annotations = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 40, 40],
                "segmentation": None,
                "area": 1200,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [20, 20, 50, 50],
                "segmentation": None,
                "area": 1500,
                "iscrowd": 0,
            },
        ]
        class_names = ["cat", "dog"]

        # Test whether labels.json file is saved correctly
        os.makedirs(self.output_dir)
        converter.save_labels(self.output_dir, images_info, annotations, class_names)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "labels.json")))

        # Test whether the content of labels.json is correct
        with open(os.path.join(self.output_dir, "labels.json"), "r") as f:
            saved_labels = json.load(f)

        self.assertEqual(saved_labels["images"], images_info)
        self.assertEqual(saved_labels["annotations"], annotations)
        self.assertEqual(
            saved_labels["categories"],
            [{"id": i, "name": name} for i, name in enumerate(class_names)],
        )


class TestYOLOConverter(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_dataset"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create sample images
        self.image_size = (100, 100)
        self.create_sample_image("0.jpg")
        self.create_sample_image("1.jpg")

        # Create sample labels
        self.labels = {
            "class_names": ["cat", "dog"],
            "0.jpg": {"boxes": [(10, 10, 50, 50)], "labels": [0]},
            "1.jpg": {"boxes": [(20, 20, 70, 70)], "labels": [1]},
        }
        with open(os.path.join(self.test_dir, "annotations.json"), "w") as f:
            json.dump(self.labels, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if hasattr(self, "output_dir") and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def create_sample_image(self, filename):
        image = Image.new("RGB", self.image_size, color="white")
        image.save(os.path.join(self.test_dir, filename))

    def test_convert_to_yolo_format(self):
        converter = YOLOConverter()
        yolo_format = converter.convert_to_yolo_format([10, 10, 50, 50], 100, 100)
        self.assertEqual(yolo_format, [0.3, 0.3, 0.4, 0.4])

    def test_process_data(self):
        self.output_dir = "output_dir"
        split_ratios = [1, 0, 0]
        converter = YOLOConverter()
        converter.process_data(
            self.labels, self.test_dir, self.output_dir, split_ratios, copy_files=True
        )

        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "train")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "val")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test")))

        # Test whether labels files exist in all output directories
        train_label_file = os.path.join(self.output_dir, "train", "labels", "0.txt")
        self.assertTrue(os.path.exists(train_label_file))
        with open(train_label_file, "r") as f:
            content = f.read()
            self.assertEqual(content.strip(), "0 0.3 0.3 0.4 0.4")

    def test_create_data_yaml(self):
        self.output_dir = "output_dir"
        converter = YOLOConverter()
        class_names = ["cat", "dog"]
        os.makedirs(self.output_dir, exist_ok=True)
        converter.create_data_yaml(self.output_dir, class_names)

        yaml_file = os.path.join(self.output_dir, "data.yaml")
        self.assertTrue(os.path.exists(yaml_file))

        with open(yaml_file, "r") as f:
            content = f.read()
            self.assertIn("train:", content)
            self.assertIn("val:", content)
            self.assertIn("test:", content)
            self.assertIn("nc: 2", content)
            self.assertIn("names: ['cat', 'dog']", content)


if __name__ == "__main__":
    unittest.main()
