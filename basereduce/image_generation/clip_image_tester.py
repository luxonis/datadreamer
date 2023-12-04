from typing import List
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

class ClipImageTester():
    def __init__(self) -> None:
        # Initialize CLIP model and processor
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def test_image(self, image: Image.Image, objects: List[str], conf_threshold=0.05) -> bool:
        """
        Tests the generated image against a set of objects using the CLIP model.

        :param image: The image to test.
        :param objects: A list of objects (text) to test against the image.
        :param conf_threshold: Confidence threshold for object detection.
        :return: True if the image passes the test, False otherwise.
        """
        # Process the inputs for the CLIP model
        inputs = self.clip_processor(text=objects, images=image, return_tensors="pt", padding=True)

        # Get similarity scores from the CLIP model
        outputs = self.clip(**inputs)
        logits_per_image = outputs.logits_per_image  # image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # label probabilities

        # Check if all objects meet the confidence threshold
        return torch.all(probs > conf_threshold).item()