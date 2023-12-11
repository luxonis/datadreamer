from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipImageTester:
    """A class for testing images against a set of textual objects using the CLIP model.

    Attributes:
        clip (CLIPModel): The CLIP model for image-text similarity evaluation.
        clip_processor (CLIPProcessor): The processor for preparing inputs to the CLIP model.

    Methods:
        test_image(image, objects, conf_threshold): Tests the given image against a list of objects.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(self) -> None:
        """Initializes the ClipImageTester with the CLIP model and processor."""
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def test_image(self, image: Image.Image, objects: List[str], conf_threshold=0.05):
        """Tests the generated image against a set of objects using the CLIP model.

        Args:
            image (Image.Image): The image to be tested.
            objects (List[str]): A list of objects (text) to test against the image.
            conf_threshold (float): Confidence threshold for considering an object as present.

        Returns:
            tuple: A tuple containing a boolean indicating if the image passes the test,
                   the probabilities of the objects, and the number of objects that passed the test.
        """
        # Process the inputs for the CLIP model
        inputs = self.clip_processor(
            text=objects, images=image, return_tensors="pt", padding=True
        )

        # Get similarity scores from the CLIP model
        outputs = self.clip(**inputs)
        logits_per_image = outputs.logits_per_image  # image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # label probabilities
        passed = torch.all(probs > conf_threshold).item()
        num_passed = torch.sum(probs > conf_threshold).item()

        # Check if all objects meet the confidence threshold
        return passed, probs, num_passed

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        self.clip = self.clip.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()
