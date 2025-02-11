from __future__ import annotations

from typing import List, Tuple

import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipImageTester:
    """A class for testing images against a set of textual objects using the CLIP model.

    Attributes:
        clip (CLIPModel): The CLIP model for image-text similarity evaluation.
        clip_processor (CLIPProcessor): The processor for preparing inputs to the CLIP model.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).

    Methods:
        test_image(image, objects, conf_threshold): Tests the given image against a list of objects.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initializes the ClipImageTester with the CLIP model and processor."""
        logger.info("Initializing CLIP image tester...")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.device = device
        self.clip.to(self.device)

    def test_image(
        self, image: Image.Image, objects: List[str], conf_threshold: float = 0.05
    ) -> Tuple[bool, torch.Tensor, int]:
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
        ).to(self.device)

        # Get similarity scores from the CLIP model
        outputs = self.clip(**inputs)
        logits_per_image = outputs.logits_per_image  # image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # label probabilities
        passed = torch.all(probs > conf_threshold).item()
        num_passed = torch.sum(probs > conf_threshold).item()

        # Check if all objects meet the confidence threshold
        return passed, probs, num_passed

    def test_images_batch(
        self,
        images: List[Image.Image],
        objects: List[List[str]],
        conf_threshold: float = 0.05,
    ) -> Tuple[List[bool], List[torch.Tensor], List[int]]:
        """Tests the generated images against a set of objects using the CLIP model.

        Args:
            images (List[Image.Image]): The images to be tested.
            objects (List[List[str]]): A list of objects (text) to test against the images.
            conf_threshold (float, optional): Confidence threshold for considering an object as present. Defaults to 0.05.

        Returns:
            Tuple[List[bool], List[torch.Tensor], List[int]]: A tuple containing a list of booleans indicating if the images pass the test,
                   a list of probabilities of the objects, and a list of the number of objects that passed the test.
        """
        # Transform the inputs for the CLIP model
        objects_array = []
        for obj_list in objects:
            objects_array.extend(obj_list)
        inputs = self.clip_processor(
            text=objects_array, images=images, return_tensors="pt", padding=True
        ).to(self.device)

        outputs = self.clip(**inputs)

        logits_per_image = outputs.logits_per_image  # image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # label probabilities

        # Cahnge the shape of the probs, passed and num_passed so they correspond to the initial tuples in the objects list
        probs_list = []
        passed_list = []
        num_passed_list = []

        start_pos = 0
        for i, obj_list in enumerate(objects):
            end_pos = start_pos + len(obj_list)
            probs_list.append(probs[i, start_pos:end_pos])
            passed_list.append(torch.all(probs_list[-1] > conf_threshold).item())
            num_passed_list.append(torch.sum(probs_list[-1] > conf_threshold).item())
            start_pos = end_pos

        # Check if all objects meet the confidence threshold
        return passed_list, probs_list, num_passed_list

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        self.clip = self.clip.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()
