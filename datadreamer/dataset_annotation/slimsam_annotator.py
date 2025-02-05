from __future__ import annotations

import logging
from typing import List

import numpy as np
import PIL
import torch
from transformers import SamModel, SamProcessor

from datadreamer.dataset_annotation.image_annotator import BaseAnnotator
from datadreamer.dataset_annotation.utils import mask_to_polygon

logger = logging.getLogger(__name__)


class SlimSAMAnnotator(BaseAnnotator):
    """A class for image annotation using the SlimSAM model, specializing in instance
    segmentation.

    Attributes:
        model (SAM): The SAM model for instance segmentation.
        processor (SamProcessor): The processor for the SAM model.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).
        size (str): The size of the SAM model to use ('base' or 'large').

    Methods:
        _init_model(): Initializes the SAM model.
        _init_processor(): Initializes the processor for the SAM model.
        annotate_batch(image, prompts, conf_threshold, use_tta, synonym_dict): Annotates the given image with bounding boxes and labels.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        seed: float = 42,
        device: str = "cuda",
        size: str = "base",
    ) -> None:
        """Initializes the SAMAnnotator with a specific seed and device.

        Args:
            seed (float): Seed for reproducibility. Defaults to 42.
            device (str): The device to run the model on. Defaults to 'cuda'.
        """
        super().__init__(seed)
        self.size = size
        self.model = self._init_model()
        self.processor = self._init_processor()
        self.device = device
        self.model.to(self.device)

    def _init_model(self) -> SamModel:
        """Initializes the SAM model for object detection.

        Returns:
            SamModel: The initialized SAM model.
        """
        logger.info(f"Initializing SlimSAM {self.size} model...")
        if self.size == "large":
            return SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50")
        return SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77")

    def _init_processor(self) -> SamProcessor:
        """Initializes the processor for the SAM model.

        Returns:
            SamProcessor: The initialized processor.
        """
        if self.size == "large":
            return SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
        return SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")

    def annotate_batch(
        self,
        images: List[PIL.Image.Image],
        boxes_batch: List[np.ndarray],
        conf_threshold: float = 0.2,
    ) -> List[List[List[float]]]:
        """Annotates images for the task of instance segmentation using the SlimSAM
        model.

        Args:
            images: The images to be annotated.
            boxes_batch: The bounding boxes of found objects.
            conf_threshold (float, optional): Confidence threshold for the annotations. Defaults to 0.2.

        Returns:
            List: A list containing the final segment masks represented as a polygon.
        """
        final_segments = []

        n = len(images)

        for i in range(n):
            boxes = boxes_batch[i].tolist()
            if len(boxes) == 0:
                final_segments.append([])
                continue

            inputs = self.processor(
                images[i], input_boxes=[boxes], return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]

            iou_scores = outputs.iou_scores.cpu()

            image_masks = []
            for j in range(len(boxes)):
                keep_idx = iou_scores[0, j] >= conf_threshold
                filtered_masks = masks[j, keep_idx].cpu().float()
                final_masks = filtered_masks.permute(1, 2, 0)
                final_masks = final_masks.mean(axis=-1)
                final_masks = (final_masks > 0).int()
                final_masks = final_masks.numpy().astype(np.uint8)
                polygon = mask_to_polygon(final_masks)
                if len(polygon) != 0:
                    image_masks.append(polygon)

            final_segments.append(image_masks)

        return final_segments

    def release(self, empty_cuda_cache: bool = False) -> None:
        """Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        self.model = self.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import requests
    from PIL import Image

    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = SlimSAMAnnotator(device="cpu", size="large")
    final_segments = annotator.annotate_batch([im], [np.array([[3, 229, 559, 650]])])
    print(len(final_segments), len(final_segments[0]))
    print(final_segments[0][0][:5])
