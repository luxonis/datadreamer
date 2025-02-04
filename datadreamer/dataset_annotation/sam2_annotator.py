from __future__ import annotations

import logging
from typing import List

import numpy as np
import PIL
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

from datadreamer.dataset_annotation.image_annotator import BaseAnnotator
from datadreamer.dataset_annotation.utils import mask_to_polygon

logger = logging.getLogger(__name__)


class SAM2Annotator(BaseAnnotator):
    """A class for image annotation using the SAM2.1 model, specializing in instance
    segmentation.

    Attributes:
        model (SAM2ImagePredictor): The SAM2.1 model for instance segmentation.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).
        size (str): The size of the SAM model to use ('base' or 'large').

    Methods:
        _init_model(): Initializes the SAM2.1 model.
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
        self.device = device
        self.model = self._init_model(device=device)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float16

    def _init_model(self, device: str) -> SAM2ImagePredictor:
        """Initializes the SAM2.1 model for object detection.

        Returns:
            SAM2ImagePredictor: The initialized SAM2.1 model.
        """
        logger.info(f"Initializing SAM2.1 {self.size} model...")
        if self.size == "large":
            return SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-large", device=device
            )
        return SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-base-plus", device=device
        )

    def annotate_batch(
        self,
        images: List[PIL.Image.Image],
        boxes_batch: List[np.ndarray],
        iou_threshold: float = 0.2,
    ) -> List[List[List[float]]]:
        """Annotates images for the task of instance segmentation using the SAM2.1
        model.

        Args:
            images: The images to be annotated.
            boxes_batch: The bounding boxes of found objects.
            iou_threshold (float, optional): Intersection over union threshold for non-maximum suppression. Defaults to 0.2.

        Returns:
            List: A list containing the final segment masks represented as a polygon.
        """
        final_segments = []

        image_batch = [np.array(img.convert("RGB")) for img in images]
        bboxes_batch = [None if len(boxes) == 0 else boxes for boxes in boxes_batch]

        with torch.inference_mode(), torch.autocast(self.device, dtype=self.dtype):
            self.model.set_image_batch(image_batch)
            masks_batch, scores_batch, _ = self.model.predict_batch(
                box_batch=bboxes_batch,
                multimask_output=False,
            )

        n = len(images)

        for i in range(n):
            boxes = boxes_batch[i].tolist()
            if boxes is None:
                final_segments.append([])
                continue

            image_masks = []
            for j in range(len(boxes)):
                mask, score = masks_batch[i][j].astype(np.uint8), scores_batch[i][j]
                if score < iou_threshold:
                    image_masks.append([])
                    continue
                if len(mask.shape) == 3:
                    mask = mask.squeeze(0)
                polygon = mask_to_polygon(mask)
                image_masks.append(polygon if len(polygon) != 0 else [])

            final_segments.append(image_masks)

        return final_segments

    def release(self, empty_cuda_cache: bool = False) -> None:
        """Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import requests
    from PIL import Image

    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = SAM2Annotator(device="cpu", size="large")
    final_segments = annotator.annotate_batch([im], [np.array([[3, 229, 559, 650]])])
    print(len(final_segments), len(final_segments[0]))
    print(final_segments[0][0][:5])
