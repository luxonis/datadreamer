from __future__ import annotations

from typing import List

import numpy as np
import PIL
import torch
from loguru import logger
from sam2.sam2_image_predictor import SAM2ImagePredictor

from datadreamer.dataset_annotation.image_annotator import BaseAnnotator
from datadreamer.dataset_annotation.utils import convert_binary_mask


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
        mask_format: str = "rle",
    ) -> None:
        """Initializes the SAM2Annotator.

        Args:
            seed (float): Seed for reproducibility. Defaults to 42.
            device (str): The device to run the model on. Defaults to 'cuda'.
            size (str): The size of the SAM2.1 model to use ('base' or 'large').
            mask_format (str): The format of the output masks ('rle' or 'polyline').
        """
        super().__init__(seed)
        self.size = size
        self.device = device
        self.model = self._init_model(device=device)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float16
        self.mask_format = mask_format

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
        conf_threshold: float = 0.2,
    ) -> List[List[List[float]]] | List[List[dict]]:
        """Annotates images for the task of instance segmentation using the SAM2.1
        model.

        Args:
            images: The images to be annotated.
            boxes_batch: The bounding boxes of found objects.
            conf_threshold (float, optional): Confidence threshold for the annotations. Defaults to 0.2.

        Returns:
            List: A list containing masks.
        """
        final_masks = []

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
                final_masks.append([])
                continue

            image_masks = []
            for j in range(len(boxes)):
                mask, score = masks_batch[i][j].astype(np.uint8), scores_batch[i][j]
                if score < conf_threshold or mask.sum() == 0:
                    image_masks.append([])
                    continue
                if len(mask.shape) == 3:
                    mask = mask.squeeze(0)

                converted_mask = convert_binary_mask(mask, self.mask_format)
                image_masks.append(converted_mask)

            final_masks.append(image_masks)

        return final_masks

    def release(self, empty_cuda_cache: bool = False) -> None:
        """Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        self.model.model = self.model.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    import requests
    from PIL import Image

    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = SAM2Annotator(device="cpu", size="large", mask_format="polyline")
    final_masks = annotator.annotate_batch([im], [np.array([[3, 229, 559, 650]])])
    print(len(final_masks), len(final_masks[0]))
    print(final_masks[0][0][:5])
