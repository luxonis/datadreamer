from __future__ import annotations

from typing import List

import numpy as np
import PIL
import torch
from loguru import logger
from transformers import SamModel, SamProcessor

from datadreamer.dataset_annotation.image_annotator import BaseAnnotator
from datadreamer.dataset_annotation.utils import convert_binary_mask


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
        mask_format: str = "rle",
    ) -> None:
        """Initializes the SlimSAMAnnotator.

        Args:
            seed (float): Seed for reproducibility. Defaults to 42.
            device (str): The device to run the model on. Defaults to 'cuda'.
            size (str): The size of the SAM model to use ('base' or 'large').
            mask_format (str): The format of the output masks ('rle' or 'polyline').
        """
        super().__init__(seed)
        self.size = size
        self.model = self._init_model()
        self.processor = self._init_processor()
        self.device = device
        self.model.to(self.device)
        self.mask_format = mask_format

    def _init_model(self) -> SamModel:
        """Initializes the SAM model for object detection.

        Returns:
            SamModel: The initialized SAM model.
        """
        logger.info(f"Initializing SlimSAM {self.size} model...")
        model_name = (
            "Zigeng/SlimSAM-uniform-50"
            if self.size == "large"
            else "Zigeng/SlimSAM-uniform-77"
        )
        return SamModel.from_pretrained(model_name)

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
        final_masks = []

        n = len(images)

        for i in range(n):
            boxes = boxes_batch[i].tolist()
            if len(boxes) == 0:
                final_masks.append([])
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
                filtered_mask = masks[j, keep_idx].cpu().float()

                if filtered_mask.shape[0] == 0 or filtered_mask.sum() == 0:
                    image_masks.append([])
                    continue

                mask = filtered_mask.permute(1, 2, 0)
                mask = mask.mean(axis=-1)
                mask = (mask > 0).int().numpy().astype(np.uint8)

                converted_mask = convert_binary_mask(mask, self.mask_format)
                image_masks.append(converted_mask)

            final_masks.append(image_masks)

        return final_masks

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
    annotator = SlimSAMAnnotator(device="cpu", size="large", mask_format="polyline")
    final_masks = annotator.annotate_batch([im], [np.array([[3, 229, 559, 650]])])
    print(len(final_masks), len(final_masks[0]))
    print(final_masks[0][0][:5])
