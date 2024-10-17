from __future__ import annotations

import logging
from typing import List, Literal, Tuple

import numpy as np
import PIL
from ultralytics import FastSAM

logger = logging.getLogger(__name__)


class FastSAMAnnotator:
    """A class for image annotation using the FastSAM model, specializing in instance
    segmentation.

    Attributes:
        model (FastSAM): The FastSAM model.


    Methods:
        annotate_batch(image, prompts, conf_threshold, use_tta, synonym_dict): Annotates the given image with bounding boxes and labels.
    """

    def __init__(
        self,
        device: str = "cuda",
        size: Literal["base", "large"] = "large",
    ) -> None:
        """Initializes the FastSAMAnnotator object.

        Args:
            size (str): The size of the FastSAM model to use ('s' or 'x').
        """
        self.size = size
        self.device = device
        self.model = self._init_model()

    def _init_model(self) -> FastSAM:
        """Initializes the FastSAM model for instance segmentation.

        Returns:
            FastSAM: The initialized FastSAM model.
        """
        model_size = "s" if self.size == "base" else "x"
        logger.info(f"Initializing FastSAM {model_size} model...")
        return FastSAM(f"FastSAM-{model_size}.pt")

    def annotate_batch(
        self,
        images: List[PIL.Image.Image],
        prompts: List[str],
        boxes_batch: List[np.ndarray],
        scores_batch: List[np.ndarray],
        labels_batch: List[np.ndarray],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.2,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Annotates images using the OWLv2 model.

        Args:
            images: The images to be annotated.
            prompts: Prompts to guide the annotation.
            conf_threshold (float, optional): Confidence threshold for the annotations. Defaults to 0.1.
            iou_threshold (float, optional): Intersection over union threshold for non-maximum suppression. Defaults to 0.2.
            use_tta (bool, optional): Flag to apply test-time augmentation. Defaults to False.
            synonym_dict (dict, optional): Dictionary for handling synonyms in labels. Defaults to None.

        Returns:
            tuple: A tuple containing the final bounding boxes, scores, and labels for the annotations.
        """
        final_segments = []

        n = len(images)

        for i in range(n):
            batch_segments = []
            for box, label in zip(boxes_batch[i], labels_batch[i]):
                result = self.model(
                    images[i],
                    device=self.device,
                    bboxes=box,
                    texts=prompts[label],
                    labels=[1],
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                mask_segment = result[0].masks.xy[0]
                print("mask", mask_segment.shape)
                batch_segments.append(mask_segment)
            final_segments.append(batch_segments)

        return boxes_batch, scores_batch, labels_batch, final_segments


if __name__ == "__main__":
    import requests
    from PIL import Image

    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = FastSAMAnnotator(device="cpu", size="base")
