from __future__ import annotations

import logging
from typing import List, Literal

import numpy as np
import PIL
from ultralytics import FastSAM

logger = logging.getLogger(__name__)


class FastSAMAnnotator:
    """A class for image annotation using the FastSAM model, specializing in instance
    segmentation.

    Attributes:
        model (FastSAM): The FastSAM model.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).
        size (str): The size of the FastSAM model to use ('s' or 'x').

    Methods:
        _init_model(): Initializes the FastSAM model.
        annotate_batch(images, boxes_batch, conf_threshold, iou_threshold): Annotates the given image with given bounding boxes.
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
        boxes_batch: List[np.ndarray],
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.2,
    ) -> List[List[List[float]]]:
        """Annotates images for the task of instance segmentation using the FastSAM
        model.

        Args:
            images: The images to be annotated.
            boxes_batch: The bounding boxes of found objects.
            conf_threshold (float, optional): Confidence threshold for the annotations. Defaults to 0.15.
            iou_threshold (float, optional): Intersection over union threshold for non-maximum suppression. Defaults to 0.2.

        Returns:
            List: A list containing the final segment masks represented as a polygon.
        """
        final_segments = []

        n = len(images)

        for i in range(n):
            result = self.model(
                images[i],
                device=self.device,
                bboxes=boxes_batch[i],
                labels=1,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            mask_segments = result[0].masks.xy
            final_segments.append(list(map(lambda x: x.tolist(), mask_segments)))

        return final_segments


if __name__ == "__main__":
    import requests
    from PIL import Image

    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = FastSAMAnnotator(device="cpu", size="large")
    final_segments = annotator.annotate_batch([im], [np.array([[3, 229, 559, 650]])])
    print(len(final_segments), len(final_segments[0]), len(final_segments[0][0]))
