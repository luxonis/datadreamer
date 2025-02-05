"""This file uses pre-trained model derived from Apple's software, provided under the
Apple Sample Code License license. The license is available at:

https://developer.apple.com/support/downloads/terms/apple-sample-code/Apple-Sample-Code-License.pdf

In addition, this file and other parts of the repository are licensed under the Apache 2.0
License. By using this file, you agree to comply with the terms of both licenses.
"""
from __future__ import annotations

import torch
from loguru import logger
from PIL import Image
from transformers import AutoModel, AutoProcessor

from datadreamer.dataset_annotation.cls_annotator import ImgClassificationAnnotator


class AIMv2Annotator(ImgClassificationAnnotator):
    """A class for image annotation using the AIMv2 model, specializing in image
    classification.

    Attributes:
        model (AutoModel): The AIMv2 model for image-text similarity evaluation.
        processor (AutoProcessor): The processor for preparing inputs to the AIMv2 model.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).
        size (str): The size of the AIMv2 model to use ('base' or 'large').

    Methods:
        _init_processor(): Initializes the AIMv2 processor.
        _init_model(): Initializes the AIMv2 model.
        annotate_batch(image, prompts, conf_threshold, use_tta, synonym_dict): Annotates the given image with bounding boxes and labels.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def _init_processor(self) -> AutoProcessor:
        """Initializes the AIMv2 processor.

        Returns:
            AutoProcessor: The initialized AIMv2 processor.
        """
        return AutoProcessor.from_pretrained("apple/aimv2-large-patch14-224-lit")

    def _init_model(self) -> AutoModel:
        """Initializes the AIMv2 model.

        Returns:
            AutoModel: The initialized AIMv2 model.
        """
        logger.info(f"Initializing AIMv2 {self.size} model...")
        return AutoModel.from_pretrained(
            "apple/aimv2-large-patch14-224-lit", trust_remote_code=True
        )


if __name__ == "__main__":
    import requests

    device = "cuda" if torch.cuda.is_available() else "cpu"
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = AIMv2Annotator(device=device)
    labels = annotator.annotate_batch([im], ["bus", "people"])
    print(labels)
    annotator.release()
