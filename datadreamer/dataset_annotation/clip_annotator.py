from __future__ import annotations

import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from datadreamer.dataset_annotation.cls_annotator import ImgClassificationAnnotator


class CLIPAnnotator(ImgClassificationAnnotator):
    """A class for image annotation using the CLIP model, specializing in image
    classification.

    Attributes:
        model (CLIPModel): The CLIP model for image-text similarity evaluation.
        processor (CLIPProcessor): The processor for preparing inputs to the CLIP model.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).
        size (str): The size of the CLIP model to use ('base' or 'large').

    Methods:
        _init_processor(): Initializes the CLIP processor.
        _init_model(): Initializes the CLIP model.
        annotate_batch(image, prompts, conf_threshold, use_tta, synonym_dict): Annotates the given image with bounding boxes and labels.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def _init_processor(self) -> CLIPProcessor:
        """Initializes the CLIP processor.

        Returns:
            CLIPProcessor: The initialized CLIP processor.
        """
        if self.size == "large":
            return CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _init_model(self) -> CLIPModel:
        """Initializes the CLIP model.

        Returns:
            CLIPModel: The initialized CLIP model.
        """
        logger.info(f"Initializing CLIP {self.size} model...")
        if self.size == "large":
            return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


if __name__ == "__main__":
    import requests

    device = "cuda" if torch.cuda.is_available() else "cpu"
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = CLIPAnnotator(device=device)
    labels = annotator.annotate_batch([im], ["bus", "people"])
    print(labels)
    annotator.release()
