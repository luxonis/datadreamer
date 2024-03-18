from __future__ import annotations

from typing import List

import numpy as np
import PIL
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from datadreamer.dataset_annotation.image_annotator import BaseAnnotator, TaskList


class CLIPAnnotator(BaseAnnotator):
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

    def __init__(
        self,
        seed: float = 42,
        device: str = "cuda",
        size: str = "base",
    ) -> None:
        """Initializes the CLIPAnnotator with a specific seed and device.

        Args:
            seed (float): Seed for reproducibility. Defaults to 42.
            device (str): The device to run the model on. Defaults to 'cuda'.
        """
        super().__init__(seed, task_definition=TaskList.CLASSIFICATION)
        self.size = size
        self.model = self._init_model()
        self.processor = self._init_processor()
        self.device = device
        self.model.to(self.device)

    def _init_processor(self):
        """Initializes the CLIP processor.

        Returns:
            CLIPProcessor: The initialized CLIP processor.
        """
        if self.size == "large":
            return CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        return CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _init_model(self):
        """Initializes the CLIP model.

        Returns:
            CLIPModel: The initialized CLIP model.
        """
        if self.size == "large":
            return CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        return CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def annotate_batch(
        self,
        images: List[PIL.Image.Image],
        objects: List[str],
        conf_threshold: float = 0.1,
        synonym_dict: dict[str, List[str]] | None = None,
    ) -> List[np.ndarray]:
        """Annotates images using the OWLv2 model.

        Args:
            images: The images to be annotated.
            objects: A list of objects (text) to test against the images.
            conf_threshold (float, optional): Confidence threshold for the annotations. Defaults to 0.1.
            synonym_dict (dict, optional): Dictionary for handling synonyms in labels. Defaults to None.

        Returns:
            List[List[int]]: A list of lists of labels for each image.
        """
        if synonym_dict is not None:
            objs_syn = set()
            for obj in objects:
                objs_syn.add(obj)
                for syn in synonym_dict[obj]:
                    objs_syn.add(syn)
            objs_syn = list(objs_syn)
            # Make a dict to transform synonym ids to original ids
            synonym_dict_rev = {}
            for key, value in synonym_dict.items():
                if key in objects:
                    synonym_dict_rev[objs_syn.index(key)] = objects.index(key)
                    for v in value:
                        synonym_dict_rev[objs_syn.index(v)] = objects.index(key)
            objects = objs_syn

        inputs = self.processor(
            text=objects, images=images, return_tensors="pt", padding=True
        ).to(self.device)

        outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image  # image-text similarity score
        probs = logits_per_image.softmax(dim=1).cpu()  # label probabilities

        labels = []
        # Get the labels for each image
        if synonym_dict is not None:
            for prob in probs:
                labels.append(
                    np.unique(
                        np.array(
                            [
                                synonym_dict_rev[label.item()]
                                for label in torch.where(prob > conf_threshold)[
                                    0
                                ].numpy()
                            ]
                        )
                    )
                )
        else:
            for prob in probs:
                labels.append(torch.where(prob > conf_threshold)[0].numpy())

        return labels

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = CLIPAnnotator(device=device)
    labels = annotator.annotate_batch([im], ["bus", "people"])
    print(labels)
    annotator.release()
