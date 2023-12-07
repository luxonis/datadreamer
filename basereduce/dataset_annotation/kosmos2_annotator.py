import torch
import torchvision.ops as ops
import numpy as np

from basereduce.dataset_annotation.image_annotator import BaseAnnotator
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
from basereduce.dataset_annotation.utils import apply_tta


class Kosmos2Annotator(BaseAnnotator):
    """
    An image annotator class that utilizes the Kosmos2 model for conditional image generation.

    Attributes:
        model (Kosmos2ForConditionalGeneration): The Kosmos2 model for conditional image generation.
        processor (AutoProcessor): The processor for the Kosmos2 model.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).

    Methods:
        _init_model(): Initializes the Kosmos2 model.
        _init_processor(): Initializes the processor for the Kosmos2 model.
        annotate(image, prompts, conf_threshold, use_tta): Annotates the given image with labels and bounding boxes.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        seed: float,
        device: str = "cuda",
    ) -> None:
        """
        Initializes the Kosmos2Annotator with a given seed and device.

        Args:
            seed (float): Seed for reproducibility.
            device (str): The device to run the model on. Defaults to 'cuda'.
        """
        super().__init__(seed)
        self.model = self._init_model()
        self.processor = self._init_processor()
        self.device = device
        self.model.to(self.device)

    def _init_model(self):
        """
        Initializes the Kosmos2 model.

        Returns:
            Kosmos2ForConditionalGeneration: The initialized Kosmos2 model.
        """
        return Kosmos2ForConditionalGeneration.from_pretrained(
            "microsoft/kosmos-2-patch14-224"
        )

    def _init_processor(self):
        """
        Initializes the processor for the Kosmos2 model.

        Returns:
            AutoProcessor: The initialized processor.
        """
        return AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    def annotate(self, image, prompts, conf_threshold=0.1, use_tta=False):

        """
        Annotates an image using the Kosmos2 model.

        Args:
            image: The image to be annotated.
            prompts: Prompts to guide the annotation.
            conf_threshold (float, optional): Confidence threshold for the annotations. Defaults to 0.1.
            use_tta (bool, optional): Flag to apply test-time augmentation. Defaults to False.

        Returns:
            tuple: A tuple containing the final bounding boxes, scores, and labels for the annotations.
        """
        image_size = image.size[::-1]

        prompt = "<grounding> An image of"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device
        )

        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=64,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        processed_text = self.processor.post_process_generation(
            generated_text, cleanup_and_extract=False
        )

        caption, entities = self.processor.post_process_generation(generated_text)

        print(image_size)

        final_boxes = []
        final_labels = []

        for label, _, boxes in entities:
            for box in boxes:
                # Scale box coordinates to image size
                scaled_box = [
                    int(coord * image_size[i % 2]) for i, coord in enumerate(box)
                ]
                final_boxes.append(scaled_box)
                final_labels.append(label)

        final_scores = [1.0] * len(final_boxes)

        final_boxes = np.array(final_boxes)
        final_scores = np.array(final_scores)
        final_labels = np.array(final_labels)

        return final_boxes, final_scores, final_labels

    def release(self, empty_cuda_cache=False) -> None:
        """
        Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        self.model = self.model.to("cpu")
        with torch.no_grad():
            torch.cuda.empty_cache()
