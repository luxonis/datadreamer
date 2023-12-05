import torch
import torchvision.ops as ops
import numpy as np

from basereduce.dataset_annotation.image_annotator import BaseAnnotator
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
from basereduce.dataset_annotation.utils import apply_tta


class Kosmos2Annotator(BaseAnnotator):
    def __init__(
        self,
        seed: float,
        device: str = "cuda",
    ) -> None:
        super().__init__(seed)
        self.model = self._init_model()
        self.processor = self._init_processor()
        self.device = device
        self.model.to(self.device)

    def _init_model(self):
        return Kosmos2ForConditionalGeneration.from_pretrained(
            "microsoft/kosmos-2-patch14-224"
        )

    def _init_processor(self):
        return AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    def annotate(self, image, prompts, conf_threshold=0.1, use_tta=False):
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
        self.model = self.model.to("cpu")
        with torch.no_grad():
            torch.cuda.empty_cache()
