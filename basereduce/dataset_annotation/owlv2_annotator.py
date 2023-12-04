import torch
import torchvision.ops as ops

from basereduce.dataset_annotation.image_annotator import BaseAnnotator
from basereduce.dataset_annotation.image_annotator import ModelName, TaskList
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from basereduce.dataset_annotation.utils import apply_tta


class OWLv2Annotator(BaseAnnotator):
    def __init__(
        self,
        seed: float,
        model_name: ModelName,
        task_definition: TaskList,
        device: str = "cuda",
    ) -> None:
        super().__init__(seed, model_name, task_definition)
        self.model = self._init_model()
        self.processor = self._init_processor()
        self.device = device
        self.model.to(self.device)

    def _init_model(self):
        return Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )

    def _init_processor(self):
        return Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble", do_pad=False
        )

    def annotate(self, image, prompts, conf_threshold=0.1, use_tta=False):

        if use_tta:
            augmented_images = apply_tta(image)
        else:
            augmented_images = [image]

        all_boxes = []
        all_scores = []
        all_labels = []

        target_sizes = torch.Tensor([augmented_images[0].size[::-1]]).to(self.device)

        for aug_image in augmented_images:
            inputs = self.processor(text=prompts, images=aug_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            #print(outputs)
            preds = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold)
            
            boxes, scores, labels = preds[0]["boxes"], preds[0]["scores"], preds[0]["labels"]
            # Flip boxes back if using TTA
            if use_tta and len(all_boxes) == 1:
                boxes[:, [0, 2]] = image.size[0] - boxes[:, [2, 0]]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # Convert list of tensors to a single tensor for NMS
        all_boxes_cat = torch.cat(all_boxes)
        all_scores_cat = torch.cat(all_scores)
        all_labels_cat = torch.cat(all_labels)

        # Apply NMS
        keep = ops.nms(all_boxes_cat, all_scores_cat, iou_threshold=0.5)

        # Select the boxes, scores, and labels that were kept by NMS
        final_boxes = all_boxes_cat[keep]
        final_scores = all_scores_cat[keep]
        final_labels = all_labels_cat[keep]

        return final_boxes, final_scores, final_labels

    def release(self, empty_cuda_cache=False) -> None:
        self.model = self.model.to('cpu')
        with torch.no_grad():
            torch.cuda.empty_cache()
