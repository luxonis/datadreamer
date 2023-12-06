import torch
import torchvision.ops as ops

from basereduce.dataset_annotation.image_annotator import BaseAnnotator
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from basereduce.dataset_annotation.utils import apply_tta

from basereduce.utils.nms import non_max_suppression

class OWLv2Annotator(BaseAnnotator):
    def __init__(
        self,
        seed: float = 42,
        device: str = "cuda",
    ) -> None:
        super().__init__(seed)
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

    def annotate(self, image, prompts, conf_threshold=0.1, use_tta=False, synonym_dict=None):
        if use_tta:
            augmented_images = apply_tta(image)
        else:
            augmented_images = [image]

        if synonym_dict is not None:
            prompts_syn = []
            for prompt in prompts:
                prompts_syn.append(prompt)
                prompts_syn.extend(synonym_dict[prompt])
            # Make a dict to transform synonym ids to original ids
            synonym_dict_rev = {}
            for key, value in synonym_dict.items():
                if key in prompts:
                    synonym_dict_rev[prompts_syn.index(key)] = prompts.index(key)
                    for v in value:
                        synonym_dict_rev[prompts_syn.index(v)] = prompts.index(key)
            prompts = prompts_syn
        

        all_boxes = []
        all_scores = []
        all_labels = []

        target_sizes = torch.Tensor([augmented_images[0].size[::-1]]).to(self.device)

        for aug_image in augmented_images:
            inputs = self.processor(
                text=prompts, images=aug_image, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # print(outputs)
            preds = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
            )

            boxes, scores, labels = (
                preds[0]["boxes"],
                preds[0]["scores"],
                preds[0]["labels"],
            )
            # Flip boxes back if using TTA
            if use_tta and len(all_boxes) == 1:
                boxes[:, [0, 2]] = image.size[0] - boxes[:, [2, 0]]

            if synonym_dict is not None:
                labels = torch.tensor([synonym_dict_rev[label.item()] for label in labels])

            all_boxes.append(boxes.to("cpu"))
            all_scores.append(scores.to("cpu"))
            all_labels.append(labels.to("cpu"))

        # Convert list of tensors to a single tensor for NMS
        all_boxes_cat = torch.cat(all_boxes)
        all_scores_cat = torch.cat(all_scores)
        all_labels_cat = torch.cat(all_labels)

        one_hot_labels = torch.nn.functional.one_hot(all_labels_cat, num_classes=len(prompts))

        # Apply NMS
        # transform predictions to shape [N, 5 + num_classes], N is the number of bboxes for nms function
        all_boxes_cat = torch.cat(
            (
                all_boxes_cat,
                all_scores_cat.unsqueeze(-1),
                one_hot_labels
            ),
            dim=1,
        )

        # output is  a list of detections, each item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
        output = non_max_suppression(all_boxes_cat.unsqueeze(0), conf_thres=conf_threshold, iou_thres=0.2)

        final_boxes = output[0][:, :4]
        final_scores = output[0][:, 4]
        final_labels = output[0][:, 5].long()

        return final_boxes, final_scores, final_labels

    def release(self, empty_cuda_cache=False) -> None:
        self.model = self.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()
