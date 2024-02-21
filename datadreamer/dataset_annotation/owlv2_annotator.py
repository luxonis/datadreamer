import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from datadreamer.dataset_annotation.image_annotator import BaseAnnotator
from datadreamer.dataset_annotation.utils import apply_tta
from datadreamer.utils.nms import non_max_suppression


class OWLv2Annotator(BaseAnnotator):
    """A class for image annotation using the OWLv2 model, specializing in object
    detection.

    Attributes:
        model (Owlv2ForObjectDetection): The OWLv2 model for object detection.
        processor (Owlv2Processor): The processor for the OWLv2 model.
        device (str): The device on which the model will run ('cuda' for GPU, 'cpu' for CPU).

    Methods:
        _init_model(): Initializes the OWLv2 model.
        _init_processor(): Initializes the processor for the OWLv2 model.
        annotate_batch(image, prompts, conf_threshold, use_tta, synonym_dict): Annotates the given image with bounding boxes and labels.
        release(empty_cuda_cache): Releases resources and optionally empties the CUDA cache.
    """

    def __init__(
        self,
        seed: float = 42,
        device: str = "cuda",
    ) -> None:
        """Initializes the OWLv2Annotator with a specific seed and device.

        Args:
            seed (float): Seed for reproducibility. Defaults to 42.
            device (str): The device to run the model on. Defaults to 'cuda'.
        """
        super().__init__(seed)
        self.model = self._init_model()
        self.processor = self._init_processor()
        self.device = device
        self.model.to(self.device)

    def _init_model(self):
        """Initializes the OWLv2 model for object detection.

        Returns:
            Owlv2ForObjectDetection: The initialized OWLv2 model.
        """
        return Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )

    def _init_processor(self):
        """Initializes the processor for the OWLv2 model.

        Returns:
            Owlv2Processor: The initialized processor.
        """
        return Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble", do_pad=False
        )

    def _generate_annotations(self, images, prompts, conf_threshold=0.1):
        """"""
        n = len(images)
        batched_prompts = [prompts] * n
        target_sizes = torch.Tensor(images[0].size[::-1]).repeat((n, 1)).to(self.device)

        inputs = self.processor(
            text=batched_prompts, images=images, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # print(outputs)
        preds = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
        )

        return preds

    def _get_annotations(
        self, pred, use_tta: bool, img_dim: int, synonym_dict, synonym_dict_rev
    ):
        boxes, scores, labels = (
            pred["boxes"],
            pred["scores"],
            pred["labels"],
        )
        # Flip boxes back if using TTA
        if use_tta:
            boxes[:, [0, 2]] = img_dim - boxes[:, [2, 0]]

        if synonym_dict is not None:
            labels = torch.tensor([synonym_dict_rev[label.item()] for label in labels])

        return boxes, scores, labels

    def annotate_batch(
        self, images, prompts, conf_threshold=0.1, use_tta=False, synonym_dict=None
    ):
        """Annotates images using the OWLv2 model.

        Args:
            images: The images to be annotated.
            prompts: Prompts to guide the annotation.
            conf_threshold (float, optional): Confidence threshold for the annotations. Defaults to 0.1.
            use_tta (bool, optional): Flag to apply test-time augmentation. Defaults to False.
            synonym_dict (dict, optional): Dictionary for handling synonyms in labels. Defaults to None.

        Returns:
            tuple: A tuple containing the final bounding boxes, scores, and labels for the annotations.
        """
        if use_tta:
            augmented_images = [apply_tta(image)[0] for image in images]

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

        preds = self._generate_annotations(images, prompts, conf_threshold)
        if use_tta:
            augmented_preds = self._generate_annotations(
                augmented_images, prompts, conf_threshold
            )
        else:
            augmented_preds = [None] * len(images)

        final_boxes = []
        final_scores = []
        final_labels = []

        for i, (pred, aug_pred) in enumerate(zip(preds, augmented_preds)):
            boxes, scores, labels = self._get_annotations(
                pred,
                False,
                images[i].size[0],
                synonym_dict,
                synonym_dict_rev if synonym_dict is not None else None,
            )

            all_boxes = [boxes.to("cpu")]
            all_scores = [scores.to("cpu")]
            all_labels = [labels.to("cpu")]

            # Flip boxes back if using TTA
            if use_tta:
                aug_boxes, aug_scores, aug_labels = self._get_annotations(
                    aug_pred,
                    True,
                    images[i].size[0],
                    synonym_dict,
                    synonym_dict_rev if synonym_dict is not None else None,
                )

                all_boxes.append(aug_boxes.to("cpu"))
                all_scores.append(aug_scores.to("cpu"))
                all_labels.append(aug_labels.to("cpu"))

            one_hot_labels = torch.nn.functional.one_hot(
                torch.cat(all_labels), num_classes=len(prompts)
            )

            # Apply NMS
            # transform predictions to shape [N, 5 + num_classes], N is the number of bboxes for nms function
            all_boxes_cat = torch.cat(
                (
                    torch.cat(all_boxes),
                    torch.cat(all_scores).unsqueeze(-1),
                    one_hot_labels,
                ),
                dim=1,
            )

            # output is a list of detections, each item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
            output = non_max_suppression(
                all_boxes_cat.unsqueeze(0), conf_thres=conf_threshold, iou_thres=0.2
            )

            output_boxes = output[0][:, :4]
            output_scores = output[0][:, 4]
            output_local_labels = output[0][:, 5].long()

            final_boxes.append(
                output_boxes.detach().cpu().numpy()
                if not isinstance(output_boxes, np.ndarray)
                else output_boxes
            )
            final_scores.append(
                output_scores.detach().cpu().numpy()
                if not isinstance(output_scores, np.ndarray)
                else output_scores
            )
            final_labels.append(
                output_local_labels.detach().cpu().numpy()
                if not isinstance(output_local_labels, np.ndarray)
                else output_local_labels
            )

        return final_boxes, final_scores, final_labels

    def release(self, empty_cuda_cache=False) -> None:
        """Releases the model and optionally empties the CUDA cache.

        Args:
            empty_cuda_cache (bool, optional): Whether to empty the CUDA cache. Defaults to False.
        """
        self.model = self.model.to("cpu")
        if empty_cuda_cache:
            with torch.no_grad():
                torch.cuda.empty_cache()
