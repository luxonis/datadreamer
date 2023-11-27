import torch

from basereduce.dataset_annotation.image_annotator import BaseAnnotator
from basereduce.dataset_annotation.image_annotator import ModelName, TaskList
from transformers import Owlv2Processor, Owlv2ForObjectDetection


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

    def annotate(self, image, prompts, conf_threshold=0.1):
        inputs = self.processor(text=prompts, images=image, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        preds = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
        )
        boxes, scores, labels = (
            preds[0]["boxes"],
            preds[0]["scores"],
            preds[0]["labels"],
        )

        return boxes, scores, labels

    def release(self, empty_cuda_cache=False) -> None:
        self.model = self.model.to('cpu')
        with torch.no_grad():
            torch.cuda.empty_cache()
