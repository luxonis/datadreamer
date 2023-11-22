from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
import enum


# Enum for generative model names, assuming this is used to determine the labeling model
class ModelName(enum.Enum):
    OWL_V2 = "google/owlv2-base-patch16-ensemble"


# Enum for different labeling tasks
class TaskList(enum.Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    # Add more tasks as needed


# Abstract base class for data labeling
class BaseAnnotator(ABC):
    def __init__(
        self, seed: float, model_name: ModelName, task_definition: TaskList
    ) -> None:
        self.seed = seed
        self.model_name = model_name
        self.task_definition = task_definition

    @abstractmethod
    def annotate(self):
        pass
