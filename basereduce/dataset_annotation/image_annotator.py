from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
import enum


# Enum for different labeling tasks
class TaskList(enum.Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    # Add more tasks as needed


# Abstract base class for data labeling
class BaseAnnotator(ABC):
    def __init__(
        self, seed: float, task_definition: TaskList = TaskList.OBJECT_DETECTION
    ) -> None:
        self.seed = seed
        self.task_definition = task_definition

    @abstractmethod
    def annotate(self):
        pass
