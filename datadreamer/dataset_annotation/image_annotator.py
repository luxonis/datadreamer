from __future__ import annotations

import enum
from abc import ABC, abstractmethod


class TaskList(enum.Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"


class BaseAnnotator(ABC):
    """Abstract base class for creating annotators.

    Attributes:
        seed (float): A seed value to ensure reproducibility in annotation processes.
        task_definition (TaskList): An enumeration of the task type. Default is OBJECT_DETECTION,
                                    which can be overridden by subclasses for specific tasks.

    Methods:
        annotate_batch(): Abstract method to be implemented by subclasses. It should contain
                    the logic for performing annotation based on the task definition.
        release(): Abstract method to be implemented by subclasses. It should contain
                    the logic for releasing the resources used by the annotator.
    """

    def __init__(
        self, seed: float, task_definition: TaskList = TaskList.OBJECT_DETECTION
    ) -> None:
        self.seed = seed
        self.task_definition = task_definition

    @abstractmethod
    def annotate_batch(self):
        pass

    @abstractmethod
    def release(self, empty_cuda_cache=False) -> None:
        pass
