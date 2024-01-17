import enum
from abc import ABC, abstractmethod


# Enum for different labeling tasks
class TaskList(enum.Enum):
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    # Add more tasks as needed


# Abstract base class for data labeling
class BaseAnnotator(ABC):
    """Abstract base class for creating annotators.

    Attributes:
        seed (float): A seed value to ensure reproducibility in annotation processes.
        task_definition (TaskList): An enumeration of the task type. Default is OBJECT_DETECTION,
                                    which can be overridden by subclasses for specific tasks.

    Methods:
        annotate(): Abstract method to be implemented by subclasses. It should contain
                    the logic for performing annotation based on the task definition.
    """

    def __init__(
        self, seed: float, task_definition: TaskList = TaskList.OBJECT_DETECTION
    ) -> None:
        self.seed = seed
        self.task_definition = task_definition

    @abstractmethod
    def annotate(self):
        pass
