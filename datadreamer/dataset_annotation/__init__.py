from __future__ import annotations

from .clip_annotator import CLIPAnnotator
from .fastsam_annotator import FastSAMAnnotator
from .image_annotator import BaseAnnotator, TaskList
from .owlv2_annotator import OWLv2Annotator

__all__ = [
    "BaseAnnotator",
    "TaskList",
    "OWLv2Annotator",
    "CLIPAnnotator",
    "FastSAMAnnotator",
]
