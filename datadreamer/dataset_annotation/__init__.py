from __future__ import annotations

from .clip_annotator import CLIPAnnotator
from .image_annotator import BaseAnnotator, TaskList
from .owlv2_annotator import OWLv2Annotator
from .slimsam_annotator import SlimSAMAnnotator

__all__ = [
    "BaseAnnotator",
    "TaskList",
    "OWLv2Annotator",
    "CLIPAnnotator",
    "SlimSAMAnnotator",
]
