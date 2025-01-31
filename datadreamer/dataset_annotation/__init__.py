from __future__ import annotations

from .aimv2_annotator import AIMv2Annotator
from .clip_annotator import CLIPAnnotator
from .cls_annotator import ImgClassificationAnnotator
from .image_annotator import BaseAnnotator, TaskList
from .owlv2_annotator import OWLv2Annotator
from .sam2_annotator import SAM2Annotator
from .slimsam_annotator import SlimSAMAnnotator

__all__ = [
    "AIMv2Annotator",
    "BaseAnnotator",
    "TaskList",
    "OWLv2Annotator",
    "ImgClassificationAnnotator",
    "CLIPAnnotator",
    "SAM2Annotator",
    "SlimSAMAnnotator",
]
