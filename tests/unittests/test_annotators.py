import pytest
import torch
from PIL import Image
import requests
from datadreamer.dataset_annotation.owlv2_annotator import OWLv2Annotator


def _check_owlv2_annotator(device: str):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = OWLv2Annotator(device=device)
    final_boxes, final_scores, final_labels = annotator.annotate(im, ["bus", "people"])
    # Assert that the boxes, scores and labels are tensors
    assert type(final_boxes) == torch.Tensor
    assert type(final_scores) == torch.Tensor
    assert type(final_labels) == torch.Tensor
    # Get the number of objects detected
    num_objects = final_boxes.shape[0]
    # Check that the boxes has correct shape
    assert final_boxes.shape == (num_objects, 4)
    # Check that the scores has correct shape
    assert final_scores.shape == (num_objects,)
    # Check that the labels has correct shape
    assert final_labels.shape == (num_objects,)
    # Check that the scores are not zero
    assert torch.all(final_scores > 0)
    # Check that the labels are bigger or equal to zero
    assert torch.all(final_labels >= 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_cuda_owlv2_annotator():
    _check_owlv2_annotator("cuda")


def test_cou_owlv2_annotator():
    _check_owlv2_annotator("cpu")
