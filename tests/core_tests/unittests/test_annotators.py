from __future__ import annotations

import numpy as np
import psutil
import pytest
import requests
import torch
from PIL import Image

from datadreamer.dataset_annotation.clip_annotator import CLIPAnnotator
from datadreamer.dataset_annotation.fastsam_annotator import SlimSAMAnnotator
from datadreamer.dataset_annotation.owlv2_annotator import OWLv2Annotator

# Get the total disk space in GB
total_disk_space = psutil.disk_usage("/").total / (1024**3)


def _check_owlv2_annotator(device: str, size: str = "base"):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = OWLv2Annotator(device=device, size=size)
    final_boxes, final_scores, final_labels = annotator.annotate_batch(
        [im], ["bus", "people"]
    )
    # Assert that the boxes, scores and labels are tensors
    assert isinstance(final_boxes, list) and len(final_boxes) == 1
    assert isinstance(final_scores, list) and len(final_scores) == 1
    assert isinstance(final_labels, list) and len(final_labels) == 1
    # Get the number of objects detected
    num_objects = final_boxes[0].shape[0]
    # Check that the boxes has correct shape
    assert final_boxes[0].shape == (num_objects, 4)
    # Check that the scores has correct shape
    assert final_scores[0].shape == (num_objects,)
    # Check that the labels has correct shape
    assert final_labels[0].shape == (num_objects,)
    # Check that the scores are not zero
    assert np.all(final_scores[0] > 0)
    # Check that the labels are bigger or equal to zero
    assert np.all(final_labels[0] >= 0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_owlv2_annotator():
    _check_owlv2_annotator("cuda")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_owlv2_annotator():
    _check_owlv2_annotator("cpu")


def _check_clip_annotator(device: str, size: str = "base"):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = CLIPAnnotator(device=device, size=size)
    labels = annotator.annotate_batch([im], ["bus", "people"])
    # Check that the labels are lists
    assert isinstance(labels, list) and len(labels) == 1
    # Check that the labels are ndarray of integers
    assert isinstance(labels[0], np.ndarray) and labels[0].dtype == np.int64

    annotator.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_clip_base_annotator():
    _check_clip_annotator("cuda")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_clip_base_annotator():
    _check_clip_annotator("cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_clip_large_annotator():
    _check_clip_annotator("cuda", size="large")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_clip_large_annotator():
    _check_clip_annotator("cpu", size="large")


def _check_slimsam_annotator(device: str, size: str = "base"):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = SlimSAMAnnotator(device=device, size=size)
    masks = annotator.annotate_batch([im], [np.array([[3, 229, 559, 650]])])
    w, h = im.width, im.height
    # Check that the masks are lists
    assert isinstance(masks, list) and len(masks) == 1
    # Check that the masks are [B, O, N, 2], where
    # - B = batch size
    # - O = number of objects
    # - N = number of points of the mask segment polygon (at least 3 to be polygon)
    assert isinstance(masks[0], list) and len(masks[0]) == 1
    assert isinstance(masks[0][0], list) and len(masks[0][0]) >= 3
    for point in masks[0][0]:
        # Check that it is a 2D point
        assert len(point) == 2
        assert 0 <= point[0] <= w and 0 <= point[1] <= h

    annotator.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_fastsam_base_annotator():
    _check_slimsam_annotator("cuda")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_fastsam_base_annotator():
    _check_slimsam_annotator("cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_fastsam_large_annotator():
    _check_slimsam_annotator("cuda", size="large")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_fastsam_large_annotator():
    _check_slimsam_annotator("cpu", size="large")
