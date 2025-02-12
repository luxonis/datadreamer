from __future__ import annotations

import numpy as np
import psutil
import pytest
import requests
import torch
from PIL import Image

from datadreamer.dataset_annotation.aimv2_annotator import AIMv2Annotator
from datadreamer.dataset_annotation.clip_annotator import CLIPAnnotator
from datadreamer.dataset_annotation.image_annotator import BaseAnnotator
from datadreamer.dataset_annotation.owlv2_annotator import OWLv2Annotator
from datadreamer.dataset_annotation.sam2_annotator import SAM2Annotator
from datadreamer.dataset_annotation.slimsam_annotator import SlimSAMAnnotator

# Get the total disk space in GB
total_disk_space = psutil.disk_usage("/").total / (1024**3)


def _check_owlv2_annotator(
    device: str, size: str = "base", use_text_prompts: bool = True
):
    annotator = OWLv2Annotator(device=device, size=size)

    if use_text_prompts:
        url = "https://ultralytics.com/images/bus.jpg"
        im = Image.open(requests.get(url, stream=True).raw)
        final_boxes, final_scores, final_labels = annotator.annotate_batch(
            [im], ["bus", "people"]
        )
    else:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        im = Image.open(requests.get(url, stream=True).raw)
        query_url = "http://images.cocodataset.org/val2017/000000058111.jpg"
        query_image = Image.open(requests.get(query_url, stream=True).raw)
        annotator = OWLv2Annotator(device=device, size=size)
        final_boxes, final_scores, final_labels = annotator.annotate_batch(
            [im], [query_image], conf_threshold=0.9
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
def test_cuda_owlv2_annotator_text():
    _check_owlv2_annotator("cuda", use_text_prompts=True)


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_owlv2_annotator_text():
    _check_owlv2_annotator("cpu", use_text_prompts=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_owlv2_annotator_image():
    _check_owlv2_annotator("cuda", use_text_prompts=False)


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_owlv2_annotator_image():
    _check_owlv2_annotator("cpu", use_text_prompts=False)


def _check_aimv2_annotator(device: str):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = AIMv2Annotator(device=device)
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
def test_cuda_aimv2_annotator():
    _check_aimv2_annotator("cuda")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_aimv2_annotator():
    _check_aimv2_annotator("cpu")


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
    annotator = SlimSAMAnnotator(device=device, size=size, mask_format="polyline")
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


def _check_annotator_rle(
    annotator_class: BaseAnnotator, device: str, size: str = "base"
):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = annotator_class(device=device, size=size, mask_format="rle")
    masks = annotator.annotate_batch([im], [np.array([[3, 229, 559, 650]])])

    # Check that the masks are lists
    assert isinstance(masks, list) and len(masks) == 1
    assert isinstance(masks[0], list) and len(masks[0]) == 1
    assert isinstance(masks[0][0], dict), "RLE masks should be stored as a dictionary"

    # Check expected RLE keys
    rle_mask = masks[0][0]
    assert (
        "counts" in rle_mask and "size" in rle_mask
    ), "RLE mask should contain 'counts' and 'size' keys"

    # Validate RLE mask content
    assert isinstance(rle_mask["counts"], str), "RLE 'counts' should be a string"
    assert (
        isinstance(rle_mask["size"], list) and len(rle_mask["size"]) == 2
    ), "RLE 'size' should be a list of length 2"

    annotator.release(empty_cuda_cache=True if device != "cpu" else False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_slimsam_base_annotator():
    _check_slimsam_annotator("cuda")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_slimsam_base_annotator():
    _check_slimsam_annotator("cpu")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_slimsam_base_annotator_rle():
    _check_annotator_rle(SlimSAMAnnotator, "cpu")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_sam2_base_annotator_rle():
    _check_annotator_rle(SAM2Annotator, "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_slimsam_large_annotator():
    _check_slimsam_annotator("cuda", size="large")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_slimsam_large_annotator():
    _check_slimsam_annotator("cpu", size="large")


def _check_sam2_annotator(device: str, size: str = "base"):
    url = "https://ultralytics.com/images/bus.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    annotator = SAM2Annotator(device=device, size=size, mask_format="polyline")
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
def test_cuda_sam2_base_annotator():
    _check_sam2_annotator("cuda")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_sam2_base_annotator():
    _check_sam2_annotator("cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or total_disk_space < 16,
    reason="Test requires GPU and 16GB of HDD",
)
def test_cuda_sam2_large_annotator():
    _check_sam2_annotator("cuda", size="large")


@pytest.mark.skipif(
    total_disk_space < 16,
    reason="Test requires at least 16GB of HDD",
)
def test_cpu_sam2_large_annotator():
    _check_sam2_annotator("cpu", size="large")
