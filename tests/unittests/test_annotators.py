import pytest
import torch


def test_owlv2_annotator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pass


if __name__ == "__main__":
    pytest.main()
