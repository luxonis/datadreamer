from io import BytesIO

import pytest
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@pytest.fixture(scope="session")
def _bus_image_bytes():
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        with session.get(
            "https://ultralytics.com/images/bus.jpg", timeout=30
        ) as response:
            response.raise_for_status()
            return response.content


@pytest.fixture
def bus_image(_bus_image_bytes):
    """Give each test its own image from one shared download."""
    with Image.open(BytesIO(_bus_image_bytes)) as image:
        return image.copy()
