from io import BytesIO
from time import sleep

import pytest
import requests
from PIL import Image


@pytest.fixture(scope="session")
def _bus_image_bytes():
    with requests.Session() as session:
        # Retry the complete download, including failures after response headers.
        for attempt in range(4):
            try:
                with session.get(
                    "https://ultralytics.com/images/bus.jpg", timeout=30, stream=True
                ) as response:
                    response.raise_for_status()
                    return response.content
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ):
                if attempt == 3:
                    raise
            except requests.HTTPError as error:
                if attempt == 3 or error.response.status_code not in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):
                    raise
            sleep(2**attempt)


@pytest.fixture
def bus_image(_bus_image_bytes):
    """Give each test its own image from one shared download."""
    with Image.open(BytesIO(_bus_image_bytes)) as image:
        return image.copy()
