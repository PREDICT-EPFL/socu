import os

import pytest
import warp as wp


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default=os.environ.get("SOCU_TEST_DEVICE"),
        help="Warp device to run the tests on (default: cuda if available, else cpu)",
    )


def pytest_configure(config):
    wp.init()
    device = config.getoption("--device")
    if device is None:
        device = "cuda" if wp.is_cuda_available() else "cpu"
    config.socu_device = wp.get_device(device)
    if config.socu_device.is_cpu:
        # must be set before JAX initializes its backends
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


@pytest.fixture(scope="session")
def device(request):
    return request.config.socu_device
