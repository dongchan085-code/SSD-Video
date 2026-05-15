import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model downloads or GPU (deselect with -m 'not slow')",
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA-capable GPU",
    )
