import importlib
import subprocess
import warnings

import pytest


packages = [
    # these are problem libraries that don't always seem to import, mostly due
    # to dependencies outside the python world
    "flwr",
    "keras",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "tensorflow",
    "torch",
]


@pytest.mark.parametrize("package_name", packages, ids=packages)
def test_import(package_name):
    """Test that certain dependencies are importable."""
    importlib.import_module(package_name)


def test_supervisor_import():
    """Test that supervisor module is importable."""
    importlib.import_module("supervisor")


def test_gpu_packages():
    try:
        subprocess.check_call(["nvidia-smi"])

        import torch

        assert torch.cuda.is_available()

        import tensorflow as tf

        assert tf.test.is_built_with_cuda()
        assert tf.config.list_physical_devices("GPU")

    except FileNotFoundError:
        warnings.warn(
            "Skipping GPU import tests since nvidia-smi is not present on test machine."
        )
