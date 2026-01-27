"""Basic tests to verify project setup is correct."""

import importlib


def test_data_package_imports() -> None:
    """Verify data package can be imported."""
    data = importlib.import_module("data")
    assert data is not None

    processors = importlib.import_module("data.processors")
    assert processors is not None


def test_models_package_imports() -> None:
    """Verify models package can be imported."""
    models = importlib.import_module("models")
    assert models is not None

    architectures = importlib.import_module("models.architectures")
    assert architectures is not None

    training = importlib.import_module("models.training")
    assert training is not None

    export = importlib.import_module("models.export")
    assert export is not None


def test_evaluation_package_imports() -> None:
    """Verify evaluation package can be imported."""
    evaluation = importlib.import_module("evaluation")
    assert evaluation is not None


def test_torch_available() -> None:
    """Verify PyTorch is installed and importable."""
    import torch

    assert torch is not None
    # Check we can create a tensor
    tensor = torch.zeros(1)
    assert tensor.shape == (1,)


def test_onnx_available() -> None:
    """Verify ONNX is installed and importable."""
    import onnx

    assert onnx is not None


def test_pandas_numpy_available() -> None:
    """Verify pandas and numpy are installed."""
    import numpy as np
    import pandas as pd

    assert np is not None
    assert pd is not None
