from types import SimpleNamespace

import numpy as np
import pytest

from inference_studio.chainner import ChainnerValueAdapter
from inference_studio.chainner.types import ChainnerTypeAdapter
from inference_studio.runtime import ImageValue, TensorValue


def port(type_name: str):
    return SimpleNamespace(output_type={"name": type_name}, input_type={"name": type_name}, associated_type=np.ndarray)


def test_ndarray_conversion_uses_port_metadata():
    adapter = ChainnerValueAdapter()
    array = np.zeros((8, 8, 3), np.float32)
    assert isinstance(adapter.to_vision(port("Image"), array), ImageValue)
    tensor = adapter.to_vision(port("Tensor"), array)
    assert isinstance(tensor, TensorValue)
    assert tensor.shape == (8, 8, 3) and tensor.dtype == "float32"
    assert ChainnerTypeAdapter.adapt({"name": "Tensor"}, np.ndarray) == ["tensor"]


def test_runtime_value_mismatch_is_explicit():
    adapter = ChainnerValueAdapter()
    tensor = TensorValue(np.zeros((2, 2)), (2, 2), "float64")
    with pytest.raises(ValueError, match="chainner_value_type_mismatch"):
        adapter.to_chainner(port("Image"), tensor)
