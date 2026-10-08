import numpy as np

from inference_studio.chainner.image import ChainnerImageAdapter


def test_chainner_image_adapter_round_trip_preserves_pixels_and_channels():
    source = np.array([[[0, 127, 255, 64], [255, 1, 128, 200]]], dtype=np.uint8)
    chainner = ChainnerImageAdapter.to_chainner(source)
    restored = ChainnerImageAdapter.to_vision(chainner)

    assert chainner.dtype == np.float32
    assert chainner.shape == source.shape
    assert chainner.min() >= 0 and chainner.max() <= 1
    assert np.array_equal(restored, source)
