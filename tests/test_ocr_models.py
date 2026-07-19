from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from services.inference import TensorInfo
from vie_plugin_line_squeeze.ocr_models import LineSqueezeTextRecognizer


class FakeRunner:
    def __init__(self, input_shape=("batch", 3, 48, 320), output_classes=4):
        self.input_infos = (TensorInfo("images", input_shape, "tensor(float)"),)
        self.output_infos = (
            TensorInfo(
                "softmax_0.tmp_0",
                ("batch", "steps", output_classes),
                "tensor(float)",
            ),
        )
        self.providers = ("CPUExecutionProvider",)
        self.run = Mock()
        self.close = Mock()


@pytest.fixture
def metadata_path(tmp_path):
    path = tmp_path / "inference.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "PreProcess": {
                    "transform_ops": [
                        {
                            "DecodeImage": {
                                "img_mode": "BGR",
                                "channel_first": False,
                            }
                        },
                        {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    ]
                },
                "PostProcess": {
                    "name": "CTCLabelDecode",
                    "character_dict": ["0", "1", "2"],
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


def test_preprocess_matches_fixed_width_ppocr_contract(metadata_path):
    runner = FakeRunner()
    recognizer = LineSqueezeTextRecognizer(str(metadata_path), runner=runner)
    image = np.full((24, 20, 3), 255, dtype=np.uint8)

    tensor = recognizer.preprocess_batch([image])

    assert tensor.shape == (1, 3, 48, 320)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[:, :, :, :40], 1.0)
    np.testing.assert_allclose(tensor[:, :, :, 40:], 0.0)


def test_predict_decodes_ctc_text_and_score(metadata_path):
    runner = FakeRunner()
    logits = np.zeros((1, 5, 4), dtype=np.float32)
    for step, (index, score) in enumerate(
        [(0, 0.9), (1, 0.8), (1, 0.7), (2, 0.6), (0, 0.5)]
    ):
        logits[0, step, index] = score
    runner.run.return_value = [logits]
    recognizer = LineSqueezeTextRecognizer(str(metadata_path), runner=runner)

    results = recognizer.predict([np.zeros((24, 20, 3), dtype=np.uint8)])

    assert results[0].text == "01"
    assert results[0].score == pytest.approx(0.7)
    runner.run.assert_called_once()


def test_initialization_accepts_dynamic_width(metadata_path):
    runner = FakeRunner(input_shape=("batch", 3, 48, "width"))

    recognizer = LineSqueezeTextRecognizer(str(metadata_path), runner=runner)

    assert recognizer.input_width == 320


def test_initialization_uses_ppocrv5_space_character(metadata_path):
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    data["Global"] = {"model_name": "en_PP-OCRv5_mobile_rec"}
    metadata_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    recognizer = LineSqueezeTextRecognizer(
        str(metadata_path),
        runner=FakeRunner(output_classes=5),
    )

    assert recognizer.characters[-1] == " "


@pytest.mark.parametrize(
    "input_shape",
    [(1, 3, 48, 320), ("batch", 3, 32, 320), ("batch", 3, 48, 640)],
)
def test_initialization_rejects_incompatible_input_metadata(
    metadata_path, input_shape
):
    with pytest.raises(ValueError, match="input"):
        LineSqueezeTextRecognizer(
            str(metadata_path),
            runner=FakeRunner(input_shape=input_shape),
        )


def test_initialization_rejects_output_character_mismatch(metadata_path):
    with pytest.raises(ValueError, match="classes"):
        LineSqueezeTextRecognizer(
            str(metadata_path),
            runner=FakeRunner(output_classes=5),
        )


def test_empty_batch_skips_runner_and_close_releases_it(metadata_path):
    runner = FakeRunner()
    recognizer = LineSqueezeTextRecognizer(str(metadata_path), runner=runner)

    assert recognizer.predict([]) == []
    recognizer.close()

    runner.run.assert_not_called()
    runner.close.assert_called_once_with()
