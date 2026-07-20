from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from services.inference import OnnxRuntimeOptions, OnnxRuntimeRunner, TensorInfo
from vie_plugin_line_squeeze.ocr_models import LineSqueezeTextRecognizer


CHARACTERS = ("0", "1", "2")


class RecordingRunner:
    input_infos = (TensorInfo("x", ("batch", 3, 48, "width"), "tensor(float)"),)
    output_infos = (TensorInfo("output", ("batch", "time", 4), "tensor(float)"),)
    providers = ("CPUExecutionProvider",)

    def __init__(self):
        self.input_shapes = []

    def run(self, inputs):
        tensor = inputs["x"]
        self.input_shapes.append(tensor.shape)
        logits = np.zeros((tensor.shape[0], 2, 4), dtype=np.float32)
        logits[:, :, 2] = 1.0
        return [logits]


def _write_metadata(path: Path) -> None:
    data = {
        "PreProcess": {
            "transform_ops": [
                {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                {"RecResizeImg": {"image_shape": [3, 48, 320]}},
            ]
        },
        "PostProcess": {
            "name": "CTCLabelDecode",
            "character_dict": list(CHARACTERS),
        },
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_recognizer_loads_metadata_and_uses_metadata_width(tmp_path):
    metadata = tmp_path / "inference.yml"
    _write_metadata(metadata)
    runner = RecordingRunner()
    recognizer = LineSqueezeTextRecognizer(str(metadata), runner=runner)
    narrow = np.zeros((48, 64, 3), dtype=np.uint8)
    wide = np.zeros((96, 643, 3), dtype=np.uint8)

    assert recognizer.characters == CHARACTERS
    results = recognizer.predict([narrow, wide])
    assert [result.text for result in results] == ["1", "1"]
    assert [result.score for result in results] == [1.0, 1.0]
    assert runner.input_shapes == [(2, 3, 48, 320)]


def test_recognizer_preprocess_matches_paddle_rec_resize(tmp_path):
    metadata = tmp_path / "inference.yml"
    _write_metadata(metadata)
    recognizer = LineSqueezeTextRecognizer(
        str(metadata),
        runner=RecordingRunner(),
    )
    image = np.arange(96 * 643 * 3, dtype=np.uint8).reshape(96, 643, 3)
    resized = cv2.resize(image, (320, 48), interpolation=cv2.INTER_LINEAR)
    expected = resized.astype(np.float32).transpose(2, 0, 1) / 255
    expected = ((expected - 0.5) / 0.5)[None]

    actual = recognizer.preprocess_batch([image])

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_exported_onnx_matches_paddle_text_on_numeric_crops():
    paddleocr = pytest.importorskip("paddleocr")
    root = Path(__file__).resolve().parents[3]
    model_dir = root / "weights/common/official/PP-en_rec_ppocr_v5"
    onnx_path = root / "weights/line_squeeze/rec_ppocrv5en_v1.onnx"
    if not onnx_path.is_file():
        pytest.skip("line-squeeze ONNX recognition model is not available")

    crops = []
    for text in ("ABC1", "ABC5", "ABC7"):
        image = np.full((64, 180, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            text,
            (5, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        crops.append(image)

    paddle_model = paddleocr.TextRecognition(
        model_dir=str(model_dir), model_name="en_PP-OCRv5_mobile_rec"
    )
    paddle_results = list(paddle_model.predict(input=crops))
    runner = OnnxRuntimeRunner(
        str(onnx_path),
        OnnxRuntimeOptions(
            providers=("CPUExecutionProvider",),
            warmup=False,
            require_cuda=False,
        ),
    )
    onnx_model = LineSqueezeTextRecognizer(
        str(model_dir / "inference.yml"),
        runner=runner,
    )

    assert [item.text for item in onnx_model.predict(crops)] == [
        item["rec_text"] for item in paddle_results
    ]
