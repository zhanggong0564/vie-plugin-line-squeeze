"""line_squeeze 插件单元测试：型号校验、线序判定、视觉相似纠正、business_post_process。"""
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from schemas.exceptions import ModelInferenceError, ProductNotRegisteredError
from schemas.inference_context import InferenceContext
from vie_plugin_line_squeeze.line_squeeze_detect import (
    VerifyLineSequenceUtils, check_infos, LineSqueezePipeline,
    LineSqueezeRecognitionResult, ProductType,
)


def test_package_metadata_requires_yolo_pipeline_framework():
    project_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = project_path.read_text(encoding="utf-8")

    assert 'version = "0.1.2"' in project
    assert 'dependencies = ["vie-framework>=2.0.1"]' in project


def test_roi_det_uses_shared_yolo_pipeline(monkeypatch):
    from vie_plugin_line_squeeze.line_squeeze_detect import RoiDet

    detector = RoiDet.__new__(RoiDet)
    detector._input_model_shape = [1, 3, 8, 10]
    detector.task = "rect"
    detector.confThreshold = 0.5
    detector.nmsThreshold = 0.6
    detector.filter_classes = None
    detector.agnostic = False
    detector.nc = 2
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    prepared = (object(), object())

    monkeypatch.setattr(
        "vie_plugin_line_squeeze.line_squeeze_detect.prepare_yolo_input",
        lambda value, shape: prepared,
    )
    assert detector.preprocess(image) is prepared


def test_roi_det_post_process_uses_shared_yolo_pipeline(monkeypatch):
    from schemas.inference_context import PreprocMeta
    from vie_plugin_line_squeeze.line_squeeze_detect import RoiDet

    detector = RoiDet.__new__(RoiDet)
    detector._input_model_shape = [1, 3, 8, 10]
    detector.task = "rect"
    detector.confThreshold = 0.5
    detector.nmsThreshold = 0.6
    detector.filter_classes = [1]
    detector.agnostic = True
    detector.nc = 2
    raw_detection = np.array(
        [[1, 2, 3, 4, 0.9, 1], [5, 6, 7, 8, 0.4, 0]], dtype=np.float32
    )
    restored = np.array(
        [[10, 20, 30, 40, 0.9, 1], [50, 60, 70, 80, 0.4, 0]],
        dtype=np.float32,
    )
    prediction = np.zeros((1, 6, 2), dtype=np.float32)
    meta = PreprocMeta(r=1.0, dw=0, dh=0, src_shape=(3, 5, 3))
    captured = {}

    def fake_nms(value, **kwargs):
        captured["nms"] = (value, kwargs)
        return [raw_detection]

    def fake_restore(detections, input_shape, src_shape):
        captured["restore"] = (detections, input_shape, src_shape)
        return restored

    monkeypatch.setattr(
        "vie_plugin_line_squeeze.line_squeeze_detect.run_yolo_nms", fake_nms
    )
    monkeypatch.setattr(
        "vie_plugin_line_squeeze.line_squeeze_detect.restore_yolo_boxes",
        fake_restore,
    )

    result = detector.post_process([prediction], meta)

    assert captured["nms"] == (
        prediction,
        {
            "task": "rect",
            "conf_threshold": 0.5,
            "iou_threshold": 0.6,
            "classes": [1],
            "agnostic": True,
            "nc": 2,
        },
    )
    assert captured["restore"][0] is raw_detection
    assert captured["restore"][1] == [8, 10]
    assert captured["restore"][2] == (3, 5, 3)
    assert result == {
        "rect": [[10.0, 20.0, 30.0, 40.0]],
        "score": [pytest.approx(0.9)],
        "cls": [1.0],
    }


def test_roi_det_post_process_preserves_empty_result(monkeypatch):
    from schemas.inference_context import PreprocMeta
    from vie_plugin_line_squeeze.line_squeeze_detect import RoiDet

    detector = RoiDet.__new__(RoiDet)
    detector._input_model_shape = [1, 3, 8, 10]
    detector.task = "rect"
    detector.confThreshold = 0.5
    detector.nmsThreshold = 0.6
    detector.filter_classes = None
    detector.agnostic = False
    detector.nc = 2
    empty = np.empty((0, 6), dtype=np.float32)
    monkeypatch.setattr(
        "vie_plugin_line_squeeze.line_squeeze_detect.run_yolo_nms",
        lambda *args, **kwargs: [empty],
    )
    monkeypatch.setattr(
        "vie_plugin_line_squeeze.line_squeeze_detect.restore_yolo_boxes",
        lambda detections, input_shape, src_shape: detections.copy(),
    )

    result = detector.post_process(
        [np.zeros((1, 6, 0), dtype=np.float32)],
        PreprocMeta(r=1.0, dw=0, dh=0, src_shape=(3, 5, 3)),
    )

    assert result == {"rect": [], "score": [], "cls": []}


def test_check_infos_visual_similar_correction():
    # 仅纠正到 valid_info(1-7) 内的目标：S->5, l->1, b->6；'O'->'0' 不适用(0 非有效线号)，原样保留
    assert check_infos(['S', 'l', 'b', '3']) == ['5', '1', '6', '3']
    assert check_infos(['O']) == ['O']


def test_pipeline_uses_onnx_text_recognizer():
    with patch(
        "vie_plugin_line_squeeze.line_squeeze_detect.LineSqueezeTextRecognizer"
    ) as recognizer_class, patch(
        "vie_plugin_line_squeeze.line_squeeze_detect.RoiDet"
    ):
        from vie_plugin_line_squeeze.line_squeeze_detect import LineSqueezePipeline

        pipeline = LineSqueezePipeline(
            "det.onnx", "rec.onnx", "inference.yml", det_nc=2
        )

    recognizer_class.assert_called_once_with("rec.onnx", "inference.yml")
    assert pipeline.ocr is recognizer_class.return_value


def test_verify_line_sequence_correct_order():
    v = VerifyLineSequenceUtils(3, verify_dc=True)
    res = v.verify_line_sequence(['1', '2', '3'], 3)
    assert res == [True, True, True]


def test_verify_line_sequence_wrong_order():
    v = VerifyLineSequenceUtils(3, verify_dc=True)
    res = v.verify_line_sequence(['2', '1', '3'], 3)
    assert res == [False, False, True]


@pytest.fixture
def judge():
    """绕过真实模型加载，构造 LineSqueezeJudgeApi。"""
    with (
        patch(
            "vie_plugin_line_squeeze.business_logic.create_inference_runner",
            side_effect=[MagicMock(), MagicMock()],
        ),
        patch("vie_plugin_line_squeeze.business_logic.LineSqueezePipeline"),
    ):
        from vie_plugin_line_squeeze.business_logic import LineSqueezeJudgeApi
        yield LineSqueezeJudgeApi(MagicMock())


def test_unregistered_product_raises(judge):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10, product_type="不存在型号")
    with pytest.raises(ProductNotRegisteredError) as ei:
        judge.business_post_process(ctx)
    assert ei.value.context.get("scenario") == "line_squeeze"


def test_business_post_process_builds_result(judge):
    ctx = InferenceContext(image=np.zeros((10, 10, 3), np.uint8), h=10, w=10,
                           product_type="七路无熔丝盒无磁环")
    # 7 路 dc，全对：识别文本 1..7，boxes 给像素 [x1,y1,x2,y2,score]
    box = np.array([1.0, 1.0, 2.0, 2.0, 0.9])
    ctx.raw_result = LineSqueezeRecognitionResult(
        dc_res=['1', '2', '3', '4', '5', '6', '7'],
        fu_res=[],
        dc_boxes=[box] * 7,
        fu_boxes=[],
    )
    judge.business_post_process(ctx)
    # 统一契约：ctx.result 为 MoMResult 对象（与 plate_screw/panel_label 一致）
    assert ctx.result.status is True
    assert len(ctx.result.detailList) == 7
    assert all(item.scene == "dc" for item in ctx.result.detailList)
    # to_dict 输出形状不变，路由层无需再 isinstance 兜底
    out = ctx.result.to_dict()
    assert out["status"] == "true"
    assert set(out.keys()) == {"status", "detailList", "error_msg", "message"}


def test_pipeline_batches_dc_and_fu_and_uses_third_character():
    pipeline = LineSqueezePipeline.__new__(LineSqueezePipeline)
    pipeline.roi_det = MagicMock()
    pipeline.roi_det.infer.return_value = {
        "rect": [[50, 0, 80, 40], [0, 0, 30, 40]],
        "score": [0.8, 0.9],
        "cls": [0, 1],
    }
    pipeline.ocr = MagicMock()
    pipeline.ocr.predict.side_effect = [
        [SimpleNamespace(text="AA1", score=0.9)],
        [SimpleNamespace(text="BB2", score=0.8)],
    ]

    result = pipeline.infer(np.zeros((50, 90, 3), dtype=np.uint8))

    assert pipeline.ocr.predict.call_count == 2
    assert len(pipeline.ocr.predict.call_args_list[0].args[0]) == 1
    assert len(pipeline.ocr.predict.call_args_list[1].args[0]) == 1
    assert result.dc_res == ["1"]
    assert result.fu_res == ["2"]


def test_pipeline_rejects_empty_roi_before_recognition():
    pipeline = LineSqueezePipeline.__new__(LineSqueezePipeline)
    pipeline.roi_det = MagicMock()
    pipeline.roi_det.infer.return_value = {
        "rect": [[0, 0, 5, 5]],
        "score": [0.9],
        "cls": [1],
    }
    pipeline.ocr = MagicMock()

    with pytest.raises(ModelInferenceError, match="ROI"):
        pipeline.infer(np.zeros((20, 20, 3), dtype=np.uint8))

    pipeline.ocr.predict.assert_not_called()


def test_pipeline_close_attempts_both_models_when_first_close_fails():
    pipeline = LineSqueezePipeline.__new__(LineSqueezePipeline)
    pipeline.roi_det = MagicMock()
    pipeline.ocr = MagicMock()
    pipeline.roi_det.close.side_effect = RuntimeError("det close failed")

    with pytest.raises(RuntimeError, match="det close failed"):
        pipeline.close()

    pipeline.roi_det.close.assert_called_once_with()
    pipeline.ocr.close.assert_called_once_with()


def test_production_modules_import_without_paddleocr():
    script = """
import importlib.abc
import sys

class BlockPaddleOCR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'paddleocr' or fullname.startswith('paddleocr.'):
            raise ModuleNotFoundError('blocked paddleocr import')
        return None

sys.meta_path.insert(0, BlockPaddleOCR())
import vie_plugin_line_squeeze.plugin
assert 'paddleocr' not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "../..:."

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(__file__) + "/..",
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
