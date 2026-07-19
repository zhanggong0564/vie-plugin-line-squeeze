from unittest.mock import MagicMock, call, patch

import pytest

from services.inference import OnnxRuntimeOptions, RunnerSpec
from services.scenario_registry import scenario_registry


def test_scene_registers_with_scenario_registry():
    from vie_plugin_line_squeeze.business_logic import LineSqueezeJudgeApi

    assert scenario_registry.snapshot()["line_squeeze"] is LineSqueezeJudgeApi


def test_business_initialization_creates_and_injects_both_runners():
    from vie_plugin_line_squeeze.business_logic import LineSqueezeJudgeApi

    settings = MagicMock()
    detection_runner = MagicMock()
    recognition_runner = MagicMock()
    with (
        patch(
            "vie_plugin_line_squeeze.business_logic.create_inference_runner",
            side_effect=[detection_runner, recognition_runner],
        ) as runner_factory,
        patch(
            "vie_plugin_line_squeeze.business_logic.LineSqueezePipeline"
        ) as pipeline_class,
    ):
        api = LineSqueezeJudgeApi(settings)

    options = OnnxRuntimeOptions.from_settings(settings)
    assert runner_factory.call_args_list == [
        call(
            RunnerSpec(
                scenario="line_squeeze",
                onnx_path="./weights/line_squeeze/det_v3.onnx",
            ),
            options,
        ),
        call(
            RunnerSpec(
                scenario="line_squeeze",
                onnx_path="./weights/line_squeeze/rec_ppocrv5en_v1.onnx",
            ),
            options,
        ),
    ]
    pipeline_class.assert_called_once_with(
        ocr_metadata_path="./weights/common/official/PP-en_rec_ppocr_v5/inference.yml",
        det_nc=2,
        det_conf_threshold=0.5,
        det_nms_threshold=0.5,
        detection_runner=detection_runner,
        recognition_runner=recognition_runner,
    )
    assert api.detector is pipeline_class.return_value


def test_second_runner_failure_closes_first_runner():
    from vie_plugin_line_squeeze.business_logic import LineSqueezeJudgeApi

    detection_runner = MagicMock()
    with (
        patch(
            "vie_plugin_line_squeeze.business_logic.create_inference_runner",
            side_effect=[detection_runner, RuntimeError("runner failed")],
        ),
        pytest.raises(Exception, match="line_squeeze 模型加载失败"),
    ):
        LineSqueezeJudgeApi(MagicMock())

    detection_runner.close.assert_called_once_with()


def test_pipeline_failure_closes_both_runners():
    from vie_plugin_line_squeeze.business_logic import LineSqueezeJudgeApi

    runners = [MagicMock(), MagicMock()]
    with (
        patch(
            "vie_plugin_line_squeeze.business_logic.create_inference_runner",
            side_effect=runners,
        ),
        patch(
            "vie_plugin_line_squeeze.business_logic.LineSqueezePipeline",
            side_effect=RuntimeError("pipeline failed"),
        ),
        pytest.raises(Exception, match="line_squeeze 模型加载失败"),
    ):
        LineSqueezeJudgeApi(MagicMock())

    for runner in runners:
        runner.close.assert_called_once_with()


def test_business_close_is_idempotent():
    from vie_plugin_line_squeeze.business_logic import LineSqueezeJudgeApi

    pipeline = MagicMock()
    with (
        patch(
            "vie_plugin_line_squeeze.business_logic.create_inference_runner",
            side_effect=[MagicMock(), MagicMock()],
        ),
        patch(
            "vie_plugin_line_squeeze.business_logic.LineSqueezePipeline",
            return_value=pipeline,
        ),
    ):
        api = LineSqueezeJudgeApi(MagicMock())

    api.close()
    api.close()

    pipeline.close.assert_called_once_with()
