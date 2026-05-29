"""line_squeeze 插件单元测试：型号校验、线序判定、视觉相似纠正、business_post_process。"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from schemas.exceptions import ProductNotRegisteredError
from schemas.inference_context import InferenceContext
from vie_plugin_line_squeeze.line_squeeze_detect import (
    VerifyLineSequenceUtils, check_infos, LineSqueezeRecognitionResult, ProductType,
)


def test_check_infos_visual_similar_correction():
    # 仅纠正到 valid_info(1-7) 内的目标：S->5, l->1, b->6；'O'->'0' 不适用(0 非有效线号)，原样保留
    assert check_infos(['S', 'l', 'b', '3']) == ['5', '1', '6', '3']
    assert check_infos(['O']) == ['O']


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
    with patch("vie_plugin_line_squeeze.business_logic.LineSqueezePipeline"):
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
