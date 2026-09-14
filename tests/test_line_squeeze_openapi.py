"""请求示例与插件实际解析、输入转换保持一致。"""

import asyncio
import copy

import numpy as np

from vie_plugin_line_squeeze.plugin import line_squeeze_router


def test_document_example_matches_request_parser():
    router = line_squeeze_router
    payload = copy.deepcopy(router.request_document_example)
    request = router.request_schema(payload)
    assert isinstance(request, router.request_document_model)
    assert request.model_dump(by_alias=True) == router.request_document_model.model_validate(payload).model_dump(by_alias=True)


def test_document_example_builds_inputs():
    router = line_squeeze_router
    request = router.request_schema(copy.deepcopy(router.request_document_example))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    inputs = router.get_inputs(request, image)
    assert inputs.product_type == request.modelParams.product_model
    assert inputs.image is image


def test_document_uses_line_sequence_name_and_preserves_path():
    route = line_squeeze_router.get_router().routes[0]
    assert route.path == "/line_squeeze_recognition"
    assert route.summary == "线序检测接口"
    assert line_squeeze_router.tag == "线序检测"
    assert line_squeeze_router.request_document_example["product"] == "线序检测"
