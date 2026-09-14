"""线序示例经过实际型号校验器验证，无需 OCR 模型。"""
import numpy as np
import pytest

from vie_plugin_line_squeeze.line_squeeze_detect import ProductType
from vie_plugin_line_squeeze.response_docs import RESPONSE_EXAMPLES


@pytest.mark.parametrize("name", ["PASS", "FAIL"])
def test_documented_positions_match_sequence_verification(name):
    expected = RESPONSE_EXAMPLES[name]["result"]
    boxes = np.array([[10, 20, 30, 40, 0.96]] * 5)
    dc = ["1", "2", "3" if name == "PASS" else "9", "4", "5"]
    passed, details = ProductType["五路有熔丝盒有磁环"](dc, ["1", "2", "3", "4", "5"], boxes, boxes)
    assert passed == (expected["verdict"] == "PASS")
    assert len(details) == len(expected["detailList"])
    for item, documented in zip(details, expected["detailList"]):
        assert item["scene"] == documented["scene"]
        assert item["status"] == (documented["verdict"] == "PASS")
        assert item["accuracy"] == documented["accuracy"]
