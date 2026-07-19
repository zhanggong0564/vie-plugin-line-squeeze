"""line_squeeze 插件用法示例。

运行（从仓库根目录）：
    python plugins/vie-plugin-line-squeeze/examples/run.py <图片路径> [产品型号]

前置：已 `pip install -e plugins/vie-plugin-line-squeeze`；检测与 OCR ONNX 权重就位。
"""
import os
import sys
import json

import cv2
import numpy as np

# 让示例在任意 cwd 下都能 import 框架（services/schemas 在仓库根，未作为包安装）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import vie_plugin_line_squeeze.plugin  # noqa: E402,F401  触发 ScenarioRegistry 注册
from services.scenario_registry import scenario_registry  # noqa: E402
from schemas.data_base import InputParamsBusiness  # noqa: E402


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    product_model = sys.argv[2] if len(sys.argv) > 2 else "五路有熔丝盒有磁环"
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")
    h, w = image.shape[:2]

    detector = scenario_registry.create("line_squeeze")
    try:
        result = detector.detect(InputParamsBusiness(image=image, product_type=product_model))
        out = result.to_dict()
        print(json.dumps(out, ensure_ascii=False, indent=2))

        # 可视化：归一化 8 点坐标 → 像素，绿框=线序正确 红框=异常
        for item in out.get("detailList", []):
            coord = item.get("coordinate", [])
            if len(coord) != 8:
                continue
            pts = np.array([[int(coord[i] * w), int(coord[i + 1] * h)] for i in range(0, 8, 2)], np.int32)
            color = (0, 255, 0) if item.get("status") == "true" else (0, 0, 255)
            cv2.polylines(image, [pts], True, color, 2)
        save_path = "line_squeeze_result.jpg"
        cv2.imwrite(save_path, image)
        print(f"可视化结果已保存: {save_path}")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
