# vie-plugin-line-squeeze

线序检测插件。插件使用 YOLO 检测 DC/FU 线号 ROI，使用 PP-OCRv5 ONNX 模型批量
识别字符，再按产品型号校验线序。

## API

- 路径：`POST /api/v1/line_squeeze_recognition`
- 表单字段：`file` 为图片，`json_data` 为 JSON 字符串
- 关键参数：`modelParams.product_model`
- 支持型号以 `line_squeeze_detect.py` 的 `ProductType` 为准

`json_data` 示例：

```json
{
  "product": "线序",
  "type": "material-no",
  "modelParams": {
    "product_model": "五路有熔丝盒有磁环"
  }
}
```

## 模型与 metadata

| 资源 | 默认路径 |
| --- | --- |
| ROI 检测模型 | `./weights/line_squeeze/det_v3.onnx` |
| OCR ONNX 模型 | `./weights/line_squeeze/rec_ppocrv5en_v1.onnx` |
| OCR metadata | `./weights/common/official/PP-en_rec_ppocr_v5/inference.yml` |

生产运行不导入 PaddleOCR。OCR 模型输入和输出必须支持动态 batch，输入尺寸和字符表
必须与 `inference.yml` 一致；英文 PP-OCRv5 的 `Global.model_name` 会启用空格类别，
因此对应 438 类 CTC 输出。如 ONNX 权重尚未导出，在框架仓库根目录执行：

```bash
bash scripts/release/export_line_squeeze_onnx.sh
```

导出脚本不会覆盖已存在的模型版本。

## 安装与运行

在框架仓库根目录执行：

```bash
conda run -n padocr pip install -e plugins/vie-plugin-line-squeeze --no-deps
conda run -n padocr python plugins/vie-plugin-line-squeeze/examples/run.py \
  /path/to/image.jpg 五路有熔丝盒有磁环
```

## 测试

在本插件目录执行：

```bash
conda run -n padocr env PYTHONPATH=../..:. python -m pytest tests/ -v
```

检测和 OCR runner 由框架 factory 创建，并在初始化失败或场景关闭时释放。
变更记录见 [CHANGELOG.md](CHANGELOG.md)。
