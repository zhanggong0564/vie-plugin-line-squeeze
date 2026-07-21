# Changelog

本插件变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 变更

- 忽略本地运行日志目录，避免生成文件污染插件工作区。
- 同步示例和配置中的模型路径，采用 `weights/{scene}/{task}_{arch}_v{N}` 命名规范。
- 场景注册迁移到框架 `ScenarioRegistry`。
- ROI 检测器改为接收框架 runner，并使用 `services.vision` 公共操作。
- OCR 从 PaddleOCR 运行时切换为 `rec_ppocrv5en_v1.onnx`，通过 metadata 保持
  PP-OCRv5 预处理、字符表和 CTC 解码契约。
- 兼容 PP-OCRv5 英文模型的动态宽度输入与空格类别（438 类 CTC 输出）metadata。
- DC/FU ROI 分别批量识别，保留第三字符提取、相似字符修正和型号判定行为。
- 双 runner 支持初始化失败回滚和服务关闭统一释放。
- 新增 ONNX OCR、批处理、无 Paddle 导入和资源生命周期测试。

## [0.1.1] - 2026-07-11

### 变更

- 复用框架 YOLO 公共推理管线，并要求兼容的框架版本。
- 统一插件元数据测试路径。

## [0.1.0] - 2026-05-29

### 新增

- 新增 `vie-plugin-line-squeeze` 插件骨架、配置、Schema 和路由。
- 新增 RoiDet 无状态适配、识别管线、线序校验工具及业务单元测试。
- 新增框架与插件二进制 wheel 构建配置和运行示例。
