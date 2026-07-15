# Changelog

本插件变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 变更

- 同步示例和配置中的模型路径，采用 `weights/{scene}/{task}_{arch}_v{N}` 命名规范。

## [0.1.1] - 2026-07-11

### 变更

- 复用框架 YOLO 公共推理管线，并要求兼容的框架版本。
- 统一插件元数据测试路径。

## [0.1.0] - 2026-05-29

### 新增

- 新增 `vie-plugin-line-squeeze` 插件骨架、配置、Schema 和路由。
- 新增 RoiDet 无状态适配、识别管线、线序校验工具及业务单元测试。
- 新增框架与插件二进制 wheel 构建配置和运行示例。
