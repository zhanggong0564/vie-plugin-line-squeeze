'''线路压缩检测业务逻辑：适配模板方法基类，型号校验 + 线序判定收敛到 business_post_process(ctx)。'''

from services.base import BusinessLogicBase
from services.inference import (
    OnnxRuntimeOptions,
    RunnerSpec,
    create_inference_runner,
)
from services.scenario_registry import scenario_registry
from schemas.data_base import MoMResult, DetectionItem, MessageType
from schemas.exceptions import ProductNotRegisteredError, ModelInferenceError
from schemas.inference_context import InferenceContext
from utils import vision_logger
from .line_squeeze_detect import LineSqueezePipeline, ProductType


@scenario_registry.register("line_squeeze")
class LineSqueezeJudgeApi(BusinessLogicBase):
    def _initialize_model(self, settings):
        from .config import LineSqueezeConfig
        cfg = LineSqueezeConfig()
        created_runners = []
        try:
            options = OnnxRuntimeOptions.from_settings(settings)
            detection_runner = create_inference_runner(
                RunnerSpec(
                    scenario="line_squeeze",
                    onnx_path=cfg.det_model_path,
                ),
                options,
            )
            created_runners.append(detection_runner)
            recognition_runner = create_inference_runner(
                RunnerSpec(
                    scenario="line_squeeze",
                    onnx_path=cfg.ocr_model_path,
                ),
                options,
            )
            created_runners.append(recognition_runner)
            self.detector = LineSqueezePipeline(
                ocr_metadata_path=cfg.ocr_metadata_path,
                det_nc=cfg.det_nc,
                det_conf_threshold=cfg.det_conf_threshold,
                det_nms_threshold=cfg.det_nms_threshold,
                detection_runner=detection_runner,
                recognition_runner=recognition_runner,
            )
        except Exception as e:
            for runner in created_runners:
                try:
                    runner.close()
                except Exception as close_error:
                    vision_logger.warning(
                        f"line_squeeze 初始化回滚清理失败: {close_error}"
                    )
            vision_logger.error(f"initialize model failed, error: {e}")
            raise ModelInferenceError(
                "line_squeeze 模型加载失败",
                scenario="line_squeeze",
                original_error=e,
            ) from e

    def business_post_process(self, ctx: InferenceContext) -> None:
        product_type = ctx.product_type
        if product_type not in ProductType:
            raise ProductNotRegisteredError(
                f"产品型号 '{product_type}' 未在 line_squeeze ProductType 中注册",
                product_type=product_type,
                scenario="line_squeeze",
            )
        rec = ctx.raw_result  # LineSqueezeRecognitionResult
        passed, infos = ProductType[product_type](
            rec.dc_res, rec.fu_res, rec.dc_boxes, rec.fu_boxes
        )
        # 统一结果契约：裸 dict → MoMResult/DetectionItem；坐标为像素值，
        # 归一化由基类 normalize_hook 统一处理（NORMALIZE 保持默认 True）。
        ctx.result = MoMResult(
            detailList=[DetectionItem(**info) for info in infos],
            status=passed,
            message=MessageType.SUCCESS.value if passed else MessageType.FAIL.value,
        )
