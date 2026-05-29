'''线路压缩检测业务逻辑：适配模板方法基类，型号校验 + 线序判定收敛到 business_post_process(ctx)。'''

from services.api import detection_factory
from services.base import BusinessLogicBase
from schemas.data_base import MoMResult, DetectionItem, MessageType
from schemas.exceptions import ProductNotRegisteredError, ModelInferenceError
from schemas.inference_context import InferenceContext
from utils import vision_logger
from .line_squeeze_detect import LineSqueezePipeline, ProductType


@detection_factory.register("line_squeeze")
class LineSqueezeJudgeApi(BusinessLogicBase):
    def _initialize_model(self, settings):
        from .config import LineSqueezeConfig
        cfg = LineSqueezeConfig()
        try:
            self.detector = LineSqueezePipeline(
                det_model_path=cfg.det_model_path,
                ocr_model_dir=cfg.ocr_model_dir,
                det_nc=cfg.det_nc,
                det_conf_threshold=cfg.det_conf_threshold,
                det_nms_threshold=cfg.det_nms_threshold,
            )
        except Exception as e:
            vision_logger.error(f"initialize model failed, error: {e}")
            raise ModelInferenceError("line_squeeze 模型加载失败", scenario="line_squeeze", original_error=e)

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
