'''Entry point: register the scene and expose ``line_squeeze_router``.'''

import numpy as np

from routers.base_router import BaseRouter
from schemas.data_base import InputParamsBusiness
from .schemas import LineSqueezeRequest
from . import business_logic  # noqa: F401  触发 ScenarioRegistry 注册


class LineSqueezeRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return LineSqueezeRequest(**json_dict)

    @staticmethod
    def _extract_product_type(request_params):
        # 本场景型号字段名为 product_model，重写基类默认的 product_type 提取，
        # 使数据回流按型号分目录而非落到 _unknown_model。
        model_params = getattr(request_params, "modelParams", None)
        return getattr(model_params, "product_model", None) if model_params else None

    def get_inputs(self, request_params: LineSqueezeRequest, image: np.ndarray):
        product_model = request_params.modelParams.product_model
        return InputParamsBusiness(image=image, product_type=product_model)


line_squeeze_router = LineSqueezeRouter(
    router_name="line_squeeze_router",
    api_path="/line_squeeze_recognition",
    summary="线路压缩检测接口",
    description="根据输入的图像和产品型号，返回线序检测结果",
    detector_type="line_squeeze",
    tag="线路压缩检测",
)
