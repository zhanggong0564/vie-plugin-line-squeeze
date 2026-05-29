'''entry_point 模块：导入 business_logic 触发工厂注册，并暴露 line_squeeze_router。'''

import numpy as np

from routers.base_router import BaseRouter
from schemas.data_base import InputParamsBusiness
from .schemas import LineSqueezeRequest
from . import business_logic  # noqa: F401  导入即触发 @detection_factory.register("line_squeeze")


class LineSqueezeRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return LineSqueezeRequest(**json_dict)

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
