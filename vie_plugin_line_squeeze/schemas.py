from pydantic import BaseModel, Field

from schemas.common import VisualReferenceParams


class ModelParams(VisualReferenceParams):
    product_model: str = Field(..., description="产品型号(例如:五路有熔丝盒有磁环)")


class LineSqueezeRequest(BaseModel):
    """线序检测请求参数。"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
