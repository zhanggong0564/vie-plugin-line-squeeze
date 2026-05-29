from pydantic import BaseModel, Field


class ModelParams(BaseModel):
    product_model: str = Field(..., description="产品型号(例如:五路有熔丝盒有磁环)")


class LineSqueezeRequest(BaseModel):
    """线路压缩检测请求体（原 master 继承 DCFuseRequest，此处独立完整定义）。"""

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
