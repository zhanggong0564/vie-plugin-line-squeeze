from pydantic import Field
from pydantic_settings import SettingsConfigDict

from services.base import SceneSettings


class LineSqueezeConfig(SceneSettings):
    model_config = SettingsConfigDict(
        env_prefix="LINE_SQUEEZE_",
        env_file=".env",
        extra="ignore",
    )

    det_model_path: str = "./weights/line_squeeze/det_v3.onnx"
    ocr_model_path: str = "./weights/line_squeeze/rec_ppocrv5en_v1.onnx"
    ocr_metadata_path: str = (
        "./weights/common/official/PP-en_rec_ppocr_v5/inference.yml"
    )
    det_nc: int = Field(default=2, gt=0)
    det_conf_threshold: float = Field(default=0.5, ge=0, le=1)
    det_nms_threshold: float = Field(default=0.5, ge=0, le=1)
