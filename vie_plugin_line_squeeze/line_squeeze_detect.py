'''线路压缩检测：RoiDet（无状态 ONNX 检测）+ OCR 识别管线 + 线序校验工具。'''

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from services.base import BaseVisionInfer
from services.inference import InferenceRunner
from services.vision.boxes import scale_boxes, sort_boxes, xywhr2xyxyxyxy
from services.vision.nms import non_max_suppression_v8
from services.vision.preprocessing import letterbox
from schemas.inference_context import PreprocMeta
from schemas.exceptions import ModelInferenceError
from utils import vision_logger
from .ocr_models import LineSqueezeTextRecognizer


class RoiDet(BaseVisionInfer):
    """ROI detector backed by an injected inference runner."""

    def __init__(
        self,
        runner: InferenceRunner,
        nc,
        confThreshold=0.5,
        nmsThreshold=0.5,
    ):
        super().__init__(
            runner,
            confThreshold=confThreshold,
            nmsThreshold=nmsThreshold,
        )
        self.task = "rect"
        self.nc = nc
        self.filter_classes = None
        self.agnostic = False

    def preprocess(self, im):
        img, r, dw, dh = letterbox(
            im=im,
            auto=False,
            new_shape=self._input_model_shape[2:],
        )
        tensor = np.stack([img])
        tensor = tensor[..., ::-1].transpose((0, 3, 1, 2))
        tensor = np.ascontiguousarray(tensor).astype(np.float32)
        tensor /= 255.0
        meta = PreprocMeta(r=r, dw=dw, dh=dh, src_shape=im.shape)
        return tensor, meta

    def post_process(self, preds, meta):
        p = non_max_suppression_v8(
            preds[0],
            task=self.task,
            conf_thres=self.confThreshold,
            iou_thres=self.nmsThreshold,
            classes=self.filter_classes,
            agnostic=self.agnostic,
            multi_label=False,
            nc=self.nc,
        )
        image_shape = meta.src_shape[:2]
        input_shape = self.input_model_shape[2:]
        res = {}
        pred = p[0].copy()
        pred[:, :4] = scale_boxes(
            input_shape,
            pred[:, :4],
            image_shape,
            xywh=False,
        )
        pred = np.concatenate([pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1)
        bbox = pred[:, :4]  # xyxy
        if self.task == "obb":
            bbox = xywhr2xyxyxyxy(pred[:, :5])
        conf = pred[:, -2]
        clas = pred[:, -1]
        # 按分数过滤（保持 master 行为）；单趟掩码避免三列过滤后错位
        keep = conf >= self.confThreshold
        res["rect"] = bbox[keep].tolist()
        res["score"] = conf[keep].tolist()
        res["cls"] = clas[keep].tolist()
        return res


VISUAL_SIMILAR_MAP = {
    's': ['5'], 'S': ['5'], 'l': ['1'], 'i': ['1'], 'I': ['1'], 'O': ['0'], 'o': ['0'],
    'b': ['6'], 'q': ['9'], 'T': ['1'], 't': ['1'], 'Z': ['2'], 'a': ['2'], 'A': ['4'],
    '+': ['3'], "G": ['5'], "B": ['5'],
}


def check_infos(infos: List[str]) -> List[str]:
    """视觉相似字符纠正（原 LineSqueezeRecognition.check_infos）。"""
    corrected = []
    valid_info = ['1', '2', '3', '4', '5', '6', '7']
    for char in infos:
        if char in valid_info:
            corrected.append(char)
        elif char in VISUAL_SIMILAR_MAP and VISUAL_SIMILAR_MAP[char][0] in valid_info:
            corrected.append(VISUAL_SIMILAR_MAP[char][0])
        else:
            corrected.append(char)
    return corrected


class VerifyLineSequenceUtils:
    """线序校验（原样移植自 master:services/LineSqueeze/business_logic.py）。"""

    def __init__(self, nums: int, verify_dc: bool = False, verify_fu: bool = False):
        self.nums = nums
        self.verify_dc = verify_dc
        self.verify_fu = verify_fu

    def __call__(self, dc_infos, fu_infos, sorted_dc_boxes, sorted_fu_boxes):
        res_infos = []
        if (len(dc_infos) != 0 and len(dc_infos) != self.nums) or (len(fu_infos) != 0 and len(fu_infos) != self.nums):
            return False, res_infos
        if self.verify_dc:
            res_info = self.verify_line_sequence(dc_infos, self.nums)
            if len(sorted_dc_boxes) == 0:
                sorted_dc_boxes = np.array([[] for _ in range(len(res_info) + 1)])
            for res, box in zip(res_info, sorted_dc_boxes):
                res_infos.append({
                    "status": res, "scene": "dc",
                    "coordinate": box[:4].tolist(),  # 像素 xyxy，归一化交给基类 normalize_hook
                    "accuracy": float(box[4]) if len(box) != 0 else 0.0,
                })
        if self.verify_fu:
            res_info = self.verify_line_sequence(fu_infos, self.nums)
            if len(sorted_fu_boxes) == 0:
                sorted_fu_boxes = np.array([[] for _ in range(len(res_info) + 1)])
            for res, box in zip(res_info, sorted_fu_boxes):
                res_infos.append({
                    "status": res, "scene": "fu",
                    "coordinate": box[:4].tolist(),  # 像素 xyxy，归一化交给基类 normalize_hook
                    "accuracy": float(box[4]) if len(box) != 0 else 0.0,
                })
        if self.verify_dc and self.verify_fu:
            return (
                all([r['status'] for r in res_infos if r['scene'] == 'dc'])
                and all([r['status'] for r in res_infos if r['scene'] == 'fu']),
                res_infos,
            )
        elif self.verify_dc:
            return all([r['status'] for r in res_infos if r['scene'] == 'dc']), res_infos
        elif self.verify_fu:
            return all([r['status'] for r in res_infos if r['scene'] == 'fu']), res_infos
        else:
            return True, res_infos

    def verify_line_sequence(self, infos: List[str], nums: int):
        res_infos = [False for _ in range(nums)]
        try:
            if len(infos) != nums:
                for info in infos:
                    info = int(info)
                    res_infos[info - 1] = True
            else:
                for i in range(nums):
                    info = int(infos[i])
                    if (info - 1) >= nums:
                        continue
                    res_infos[info - 1] = (i + 1) == info
        except ValueError:
            vision_logger.warning(f"verify_line_sequence error, infos: {infos}")
            return res_infos
        return res_infos


# 型号 -> 校验器（原样移植自 master）
ProductType: Dict[str, VerifyLineSequenceUtils] = {
    "五路有熔丝盒有磁环": VerifyLineSequenceUtils(5, verify_dc=True, verify_fu=True),
    "五路有熔丝盒无磁环": VerifyLineSequenceUtils(5, verify_dc=True, verify_fu=True),
    "六路有熔丝盒无磁环": VerifyLineSequenceUtils(6, verify_dc=True, verify_fu=True),
    "六路无熔丝盒无磁环": VerifyLineSequenceUtils(6, verify_dc=True),
    "七路无熔丝盒无磁环": VerifyLineSequenceUtils(7, verify_dc=True),
    "七路有熔丝盒无磁环": VerifyLineSequenceUtils(7, verify_dc=True, verify_fu=True),
}


@dataclass
class LineSqueezeRecognitionResult:
    """识别管线产出的中间结果（型号校验前）。boxes 为像素 [x1,y1,x2,y2,score]。"""
    dc_res: List[str] = field(default_factory=list)
    fu_res: List[str] = field(default_factory=list)
    dc_boxes: List = field(default_factory=list)
    fu_boxes: List = field(default_factory=list)


class LineSqueezePipeline:
    """RoiDet + OCR 识别管线。infer(image) 产出归一化 boxes + OCR 文本，型号校验交给 business_post_process。"""

    def __init__(
        self,
        ocr_metadata_path: str,
        det_nc: int = 2,
        det_conf_threshold: float = 0.5,
        det_nms_threshold: float = 0.5,
        *,
        detection_runner: InferenceRunner,
        recognition_runner: InferenceRunner,
    ):
        self.roi_det = RoiDet(
            detection_runner,
            det_nc,
            confThreshold=det_conf_threshold,
            nmsThreshold=det_nms_threshold,
        )
        self.ocr = LineSqueezeTextRecognizer(
            ocr_metadata_path,
            runner=recognition_runner,
        )
        self.classes2names = {0: "fu_line", 1: "dc_line"}

    def infer(self, image: np.ndarray) -> LineSqueezeRecognitionResult:
        results = self.roi_det.infer(image)  # {rect, score, cls}
        classes = results['cls']
        score = results['score']
        if len(results['rect']) == 0:
            return LineSqueezeRecognitionResult()
        rect = np.concatenate((np.array(results['rect']), np.array(score).reshape(-1, 1)), axis=1)
        dc_boxes = [box for cls, box in zip(classes, rect) if cls == 1]
        fu_boxes = [box for cls, box in zip(classes, rect) if cls == 0]
        # ⚠️ 新框架 sort_boxes 返回 (boxes, indices)，需解包（master 旧版只返回 boxes）
        sorted_dc_boxes, _ = sort_boxes(dc_boxes)
        sorted_fu_boxes, _ = sort_boxes(fu_boxes)
        dc_rois = self._crop_rois(image, sorted_dc_boxes)
        fu_rois = self._crop_rois(image, sorted_fu_boxes)
        dc_res = self._recognize(dc_rois)
        fu_res = self._recognize(fu_rois)
        dc_res = check_infos(dc_res)
        fu_res = check_infos(fu_res)
        return LineSqueezeRecognitionResult(dc_res, fu_res, sorted_dc_boxes, sorted_fu_boxes)

    @staticmethod
    def _crop_rois(image: np.ndarray, boxes) -> list[np.ndarray]:
        rois = []
        for box in boxes:
            height, width = image.shape[:2]
            x1 = max(int(box[0]), 0)
            y1 = max(int(box[1]) + 10, 0)
            x2 = min(int(box[2]), width)
            y2 = min(int(box[3]) - 10, height)
            if x2 <= x1 or y2 <= y1:
                raise ModelInferenceError(
                    "line_squeeze detected ROI is empty after clipping",
                    scenario="line_squeeze",
                )
            roi = image[y1:y2, x1:x2]
            rois.append(roi)
        return rois

    def _recognize(self, rois: list[np.ndarray]) -> list[str]:
        if not rois:
            return []
        return [
            result.text[2]
            for result in self.ocr.predict(rois)
            if len(result.text) > 2
        ]

    def close(self) -> None:
        first_error = None
        for model in (self.roi_det, self.ocr):
            try:
                model.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
