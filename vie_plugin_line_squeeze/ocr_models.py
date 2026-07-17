"""Line-squeeze ONNX OCR adapters."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from services.base import BaseCtcRecognitionPipeline, OnnxRuntimeRunner


def _load_recognition_metadata(metadata_path: str) -> tuple[tuple[str, ...], int, int]:
    path = Path(metadata_path)
    if path.is_dir():
        path = path / "inference.yml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        operations = data["PreProcess"]["transform_ops"]
        resize = next(item["RecResizeImg"] for item in operations if "RecResizeImg" in item)
        image_shape = resize["image_shape"]
        postprocess = data["PostProcess"]
        characters = postprocess["character_dict"]
    except (OSError, KeyError, StopIteration, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid recognition metadata: {path}") from exc
    if (
        postprocess.get("name") != "CTCLabelDecode"
        or not isinstance(characters, list)
        or not characters
    ):
        raise ValueError("PostProcess must declare a non-empty CTC character_dict")
    if (
        not isinstance(image_shape, (list, tuple))
        or len(image_shape) < 3
        or image_shape[0] != 3
        or not all(isinstance(value, int) and value > 0 for value in image_shape[:3])
    ):
        raise ValueError("RecResizeImg.image_shape must start with [3, height, width]")
    return tuple(characters), image_shape[1], image_shape[2]


class LineSqueezeTextRecognizer(BaseCtcRecognitionPipeline):
    """PP-OCRv5 English dynamic-width recognizer backed by ONNX Runtime."""

    def __init__(self, model_path: str, metadata_path: str, runner=None) -> None:
        characters, input_height, metadata_width = _load_recognition_metadata(
            metadata_path
        )
        self.metadata_width = metadata_width
        selected_runner = (
            runner
            if runner is not None
            else OnnxRuntimeRunner(model_path, execution_mode="sequential")
        )
        if selected_runner.output_infos:
            output_shape = selected_runner.output_infos[0].shape
            output_classes = output_shape[-1] if output_shape else None
            if isinstance(output_classes, int) and output_classes > 0:
                if output_classes == len(characters) + 2:
                    characters = (*characters, " ")
                elif output_classes != len(characters) + 1:
                    raise ValueError(
                        "recognition ONNX output classes do not match metadata"
                    )
        super().__init__(
            selected_runner,
            characters,
            input_height=input_height,
            max_width=3200,
        )

    def _target_width(self, images: Sequence[np.ndarray]) -> int:
        super()._target_width(images)
        max_ratio = max(image.shape[1] / float(image.shape[0]) for image in images)
        return min(
            self.max_width,
            max(self.metadata_width, int(self.input_height * max_ratio)),
        )

    def preprocess_image(self, image: np.ndarray, target_width: int) -> np.ndarray:
        ratio = image.shape[1] / float(image.shape[0])
        resized_width = min(target_width, int(np.ceil(self.input_height * ratio)))
        resized = cv2.resize(
            image,
            (resized_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255
        chw = (chw - 0.5) / 0.5
        output = np.zeros((3, self.input_height, target_width), np.float32)
        output[:, :, :resized_width] = chw
        return output
