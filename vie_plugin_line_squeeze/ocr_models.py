"""PP-OCRv5 ONNX text recognition for the line-squeeze scene."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from services.base import BaseCtcRecognitionPipeline
from services.inference import InferenceRunner


def _load_metadata(metadata_path: str) -> dict:
    path = Path(metadata_path)
    if path.is_dir():
        path = path / "inference.yml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid inference metadata: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("inference metadata must be a mapping")
    return data


def _transform(data: dict, name: str) -> dict:
    try:
        operations = data["PreProcess"]["transform_ops"]
    except (KeyError, TypeError) as exc:
        raise ValueError("metadata is missing PreProcess.transform_ops") from exc
    if not isinstance(operations, list):
        raise ValueError("PreProcess.transform_ops must be a list")
    matches = [
        operation[name]
        for operation in operations
        if isinstance(operation, dict) and name in operation
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f"metadata must contain exactly one valid {name}")
    return matches[0]


class LineSqueezeTextRecognizer(BaseCtcRecognitionPipeline):
    """Fixed-width PP-OCRv5 CTC recognizer backed by an injected runner."""

    def __init__(self, metadata_path: str, *, runner: InferenceRunner) -> None:
        data = _load_metadata(metadata_path)
        decode = _transform(data, "DecodeImage")
        resize = _transform(data, "RecResizeImg")
        if (
            decode.get("img_mode") != "BGR"
            or decode.get("channel_first") is not False
        ):
            raise ValueError(
                "DecodeImage requires img_mode='BGR' and channel_first=false"
            )

        image_shape = resize.get("image_shape")
        if (
            not isinstance(image_shape, (list, tuple))
            or len(image_shape) < 3
            or image_shape[0] != 3
            or not all(
                isinstance(value, int) and value > 0
                for value in image_shape[:3]
            )
        ):
            raise ValueError(
                "RecResizeImg.image_shape must start with [3, height, width]"
            )

        try:
            postprocess = data["PostProcess"]
            characters = postprocess["character_dict"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "metadata is missing PostProcess.character_dict"
            ) from exc
        if postprocess.get("name") != "CTCLabelDecode":
            raise ValueError("PostProcess.name must be CTCLabelDecode")
        if (
            not isinstance(characters, list)
            or not characters
            or not all(isinstance(value, str) for value in characters)
        ):
            raise ValueError("character_dict must be a non-empty string list")
        use_space_char = postprocess.get("use_space_char")
        if use_space_char is None:
            model_name = data.get("Global", {}).get("model_name")
            use_space_char = model_name == "en_PP-OCRv5_mobile_rec"
        if use_space_char:
            characters = [*characters, " "]

        if len(runner.input_infos) != 1 or len(runner.output_infos) != 1:
            raise ValueError("recognition model requires one input and one output")
        input_shape = runner.input_infos[0].shape
        if (
            len(input_shape) != 4
            or isinstance(input_shape[0], int)
            or input_shape[1] != 3
            or input_shape[2] != image_shape[1]
            or (
                isinstance(input_shape[3], int)
                and input_shape[3] != image_shape[2]
            )
        ):
            raise ValueError(
                "recognition model input must use dynamic batch and match metadata input shape"
            )
        output_shape = runner.output_infos[0].shape
        if not output_shape or isinstance(output_shape[0], int):
            raise ValueError("recognition model output must use dynamic batch")
        output_classes = output_shape[-1]
        expected_classes = len(characters) + 1
        if (
            isinstance(output_classes, int)
            and output_classes != expected_classes
        ):
            raise ValueError(
                f"recognition output classes must be {expected_classes}"
            )

        self.input_width = image_shape[2]
        super().__init__(
            runner,
            characters,
            input_height=image_shape[1],
            max_width=image_shape[2],
        )

    def _target_width(self, images: Sequence[np.ndarray]) -> int:
        super()._target_width(images)
        return self.input_width

    def preprocess_image(
        self,
        image: np.ndarray,
        target_width: int,
    ) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("recognition image must be a three-channel BGR array")
        ratio = image.shape[1] / float(image.shape[0])
        resized_width = max(
            1,
            min(target_width, int(np.ceil(self.input_height * ratio))),
        )
        resized = cv2.resize(
            image,
            (resized_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255
        chw = (chw - 0.5) / 0.5
        output = np.zeros(
            (3, self.input_height, target_width),
            dtype=np.float32,
        )
        output[:, :, :resized_width] = chw
        return output
