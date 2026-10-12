"""Player pose model adapter."""

from concurrent.futures import Future
from typing import Any

import numpy as np

from models.inference import _YoloModelAdapter


def detect_pose(
    model: Any,
    frame: np.ndarray,
    image_size: int,
    device: int | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return person boxes and their keypoints from a pose-model result."""
    result = model.predict(
        frame,
        imgsz=image_size,
        conf=0.25,
        device=device,
        verbose=False,
    )[0]
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 17, 2), dtype=np.float32),
            np.empty((0, 17), dtype=np.float32),
        )
    person = result.boxes.cls.detach().cpu().numpy().astype(np.int32) == 0
    boxes = result.boxes.xyxy.detach().cpu().numpy()[person]
    box_scores = result.boxes.conf.detach().cpu().numpy()[person]
    keypoints = result.keypoints.data.detach().cpu().numpy()[person]
    return boxes, box_scores, keypoints[:, :, :2], keypoints[:, :, 2]


class PlayerPoseDetector(_YoloModelAdapter):
    """Load the player pose model and delegate frame inference to its executor."""

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return detect_pose(self.model, frame, self.image_size, self.device)

    def result(
        self,
        future: Future[Any] | None,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.inference.result(future, frame)


__all__ = ["PlayerPoseDetector", "detect_pose"]
