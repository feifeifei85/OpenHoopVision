"""Court keypoint model adapter."""

from concurrent.futures import Future
from typing import Any

import numpy as np

from models.inference import _YoloModelAdapter
from tracking.court import COURT_KEYPOINT_COUNT


def detect_court_keypoints(
    model: Any,
    frame: np.ndarray,
    image_size: int,
    device: int | str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the best court-keypoint prediction in image coordinates."""
    result = model.predict(frame, imgsz=image_size, conf=0.01, device=device, verbose=False)[0]
    if result.keypoints is None:
        return None, None
    keypoints = result.keypoints.data.detach().cpu().numpy()
    if keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return None, None
    if keypoints.shape[1] != COURT_KEYPOINT_COUNT:
        raise RuntimeError(
            f"court model must output exactly {COURT_KEYPOINT_COUNT} keypoints, "
            f"got {keypoints.shape[1]}"
        )
    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes):
        instance_scores = result.boxes.conf.detach().cpu().numpy().reshape(-1)[: len(keypoints)]
    else:
        instance_scores = (
            keypoints[:, :, 2].mean(axis=1)
            if keypoints.shape[2] >= 3
            else np.zeros(len(keypoints))
        )
    best_index = int(np.argmax(instance_scores))
    scores = (
        keypoints[best_index, :, 2]
        if keypoints.shape[2] >= 3
        else np.ones(COURT_KEYPOINT_COUNT, dtype=np.float32)
    )
    return keypoints[best_index, :, :2].astype(np.float32), scores.astype(np.float32)


class CourtPoseDetector(_YoloModelAdapter):
    """Load the court pose model and delegate frame inference to its executor."""

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        return detect_court_keypoints(self.model, frame, self.image_size, self.device)

    def result(
        self,
        future: Future[Any] | None,
        frame: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.inference.result(future, frame)


__all__ = ["CourtPoseDetector", "detect_court_keypoints"]
