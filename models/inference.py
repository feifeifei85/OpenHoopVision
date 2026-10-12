from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any


class CudaInferenceExecutor:
    """Run model inference on a dedicated CUDA stream when CUDA is available."""

    def __init__(self, device: int | str, infer: Callable[[Any], Any]) -> None:
        self.device = device
        self.infer = infer
        self.executor: ThreadPoolExecutor | None = None
        self.stream: Any | None = None
        self.torch: Any | None = None

    def start(self) -> None:
        import torch

        self.torch = torch
        if torch.cuda.is_available():
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.stream = torch.cuda.Stream(device=torch.device(f"cuda:{self.device}"))

    def submit(self, frame: Any) -> Future[Any] | None:
        if self.executor is None:
            return None
        return self.executor.submit(self._infer_on_stream, frame)

    def result(self, future: Future[Any] | None, frame: Any) -> Any:
        return future.result() if future is not None else self.infer(frame)

    def _infer_on_stream(self, frame: Any) -> Any:
        if self.stream is None:
            return self.infer(frame)
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.stream):
            result = self.infer(frame)
        self.stream.synchronize()
        return result

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


class _YoloModelAdapter:
    """Share YOLO loading and inference lifecycle across pose-model adapters."""

    def __init__(self, checkpoint: str | Path, image_size: int, device: int | str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint))
        self.image_size = image_size
        self.device = device
        self.inference = CudaInferenceExecutor(device, self.detect)

    def detect(self, frame: Any) -> Any:
        raise NotImplementedError

    def start(self) -> None:
        self.inference.start()

    def submit(self, frame: Any) -> Future[Any] | None:
        return self.inference.submit(frame)

    def result(self, future: Future[Any] | None, frame: Any) -> Any:
        return self.inference.result(future, frame)

    def close(self) -> None:
        self.inference.close()


__all__ = ["CudaInferenceExecutor"]
