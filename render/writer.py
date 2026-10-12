import subprocess
from pathlib import Path

import numpy as np


class FfmpegVideoWriter:
    """Encode BGR frames to an H.264 video using FFmpeg stdin."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        width, height = size
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(float(fps)),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def is_opened(self) -> bool:
        return self.process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        self.process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))

    def release(self) -> None:
        process = self.process
        if process is None:
            return
        close_error = None
        try:
            try:
                process.stdin.close()
            except BrokenPipeError as exc:
                close_error = exc
            error = process.stderr.read().decode(errors="replace").strip()
        finally:
            try:
                return_code = process.wait()
            finally:
                self.process = None
        if return_code:
            raise RuntimeError(error or f"ffmpeg video encoding failed ({return_code})") from close_error
        if close_error is not None:
            raise close_error


__all__ = ["FfmpegVideoWriter"]
