import cv2
import numpy as np

from tracking.court import COURT_LENGTH_M, COURT_PATHS, COURT_WIDTH_M
from render.primitives import color


class CourtMiniMap:
    def __init__(self, frame_width, frame_height):
        self.frame_scale = frame_height / 720.0
        self.margin = max(1, round(9 * self.frame_scale))
        requested = round(frame_height * 0.19 * COURT_LENGTH_M / COURT_WIDTH_M)
        self.width = min(
            max(1, requested),
            max(1, frame_width - 2 * self.margin),
        )
        self.height = max(1, round(self.width * COURT_WIDTH_M / COURT_LENGTH_M))
        self.pad = max(1, round(self.width * 0.035))
        self.marker_radius = max(1, round(self.width * 0.024))
        self.text_scale = 0.32 * self.frame_scale
        self.line_thickness = max(1, round(self.frame_scale))
        self.marker_thickness = max(1, round(2 * self.frame_scale))
        self.scale = np.asarray((self.width - 2 * self.pad, self.height - 2 * self.pad), np.float32)
        self.scale /= (COURT_LENGTH_M, COURT_WIDTH_M)
        self.base = self._build_base()

    def _pixel(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        return np.rint(points * self.scale + self.pad).astype(np.int32)

    def _build_base(self):
        canvas = np.full((self.height, self.width, 3), [29, 36, 47], dtype=np.uint8)
        cv2.polylines(
            canvas,
            [self._pixel(path) for path in COURT_PATHS],
            False,
            [142, 150, 161],
            self.line_thickness,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (0, 0),
            (self.width - 1, self.height - 1),
            [188, 193, 201],
            self.line_thickness,
            cv2.LINE_AA,
        )
        return canvas

    def draw(self, frame, players):
        panel = self.base.copy()
        court_limit = np.asarray((COURT_LENGTH_M, COURT_WIDTH_M), dtype=np.float32)
        for item in players:
            position = item.get("court_xy")
            if position is None:
                continue
            position = np.asarray(position, dtype=np.float32).reshape(2)
            if (
                not np.isfinite(position).all()
                or np.any(position < -2.0)
                or np.any(position > court_limit + 2.0)
            ):
                continue
            center = tuple(self._pixel(np.clip(position, (0, 0), court_limit))[0])
            radius = self.marker_radius
            marker_color = item["color"] if "color" in item else color(item["track_id"])
            marker = tuple(int(value) for value in marker_color)
            if item.get("holder", False):
                cv2.circle(
                    panel,
                    center,
                    radius + round(4 * self.frame_scale),
                    [0, 190, 255],
                    self.marker_thickness,
                    cv2.LINE_AA,
                )
            cv2.circle(panel, center, radius, [235, 238, 242], -1, cv2.LINE_AA)
            cv2.circle(
                panel,
                center,
                max(1, radius - round(2 * self.frame_scale)),
                marker,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                panel,
                center,
                radius,
                [235, 238, 242],
                self.line_thickness,
                cv2.LINE_AA,
            )
            text = str(int(item["track_id"]))
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, self.text_scale, self.line_thickness
            )
            origin = (round(center[0] - text_width / 2), round(center[1] + (text_height - baseline) / 2))
            cv2.putText(
                panel,
                text,
                origin,
                cv2.FONT_HERSHEY_DUPLEX,
                self.text_scale,
                (10, 14, 20),
                self.marker_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                text,
                origin,
                cv2.FONT_HERSHEY_DUPLEX,
                self.text_scale,
                (255, 255, 255),
                self.line_thickness,
                cv2.LINE_AA,
            )
        frame_h, frame_w = frame.shape[:2]
        x0 = max(0, frame_w - self.margin - self.width)
        y0 = self.margin
        x1 = min(frame_w, x0 + self.width)
        y1 = min(frame_h, y0 + self.height)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        panel = panel[: y1 - y0, : x1 - x0]
        cv2.addWeighted(
            panel, 0.94, roi, 1 - 0.94, 0, roi
        )
        cv2.rectangle(
            frame,
            (x0, y0),
            (x1 - 1, y1 - 1),
            [188, 193, 201],
            self.line_thickness,
            cv2.LINE_AA,
        )
