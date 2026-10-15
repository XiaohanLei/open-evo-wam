"""Unchanged dense native camera layout from robotwin_dense_contract.py."""

from typing import Any
import numpy as np
import cv2


def compose_views(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    panels = (
        cv2.resize(head, (640, 240), interpolation=cv2.INTER_CUBIC),
        np.concatenate(
            (
                cv2.resize(left, (320, 120), interpolation=cv2.INTER_CUBIC),
                cv2.resize(right, (320, 120), interpolation=cv2.INTER_CUBIC),
            ),
            axis=1,
        ),
    )
    return np.ascontiguousarray(np.concatenate(panels, axis=0), dtype=np.uint8)


def observation_image(observation: dict[str, Any]) -> np.ndarray:
    cameras = observation["observation"]
    return compose_views(
        *(
            np.asarray(cameras[name]["rgb"])[..., :3]
            for name in ("head_camera", "left_camera", "right_camera")
        )
    )
