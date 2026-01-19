"""Video analysis for GMFM-66 hard facts using MediaPipe Pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseFrame:
    """Container for pose landmarks and frame index."""

    frame_idx: int
    landmarks: Dict[str, Tuple[float, float, float]]


class MovementAnalyzer:
    """Analyze movement events from video using MediaPipe Pose."""

    def __init__(
        self,
        frame_stride: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.frame_stride = max(1, frame_stride)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmark_names = {
            "left_ankle": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            "right_ankle": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            "left_knee": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            "right_knee": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
            "left_hip": mp.solutions.pose.PoseLandmark.LEFT_HIP,
            "right_hip": mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            "left_wrist": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            "right_wrist": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        }

    def _iter_pose_frames(self, video_path: str) -> Iterable[PoseFrame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % self.frame_stride != 0:
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.pose.process(rgb)
                if result.pose_landmarks:
                    landmarks = {}
                    for name, idx in self._landmark_names.items():
                        lm = result.pose_landmarks.landmark[idx]
                        landmarks[name] = (lm.x, lm.y, lm.z)
                    yield PoseFrame(frame_idx=frame_idx, landmarks=landmarks)
                frame_idx += 1
        finally:
            cap.release()

    @staticmethod
    def _knee_angle(
        hip: Tuple[float, float, float],
        knee: Tuple[float, float, float],
        ankle: Tuple[float, float, float],
    ) -> float:
        hip_v = np.array(hip[:2]) - np.array(knee[:2])
        ankle_v = np.array(ankle[:2]) - np.array(knee[:2])
        hip_norm = np.linalg.norm(hip_v)
        ankle_norm = np.linalg.norm(ankle_v)
        if hip_norm == 0 or ankle_norm == 0:
            return 0.0
        cos_angle = np.clip(
            np.dot(hip_v, ankle_v) / (hip_norm * ankle_norm), -1.0, 1.0
        )
        return float(np.degrees(np.arccos(cos_angle)))

    def detect_flight_phase(self, video_path: str) -> Dict[str, float | bool]:
        """Detect flight phase (both feet off ground) using ankle height."""
        ground_level = 0.0
        frames_with_pose = 0
        flight_frames = 0
        flight_threshold = 0.05  # normalized y-distance above ground

        for frame in self._iter_pose_frames(video_path):
            frames_with_pose += 1
            left_ankle = frame.landmarks["left_ankle"][1]
            right_ankle = frame.landmarks["right_ankle"][1]
            ground_level = max(ground_level, left_ankle, right_ankle)

            if (
                left_ankle < ground_level - flight_threshold
                and right_ankle < ground_level - flight_threshold
            ):
                flight_frames += 1

        if frames_with_pose == 0:
            raise ValueError("No person detected in video for flight phase analysis.")

        confidence = flight_frames / frames_with_pose
        return {
            "flight_phase_detected": flight_frames > 0,
            "flight_phase_confidence": float(confidence),
        }

    def detect_hand_support(self, video_path: str) -> Dict[str, float | bool | str]:
        """Detect hand support via wrist proximity to knees and low wrist motion."""
        frames_with_pose = 0
        support_frames = 0
        knee_support_frames = 0
        wrist_positions: List[Tuple[np.ndarray, np.ndarray]] = []
        hip_positions: List[np.ndarray] = []

        for frame in self._iter_pose_frames(video_path):
            frames_with_pose += 1
            left_wrist = np.array(frame.landmarks["left_wrist"][:2])
            right_wrist = np.array(frame.landmarks["right_wrist"][:2])
            left_knee = np.array(frame.landmarks["left_knee"][:2])
            right_knee = np.array(frame.landmarks["right_knee"][:2])
            left_hip = np.array(frame.landmarks["left_hip"][:2])
            right_hip = np.array(frame.landmarks["right_hip"][:2])

            wrist_positions.append((left_wrist, right_wrist))
            hip_positions.append((left_hip + right_hip) / 2)

            if len(wrist_positions) < 2:
                continue

            prev_left, prev_right = wrist_positions[-2]
            wrist_velocity = (
                np.linalg.norm(left_wrist - prev_left)
                + np.linalg.norm(right_wrist - prev_right)
            )

            knee_distance = min(
                np.linalg.norm(left_wrist - left_knee),
                np.linalg.norm(right_wrist - right_knee),
            )

            hips_upward = hip_positions[-1][1] < hip_positions[-2][1]

            if wrist_velocity < 0.02 and hips_upward:
                support_frames += 1
                if knee_distance < 0.08:
                    knee_support_frames += 1

        if frames_with_pose == 0:
            raise ValueError("No person detected in video for hand support analysis.")

        support_confidence = support_frames / frames_with_pose
        support_detected = support_frames > 0
        support_mode = "knees" if knee_support_frames > 0 else "unknown_support"

        return {
            "hand_support_detected": support_detected,
            "hand_support_confidence": float(support_confidence),
            "hand_support_mode": support_mode,
        }

    def detect_symmetry(self, video_path: str) -> Dict[str, float | bool]:
        """Estimate movement symmetry using left/right ankle trajectories."""
        frames_with_pose = 0
        diffs: List[float] = []

        for frame in self._iter_pose_frames(video_path):
            frames_with_pose += 1
            left_ankle = np.array(frame.landmarks["left_ankle"][:2])
            right_ankle = np.array(frame.landmarks["right_ankle"][:2])
            diffs.append(float(np.linalg.norm(left_ankle - right_ankle)))

        if frames_with_pose == 0:
            raise ValueError("No person detected in video for symmetry analysis.")

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        symmetry_threshold = 0.12  # normalized distance threshold
        return {
            "symmetry_good": mean_diff < symmetry_threshold,
            "symmetry_score": float(max(0.0, 1.0 - mean_diff / symmetry_threshold)),
        }

    def detect_trunk_stability(self, video_path: str) -> Dict[str, float | bool]:
        """Estimate trunk stability using hip center sway."""
        frames_with_pose = 0
        hip_centers: List[np.ndarray] = []

        for frame in self._iter_pose_frames(video_path):
            frames_with_pose += 1
            left_hip = np.array(frame.landmarks["left_hip"][:2])
            right_hip = np.array(frame.landmarks["right_hip"][:2])
            hip_centers.append((left_hip + right_hip) / 2)

        if frames_with_pose == 0:
            raise ValueError("No person detected in video for trunk stability analysis.")

        hip_array = np.vstack(hip_centers)
        sway = float(np.std(hip_array[:, 0]))
        sway_threshold = 0.05
        return {
            "trunk_stability_good": sway < sway_threshold,
            "trunk_sway": sway,
        }

    def analyze_video(self, video_path: str) -> Dict[str, float | bool | str]:
        """Run a set of common checks and return a summary dict."""
        pose_frames = list(self._iter_pose_frames(video_path))
        if not pose_frames:
            raise ValueError("No person detected in video for analysis.")

        max_knee_angle = 0.0
        for frame in pose_frames:
            left_angle = self._knee_angle(
                frame.landmarks["left_hip"],
                frame.landmarks["left_knee"],
                frame.landmarks["left_ankle"],
            )
            right_angle = self._knee_angle(
                frame.landmarks["right_hip"],
                frame.landmarks["right_knee"],
                frame.landmarks["right_ankle"],
            )
            max_knee_angle = max(max_knee_angle, left_angle, right_angle)

        flight_info = self.detect_flight_phase(video_path)
        hand_support_info = self.detect_hand_support(video_path)
        symmetry_info = self.detect_symmetry(video_path)
        stability_info = self.detect_trunk_stability(video_path)

        return {
            **flight_info,
            **hand_support_info,
            **symmetry_info,
            **stability_info,
            "max_knee_angle": float(max_knee_angle),
        }
