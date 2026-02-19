"""
Data processing utilities for video action recognition.
Wraps frame extraction, dataframe building, and PyTorch Dataset.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import torch  # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset  # pyright: ignore[reportMissingImports]


class VideoDataProcessor:
    """Handles video paths, frame extraction, and training array preparation."""

    def __init__(
        self,
        directory: str,
        normal_directory: str,
        frame_size: int = 64,
        num_frames: int = 40,
    ):
        self.directory = directory
        self.normal_directory = normal_directory
        self.frame_size = frame_size
        self.num_frames = num_frames
        self._dataset_path: List[str] = []
        self._label_types: List[str] = []
        self._label_types_normal: List[str] = []
        self._selected_classes: List[str] = []
        self._train_df: Optional[pd.DataFrame] = None

    def build_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame of (video_path, label) from abnormal and normal directories."""
        self._dataset_path = os.listdir(self.directory)
        self._label_types = os.listdir(self.directory)
        self._label_types_normal = os.listdir(self.normal_directory)
        self._selected_classes = self._label_types + self._label_types_normal

        rooms = []
        for item in self._dataset_path:
            all_rooms = os.listdir(self.directory + "/" + item)
            for room in all_rooms:
                rooms.append((str(self.directory + "/" + item) + "/" + room, item))
        for item in self._label_types_normal:
            all_rooms = os.listdir(self.normal_directory + "/" + item)
            for room in all_rooms:
                rooms.append(
                    (str(self.normal_directory + "/" + item) + "/" + room, item)
                )
        self._train_df = pd.DataFrame(
            {"video_path": [r[0] for r in rooms], "label": [r[1] for r in rooms]}
        )
        return self._train_df

    @property
    def selected_classes(self) -> List[str]:
        """Label list (abnormal + normal). Call build_dataframe() first."""
        if not self._selected_classes:
            self.build_dataframe()
        return self._selected_classes

    @property
    def train_df(self) -> pd.DataFrame:
        """DataFrame of video paths and labels. Call build_dataframe() first."""
        if self._train_df is None:
            self.build_dataframe()
        return self._train_df  # type: ignore[return-value]

    def extract_frames(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
    ) -> np.ndarray:
        """Extract uniformly sampled frames, resized to frame_size, normalized to [0,1]."""
        n = num_frames if num_frames is not None else self.num_frames
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // n, 1)
        count = 0
        while len(frames) < n and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            count += 1
        cap.release()
        while len(frames) < n:
            frames.append(frames[-1])
        return np.array(frames, dtype=np.float32)

    def extract_original_frames(
        self,
        video_path: str,
        num_frames: Optional[int] = None,
    ) -> np.ndarray:
        """Extract uniformly sampled frames at original resolution, normalized to [0,1]."""
        n = num_frames if num_frames is not None else self.num_frames
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(total_frames // n, 1)
        count = 0
        while len(frames) < n and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            count += 1
        cap.release()
        while len(frames) < n:
            frames.append(frames[-1])
        return np.array(frames, dtype=np.float32)

    def extract_frames_for_sliding(
        self,
        video_path: str,
        frame_size: Optional[int] = None,
        max_frames: int = 40,
    ) -> np.ndarray:
        """Extract consecutive frames (one per step) for sliding-window prediction."""
        fs = frame_size if frame_size is not None else self.frame_size
        frames = []
        cap = cv2.VideoCapture(video_path)
        count = 0
        while count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (fs, fs))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            count += 1
        cap.release()
        if not frames:
            return np.zeros((0, fs, fs, 3), dtype=np.float32)
        return np.array(frames, dtype=np.float32)

    def extract_frames_for_sliding_original(
        self,
        video_path: str,
        max_frames: int = 40,
    ) -> np.ndarray:
        """Extract consecutive frames at original resolution (no resizing) for display.
        Same frame count and order as extract_frames_for_sliding, so indices align."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        count = 0
        while count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            count += 1
        cap.release()
        if not frames:
            return np.array(frames, dtype=np.float32)
        return np.array(frames, dtype=np.float32)

    def load_training_arrays(
        self,
        videos_per_class: int = 30,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """
        Load X, y from directories, shuffle and split.
        Returns X_train, X_test, y_train, y_test.
        """
        classes = self.selected_classes
        X: List[np.ndarray] = []
        y: List[int] = []
        for class_index, cls in enumerate(classes):
            if cls == "Normal Videos":
                class_path = os.path.join(self.normal_directory, cls)
                videos = os.listdir(class_path)[:videos_per_class]
            else:
                class_path = os.path.join(self.directory, cls)
                videos = os.listdir(class_path)[:videos_per_class]
            for video in videos:
                video_path = os.path.join(class_path, video)
                frames = self.extract_frames(video_path, num_frames=self.num_frames)
                if frames.shape == (self.num_frames, self.frame_size, self.frame_size, 3):
                    X.append(frames)
                    y.append(class_index)

        X_arr = np.array(X)
        y_arr = np.array(y, dtype=np.int64)
        idx = np.random.RandomState(random_state).permutation(len(y_arr))
        X_arr, y_arr = X_arr[idx], y_arr[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=random_state, shuffle=True
        )
        return X_train, X_test, y_train, y_test


class VideoDataset(Dataset):
    """PyTorch Dataset for video tensors (N, T, H, W, C) -> (N, T, C, H, W)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, T, H, W, C) numpy
        self.X = torch.from_numpy(X).float().permute(0, 1, 4, 2, 3)  # (N, T, C, H, W)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
