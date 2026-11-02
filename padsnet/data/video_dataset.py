import os
import re
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from .vocal_fold_features import mask_preprocessing


class ContrastiveAugment:
    """对比学习专用数据增强器，生成多样化正样本对"""

    def __init__(self, resize_shape=(224, 224)):
        self.resize_shape = resize_shape
        self.augmentations = [
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(size=resize_shape, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            ]),
            transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(hue=0.1)
            ]),
            transforms.Compose([
                transforms.RandomGrayscale(p=0.3),
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomResizedCrop(size=resize_shape, scale=(0.7, 1.0))
            ])
        ]

    def __call__(self, frame):
        """对单帧图像应用随机增强"""
        frame_pil = transforms.ToPILImage()(frame)
        aug = random.choice(self.augmentations)
        return aug(frame_pil)


class FramesDataset(Dataset):
    """Load 32-frame PADSNet clips for training, validation, or testing."""

    def __init__(self, data_root, mask_root, split_file, mode='train',
                 num_frames_3d_32=32, skip_frames_3d_32=1, slide_step=32,
                 resize_shape=(224, 224), mask_dtype=torch.long,
                 compute_distance_features=True):
        self.data_root = data_root
        self.mask_root = mask_root
        self.mode = mode

        self.num_frames_3d_32 = num_frames_3d_32
        self.skip_frames_3d_32 = skip_frames_3d_32
        self.slide_step = slide_step

        self.resize_shape = resize_shape
        self.mask_dtype = mask_dtype
        self.compute_distance_features = bool(compute_distance_features)
        self._frame_lists = {}

        self.raw_data_list = self._parse_split_file(split_file)

        if self.mode == 'train' or self.mode == 'val':
            self.expanded_data_list = self._expand_data_with_sliding_window()
        else:
            self.expanded_data_list = []

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.contrastive_aug = ContrastiveAugment(resize_shape) if mode == 'train' else None

    def _parse_split_file(self, split_file):
        data_list = []
        with open(split_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # Split files were generated on Windows and contain "class\\patient".
                    # Normalize the relative component so the same files work in WSL/Linux.
                    relative_video_path = parts[0].replace("\\", os.sep).replace("/", os.sep)
                    video_path = os.path.join(self.data_root, relative_video_path)
                    label = int(parts[1])
                    data_list.append((video_path, label))
        return data_list

    def _sample_frames_3d_32(self, total_frames):
        """【训练/验证专用】带帧间隔的滑动窗口采样"""
        if total_frames <= 0:
            raise ValueError("A sample directory contains no readable image frames.")
        frame_indices_list = []
        stride = self.skip_frames_3d_32
        required_span = (self.num_frames_3d_32 - 1) * stride + 1

        if total_frames >= required_span:
            max_start_idx = total_frames - required_span
            start_indices = list(range(0, max_start_idx + 1, self.slide_step))

            for start_idx in start_indices:
                window_indices = [start_idx + i * stride for i in range(self.num_frames_3d_32)]
                frame_indices_list.append(window_indices)
        else:
            # 视频长度不足时，使用乒乓反射补齐到 required_span 长度
            sequence = []
            while len(sequence) < required_span:
                sequence.extend(list(range(total_frames)))
                if len(sequence) < required_span and total_frames > 1:
                    sequence.extend(list(range(total_frames - 2, 0, -1)))

            window_indices = [sequence[i * stride] for i in range(self.num_frames_3d_32)]
            frame_indices_list.append(window_indices)

        return frame_indices_list

    def _sample_frames_3d_32_dense(self, total_frames):
        """【测试专用】带帧间隔的密集采样，返回全视频覆盖的切片"""
        if total_frames <= 0:
            raise ValueError("A sample directory contains no readable image frames.")
        frame_indices_list = []
        stride = self.skip_frames_3d_32
        required_span = (self.num_frames_3d_32 - 1) * stride + 1

        if total_frames >= required_span:
            max_start_idx = total_frames - required_span
            start_indices = list(range(0, max_start_idx + 1, self.slide_step))

            # 确保包含最后一段，覆盖视频末尾动态
            if not start_indices or start_indices[-1] != max_start_idx:
                start_indices.append(max_start_idx)

            for start_idx in start_indices:
                window_indices = [start_idx + i * stride for i in range(self.num_frames_3d_32)]
                frame_indices_list.append(window_indices)
        else:
            sequence = []
            while len(sequence) < required_span:
                sequence.extend(list(range(total_frames)))
                if len(sequence) < required_span and total_frames > 1:
                    sequence.extend(list(range(total_frames - 2, 0, -1)))

            window_indices = [sequence[i * stride] for i in range(self.num_frames_3d_32)]
            frame_indices_list.append(window_indices)

        return frame_indices_list

    def _get_sorted_frame_list(self, video_path):
        """Cache directory metadata; many sliding clips reuse the same video."""
        frames_list = self._frame_lists.get(video_path)
        if frames_list is None:
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            frames_list = [
                name for name in os.listdir(video_path)
                if name.lower().endswith(valid_extensions) and re.search(r'\d+', name)
            ]
            frames_list.sort(key=lambda x: int(re.search(r'\d+', x).group()))
            if not frames_list:
                raise ValueError(f"No numbered image frames found in: {video_path}")
            self._frame_lists[video_path] = frames_list
        return frames_list

    def _expand_data_with_sliding_window(self):
        expanded_list = []
        for video_path, label in self.raw_data_list:
            frames_list = self._get_sorted_frame_list(video_path)
            total_frames = len(frames_list)

            sub_sample_indices_list = self._sample_frames_3d_32(total_frames)

            for sub_indices in sub_sample_indices_list:
                expanded_list.append((video_path, label, sub_indices))

        print(f"原始视频数: {len(self.raw_data_list)} → 滑动窗口后子切片数: {len(expanded_list)}")
        return expanded_list

    def _load_frames(self, frames_list, frame_indices, frames_path, mask_root):
        frames = []
        masks = []
        seq_distance_features = []
        for idx in frame_indices:
            if 0 <= idx < len(frames_list):
                frame_name = frames_list[idx]
                frame_path = os.path.join(frames_path, frame_name)
                mask_path = os.path.join(mask_root, frame_name)

                frame = cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = cv2.resize(frame, self.resize_shape)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame = np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8)
                frames.append(frame)

                # Load the mask. Distance features are optional because the
                # visual-only models never consume them; skipping the contour
                # scan avoids substantial CPU work for every frame.
                if os.path.exists(mask_path):
                    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if self.compute_distance_features:
                        _, _, feature_vector = mask_preprocessing(mask)
                    else:
                        feature_vector = np.zeros(8, dtype=np.float32)
                    mask_input = cv2.resize(mask, self.resize_shape)
                else:
                    mask_input = np.zeros(self.resize_shape, dtype=np.uint8)
                    feature_vector = np.zeros(8, dtype=np.float32)

                seq_distance_features.append(feature_vector)
                masks.append(mask_input)
            else:
                # Padding logic
                if frames:
                    frames.append(frames[-1].copy())
                    masks.append(masks[-1].copy())
                    seq_distance_features.append(seq_distance_features[-1].copy())
                else:
                    frames.append(np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8))
                    masks.append(np.zeros(self.resize_shape, dtype=np.uint8))
                    seq_distance_features.append(np.zeros(8, dtype=np.float32))

        return frames, masks, seq_distance_features

    def _process_single_clip(self, frames_path, label, frame_indices_3d_32):
        """Load and transform one 32-frame clip."""
        frames_list = self._get_sorted_frame_list(frames_path)

        frames_3d_32, masks_3d_32, seq_distance_features = self._load_frames(
            frames_list, frame_indices_3d_32, frames_path, self.mask_root
        )

        processed_frames_3d_32 = []
        for frame in frames_3d_32:
            if self.mode == 'train' and self.contrastive_aug is not None:
                frame_aug = self.contrastive_aug(frame)
                processed_frames_3d_32.append(self.base_transform(frame_aug))
            else:
                processed_frames_3d_32.append(self.base_transform(transforms.ToPILImage()(frame)))

        return {
            'input_3d_32': torch.stack(processed_frames_3d_32, dim=1),
            'mask_3d_32': torch.tensor(np.array(masks_3d_32), dtype=self.mask_dtype),
            'label': torch.tensor(label, dtype=torch.long),
            'distance_features': torch.from_numpy(np.array(seq_distance_features, dtype=np.float32)),
            'video_path': frames_path
        }

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            frames_path, label, frame_indices_3d_32 = self.expanded_data_list[idx]
            return self._process_single_clip(frames_path, label, frame_indices_3d_32)

        frames_path, label = self.raw_data_list[idx]
        frames_list = self._get_sorted_frame_list(frames_path)
        total_frames = len(frames_list)

        all_indices = self._sample_frames_3d_32_dense(total_frames)

        all_input_3d = []
        all_mask_3d = []
        all_dist_feats = []

        for indices in all_indices:
            clip_data = self._process_single_clip(frames_path, label, indices)
            all_input_3d.append(clip_data['input_3d_32'])
            all_mask_3d.append(clip_data['mask_3d_32'])
            all_dist_feats.append(clip_data['distance_features'])

        return {
            'input_3d_32': torch.stack(all_input_3d),
            'mask_3d_32': torch.stack(all_mask_3d),
            'distance_features': torch.stack(all_dist_feats),
            'label': torch.tensor(label, dtype=torch.long),
            'video_path': frames_path
        }

    def __len__(self):
        if self.mode == 'train' or self.mode == 'val':
            return len(self.expanded_data_list)
        return len(self.raw_data_list)
