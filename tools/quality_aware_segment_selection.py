import argparse
import os
import cv2
import re
import shutil
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_closing
from scipy.signal import find_peaks

from padsnet.data.vocal_fold_features import mask_preprocessing


def cv_imread_chinese(file_path, flags=cv2.IMREAD_GRAYSCALE):
    """Read image paths containing non-ASCII characters."""
    if not os.path.exists(file_path):
        return None
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)


def calculate_periodicity(signal):
    if len(signal) < 10:
        return 0.0
    norm_sig = signal - np.mean(signal)
    var = np.var(signal)
    if var < 1e-5:
        return 0.0
    autocorr = np.correlate(norm_sig, norm_sig, mode='full')
    autocorr = autocorr[len(autocorr) // 2:] / (var * len(signal))
    peaks, _ = find_peaks(autocorr[2:], distance=3)
    if len(peaks) == 0:
        return 0.0
    return max(0.0, autocorr[peaks[0] + 2])


def filter_and_extract_segments(all_distances, useless_flags):
    N = len(all_distances)
    all_widths = np.sum(all_distances, axis=1)

    baseline_width = np.median(np.sort(all_widths)[:N // 3])
    dynamic_range = np.percentile(all_widths, 90) - np.percentile(all_widths, 10)
    open_threshold = np.percentile(all_widths, 85) if dynamic_range < 5 else baseline_width + dynamic_range * 0.6

    is_open = all_widths > open_threshold
    is_negative = np.any(all_distances < 0, axis=1)
    is_invalid = is_open | is_negative | useless_flags
    is_valid_smoothed = binary_closing(~is_invalid, structure=np.ones(3))

    segments = []
    start = -1
    for i in range(N):
        if is_valid_smoothed[i] and start == -1:
            start = i
        elif not is_valid_smoothed[i] and start != -1:
            segments.append((start, i - 1))
            start = -1
    if start != -1:
        segments.append((start, N - 1))

    valid_segments = [seg for seg in segments if (seg[1] - seg[0] + 1) >= 15]
    return valid_segments, all_widths


def get_best_vibration_segment(all_distances, all_blur_scores, useless_flags):
    segments, all_widths = filter_and_extract_segments(all_distances, useless_flags)
    if not segments:
        return None

    max_len = max(seg[1] - seg[0] + 1 for seg in segments)
    max_blur = max(all_blur_scores) if max(all_blur_scores) > 0 else 1.0
    scores = []

    for start, end in segments:
        length = end - start + 1
        score_len = length / max_len
        score_blur = np.mean(all_blur_scores[start:end + 1]) / max_blur
        score_period = calculate_periodicity(all_widths[start:end + 1])

        mean_dL, mean_dR = np.mean(all_distances[start:end + 1, 0]), np.mean(all_distances[start:end + 1, 1])
        score_sym = 1.0 - (abs(mean_dL - mean_dR) / (mean_dL + mean_dR + 1e-5))

        total_score = (0.35 * score_len) + (0.30 * score_blur) + (0.25 * score_period) + (0.10 * score_sym)
        scores.append(total_score)

    return segments[np.argmax(scores)]


def extract_frame_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def extract_distance_features(mask):
    """Return the left/right mean distances from a mask."""
    _, _, feature_vector = mask_preprocessing(mask, visualize=False)
    dL = feature_vector[0]
    dR = feature_vector[3]
    return float(dL), float(dR)


def process_all_data(data_root, mask_root, output_root):
    """Optionally select one quality-aware vibration segment per sample."""
    os.makedirs(output_root, exist_ok=True)

    if not os.path.exists(data_root):
        print(f"错误: 找不到帧数据根目录 {data_root}")
        return

    categories = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    print(">> 开始提取【完美发声段】(算法: 中线距离特征 + 病理感知提取)...")

    for category in categories:
        category_path = os.path.join(data_root, category)
        save_category_path = os.path.join(output_root, category)
        os.makedirs(save_category_path, exist_ok=True)

        patients = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]

        for patient in tqdm(patients, desc=f"处理类别: {category}"):
            patient_path = os.path.join(category_path, patient)
            save_patient_path = os.path.join(save_category_path, patient)

            frame_files = [f for f in os.listdir(patient_path) if f.lower().endswith(('.jpg', '.png'))]
            if not frame_files:
                continue

            frame_files.sort(key=extract_frame_number)

            all_distances = []
            all_blur_scores = []
            useless_flags = []
            analyzed_frame_files = []

            for frame_file in frame_files:
                frame_path = os.path.join(patient_path, frame_file)

                mask_path = os.path.join(mask_root, frame_file)

                frame_img = cv_imread_chinese(frame_path, cv2.IMREAD_GRAYSCALE)
                if frame_img is None:
                    continue
                analyzed_frame_files.append(frame_file)

                blur_score = cv2.Laplacian(frame_img, cv2.CV_64F).var()
                all_blur_scores.append(blur_score)

                useless = False
                dL, dR = 0.0, 0.0
                mask_img = None

                if os.path.exists(mask_path):
                    mask_img = cv_imread_chinese(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is None or np.sum(mask_img > 0) < 50:
                        useless = True
                    else:
                        dL, dR = extract_distance_features(mask_img)
                        if dL == 0.0 and dR == 0.0:
                            useless = True
                else:
                    useless = True

                useless_flags.append(useless)
                all_distances.append([dL, dR])

            if len(all_distances) == 0:
                continue

            all_distances = np.array(all_distances)
            all_blur_scores = np.array(all_blur_scores)
            useless_flags = np.array(useless_flags)

            best_seg = get_best_vibration_segment(all_distances, all_blur_scores, useless_flags)

            if best_seg:
                start_idx, end_idx = best_seg
                os.makedirs(save_patient_path, exist_ok=True)

                for i in range(start_idx, end_idx + 1):
                    src_frame = os.path.join(patient_path, analyzed_frame_files[i])
                    dst_frame = os.path.join(save_patient_path, analyzed_frame_files[i])
                    shutil.copy2(src_frame, dst_frame)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optional quality-aware vibration-segment selection. This is not required by PADSNet training."
    )
    parser.add_argument("--data-root", default="data/frames/train")
    parser.add_argument("--mask-root", default="data/masks")
    parser.add_argument("--output-root", default="outputs/selected_segments")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_data(args.data_root, args.mask_root, args.output_root)
    print("✨ 所有数据处理完毕！")
