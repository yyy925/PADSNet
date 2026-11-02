import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from padsnet.models import LaryngealRecognitionModel
from padsnet.data import FramesDataset


def test_model_multiview(data_root, mask_root, split_file, model_path, num_classes=3,
                         save_dir=None, class_names=None, top_k=5, slide_step=32,
                         num_workers=4, mini_batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    test_dataset = FramesDataset(
        data_root=data_root,
        mask_root=mask_root,
        split_file=split_file,
        mode='test',
        slide_step=slide_step
    )
    print(f"Test samples (Videos): {len(test_dataset)}")

    # Each sample can contain a different number of clips.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = LaryngealRecognitionModel(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_sample_names = []

    print(f"Start Multi-view Testing... Fusion: class-wise top-{top_k} mean")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # [1, N, C, T, H, W] -> [N, C, T, H, W]
            video_clips = batch['input_3d_32'].squeeze(0).to(device)
            mask_clips = batch['mask_3d_32'].squeeze(0).to(device)
            dist_clips = batch['distance_features'].squeeze(0).to(device)
            label = batch['label'].item()
            video_path = batch['video_path'][0]
            sample_name = os.path.basename(video_path)

            n_clips = video_clips.shape[0]

            clip_outputs = []

            for k in range(0, n_clips, mini_batch_size):
                end_k = min(k + mini_batch_size, n_clips)

                sub_video = video_clips[k:end_k]
                sub_mask = mask_clips[k:end_k]
                sub_dist = dist_clips[k:end_k]

                outputs = model(sub_video, sub_mask, sub_dist)
                probs = F.softmax(outputs, dim=1)
                clip_outputs.append(probs)

            all_clip_probs = torch.cat(clip_outputs, dim=0)

            # Class-wise top-k mean pooling.
            k_top = min(top_k, all_clip_probs.shape[0])
            topk_scores = torch.topk(all_clip_probs, k=k_top, dim=0).values
            patient_score = torch.mean(topk_scores, dim=0)
            patient_prob = patient_score / patient_score.sum().clamp_min(1e-12)
            pred_label = torch.argmax(patient_prob).item()

            all_preds.append(pred_label)
            all_labels.append(label)
            all_probs.append(patient_prob.cpu().numpy())
            all_sample_names.append(sample_name)

            if i % 10 == 0:
                print(
                    f"Processed {i}/{len(test_dataset)}: {sample_name} "
                    f"(Clips: {n_clips}, top-k: {k_top}) -> Pred: {pred_label}, True: {label}"
                )

    accuracy = accuracy_score(all_labels, all_preds)
    label_ids = list(range(num_classes))
    class_names = class_names or [f"class-{index}" for index in label_ids]
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)
    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        digits=4,
        target_names=class_names,
        zero_division=0,
    )

    print(f"\n===== Multi-view Testing Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        result_columns = {
            'Sample': all_sample_names,
            'Label': all_labels,
            'Prediction': all_preds,
            'Correct': [p == l for p, l in zip(all_preds, all_labels)],
        }
        for class_index in range(num_classes):
            result_columns[f'Prob_Class{class_index}'] = [p[class_index] for p in all_probs]
        results_df = pd.DataFrame(result_columns)
        results_df.to_excel(os.path.join(save_dir, "multiview_results.xlsx"), index=False)

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            vmin=0,
            vmax=max(1, int(cm.max())),
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={"size": 16}
        )
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        plt.ylabel('True Label', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=18)
        plt.savefig(os.path.join(save_dir, "multiview_cm.png"), dpi=300, bbox_inches='tight')
        plt.close()

        with open(os.path.join(save_dir, "multiview_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"Fusion: class-wise top-{top_k} mean\n\n")
            f.write(report)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PADSNet with class-wise top-k clip fusion.")
    parser.add_argument("--data-root", default="data/frames/test")
    parser.add_argument("--mask-root", default="data/masks")
    parser.add_argument("--test-split", default="splits/test.txt")
    parser.add_argument("--checkpoint", default="weights/padsnet_model_epoch_100.pth")
    parser.add_argument("--output-dir", default="outputs/evaluation/padsnet")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["high-risk", "low-risk", "normal"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--slide-step", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if len(args.class_names) != args.num_classes:
        raise ValueError("--class-names must contain exactly --num-classes values.")
    test_model_multiview(
        data_root=args.data_root,
        mask_root=args.mask_root,
        split_file=args.test_split,
        model_path=args.checkpoint,
        num_classes=args.num_classes,
        save_dir=args.output_dir,
        class_names=args.class_names,
        top_k=args.top_k,
        slide_step=args.slide_step,
        num_workers=args.workers,
        mini_batch_size=args.mini_batch_size,
    )
