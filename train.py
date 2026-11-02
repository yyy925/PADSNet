import argparse
import os
import cv2

from padsnet.models import LaryngealRecognitionModel
from padsnet.data import FramesDataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm


def load_pretrained_weights(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        print(f"⚠️ 警告: 预训练权重文件未找到: {pretrained_path}")
        return model

    print(f">> 正在加载预训练权重: {pretrained_path}")
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return model

    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = {}
    matched_count = 0

    for k, v in pretrained_dict.items():
        clean_k = k.replace('backbone.', '').replace('module.', '')
        target_k = f"backbone.{clean_k}"

        if target_k in model_dict:
            if model_dict[target_k].shape == v.shape:
                new_state_dict[target_k] = v
                matched_count += 1
            elif "patch_embed.proj.weight" in clean_k:
                print(f"   💡 适配首层卷积通道: {v.shape} -> {model_dict[target_k].shape}")
                new_v = torch.zeros_like(model_dict[target_k])
                # Copy RGB weights; initialize the mask channel to zero.
                new_v[:, :3, :, :, :] = v
                new_state_dict[target_k] = new_v
                matched_count += 1

    if matched_count == 0:
        print("❌ 警告：没有加载任何层！请检查权重文件是否正确。")
        print(f"   权重文件键名示例: {list(pretrained_dict.keys())[:3]}")
        print(f"   模型键名示例: {list(model_dict.keys())[:3]}")
    else:
        print(f">> 成功加载 {matched_count} 层参数")

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    return model


def prepare_split_files(original_split_file, output_dir, val_ratio=0.2, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    with open(original_split_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    data = []
    labels = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            data.append(line)
            labels.append(int(parts[1]))

    train_lines, val_lines, train_labels, _ = train_test_split(
        data, labels, test_size=val_ratio, random_state=seed, stratify=labels
    )

    train_file_path = os.path.join(output_dir, 'temp_train_split.txt')
    val_file_path = os.path.join(output_dir, 'temp_val_split.txt')

    with open(train_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    with open(val_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))

    print(f"数据划分完成 (Video-Level): Train={len(train_lines)}, Val={len(val_lines)}")
    return train_file_path, val_file_path, train_labels


def calculate_class_weights(labels, num_classes=3):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    print("类别分布与权重:")
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count > 0:
            w = total / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)
        print(f"  Class {i}: {count} samples, Weight: {w:.4f}")
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


class AWBLoss(nn.Module):
    def __init__(self, num_classes=3, eps=1e-8):
        super(AWBLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, pred, target):
        probs = F.softmax(pred, dim=1)
        B = probs.size(0)
        target_probs = probs.gather(1, target.view(-1, 1)).squeeze(1)
        unique_classes, counts = torch.unique(target, return_counts=True)
        class_counts = torch.zeros(self.num_classes, device=pred.device)
        class_counts[unique_classes] = counts.float()
        if len(counts) > 0:
            max_Nk = torch.max(counts).float()
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        L1_loss = 0.0
        L2_loss = 0.0
        valid_classes_count = 0
        for k in unique_classes:
            k = k.item()
            Nk = class_counts[k]
            mask = (target == k)
            p_k = target_probs[mask]
            omega_k = torch.log((max_Nk / Nk) + 1)
            log_p_k = torch.log(p_k + self.eps)
            q_k = torch.mean(log_p_k)
            L1_loss += -1 * omega_k * q_k
            if Nk > 1:
                pk_mu = torch.mean(p_k)
                pk_delta = torch.std(p_k)
                term = pk_delta / (pk_mu + self.eps)
                L2_term = (Nk / B) * omega_k * term
                L2_loss += L2_term
            valid_classes_count += 1
        if valid_classes_count > 0:
            L1_loss = L1_loss / valid_classes_count
        return L1_loss + L2_loss


def train_model(data_root,
                mask_root,
                train_split_file,
                num_classes=3,
                loss_type='awb_loss',
                epochs=50,
                batch_size=8,
                base_lr=1e-4,
                weight_decay=1e-4,
                manual_class_weights=None,
                num_workers=0,
                seed=42,
                save_dir="./checkpoints",
                save_freq=20, resume_epoch=None,
                warmup_epochs=10,
                warmup_start_lr=1e-5,
                val_split_random_state=42,
                val_batch_size=None,
                accumulation_steps=4,
                pretrained_path=None,
                val_ratio=0.2,
                slide_step=32,
                balanced_sampling=False,
                ):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()

    temp_train_split, temp_val_split, train_labels = prepare_split_files(
        train_split_file, output_dir=save_dir, val_ratio=val_ratio, seed=val_split_random_state
    )

    if manual_class_weights is not None:
        class_weights = torch.tensor(manual_class_weights, dtype=torch.float32)
    else:
        mapped_labels = [l for l in train_labels]
        class_weights = calculate_class_weights(mapped_labels, num_classes=num_classes)

    train_dataset = FramesDataset(
        data_root=data_root,
        mask_root=mask_root,
        split_file=temp_train_split,
        mode='train',
        slide_step=slide_step,
    )
    val_dataset = FramesDataset(
        data_root=data_root,
        mask_root=mask_root,
        split_file=temp_val_split,
        mode='val',
        slide_step=slide_step,
    )

    val_batch_size = val_batch_size if val_batch_size is not None else batch_size

    if balanced_sampling:
        clip_labels = [sample[1] for sample in train_dataset.expanded_data_list]
        class_counts = Counter(clip_labels)
        print(f"  [Sampler] Training clip distribution: {dict(class_counts)}")
        sample_weights = [1.0 / (class_counts[label] + 1e-6) for label in clip_labels]

        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        print("WeightedRandomSampler enabled.")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )

    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(
        f"Dataset Ready. Train Clips: {len(train_dataset)} | Val Clips: {len(val_dataset)} | Batch Size: {batch_size}")

    print(f"Initializing Model on {device} (Acc Steps: {accumulation_steps})...")
    model = LaryngealRecognitionModel(num_classes=num_classes)

    if pretrained_path:
        model = load_pretrained_weights(model, pretrained_path)

        print(">> 正在冻结 backbone 浅层参数以防止过拟合...")
        for name, param in model.backbone.named_parameters():
            if 'layers.0' in name or 'layers.1' in name or 'layers.2' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        print("⚠️ 警告: 未指定预训练权重路径，模型将随机初始化！")

    model = model.to(device)

    if loss_type == 'awb_loss':
        criterion = AWBLoss(num_classes=num_classes)
        print("✅ 使用 AWB Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("✅ 使用 加权 CrossEntropy Loss 应对类别不平衡")

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (warmup_start_lr + (epoch / warmup_epochs) * (
            base_lr - warmup_start_lr)) / base_lr)
    annealing_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=base_lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, annealing_scheduler], milestones=[warmup_epochs])

    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_train_acc = 0.0
    start_epoch = 0

    if resume_epoch is not None:
        cp_path = os.path.join(save_dir, 'best_model.pth')
        if os.path.exists(cp_path):
            checkpoint = torch.load(cp_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            print(f"Resumed from epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}")

    def evaluate_val(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        all_preds = []
        all_labels = []

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                input_3d_32 = batch['input_3d_32'].to(device)
                mask_3d_32 = batch['mask_3d_32'].to(device)
                distance_features = batch['distance_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_3d_32, mask_3d_32, distance_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_total += labels.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / val_total if val_total > 0 else float('inf')
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return avg_val_loss, val_acc, val_f1, val_total

    optimizer.zero_grad()

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

        for i, batch in enumerate(train_pbar):
            input_3d_32 = batch['input_3d_32'].to(device)
            mask_3d_32 = batch['mask_3d_32'].to(device)
            distance_features = batch['distance_features'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_3d_32, mask_3d_32, distance_features)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            current_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})

        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        avg_train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0

        avg_val_loss, val_acc, val_f1, val_total_count = evaluate_val(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]['lr']
        print(f"\n > End Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} | Train Acc {avg_train_acc:.2f}%")
        print(f" > Validation : Val Loss {avg_val_loss:.4f} | Clip Acc {val_acc:.2f}% | F1(Macro) {val_f1:.4f}")
        print(f" > Time: {time.time() - start_time:.1f}s | LR: {lr:.6f}")

        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Val/Clip_Acc', val_acc, epoch)

        scheduler.step()

        # Prefer lower validation loss; use macro F1 as the tie-breaker.
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_f1 = val_f1
            is_best = True
        elif abs(avg_val_loss - best_val_loss) < 1e-4 and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            is_best = True

        if is_best and val_total_count > 0:
            best_val_acc = val_acc
            best_train_acc = avg_train_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_train_acc': best_train_acc,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'best_val_f1': best_val_f1,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(
                f"  🌟 [保存最佳模型] 更新！当前 Val Loss: {best_val_loss:.4f} | F1: {best_val_f1:.4f} | Acc: {best_val_acc:.2f}%")

        if (epoch + 1) % save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_acc': avg_train_acc,
                'val_acc': val_acc,
            }, os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'best_train_acc': best_train_acc,
        'best_val_acc': best_val_acc,
    }, os.path.join(save_dir, 'final_model.pth'))

    writer.close()
    print(f"\n训练完成，最佳模型的验证 Loss: {best_val_loss:.4f} | F1: {best_val_f1:.4f} | Acc: {best_val_acc:.4f}")
    return best_val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train PADSNet on 300-frame laryngeal samples.")
    parser.add_argument(
        "--data-root",
        default="data/frames/train",
        help="Frame root; default: data/frames/train",
    )
    parser.add_argument(
        "--mask-root",
        default="data/masks",
        help="Flat mask directory; default: data/masks",
    )
    parser.add_argument(
        "--train-split",
        default="splits/train.txt",
        help="Video-level split file; default: splits/train.txt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/checkpoints/padsnet",
        help="Checkpoint and TensorBoard directory.",
    )
    parser.add_argument(
        "--pretrained",
        default="weights/swin_tiny_kinetics400.pth",
        help="Video Swin Tiny checkpoint; use an empty string to train from scratch.",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--loss", choices=("awb_loss", "ce_loss"), default="awb_loss")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--slide-step", type=int, default=32)
    parser.add_argument("--balanced-sampling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_model(
        data_root=args.data_root,
        mask_root=args.mask_root,
        train_split_file=args.train_split,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        num_workers=args.workers,
        seed=args.seed,
        save_dir=args.output_dir,
        warmup_epochs=args.warmup_epochs,
        val_split_random_state=args.seed,
        accumulation_steps=args.accumulation_steps,
        pretrained_path=args.pretrained or None,
        val_ratio=args.val_ratio,
        slide_step=args.slide_step,
        balanced_sampling=args.balanced_sampling,
    )
