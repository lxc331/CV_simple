import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as data
from torch import nn
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import VisionTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
TRAIN_PARAMETER_DIR = BASE_DIR / "train_parameter"


class ModelEMA:
    """维护模型参数的指数滑动平均，用于获得更稳定的验证结果。"""

    def __init__(self, model, decay=0.9995):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_parameters = dict(self.module.named_parameters())
        model_parameters = dict(model.named_parameters())
        for name, ema_parameter in ema_parameters.items():
            ema_parameter.mul_(self.decay).add_(
                model_parameters[name].detach(), alpha=1.0 - self.decay
            )

        ema_buffers = dict(self.module.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(model_buffers[name])


def deal_train_and_val_data():
    """加载 FashionMNIST，并使用不同预处理构建训练集和验证集。"""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=6,
                translate=(0.06, 0.06),
                scale=(0.97, 1.03),
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.RandomErasing(p=0.08, scale=(0.02, 0.10), value=0),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    train_dataset = FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=train_transform
    )
    val_dataset = FashionMNIST(
        root=str(DATA_DIR), train=True, download=True, transform=val_transform
    )

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_size = round(len(indices) * 0.8)
    train_data = data.Subset(train_dataset, indices[:train_size])
    val_data = data.Subset(val_dataset, indices[train_size:])

    pin_memory = torch.cuda.is_available()
    train_dataloader = data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_dataloader = data.DataLoader(
        val_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return train_dataloader, val_dataloader


def cosine_learning_rate(optimizer, epoch, epochs, base_lr, min_lr, warmup_epochs=5):
    """线性预热后使用余弦退火调整学习率。"""
    if epoch < warmup_epochs:
        learning_rate = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
        learning_rate = min_lr + 0.5 * (base_lr - min_lr) * (
            1 + math.cos(math.pi * progress)
        )

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return learning_rate


def build_optimizer(model, learning_rate):
    """仅对矩阵权重使用 weight decay。"""
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim <= 1
            or name.endswith("bias")
            or "class_token" in name
            or "position_embedding" in name
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": 0.05},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def create_grad_scaler(enabled):
    """兼容新旧 PyTorch 的 GradScaler 接口。"""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled):
    """兼容新旧 PyTorch 的 autocast 接口。"""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def mixup_batch(images, labels, alpha=0.2, probability=0.5):
    """以一定概率混合两张图片和标签。"""
    if alpha <= 0 or torch.rand(1).item() > probability:
        return images, labels, labels, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


@torch.no_grad()
def evaluate_model(model, val_dataloader, criterion, device, amp_enabled):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_num = 0

    for b_x, b_y in val_dataloader:
        b_x = b_x.to(device, non_blocking=True)
        b_y = b_y.to(device, non_blocking=True)

        with autocast_context(amp_enabled):
            output = model(b_x)
            loss = criterion(output, b_y)

        prediction = torch.argmax(output, dim=1)
        val_loss += loss.item() * b_x.size(0)
        val_correct += torch.sum(prediction == b_y).item()
        val_num += b_x.size(0)

    return val_loss / val_num, val_correct / val_num


def train_model(model, train_dataloader, val_dataloader, epochs):
    """使用单张 GPU 或 CPU 训练，并保存最佳 EMA 权重。"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    base_lr = 5e-4
    model = model.to(device)
    optimizer = build_optimizer(model, base_lr)
    model_ema = ModelEMA(model, decay=0.9995)

    amp_enabled = device.type == "cuda"
    scaler = create_grad_scaler(amp_enabled)
    best_model_wts = copy.deepcopy(model_ema.module.state_dict())
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    total_time = 0.0

    print("training device: {}".format(device))

    for epoch in range(epochs):
        epoch_start_time = time.time()
        learning_rate = cosine_learning_rate(
            optimizer, epoch, epochs, base_lr=base_lr, min_lr=1e-5
        )
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 10)
        print("learning rate: {:.6f}".format(learning_rate))

        train_loss = 0.0
        train_correct = 0.0
        train_num = 0
        model.train()

        for b_x, b_y in train_dataloader:
            b_x = b_x.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            mixed_x, target_a, target_b, lam = mixup_batch(b_x, b_y)

            with autocast_context(amp_enabled):
                output = model(mixed_x)
                loss = lam * criterion(output, target_a) + (
                    1.0 - lam
                ) * criterion(output, target_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            model_ema.update(model)

            prediction = torch.argmax(output, dim=1)
            train_loss += loss.item() * b_x.size(0)
            train_correct += lam * torch.sum(prediction == target_a).item()
            train_correct += (1.0 - lam) * torch.sum(
                prediction == target_b
            ).item()
            train_num += b_x.size(0)

        train_loss_value = train_loss / train_num
        train_acc_value = train_correct / train_num
        val_loss_value, val_acc_value = evaluate_model(
            model_ema.module,
            val_dataloader,
            criterion,
            device,
            amp_enabled,
        )

        train_loss_all.append(train_loss_value)
        val_loss_all.append(val_loss_value)
        train_acc_all.append(train_acc_value)
        val_acc_all.append(val_acc_value)

        print(
            "{} epoch, train loss: {:.4f}, train acc: {:.4f}".format(
                epoch + 1, train_loss_value, train_acc_value
            )
        )
        print(
            "{} epoch, val loss: {:.4f}, val acc: {:.4f}".format(
                epoch + 1, val_loss_value, val_acc_value
            )
        )

        if val_acc_value > best_acc:
            best_acc = val_acc_value
            best_model_wts = copy.deepcopy(model_ema.module.state_dict())

        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        print(
            "{} epoch consume time: {:.0f}m {:.0f}s".format(
                epoch + 1, epoch_time // 60, epoch_time % 60
            )
        )

    print("-" * 10)
    print("Best validation accuracy: {:.4f}".format(best_acc))
    print("Total train time: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PARAMETER_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_wts, MODEL_DIR / "best_model.pth")

    train_process = pd.DataFrame(
        {
            "epoch": range(1, epochs + 1),
            "train_loss": train_loss_all,
            "train_acc": train_acc_all,
            "val_loss": val_loss_all,
            "val_acc": val_acc_all,
        }
    )
    train_process.to_csv(TRAIN_PARAMETER_DIR / "train_parameter.csv", index=False)
    return train_process


def matplot_acc_loss(train_process):
    """绘制并保存训练过程中的损失和准确率曲线。"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss"], "ro-", label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], "bs-", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc"], "ro-", label="Train Accuracy")
    plt.plot(train_process["epoch"], train_process["val_acc"], "bs-", label="Val Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    TRAIN_PARAMETER_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(TRAIN_PARAMETER_DIR / "model_acc_loss.png")
    plt.close()


if __name__ == "__main__":
    train_dataloader, val_dataloader = deal_train_and_val_data()
    vit_model = VisionTransformer()
    train_process = train_model(vit_model, train_dataloader, val_dataloader, 50)
    matplot_acc_loss(train_process)

