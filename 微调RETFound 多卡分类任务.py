#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import os
import time
from pathlib import Path
import warnings
import faulthandler
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from timm.models.layers import trunc_normal_
from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from huggingface_hub import hf_hub_download

try:
    import optuna  # type: ignore
except ImportError:  # pragma: no cover - optuna is optional at runtime
    optuna = None

import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import evaluate  # 评估沿用原函数

from tqdm import tqdm
faulthandler.enable()
warnings.simplefilter(action="ignore", category=FutureWarning)

# =========================
# 小工具：返回 AMP/空 上下文实例（可用于 CPU/非AMP）
# =========================
def autocast_context(enabled=True, dtype=torch.bfloat16):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()

# =========================
# CSV 数据集（直接用 path 列）
# =========================
class CSVDataset(Dataset):
    def __init__(
        self, csv_file, split, path_col="path", label_col="partition",
        split_col="split", transform=None, ignore_unlabeled=True, class_to_idx=None
    ):
        """
        split: 'train' | 'iv' | 'ev'（大小写不敏感）
        """
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col
        self.split_col = split_col

        df = pd.read_csv(csv_file)

        # 仅该 split
        df = df[df[split_col].astype(str).str.lower() == str(split).lower()]

        # 过滤无标签
        if ignore_unlabeled:
            df = df[~df[label_col].isna()]
            df = df[df[label_col].astype(str).str.len() > 0]

        # 路径存在性检查
        df[path_col] = df[path_col].astype(str)
        df["__exists__"] = df[path_col].apply(lambda p: Path(p).is_file())
        missing = (~df["__exists__"]).sum()
        if missing > 0:
            print(f"[WARN] {missing} files not found; dropped. Example: "
                  f"{df.loc[~df['__exists__'], path_col].head(3).tolist()}")
        df = df[df["__exists__"]].copy()

        # 标签归一：数字/浮点转 int，其他保留 str
        def norm_label(v):
            try:
                return int(float(v))
            except Exception:
                return str(v)
        df["__y__"] = df[label_col].apply(norm_label)

        # 类别映射（未指定则按该 split 构建；真正训练会用 train 的映射对齐）
        if class_to_idx is None:
            uniq = df["__y__"].unique().tolist()
            nums = sorted([u for u in uniq if isinstance(u, int)])
            strs = sorted([u for u in uniq if isinstance(u, str)])
            classes = nums + strs
            class_to_idx = {c: i for i, c in enumerate(classes)}

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # 样本列表
        self.samples = []
        for _, r in df.iterrows():
            y = r["__y__"]
            if y not in self.class_to_idx:
                continue
            self.samples.append((r[path_col], self.class_to_idx[y]))

        self._report()

    def _report(self):
        print(f"[INFO] Loaded {len(self.samples)} samples for split.")
        counter = Counter([y for _, y in self.samples])
        if len(counter) > 0:
            pretty = ", ".join([f"{self.idx_to_class[k]}({k}):{v}" for k, v in sorted(counter.items())])
            print(f"[INFO] Class histogram: {pretty}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def get_class_counts(self):
        """返回各类别的样本数量，用于计算权重"""
        counter = Counter([y for _, y in self.samples])
        counts = [0] * len(self.class_to_idx)
        for class_idx, count in counter.items():
            counts[class_idx] = count
        return counts


def build_transforms(args):
    # 没有 --norm 时回退 IMAGENET
    norm_name = getattr(args, "norm", "IMAGENET")
    mean_std = {
        "IMAGENET": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }
    mean, std = mean_std.get(str(norm_name).upper(), mean_std["IMAGENET"])

    train_tf = create_transform(
        input_size=args.input_size, is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa if args.aa and args.aa.lower() != "none" else None,
        interpolation="bicubic",
        re_prob=args.reprob, re_mode=args.remode, re_count=args.recount,
        mean=mean, std=std,
    )
    eval_tf = create_transform(
        input_size=args.input_size, is_training=False,
        interpolation="bicubic", mean=mean, std=std,
    )
    return train_tf, eval_tf


# =========================
# 训练（支持单卡/多卡 + 梯度累积）
# =========================
def train_one_epoch(
    model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
    clip_grad=None, mixup_fn=None, log_writer=None, args=None
):
    model.train(True)
    header = f"Epoch[{epoch}]"
    num_steps = len(data_loader)
    running_loss = 0.0

    optimizer.zero_grad(set_to_none=True)
    accum_iter = max(1, int(args.accum_iter))

    # DDP 情况下，确保每个 epoch 刷新 shuffle
    if isinstance(getattr(data_loader, "sampler", None), DistributedSampler):
        data_loader.sampler.set_epoch(epoch)

    for step, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        with autocast_context(enabled=True, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / accum_iter  # 累积时缩放

        loss_value = loss.item() * accum_iter
        running_loss += loss_value

        grad_update = ((step + 1) % accum_iter == 0) or (step + 1 == num_steps)
        loss_scaler(
            loss, optimizer, clip_grad=clip_grad, parameters=model.parameters(),
            create_graph=False, update_grad=grad_update
        )
        if grad_update:
            optimizer.zero_grad(set_to_none=True)

        if (step % 50 == 0) or (step + 1 == num_steps):
            lr = optimizer.param_groups[0]["lr"]
            if misc.is_main_process():
                print(f"{header} [{step+1:>5d}/{num_steps}] loss={loss_value:.4f} lr={lr:.2e}")

    # 进程间取平均（可选；这里只在主进程记录）
    epoch_loss = running_loss / max(1, num_steps)
    if log_writer is not None and misc.is_main_process():
        log_writer.add_scalar("loss/train", epoch_loss, epoch)
        log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
    return {"loss": epoch_loss}


def get_args_parser():
    parser = argparse.ArgumentParser(
        "RETFound DINOv2 fine-tuning (single/multi-GPU, CSV with absolute paths)", add_help=False
    )
    # Core（适配 10–20k 图像微调）
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Per-GPU batch size (effective = batch_size * accum_iter * world_size)")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--accum_iter", default=2, type=int, help="Gradient accumulation steps")

    # Model
    parser.add_argument("--model", default="RETFound_dinov2", type=str,
                        help="RETFound_dinov2 / RETFound_mae / Dinov2 / Dinov3 ...")
    parser.add_argument("--model_arch", default="dinov2_vitb14", type=str)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--global_pool", action="store_true"); parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool")

    # Optim / schedule
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None,
                        help="If None, lr = blr * (batch_size*accum_iter*world_size)/256")
    parser.add_argument("--blr", type=float, default=2e-3,
                        help="base lr; lr = blr * (batch_size*accum_iter*world_size)/256")
    parser.add_argument("--layer_decay", type=float, default=0.65)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Aug
    parser.add_argument("--color_jitter", type=float, default=None)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--smoothing", type=float, default=0.0)  # 改为0.0，避免伤害少数类
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)

    # Mixup/Cutmix（默认关；若开请改用 SoftTargetCrossEntropy）
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # Finetune
    parser.add_argument("--finetune", default="RETFound_dinov2", type=str,
                        help="RETFound_dinov2 / RETFound_mae will be downloaded from HF")
    parser.add_argument("--task", default="csv_paths_run", type=str)
    parser.add_argument("--adaptation", default="finetune", choices=["finetune", "lp"])

    # CSV columns
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--path_col", type=str, default="path",
                        help="column name for absolute image path")
    parser.add_argument("--label_col", type=str, default="partition",
                        help="label column name (e.g., eye_history or partition)")
    parser.add_argument("--split_col", type=str, default="split",
                        help="column that has train/iv/ev")
    parser.add_argument("--ignore_unlabeled", action="store_true", default=True,
                        help="drop rows with empty/NaN label")

    # Runtime & IO
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true"); parser.set_defaults(pin_mem=True)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_logs")
    parser.add_argument("--savemodel", action="store_true", default=True)

    # 分布式参数（按 util.misc 的约定）
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_eval", action="store_true", default=False,
                        help="use DistributedSampler for val/test")

    # 归一化 preset
    parser.add_argument("--norm", default="IMAGENET", type=str,
                        help="Normalization preset, default IMAGENET")

    # 可选：手动覆盖类别数（否则从 train 推断）
    parser.add_argument("--nb_classes", type=int, default=None,
                        help="override class count (otherwise infer from train split)")

    # 类别权重和采样相关参数
    parser.add_argument("--use_class_weights", action="store_true", default=True,
                        help="use class weights in CrossEntropyLoss")
    parser.add_argument("--use_weighted_sampler", action="store_true", default=True,
                        help="use WeightedRandomSampler for training")

    # Optuna tuning
    parser.add_argument("--optuna_trials", type=int, default=0,
                        help="number of Optuna trials (0 to disable)")
    parser.add_argument("--optuna_timeout", type=int, default=None,
                        help="overall Optuna timeout in seconds")
    parser.add_argument("--optuna_direction", type=str, default="maximize",
                        help="study direction, typically maximize validation score")
    parser.add_argument("--optuna_storage", type=str, default=None,
                        help="Optuna storage URL for persistent studies")
    parser.add_argument("--optuna_study_name", type=str, default=None,
                        help="Optuna study name")
    parser.add_argument("--optuna_resume", action="store_true", default=False,
                        help="resume Optuna study if it already exists")
    parser.add_argument("--optuna_seed", type=int, default=None,
                        help="random seed for Optuna sampler")
    parser.add_argument("--optuna_pruner", type=str, default="none",
                        help="pruner type: none or median")
    parser.add_argument("--optuna_epochs", type=int, default=None,
                        help="override training epochs during Optuna trials")
    parser.add_argument("--optuna_eval_test", action="store_true", default=False,
                        help="run test evaluation inside Optuna trials")
    parser.add_argument("--optuna_n_jobs", type=int, default=1,
                        help="number of parallel Optuna jobs")
    parser.add_argument("--optuna_sampler", type=str, default="tpe",
                        help="sampler to use: tpe or random")

    return parser


def run_experiment(args: argparse.Namespace, trial: Optional["optuna.trial.Trial"] = None) -> float:
    # 由 util.misc 自动设置 args.distributed / args.gpu / ranks
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    if misc.is_main_process():
        print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
        print(f"{args}".replace(", ", ",\n"))
        if trial is not None:
            print(f"[Optuna] Running trial #{trial.number}")

    if trial is not None:
        args.blr = trial.suggest_float("blr", 5e-4, 5e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-2, log=True)
        args.drop_path = trial.suggest_float("drop_path", 0.0, 0.4)
        args.layer_decay = trial.suggest_float("layer_decay", 0.5, 0.9)
        args.smoothing = trial.suggest_float("smoothing", 0.0, 0.15)
        args.accum_iter = trial.suggest_categorical("accum_iter", [1, 2])
        args.use_class_weights = trial.suggest_categorical("use_class_weights", [True, False])
        args.use_weighted_sampler = trial.suggest_categorical("use_weighted_sampler", [True, False])
        args.lr = None

    # DDP 建议：多卡时固定本地设备
    if args.distributed and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True

    start_time = time.time()
    max_score = float("-inf")
    best_epoch = 0
    log_writer: Optional[SummaryWriter] = None

    try:
        # ====== 数据与类别数（先于建模，以便确定 nb_classes）======
        train_tf, eval_tf = build_transforms(args)

        tmp_train = CSVDataset(
            args.csv_file, split="train",
            path_col=args.path_col, label_col=args.label_col, split_col=args.split_col,
            transform=train_tf, ignore_unlabeled=args.ignore_unlabeled, class_to_idx=None
        )
        class_to_idx = tmp_train.class_to_idx
        inferred_num_classes = len(class_to_idx)
        if args.nb_classes is None:
            args.nb_classes = inferred_num_classes
            if misc.is_main_process():
                print(f"[INFO] nb_classes inferred from train split: {args.nb_classes}")
        elif args.nb_classes != inferred_num_classes and misc.is_main_process():
            print(f"[WARN] nb_classes ({args.nb_classes}) != inferred ({inferred_num_classes}); "
                  f"proceeding with nb_classes={args.nb_classes}")

        dataset_train = tmp_train
        dataset_val = CSVDataset(
            args.csv_file, split="iv",
            path_col=args.path_col, label_col=args.label_col, split_col=args.split_col,
            transform=eval_tf, ignore_unlabeled=args.ignore_unlabeled, class_to_idx=class_to_idx
        )
        dataset_test = CSVDataset(
            args.csv_file, split="ev",
            path_col=args.path_col, label_col=args.label_col, split_col=args.split_col,
            transform=eval_tf, ignore_unlabeled=args.ignore_unlabeled, class_to_idx=class_to_idx
        )

        # ====== 计算类别权重 ======
        if args.use_class_weights:
            class_counts = dataset_train.get_class_counts()
            counts_tensor = torch.tensor(class_counts, dtype=torch.float32)
            denom = counts_tensor.clamp_min(1.0)
            class_weights = (counts_tensor.sum() / (len(counts_tensor) * denom)).to(device)
            if misc.is_main_process():
                print(f"[INFO] Class counts: {class_counts}")
                print(f"[INFO] Class weights: {class_weights.cpu().tolist()}")
            criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=args.smoothing
            )
        else:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)

        # ====== 模型 ======
        if args.model == "RETFound_mae":
            model = models.__dict__[args.model](
                img_size=args.input_size, num_classes=args.nb_classes,
                drop_path_rate=args.drop_path, global_pool=args.global_pool,
            )
        else:
            model = models.__dict__[args.model](
                num_classes=args.nb_classes, drop_path_rate=args.drop_path, args=args,
            )

        # 预训练权重
        if args.finetune and not args.eval:
            if misc.is_main_process():
                print(f"[Load Pretrain] {args.finetune}")
            if args.model in ["Dinov3", "Dinov2"]:
                checkpoint_path = args.finetune
            elif args.model in ["RETFound_dinov2", "RETFound_mae"]:
                checkpoint_path = hf_hub_download(
                    repo_id=f"YukunZhou/{args.finetune}",
                    filename=f"{args.finetune}.pth",
                )
            else:
                raise ValueError("Unsupported model for finetuning")

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if args.model in ["Dinov3", "Dinov2"]:
                checkpoint_model = checkpoint
            elif args.model == "RETFound_dinov2":
                checkpoint_model = checkpoint["teacher"]
            else:
                checkpoint_model = checkpoint["model"]

            ckpt = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
            ckpt = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in ckpt.items()}
            ckpt = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in ckpt.items()}

            # 删除不匹配的分类头
            state_dict = model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if k in ckpt and ckpt[k].shape != state_dict[k].shape:
                    if misc.is_main_process():
                        print(f"[Pretrain] remove {k}")
                    del ckpt[k]

            interpolate_pos_embed(model, ckpt)
            _ = model.load_state_dict(ckpt, strict=False)

            # 重新初始化分类头
            if hasattr(model, "head") and hasattr(model.head, "weight"):
                trunc_normal_(model.head.weight, std=2e-5)
                if getattr(model.head, "bias", None) is not None:
                    torch.nn.init.zeros_(model.head.bias)

        # ====== DDP 包装（如启用）
        model.to(device)
        model_without_ddp = model
        if args.distributed:
            ddp_kwargs = dict(broadcast_buffers=False)
            if args.adaptation == "lp":
                ddp_kwargs["find_unused_parameters"] = True
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu] if torch.cuda.is_available() else None, **ddp_kwargs
            )
            model_without_ddp = model.module

        # ====== Sampler / DataLoader（单/多卡）======
        if args.distributed:
            sampler_train = DistributedSampler(
                dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
            )
            if misc.is_main_process():
                print("[INFO] Using DistributedSampler for training (WeightedRandomSampler disabled in distributed mode)")
            if args.dist_eval:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
                sampler_test = DistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_val = SequentialSampler(dataset_val)
                sampler_test = SequentialSampler(dataset_test)
        else:
            if args.use_weighted_sampler:
                class_counts = dataset_train.get_class_counts()
                sample_weights = []
                for _, target in dataset_train.samples:
                    sample_weights.append(1.0 / max(class_counts[target], 1))

                sampler_train = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                if misc.is_main_process():
                    print("[INFO] Using WeightedRandomSampler for training")
            else:
                sampler_train = RandomSampler(dataset_train)
                if misc.is_main_process():
                    print("[INFO] Using RandomSampler for training")

            sampler_val = SequentialSampler(dataset_val)
            sampler_test = SequentialSampler(dataset_test)

        if misc.is_main_process() and (not args.eval):
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
        )
        if misc.is_main_process():
            print(f"[INFO] len(train_set)={len(dataset_train)} (batches: {len(data_loader_train)})")

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
        )

        # Mixup/Cutmix
        mixup_fn = None
        mixup_active = (args.mixup > 0) or (args.cutmix > 0.) or (args.cutmix_minmax is not None)
        if mixup_active:
            if misc.is_main_process():
                print("[INFO] Mixup activated")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes
            )
            criterion = SoftTargetCrossEntropy()

        # Eval-only resume
        if args.resume and args.eval:
            checkpoint = torch.load(args.resume, map_location="cpu")
            if misc.is_main_process():
                print(f"[Eval] Load checkpoint: {args.resume}")
            model_without_ddp.load_state_dict(checkpoint["model"])

        if args.adaptation == "lp":
            for name, p in model_without_ddp.named_parameters():
                p.requires_grad = ("head" in name)
            if misc.is_main_process():
                print("[Adaptation] Linear probe: train head only.")
        else:
            if misc.is_main_process():
                print("[Adaptation] Full finetune.")

        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        if misc.is_main_process():
            print(f"[INFO] Trainable params (M): {n_parameters/1e6:.2f}")

        world_size = misc.get_world_size()
        eff_bs = args.batch_size * args.accum_iter * world_size
        if args.lr is None:
            args.lr = args.blr * eff_bs / 256
        if misc.is_main_process():
            print(f"base lr: {args.blr:.2e}  actual lr: {args.lr:.2e}  accum_iter: {args.accum_iter}  "
                  f"world_size: {world_size}  eff_bs: {eff_bs}")

        no_weight_decay = (model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, "no_weight_decay") else [])
        param_groups = lrd.param_groups_lrd(
            model_without_ddp, weight_decay=args.weight_decay,
            no_weight_decay_list=no_weight_decay, layer_decay=args.layer_decay,
        )
        for g in param_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        loss_scaler = NativeScaler()
        if misc.is_main_process():
            print(f"criterion = {criterion}")

        misc.load_model(args=args, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            if "checkpoint" in locals() and isinstance(checkpoint, dict) and ("epoch" in checkpoint):
                if misc.is_main_process():
                    print(f"[Eval] best epoch = {checkpoint['epoch']}")
            _test_stats, _auc_roc = evaluate(
                data_loader_test, model, device, args, epoch=0, mode="test",
                num_class=args.nb_classes, log_writer=log_writer
            )
            if log_writer is not None:
                log_writer.flush()
                log_writer.close()
            return _auc_roc

        if misc.is_main_process():
            desc = f"Trial {trial.number}" if trial is not None else "Epochs"
            epoch_iter = tqdm(range(args.start_epoch, args.epochs), desc=desc)
        else:
            epoch_iter = range(args.start_epoch, args.epochs)

        for epoch in epoch_iter:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                clip_grad=args.clip_grad, mixup_fn=mixup_fn,
                log_writer=log_writer, args=args
            )

            val_stats, val_score = evaluate(
                data_loader_val, model, device, args, epoch, mode="val",
                num_class=args.nb_classes, log_writer=log_writer
            )

            if trial is not None:
                trial.report(val_score, epoch)
                if trial.should_prune():
                    if log_writer is not None:
                        log_writer.flush()
                        log_writer.close()
                    raise optuna.TrialPruned()  # type: ignore[misc]

            if misc.is_main_process():
                if max_score < val_score:
                    max_score = val_score
                    best_epoch = epoch
                    if args.output_dir and args.savemodel:
                        misc.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, mode="best"
                        )
                print(f"[VAL] Best epoch={best_epoch}, Best score={max_score:.4f}")

                if log_writer is not None:
                    log_writer.add_scalar("loss/val", val_stats["loss"], epoch)
                    log_writer.flush()

                log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                             "epoch": epoch, "n_parameters": n_parameters}
                if args.output_dir:
                    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)
                    with open(os.path.join(args.output_dir, args.task, "log.txt"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

            if args.distributed:
                torch.distributed.barrier()

        if log_writer is not None:
            log_writer.flush()
            log_writer.close()

        if trial is None or args.optuna_eval_test:
            ckpt_path = os.path.join(args.output_dir, args.task, "checkpoint-best.pth")
            if args.savemodel and os.path.isfile(ckpt_path):
                if misc.is_main_process():
                    print(f"[TEST] Loading best checkpoint from {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
                model.to(device)
                if misc.is_main_process():
                    print(f"[TEST] best epoch = {checkpoint.get('epoch', -1)}")
            if args.distributed:
                for p in model.parameters():
                    torch.distributed.broadcast(p.data, src=0)

            _test_stats, _auc_roc = evaluate(
                data_loader_test, model, device, args, -1, mode="test",
                num_class=args.nb_classes, log_writer=None
            )
            if trial is None and misc.is_main_process():
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print(f"Training time {total_time_str}")
                print(f"[TEST] score={_auc_roc:.4f}")
        else:
            if misc.is_main_process():
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print(f"[Optuna][Trial {trial.number}] best score={max_score:.4f} (time={total_time_str})")

        return max_score

    finally:
        if log_writer is not None:
            log_writer.flush()
            log_writer.close()
        if args.distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _clone_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def run_optuna(args: argparse.Namespace) -> None:
    if optuna is None:
        raise ImportError("Optuna is not installed. Please `pip install optuna`.")
    if args.optuna_trials <= 0:
        raise ValueError("optuna_trials must be > 0 to start tuning")

    sampler: Optional[optuna.samplers.BaseSampler]
    if args.optuna_sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.optuna_seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)

    pruner: Optional[optuna.pruners.BasePruner] = None
    if args.optuna_pruner == "median":
        pruner = optuna.pruners.MedianPruner()

    study_kwargs = {"direction": args.optuna_direction, "sampler": sampler, "pruner": pruner}
    if args.optuna_storage:
        study_kwargs["storage"] = args.optuna_storage
        if args.optuna_study_name:
            study_kwargs["study_name"] = args.optuna_study_name
        study_kwargs["load_if_exists"] = args.optuna_resume
    elif args.optuna_study_name:
        study_kwargs["study_name"] = args.optuna_study_name

    study = optuna.create_study(**study_kwargs)

    def objective(trial: "optuna.trial.Trial") -> float:
        trial_args = _clone_args(args)
        trial_args.optuna_trials = 0
        trial_args.optuna_timeout = None
        trial_args.optuna_n_jobs = 1
        trial_args.optuna_direction = args.optuna_direction
        trial_args.optuna_pruner = args.optuna_pruner
        trial_args.optuna_sampler = args.optuna_sampler
        trial_args.optuna_eval_test = args.optuna_eval_test
        trial_args.task = f"{args.task}_trial{trial.number:04d}"
        if args.optuna_epochs is not None:
            trial_args.epochs = args.optuna_epochs
        trial_args.start_epoch = 0
        trial_args.resume = ""
        trial_args.savemodel = bool(args.optuna_eval_test)
        base_output = Path(args.output_dir)
        base_output.mkdir(parents=True, exist_ok=True)
        (Path(args.log_dir)).mkdir(parents=True, exist_ok=True)
        return run_experiment(trial_args, trial)

    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=args.optuna_timeout,
        n_jobs=args.optuna_n_jobs,
        gc_after_trial=True,
    )

    if len(study.trials) == 0:
        return

    if study.best_trial is not None:
        if optuna.study.StudyDirection.MAXIMIZE == study.direction:
            direction_str = "maximize"
        else:
            direction_str = "minimize"
        print(f"[Optuna] Best trial #{study.best_trial.number} {direction_str} value={study.best_value:.4f}")
        print("[Optuna] Best parameters:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")

        summary_path = Path(args.output_dir) / args.task / "optuna_best.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump({
                "value": study.best_value,
                "params": study.best_trial.params,
                "direction": args.optuna_direction,
                "datetime": datetime.datetime.now().isoformat(),
            }, fp, ensure_ascii=False, indent=2)


def main(args: argparse.Namespace) -> None:
    if args.optuna_trials and args.optuna_trials > 0:
        run_optuna(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir and (not os.path.exists(args.output_dir)):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
