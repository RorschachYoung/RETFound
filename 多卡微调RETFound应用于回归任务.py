#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-task regression training script (EfficientNet-B4 + TabTransformer-style head)
- Fixes LRD (layer-wise LR decay) for non-ResNet backbones by providing a generic
  param_groups_lrd() in-script (no reliance on util/lr_decay.py having layer1..4).
- Correctly applies per-group learning-rate scaling AFTER optimizer creation.
- Keeps your requested targets: tc, ldl, hdl, tg, apoa, apob.
- Adds R2 / Pearson R metrics; macro R2 used for model selection.
- Includes clean DDP teardown to avoid NCCL warnings on crash/exit.
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path
import warnings
import faulthandler
from contextlib import nullcontext
from typing import Dict, Iterable, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from timm.models.layers import trunc_normal_
from timm.data import create_transform
import timm

try:
    import optuna
except ImportError:  # pragma: no cover - optuna is optional at runtime
    optuna = None

# project utils (unchanged)
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from tqdm import tqdm
faulthandler.enable()
warnings.simplefilter(action="ignore", category=FutureWarning)

# =========================
# AMP helper
# =========================
def autocast_context(enabled=True, dtype=torch.bfloat16):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()

# =========================
# Generic Layer-wise LR Decay (LRD) for diverse backbones
# Replaces util/lr_decay.py ResNet-only logic.
# =========================
_DECAY_EXCLUDES = ("bias", "bn", "norm", "ln", "embedding")

def _is_no_weight_decay(name: str, param: nn.Parameter, extra_nowd: Iterable[str]) -> bool:
    if param.ndim == 1:  # norm/bn/scale weights
        return True
    lname = name.lower()
    if lname.endswith(".bias"):
        return True
    if any(k in lname for k in _DECAY_EXCLUDES):
        return True
    if extra_nowd and name in set(extra_nowd):
        return True
    return False


def _count_attr_len(obj, attr: str) -> int:
    m = getattr(obj, attr, None)
    if m is None:
        return 0
    try:
        return len(m)
    except TypeError:
        return 0


def _infer_layering(model: nn.Module) -> Tuple[int, List[Tuple[re.Pattern, int]]]:
    """
    Returns (num_layers, rules). Rules are [(regex, base_id)] mapping module name -> layer id.
    Top/head gets layer_id = num_layers-1; stem gets 0.
    Handles RegressionModel(backbone, head) with various backbones (ViT blocks, EfficientNet blocks, ConvNeXt stages, ResNet layer1..4).
    """
    back = getattr(model, "backbone", model)

    # ViT/MAE-style blocks
    n_blocks = _count_attr_len(back, "blocks")
    if n_blocks > 0:
        num_layers = n_blocks + 1  # + head
        rules = [
            (re.compile(r"(?:^|\.)(?:backbone\.)?blocks\.(\d+)\."), 1),
        ]
        return num_layers, rules

    # ConvNeXt-style stages
    n_stages = _count_attr_len(back, "stages")
    if n_stages > 0:
        num_layers = n_stages + 1
        rules = [
            (re.compile(r"(?:^|\.)(?:backbone\.)?stages\.(\d+)\."), 1),
        ]
        return num_layers, rules

    # EfficientNet/MBConv blocks (Sequential)
    n_blocks2 = _count_attr_len(back, "blocks")
    if n_blocks2 > 0:
        num_layers = n_blocks2 + 1
        rules = [
            (re.compile(r"(?:^|\.)(?:backbone\.)?blocks\.(\d+)\."), 1),
        ]
        return num_layers, rules

    # ResNet layer1..layer4
    if all(hasattr(back, f"layer{i}") for i in range(1, 5)):
        num_layers = 4 + 1
        rules = [
            (re.compile(r"(?:^|\.)(?:backbone\.)?layer1\."), 1),
            (re.compile(r"(?:^|\.)(?:backbone\.)?layer2\."), 2),
            (re.compile(r"(?:^|\.)(?:backbone\.)?layer3\."), 3),
            (re.compile(r"(?:^|\.)(?:backbone\.)?layer4\."), 4),
        ]
        return num_layers, rules

    # Fallback: single layer
    return 1, []


def _get_layer_id(name: str, num_layers: int, rules: List[Tuple[re.Pattern, int]]) -> int:
    lname = name.lower()
    # heads
    if ".head." in lname or lname.startswith("head."):
        return max(0, num_layers - 1)
    if ".regression_head." in lname or lname.startswith("regression_head."):
        return max(0, num_layers - 1)
    if ".classifier." in lname or lname.startswith("classifier."):
        return max(0, num_layers - 1)

    # match rules
    for pat, base in rules:
        m = pat.search(name)
        if m:
            if m.groups():
                try:
                    idx = int(m.group(1))
                    return min(base + idx, max(0, num_layers - 1))
                except Exception:
                    return base
            return base

    return 0


def param_groups_lrd(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: Iterable[str] = (),
    layer_decay: float = 1.0,
):
    """Generic LLRD param grouping.
    Returns groups with 'lr_scale' populated; set actual lr via `pg['lr'] = base_lr * lr_scale` after optimizer creation.
    """
    num_layers, rules = _infer_layering(model)
    if num_layers < 1:
        num_layers = 1

    layer_scales = [layer_decay ** (num_layers - 1 - i) for i in range(num_layers)]

    buckets: Dict[Tuple[int, bool], Dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer_id = _get_layer_id(name, num_layers, rules)
        is_no_decay = _is_no_weight_decay(name, p, no_weight_decay_list)
        key = (layer_id, is_no_decay)
        if key not in buckets:
            buckets[key] = {
                "params": [],
                "weight_decay": 0.0 if is_no_decay else weight_decay,
                "lr_scale": layer_scales[layer_id],
            }
        buckets[key]["params"].append(p)

    groups = [g for g in buckets.values() if len(g["params"]) > 0]
    if not groups:  # extreme fallback
        all_params = [p for _, p in model.named_parameters() if p.requires_grad]
        groups = [{"params": all_params, "weight_decay": weight_decay, "lr_scale": 1.0}]
    return groups

# =========================
# TabTransformer-style Multi-task head
# =========================
class MultiTaskTabTransformerHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_tasks: int,
        embed_dim: int = None,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        embed_dim = embed_dim or in_dim
        self.num_tasks = num_tasks

        self.proj = nn.Linear(in_dim, embed_dim)
        self.task_tokens = nn.Parameter(torch.zeros(num_tasks, embed_dim))  # [T, E]
        trunc_normal_(self.task_tokens, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        x = self.proj(x)  # [B, E]
        tokens = x.unsqueeze(1) + self.task_tokens.unsqueeze(0)  # [B, T, E]
        z = self.encoder(tokens)  # [B, T, E]
        z = self.norm(z)
        y = self.head(z).squeeze(-1)  # [B, T]
        return y


# =========================
# CSV Dataset (multi-task regression)
# =========================
class CSVDatasetRegression(Dataset):
    def __init__(
        self,
        csv_file,
        split,
        path_col="path",
        split_col="split",
        targets=("tc", "ldl", "hdl", "tg", "apoa", "apob"),
        transform=None,
        dropna=True,
    ):
        self.transform = transform
        self.path_col = path_col
        self.split_col = split_col
        self.targets = list(targets)

        df = pd.read_csv(csv_file)
        df = df[df[split_col].astype(str).str.lower() == str(split).lower()].copy()

        df[path_col] = df[path_col].astype(str)
        df["__exists__"] = df[path_col].apply(lambda p: Path(p).is_file())
        missing = (~df["__exists__"]).sum()
        if missing > 0:
            print(f"[WARN] {missing} files not found; dropped. Example: "
                  f"{df.loc[~df['__exists__'], path_col].head(3).tolist()}")
        df = df[df["__exists__"]].copy()

        missing_cols = [c for c in self.targets if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"CSV 缺少目标列: {missing_cols}")
        cols = [path_col] + self.targets
        df = df[cols].copy()

        if dropna:
            before = len(df)
            df = df.dropna(subset=self.targets, how="any").copy()
            after = len(df)
            if before != after:
                print(f"[INFO] Dropped {before - after} rows due to NaN in targets.")

        self.samples = []
        for _, r in df.iterrows():
            y = [float(r[c]) for c in self.targets]
            self.samples.append((r[path_col], np.array(y, dtype=np.float32)))

        self._report_stats(df)

    def _report_stats(self, df):
        print(f"[INFO] Loaded {len(self.samples)} samples for split.")
        desc = df[self.targets].describe().T
        fields = ["mean", "std", "min", "max"]
        print("[INFO] Target stats (per split):")
        for t in self.targets:
            vals = {k: float(desc.loc[t, k]) for k in fields}
            print(f"  - {t:>5s}: mean={vals['mean']:.4f}, std={vals['std']:.4f}, min={vals['min']:.4f}, max={vals['max']:.4f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(target)


def build_transforms(args):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
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
# Train one epoch
# =========================
def train_one_epoch(
    model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
    clip_grad=None, log_writer=None, args=None
):
    model.train(True)
    header = f"Epoch[{epoch}]"
    num_steps = len(data_loader)
    running_loss = 0.0

    optimizer.zero_grad(set_to_none=True)
    accum_iter = max(1, int(args.accum_iter))

    if isinstance(getattr(data_loader, "sampler", None), DistributedSampler):
        data_loader.sampler.set_epoch(epoch)

    for step, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(enabled=True, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets) / accum_iter

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

    epoch_loss = running_loss / max(1, num_steps)
    if log_writer is not None and misc.is_main_process():
        log_writer.add_scalar("loss/train", epoch_loss, epoch)
        log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
    return {"loss": epoch_loss}


# =========================
# Evaluation (MAE/RMSE/R2/PearsonR)
# =========================
@torch.no_grad()
def evaluate_regression(
    data_loader, model, device, args, epoch, mode="val", num_tasks=6, target_names=None, log_writer=None
):
    model.eval()
    crit_mse = nn.MSELoss(reduction="none")

    all_pred, all_true = [], []
    total_loss, n_batches = 0.0, 0

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(enabled=True, dtype=torch.bfloat16):
            outputs = model(images)
        batch_mse = crit_mse(outputs, targets).mean(dim=0)
        total_loss += batch_mse.mean().item()
        n_batches += 1
        all_pred.append(outputs.detach().float().cpu())
        all_true.append(targets.detach().float().cpu())

    if n_batches == 0:
        return {"loss": 0.0}, 0.0

    y_pred = torch.cat(all_pred, dim=0).numpy()
    y_true = torch.cat(all_true, dim=0).numpy()

    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    r2, pr = [], []
    for t in range(y_true.shape[1]):
        y, yhat = y_true[:, t], y_pred[:, t]
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2.append(1.0 - ss_res / ss_tot)
        pr.append(pearsonr(y, yhat)[0])
    r2 = np.array(r2)
    pr = np.array(pr)
    macro_r2 = float(np.mean(r2))
    macro_pr = float(np.mean(pr))

    names = target_names or [f"t{idx}" for idx in range(num_tasks)]
    if misc.is_main_process():
        print(f"[{mode.upper()}][epoch={epoch}] macro_R2={macro_r2:.4f} macro_PearsonR={macro_pr:.4f} "
              f"(loss_mse={total_loss/n_batches:.4f})")
        for n, a, rm, s, p in zip(names, mae, rmse, r2, pr):
            print(f"  - {n:>5s}: MAE={a:.4f} RMSE={rm:.4f} R2={s:.4f} PearsonR={p:.4f}")

    if log_writer is not None and misc.is_main_process():
        log_writer.add_scalar(f"loss/{mode}_mse", total_loss / n_batches, epoch)
        log_writer.add_scalar(f"{mode}/macro_R2", macro_r2, epoch)
        log_writer.add_scalar(f"{mode}/macro_PearsonR", macro_pr, epoch)
        for i, n in enumerate(names):
            log_writer.add_scalar(f"{mode}/MAE_{n}", mae[i], epoch)
            log_writer.add_scalar(f"{mode}/RMSE_{n}", rmse[i], epoch)
            log_writer.add_scalar(f"{mode}/R2_{n}", r2[i], epoch)
            log_writer.add_scalar(f"{mode}/PearsonR_{n}", pr[i], epoch)

    return {"loss": total_loss / n_batches}, macro_r2


# =========================
# Arg parser
# =========================
def get_args_parser():
    parser = argparse.ArgumentParser(
        "EfficientNet-B4 fine-tuning -> Multi-task Regression (CSV with absolute paths)", add_help=False
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--accum_iter", default=2, type=int)

    # Model
    parser.add_argument("--model", default="efficientnet_b4", type=str)
    parser.add_argument("--input_size", default=380, type=int)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--head_embed_dim", type=int, default=None)
    parser.add_argument("--head_depth", type=int, default=2)
    parser.add_argument("--head_num_heads", type=int, default=4)
    parser.add_argument("--head_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--head_dropout", type=float, default=0.1)

    # Optim / schedule
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=2e-3)
    parser.add_argument("--layer_decay", type=float, default=0.65)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--huber_beta", type=float, default=1.0)

    # Aug
    parser.add_argument("--color_jitter", type=float, default=None)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)

    # Finetune
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--task", default="efficientnet_b4_regression", type=str)
    parser.add_argument("--adaptation", default="finetune", choices=["finetune", "lp"])

    # CSV columns
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--path_col", type=str, default="path")
    parser.add_argument("--split_col", type=str, default="split")

    parser.add_argument(
        "--targets", nargs="+",
        default=["tc", "ldl", "hdl", "tg", "apoa", "apob"],
        help="regression targets to extract from CSV"
    )

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

    # Optuna hyper-parameter search
    parser.add_argument("--optuna_trials", type=int, default=0,
                        help="number of Optuna trials to run; 0 disables tuning")
    parser.add_argument("--optuna_timeout", type=int, default=None,
                        help="maximum search time in seconds")
    parser.add_argument("--optuna_direction", type=str, default="maximize",
                        choices=["maximize", "minimize"],
                        help="direction for Optuna study")
    parser.add_argument("--optuna_storage", type=str, default=None,
                        help="Optuna storage URL for persistent studies")
    parser.add_argument("--optuna_study_name", type=str, default=None,
                        help="name of the Optuna study")
    parser.add_argument("--optuna_resume", action="store_true", default=False,
                        help="resume Optuna study if it exists")
    parser.add_argument("--optuna_seed", type=int, default=None,
                        help="random seed for Optuna sampler")
    parser.add_argument("--optuna_pruner", type=str, default="none",
                        choices=["none", "median"],
                        help="pruning strategy for Optuna trials")
    parser.add_argument("--optuna_epochs", type=int, default=None,
                        help="override training epochs during Optuna search")
    parser.add_argument("--optuna_eval_test", action="store_true", default=False,
                        help="run test evaluation inside Optuna trials")
    parser.add_argument("--optuna_n_jobs", type=int, default=1,
                        help="number of parallel Optuna workers")
    parser.add_argument("--optuna_sampler", type=str, default="tpe",
                        choices=["tpe", "random"],
                        help="Optuna sampler to use")

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_eval", action="store_true", default=False)

    # Normalization preset (kept for compatibility)
    parser.add_argument("--norm", default="IMAGENET", type=str)

    return parser


# =========================
# Main utilities
# =========================
def _clone_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def run_experiment(args: argparse.Namespace, trial: Optional["optuna.trial.Trial"] = None) -> float:
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    if misc.is_main_process():
        print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
        print(f"{args}".replace(", ", ",\n"))
        if trial is not None:
            print(f"[Optuna] Running trial #{trial.number}")

    if args.distributed and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True

    start_time = time.time()
    max_score, best_epoch = -1e9, 0

    try:
        # Data
        train_tf, eval_tf = build_transforms(args)
        dataset_train = CSVDatasetRegression(
            args.csv_file, split="train",
            path_col=args.path_col, split_col=args.split_col,
            targets=args.targets, transform=train_tf, dropna=True
        )
        dataset_val = CSVDatasetRegression(
            args.csv_file, split="iv",
            path_col=args.path_col, split_col=args.split_col,
            targets=args.targets, transform=eval_tf, dropna=True
        )
        dataset_test = CSVDatasetRegression(
            args.csv_file, split="ev",
            path_col=args.path_col, split_col=args.split_col,
            targets=args.targets, transform=eval_tf, dropna=True
        )
        num_tasks = len(args.targets)

        # Model
        if misc.is_main_process():
            print(f"[Model] Creating {args.model} with pretrained={args.pretrained}")
        backbone = timm.create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=0,
            drop_path_rate=args.drop_path,
        )
        feat_dim = backbone.num_features
        if misc.is_main_process():
            print(f"[Model] Feature dimension: {feat_dim}")

        if args.finetune and os.path.isfile(args.finetune):
            if misc.is_main_process():
                print(f"[Load Checkpoint] {args.finetune}")
            checkpoint = torch.load(args.finetune, map_location="cpu")
            checkpoint_model = checkpoint.get("model", checkpoint)
            backbone.load_state_dict(checkpoint_model, strict=False)

        head_embed_dim = args.head_embed_dim or feat_dim
        if head_embed_dim % args.head_num_heads != 0:
            raise ValueError(
                f"head_embed_dim ({head_embed_dim}) must be divisible by head_num_heads ({args.head_num_heads})"
            )

        regression_head = MultiTaskTabTransformerHead(
            in_dim=feat_dim,
            num_tasks=num_tasks,
            embed_dim=head_embed_dim,
            depth=args.head_depth,
            num_heads=args.head_num_heads,
            mlp_ratio=args.head_mlp_ratio,
            dropout=args.head_dropout,
        )
        for m in regression_head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        class RegressionModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                features = self.backbone(x)
                return self.head(features)

            def no_weight_decay(self):
                if hasattr(self.backbone, "no_weight_decay"):
                    return {"backbone." + k for k in self.backbone.no_weight_decay()}
                return set()

        model = RegressionModel(backbone, regression_head)

        # DDP
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

        # Samplers/Loaders
        if args.distributed:
            sampler_train = DistributedSampler(
                dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
            )
            if misc.is_main_process():
                print("[INFO] Using DistributedSampler for training")
            if args.dist_eval:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
                sampler_test = DistributedSampler(dataset_test, shuffle=False)
            else:
                sampler_val = SequentialSampler(dataset_val)
                sampler_test = SequentialSampler(dataset_test)
        else:
            sampler_train = RandomSampler(dataset_train)
            sampler_val = SequentialSampler(dataset_val)
            sampler_test = SequentialSampler(dataset_test)

        log_writer = None
        if misc.is_main_process() and (not args.eval) and args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            log_suffix = args.task if trial is None else f"{args.task}"
            log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, log_suffix))

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

        # Loss
        criterion = nn.SmoothL1Loss(beta=args.huber_beta)

        # Eval-only resume
        if args.resume and args.eval:
            checkpoint = torch.load(args.resume, map_location="cpu")
            if misc.is_main_process():
                print(f"[Eval] Load checkpoint: {args.resume}")
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        # Adaptation
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

        # LR schedule
        world_size = misc.get_world_size()
        eff_bs = args.batch_size * args.accum_iter * world_size
        if args.lr is None:
            args.lr = args.blr * eff_bs / 256
        if misc.is_main_process():
            print(
                f"base lr: {args.blr:.2e}  actual lr: {args.lr:.2e}  accum_iter: {args.accum_iter}  "
                f"world_size: {world_size}  eff_bs: {eff_bs}"
            )

        # Optimizer groups with LLRD
        no_weight_decay = (
            model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, "no_weight_decay") else []
        )
        param_groups = param_groups_lrd(
            model_without_ddp, weight_decay=args.weight_decay,
            no_weight_decay_list=no_weight_decay, layer_decay=args.layer_decay,
        )
        for g in param_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        # Apply per-group lr scales
        for pg in optimizer.param_groups:
            scale = pg.get("lr_scale", 1.0)
            pg["lr"] = args.lr * scale

        loss_scaler = NativeScaler()
        if misc.is_main_process():
            print(f"criterion = {criterion}")

        # (optional) restore training state
        misc.load_model(args=args, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler)

        # Eval only path
        if args.eval:
            _val_stats, _val_r2 = evaluate_regression(
                data_loader_test, model, device, args, epoch=0, mode="test",
                num_tasks=num_tasks, target_names=args.targets, log_writer=log_writer
            )
            if log_writer is not None:
                log_writer.flush()
                log_writer.close()
            return _val_r2

        if misc.is_main_process():
            desc = f"Trial {trial.number}" if trial is not None else "Epochs"
            epoch_iter = tqdm(range(args.start_epoch, args.epochs), desc=desc)
        else:
            epoch_iter = range(args.start_epoch, args.epochs)

        for epoch in epoch_iter:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                clip_grad=args.clip_grad, log_writer=log_writer, args=args
            )

            if epoch % 2 == 0:
                _, _ = evaluate_regression(
                    data_loader_train, model, device, args, epoch, mode="train",
                    num_tasks=num_tasks, target_names=args.targets, log_writer=log_writer
                )

            val_stats, val_score = evaluate_regression(
                data_loader_val, model, device, args, epoch, mode="val",
                num_tasks=num_tasks, target_names=args.targets, log_writer=log_writer
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
                print(f"[VAL] Best epoch={best_epoch}, Best macro_R2={max_score:.4f}")

                if log_writer is not None:
                    log_writer.add_scalar("loss/val_epoch_mse", val_stats["loss"], epoch)
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

        if trial is None:
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

            _test_stats, _test_r2 = evaluate_regression(
                data_loader_test, model, device, args, -1, mode="test",
                num_tasks=num_tasks, target_names=args.targets, log_writer=None
            )
            if misc.is_main_process():
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print(f"Training time {total_time_str}")
                print(f"[TEST] macro_R2={_test_r2:.4f}")
        else:
            if misc.is_main_process():
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print(f"[Optuna][Trial {trial.number}] best macro_R2={max_score:.4f} (time={total_time_str})")

        return max_score

    finally:
        if args.distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run_optuna(args: argparse.Namespace) -> None:
    if optuna is None:
        raise ImportError("Optuna is required for hyper-parameter tuning but is not installed.")

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

    study_kwargs = {"direction": args.optuna_direction, "sampler": sampler}
    if pruner is not None:
        study_kwargs["pruner"] = pruner
    if args.optuna_storage:
        study_kwargs["storage"] = args.optuna_storage
        if args.optuna_study_name:
            study_kwargs["study_name"] = args.optuna_study_name
        study_kwargs["load_if_exists"] = args.optuna_resume
    elif args.optuna_study_name:
        study_kwargs["study_name"] = args.optuna_study_name

    study = optuna.create_study(**study_kwargs)

    base_output = Path(args.output_dir) if args.output_dir else None
    base_log = Path(args.log_dir) if args.log_dir else None
    if base_output is not None:
        base_output.mkdir(parents=True, exist_ok=True)
    if base_log is not None:
        base_log.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_args = _clone_args(args)
        trial_args.optuna_trials = 0
        trial_args.optuna_timeout = None
        trial_args.optuna_direction = args.optuna_direction
        trial_args.optuna_eval_test = args.optuna_eval_test
        trial_args.optuna_pruner = args.optuna_pruner
        trial_args.optuna_sampler = args.optuna_sampler
        trial_args.optuna_n_jobs = 1

        # Sample hyper-parameters
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 0.1, log=True)
        layer_decay = trial.suggest_float("layer_decay", 0.4, 0.95)
        drop_path = trial.suggest_float("drop_path", 0.0, 0.4)
        head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5)
        head_depth = trial.suggest_int("head_depth", 1, 4)
        head_mlp_ratio = trial.suggest_float("head_mlp_ratio", 1.0, 4.0)
        head_num_heads = trial.suggest_categorical("head_num_heads", [2, 4, 8])
        huber_beta = trial.suggest_float("huber_beta", 0.5, 2.0)

        trial_args.lr = lr
        trial_args.blr = lr
        trial_args.weight_decay = weight_decay
        trial_args.layer_decay = layer_decay
        trial_args.drop_path = drop_path
        trial_args.head_dropout = head_dropout
        trial_args.head_depth = head_depth
        trial_args.head_mlp_ratio = head_mlp_ratio
        trial_args.head_num_heads = head_num_heads
        trial_args.huber_beta = huber_beta
        trial_args.savemodel = bool(args.optuna_eval_test)
        trial_args.task = f"{args.task}_trial{trial.number}"

        if args.optuna_epochs is not None:
            trial_args.epochs = args.optuna_epochs

        if base_output is not None:
            trial_args.output_dir = str(base_output)
        if base_log is not None:
            trial_args.log_dir = str(base_log)

        trial_args.seed = (args.seed if args.seed is not None else 0) + trial.number

        return run_experiment(trial_args, trial=trial)

    study.optimize(
        objective,
        n_trials=args.optuna_trials,
        timeout=args.optuna_timeout,
        n_jobs=args.optuna_n_jobs,
        gc_after_trial=True,
    )

    if misc.is_main_process():
        print("[Optuna] Best trial:")
        print(f"  value: {study.best_value}")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        if base_output is not None:
            summary_path = base_output / "optuna_best.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({"value": study.best_value, "params": study.best_trial.params}, f, ensure_ascii=False, indent=2)


def main(args: argparse.Namespace) -> None:
    if args.optuna_trials and args.optuna_trials > 0:
        run_optuna(args)
    else:
        if args.output_dir and (not os.path.exists(args.output_dir)):
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.log_dir and (not os.path.exists(args.log_dir)):
            Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        run_experiment(args)


if __name__ == "__main__":
    parsed = get_args_parser()
    args = parsed.parse_args()
    main(args)
