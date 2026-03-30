"""
benchmark.py – Entry point for training & evaluating image-classification models.

Usage:
    python benchmark.py \
        --model resnet50 \
        --dataset /path/to/dataset   (or use --download for built-in datasets) \
        --loss ce \
        --epochs 100 \
        --batch_size 16 \
        --size 224 \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --workers 2 \
        --seed 42 \
        --save_dir ./results
"""

import argparse
import os
import sys
import time
import copy
import random
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset   import build_splits, JujubeDataset
from utils.trainer   import train_one_epoch, evaluate
from utils.metrics   import compute_metrics, measure_fps
from utils.plots     import plot_history, plot_confusion_matrix, save_results_csv
from utils.download  import download_dataset

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Jujube / Generic Image-Classification Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--dataset",     type=str, default=None,
                   help="Path to dataset root (folder-per-class layout).")
    p.add_argument("--download",    type=str, default=None,
                   help="Download dataset.  Format: 'kaggle:<slug>' or 'gdrive:<id>'.")
    p.add_argument("--classes",     type=str, default=None,
                   help="Comma-separated class names (auto-detected if omitted).")
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio",   type=float, default=0.15)

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model",       type=str,  default="resnet50",
                   help="Model name (matches models/<name>.py  build_model() function).")
    p.add_argument("--pretrained",  action="store_true", default=True,
                   help="Use ImageNet pre-trained weights when available.")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--loss",        type=str,  default="ce",
                   help="Loss name (matches losses/<name>.py  build_loss() function).")
    p.add_argument("--epochs",      type=int,  default=100)
    p.add_argument("--batch_size",  type=int,  default=16)
    p.add_argument("--size",        type=int,  default=224,  help="Input image size.")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--lr_step",     type=int,  default=10,  help="StepLR step size.")
    p.add_argument("--lr_gamma",    type=float, default=0.1, help="StepLR gamma.")
    p.add_argument("--workers",     type=int,  default=2)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--augment",     action="store_true", default=False,
                   help="Enable random augmentation on the training split.")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--save_dir",    type=str,  default="./results")
    p.add_argument("--no_plot",     action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic model / loss loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, num_classes: int, pretrained: bool):
    """
    Try models/<model_name>.py  → build_model(num_classes, pretrained).
    Fall back to torchvision built-ins.
    """
    module_path = Path("models") / f"{model_name}.py"
    if module_path.exists():
        spec   = importlib.util.spec_from_file_location(model_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_model(num_classes=num_classes, pretrained=pretrained)

    # ── torchvision fallback ──────────────────────────────────────────────────
    import torchvision.models as tv
    tv_factory = {
        "resnet18":   tv.resnet18,
        "resnet34":   tv.resnet34,
        "resnet50":   tv.resnet50,
        "resnet101":  tv.resnet101,
        "vgg16":      tv.vgg16,
        "vgg19":      tv.vgg19,
        "efficientnet_b0": tv.efficientnet_b0,
        "efficientnet_b3": tv.efficientnet_b3,
        "mobilenet_v3_small": tv.mobilenet_v3_small,
        "mobilenet_v3_large": tv.mobilenet_v3_large,
        "densenet121": tv.densenet121,
        "densenet201": tv.densenet201,
        "convnext_tiny":  tv.convnext_tiny,
        "convnext_small": tv.convnext_small,
        "vit_b_16": tv.vit_b_16,
        "swin_t":   tv.swin_t,
    }
    if model_name not in tv_factory:
        raise ValueError(
            f"Unknown model '{model_name}'.  "
            f"Create models/{model_name}.py with a build_model() function, "
            f"or use one of: {list(tv_factory.keys())}"
        )

    weights = "DEFAULT" if pretrained else None
    model   = tv_factory[model_name](weights=weights)

    # Replace the classification head
    if hasattr(model, "fc"):                         # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    elif hasattr(model, "heads"):                    # ViT
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model


def load_loss(loss_name: str):
    """
    Try losses/<loss_name>.py  → build_loss().
    Fall back to common built-ins.
    """
    module_path = Path("losses") / f"{loss_name}.py"
    if module_path.exists():
        spec   = importlib.util.spec_from_file_location(loss_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_loss()

    builtin = {
        "ce":         nn.CrossEntropyLoss(),
        "bce":        nn.BCEWithLogitsLoss(),
        "mse":        nn.MSELoss(),
        "label_smooth": nn.CrossEntropyLoss(label_smoothing=0.1),
    }
    if loss_name not in builtin:
        raise ValueError(
            f"Unknown loss '{loss_name}'.  "
            f"Create losses/{loss_name}.py with a build_loss() function, "
            f"or use one of: {list(builtin.keys())}"
        )
    return builtin[loss_name]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Model  : {args.model}")
    print(f"  Loss   : {args.loss}")
    print(f"{'='*60}\n")

    # ── Download (optional) ───────────────────────────────────────────────────
    dataset_root = args.dataset
    if args.download:
        dataset_root = download_dataset(args.download, dest="./data")

    if dataset_root is None:
        sys.exit("ERROR: Provide --dataset <path> or --download <source>.")

    # ── Data ──────────────────────────────────────────────────────────────────
    class_names = (
        [c.strip() for c in args.classes.split(",")]
        if args.classes
        else sorted([
            d for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
        ])
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    train_s, val_s, test_s = build_splits(
        dataset_root, class_names,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])
    tf_eval = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(), normalize,
    ])
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(), normalize,
    ]) if args.augment else tf_eval

    train_loader = DataLoader(JujubeDataset(train_s, tf_train),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(JujubeDataset(val_s, tf_eval),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(JujubeDataset(test_s, tf_eval),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"  Train: {len(train_s)}  |  Val: {len(val_s)}  |  Test: {len(test_s)}\n")

    # ── Model & Loss ──────────────────────────────────────────────────────────
    model     = load_model(args.model, num_classes, args.pretrained).to(device)
    criterion = load_loss(args.loss)

    print(f"  Params: {count_params(model):.2f}M")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    # ── Save dir ──────────────────────────────────────────────────────────────
    run_dir = os.path.join(args.save_dir, f"{args.model}_{args.loss}")
    os.makedirs(run_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    best_model_wt = copy.deepcopy(model.state_dict())
    start_time    = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc       = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            best_model_wt = copy.deepcopy(model.state_dict())
            ckpt_path = os.path.join(run_dir, f"{args.model}_best.pth")
            torch.save({
                "epoch": epoch, "model_state": best_model_wt,
                "val_acc": best_val_acc, "model_name": args.model,
                "class_names": class_names,
            }, ckpt_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
            f"Val   Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {time.time()-t0:.1f}s"
        )

    print(f"\nTraining done in {(time.time()-start_time)/60:.1f} min")
    print(f"Best Val Acc: {best_val_acc:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.load_state_dict(best_model_wt)
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    report = compute_metrics(test_labels, test_preds, class_names)

    # ── FLOPs ─────────────────────────────────────────────────────────────────
    try:
        from thop import profile
        dummy = torch.randn(1, 3, args.size, args.size).to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        flops_G  = flops / 1e9
        params_M = params / 1e6
    except ImportError:
        print("  (thop not installed – skipping FLOPs count)")
        flops_G  = float("nan")
        params_M = count_params(model)

    # ── FPS ───────────────────────────────────────────────────────────────────
    fps_bs1,  lat_ms = measure_fps(model, device, args.size, batch_size=1)
    fps_bs32, _      = measure_fps(model, device, args.size, batch_size=32)

    # ── Print summary ─────────────────────────────────────────────────────────
    macro_p  = report["macro avg"]["precision"]
    macro_r  = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"\n{'='*70}")
    print(f"  {args.model.upper()}  [{args.loss}]  —  TEST RESULTS")
    print(f"{'='*70}")
    print(f"  {'Accuracy':<14}{'Precision':<14}{'Recall':<14}{'F1':<14}{'FLOPS(G)':<12}{'Params(M)'}")
    print(f"  {'-'*67}")
    print(f"  {test_acc:<14.4f}{macro_p:<14.4f}{macro_r:<14.4f}{macro_f1:<14.4f}{flops_G:<12.3f}{params_M:.2f}")

    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<15}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support'}")
    print(f"  {'-'*55}")
    for cn in class_names:
        p = report[cn]["precision"]
        r = report[cn]["recall"]
        f = report[cn]["f1-score"]
        s = int(report[cn]["support"])
        print(f"  {cn:<15}{p:<12.4f}{r:<12.4f}{f:<12.4f}{s}")

    print(f"\n  FPS (batch=1) : {fps_bs1:.1f} img/s  |  Latency: {lat_ms:.2f} ms/img")
    print(f"  FPS (batch=32): {fps_bs32:.1f} img/s")

    # ── Save artefacts ────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_history(history, args.model, run_dir)
        plot_confusion_matrix(test_labels, test_preds, class_names, args.model, run_dir)

    save_results_csv(
        run_dir, args.model, args.loss,
        test_acc, macro_p, macro_r, macro_f1,
        flops_G, params_M, fps_bs1, lat_ms,
    )
    print(f"\n  Results saved → {run_dir}\n")


if __name__ == "__main__":
    main()
