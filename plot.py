"""utils/plots.py – training curves, confusion matrix, CSV summary."""

import os
import csv

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_history(history, model_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val",   markersize=3)
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val",   markersize=3)
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    out = os.path.join(save_dir, f"{model_name}_curves.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved training curves → {out}")


def plot_confusion_matrix(labels, preds, class_names, model_name, save_dir):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    out = os.path.join(save_dir, f"{model_name}_cm.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Saved confusion matrix → {out}")


def save_results_csv(
    save_dir, model_name, loss_name,
    acc, precision, recall, f1,
    flops_g, params_m, fps_bs1, latency_ms,
):
    csv_path = os.path.join(save_dir, "results_summary.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "model", "loss", "accuracy", "precision", "recall", "f1",
                "flops_G", "params_M", "fps_bs1", "latency_ms",
            ])
        writer.writerow([
            model_name, loss_name,
            f"{acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}",
            f"{flops_g:.3f}", f"{params_m:.2f}", f"{fps_bs1:.1f}", f"{latency_ms:.2f}",
        ])
    print(f"  Saved results CSV → {csv_path}")
