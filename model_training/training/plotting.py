import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def plot_loss_curve(
    run_dir: str,
    output_path: Optional[str] = None,
    component: str = "intent_ffn",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot_loss_curve")
        return

    run_dir = Path(run_dir)
    epochs_file = run_dir / f"{component}_epochs.json"
    if not epochs_file.exists():
        logger.warning(f"No epoch data at {epochs_file}")
        return

    with open(epochs_file) as f:
        epochs = json.load(f)

    if not epochs:
        return

    train_losses = [e.get("train_loss") for e in epochs if "train_loss" in e]
    val_losses = [e.get("val_loss") for e in epochs if "val_loss" in e]
    epochs_x = list(range(1, len(epochs) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    if train_losses:
        ax.plot(epochs_x[:len(train_losses)], train_losses, label="Train Loss", marker="o")
    if val_losses:
        ax.plot(epochs_x[:len(val_losses)], val_losses, label="Validation Loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{component.replace('_', ' ').title()} Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_path) if output_path else run_dir / "loss_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Loss curve saved to {out}")


def plot_accuracy_curve(
    run_dir: str,
    output_path: Optional[str] = None,
    component: str = "intent_ffn",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot_accuracy_curve")
        return

    run_dir = Path(run_dir)
    epochs_file = run_dir / f"{component}_epochs.json"
    if not epochs_file.exists():
        return

    with open(epochs_file) as f:
        epochs = json.load(f)

    if not epochs:
        return

    train_accs = []
    val_accs = []
    train_acc_key = next((k for k in epochs[0] if "train" in k and "acc" in k), None)
    val_acc_key = next((k for k in epochs[0] if "val" in k and "acc" in k), None) if len(epochs) > 0 else None

    for e in epochs:
        if train_acc_key:
            train_accs.append(e.get(train_acc_key, 0))
        if val_acc_key:
            val_accs.append(e.get(val_acc_key, 0))

    epochs_x = list(range(1, len(epochs) + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    if train_accs:
        ax.plot(epochs_x[:len(train_accs)], train_accs, label="Train Accuracy", marker="o")
    if val_accs:
        ax.plot(epochs_x[:len(val_accs)], val_accs, label="Validation Accuracy", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{component.replace('_', ' ').title()} Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_path) if output_path else run_dir / "accuracy_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Accuracy curve saved to {out}")


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping plot_confusion_matrix")
        return

    import numpy as np
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.2), 8))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, int(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {output_path}")
