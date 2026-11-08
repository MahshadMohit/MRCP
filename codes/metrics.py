import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    brier = brier_score_loss(y_true, y_prob)

    return {
        "AUROC": auroc,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "Brier": brier,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }

def get_fn_fp_indices(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    return fn_idx, fp_idx

def plot_roc_curve(y_true, y_prob, save_path=None, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_calibration_curve(y_true, y_prob, n_bins=10, save_path=None, title="Calibration Curve"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            if logits.shape[-1] > 1:
                probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits).squeeze()
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    fn_idx, fp_idx = get_fn_fp_indices(y_true, y_prob, threshold=threshold)
    return metrics, y_true, y_prob, fn_idx, fp_idx