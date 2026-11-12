"""
generate_dynamic_plots.py — Vẽ lại các biểu đồ trực quan hóa đồng bộ 100% với final_metrics.json
và confusion_matrix.csv.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def plot_confusion_matrix(metrics: dict) -> None:
    cm_val = metrics["confusion_matrix"]
    if isinstance(cm_val, dict) and "matrix" in cm_val:
        cm = np.array(cm_val["matrix"])
    else:
        cm = np.array(cm_val)
    total_count = metrics["total_samples"]
    target_names = ["NEGATIVE", "POSITIVE"]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total_count) * 100
            text_color = "white" if val > 45 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontsize=13, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(target_names, fontsize=11, fontweight="bold")
    ax.set_yticklabels(target_names, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label (Nhãn dự đoán)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label (Nhãn thực tế)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix — DistilBERT Fine-tuned (Test Set)", fontsize=12, fontweight="bold", pad=18)
    plt.tight_layout()
    out_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] Đã xuất: {out_path.name}")


"""
generate_dynamic_plots.py — Vẽ lại các biểu đồ trực quan hóa đồng bộ 100% với final_metrics.json
và confusion_matrix.csv. Hỗ trợ cả biểu đồ độc lập từng phần (không gộp chung) và biểu đồ tổng hợp.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def plot_confusion_matrix(metrics: dict) -> None:
    cm_val = metrics["confusion_matrix"]
    if isinstance(cm_val, dict) and "matrix" in cm_val:
        cm = np.array(cm_val["matrix"])
    else:
        cm = np.array(cm_val)
    total_count = metrics["total_samples"]
    target_names = ["NEGATIVE", "POSITIVE"]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = (val / total_count) * 100
            text_color = "white" if val > 45 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=text_color, fontsize=13, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(target_names, fontsize=11, fontweight="bold")
    ax.set_yticklabels(target_names, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label (Nhãn dự đoán)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label (Nhãn thực tế)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Confusion Matrix — DistilBERT Fine-tuned (Test Set)", fontsize=12, fontweight="bold", pad=18)
    plt.tight_layout()
    out_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] Đã xuất: {out_path.name}")


def plot_test_benchmark_alone(metrics: dict) -> None:
    """Biểu đồ độc lập: Đối chiếu hiệu năng Benchmark vs. Wilson CI."""
    test_acc = metrics["test_accuracy"]
    ci = metrics["wilson_95_ci"]
    ci_lower = ci[0]
    ci_upper = ci[1]

    if "majority_baseline_train_derived" in metrics:
        maj_acc = metrics["majority_baseline_train_derived"]["accuracy"]
    else:
        maj_acc = metrics.get("majority_baseline", 52.0)

    base_val = metrics.get("untrained_head_val_baseline")
    if base_val is None:
        base_val = metrics.get("baseline_val_metrics", {}).get("accuracy", 44.0)
    if base_val < 1.0:
        base_val *= 100
    if base_val < 1.0:
        base_val *= 100

    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Untrained Baseline\n(Đo trên Val Set)", "Train Majority\n(Áp trên Test Set)", "DistilBERT\nFine-tuned (Test Set)"]
    accs = [base_val, maj_acc, test_acc]
    bars = ax.bar(models, accs, color=["#7f8c8d", "#e67e22", "#27ae60"], width=0.45, edgecolor="#2c3e50", linewidth=1.2)
    err_l = test_acc - ci_lower
    err_u = ci_upper - test_acc
    ax.errorbar(2, test_acc, yerr=[[err_l], [err_u]], fmt="none", ecolor="#1b4d3e", elinewidth=2.5, capsize=6, capthick=2)
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Đối Chiếu Hiệu Năng: Baseline vs. Fine-tuned", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 2, f"{a:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(2, ci_upper + 3, f"95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]", ha="center", va="bottom", fontsize=9, color="#1b4d3e", fontweight="bold")

    plt.tight_layout()
    out_path = REPORTS_DIR / "test_benchmark_comparison.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] Đã xuất: {out_path.name}")


def plot_test_per_class_alone(metrics: dict) -> None:
    """Biểu đồ độc lập: Chỉ số phân loại từng lớp NEGATIVE vs POSITIVE."""
    clf_report = metrics["classification_report"]
    metrics_names = ["Precision", "Recall", "F1-Score"]
    neg_vals = [clf_report["NEGATIVE"]["precision"] * 100, clf_report["NEGATIVE"]["recall"] * 100, clf_report["NEGATIVE"]["f1-score"] * 100]
    pos_vals = [clf_report["POSITIVE"]["precision"] * 100, clf_report["POSITIVE"]["recall"] * 100, clf_report["POSITIVE"]["f1-score"] * 100]
    neg_sup = int(clf_report["NEGATIVE"]["support"])
    pos_sup = int(clf_report["POSITIVE"]["support"])

    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(metrics_names))
    w = 0.35
    rects1 = ax.bar(x_pos - w / 2, neg_vals, w, label=f"NEGATIVE ({neg_sup} mẫu)", color="#34495e", alpha=0.85)
    rects2 = ax.bar(x_pos + w / 2, pos_vals, w, label=f"POSITIVE ({pos_sup} mẫu)", color="#3498db", alpha=0.85)
    ax.set_ylabel("Tỷ lệ (%)", fontsize=11, fontweight="bold")
    ax.set_title("Đánh Giá Chi Tiết Từng Lớp: NEGATIVE vs POSITIVE", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="#f8f9fa")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for r in rects1 + rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    plt.tight_layout()
    out_path = REPORTS_DIR / "test_per_class_metrics.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [OK] Đã xuất: {out_path.name}")


def plot_test_comparison(metrics: dict) -> None:
    test_acc = metrics["test_accuracy"]
    ci = metrics["wilson_95_ci"]
    ci_lower = ci[0]
    ci_upper = ci[1]

    if "majority_baseline_train_derived" in metrics:
        maj_acc = metrics["majority_baseline_train_derived"]["accuracy"]
    else:
        maj_acc = metrics.get("majority_baseline", 52.0)

    base_val = metrics.get("untrained_head_val_baseline")
    if base_val is None:
        base_val = metrics.get("baseline_val_metrics", {}).get("accuracy", 44.0)
    if base_val < 1.0:
        base_val *= 100
    if base_val < 1.0:
        base_val *= 100

    clf_report = metrics["classification_report"]

    fig_comp, (ax_comp1, ax_comp2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar 1: So sánh mốc baseline
    models = ["Untrained Baseline\n(Đo trên Val Set)", "Train Majority\n(Áp trên Test Set)", "DistilBERT\nFine-tuned (Test Set)"]
    accs = [base_val, maj_acc, test_acc]
    bars_comp = ax_comp1.bar(models, accs, color=["#7f8c8d", "#e67e22", "#27ae60"], width=0.5, edgecolor="#2c3e50", linewidth=1.2)
    err_l = test_acc - ci_lower
    err_u = ci_upper - test_acc
    ax_comp1.errorbar(2, test_acc, yerr=[[err_l], [err_u]], fmt="none", ecolor="#1b4d3e", elinewidth=2.5, capsize=6, capthick=2)
    ax_comp1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax_comp1.set_title("Đối Chiếu Hiệu Năng: Baseline vs. Fine-tuned", fontsize=12, fontweight="bold", pad=12)
    ax_comp1.set_ylim(0, 100)
    ax_comp1.grid(axis="y", linestyle="--", alpha=0.5)

    for b, a in zip(bars_comp, accs):
        ax_comp1.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 2, f"{a:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_comp1.text(2, ci_upper + 3, f"95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]", ha="center", va="bottom", fontsize=9, color="#1b4d3e", fontweight="bold")

    # Bar 2: Per-class metrics
    metrics_names = ["Precision", "Recall", "F1-Score"]
    neg_vals = [clf_report["NEGATIVE"]["precision"] * 100, clf_report["NEGATIVE"]["recall"] * 100, clf_report["NEGATIVE"]["f1-score"] * 100]
    pos_vals = [clf_report["POSITIVE"]["precision"] * 100, clf_report["POSITIVE"]["recall"] * 100, clf_report["POSITIVE"]["f1-score"] * 100]
    neg_sup = int(clf_report["NEGATIVE"]["support"])
    pos_sup = int(clf_report["POSITIVE"]["support"])
    x_pos = np.arange(len(metrics_names))
    w = 0.35
    rects1 = ax_comp2.bar(x_pos - w / 2, neg_vals, w, label=f"NEGATIVE ({neg_sup} mẫu)", color="#34495e", alpha=0.85)
    rects2 = ax_comp2.bar(x_pos + w / 2, pos_vals, w, label=f"POSITIVE ({pos_sup} mẫu)", color="#3498db", alpha=0.85)
    ax_comp2.set_ylabel("Tỷ lệ (%)", fontsize=11, fontweight="bold")
    ax_comp2.set_title("Đánh Giá Chi Tiết Từng Lớp: NEGATIVE vs POSITIVE", fontsize=12, fontweight="bold", pad=12)
    ax_comp2.set_xticks(x_pos)
    ax_comp2.set_xticklabels(metrics_names, fontsize=10, fontweight="bold")
    ax_comp2.set_ylim(0, 105)
    ax_comp2.legend(frameon=True, facecolor="#f8f9fa")
    ax_comp2.grid(axis="y", linestyle="--", alpha=0.5)

    for r in rects1 + rects2:
        h = r.get_height()
        ax_comp2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=8.5)

    plt.tight_layout()
    out_path = REPORTS_DIR / "test_comparison_and_per_class.png"
    fig_comp.savefig(out_path, dpi=120)
    plt.close(fig_comp)
    print(f"  [OK] Đã xuất: {out_path.name}")

    # Đồng thời xuất 2 biểu đồ độc lập
    plot_test_benchmark_alone(metrics)
    plot_test_per_class_alone(metrics)


def main() -> None:
    with open(REPORTS_DIR / "final_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    plot_confusion_matrix(metrics)
    plot_test_comparison(metrics)


if __name__ == "__main__":
    main()


