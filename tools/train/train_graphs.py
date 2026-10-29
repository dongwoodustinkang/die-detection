import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로만 저장 (Linux 서버에서도 동작)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_history_plot(history, best_epoch, path):
    """epoch별 Training/Validation Acc·Loss를 한 그래프로 저장한다."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], color="tab:blue", label="Training Acc")
    ax.plot(epochs, history["val_acc"], color="tab:blue", linestyle="--", label="Validation Acc")
    ax.plot(epochs, history["train_loss"], color="tab:orange", label="Training Loss")
    ax.plot(epochs, history["val_loss"], color="tab:orange", linestyle="--", label="Validation Loss")
    ax.axvline(best_epoch, color="gray", linestyle=":", label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Value")
    ax.set_title(path.stem)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)