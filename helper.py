import matplotlib.pyplot as plt
from IPython import display
import os
import cv2
import numpy as np
from PIL import Image
import io

plt.ion()
plt.style.use("dark_background")

# Global figure setup
_fig, _ax = plt.subplots(figsize=(15, 9))
_ax.set_title("Snake AI Training Progress", fontsize=16, color="#00FFCC")
_ax.set_xlabel("Number of Games", fontsize=13, color="#AAAAAA")
_ax.set_ylabel("Score", fontsize=13, color="#AAAAAA")
_ax.grid(True, linestyle="--", alpha=0.3)

# Initialize plot lines
_score_line, = _ax.plot([], [], label="Score per Game", color="#00FFCC", linewidth=2)
_mean_line, = _ax.plot([], [], label="Average Score", color="#FF8800", linewidth=2)
_ax.legend(facecolor="#111111", edgecolor="#333333", fontsize=12)

# --- Video Writer for Plot ---
os.makedirs("videos", exist_ok=True)
plot_video_path = "videos/training_plot.mp4"
_plot_writer = cv2.VideoWriter(plot_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (1500, 900))


def plot(scores, mean_scores):
    """
    In-place update + video frame capture for the training progress.
    """
    _score_line.set_data(range(len(scores)), scores)
    _mean_line.set_data(range(len(mean_scores)), mean_scores)
    _ax.set_xlim(0, len(scores) + 5)
    y_max = max(max(scores, default=0), max(mean_scores, default=0)) + 5
    _ax.set_ylim(0, y_max)

    for artist in list(_ax.texts):
        artist.remove()

    if len(scores) > 0:
        _ax.text(len(scores) - 1, scores[-1] + 1, f"{scores[-1]}", color="#00FFCC", fontsize=12)
    if len(mean_scores) > 0:
        _ax.text(len(mean_scores) - 1, mean_scores[-1] + 1, f"{mean_scores[-1]:.2f}", color="#FF8800", fontsize=12)

    display.clear_output(wait=True)
    display.display(_fig)
    plt.pause(0.05)

    # Save PNG snapshot
    os.makedirs("plots", exist_ok=True)
    _fig.savefig("plots/training_progress.png", dpi=120, bbox_inches="tight")

    # --- Safe frame capture for video ---
    buf = io.BytesIO()
    _fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    frame = np.array(img)
    buf.close()

    # Resize safely for video dimensions
    frame_resized = cv2.resize(frame, (1500, 900))
    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
    _plot_writer.write(frame_bgr)



def close_plot_video():
    """Safely release video writer after training."""
    _plot_writer.release()
    plt.close(_fig)
