"""
Hooke's Law TensorFlow Linear Regression Model
Learns: Length = slope * Weight + intercept
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

from data import get_dataset, get_spring_constant, get_initial_length

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dark-theme plot style ─────────────────────────────────────────────────────
DARK_BG      = "#0f0e17"
CARD_BG      = "#1a1933"
CYAN         = "#00d4ff"
GREEN        = "#00ff88"
GOLD         = "#ffd700"
RED          = "#ff4757"
TEXT_PRIMARY = "#fffffe"
TEXT_SEC     = "#a7a9be"

def _apply_dark_style(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_SEC, labelsize=10)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    ax.title.set_color(TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d2b55")
    ax.grid(True, alpha=0.3, color="#2d2b55", linestyle="--")


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class TrainResult:
    slope: float
    intercept: float
    final_loss: float
    epochs: int
    loss_history: List[float]
    loss_curve_path: str
    fitting_path: str
    predicted_length: Optional[float] = None


# ── Global model state ────────────────────────────────────────────────────────
_model: Optional[tf.keras.Model] = None
_train_result: Optional[TrainResult] = None


class HookesLawModel:
    """TensorFlow linear regression model for Hooke's Law (F = kx)."""

    def __init__(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, input_shape=[1], name="linear")
        ])
        self._model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss="mean_squared_error"
        )

    def fit(self, x, y, epochs=500, **kwargs):
        return self._model.fit(x, y, epochs=epochs, **kwargs)

    def predict(self, x, **kwargs):
        return self._model.predict(x, **kwargs)

    def get_weights(self):
        return self._model.layers[0].get_weights()


def _build_model() -> HookesLawModel:
    return HookesLawModel()


def is_trained() -> bool:
    return _model is not None and _train_result is not None


def get_model_params() -> dict:
    if _model is None:
        return {}
    w = _model.get_weights()
    slope     = float(w[0].flatten()[0])
    intercept = float(w[1].flatten()[0])
    return {"slope": slope, "intercept": intercept}


def get_last_result() -> Optional[TrainResult]:
    return _train_result


# ── PNG generators ────────────────────────────────────────────────────────────
def _save_loss_curve(loss_history: List[float]) -> str:
    path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    epochs_arr = np.arange(1, len(loss_history) + 1)
    ax.plot(epochs_arr, loss_history, color=CYAN, linewidth=2, label="Training Loss (MSE)")
    ax.fill_between(epochs_arr, loss_history, alpha=0.15, color=CYAN)

    # Annotate final loss
    final = loss_history[-1]
    ax.annotate(
        f"Final: {final:.4f}",
        xy=(epochs_arr[-1], final),
        xytext=(-80, 20),
        textcoords="offset points",
        color=GOLD,
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss (log scale)", fontsize=12)
    ax.set_title("Training Loss Curve — Hooke's Law TF Model", fontsize=14, fontweight="bold", pad=15)
    ax.legend(facecolor=CARD_BG, edgecolor="#2d2b55", labelcolor=TEXT_PRIMARY)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


def _save_spring_fitting(slope: float, intercept: float, new_mass: Optional[float] = None) -> str:
    path = os.path.join(OUTPUT_DIR, "spring_fitting.png")
    weights, measured_lengths, true_lengths = get_dataset()

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    # Measured data
    ax.scatter(weights, measured_lengths, color=CYAN, s=80, zorder=5,
               label="Measured Data (noisy)", edgecolors="white", linewidths=0.5)

    # True law
    x_line = np.linspace(-0.5, (new_mass or 12) + 1, 200)
    ax.plot(x_line, 2.0 * x_line + 10.0, color=GREEN, linestyle="--",
            linewidth=2, label="True Law  (y = 2x + 10)", zorder=4)

    # AI prediction line
    pred_line = slope * x_line + intercept
    ax.plot(x_line, pred_line, color=RED, linewidth=2.5, zorder=6,
            label=f"AI Model  (y = {slope:.2f}x + {intercept:.2f})")

    # New mass prediction star
    if new_mass is not None:
        pred_y = slope * new_mass + intercept
        ax.scatter([new_mass], [pred_y], color=GOLD, s=250, zorder=7,
                   marker="*", label=f"Prediction @ {new_mass} kg → {pred_y:.2f} cm")
        ax.axvline(x=new_mass, color=GOLD, linestyle=":", alpha=0.5)
        ax.axhline(y=pred_y, color=GOLD, linestyle=":", alpha=0.5)

    ax.set_xlabel("Mass (kg)", fontsize=12)
    ax.set_ylabel("Spring Length (cm)", fontsize=12)
    ax.set_title("Spring Experiment — Hooke's Law Regression", fontsize=14, fontweight="bold", pad=15)
    ax.legend(facecolor=CARD_BG, edgecolor="#2d2b55", labelcolor=TEXT_PRIMARY, fontsize=10)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ── Public API ────────────────────────────────────────────────────────────────
def train_model(epochs: int = 500, new_mass_kg: Optional[float] = None) -> TrainResult:
    global _model, _train_result

    weights, measured_lengths, _ = get_dataset()
    _model = _build_model()

    history = _model.fit(
        weights, measured_lengths,
        epochs=epochs,
        verbose=0,
        batch_size=len(weights)
    )

    params = get_model_params()
    slope     = params["slope"]
    intercept = params["intercept"]
    loss_hist = [float(v) for v in history.history["loss"]]
    final_loss = loss_hist[-1]

    loss_path    = _save_loss_curve(loss_hist)
    fitting_path = _save_spring_fitting(slope, intercept, new_mass_kg)

    predicted = None
    if new_mass_kg is not None:
        predicted = float(_model.predict(
            np.array([[new_mass_kg]]), verbose=0
        )[0][0])

    _train_result = TrainResult(
        slope=slope,
        intercept=intercept,
        final_loss=final_loss,
        epochs=epochs,
        loss_history=loss_hist,
        loss_curve_path=loss_path,
        fitting_path=fitting_path,
        predicted_length=predicted,
    )
    return _train_result


def predict(mass_kg: float) -> float:
    if not is_trained():
        raise RuntimeError("Model not trained yet. Call /train first.")
    return float(_model.predict(np.array([[mass_kg]]), verbose=0)[0][0])
