"""Week 4 물리 ML 실험 앱 — PySide6"""
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QSplitter, QFormLayout, QPushButton, QProgressBar,
    QTextEdit, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QScrollArea,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

# ── 중력 가속도 ──────────────────────────────────────────────────────
_G = 9.81


# ══════════════════════════════════════════════════════════════════════
# 순수 함수 (테스트 가능)
# ══════════════════════════════════════════════════════════════════════

def calculate_true_period(L: float, theta0_deg: float) -> float:
    """진자 실제 주기 (타원 적분 근사)"""
    theta0_rad = np.deg2rad(theta0_deg)
    T_small = 2 * np.pi * np.sqrt(L / _G)
    correction = (1 + (1 / 16) * theta0_rad ** 2 + (11 / 3072) * theta0_rad ** 4)
    return T_small * correction


def overfitting_true_function(x: np.ndarray) -> np.ndarray:
    """과적합 데모의 실제 함수: y = sin(2x) + 0.5x"""
    return np.sin(2 * x) + 0.5 * x


def generate_projectile_data(n_samples: int = 2000, noise_level: float = 0.5):
    """포물선 운동 데이터 생성. Returns (X, Y)"""
    v0 = np.random.uniform(10, 50, n_samples)
    theta = np.random.uniform(20, 70, n_samples)
    theta_rad = np.deg2rad(theta)
    t_max = 2 * v0 * np.sin(theta_rad) / _G
    t = np.random.uniform(0, t_max * 0.9, n_samples)
    x = v0 * np.cos(theta_rad) * t + np.random.normal(0, noise_level, n_samples)
    y = v0 * np.sin(theta_rad) * t - 0.5 * _G * t ** 2 + np.random.normal(0, noise_level, n_samples)
    valid = y >= 0
    X = np.column_stack([v0[valid], theta[valid], t[valid]])
    Y = np.column_stack([x[valid], y[valid]])
    return X, Y


def lab1_get_function(name: str):
    """Lab1 함수 이름 → numpy 함수 반환"""
    fns = {
        "sin(x)":          lambda x: np.sin(x),
        "cos(x)+0.5sin(2x)": lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
        "x·sin(x)":        lambda x: x * np.sin(x),
    }
    if name not in fns:
        raise ValueError(f"Unknown function: {name}")
    return fns[name]


# ══════════════════════════════════════════════════════════════════════
# Worker Thread
# ══════════════════════════════════════════════════════════════════════

class TrainingWorker(QThread):
    progress = Signal(int)
    log_message = Signal(str)
    finished = Signal(object)   # list[Figure]
    error = Signal(str)

    def __init__(self, task_fn, params: dict):
        super().__init__()
        self._task_fn = task_fn
        self._params = params

    def run(self):
        try:
            figures = self._task_fn(self._params, self.progress.emit, self.log_message.emit)
            self.finished.emit(figures)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════
# Base Lab Tab
# ══════════════════════════════════════════════════════════════════════

class BaseLabTab(QWidget):
    run_started = Signal()
    run_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._canvas_list: list[FigureCanvas] = []
        self._setup_ui()

    def _build_param_panel(self, form: QFormLayout):
        raise NotImplementedError

    def _collect_params(self) -> dict:
        raise NotImplementedError

    def _task_fn(self, params: dict, progress_cb, log_cb) -> list:
        raise NotImplementedError

    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)

        # 왼쪽 파라미터 패널
        left = QWidget()
        left.setFixedWidth(240)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        form.setSpacing(8)
        self._build_param_panel(form)
        left_layout.addLayout(form)
        left_layout.addSpacing(12)
        self._run_btn = QPushButton("▶  실행")
        self._run_btn.setFont(QFont("", 11, QFont.Bold))
        self._run_btn.setMinimumHeight(36)
        self._run_btn.clicked.connect(self._on_run)
        left_layout.addWidget(self._run_btn)
        left_layout.addStretch()

        # 오른쪽 결과 패널
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMaximumHeight(18)
        right_layout.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(90)
        self._log.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self._log)

        # 스크롤 가능한 캔버스 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        canvas_container = QWidget()
        self._canvas_area = QVBoxLayout(canvas_container)
        self._canvas_area.setAlignment(Qt.AlignTop)
        scroll.setWidget(canvas_container)
        right_layout.addWidget(scroll)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(splitter)

    def _on_run(self):
        if self._worker and self._worker.isRunning():
            return
        try:
            params = self._collect_params()
        except ValueError as e:
            QMessageBox.warning(self, "파라미터 오류", str(e))
            return
        self._log.clear()
        self._progress.setValue(0)
        self._clear_canvases()
        self.run_started.emit()
        self._run_btn.setEnabled(False)
        self._worker = TrainingWorker(self._task_fn, params)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.log_message.connect(self._log.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, figures):
        for fig in figures:
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(350)
            self._canvas_area.addWidget(canvas)
            self._canvas_list.append(canvas)
        self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self.run_finished.emit()

    def _on_error(self, msg):
        QMessageBox.critical(self, "오류 발생", msg)
        self._run_btn.setEnabled(True)
        self.run_finished.emit()

    def _clear_canvases(self):
        for canvas in self._canvas_list:
            self._canvas_area.removeWidget(canvas)
            canvas.deleteLater()
        self._canvas_list.clear()

# ══════════════════════════════════════════════════════════════════════
# Lab 1: 1D 함수 근사
# ══════════════════════════════════════════════════════════════════════

class Lab1Tab(BaseLabTab):
    """Lab 1: Universal Approximation — 1D 함수 근사"""

    def _build_param_panel(self, form: QFormLayout):
        self._func = QComboBox()
        self._func.addItems(["sin(x)", "cos(x)+0.5sin(2x)", "x·sin(x)"])
        form.addRow("함수:", self._func)

        self._arch = QComboBox()
        self._arch.addItems(["Small [32]", "Medium [64,64]", "Large [128,128,64]"])
        self._arch.setCurrentIndex(2)
        form.addRow("네트워크:", self._arch)

        self._epochs = QSpinBox()
        self._epochs.setRange(100, 10000)
        self._epochs.setSingleStep(500)
        self._epochs.setValue(3000)
        form.addRow("Epochs:", self._epochs)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(0.0001, 0.1)
        self._lr.setSingleStep(0.001)
        self._lr.setDecimals(4)
        self._lr.setValue(0.01)
        form.addRow("Learning Rate:", self._lr)

    def _collect_params(self) -> dict:
        arch_map = {
            "Small [32]": [32],
            "Medium [64,64]": [64, 64],
            "Large [128,128,64]": [128, 128, 64],
        }
        return {
            "func_name": self._func.currentText(),
            "hidden_layers": arch_map[self._arch.currentText()],
            "epochs": self._epochs.value(),
            "lr": self._lr.value(),
        }

    def _task_fn(self, params, progress_cb, log_cb):
        import tensorflow as tf
        from tensorflow import keras

        func_name = params["func_name"]
        hidden_layers = params["hidden_layers"]
        epochs = params["epochs"]
        lr = params["lr"]

        fn = lab1_get_function(func_name)
        x_train = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
        x_test = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
        y_train = fn(x_train)
        y_test = fn(x_test)

        # 모델 구성
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(1,)))
        for units in hidden_layers:
            model.add(keras.layers.Dense(units, activation="tanh"))
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )

        log_cb(f"학습 시작: {func_name} / 구조: {hidden_layers}")

        # 직접 epoch 루프로 진행률 보고
        batch_size = 32
        for epoch in range(epochs):
            model.train_on_batch(x_train, y_train)
            if epoch % max(1, epochs // 100) == 0:
                progress_cb(int(epoch / epochs * 90))

        y_pred = model.predict(x_test, verbose=0)
        mse = float(np.mean((y_pred - y_test) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_test)))
        log_cb(f"완료 — MSE: {mse:.6f}, MAE: {mae:.6f}")
        progress_cb(95)

        # Figure 1: 함수 근사 + 오차
        fig1 = Figure(figsize=(10, 4))
        ax1 = fig1.add_subplot(121)
        ax1.plot(x_test, y_test, "b-", linewidth=2, label="True", alpha=0.7)
        ax1.plot(x_test, y_pred, "r--", linewidth=2, label="NN Prediction")
        ax1.scatter(x_train[::10], y_train[::10], c="black", s=15, alpha=0.3)
        ax1.set_title(f"{func_name}\nMSE: {mse:.6f}", fontweight="bold")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig1.add_subplot(122)
        error = np.abs(y_pred - y_test)
        ax2.plot(x_test, error, "r-", linewidth=1.5)
        ax2.fill_between(x_test.flatten(), 0, error.flatten(), color="r", alpha=0.3)
        ax2.set_title(f"Absolute Error (Max: {error.max():.4f})", fontweight="bold")
        ax2.set_xlabel("x")
        ax2.set_ylabel("|error|")
        ax2.grid(True, alpha=0.3)

        fig1.suptitle("1D 함수 근사", fontsize=13, fontweight="bold")
        fig1.tight_layout()

        progress_cb(100)
        return [fig1]

# ══════════════════════════════════════════════════════════════════════
# Lab 2: 포물선 운동 회귀
# ══════════════════════════════════════════════════════════════════════

class Lab2Tab(BaseLabTab):
    """Lab 2: Projectile Motion Regression"""

    def _build_param_panel(self, form: QFormLayout):
        self._n_samples = QSpinBox()
        self._n_samples.setRange(500, 5000)
        self._n_samples.setSingleStep(500)
        self._n_samples.setValue(2000)
        form.addRow("학습 샘플:", self._n_samples)

        self._epochs = QSpinBox()
        self._epochs.setRange(20, 500)
        self._epochs.setSingleStep(20)
        self._epochs.setValue(100)
        form.addRow("Epochs:", self._epochs)

        self._v0 = QDoubleSpinBox()
        self._v0.setRange(10.0, 50.0)
        self._v0.setSingleStep(5.0)
        self._v0.setValue(30.0)
        form.addRow("초기속도 v₀ (m/s):", self._v0)

        self._theta = QDoubleSpinBox()
        self._theta.setRange(20.0, 70.0)
        self._theta.setSingleStep(5.0)
        self._theta.setValue(45.0)
        form.addRow("발사각 θ (°):", self._theta)

    def _collect_params(self) -> dict:
        return {
            "n_samples": self._n_samples.value(),
            "epochs": self._epochs.value(),
            "v0": self._v0.value(),
            "theta": self._theta.value(),
        }

    def _task_fn(self, params, progress_cb, log_cb):
        from tensorflow import keras

        n_samples = params["n_samples"]
        epochs = params["epochs"]
        v0_test = params["v0"]
        theta_test = params["theta"]

        # 데이터 생성
        X_train, Y_train = generate_projectile_data(n_samples, noise_level=0.5)
        log_cb(f"학습 데이터: {X_train.shape[0]} 샘플")
        progress_cb(5)

        # 모델
        model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(2, activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])

        history = model.fit(
            X_train, Y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[keras.callbacks.LambdaCallback(
                on_epoch_end=lambda ep, _: progress_cb(5 + int(ep / epochs * 85))
            )],
        )
        progress_cb(90)

        final_loss = history.history["loss"][-1]
        final_val = history.history["val_loss"][-1]
        log_cb(f"Train MSE: {final_loss:.4f} / Val MSE: {final_val:.4f}")

        # 테스트 궤적 예측
        theta_rad = np.deg2rad(theta_test)
        t_max = 2 * v0_test * np.sin(theta_rad) / _G
        t = np.linspace(0, t_max, 60)
        X_input = np.column_stack([
            np.full(60, v0_test), np.full(60, theta_test), t
        ])
        pred = model.predict(X_input, verbose=0)
        x_pred, y_pred = pred[:, 0], pred[:, 1]
        x_true = v0_test * np.cos(theta_rad) * t
        y_true = v0_test * np.sin(theta_rad) * t - 0.5 * _G * t ** 2

        mse_traj = float(np.mean((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2))
        log_cb(f"궤적 MSE: {mse_traj:.4f}")
        progress_cb(95)

        # Figure 1: 궤적
        fig1 = Figure(figsize=(10, 4))
        ax1 = fig1.add_subplot(121)
        ax1.plot(x_true, y_true, "b-", linewidth=2.5, label="True", alpha=0.7)
        ax1.plot(x_pred, y_pred, "r--", linewidth=2, label="NN Pred")
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_title(f"v₀={v0_test} m/s, θ={theta_test}°\nMSE: {mse_traj:.4f}", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig1.add_subplot(122)
        ax2.plot(history.history["loss"], "b-", linewidth=2, label="Train")
        ax2.plot(history.history["val_loss"], "r--", linewidth=2, label="Val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE")
        ax2.set_title("Training History", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        fig1.suptitle("Projectile Motion Regression", fontsize=13, fontweight="bold")
        fig1.tight_layout()
        progress_cb(100)
        return [fig1]


# ══════════════════════════════════════════════════════════════════════
# Lab 3: 과적합 vs 과소적합
# ══════════════════════════════════════════════════════════════════════

class Lab3Tab(BaseLabTab):
    """Lab 3: Overfitting vs Underfitting"""

    def _build_param_panel(self, form: QFormLayout):
        self._n_train = QSpinBox()
        self._n_train.setRange(30, 500)
        self._n_train.setSingleStep(10)
        self._n_train.setValue(100)
        form.addRow("학습 샘플:", self._n_train)

        self._noise = QDoubleSpinBox()
        self._noise.setRange(0.05, 1.0)
        self._noise.setSingleStep(0.05)
        self._noise.setDecimals(2)
        self._noise.setValue(0.3)
        form.addRow("노이즈:", self._noise)

        self._epochs = QSpinBox()
        self._epochs.setRange(50, 1000)
        self._epochs.setSingleStep(50)
        self._epochs.setValue(200)
        form.addRow("Epochs:", self._epochs)

    def _collect_params(self) -> dict:
        return {
            "n_train": self._n_train.value(),
            "noise": self._noise.value(),
            "epochs": self._epochs.value(),
        }

    def _task_fn(self, params, progress_cb, log_cb):
        from tensorflow import keras

        n_train = params["n_train"]
        noise = params["noise"]
        epochs = params["epochs"]

        # 데이터
        x_tr = np.random.uniform(-2, 2, n_train).reshape(-1, 1)
        y_tr = overfitting_true_function(x_tr) + np.random.normal(0, noise, (n_train, 1))
        x_val = np.random.uniform(-2, 2, 50).reshape(-1, 1)
        y_val = overfitting_true_function(x_val) + np.random.normal(0, noise, (50, 1))
        x_test = np.linspace(-2, 2, 200).reshape(-1, 1)
        y_test = overfitting_true_function(x_test)

        model_defs = {
            "Underfit [4]": keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(4, activation="relu"),
                keras.layers.Dense(1),
            ]),
            "Good Fit [32,16]": keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ]),
            "Overfit [256,128,64,32]": keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1),
            ]),
        }

        histories = {}
        predictions = {}
        colors = {"Underfit [4]": "blue", "Good Fit [32,16]": "green", "Overfit [256,128,64,32]": "red"}
        total = len(model_defs) * epochs
        done = 0

        for name, model in model_defs.items():
            model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")

            def make_cb(n=name):
                nonlocal done
                def on_epoch_end(ep, logs):
                    nonlocal done
                    done += 1
                    progress_cb(int(done / total * 90))
                return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

            h = model.fit(
                x_tr, y_tr,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=16,
                verbose=0,
                callbacks=[make_cb()],
            )
            histories[name] = h
            predictions[name] = model.predict(x_test, verbose=0)
            t_mse = float(np.mean((predictions[name] - y_test) ** 2))
            log_cb(f"{name}: Train={h.history['loss'][-1]:.4f} Val={h.history['val_loss'][-1]:.4f} TestMSE={t_mse:.4f}")

        progress_cb(92)

        # Figure 1: 예측 비교
        fig1 = Figure(figsize=(12, 4))
        for i, (name, y_pred) in enumerate(predictions.items()):
            ax = fig1.add_subplot(1, 3, i + 1)
            ax.scatter(x_tr, y_tr, alpha=0.4, s=20, color="gray", label="Train data")
            ax.plot(x_test, y_test, "k-", linewidth=2, label="True", alpha=0.7)
            ax.plot(x_test, y_pred, color=colors[name], linestyle="--", linewidth=2, label=name)
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig1.suptitle("과적합 vs 과소적합 비교", fontsize=13, fontweight="bold")
        fig1.tight_layout()

        # Figure 2: 학습 곡선
        fig2 = Figure(figsize=(12, 4))
        for i, name in enumerate(model_defs.keys()):
            ax = fig2.add_subplot(1, 3, i + 1)
            h = histories[name]
            ax.plot(h.history["loss"], color=colors[name], linestyle="-", linewidth=2, label="Train")
            ax.plot(h.history["val_loss"], color=colors[name], linestyle="--", linewidth=2, label="Val")
            ax.set_title(name, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")
        fig2.suptitle("학습 곡선 (Train vs Val Loss)", fontsize=13, fontweight="bold")
        fig2.tight_layout()

        progress_cb(100)
        return [fig1, fig2]
