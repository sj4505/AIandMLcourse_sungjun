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
