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
