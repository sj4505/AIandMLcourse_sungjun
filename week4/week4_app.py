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
