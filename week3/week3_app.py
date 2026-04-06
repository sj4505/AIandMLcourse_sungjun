"""Week 3 Neural Network 실험 앱 — PySide6"""
import sys
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QSplitter, QFormLayout, QPushButton, QProgressBar,
    QTextEdit, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


import importlib.util as _ilu
import matplotlib.pyplot as _plt_suppress

def _load_module(name: str, filename: str):
    """Load a week3 script suppressing plt.show/savefig side effects."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    _orig_show = _plt_suppress.show
    _orig_savefig = _plt_suppress.savefig
    _plt_suppress.show = lambda *a, **k: None
    _plt_suppress.savefig = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        _plt_suppress.show = _orig_show
        _plt_suppress.savefig = _orig_savefig
    return mod

_p01 = _load_module("p01", "01_perceptron.py")
Perceptron = _p01.Perceptron

_p04 = _load_module("p04", "04_mlp_numpy.py")
MLP = _p04.MLP

_p05 = _load_module("p05", "05_universal_approximation.py")
UniversalApproximator = _p05.UniversalApproximator


class TrainingWorker(QThread):
    """백그라운드 연산 스레드"""
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
            self.error.emit(str(e))


class BaseLabTab(QWidget):
    """모든 LabTab의 공통 베이스"""

    run_started = Signal()
    run_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._canvas_list: list[FigureCanvas] = []
        self._setup_ui()

    # ── 서브클래스가 구현할 메서드 ──────────────────────────────
    def _build_param_panel(self, form: QFormLayout):
        """파라미터 위젯을 form에 추가. 서브클래스 구현."""
        raise NotImplementedError

    def _collect_params(self) -> dict:
        """현재 위젯 값을 dict로 반환. 서브클래스 구현."""
        raise NotImplementedError

    def _task_fn(self, params: dict, progress_cb, log_cb) -> list:
        """연산 수행 후 Figure 리스트 반환. 서브클래스 구현."""
        raise NotImplementedError

    # ── 공통 UI 구성 ────────────────────────────────────────────
    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)

        # 왼쪽: 파라미터 패널
        left = QWidget()
        left.setMaximumWidth(260)
        left_layout = QVBoxLayout(left)
        form = QFormLayout()
        self._build_param_panel(form)
        left_layout.addLayout(form)
        self._run_btn = QPushButton("실행")
        self._run_btn.setFont(QFont("", 12, QFont.Bold))
        self._run_btn.clicked.connect(self._on_run)
        left_layout.addWidget(self._run_btn)
        left_layout.addStretch()

        # 오른쪽: 결과 패널
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        right_layout.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        right_layout.addWidget(self._log)
        self._canvas_area = QVBoxLayout()
        right_layout.addLayout(self._canvas_area)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        main = QVBoxLayout(self)
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
        self._worker.log_message.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, figures):
        for fig in figures:
            canvas = FigureCanvas(fig)
            self._canvas_area.addWidget(canvas)
            self._canvas_list.append(canvas)
        self._progress.setValue(100)
        self._run_btn.setEnabled(True)
        self.run_finished.emit()

    def _on_error(self, msg):
        QMessageBox.critical(self, "오류", msg)
        self._run_btn.setEnabled(True)
        self.run_finished.emit()

    def _append_log(self, msg):
        self._log.append(msg)

    def _clear_canvases(self):
        for canvas in self._canvas_list:
            self._canvas_area.removeWidget(canvas)
            canvas.deleteLater()
        self._canvas_list.clear()

    def set_run_enabled(self, enabled: bool):
        self._run_btn.setEnabled(enabled)


# ── Placeholder Lab Tabs (Tasks 3-7 will replace these) ─────────────
class Lab1Tab(BaseLabTab):
    """Lab 1: Perceptron — AND/OR/XOR 게이트"""

    def _build_param_panel(self, form: QFormLayout):
        self._gate = QComboBox()
        self._gate.addItems(["AND", "OR", "XOR"])
        form.addRow("Gate:", self._gate)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(0.01, 1.0)
        self._lr.setSingleStep(0.01)
        self._lr.setValue(0.1)
        form.addRow("Learning Rate:", self._lr)

        self._epochs = QSpinBox()
        self._epochs.setRange(10, 10000)
        self._epochs.setValue(100)
        form.addRow("Epochs:", self._epochs)

    def _collect_params(self) -> dict:
        return {
            "gate": self._gate.currentText(),
            "lr": self._lr.value(),
            "epochs": self._epochs.value(),
        }

    def _task_fn(self, params, progress_cb, log_cb):
        import numpy as np
        from matplotlib.figure import Figure

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        gate_map = {
            "AND": np.array([0, 0, 0, 1]),
            "OR":  np.array([0, 1, 1, 1]),
            "XOR": np.array([0, 1, 1, 0]),
        }
        y = gate_map[params["gate"]]

        p = Perceptron(input_size=2, learning_rate=params["lr"])
        epochs = params["epochs"]
        for i in range(epochs):
            for inputs, label in zip(X, y):
                pred = p.predict(inputs)
                err = label - pred
                p.weights += params["lr"] * err * inputs
                p.bias += params["lr"] * err
            progress_cb(int((i + 1) / epochs * 100))

        correct = sum(p.predict(x) == l for x, l in zip(X, y))
        log_cb(f"Gate: {params['gate']}  정확도: {correct}/4")

        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        x_min, x_max, y_min, y_max = -0.5, 1.5, -0.5, 1.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100),
        )
        Z = np.array([p.predict(np.array([xi, yi]))
                      for xi, yi in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5],
                    colors=["blue", "red"])
        for point, label in zip(X, y):
            c = "red" if label == 1 else "blue"
            m = "o" if label == 1 else "x"
            ax.scatter(point[0], point[1], c=c, marker=m, s=200,
                       edgecolors="black", linewidth=2)
        ax.set_title(f"{params['gate']} Gate — Decision Boundary")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return [fig]

class Lab2Tab(BaseLabTab):
    """Lab 2: Activation Functions 비교"""

    def _build_param_panel(self, form: QFormLayout):
        self._x_min = QDoubleSpinBox()
        self._x_min.setRange(-20.0, 0.0)
        self._x_min.setValue(-5.0)
        form.addRow("X 최솟값:", self._x_min)

        self._x_max = QDoubleSpinBox()
        self._x_max.setRange(0.0, 20.0)
        self._x_max.setValue(5.0)
        form.addRow("X 최댓값:", self._x_max)

        self._cb_sigmoid = QCheckBox("Sigmoid")
        self._cb_sigmoid.setChecked(True)
        self._cb_tanh = QCheckBox("Tanh")
        self._cb_tanh.setChecked(True)
        self._cb_relu = QCheckBox("ReLU")
        self._cb_relu.setChecked(True)
        self._cb_leaky = QCheckBox("Leaky ReLU")
        self._cb_leaky.setChecked(True)
        for cb in [self._cb_sigmoid, self._cb_tanh, self._cb_relu, self._cb_leaky]:
            form.addRow("", cb)

    def _collect_params(self) -> dict:
        selected = []
        if self._cb_sigmoid.isChecked():
            selected.append("Sigmoid")
        if self._cb_tanh.isChecked():
            selected.append("Tanh")
        if self._cb_relu.isChecked():
            selected.append("ReLU")
        if self._cb_leaky.isChecked():
            selected.append("Leaky ReLU")
        if not selected:
            raise ValueError("최소 하나의 함수를 선택하세요.")
        return {
            "x_min": self._x_min.value(),
            "x_max": self._x_max.value(),
            "selected": selected,
        }

    def _task_fn(self, params, progress_cb, log_cb):
        import numpy as np
        from matplotlib.figure import Figure

        x = np.linspace(params["x_min"], params["x_max"], 300)
        sel = params["selected"]

        def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        def tanh(x): return np.tanh(x)
        def relu(x): return np.maximum(0, x)
        def leaky(x): return np.where(x > 0, x, 0.01 * x)

        fn_map = {
            "Sigmoid":    (sigmoid,  lambda x: sigmoid(x) * (1 - sigmoid(x))),
            "Tanh":       (tanh,     lambda x: 1 - np.tanh(x) ** 2),
            "ReLU":       (relu,     lambda x: np.where(x > 0, 1, 0).astype(float)),
            "Leaky ReLU": (leaky,    lambda x: np.where(x > 0, 1, 0.01)),
        }

        fig = Figure(figsize=(10, 4))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        progress_cb(30)

        for name in sel:
            fn, dfn = fn_map[name]
            ax0.plot(x, fn(x), label=name, linewidth=2)
            ax1.plot(x, dfn(x), label=f"{name}'", linewidth=2)

        for ax, title, ylabel in zip(
            [ax0, ax1],
            ["Activation Functions", "Derivatives (Gradients)"],
            ["f(x)", "f'(x)"],
        ):
            ax.axhline(0, color="k", alpha=0.3)
            ax.axvline(0, color="k", alpha=0.3)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("x")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        progress_cb(100)
        log_cb(f"표시 함수: {', '.join(sel)}")
        fig.tight_layout()
        return [fig]


class Lab3Tab(BaseLabTab):
    """Lab 3: Forward Propagation 단계별 시각화"""

    def _build_param_panel(self, form: QFormLayout):
        self._hidden = QSpinBox()
        self._hidden.setRange(2, 20)
        self._hidden.setValue(3)
        form.addRow("Hidden Units:", self._hidden)

        self._x1 = QDoubleSpinBox()
        self._x1.setRange(-2.0, 2.0)
        self._x1.setSingleStep(0.1)
        self._x1.setValue(0.5)
        form.addRow("Input X1:", self._x1)

        self._x2 = QDoubleSpinBox()
        self._x2.setRange(-2.0, 2.0)
        self._x2.setSingleStep(0.1)
        self._x2.setValue(0.8)
        form.addRow("Input X2:", self._x2)

    def _collect_params(self) -> dict:
        return {
            "hidden": self._hidden.value(),
            "x1": self._x1.value(),
            "x2": self._x2.value(),
        }

    def _task_fn(self, params, progress_cb, log_cb):
        import numpy as np
        import matplotlib.patches as mpatches
        from matplotlib.figure import Figure

        n_h = params["hidden"]
        X = np.array([params["x1"], params["x2"]])

        np.random.seed(42)
        W1 = np.random.randn(2, n_h) * 0.5
        b1 = np.random.randn(n_h) * 0.1
        W2 = np.random.randn(n_h, 1) * 0.5
        b2 = np.random.randn(1) * 0.1

        def sigmoid(x): return 1 / (1 + np.exp(-x))
        def relu(x): return np.maximum(0, x)

        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        progress_cb(50)
        log_cb(f"z1 = {np.round(z1, 3)}")
        log_cb(f"a1 (ReLU) = {np.round(a1, 3)}")
        log_cb(f"z2 = {z2[0]:.4f},  a2 (Sigmoid) = {a2[0]:.4f}")

        fig = Figure(figsize=(10, 4))
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        # Network diagram
        ax0.set_xlim(0, 4)
        ax0.set_ylim(0, 4)
        ax0.axis("off")
        ax0.set_title(f"Network Architecture (2-{n_h}-1)", fontweight="bold")
        iy = np.linspace(0.5, 3.5, 2)
        hy = np.linspace(0.5, 3.5, n_h)
        for y in iy:
            ax0.add_patch(mpatches.Circle((0.5, y), 0.2, color="lightblue", ec="black", lw=2))
        for y in hy:
            ax0.add_patch(mpatches.Circle((2, y), 0.2, color="lightgreen", ec="black", lw=2))
        ax0.add_patch(mpatches.Circle((3.5, 2), 0.2, color="lightcoral", ec="black", lw=2))
        for iy_ in iy:
            for hy_ in hy:
                ax0.plot([0.7, 1.8], [iy_, hy_], "k-", alpha=0.3, lw=1)
        for hy_ in hy:
            ax0.plot([2.2, 3.3], [hy_, 2], "k-", alpha=0.3, lw=1)
        ax0.text(0.5, -0.1, "Input", ha="center", fontsize=9, fontweight="bold")
        ax0.text(2, -0.1, f"Hidden\n(ReLU)", ha="center", fontsize=9, fontweight="bold")
        ax0.text(3.5, -0.1, "Output\n(Sigmoid)", ha="center", fontsize=9, fontweight="bold")

        # z/a bar chart
        show = min(n_h, 6)
        idx = np.arange(show)
        w = 0.3
        ax1.bar(idx - w / 2, z1[:show], w, label="z1 (before ReLU)", color="orange", alpha=0.8)
        ax1.bar(idx + w / 2, a1[:show], w, label="a1 (after ReLU)", color="green", alpha=0.8)
        ax1.set_title("Hidden Layer: z1 vs a1 (ReLU)", fontweight="bold")
        ax1.set_xlabel("Neuron")
        ax1.set_ylabel("Value")
        ax1.set_xticks(idx)
        ax1.set_xticklabels([f"H{i+1}" for i in range(show)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        progress_cb(100)
        fig.tight_layout()
        return [fig]


class Lab4Tab(BaseLabTab):
    """Lab 4: MLP로 XOR 해결"""

    def _build_param_panel(self, form: QFormLayout):
        self._hidden = QSpinBox()
        self._hidden.setRange(2, 32)
        self._hidden.setValue(4)
        form.addRow("Hidden Size:", self._hidden)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(0.01, 2.0)
        self._lr.setSingleStep(0.01)
        self._lr.setValue(0.5)
        form.addRow("Learning Rate:", self._lr)

        self._epochs = QSpinBox()
        self._epochs.setRange(1000, 50000)
        self._epochs.setSingleStep(1000)
        self._epochs.setValue(10000)
        form.addRow("Epochs:", self._epochs)

    def _collect_params(self) -> dict:
        return {
            "hidden": self._hidden.value(),
            "lr": self._lr.value(),
            "epochs": self._epochs.value(),
        }

    def _task_fn(self, params, progress_cb, log_cb):
        import numpy as np
        from matplotlib.figure import Figure

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)

        mlp = MLP(
            input_size=2,
            hidden_size=params["hidden"],
            output_size=1,
            learning_rate=params["lr"],
        )

        epochs = params["epochs"]
        report_every = max(1, epochs // 10)

        for epoch in range(epochs):
            output = mlp.forward(X)
            loss = float(np.mean((output - y) ** 2))
            mlp.backward(X, y, output)
            mlp.loss_history.append(loss)
            if (epoch + 1) % report_every == 0:
                progress_cb(int((epoch + 1) / epochs * 100))
                log_cb(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.6f}")

        preds = mlp.predict(X)
        acc = float(np.mean(preds == y.astype(int))) * 100
        log_cb(f"\n정확도: {acc:.1f}%")

        fig = Figure(figsize=(14, 4))
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        # Loss
        ax0.plot(mlp.loss_history, lw=2)
        ax0.set_yscale("log")
        ax0.set_title("Training Loss (MSE)", fontweight="bold")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.grid(True, alpha=0.3)

        # Decision boundary
        xx, yy = np.meshgrid(
            np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200)
        )
        Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cf = ax1.contourf(xx, yy, Z, levels=20, cmap="RdYlBu", alpha=0.8)
        fig.colorbar(cf, ax=ax1)
        for point, label in zip(X, y):
            c = "red" if label[0] == 1 else "blue"
            m = "o" if label[0] == 1 else "x"
            ax1.scatter(point[0], point[1], c=c, marker=m, s=200,
                        edgecolors="black", lw=2, zorder=5)
        ax1.set_title("Decision Boundary", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Hidden activations
        ha = mlp.a1
        im = ax2.imshow(ha.T, cmap="viridis", aspect="auto")
        ax2.set_yticks(range(params["hidden"]))
        ax2.set_yticklabels([f"H{i+1}" for i in range(params["hidden"])])
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
        ax2.set_title("Hidden Layer Activations", fontweight="bold")
        fig.colorbar(im, ax=ax2)
        for i in range(params["hidden"]):
            for j in range(4):
                ax2.text(j, i, f"{ha[j, i]:.2f}",
                         ha="center", va="center", color="white", fontsize=8)

        fig.tight_layout()
        return [fig]


class Lab5Tab(BaseLabTab):
    """Lab 5: Universal Approximation Theorem"""

    def _build_param_panel(self, form: QFormLayout):
        self._fn = QComboBox()
        self._fn.addItems(["Sine", "Step", "Complex"])
        form.addRow("Target Function:", self._fn)

        self._neurons = QComboBox()
        self._neurons.addItems(["3", "10", "50"])
        self._neurons.setCurrentIndex(2)
        form.addRow("Neurons:", self._neurons)

        self._epochs = QSpinBox()
        self._epochs.setRange(1000, 20000)
        self._epochs.setSingleStep(1000)
        self._epochs.setValue(5000)
        form.addRow("Epochs:", self._epochs)

    def _collect_params(self) -> dict:
        return {
            "fn": self._fn.currentText(),
            "neurons": int(self._neurons.currentText()),
            "epochs": self._epochs.value(),
        }

    def _task_fn(self, params, progress_cb, log_cb):
        import numpy as np
        from matplotlib.figure import Figure

        fn_map = {
            "Sine":    lambda x: np.sin(2 * np.pi * x),
            "Step":    lambda x: np.where(x < 0.5, 0, 1).astype(float),
            "Complex": lambda x: (np.sin(2 * np.pi * x)
                                  + 0.5 * np.sin(4 * np.pi * x)
                                  + 0.3 * np.cos(6 * np.pi * x)),
        }
        target_fn = fn_map[params["fn"]]
        x_train = np.linspace(0, 1, 100).reshape(-1, 1)
        x_test = np.linspace(0, 1, 200).reshape(-1, 1)
        y_train = target_fn(x_train)
        y_true = target_fn(x_test)

        n = params["neurons"]
        lr = 0.05 if n < 20 else 0.01
        model = UniversalApproximator(n_hidden=n, activation="tanh")
        epochs = params["epochs"]
        report = max(1, epochs // 10)

        for epoch in range(epochs):
            z1 = x_train @ model.W1 + model.b1
            a1 = np.tanh(z1)
            out = a1 @ model.W2 + model.b2
            loss = float(np.mean((out - y_train) ** 2))
            dL = 2 * (out - y_train) / len(x_train)
            dW2 = a1.T @ dL
            db2 = np.sum(dL, axis=0)
            da1 = dL @ model.W2.T
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = x_train.T @ dz1
            db1 = np.sum(dz1, axis=0)
            model.W2 -= lr * dW2
            model.b2 -= lr * db2
            model.W1 -= lr * dW1
            model.b1 -= lr * db1
            if (epoch + 1) % report == 0:
                progress_cb(int((epoch + 1) / epochs * 100))
                log_cb(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.6f}")

        z1 = x_test @ model.W1 + model.b1
        y_pred = np.tanh(z1) @ model.W2 + model.b2
        mse = float(np.mean((y_pred - y_true) ** 2))
        log_cb(f"\nMSE: {mse:.6f}")

        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        ax.plot(x_test, y_true, "b-", lw=2, label="True Function", alpha=0.7)
        ax.plot(x_test, y_pred, "r--", lw=2, label=f"NN ({n} neurons)")
        ax.scatter(x_train[::10], y_train[::10], c="green", s=30, alpha=0.5,
                   label="Train data")
        ax.set_title(f"{params['fn']} — {n} neurons  MSE={mse:.4f}", fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return [fig]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 3: Neural Network 실험")
        self.resize(1100, 750)

        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._lab_tabs: list[BaseLabTab] = [
            Lab1Tab(), Lab2Tab(), Lab3Tab(), Lab4Tab(), Lab5Tab()
        ]
        labels = [
            "Lab1: Perceptron",
            "Lab2: Activation Fn",
            "Lab3: Forward Prop",
            "Lab4: MLP/XOR",
            "Lab5: Universal Approx",
        ]
        for tab, label in zip(self._lab_tabs, labels):
            self._tabs.addTab(tab, label)
            tab.run_started.connect(self._disable_all_runs)
            tab.run_finished.connect(self._enable_all_runs)

    def _disable_all_runs(self):
        for tab in self._lab_tabs:
            tab.set_run_enabled(False)

    def _enable_all_runs(self):
        for tab in self._lab_tabs:
            tab.set_run_enabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
