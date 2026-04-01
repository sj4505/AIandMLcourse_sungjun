# Week 3 PySide6 앱 디자인 스펙

**날짜**: 2026-04-01
**버전**: 1.0

---

## 개요

`week3/week3_app.py` — PySide6 기반 단일 파일 앱.
Week 3의 5개 Neural Network 실험 Lab을 탭으로 구성하고, 연산을 백그라운드 스레드에서 실행하며 진행 상태와 결과 그래프를 앱 내부에 표시한다.

기존 5개 스크립트(`01_perceptron.py` ~ `05_universal_approximation.py`)는 수정 없이 유지하며, 핵심 클래스/함수만 import하여 사용한다.

---

## 파일 구조

```
week3/
├── week3_app.py                    ← 신규 추가
├── 01_perceptron.py                ← 유지 (Perceptron 클래스 import)
├── 02_activation_functions.py      ← 유지 (함수 import)
├── 03_forward_propagation.py       ← 유지 (클래스 import)
├── 04_mlp_numpy.py                 ← 유지 (MLP 클래스 import)
├── 05_universal_approximation.py   ← 유지 (함수 import)
└── outputs/                        ← 기존 출력 폴더
```

---

## 아키텍처

### 클래스 구조

```
MainWindow (QMainWindow)
└── QTabWidget
    ├── Lab1Tab (QWidget) — Perceptron
    ├── Lab2Tab (QWidget) — Activation Functions
    ├── Lab3Tab (QWidget) — Forward Propagation
    ├── Lab4Tab (QWidget) — MLP / XOR
    └── Lab5Tab (QWidget) — Universal Approximation

TrainingWorker (QThread)
    signals:
        progress(int)       — 0~100, progress bar 업데이트
        log_message(str)    — 에포크 로그 출력
        finished(object)    — matplotlib Figure 리스트 반환
        error(str)          — 예외 발생 시 에러 메시지
```

### 각 LabTab 레이아웃

```
QWidget (LabTab)
└── QSplitter (horizontal)
    ├── 왼쪽: 파라미터 패널 (QFormLayout)
    │   ├── 파라미터 입력 위젯들
    │   └── 실행 버튼 (QPushButton "실행")
    └── 오른쪽: 결과 패널 (QVBoxLayout)
        ├── QProgressBar
        ├── QTextEdit (로그, read-only)
        └── FigureCanvas (matplotlib, 실행 완료 후 표시)
```

---

## 각 Lab 파라미터

### Lab 1 — Perceptron (AND/OR/XOR 게이트)

| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Gate | QComboBox (AND / OR / XOR) | AND |
| Learning Rate | QDoubleSpinBox (0.01~1.0, step 0.01) | 0.1 |
| Epochs | QSpinBox (10~10000) | 100 |

생성 그래프: 선택한 게이트의 결정 경계 시각화 (contourf + scatter)

### Lab 2 — Activation Functions

| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| X 최솟값 | QDoubleSpinBox (-20~0) | -5.0 |
| X 최댓값 | QDoubleSpinBox (0~20) | 5.0 |
| 표시 함수 | QCheckBox ×4 (Sigmoid / Tanh / ReLU / Leaky ReLU) | 전체 선택 |

생성 그래프: 함수 비교, Gradient 비교 (2×2 subplot)

### Lab 3 — Forward Propagation

| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Hidden Units | QSpinBox (2~20) | 3 |
| Input X1 | QDoubleSpinBox (-2.0~2.0) | 0.5 |
| Input X2 | QDoubleSpinBox (-2.0~2.0) | 0.8 |

생성 그래프: 네트워크 다이어그램, Layer별 z/a 값 시각화 (2×2 subplot)

### Lab 4 — MLP / XOR

| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Hidden Size | QSpinBox (2~32) | 4 |
| Learning Rate | QDoubleSpinBox (0.01~2.0, step 0.01) | 0.5 |
| Epochs | QSpinBox (1000~50000, step 1000) | 10000 |

생성 그래프: Training Loss, 결정 경계, 은닉층 활성화 (1×3 subplot)

### Lab 5 — Universal Approximation

| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Target Function | QComboBox (Sine / Step / Complex) | Sine |
| Neurons | QComboBox (3 / 10 / 50) | 50 |
| Epochs | QSpinBox (1000~20000, step 1000) | 5000 |

생성 그래프: 목표 함수 vs 근사 함수 비교, MSE 비교 (뉴런 수별)

---

## 데이터 흐름

1. 사용자가 파라미터 입력 후 **실행** 버튼 클릭
2. `TrainingWorker(QThread)` 인스턴스 생성, 파라미터 전달
3. `worker.start()` → 백그라운드에서 Numpy 연산 실행
4. 연산 중 `progress(int)` signal emit → progress bar 업데이트
5. 연산 중 `log_message(str)` signal emit → QTextEdit 로그 출력
6. 연산 완료 → `finished(figures)` signal emit → FigureCanvas 갱신
7. 예외 발생 → `error(str)` signal → QMessageBox 표시
8. 연산 중 다른 탭 자유롭게 탐색 가능

---

## 기술 스택

| 항목 | 선택 |
|---|---|
| GUI 프레임워크 | PySide6 |
| 그래프 임베드 | `matplotlib.backends.backend_qtagg.FigureCanvasQTAgg` |
| 백그라운드 연산 | `QThread` |
| ML/수치 연산 | NumPy (기존 week3 코드 동일) |
| Python | 3.10+ |
| PDF 생성 | `reportlab` 또는 `weasyprint` (PRD/TRD 문서용) |

---

## 에러 처리

- 잘못된 파라미터 입력 시 실행 전 QMessageBox로 안내
- 연산 중 예외 → `error` signal → QMessageBox
- 연산 중 모든 탭의 실행 버튼 비활성화 (한 번에 하나만 실행)
- import 실패 시 앱 시작 시 경고 다이얼로그

---

## 제약 사항

- Stop 기능 없음 — 실행 시작 후 완료까지 대기
- 동시에 여러 Lab 연산 불가 (순차 실행)
- 기존 5개 스크립트 코드 수정 없음 (클래스/함수만 import)

---

## PRD / TRD 산출물

두 문서 모두 마크다운으로 작성 후 PDF로 변환하여 제출.

### PRD (Product Requirements Document)
- 제품 개요 및 목표
- 사용자 시나리오 (학습자가 파라미터를 바꿔가며 신경망 동작 이해)
- 5개 Lab 기능 명세
- UI/UX 요구사항

### TRD (Technical Requirements Document)
- 기술 스택 및 의존성
- 클래스/컴포넌트 명세
- 데이터 흐름
- 파일 구조
- 빌드 및 실행 방법
