# Week 4 PySide6 앱 디자인 스펙

**날짜**: 2026-04-01
**버전**: 1.0

---

## 개요

`week4/week4_app.py` — PySide6 기반 단일 파일 앱.
4개의 Neural Network 실험 Lab을 탭으로 구성하고, 학습을 백그라운드 스레드에서 실행하며 진행 상태와 결과 그래프를 앱 내부에 표시한다.

기존 4개 스크립트(`01perfect1d.py` ~ `04pendulum.py`)는 수정 없이 유지한다.

---

## 파일 구조

```
week4/
├── week4_app.py        ← 신규 추가
├── 01perfect1d.py      ← 유지
├── 02projectile.py     ← 유지
├── 03overfitting.py    ← 유지
├── 04pendulum.py       ← 유지
└── outputs/            ← 기존 출력 폴더
```

---

## 아키텍처

### 클래스 구조

```
MainWindow (QMainWindow)
└── QTabWidget
    ├── Lab1Tab (QWidget)
    ├── Lab2Tab (QWidget)
    ├── Lab3Tab (QWidget)
    └── Lab4Tab (QWidget)

TrainingWorker (QThread)
    signals:
        progress(int)       — 0~100, progress bar 업데이트
        log_message(str)    — 에포크 로그 출력
        finished(list)      — matplotlib Figure 리스트 반환
        error(str)          — 예외 발생 시 에러 메시지
```

### 각 LabTab 레이아웃

```
QWidget (LabTab)
├── QSplitter (horizontal)
│   ├── 왼쪽: 파라미터 패널 (QFormLayout)
│   │   ├── 파라미터 입력 위젯들 (QSpinBox / QLineEdit / QComboBox)
│   │   └── 실행 버튼 (QPushButton)
│   └── 오른쪽: 결과 패널 (QVBoxLayout)
│       ├── QProgressBar
│       ├── QTextEdit (로그, read-only)
│       └── FigureCanvas (matplotlib, 학습 완료 후 표시)
└── QStatusBar 영역 (메인 윈도우 공유)
```

---

## 데이터 흐름

1. 사용자가 파라미터 입력 후 **실행** 버튼 클릭
2. `TrainingWorker(QThread)` 인스턴스 생성, 파라미터 전달
3. `worker.start()` → 백그라운드에서 TensorFlow/Keras 학습 실행
4. 학습 중 Keras callback이 `progress` / `log_message` signal emit
5. UI 스레드에서 signal 수신 → progress bar / 로그 업데이트 (비블로킹)
6. 학습 완료 → `finished(figures)` signal emit → `FigureCanvas` 갱신
7. 예외 발생 → `error(str)` signal → 에러 메시지 다이얼로그 표시
8. 학습 중 다른 탭 자유롭게 탐색 가능

---

## 각 Lab 파라미터

### Lab 1 — 1D 함수 근사
| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Epochs | QSpinBox (100~10000) | 3000 |
| Hidden Layers | QLineEdit (쉼표 구분) | `128, 128, 64` |
| Activation | QComboBox | `tanh` |
| Learning Rate | QLineEdit | `0.001` |

생성 그래프: `perfect_1d_approximation`, `network_size_comparison`, `extreme_function_test`

### Lab 2 — 포물선 운동
| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Epochs | QSpinBox | 2000 |
| 초기 속력 v₀ (m/s) | QDoubleSpinBox | `20, 30, 40` (쉼표 구분) |
| 발사 각도 θ (°) | QLineEdit | `30, 45, 60` |

생성 그래프: `02_projectile_trajectories`, `02_projectile_analysis`

### Lab 3 — 과적합 vs 과소적합
| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Epochs | QSpinBox | 200 |
| 노이즈 크기 | QDoubleSpinBox (0.0~1.0) | `0.3` |

생성 그래프: `03_overfitting_comparison`, `03_training_curves`, `03_error_analysis`, `03_comparison_table`

### Lab 4 — 진자 주기 예측
| 파라미터 | 위젯 | 기본값 |
|---|---|---|
| Epochs | QSpinBox | 2000 |
| 길이 L (m) | QLineEdit | `0.5, 1.0, 2.0` |
| 초기 각도 θ₀ (°) | QLineEdit | `5, 30, 80` |

생성 그래프: `04_pendulum_prediction`, `04_pendulum_simulation`, `04_pendulum_analysis`

---

## 기술 스택

| 항목 | 선택 |
|---|---|
| GUI 프레임워크 | PySide6 |
| 그래프 임베드 | `matplotlib.backends.backend_qtagg.FigureCanvasQTAgg` |
| 백그라운드 학습 | `QThread` + Keras `Callback` |
| ML 프레임워크 | TensorFlow / Keras (기존 동일) |
| Python | 3.10+ |

---

## 에러 처리

- TensorFlow import 실패 시 앱 시작 시 경고 다이얼로그
- 파라미터 파싱 실패(잘못된 입력) 시 실행 전 QMessageBox로 안내
- 학습 중 예외 → `error` signal → QMessageBox
- 학습 중 모든 탭의 실행 버튼 비활성화 (어느 Lab이든 하나만 실행 가능)

---

## 제약 사항

- Stop 기능 없음 — 학습 시작 후 완료까지 대기
- 동시에 여러 Lab 학습 불가 (탭별 순차 실행)
- 기존 4개 스크립트 코드 수정 없음
