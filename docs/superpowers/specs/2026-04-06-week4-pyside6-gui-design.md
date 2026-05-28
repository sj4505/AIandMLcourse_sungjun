# Week4 PySide6 GUI Design

## Overview

Week4의 4개 물리 ML Lab (1D 함수 근사, 포물선 운동, 과적합, 진자)을 PySide6 기반 인터랙티브 GUI 앱으로 통합한다. 기존 스크립트의 ML 로직은 그대로 유지하고, matplotlib 출력을 GUI 캔버스에 임베드한다.

## Architecture

### File Structure
```
week4/
├── gui_app.py              # 진입점 (python week4/gui_app.py)
├── gui/
│   ├── __init__.py
│   ├── main_window.py      # QMainWindow + QTabWidget
│   ├── worker.py           # QThread 기반 ML 학습 워커
│   └── tabs/
│       ├── __init__.py
│       ├── tab_1d.py       # Lab 1: 1D 함수 근사
│       ├── tab_projectile.py  # Lab 2: 포물선 운동
│       ├── tab_overfitting.py # Lab 3: 과적합 데모
│       └── tab_pendulum.py    # Lab 4: 진자 주기
└── tests/
    ├── __init__.py
    ├── test_physics.py     # 순수 함수 단위 테스트 (pytest)
    └── test_gui.py         # GUI 초기화 테스트 (pytest-qt)
```

### Components

**MainWindow** (`main_window.py`)
- `QMainWindow` with `QTabWidget`
- 4개 탭: Lab1, Lab2, Lab3, Lab4
- 상단 타이틀 + 상태바

**BaseTab** (공통 구조)
- 좌측 패널: 파라미터 컨트롤 (`QSpinBox`, `QDoubleSpinBox`, `QComboBox`)
- 우측: `FigureCanvasQTAgg` (matplotlib 캔버스)
- 하단: Run 버튼 + 진행 상태 표시 (`QProgressBar` + `QLabel`)

**MLWorker** (`worker.py`)
- `QThread` 서브클래스
- 시그널: `progress(int)`, `status(str)`, `finished(dict)`, `error(str)`
- TensorFlow 학습을 별도 스레드에서 실행하여 UI 블로킹 방지

### Per-Tab Parameters

| Tab | 파라미터 |
|-----|---------|
| Lab1 | 함수 선택(sin/cos+sin/x·sin), 네트워크 크기, epochs |
| Lab2 | 초기속도(v₀), 발사각(θ), n_samples, epochs |
| Lab3 | noise_level, epochs, n_train |
| Lab4 | 길이(L), 초기각도(θ₀), n_samples, epochs |

## Data Flow

```
User → [파라미터 입력] → Run 버튼 클릭
     → MLWorker (QThread) → TF 학습 → progress 시그널
     → UI 업데이트 (QProgressBar)
     → finished 시그널 → matplotlib 렌더링
     → FigureCanvasQTAgg 표시
```

## Testing Strategy (TDD)

**test_physics.py** (순수 함수, 의존성 없음):
- `calculate_true_period(L, theta)` - 진자 주기 공식 검증
- `generate_projectile_data()` - 데이터 shape/range 검증
- `true_function(x)` - sin(2x)+0.5x 검증

**test_gui.py** (pytest-qt):
- MainWindow 초기화, 탭 개수 확인
- 각 탭 위젯 존재 확인
- Run 버튼 클릭 가능 여부

## Key Decisions

1. **단일 앱 진입점**: `gui_app.py` 하나로 4개 Lab 모두 접근
2. **QThread 분리**: TF 학습은 반드시 워커 스레드에서 실행
3. **기존 코드 재사용**: 각 `.py`의 함수들을 import해서 사용
4. **matplotlib 임베드**: `FigureCanvasQTAgg` + `NavigationToolbar2QT`
