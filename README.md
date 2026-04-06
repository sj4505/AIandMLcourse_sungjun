# Sungjun An — Portfolio

개인 프로젝트 및 AI·ML 실습 결과물 모음입니다.

---

## Projects

### 1. Budget App — 스마트 가계부 웹앱
**`budget-app/index.html`**

SMS 문자를 붙여넣으면 OpenAI가 자동으로 지출을 분석해주는 가계부 앱입니다.

**주요 기능:**
- SMS 문자 파싱 + OpenAI 기반 지출 자동 분류
- 더치페이 계산기
- 할부 / 대출 관리 및 진행률 추적
- 월별 예산 설정 및 일일 가용 금액 자동 계산
- 브라우저 localStorage 기반 (서버 불필요, 바로 실행)

**실행 방법:** `budget-app/index.html`을 브라우저에서 열기

---

### 2. Focus Flow — 뽀모도로 태스크 매니저
**`focus-flow/index.html`**

작업을 30분 단위로 쪼개고 우선순위를 정해준 뒤, 뽀모도로 타이머로 집중할 수 있는 웹앱입니다.

**주요 기능:**
- 태스크 자동 분할 (30분 단위 시간 블록)
- 우선순위 자동 정렬
- 뽀모도로 타이머 연동
- 브라우저 localStorage 기반 (서버 불필요, 바로 실행)

**실행 방법:** `focus-flow/index.html`을 브라우저에서 열기

---

### 3. Physics Web — 부산대 물리학과 웹사이트
**`physics-web/`**

FastAPI 기반 물리학과 소개 웹사이트입니다.

**기술 스택:** Python, FastAPI, Jinja2 Templates

**실행 방법:**
```bash
cd physics-web
pip install -r requirements.txt
python main.py
# http://localhost:8000 에서 확인
```

---

### 4. Week 3 — 신경망 실험 PySide6 GUI 앱
**`week3/week3_app.py`**

Perceptron, 활성화 함수, MLP, Universal Approximation 등 신경망 핵심 개념을 인터랙티브하게 실험할 수 있는 데스크탑 앱입니다.

**주요 기능:**
- Lab 1: Perceptron — AND/OR/XOR 게이트 학습
- Lab 2: 활성화 함수 비교 (Sigmoid, Tanh, ReLU, Leaky ReLU)
- Lab 3: 순전파(Forward Propagation) 시각화
- Lab 4: MLP (numpy 직접 구현)
- Lab 5: Universal Approximation Theorem 실험

**기술 스택:** Python, PySide6, TensorFlow, matplotlib

**실행 방법:**
```bash
pip install PySide6 tensorflow numpy matplotlib
python week3/week3_app.py
```

---

### 5. Week 4 — 물리 ML 실험 PySide6 GUI 앱
**`week4/week4_app.py`**

물리 데이터를 Neural Network로 학습·예측하는 4가지 실험을 인터랙티브하게 실행할 수 있는 데스크탑 앱입니다. TDD로 개발되었습니다.

**주요 기능:**
- Lab 1: 1D 함수 근사 (Universal Approximation Theorem)
- Lab 2: 포물선 운동 궤적 회귀
- Lab 3: 과적합 vs 과소적합 시각화
- Lab 4: 진자 주기 예측 (비선형 물리 법칙 학습)

**기술 스택:** Python, PySide6, TensorFlow/Keras, matplotlib, pytest

**실행 방법:**
```bash
pip install PySide6 tensorflow numpy matplotlib
python week4/week4_app.py
```

---

## Tech Stack

| 분야 | 기술 |
|------|------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| Desktop GUI | Python, PySide6 |
| ML / AI | TensorFlow/Keras, NumPy, OpenAI API |
| Testing | pytest, pytest-qt |
| Tools | uv, Git |

---

## Contact

**Sungjun An**  
ansungjun1610@gmail.com
