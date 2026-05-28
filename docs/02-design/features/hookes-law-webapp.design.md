# Design: hookes-law-webapp

## 1. Architecture Overview

```
Browser (TailwindCSS SPA)
        │  HTTP (fetch)
        ▼
FastAPI (main.py)  ──→  model.py (TF training/predict)
        │                      │
        │                      ▼
        │               output/
        │               ├── loss_curve.png
        │               └── spring_fitting.png
        ▼
GET /images/{name} → serves PNG files
```

---

## 2. Data Layer (data.py)

### 2.1 Dataset Spec
```python
weights = np.array([0,1,2,3,4,5,6,7,8,9,10], dtype=float)
true_lengths = 2 * weights + 10        # k=2 cm/kg, L0=10 cm
noise = Normal(mean=0, std=1.5)
measured_lengths = true_lengths + noise (seed=42)
```

### 2.2 Functions
| Function | Return | Description |
|----------|--------|-------------|
| `get_dataset()` | `(weights, measured_lengths, true_lengths)` | week2 동일 데이터 반환 |
| `get_spring_constant()` | `float` | k = 2 (cm/kg) |

---

## 3. Model Layer (model.py)

### 3.1 TF Model Architecture
```
Input(shape=[1])
  └─ Dense(units=1, activation='linear')  # y = w*x + b
Output: spring length (cm)
```

### 3.2 Training Config
| Param | Value |
|-------|-------|
| Optimizer | Adam (lr=0.01) |
| Loss | MSE |
| Epochs | 500 |
| Verbose | 0 |

### 3.3 Output PNGs

#### loss_curve.png
- X: Epoch (0~500)
- Y: MSE Loss (log scale)
- Style: dark background (#1a1a2e), neon color (#00d4ff)
- Figsize: (10, 5)
- Grid: True, alpha=0.3

#### spring_fitting.png
- Scatter: measured data (cyan dots)
- Line 1: True Law (green dashed, y=2x+10)
- Line 2: AI Prediction (red solid)
- Prediction point: new mass (gold star marker)
- Dark background style
- Figsize: (10, 6)

### 3.4 Functions
| Function | Params | Return | Description |
|----------|--------|--------|-------------|
| `HookesLawModel()` | - | TF model | 모델 생성 |
| `train_model(epochs)` | epochs=500 | `TrainResult` | 학습 + PNG 저장 |
| `predict(mass_kg)` | float | float | 길이 예측 (cm) |
| `get_model_params()` | - | `{slope, intercept}` | 학습된 파라미터 |
| `is_trained()` | - | bool | 학습 여부 |

### 3.5 TrainResult Schema
```python
{
  "slope": float,        # learned weight (≈2.0)
  "intercept": float,    # learned bias (≈10.0)
  "final_loss": float,
  "epochs": int,
  "loss_history": List[float],
  "loss_curve_path": str,
  "fitting_path": str
}
```

---

## 4. API Layer (main.py)

### 4.1 POST /train
```
Request:  { "epochs": 500, "new_mass_kg": 15.0 }
Response: {
  "success": true,
  "slope": 1.98,
  "intercept": 10.12,
  "final_loss": 2.34,
  "epochs": 500,
  "loss_history": [...],
  "predicted_length": 40.02,
  "images": {
    "loss_curve": "/images/loss_curve.png",
    "spring_fitting": "/images/spring_fitting.png"
  }
}
```

### 4.2 POST /predict
```
Request:  { "mass_kg": 15.0 }
Response: {
  "mass_kg": 15.0,
  "predicted_length_cm": 40.02,
  "theoretical_length_cm": 40.0,
  "model_equation": "Length = 1.98 × Mass + 10.12"
}
```

### 4.3 GET /status
```
Response: {
  "is_trained": true,
  "slope": 1.98,
  "intercept": 10.12,
  "final_loss": 2.34,
  "model_equation": "Length = 1.98 × Mass + 10.12"
}
```

### 4.4 GET /images/{name}
- name: `loss_curve.png` or `spring_fitting.png`
- Returns: FileResponse (image/png)

### 4.5 CORS
- Allow Origins: `["*"]`
- Allow Methods: `["*"]`
- Allow Headers: `["*"]`

---

## 5. Frontend Design (index.html)

### 5.1 Layout
```
┌──────────────────────────────────────────────────┐
│  🌊 HOOKE'S LAW  ·  AI Linear Regression         │
│  (Hero: 훅의 법칙 수식 + 설명 카드)              │
├──────────────┬───────────────────────────────────┤
│  Control     │  Visualization                    │
│  Panel       │  ┌──────────────┬───────────────┐ │
│              │  │  Loss Curve  │ Spring Fitting │ │
│  [Epochs]    │  │    PNG       │    PNG         │ │
│  [Mass kg]   │  └──────────────┴───────────────┘ │
│  [TRAIN]     │                                   │
│  [PREDICT]   │  Results Card                     │
│              │  "Length = X × Mass + Y"           │
│  Status      │  Prediction: Z cm                 │
└──────────────┴───────────────────────────────────┘
```

### 5.2 Color System (Dark Theme)
| Token | Value | Usage |
|-------|-------|-------|
| bg-primary | #0f0e17 | Page background |
| bg-card | #1a1933 | Card backgrounds |
| accent-cyan | #00d4ff | Primary accent |
| accent-green | #00ff88 | Success / True Law |
| accent-gold | #ffd700 | Prediction highlight |
| accent-red | #ff4757 | AI prediction line |
| text-primary | #fffffe | Main text |
| text-secondary | #a7a9be | Secondary text |

### 5.3 Components
1. **HeroSection**: 훅의 법칙 수식 (F=kx, LaTeX-style), 설명 카드
2. **ControlPanel**: epochs slider, mass input, Train/Predict buttons
3. **StatusBadge**: 학습 전/후 상태 표시 (애니메이션)
4. **TrainingProgress**: 학습 중 spinner + progress indicator
5. **ImageViewer**: loss_curve + spring_fitting 이미지 2분할 표시
6. **ResultCard**: 예측 결과, 학습된 수식, 실제 수식 비교
7. **MetricsBar**: Final Loss, Slope, Intercept 수치 표시

### 5.4 UX Interactions
- Train 버튼 클릭 → loading spinner → 완료 시 이미지 fade-in
- Predict 버튼 → 즉시 결과 표시 (학습 완료 후)
- Epochs slider → 실시간 값 표시
- 이미지 클릭 → modal 확대 보기

---

## 6. File Structure

```
week2/LinRegSpr/
├── main.py              # FastAPI app entry
├── model.py             # TF HookesLawModel
├── data.py              # Dataset provider
├── requirements.txt     # Dependencies
├── output/              # Auto-created, PNG outputs
└── static/
    └── index.html       # TailwindCSS SPA (CDN-based)
```

---

## 7. Implementation Checklist

- [ ] data.py: get_dataset(), get_spring_constant()
- [ ] model.py: HookesLawModel class, train_model(), predict(), dark-themed PNGs
- [ ] main.py: FastAPI app, CORS, /train, /predict, /status, /images/{name}, static mount
- [ ] static/index.html: Full TailwindCSS dark-theme SPA
- [ ] output/ directory auto-creation
- [ ] requirements.txt
