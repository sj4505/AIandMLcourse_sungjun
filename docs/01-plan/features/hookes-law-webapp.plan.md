# Plan: hookes-law-webapp

## Executive Summary

| Item | Detail |
|------|--------|
| Feature | Hooke's Law Interactive Web App |
| Start Date | 2026-03-18 |
| Target Date | 2026-03-18 |
| Stack | FastAPI + TailwindCSS + TensorFlow |

### 1.3 Value Delivered (4-Perspective)

| Perspective | Content |
|-------------|---------|
| Problem | 훅의 법칙(F=kx)을 시각적으로 이해하기 어렵고, TensorFlow 학습 과정을 직관적으로 볼 수 없음 |
| Solution | FastAPI 백엔드 + TailwindCSS UI로 TF 모델 훈련→예측→시각화를 원스탑 웹앱으로 제공 |
| Function UX Effect | 새 질량 입력 시 즉시 예측, Loss curve + Spring fitting PNG를 웹에서 바로 확인 가능 |
| Core Value | 전문가 수준의 인터랙티브 ML 교육 데모: Hooke's Law를 TF linear regression으로 증명 |

---

## 1. Feature Overview

**Feature Name**: hookes-law-webapp
**Description**: FastAPI + TailwindCSS + TensorFlow 기반의 훅의 법칙 학습 웹 애플리케이션.
TensorFlow로 훅의 법칙(F=kx, 즉 Length = 2·Weight + 10)을 선형회귀로 학습하고,
epoch별 Loss curve, spring fitting 결과를 PNG로 저장. 새 질량 입력 시 길이 예측.

**Project Level**: Dynamic (FastAPI backend + TF + Frontend UI)

---

## 2. Goals & Success Criteria

| # | Goal | Acceptance Criteria |
|---|------|---------------------|
| G1 | FastAPI 서버 실행 | `uvicorn main:app` 정상 기동, `/docs` swagger 접근 |
| G2 | TF 모델 학습 | week2 데이터로 학습, Loss < 5.0 달성 |
| G3 | Loss curve PNG 저장 | `output/loss_curve.png` 생성 |
| G4 | Spring fitting PNG 저장 | `output/spring_fitting.png` 생성 |
| G5 | 예측 API | POST `/predict` → 질량 입력 → 길이 반환 |
| G6 | 학습 API | POST `/train` → 모델 학습 + PNG 저장 |
| G7 | 전문가급 UI | TailwindCSS, 다크테마, 애니메이션, 반응형 |
| G8 | 교육 콘텐츠 | 훅의 법칙 수식 설명, 학습 과정 인터랙티브 표시 |

---

## 3. Scope

### In Scope
- FastAPI 백엔드 (train, predict, status, images API)
- TensorFlow Sequential 선형회귀 모델
- week2 데이터 (weights 0~10kg, Length = 2·w + 10 + noise)
- Epoch별 Loss curve PNG → `output/` 저장
- Spring fitting PNG → `output/` 저장
- TailwindCSS CDN 기반 SPA 프론트엔드
- 실시간 학습 진행 상황 표시
- 새 질량 → 예측 길이 반환 UI

### Out of Scope
- 사용자 인증
- 데이터베이스
- Docker 배포

---

## 4. Technical Requirements

| Component | Technology | Version |
|-----------|-----------|---------|
| Backend | FastAPI | latest |
| Backend Server | Uvicorn | latest |
| ML Framework | TensorFlow | 2.x |
| Data Processing | NumPy | latest |
| Visualization | Matplotlib | latest |
| Frontend | HTML + TailwindCSS CDN | 3.x |
| Frontend Charts | Chart.js CDN | latest |
| Python | Python | 3.9+ |

---

## 5. Folder Structure

```
week2/LinRegSpr/
├── main.py              # FastAPI app
├── model.py             # TF model + training logic
├── data.py              # Hooke's Law dataset
├── requirements.txt
├── output/              # PNG 저장 디렉토리
│   ├── loss_curve.png
│   └── spring_fitting.png
└── static/
    └── index.html       # TailwindCSS SPA
```

---

## 6. API Design

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | 메인 UI (index.html) |
| POST | `/train` | 모델 학습 + PNG 생성 |
| POST | `/predict` | 질량 → 길이 예측 |
| GET | `/status` | 모델 상태 (학습 여부, params) |
| GET | `/images/{name}` | PNG 이미지 서빙 |

---

## 7. Implementation Order

1. `data.py` — Hooke's Law 데이터셋 (week2 동일 로직)
2. `model.py` — TF 모델 정의 + 학습 + PNG 저장
3. `main.py` — FastAPI 라우터
4. `static/index.html` — TailwindCSS UI
5. `requirements.txt`

---

## 8. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| TF 학습 시간 UI 블로킹 | High | FastAPI BackgroundTasks 또는 동기 처리 후 결과 반환 |
| PNG 경로 문제 | Medium | 절대경로 기반 output dir |
| CORS 이슈 | Low | FastAPI CORS middleware 추가 |
