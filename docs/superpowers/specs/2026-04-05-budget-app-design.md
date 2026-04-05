# 가계부 앱 설계 스펙
**날짜**: 2026-04-05  
**버전**: 1.0 (MVP)

---

## 1. 개요

대학생을 위한 미니멀 개인 가계부 웹앱. 알바 수입과 용돈을 기반으로 가처분 소득·일일 예산을 실시간 계산하고, SMS 클립보드 파싱으로 지출을 빠르게 기록한다.

**디자인 철학**: Dieter Rams — 필요한 것만, 군더더기 없이.  
**무드**: 다크 베이스, 숫자가 주인공, 모노스페이스 혼합.

---

## 2. 기술 스택

| 항목 | 선택 |
|---|---|
| 구조 | 단일 HTML 파일 |
| 스타일 | Tailwind CSS (CDN) |
| 데이터 저장 | localStorage |
| AI | OpenAI API (GPT-4o mini) — 브라우저 직접 호출 |
| 폰트 | JetBrains Mono (숫자), Noto Sans KR (텍스트) |

---

## 3. 디자인 시스템

### 컬러
| 상태 | 컬러 | 조건 |
|---|---|---|
| 여유 | `#34d399` (Emerald) | 잔여 예산 50% 이상 |
| 주의 | `#fbbf24` (Amber) | 잔여 예산 20~50% |
| 위험 | `#f87171` (Red) | 잔여 예산 20% 미만 |
| 초과 | `#ef4444` (Red 강조) | 마이너스 |

- 다크 베이스: `#0f172a`
- 보조 배경: `#0a1120`
- 보더: `#1e293b`
- 텍스트 주: `#f1f5f9`, 보조: `#94a3b8`, 흐린: `#475569`

---

## 4. 화면 구조

하단 탭 4개.

### TAB 1 — 홈 대시보드
- 오늘의 잔여 예산 (대형 숫자 + 상태 컬러 + 진행 바)
- 이번달 수입/지출 분해 내역:
  - 알바 예상 수입 + 용돈
  - 고정 지출 합계
  - 할부 월납 합계
  - 대출 일일 적립 × 이번달 남은 일수
  - **가처분 소득** (굵게 강조)
- 최근 지출 3~5건

### TAB 2 — 지출 입력
**SMS 파싱 흐름 (메인)**:
1. textarea에 카드 결제 문자 붙여넣기
2. "분석" 버튼 → GPT-4o mini 호출
3. 결과 표시: 금액, 가맹점, 카테고리, 날짜
4. AI 피드백 한 줄 (예: "오늘 일일 예산의 86% 소진")
5. 더치페이 토글 → 인원 수 입력 → 실부담 자동 계산
6. 저장 버튼

**수동 입력 (보조)**: 하단 "직접 입력 →" 링크로 폼 전환

### TAB 3 — 할부 & 대출
**할부 섹션**:
- 항목별 카드: 이름 / 진행률 바 / 월납금 / 잔여금
- 마지막 1개월 남은 항목: 빨간 경고 배지 표시
- "+ 할부 추가" 버튼

**대출 섹션**:
- 항목별 카드: 이름 / 진행률 바 / 일 적립액 / 잔여 잔액 / D-day
- "+ 대출 추가" 버튼

### TAB 4 — 설정
섹션별 구성:
- **수입**: 용돈(월), 알바 패턴(요일/시간), 시급, 이번주 보정
- **고정 지출**: 목록 + 추가/삭제
- **카테고리**: 기본 13개 표시 + 사용자 추가/삭제
- **API**: OpenAI API 키 (localStorage 저장, password 입력)

---

## 5. 데이터 모델 (localStorage)

```
budgetApp_settings     { allowance, baseDate, apiKey }
budgetApp_work         { pattern: [{day, hours}], hourlyWage }
budgetApp_workLog      [{ weekKey, actualHours }]
budgetApp_fixed        [{ id, name, amount, dueDay, category }]
budgetApp_expenses     [{ id, date, amount, myAmount, merchant,
                          category, memo, source, isDutch, splitCount }]
budgetApp_installments [{ id, name, totalAmount, months, paidMonths,
                          startDate, monthlyAmount }]
budgetApp_loans        [{ id, name, remainingBalance, targetDate }]
budgetApp_categories   [{ id, name, isDefault }]
```

---

## 6. 일일 예산 계산 공식

```
월 알바 수입    = 주당 시간(패턴 + 보정) × 시급 × (이번달 해당 요일 수)
월 예상 수입    = 월 알바 수입 + 용돈
월 고정 지출    = Σ 고정지출 + Σ 할부 월납금
월 대출 적립    = Σ (대출 잔액 ÷ targetDate까지 남은 일수) × 이번달 남은 일수
가처분 소득     = 월 예상 수입 − 월 고정 지출 − 월 대출 적립
일일 예산       = 가처분 소득 ÷ 이번달 남은 일수  (오늘 포함)
오늘 잔여 예산  = 일일 예산 − 오늘 지출 합계 (myAmount 기준)
```

---

## 7. AI 통합 (GPT-4o mini)

### SMS 파싱 프롬프트 구조
```
system: 한국 카드사 SMS에서 금액, 가맹점명, 날짜를 추출하고
        아래 카테고리 중 하나로 분류하세요: {categories}
        JSON으로 반환: { amount, merchant, category, date, feedback }
        feedback은 오늘 예산 소진율을 포함한 한 줄 피드백.

user: {sms_text}
     오늘 일일 예산: {daily_budget}, 오늘 지출 합계: {today_spent}
```

### 카테고리 주입
`budgetApp_categories`에서 목록을 읽어 프롬프트에 동적 삽입. 사용자 추가 카테고리도 자동 포함.

---

## 8. 카테고리 기본값 (13개)

| 기본 제공 (삭제 불가) | 사용자 추가 가능 |
|---|---|
| 식비, 카페, 편의점, 교통, 쇼핑, 구독, 의료, 문화, 자기관리, 단백질음료, 운동, 사줌, 기타 | ✓ 무제한 추가 |

---

## 9. 더치페이 처리

- 지출 입력 시 "더치페이" 토글 제공
- 인원 수 입력 → `myAmount = amount ÷ splitCount` 자동 계산
- 예산 차감은 `myAmount` 기준
- 내역 표시: "₩48,000 (더치 ÷3 → ₩16,000)"
- 저장 필드: `{ amount, myAmount, splitCount, isDutch }`

---

## 10. 2차 버전 (MVP 제외)

- 주간 점수 시스템
- AI 챗봇
- Gmail 자동 영수증 연동
