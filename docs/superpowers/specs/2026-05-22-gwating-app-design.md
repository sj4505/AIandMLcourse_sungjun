# Design Spec — 과팅 매칭 MVP (gwating-app)

**Date:** 2026-05-22  
**Status:** Approved  
**Target Users:** 부산대학교 학생 전용 (MVP 범위)

---

## 1. Overview

부산대 학생들이 소규모 그룹 과팅 상대 팀을 찾는 매칭 데모 앱.  
개인 성향 테스트 → 팀 생성 → 팀-팀 궁합 점수 → 추천 리스트 순서로 진행.

**핵심 원칙:**
- 그룹 중심 (개인 프로필이 아닌 팀 카드가 핵심 단위)
- 분위기 매칭 우선 (외모 아닌 대화 스타일·역할 밸런스 기반)
- 가볍고 저부담 (부산대 학생들의 캐주얼한 과팅 문화에 맞게)
- 설명 가능한 점수 (매칭 이유 2~3개 항상 표시)

**부산대 MVP 제약:**
- 학교 필드는 "부산대학교"로 고정 (선택 불가)
- Mock 팀 데이터는 전부 부산대 학생 팀으로 구성
- 지역은 "부산" 고정 (추후 확장 예정 필드로 코드상 남겨둠)
- Copy에서 부산대를 명시: "부산대 과팅 매칭", "부산대생 전용 베타"

---

## 2. Visual Design

**기반:** `design_upgraded_group_blind_date.md` (Airbnb-inspired light theme)

### Colors

```ts
colors: {
  primary:        "#ff5a6f",  // Coral — CTA, 선택 상태, 점수 강조
  primaryActive:  "#e6475d",  // Pressed 상태
  primarySoft:    "#fff0f2",  // 선택된 칩 배경, 강조 영역
  primaryDisabled:"#ffd6dd",  // 비활성 버튼
  canvas:         "#ffffff",
  canvasWarm:     "#fffaf7",  // 히어로, 온보딩 배경
  surfaceSoft:    "#f7f7f7",  // 비활성 칩, 미묘한 구분
  ink:            "#222222",  // 주요 텍스트
  body:           "#3f3f3f",
  muted:          "#6a6a6a",
  hairline:       "#dddddd",
  mint:           "#dff8ec",  // 활발한 친목형
  mintInk:        "#147a55",
  lavender:       "#f0eaff",  // 예의/안전 중시형
  lavenderInk:    "#5b3ab8",
  sky:            "#eaf5ff",  // 자연스러운 소개팅형, 이유 패널
  skyInk:         "#1f6fb2",
  amber:          "#fff3d8",  // 게임/술자리형
  amberInk:       "#9a6700",
}
```

### Typography

Font: **Pretendard** (Korean) + Inter fallback

| 용도 | Size | Weight |
|------|---:|---:|
| Hero headline | 32px | 700 |
| Page title | 28px | 700 |
| Card title / 팀명 | 21px | 650 |
| Section label | 16px | 600 |
| Body copy | 16px | 400 |
| Card metadata | 14px | 400 |
| Mood chip | 12px | 600 |
| Match score | 48px | 750 |

### Shape & Elevation

- Input / 버튼: `border-radius: 8px`
- 카드 / 퀴즈 선택지: `border-radius: 14px`
- 추천 카드 / 메인 패널: `border-radius: 20px`
- Mood chip: `border-radius: 9999px`

Shadow (단일 티어):
```css
box-shadow: rgba(0,0,0,0.02) 0 0 0 1px,
            rgba(0,0,0,0.04) 0 2px 6px,
            rgba(0,0,0,0.10) 0 4px 8px;
```

---

## 3. Pages & Routes

| Route | 페이지 | 목적 |
|-------|--------|------|
| `/` | 홈 | 소개 + 시작 CTA |
| `/test` | 성향 테스트 | 상황형 퀴즈 10문항 (trait당 2개) |
| `/team/create` | 팀 생성 | 팀 정보 입력 + 분위기 선택 |
| `/team/demo` | 팀 상세 | 팀 프로필 요약 + trait 바 |
| `/match` | 매칭 결과 | 추천 팀 랭킹 리스트 |

동적 라우팅 없음. 모두 정적 페이지.

---

## 4. State Management

**localStorage만 사용. Context Provider / Zustand / 백엔드 없음.**

```ts
// localStorage keys
"gwating_user": UserProfile
"gwating_team": TeamProfile
```

```ts
type TraitKey =
  | "atmosphereCoordination"
  | "consideration"
  | "participation"
  | "respectfulness"
  | "communicationBalance";

type MoodKey =
  | "comfortableTalk"   // 편한 대화형   → coral (primary)
  | "activeSocial"      // 활발한 친목형  → mint
  | "gamesAndDrinks"    // 게임/술자리형  → amber
  | "respectfulSafe"    // 예의/안전 중시형 → lavender
  | "naturalIntro";     // 자연스러운 소개팅형 → sky

type UserProfile = {
  nickname: string;
  traits: Record<TraitKey, number>;  // 1~5
};

type TeamProfile = {
  teamName: string;
  school: "부산대학교";              // MVP: 고정값
  region: "부산";                   // MVP: 고정값
  size: number;
  ageRange: string;
  mood: MoodKey;
  members: UserProfile[];
};
```

빈 localStorage 상태: 친근한 복구 메시지 + "처음부터 시작하기" CTA 표시.

---

## 5. Personality Test (성향 테스트)

**형식:** 상황형 객관식 (4지선다), 10문항 (trait당 2문항)  
**저장:** `/test` 완료 시 `gwating_user.traits` → localStorage

### Trait → 문항 매핑 예시

| Trait | Q# | 상황 |
|-------|----|------|
| atmosphereCoordination | Q1 | 대화가 끊겼을 때 |
| atmosphereCoordination | Q2 | 분위기가 가라앉을 때 |
| consideration | Q3 | 말수 적은 사람이 있을 때 |
| consideration | Q4 | 누군가 불편해 보일 때 |
| participation | Q5 | 게임 제안이 나왔을 때 |
| participation | Q6 | 자기소개 순서가 됐을 때 |
| respectfulness | Q7 | 상대가 답하기 싫어보이는 질문을 받을 때 |
| respectfulness | Q8 | 자리가 예상보다 가까울 때 |
| communicationBalance | Q9 | 한 사람이 대화를 독점할 때 |
| communicationBalance | Q10 | 조용한 사람과 짝이 됐을 때 |

각 선택지: 점수 1~5 가중치 매핑 (data/questions.ts에 정의)

---

## 6. Matching Score

```
Final Score = 40% × vibeScore + 35% × traitScore + 25% × conditionScore
```

- **vibeScore (40%):** 5×5 가중치 행렬. 같은 mood면 1.0, 인접하면 0.6~0.8, 반대면 0.2. (`data/moodWeights.ts`)
- **traitScore (35%):** 두 팀의 trait 평균 벡터 간 코사인 유사도 근사 (정규화 후 dot product)
- **conditionScore (25%):** 인원 수 일치 (50%) + 나이대 겹침 (50%)

점수 표현:
- 80%↑ → "Strong vibe fit"
- 60~79% → "Good with some differences"
- 60%↓ → "Different atmosphere preferences"

이유 문장은 항상 2~3개 표시. 판단이 아닌 예측 어조.

---

## 7. Components

| 컴포넌트 | 역할 |
|----------|------|
| `AppHeader` | 로고 "부산대 과팅" + 현재 스텝 표시 |
| `Button` | primary (coral) / secondary (outline) |
| `MoodChip` | 분위기 칩, 5가지 파스텔 색상 |
| `MoodSelector` | 5개 MoodChip 선택 UI |
| `QuizCard` | 상황 문항 + 4지선다 + 진행 바 |
| `TeamCreateForm` | 팀명/인원/나이대 입력 + MoodSelector |
| `TeamProfileCard` | 팀 요약 카드 (trait 강점 문장 포함) |
| `RecommendationTeamCard` | 상대팀 카드 (점수 + 이유 2~3개) |
| `MatchScoreCard` | 대형 점수 숫자 + 헤드라인 |
| `MatchReasonList` | sky 배경 이유 항목 리스트 |

---

## 8. File Structure

```
gwating-app/                 ← 현재 레포 루트에 새 폴더
├── app/
│   ├── layout.tsx           ← Pretendard 폰트, 흰 배경
│   ├── page.tsx             ← /
│   ├── test/page.tsx        ← /test
│   ├── team/
│   │   ├── create/page.tsx  ← /team/create
│   │   └── demo/page.tsx    ← /team/demo
│   └── match/page.tsx       ← /match
├── components/
│   ├── AppHeader.tsx
│   ├── Button.tsx
│   ├── MoodChip.tsx
│   ├── MoodSelector.tsx
│   ├── QuizCard.tsx
│   ├── TeamCreateForm.tsx
│   ├── TeamProfileCard.tsx
│   ├── RecommendationTeamCard.tsx
│   ├── MatchScoreCard.tsx
│   └── MatchReasonList.tsx
├── lib/
│   ├── storage.ts
│   ├── matching.ts
│   └── scoring.ts
├── types/
│   └── matching.ts
├── data/
│   ├── questions.ts         ← 10문항 + 가중치
│   ├── mockTeams.ts         ← 부산대 팀 5개
│   └── moodWeights.ts       ← 5×5 vibe 궁합 행렬
├── tailwind.config.ts
├── next.config.ts
└── package.json
```

---

## 9. Mock Data Scope

`data/mockTeams.ts`에 정의할 상대팀 5개:

| 팀명 | 인원 | 나이대 | 분위기 |
|------|---:|--------|--------|
| 용두산 삼총사 | 3 | 22~23 | comfortableTalk |
| 남포동 클럽 | 4 | 21~24 | activeSocial |
| 해운대 게임단 | 3 | 22~24 | gamesAndDrinks |
| 온천장 신사단 | 3 | 21~22 | respectfulSafe |
| 서면 인트로 | 4 | 20~23 | naturalIntro |

모두 학교: 부산대학교, 지역: 부산.

---

## 10. Known MVP Tradeoffs

- 실제 인증/로그인 없음 (단일 사용자 데모)
- 팀원 여러 명 입력 UI 없음 (팀장 1명 trait만 사용해 팀 대표값으로 활용)
- 사진 없음 (팀 이니셜 아바타로 대체)
- 실제 초대/수락 플로우 없음
- 매칭 점수는 데모용 근사값

---

## 11. Out of Scope (이번 MVP 제외)

- 백엔드, DB, Supabase
- 로그인, 인증
- Zustand, Context Provider
- 실시간 채팅
- 타 학교 확장
- 사진 업로드
