# 과팅 매칭 MVP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 부산대 학생 전용 그룹 과팅 매칭 데모 앱 — 성향 테스트 → 팀 생성 → 궁합 추천 5페이지 Next.js 앱

**Architecture:** Next.js 14 App Router 정적 5페이지, localStorage만으로 상태 공유, mock 팀 5개로 매칭 시연. 순수 함수 매칭 로직(lib/)과 UI(components/)를 완전 분리. Context/Zustand 없음.

**Tech Stack:** Next.js 14, TypeScript, Tailwind CSS (커스텀 토큰), Jest + ts-jest (유닛 테스트), localStorage

---

## File Map

```
gwating-app/                         ← 레포 루트에 생성
├── __tests__/
│   ├── storage.test.ts
│   ├── scoring.test.ts
│   └── matching.test.ts
├── app/
│   ├── globals.css                  ← Pretendard CDN + base styles
│   ├── layout.tsx                   ← 폰트·메타·공통 래퍼
│   ├── page.tsx                     ← 홈 /
│   ├── test/page.tsx                ← /test 성향 퀴즈
│   ├── team/
│   │   ├── create/page.tsx          ← /team/create
│   │   └── demo/page.tsx            ← /team/demo
│   └── match/page.tsx               ← /match
├── components/
│   ├── AppHeader.tsx
│   ├── Button.tsx
│   ├── MoodChip.tsx
│   ├── MoodSelector.tsx
│   ├── QuizCard.tsx
│   ├── MemberRoleCard.tsx          ← 팀원 닉네임+역할 선택 카드
│   ├── TeamProfileCard.tsx
│   ├── RecommendationTeamCard.tsx
│   ├── MatchScoreCard.tsx
│   └── MatchReasonList.tsx
│   (TeamCreateForm 로직은 /team/create/page.tsx 인라인)
├── data/
│   ├── questions.ts
│   ├── mockTeams.ts
│   └── moodWeights.ts
├── lib/
│   ├── storage.ts
│   ├── scoring.ts
│   └── matching.ts
├── types/
│   └── matching.ts
├── jest.config.ts
├── tailwind.config.ts
└── next.config.ts
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `gwating-app/` (전체 Next.js 프로젝트)
- Modify: `gwating-app/tailwind.config.ts`
- Create: `gwating-app/jest.config.ts`

- [ ] **Step 1: Next.js 앱 생성**

현재 레포 루트(`AIandMLcourse/`)에서 실행:

```bash
npx create-next-app@14 gwating-app \
  --typescript \
  --tailwind \
  --app \
  --no-src-dir \
  --import-alias "@/*" \
  --no-eslint
```

프롬프트가 뜨면 모두 기본값(Enter). 완료 후:

```bash
cd gwating-app
```

- [ ] **Step 2: Jest 의존성 설치**

```bash
npm install -D jest @types/jest ts-jest
```

- [ ] **Step 3: jest.config.ts 작성**

`gwating-app/jest.config.ts`:

```ts
import type { Config } from "jest";

const config: Config = {
  preset: "ts-jest",
  testEnvironment: "node",
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
  },
};

export default config;
```

- [ ] **Step 4: package.json에 test 스크립트 추가**

`gwating-app/package.json`의 `"scripts"` 블록에 추가:

```json
"test": "jest",
"test:watch": "jest --watch"
```

- [ ] **Step 5: tailwind.config.ts 교체**

`gwating-app/tailwind.config.ts` 전체를 아래로 교체:

```ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary:           "#ff5a6f",
        "primary-active":  "#e6475d",
        "primary-soft":    "#fff0f2",
        "primary-disabled":"#ffd6dd",
        canvas:            "#ffffff",
        "canvas-warm":     "#fffaf7",
        "surface-soft":    "#f7f7f7",
        ink:               "#222222",
        body:              "#3f3f3f",
        muted:             "#6a6a6a",
        hairline:          "#dddddd",
        "hairline-soft":   "#ebebeb",
        mint:              "#dff8ec",
        "mint-ink":        "#147a55",
        lavender:          "#f0eaff",
        "lavender-ink":    "#5b3ab8",
        sky:               "#eaf5ff",
        "sky-ink":         "#1f6fb2",
        amber:             "#fff3d8",
        "amber-ink":       "#9a6700",
      },
      fontFamily: {
        sans: ["Pretendard Variable", "Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        sm:   "8px",
        md:   "14px",
        lg:   "20px",
        full: "9999px",
      },
      boxShadow: {
        card: "rgba(0,0,0,0.02) 0 0 0 1px, rgba(0,0,0,0.04) 0 2px 6px, rgba(0,0,0,0.10) 0 4px 8px",
      },
      fontSize: {
        hero:    ["32px", { lineHeight: "1.18", fontWeight: "700" }],
        score:   ["48px", { lineHeight: "1.0",  fontWeight: "750" }],
      },
    },
  },
  plugins: [],
};

export default config;
```

- [ ] **Step 6: 빌드 확인**

```bash
npm run build
```

Expected: 오류 없이 완료 (`✓ Compiled successfully`)

- [ ] **Step 7: Commit**

```bash
cd ..
git add gwating-app/
git commit -m "feat: scaffold gwating-app (Next.js 14 + Tailwind + Jest)"
```

---

## Task 2: Core Types

**Files:**
- Create: `gwating-app/types/matching.ts`

- [ ] **Step 1: types/matching.ts 작성**

```ts
export type TraitKey =
  | "atmosphereCoordination"
  | "consideration"
  | "participation"
  | "respectfulness"
  | "communicationBalance";

export type MoodKey =
  | "comfortableTalk"
  | "activeSocial"
  | "gamesAndDrinks"
  | "respectfulSafe"
  | "naturalIntro";

export type MemberRole =
  | "moodMaker"
  | "coordinator"
  | "considerate"
  | "reactor";

export type TeamMember = {
  nickname: string;
  role: MemberRole;
  traits?: Record<TraitKey, number>;
  isLeader?: boolean;
};

export type UserProfile = {
  nickname: string;
  traits: Record<TraitKey, number>;
};

export type TeamProfile = {
  teamName: string;
  school: "부산대학교";
  region: "부산";
  size: number;
  ageRange: string;
  mood: MoodKey;
  members: TeamMember[];
};

export type MatchResult = {
  team: TeamProfile;
  score: number;
  vibeScore: number;
  roleScore: number;
  conditionScore: number;
  reasons: string[];
  label: "Strong vibe fit" | "Good with some differences" | "Different atmosphere preferences";
};
```

- [ ] **Step 2: TypeScript 컴파일 확인**

```bash
cd gwating-app && npx tsc --noEmit
```

Expected: 출력 없음 (오류 없음)

- [ ] **Step 3: Commit**

```bash
cd ..
git add gwating-app/types/matching.ts
git commit -m "feat: add core types (TraitKey, MoodKey, MemberRole, TeamMember, TeamProfile, MatchResult)"
```

---

## Task 3: Data Layer

**Files:**
- Create: `gwating-app/data/questions.ts`
- Create: `gwating-app/data/moodWeights.ts`
- Create: `gwating-app/data/mockTeams.ts`

- [ ] **Step 1: data/questions.ts 작성**

```ts
import { TraitKey } from "@/types/matching";

export type QuizChoice = {
  text: string;
  score: number;
};

export type QuizQuestion = {
  id: number;
  trait: TraitKey;
  situation: string;
  choices: QuizChoice[];
};

export const questions: QuizQuestion[] = [
  {
    id: 1,
    trait: "atmosphereCoordination",
    situation: "과팅 자리에서 대화가 갑자기 끊겼어. 어떻게 해?",
    choices: [
      { text: "바로 새 화제를 꺼낸다", score: 5 },
      { text: "잠깐 기다렸다가 말을 꺼낸다", score: 4 },
      { text: "누군가 먼저 말하겠지", score: 2 },
      { text: "폰 만지며 버틴다", score: 1 },
    ],
  },
  {
    id: 2,
    trait: "atmosphereCoordination",
    situation: "분위기가 예상보다 많이 가라앉아 있어. 어떻게 해?",
    choices: [
      { text: "게임이나 공통 화제를 제안한다", score: 5 },
      { text: "옆 사람한테 살짝 말을 건다", score: 4 },
      { text: "분위기가 풀릴 때까지 기다린다", score: 2 },
      { text: "그냥 빨리 끝나길 바란다", score: 1 },
    ],
  },
  {
    id: 3,
    trait: "consideration",
    situation: "말수가 거의 없는 사람이 한 명 있어. 어떻게 해?",
    choices: [
      { text: "자연스럽게 그 사람에게 질문을 돌린다", score: 5 },
      { text: "눈을 마주쳐서 참여를 유도한다", score: 4 },
      { text: "본인이 말하고 싶으면 하겠지", score: 2 },
      { text: "신경 쓰지 않는다", score: 1 },
    ],
  },
  {
    id: 4,
    trait: "consideration",
    situation: "옆 사람이 불편해 보이는 상황이야. 어떻게 해?",
    choices: [
      { text: "조용히 괜찮냐고 물어본다", score: 5 },
      { text: "화제를 바꿔서 분위기를 돌린다", score: 4 },
      { text: "내 일에 집중한다", score: 2 },
      { text: "못 본 척한다", score: 1 },
    ],
  },
  {
    id: 5,
    trait: "participation",
    situation: "누군가 게임 제안을 했어. 어떻게 해?",
    choices: [
      { text: "바로 '오 좋아!' 하고 참여한다", score: 5 },
      { text: "다수가 원하면 따른다", score: 4 },
      { text: "지켜보다가 분위기 보고 한다", score: 2 },
      { text: "별로지만 억지로 참여한다", score: 1 },
    ],
  },
  {
    id: 6,
    trait: "participation",
    situation: "자기소개 순서가 돌아왔어. 어떻게 해?",
    choices: [
      { text: "재미있는 에피소드 하나 섞어서 자연스럽게 한다", score: 5 },
      { text: "준비한 말을 짧고 깔끔하게 한다", score: 4 },
      { text: "최대한 짧게 끝낸다", score: 2 },
      { text: "긴장해서 말이 잘 안 나온다", score: 1 },
    ],
  },
  {
    id: 7,
    trait: "respectfulness",
    situation: "상대가 대답하기 싫어하는 것 같은 질문을 받았어. 어떻게 해?",
    choices: [
      { text: "답하기 불편할 수 있다고 먼저 양해를 구한다", score: 5 },
      { text: "질문을 살짝 바꿔서 물어본다", score: 4 },
      { text: "일단 물어보고 반응 보다가 사과한다", score: 2 },
      { text: "솔직한 게 좋으니 그냥 계속 묻는다", score: 1 },
    ],
  },
  {
    id: 8,
    trait: "respectfulness",
    situation: "자리가 생각보다 가깝고 신체 접촉이 생길 것 같아. 어떻게 해?",
    choices: [
      { text: "자연스럽게 거리를 만들어주거나 양해를 구한다", score: 5 },
      { text: "상대 표정 보고 불편하면 조심한다", score: 4 },
      { text: "어색해서 그냥 있는다", score: 2 },
      { text: "의식 안 하고 편하게 있는다", score: 1 },
    ],
  },
  {
    id: 9,
    trait: "communicationBalance",
    situation: "한 사람이 대화를 혼자 독점하고 있어. 어떻게 해?",
    choices: [
      { text: "다른 사람에게 질문을 넘겨서 균형을 맞춘다", score: 5 },
      { text: "자연스럽게 끼어들어 화제를 바꾼다", score: 4 },
      { text: "나도 같이 참여하며 지켜본다", score: 2 },
      { text: "그 사람이 지칠 때까지 기다린다", score: 1 },
    ],
  },
  {
    id: 10,
    trait: "communicationBalance",
    situation: "짝 대화에서 내 파트너가 많이 조용해. 어떻게 해?",
    choices: [
      { text: "가볍게 물어보면서 대화를 이끌어 본다", score: 5 },
      { text: "관심사를 물어보며 편하게 해준다", score: 4 },
      { text: "나도 조용히 있는다", score: 2 },
      { text: "불편해서 다른 대화에 끼려고 한다", score: 1 },
    ],
  },
];
```

- [ ] **Step 2: data/moodWeights.ts 작성**

```ts
import { MoodKey } from "@/types/matching";

export const moodWeights: Record<MoodKey, Record<MoodKey, number>> = {
  comfortableTalk: {
    comfortableTalk: 1.0,
    activeSocial:    0.6,
    gamesAndDrinks:  0.3,
    respectfulSafe:  0.8,
    naturalIntro:    0.9,
  },
  activeSocial: {
    comfortableTalk: 0.6,
    activeSocial:    1.0,
    gamesAndDrinks:  0.8,
    respectfulSafe:  0.4,
    naturalIntro:    0.7,
  },
  gamesAndDrinks: {
    comfortableTalk: 0.3,
    activeSocial:    0.8,
    gamesAndDrinks:  1.0,
    respectfulSafe:  0.2,
    naturalIntro:    0.5,
  },
  respectfulSafe: {
    comfortableTalk: 0.8,
    activeSocial:    0.4,
    gamesAndDrinks:  0.2,
    respectfulSafe:  1.0,
    naturalIntro:    0.7,
  },
  naturalIntro: {
    comfortableTalk: 0.9,
    activeSocial:    0.7,
    gamesAndDrinks:  0.5,
    respectfulSafe:  0.7,
    naturalIntro:    1.0,
  },
};
```

- [ ] **Step 3: data/mockTeams.ts 작성**

```ts
import { TeamProfile } from "@/types/matching";

export const mockTeams: TeamProfile[] = [
  {
    teamName: "용두산 삼총사",
    school: "부산대학교",
    region: "부산",
    size: 3,
    ageRange: "22~23",
    mood: "comfortableTalk",
    members: [
      {
        nickname: "민준",
        role: "coordinator",
        isLeader: true,
        traits: {
          atmosphereCoordination: 4,
          consideration: 4,
          participation: 3,
          respectfulness: 5,
          communicationBalance: 5,
        },
      },
      { nickname: "서연", role: "coordinator" },
      { nickname: "지호", role: "considerate" },
    ],
  },
  {
    teamName: "남포동 클럽",
    school: "부산대학교",
    region: "부산",
    size: 4,
    ageRange: "21~24",
    mood: "activeSocial",
    members: [
      {
        nickname: "현우",
        role: "moodMaker",
        isLeader: true,
        traits: {
          atmosphereCoordination: 5,
          consideration: 3,
          participation: 5,
          respectfulness: 3,
          communicationBalance: 4,
        },
      },
      { nickname: "은지", role: "moodMaker" },
      { nickname: "태양", role: "reactor" },
      { nickname: "소희", role: "reactor" },
    ],
  },
  {
    teamName: "해운대 게임단",
    school: "부산대학교",
    region: "부산",
    size: 3,
    ageRange: "22~24",
    mood: "gamesAndDrinks",
    members: [
      {
        nickname: "준혁",
        role: "moodMaker",
        isLeader: true,
        traits: {
          atmosphereCoordination: 5,
          consideration: 2,
          participation: 5,
          respectfulness: 3,
          communicationBalance: 3,
        },
      },
      { nickname: "다은", role: "reactor" },
      { nickname: "성민", role: "reactor" },
    ],
  },
  {
    teamName: "온천장 신사단",
    school: "부산대학교",
    region: "부산",
    size: 3,
    ageRange: "21~22",
    mood: "respectfulSafe",
    members: [
      {
        nickname: "도윤",
        role: "considerate",
        isLeader: true,
        traits: {
          atmosphereCoordination: 3,
          consideration: 5,
          participation: 3,
          respectfulness: 5,
          communicationBalance: 4,
        },
      },
      { nickname: "나연", role: "considerate" },
      { nickname: "재원", role: "coordinator" },
    ],
  },
  {
    teamName: "서면 인트로",
    school: "부산대학교",
    region: "부산",
    size: 4,
    ageRange: "20~23",
    mood: "naturalIntro",
    members: [
      {
        nickname: "수아",
        role: "coordinator",
        isLeader: true,
        traits: {
          atmosphereCoordination: 4,
          consideration: 4,
          participation: 4,
          respectfulness: 4,
          communicationBalance: 4,
        },
      },
      { nickname: "찬호", role: "moodMaker" },
      { nickname: "예린", role: "considerate" },
      { nickname: "민서", role: "reactor" },
    ],
  },
];
```

- [ ] **Step 4: Commit**

```bash
git add gwating-app/data/
git commit -m "feat: add data layer (10 quiz questions, 5x5 mood weights, 5 mock teams)"
```

---

## Task 4: lib/storage.ts

**Files:**
- Create: `gwating-app/lib/storage.ts`
- Create: `gwating-app/__tests__/storage.test.ts`

- [ ] **Step 1: 테스트 파일 작성 (실패 확인용)**

`gwating-app/__tests__/storage.test.ts`:

```ts
import { saveUser, loadUser, saveTeam, loadTeam, clearAll } from "@/lib/storage";
import { UserProfile, TeamProfile } from "@/types/matching";

const mockUser: UserProfile = {
  nickname: "테스터",
  traits: {
    atmosphereCoordination: 4,
    consideration: 3,
    participation: 5,
    respectfulness: 4,
    communicationBalance: 3,
  },
};

const mockTeam: TeamProfile = {
  teamName: "테스트팀",
  school: "부산대학교",
  region: "부산",
  size: 3,
  ageRange: "22~23",
  mood: "naturalIntro",
  members: [{ nickname: "테스터", role: "coordinator", isLeader: true }],
};

describe("storage", () => {
  beforeEach(() => {
    // Node 환경에서 localStorage mock
    const store: Record<string, string> = {};
    global.localStorage = {
      getItem: (key: string) => store[key] ?? null,
      setItem: (key: string, val: string) => { store[key] = val; },
      removeItem: (key: string) => { delete store[key]; },
      clear: () => { Object.keys(store).forEach(k => delete store[k]); },
      length: 0,
      key: () => null,
    };
  });

  it("saves and loads user", () => {
    saveUser(mockUser);
    expect(loadUser()).toEqual(mockUser);
  });

  it("returns null when no user saved", () => {
    expect(loadUser()).toBeNull();
  });

  it("saves and loads team", () => {
    saveTeam(mockTeam);
    expect(loadTeam()).toEqual(mockTeam);
  });

  it("returns null when no team saved", () => {
    expect(loadTeam()).toBeNull();
  });

  it("clearAll removes both keys", () => {
    saveUser(mockUser);
    saveTeam(mockTeam);
    clearAll();
    expect(loadUser()).toBeNull();
    expect(loadTeam()).toBeNull();
  });
});
```

- [ ] **Step 2: 테스트 실행 (FAIL 확인)**

```bash
cd gwating-app && npm test -- --testPathPattern=storage
```

Expected: `FAIL __tests__/storage.test.ts` — `Cannot find module '@/lib/storage'`

- [ ] **Step 3: lib/storage.ts 구현**

```ts
import { UserProfile, TeamProfile } from "@/types/matching";

const KEYS = {
  user: "gwating_user",
  team: "gwating_team",
} as const;

export function saveUser(profile: UserProfile): void {
  localStorage.setItem(KEYS.user, JSON.stringify(profile));
}

export function loadUser(): UserProfile | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(KEYS.user);
  return raw ? (JSON.parse(raw) as UserProfile) : null;
}

export function saveTeam(team: TeamProfile): void {
  localStorage.setItem(KEYS.team, JSON.stringify(team));
}

export function loadTeam(): TeamProfile | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(KEYS.team);
  return raw ? (JSON.parse(raw) as TeamProfile) : null;
}

export function clearAll(): void {
  localStorage.removeItem(KEYS.user);
  localStorage.removeItem(KEYS.team);
}
```

- [ ] **Step 4: 테스트 실행 (PASS 확인)**

```bash
npm test -- --testPathPattern=storage
```

Expected: `PASS __tests__/storage.test.ts` (5 tests)

- [ ] **Step 5: Commit**

```bash
cd .. && git add gwating-app/lib/storage.ts gwating-app/__tests__/storage.test.ts
git commit -m "feat: add storage lib with localStorage read/write helpers"
```

---

## Task 5: lib/scoring.ts

**Files:**
- Create: `gwating-app/lib/scoring.ts`
- Create: `gwating-app/__tests__/scoring.test.ts`

- [ ] **Step 1: 테스트 파일 작성**

`gwating-app/__tests__/scoring.test.ts`:

```ts
import { classifyRole, buildRoleVector, roleComplementarityScore } from "@/lib/scoring";
import { TraitKey } from "@/types/matching";

const highAtmParticipation: Record<TraitKey, number> = {
  atmosphereCoordination: 5,
  consideration: 1,
  participation: 5,
  respectfulness: 1,
  communicationBalance: 1,
};

const highConsiderationRespect: Record<TraitKey, number> = {
  atmosphereCoordination: 1,
  consideration: 5,
  participation: 1,
  respectfulness: 5,
  communicationBalance: 1,
};

const highCommBalance: Record<TraitKey, number> = {
  atmosphereCoordination: 2,
  consideration: 1,
  participation: 2,
  respectfulness: 1,
  communicationBalance: 5,
};

describe("classifyRole", () => {
  it("high atmosphereCoordination+participation → moodMaker", () => {
    expect(classifyRole(highAtmParticipation)).toBe("moodMaker");
  });

  it("high consideration+respectfulness → considerate", () => {
    expect(classifyRole(highConsiderationRespect)).toBe("considerate");
  });

  it("high communicationBalance → coordinator", () => {
    expect(classifyRole(highCommBalance)).toBe("coordinator");
  });
});

describe("buildRoleVector", () => {
  it("normalizes role counts to fractions", () => {
    const members = [
      { role: "moodMaker" as const },
      { role: "moodMaker" as const },
      { role: "reactor" as const },
      { role: "reactor" as const },
    ];
    const v = buildRoleVector(members);
    expect(v.moodMaker).toBeCloseTo(0.5);
    expect(v.reactor).toBeCloseTo(0.5);
    expect(v.coordinator).toBe(0);
    expect(v.considerate).toBe(0);
  });
});

describe("roleComplementarityScore", () => {
  it("identical teams score lower than complementary teams", () => {
    const allMoodMaker = buildRoleVector([
      { role: "moodMaker" }, { role: "moodMaker" },
    ]);
    const allConsiderate = buildRoleVector([
      { role: "considerate" }, { role: "considerate" },
    ]);
    const sameSame = roleComplementarityScore(allMoodMaker, allMoodMaker);
    const diff = roleComplementarityScore(allMoodMaker, allConsiderate);
    expect(diff).toBeGreaterThan(sameSame);
  });
});
```

- [ ] **Step 2: 테스트 실행 (FAIL 확인)**

```bash
npm test -- --testPathPattern=scoring
```

Expected: `FAIL __tests__/scoring.test.ts` — `Cannot find module '@/lib/scoring'`

- [ ] **Step 3: lib/scoring.ts 구현**

```ts
import { TraitKey, MemberRole } from "@/types/matching";

const ROLE_WEIGHTS: Record<MemberRole, Record<TraitKey, number>> = {
  moodMaker: {
    atmosphereCoordination: 0.6,
    consideration:          0.0,
    participation:          0.4,
    respectfulness:         0.0,
    communicationBalance:   0.0,
  },
  coordinator: {
    atmosphereCoordination: 0.4,
    consideration:          0.0,
    participation:          0.0,
    respectfulness:         0.0,
    communicationBalance:   0.6,
  },
  considerate: {
    atmosphereCoordination: 0.0,
    consideration:          0.6,
    participation:          0.0,
    respectfulness:         0.4,
    communicationBalance:   0.0,
  },
  reactor: {
    atmosphereCoordination: 0.0,
    consideration:          0.0,
    participation:          0.5,
    respectfulness:         0.0,
    communicationBalance:   0.5,
  },
};

export function classifyRole(traits: Record<TraitKey, number>): MemberRole {
  const roles: MemberRole[] = ["moodMaker", "coordinator", "considerate", "reactor"];
  let best: MemberRole = "coordinator";
  let bestScore = -1;

  for (const role of roles) {
    const weights = ROLE_WEIGHTS[role];
    const score = (Object.keys(weights) as TraitKey[]).reduce(
      (sum, key) => sum + traits[key] * weights[key],
      0
    );
    if (score > bestScore) {
      bestScore = score;
      best = role;
    }
  }
  return best;
}

export type RoleVector = Record<MemberRole, number>;

export function buildRoleVector(members: { role: MemberRole }[]): RoleVector {
  const counts: RoleVector = { moodMaker: 0, coordinator: 0, considerate: 0, reactor: 0 };
  for (const m of members) counts[m.role]++;
  const total = members.length || 1;
  return {
    moodMaker:   counts.moodMaker   / total,
    coordinator: counts.coordinator / total,
    considerate: counts.considerate / total,
    reactor:     counts.reactor     / total,
  };
}

export function roleComplementarityScore(a: RoleVector, b: RoleVector): number {
  const dot =
    a.moodMaker   * b.moodMaker +
    a.coordinator * b.coordinator +
    a.considerate * b.considerate +
    a.reactor     * b.reactor;

  const complementarity = 1 - dot;
  const hasCommonRole = (Object.keys(a) as MemberRole[]).some(
    (r) => a[r] > 0 && b[r] > 0
  );
  return Math.min(1, complementarity + (hasCommonRole ? 0.1 : 0));
}
```

- [ ] **Step 4: 테스트 실행 (PASS 확인)**

```bash
npm test -- --testPathPattern=scoring
```

Expected: `PASS __tests__/scoring.test.ts` (5 tests)

- [ ] **Step 5: Commit**

```bash
cd .. && git add gwating-app/lib/scoring.ts gwating-app/__tests__/scoring.test.ts
git commit -m "feat: add scoring lib (classifyRole, buildRoleVector, roleComplementarityScore)"
```

---

## Task 6: lib/matching.ts

**Files:**
- Create: `gwating-app/lib/matching.ts`
- Create: `gwating-app/__tests__/matching.test.ts`

- [ ] **Step 1: 테스트 파일 작성**

`gwating-app/__tests__/matching.test.ts`:

```ts
import { calculateMatchScore, rankTeams } from "@/lib/matching";
import { TeamProfile } from "@/types/matching";

const myTeam: TeamProfile = {
  teamName: "내팀",
  school: "부산대학교",
  region: "부산",
  size: 3,
  ageRange: "22~23",
  mood: "comfortableTalk",
  members: [
    { nickname: "나", role: "coordinator", isLeader: true,
      traits: { atmosphereCoordination: 4, consideration: 4, participation: 3, respectfulness: 5, communicationBalance: 5 } },
    { nickname: "친구1", role: "considerate" },
    { nickname: "친구2", role: "reactor" },
  ],
};

const sameMoodTeam: TeamProfile = {
  teamName: "비슷한팀",
  school: "부산대학교",
  region: "부산",
  size: 3,
  ageRange: "22~23",
  mood: "comfortableTalk",
  members: [
    { nickname: "가", role: "moodMaker", isLeader: true,
      traits: { atmosphereCoordination: 4, consideration: 3, participation: 4, respectfulness: 4, communicationBalance: 3 } },
    { nickname: "나", role: "reactor" },
    { nickname: "다", role: "reactor" },
  ],
};

const differentMoodTeam: TeamProfile = {
  teamName: "다른팀",
  school: "부산대학교",
  region: "부산",
  size: 3,
  ageRange: "22~23",
  mood: "gamesAndDrinks",
  members: [
    { nickname: "가", role: "moodMaker", isLeader: true,
      traits: { atmosphereCoordination: 5, consideration: 2, participation: 5, respectfulness: 2, communicationBalance: 3 } },
    { nickname: "나", role: "reactor" },
    { nickname: "다", role: "reactor" },
  ],
};

describe("calculateMatchScore", () => {
  it("returns score between 0 and 100", () => {
    const result = calculateMatchScore(myTeam, sameMoodTeam);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
  });

  it("same mood gives higher vibeScore than different mood", () => {
    const sameMood = calculateMatchScore(myTeam, sameMoodTeam);
    const diffMood = calculateMatchScore(myTeam, differentMoodTeam);
    expect(sameMood.vibeScore).toBeGreaterThan(diffMood.vibeScore);
  });

  it("provides 2-3 reasons", () => {
    const result = calculateMatchScore(myTeam, sameMoodTeam);
    expect(result.reasons.length).toBeGreaterThanOrEqual(2);
    expect(result.reasons.length).toBeLessThanOrEqual(3);
  });

  it("label is 'Strong vibe fit' when score >= 80", () => {
    const result = calculateMatchScore(myTeam, sameMoodTeam);
    if (result.score >= 80) {
      expect(result.label).toBe("Strong vibe fit");
    }
  });
});

describe("rankTeams", () => {
  it("returns teams sorted by score descending", () => {
    const ranked = rankTeams(myTeam, [differentMoodTeam, sameMoodTeam]);
    expect(ranked[0].score).toBeGreaterThanOrEqual(ranked[1].score);
  });
});
```

- [ ] **Step 2: 테스트 실행 (FAIL 확인)**

```bash
npm test -- --testPathPattern=matching
```

Expected: `FAIL __tests__/matching.test.ts` — `Cannot find module '@/lib/matching'`

- [ ] **Step 3: lib/matching.ts 구현**

```ts
import { TeamProfile, MatchResult, MoodKey } from "@/types/matching";
import { moodWeights } from "@/data/moodWeights";
import { buildRoleVector, roleComplementarityScore } from "@/lib/scoring";

const MOOD_LABELS: Record<MoodKey, string> = {
  comfortableTalk: "편한 대화형",
  activeSocial:    "활발한 친목형",
  gamesAndDrinks:  "게임/술자리형",
  respectfulSafe:  "예의/안전 중시형",
  naturalIntro:    "자연스러운 소개팅형",
};

function calcVibeScore(myMood: MoodKey, theirMood: MoodKey): number {
  return moodWeights[myMood][theirMood];
}

function calcRoleBalanceScore(my: TeamProfile, their: TeamProfile): number {
  return roleComplementarityScore(
    buildRoleVector(my.members),
    buildRoleVector(their.members)
  );
}

function calcConditionScore(my: TeamProfile, their: TeamProfile): number {
  const sizeDiff = Math.abs(my.size - their.size);
  const sizeScore = sizeDiff === 0 ? 1.0 : sizeDiff === 1 ? 0.5 : 0.0;

  const parseRange = (r: string): [number, number] => {
    const [min, max] = r.split("~").map(Number);
    return [min, max];
  };
  const [myMin, myMax] = parseRange(my.ageRange);
  const [thMin, thMax] = parseRange(their.ageRange);
  const ageScore = Math.max(myMin, thMin) <= Math.min(myMax, thMax) ? 1.0 : 0.0;

  return sizeScore * 0.5 + ageScore * 0.5;
}

function scoreToLabel(score: number): MatchResult["label"] {
  if (score >= 80) return "Strong vibe fit";
  if (score >= 60) return "Good with some differences";
  return "Different atmosphere preferences";
}

function generateReasons(
  my: TeamProfile,
  their: TeamProfile,
  vibeRaw: number,
  roleRaw: number,
  condRaw: number
): string[] {
  const reasons: string[] = [];

  if (vibeRaw >= 0.8) {
    reasons.push(`두 팀 모두 ${MOOD_LABELS[my.mood]} 분위기를 선호해요.`);
  } else if (vibeRaw >= 0.5) {
    reasons.push(
      `${MOOD_LABELS[their.mood]}인 상대팀이 여러분의 분위기에 잘 맞춰줄 수 있어요.`
    );
  }

  if (roleRaw >= 0.7) {
    reasons.push("두 팀의 역할 구성이 서로를 잘 보완해요.");
  } else if (roleRaw >= 0.5) {
    reasons.push("상대팀이 초반 어색함을 줄여줄 수 있는 역할을 갖고 있어요.");
  }

  if (condRaw >= 0.75) {
    reasons.push("팀 인원과 나이대가 비슷해 편안한 만남이 될 거예요.");
  } else if (my.size === their.size) {
    reasons.push("팀 인원이 같아서 자리 구성이 자연스러워요.");
  }

  if (reasons.length < 2) {
    reasons.push("두 팀이 가볍고 부담 없는 만남을 만들 수 있어요.");
  }

  return reasons.slice(0, 3);
}

export function calculateMatchScore(
  myTeam: TeamProfile,
  candidate: TeamProfile
): MatchResult {
  const vibeRaw = calcVibeScore(myTeam.mood, candidate.mood);
  const roleRaw = calcRoleBalanceScore(myTeam, candidate);
  const condRaw = calcConditionScore(myTeam, candidate);
  const score = Math.round(vibeRaw * 40 + roleRaw * 35 + condRaw * 25);

  return {
    team:           candidate,
    score,
    vibeScore:      Math.round(vibeRaw * 100),
    roleScore:      Math.round(roleRaw * 100),
    conditionScore: Math.round(condRaw * 100),
    reasons:        generateReasons(myTeam, candidate, vibeRaw, roleRaw, condRaw),
    label:          scoreToLabel(score),
  };
}

export function rankTeams(
  myTeam: TeamProfile,
  candidates: TeamProfile[]
): MatchResult[] {
  return candidates
    .map((c) => calculateMatchScore(myTeam, c))
    .sort((a, b) => b.score - a.score);
}
```

- [ ] **Step 4: 테스트 실행 (PASS 확인)**

```bash
npm test -- --testPathPattern=matching
```

Expected: `PASS __tests__/matching.test.ts` (5 tests)

- [ ] **Step 5: 전체 테스트 확인**

```bash
npm test
```

Expected: `Test Suites: 3 passed, 3 total` / `Tests: 13 passed, 13 total`

- [ ] **Step 6: Commit**

```bash
cd .. && git add gwating-app/lib/matching.ts gwating-app/__tests__/matching.test.ts
git commit -m "feat: add matching lib (calculateMatchScore, rankTeams)"
```

---

## Task 7: Shared Components

**Files:**
- Create: `gwating-app/components/AppHeader.tsx`
- Create: `gwating-app/components/Button.tsx`
- Create: `gwating-app/components/MoodChip.tsx`
- Create: `gwating-app/components/MoodSelector.tsx`

- [ ] **Step 1: components/Button.tsx 작성**

```tsx
import { ButtonHTMLAttributes } from "react";

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary";
  fullWidth?: boolean;
};

export function Button({
  variant = "primary",
  fullWidth = false,
  className = "",
  children,
  ...props
}: Props) {
  const base = "h-12 px-6 rounded-sm text-base font-semibold transition-colors disabled:opacity-50";
  const variants = {
    primary:   "bg-primary text-white hover:bg-primary-active",
    secondary: "bg-white text-ink border border-hairline hover:bg-surface-soft",
  };
  return (
    <button
      className={`${base} ${variants[variant]} ${fullWidth ? "w-full" : ""} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
```

- [ ] **Step 2: components/MoodChip.tsx 작성**

```tsx
import { MoodKey } from "@/types/matching";

const MOOD_CONFIG: Record<MoodKey, { label: string; bg: string; text: string; border: string }> = {
  comfortableTalk: { label: "편한 대화형",      bg: "bg-primary-soft",  text: "text-primary",      border: "border-primary" },
  activeSocial:    { label: "활발한 친목형",     bg: "bg-mint",          text: "text-mint-ink",     border: "border-mint-ink" },
  gamesAndDrinks:  { label: "게임/술자리형",     bg: "bg-amber",         text: "text-amber-ink",    border: "border-amber-ink" },
  respectfulSafe:  { label: "예의/안전 중시형",  bg: "bg-lavender",      text: "text-lavender-ink", border: "border-lavender-ink" },
  naturalIntro:    { label: "자연스러운 소개팅형", bg: "bg-sky",          text: "text-sky-ink",      border: "border-sky-ink" },
};

type Props = {
  mood: MoodKey;
  selected?: boolean;
  onClick?: () => void;
};

export function MoodChip({ mood, selected = false, onClick }: Props) {
  const cfg = MOOD_CONFIG[mood];
  return (
    <button
      type="button"
      onClick={onClick}
      className={`
        inline-flex items-center px-4 py-2 rounded-full text-xs font-semibold border transition-all
        ${selected
          ? `${cfg.bg} ${cfg.text} ${cfg.border}`
          : "bg-white text-muted border-hairline hover:border-body"}
      `}
    >
      {cfg.label}
    </button>
  );
}

export { MOOD_CONFIG };
```

- [ ] **Step 3: components/MoodSelector.tsx 작성**

```tsx
import { MoodKey } from "@/types/matching";
import { MoodChip } from "./MoodChip";

const ALL_MOODS: MoodKey[] = [
  "comfortableTalk",
  "activeSocial",
  "gamesAndDrinks",
  "respectfulSafe",
  "naturalIntro",
];

type Props = {
  value: MoodKey | null;
  onChange: (mood: MoodKey) => void;
};

export function MoodSelector({ value, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {ALL_MOODS.map((mood) => (
        <MoodChip
          key={mood}
          mood={mood}
          selected={value === mood}
          onClick={() => onChange(mood)}
        />
      ))}
    </div>
  );
}
```

- [ ] **Step 4: components/AppHeader.tsx 작성**

```tsx
import Link from "next/link";

type Props = {
  step?: number;
  totalSteps?: number;
};

export function AppHeader({ step, totalSteps }: Props) {
  return (
    <header className="h-14 md:h-16 border-b border-hairline-soft bg-white sticky top-0 z-10">
      <div className="max-w-[1120px] mx-auto px-4 h-full flex items-center justify-between">
        <Link href="/" className="flex items-center gap-1.5">
          <span className="text-primary font-bold text-xl leading-none">●</span>
          <span className="font-bold text-ink text-base">부산대 과팅</span>
          <span className="text-xs text-muted border border-hairline rounded-full px-2 py-0.5 ml-1">
            베타
          </span>
        </Link>
        {step !== undefined && totalSteps !== undefined && (
          <span className="text-xs font-semibold text-muted">
            {step} / {totalSteps} 단계
          </span>
        )}
      </div>
    </header>
  );
}
```

- [ ] **Step 5: Commit**

```bash
cd .. && git add gwating-app/components/
git commit -m "feat: add shared components (Button, MoodChip, MoodSelector, AppHeader)"
```

---

## Task 8: Layout & Global Styles

**Files:**
- Modify: `gwating-app/app/globals.css`
- Modify: `gwating-app/app/layout.tsx`

- [ ] **Step 1: app/globals.css 교체**

```css
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css');
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  font-family: "Pretendard Variable", Inter, system-ui, sans-serif;
}

* {
  -webkit-font-smoothing: antialiased;
}

body {
  background-color: #ffffff;
  color: #222222;
}
```

- [ ] **Step 2: app/layout.tsx 교체**

```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "부산대 과팅 매칭",
  description: "부산대생 전용 그룹 과팅 매칭 서비스 베타",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="min-h-screen bg-canvas text-ink">{children}</body>
    </html>
  );
}
```

- [ ] **Step 3: 개발 서버 확인**

```bash
cd gwating-app && npm run dev
```

브라우저에서 `http://localhost:3000` 열기. Expected: 빈 흰 페이지, Pretendard 폰트 로드됨 (Network 탭에서 확인).

- [ ] **Step 4: Commit**

```bash
cd .. && git add gwating-app/app/globals.css gwating-app/app/layout.tsx
git commit -m "feat: add layout and global styles (Pretendard font, canvas white)"
```

---

## Task 9: Home Page

**Files:**
- Modify: `gwating-app/app/page.tsx`

- [ ] **Step 1: app/page.tsx 작성**

```tsx
import Link from "next/link";
import { AppHeader } from "@/components/AppHeader";
import { Button } from "@/components/Button";
import { MoodChip } from "@/components/MoodChip";

export default function HomePage() {
  return (
    <>
      <AppHeader />
      <main>
        {/* Hero */}
        <section className="bg-canvas-warm py-16 px-4">
          <div className="max-w-[560px] mx-auto text-center">
            <p className="text-xs font-semibold text-primary mb-3 tracking-wide uppercase">
              부산대생 전용 베타
            </p>
            <h1 className="text-[32px] font-bold text-ink leading-tight mb-4">
              우리 팀 분위기에 딱 맞는<br />과팅 상대를 찾아보세요
            </h1>
            <p className="text-base text-body mb-8">
              성향 테스트로 역할을 파악하고, 팀을 만들어 궁합 점수를 확인하세요.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Link href="/test">
                <Button variant="primary" className="rounded-full">
                  성향 테스트 시작
                </Button>
              </Link>
              <Link href="/team/create">
                <Button variant="secondary" className="rounded-full">
                  팀 바로 만들기
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="py-14 px-4">
          <div className="max-w-[700px] mx-auto">
            <h2 className="text-2xl font-bold text-ink mb-10 text-center">어떻게 진행되나요?</h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              {[
                { step: "01", title: "성향 테스트", desc: "10개 상황 문항으로 나의 과팅 스타일을 파악해요." },
                { step: "02", title: "팀 생성",     desc: "팀명과 분위기를 선택하고 팀원의 역할을 입력해요." },
                { step: "03", title: "매칭 추천",   desc: "궁합 점수와 이유를 바탕으로 상대팀을 추천해드려요." },
              ].map(({ step, title, desc }) => (
                <div key={step} className="bg-surface-soft rounded-lg p-6">
                  <p className="text-xs font-semibold text-primary mb-2">{step}</p>
                  <h3 className="text-base font-semibold text-ink mb-2">{title}</h3>
                  <p className="text-sm text-body">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Mood examples */}
        <section className="py-10 px-4 border-t border-hairline-soft">
          <div className="max-w-[560px] mx-auto">
            <p className="text-sm font-semibold text-muted text-center mb-4">어떤 분위기를 원하세요?</p>
            <div className="flex flex-wrap gap-2 justify-center">
              <MoodChip mood="comfortableTalk" />
              <MoodChip mood="activeSocial" />
              <MoodChip mood="gamesAndDrinks" />
              <MoodChip mood="respectfulSafe" />
              <MoodChip mood="naturalIntro" />
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
```

- [ ] **Step 2: 브라우저에서 확인**

`http://localhost:3000` — Hero 섹션, How it works 3단계, 분위기 칩 5개 표시 확인.

- [ ] **Step 3: Commit**

```bash
cd .. && git add gwating-app/app/page.tsx
git commit -m "feat: add home page with hero, how-it-works, mood chips"
```

---

## Task 10: Quiz Page (/test)

**Files:**
- Create: `gwating-app/components/QuizCard.tsx`
- Create: `gwating-app/app/test/page.tsx`

- [ ] **Step 1: components/QuizCard.tsx 작성**

```tsx
import { QuizQuestion, QuizChoice } from "@/data/questions";

type Props = {
  question: QuizQuestion;
  current: number;
  total: number;
  onSelect: (score: number) => void;
};

export function QuizCard({ question, current, total, onSelect }: Props) {
  const progress = (current / total) * 100;

  return (
    <div className="bg-white rounded-lg shadow-card p-6 max-w-[560px] w-full mx-auto">
      {/* Progress */}
      <div className="flex justify-between text-xs text-muted mb-3">
        <span>Question {current} of {total}</span>
      </div>
      <div className="h-1.5 bg-surface-soft rounded-full mb-6">
        <div
          className="h-full bg-primary rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Question */}
      <p className="text-base font-semibold text-ink mb-6 leading-snug">
        {question.situation}
      </p>

      {/* Choices */}
      <div className="flex flex-col gap-3">
        {question.choices.map((choice: QuizChoice, i: number) => (
          <button
            key={i}
            type="button"
            onClick={() => onSelect(choice.score)}
            className="
              text-left px-4 py-4 rounded-md border border-hairline text-sm text-body
              hover:border-primary hover:bg-primary-soft transition-all min-h-[56px]
            "
          >
            {choice.text}
          </button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: app/test/page.tsx 작성**

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/AppHeader";
import { QuizCard } from "@/components/QuizCard";
import { Button } from "@/components/Button";
import { questions } from "@/data/questions";
import { saveUser } from "@/lib/storage";
import { classifyRole } from "@/lib/scoring";
import { TraitKey, MemberRole, UserProfile } from "@/types/matching";

const ROLE_LABELS: Record<MemberRole, { name: string; desc: string; emoji: string }> = {
  moodMaker:   { name: "분위기 메이커형", desc: "에너지를 끌어올리고 자리를 살려주는 역할이에요.",  emoji: "🔥" },
  coordinator: { name: "조율자형",        desc: "대화 흐름을 이어주고 균형을 맞추는 역할이에요.",    emoji: "🎯" },
  considerate: { name: "배려형",          desc: "모두를 세심하게 챙기는 역할이에요.",               emoji: "🤍" },
  reactor:     { name: "리액션형",        desc: "분위기를 살려주는 반응으로 자리를 따뜻하게 해요.", emoji: "✨" },
};

type TraitScores = Partial<Record<TraitKey, number[]>>;

export default function TestPage() {
  const router = useRouter();
  const [currentIdx, setCurrentIdx] = useState(0);
  const [traitScores, setTraitScores] = useState<TraitScores>({});
  const [nickname, setNickname] = useState("");
  const [showNickname, setShowNickname] = useState(false);
  const [resultRole, setResultRole] = useState<MemberRole | null>(null);
  const [finalTraits, setFinalTraits] = useState<Record<TraitKey, number> | null>(null);

  const current = questions[currentIdx];
  const isDone = currentIdx >= questions.length;

  function handleSelect(score: number) {
    const trait = current.trait;
    const updated: TraitScores = {
      ...traitScores,
      [trait]: [...(traitScores[trait] ?? []), score],
    };
    setTraitScores(updated);

    if (currentIdx + 1 >= questions.length) {
      // 계산
      const allTraits: TraitKey[] = [
        "atmosphereCoordination", "consideration", "participation",
        "respectfulness", "communicationBalance",
      ];
      const traits = allTraits.reduce((acc, key) => {
        const scores = updated[key] ?? [3];
        acc[key] = Math.round(scores.reduce((s, v) => s + v, 0) / scores.length);
        return acc;
      }, {} as Record<TraitKey, number>);

      setFinalTraits(traits);
      setResultRole(classifyRole(traits));
      setShowNickname(true);
    } else {
      setCurrentIdx((i) => i + 1);
    }
  }

  function handleSave() {
    if (!nickname.trim() || !finalTraits) return;
    const profile: UserProfile = { nickname: nickname.trim(), traits: finalTraits };
    saveUser(profile);
    router.push("/team/create");
  }

  if (showNickname && resultRole) {
    const info = ROLE_LABELS[resultRole];
    return (
      <>
        <AppHeader step={3} totalSteps={3} />
        <main className="py-12 px-4">
          <div className="max-w-[480px] mx-auto text-center">
            <div className="text-5xl mb-4">{info.emoji}</div>
            <h2 className="text-2xl font-bold text-ink mb-2">{info.name}</h2>
            <p className="text-sm text-body mb-8">{info.desc}</p>
            <div className="mb-6 text-left">
              <label className="block text-sm font-semibold text-ink mb-2">
                닉네임을 입력해주세요
              </label>
              <input
                type="text"
                value={nickname}
                onChange={(e) => setNickname(e.target.value)}
                placeholder="예: 민준"
                maxLength={10}
                className="w-full border border-hairline rounded-sm px-4 h-12 text-base text-ink focus:outline-none focus:border-primary"
              />
            </div>
            <Button fullWidth onClick={handleSave} disabled={!nickname.trim()}>
              팀 만들러 가기
            </Button>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <AppHeader step={1} totalSteps={3} />
      <main className="py-10 px-4 bg-canvas-warm min-h-screen">
        <div className="mb-8 text-center">
          <h1 className="text-2xl font-bold text-ink">나의 과팅 스타일은?</h1>
          <p className="text-sm text-muted mt-1">상황을 읽고 솔직하게 골라주세요</p>
        </div>
        {!isDone && (
          <QuizCard
            question={current}
            current={currentIdx + 1}
            total={questions.length}
            onSelect={handleSelect}
          />
        )}
      </main>
    </>
  );
}
```

- [ ] **Step 3: 브라우저에서 확인**

`http://localhost:3000/test` — 10문항이 순서대로 표시되고, 완료 후 역할 결과 + 닉네임 입력 화면으로 전환됨.

- [ ] **Step 4: Commit**

```bash
cd .. && git add gwating-app/components/QuizCard.tsx gwating-app/app/test/
git commit -m "feat: add quiz page with 10 situational questions and role auto-classification"
```

---

## Task 11: Team Create Page (/team/create)

**Files:**
- Create: `gwating-app/components/MemberRoleCard.tsx`
- Create: `gwating-app/components/TeamCreateForm.tsx`
- Create: `gwating-app/app/team/create/page.tsx`

- [ ] **Step 1: components/MemberRoleCard.tsx 작성**

```tsx
import { MemberRole } from "@/types/matching";

const ROLES: { value: MemberRole; label: string; emoji: string }[] = [
  { value: "moodMaker",   label: "분위기 메이커형", emoji: "🔥" },
  { value: "coordinator", label: "조율자형",         emoji: "🎯" },
  { value: "considerate", label: "배려형",            emoji: "🤍" },
  { value: "reactor",     label: "리액션형",          emoji: "✨" },
];

type Props = {
  index: number;
  nickname: string;
  role: MemberRole | "";
  onNicknameChange: (val: string) => void;
  onRoleChange: (val: MemberRole) => void;
  onRemove: () => void;
};

export function MemberRoleCard({
  index, nickname, role, onNicknameChange, onRoleChange, onRemove,
}: Props) {
  return (
    <div className="border border-hairline rounded-md p-4 bg-surface-soft">
      <div className="flex justify-between items-center mb-3">
        <span className="text-xs font-semibold text-muted">팀원 {index + 1}</span>
        <button
          type="button"
          onClick={onRemove}
          className="text-xs text-muted hover:text-primary"
        >
          삭제
        </button>
      </div>
      <input
        type="text"
        value={nickname}
        onChange={(e) => onNicknameChange(e.target.value)}
        placeholder="닉네임"
        maxLength={10}
        className="w-full border border-hairline rounded-sm px-3 h-10 text-sm text-ink mb-3 focus:outline-none focus:border-primary bg-white"
      />
      <div className="grid grid-cols-2 gap-2">
        {ROLES.map((r) => (
          <button
            key={r.value}
            type="button"
            onClick={() => onRoleChange(r.value)}
            className={`
              flex items-center gap-1.5 px-3 py-2 rounded-sm border text-xs font-semibold transition-all
              ${role === r.value
                ? "bg-primary-soft border-primary text-primary"
                : "bg-white border-hairline text-muted hover:border-body"}
            `}
          >
            <span>{r.emoji}</span>
            <span>{r.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: app/team/create/page.tsx 작성**

```tsx
"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/AppHeader";
import { Button } from "@/components/Button";
import { MoodSelector } from "@/components/MoodSelector";
import { MemberRoleCard } from "@/components/MemberRoleCard";
import { loadUser, saveTeam } from "@/lib/storage";
import { classifyRole } from "@/lib/scoring";
import { MoodKey, MemberRole, TeamProfile, TeamMember } from "@/types/matching";

type MemberDraft = { nickname: string; role: MemberRole | "" };

export default function TeamCreatePage() {
  const router = useRouter();
  const [teamName, setTeamName] = useState("");
  const [ageRange, setAgeRange] = useState("");
  const [mood, setMood] = useState<MoodKey | null>(null);
  const [leader, setLeader] = useState<TeamMember | null>(null);
  const [extraMembers, setExtraMembers] = useState<MemberDraft[]>([
    { nickname: "", role: "" },
  ]);

  useEffect(() => {
    const user = loadUser();
    if (!user) return;
    const role = classifyRole(user.traits);
    setLeader({
      nickname:  user.nickname,
      role,
      traits:    user.traits,
      isLeader:  true,
    });
  }, []);

  function addMember() {
    if (extraMembers.length >= 4) return;
    setExtraMembers((prev) => [...prev, { nickname: "", role: "" }]);
  }

  function removeMember(i: number) {
    setExtraMembers((prev) => prev.filter((_, idx) => idx !== i));
  }

  function updateMember(i: number, patch: Partial<MemberDraft>) {
    setExtraMembers((prev) =>
      prev.map((m, idx) => (idx === i ? { ...m, ...patch } : m))
    );
  }

  const isValid =
    teamName.trim() &&
    ageRange.trim() &&
    mood &&
    leader &&
    extraMembers.every((m) => m.nickname.trim() && m.role !== "");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!isValid || !leader || !mood) return;

    const members: TeamMember[] = [
      leader,
      ...extraMembers.map((m) => ({
        nickname: m.nickname.trim(),
        role:     m.role as MemberRole,
      })),
    ];

    const team: TeamProfile = {
      teamName:  teamName.trim(),
      school:    "부산대학교",
      region:    "부산",
      size:      members.length,
      ageRange:  ageRange.trim(),
      mood,
      members,
    };
    saveTeam(team);
    router.push("/team/demo");
  }

  if (!leader) {
    return (
      <>
        <AppHeader />
        <main className="py-20 px-4 text-center max-w-[480px] mx-auto">
          <p className="text-body mb-6">성향 테스트를 먼저 완료해야 팀을 만들 수 있어요.</p>
          <Button onClick={() => router.push("/test")}>성향 테스트 하러 가기</Button>
        </main>
      </>
    );
  }

  return (
    <>
      <AppHeader step={2} totalSteps={3} />
      <main className="py-10 px-4">
        <div className="max-w-[560px] mx-auto">
          <h1 className="text-2xl font-bold text-ink mb-1">팀 만들기</h1>
          <p className="text-sm text-muted mb-8">팀 정보를 입력하고 팀원 역할을 골라주세요</p>

          <form onSubmit={handleSubmit} className="flex flex-col gap-6">
            {/* 팀명 */}
            <div>
              <label className="block text-sm font-semibold text-ink mb-2">팀 이름</label>
              <input
                type="text"
                value={teamName}
                onChange={(e) => setTeamName(e.target.value)}
                placeholder="예: 서면 드리머즈"
                maxLength={20}
                className="w-full border border-hairline rounded-sm px-4 h-12 text-base text-ink focus:outline-none focus:border-primary"
              />
            </div>

            {/* 나이대 */}
            <div>
              <label className="block text-sm font-semibold text-ink mb-2">나이대</label>
              <input
                type="text"
                value={ageRange}
                onChange={(e) => setAgeRange(e.target.value)}
                placeholder="예: 22~24"
                maxLength={10}
                className="w-full border border-hairline rounded-sm px-4 h-12 text-base text-ink focus:outline-none focus:border-primary"
              />
            </div>

            {/* 원하는 분위기 */}
            <div>
              <label className="block text-sm font-semibold text-ink mb-2">원하는 과팅 분위기</label>
              <MoodSelector value={mood} onChange={setMood} />
            </div>

            {/* 팀장 */}
            <div>
              <label className="block text-sm font-semibold text-ink mb-2">팀장 (나)</label>
              <div className="border border-primary bg-primary-soft rounded-md p-4 text-sm">
                <span className="font-semibold text-ink">{leader.nickname}</span>
                <span className="ml-2 text-primary font-semibold">
                  {leader.role === "moodMaker"   && "🔥 분위기 메이커형"}
                  {leader.role === "coordinator" && "🎯 조율자형"}
                  {leader.role === "considerate" && "🤍 배려형"}
                  {leader.role === "reactor"     && "✨ 리액션형"}
                </span>
                <span className="ml-2 text-xs text-muted">(성향 테스트 결과)</span>
              </div>
            </div>

            {/* 팀원 */}
            <div>
              <label className="block text-sm font-semibold text-ink mb-2">
                팀원 ({extraMembers.length}/4)
              </label>
              <div className="flex flex-col gap-3">
                {extraMembers.map((m, i) => (
                  <MemberRoleCard
                    key={i}
                    index={i}
                    nickname={m.nickname}
                    role={m.role}
                    onNicknameChange={(val) => updateMember(i, { nickname: val })}
                    onRoleChange={(val) => updateMember(i, { role: val })}
                    onRemove={() => removeMember(i)}
                  />
                ))}
                {extraMembers.length < 4 && (
                  <button
                    type="button"
                    onClick={addMember}
                    className="border border-dashed border-hairline rounded-md py-3 text-sm text-muted hover:border-primary hover:text-primary transition-colors"
                  >
                    + 팀원 추가
                  </button>
                )}
              </div>
            </div>

            <Button type="submit" fullWidth disabled={!isValid}>
              팀 프로필 만들기
            </Button>
          </form>
        </div>
      </main>
    </>
  );
}
```

- [ ] **Step 3: 브라우저에서 확인**

`http://localhost:3000/team/create` — 팀장 자동 표시, 팀원 추가/삭제, 역할 선택, 분위기 선택 확인. Submit 시 `/team/demo`로 이동.

- [ ] **Step 4: Commit**

```bash
cd .. && git add gwating-app/components/MemberRoleCard.tsx gwating-app/app/team/create/
git commit -m "feat: add team create page with member role selection"
```

---

## Task 12: Team Demo Page (/team/demo)

**Files:**
- Create: `gwating-app/components/TeamProfileCard.tsx`
- Create: `gwating-app/app/team/demo/page.tsx`

- [ ] **Step 1: components/TeamProfileCard.tsx 작성**

```tsx
import { TeamProfile, MemberRole } from "@/types/matching";
import { MoodChip } from "./MoodChip";

const ROLE_INFO: Record<MemberRole, { label: string; emoji: string }> = {
  moodMaker:   { label: "분위기 메이커형", emoji: "🔥" },
  coordinator: { label: "조율자형",         emoji: "🎯" },
  considerate: { label: "배려형",           emoji: "🤍" },
  reactor:     { label: "리액션형",         emoji: "✨" },
};

type Props = { team: TeamProfile };

export function TeamProfileCard({ team }: Props) {
  const initials = team.teamName.slice(0, 2);
  const leaderRoles = team.members.filter((m) => m.isLeader).map((m) => m.role);

  return (
    <div className="bg-white rounded-lg shadow-card p-6 max-w-[480px] w-full">
      {/* 팀 아이덴티티 */}
      <div className="flex items-center gap-4 mb-5">
        <div className="w-14 h-14 rounded-full bg-primary-soft flex items-center justify-center text-xl font-bold text-primary">
          {initials}
        </div>
        <div>
          <h2 className="text-xl font-bold text-ink">{team.teamName}</h2>
          <p className="text-sm text-muted">
            {team.school} · {team.size}명 · {team.ageRange}세
          </p>
        </div>
      </div>

      {/* 분위기 */}
      <div className="mb-5">
        <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">원하는 분위기</p>
        <MoodChip mood={team.mood} selected />
      </div>

      {/* 멤버 역할 */}
      <div>
        <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">팀원 구성</p>
        <div className="flex flex-col gap-2">
          {team.members.map((m, i) => {
            const info = ROLE_INFO[m.role];
            return (
              <div key={i} className="flex items-center justify-between text-sm">
                <span className="text-ink font-medium">
                  {m.nickname}
                  {m.isLeader && (
                    <span className="ml-1.5 text-xs text-primary font-semibold">팀장</span>
                  )}
                </span>
                <span className="text-muted">
                  {info.emoji} {info.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: app/team/demo/page.tsx 작성**

```tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/AppHeader";
import { Button } from "@/components/Button";
import { TeamProfileCard } from "@/components/TeamProfileCard";
import { loadTeam } from "@/lib/storage";
import { TeamProfile } from "@/types/matching";

export default function TeamDemoPage() {
  const router = useRouter();
  const [team, setTeam] = useState<TeamProfile | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTeam(loadTeam());
    setLoading(false);
  }, []);

  if (loading) return null;

  if (!team) {
    return (
      <>
        <AppHeader />
        <main className="py-20 px-4 text-center max-w-[480px] mx-auto">
          <p className="text-body mb-6">팀 정보가 없어요. 팀을 먼저 만들어주세요.</p>
          <Button onClick={() => router.push("/team/create")}>팀 만들러 가기</Button>
        </main>
      </>
    );
  }

  return (
    <>
      <AppHeader step={2} totalSteps={3} />
      <main className="py-10 px-4 bg-canvas-warm min-h-screen">
        <div className="max-w-[560px] mx-auto">
          <h1 className="text-2xl font-bold text-ink mb-1">우리 팀 프로필</h1>
          <p className="text-sm text-muted mb-8">이 팀으로 매칭을 진행할게요</p>
          <TeamProfileCard team={team} />
          <div className="mt-6 flex flex-col gap-3">
            <Button fullWidth onClick={() => router.push("/match")}>
              매칭 팀 찾기
            </Button>
            <Button
              variant="secondary"
              fullWidth
              onClick={() => router.push("/team/create")}
            >
              팀 수정하기
            </Button>
          </div>
        </div>
      </main>
    </>
  );
}
```

- [ ] **Step 3: 브라우저에서 확인**

`http://localhost:3000/team/demo` — 팀 카드에 이름, 학교, 인원, 나이, 분위기, 역할 목록 표시. "매칭 팀 찾기" 클릭 시 `/match`로 이동.

- [ ] **Step 4: Commit**

```bash
cd .. && git add gwating-app/components/TeamProfileCard.tsx gwating-app/app/team/demo/
git commit -m "feat: add team demo page with TeamProfileCard"
```

---

## Task 13: Match Page (/match)

**Files:**
- Create: `gwating-app/components/MatchScoreCard.tsx`
- Create: `gwating-app/components/MatchReasonList.tsx`
- Create: `gwating-app/components/RecommendationTeamCard.tsx`
- Create: `gwating-app/app/match/page.tsx`

- [ ] **Step 1: components/MatchScoreCard.tsx 작성**

```tsx
import { MatchResult } from "@/types/matching";

const LABEL_COLORS: Record<MatchResult["label"], string> = {
  "Strong vibe fit":                "text-primary",
  "Good with some differences":     "text-amber-ink",
  "Different atmosphere preferences": "text-muted",
};

type Props = { score: number; label: MatchResult["label"] };

export function MatchScoreCard({ score, label }: Props) {
  return (
    <div className="text-center">
      <p className="text-[48px] font-extrabold text-primary leading-none">{score}%</p>
      <p className={`text-sm font-semibold mt-1 ${LABEL_COLORS[label]}`}>{label}</p>
    </div>
  );
}
```

- [ ] **Step 2: components/MatchReasonList.tsx 작성**

```tsx
type Props = { reasons: string[] };

export function MatchReasonList({ reasons }: Props) {
  return (
    <ul className="flex flex-col gap-2">
      {reasons.map((r, i) => (
        <li key={i} className="flex gap-2 text-sm text-sky-ink bg-sky rounded-md px-3 py-2">
          <span className="shrink-0">✦</span>
          <span>{r}</span>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 3: components/RecommendationTeamCard.tsx 작성**

```tsx
import { MatchResult, MemberRole } from "@/types/matching";
import { MoodChip } from "./MoodChip";
import { MatchScoreCard } from "./MatchScoreCard";
import { MatchReasonList } from "./MatchReasonList";

const ROLE_EMOJI: Record<MemberRole, string> = {
  moodMaker: "🔥", coordinator: "🎯", considerate: "🤍", reactor: "✨",
};

type Props = { result: MatchResult; rank: number };

export function RecommendationTeamCard({ result, rank }: Props) {
  const { team, score, label, reasons } = result;
  const initials = team.teamName.slice(0, 2);

  return (
    <div className="bg-white rounded-lg shadow-card p-5 flex flex-col gap-4">
      {/* 헤더 */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full bg-surface-soft flex items-center justify-center text-base font-bold text-ink shrink-0">
            {initials}
          </div>
          <div>
            <div className="flex items-center gap-2">
              {rank <= 3 && (
                <span className="text-xs font-bold text-primary bg-primary-soft border border-primary-disabled rounded-full px-2 py-0.5">
                  #{rank}
                </span>
              )}
              <h3 className="text-base font-bold text-ink">{team.teamName}</h3>
            </div>
            <p className="text-xs text-muted mt-0.5">
              {team.school} · {team.size}명 · {team.ageRange}세
            </p>
          </div>
        </div>
        <MatchScoreCard score={score} label={label} />
      </div>

      {/* 분위기 + 역할 */}
      <div className="flex flex-wrap items-center gap-2">
        <MoodChip mood={team.mood} selected />
        {team.members.slice(0, 4).map((m, i) => (
          <span key={i} className="text-base" title={m.role}>
            {ROLE_EMOJI[m.role]}
          </span>
        ))}
      </div>

      {/* 이유 */}
      <MatchReasonList reasons={reasons} />
    </div>
  );
}
```

- [ ] **Step 4: app/match/page.tsx 작성**

```tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/AppHeader";
import { Button } from "@/components/Button";
import { RecommendationTeamCard } from "@/components/RecommendationTeamCard";
import { loadTeam } from "@/lib/storage";
import { rankTeams } from "@/lib/matching";
import { mockTeams } from "@/data/mockTeams";
import { TeamProfile, MatchResult } from "@/types/matching";

export default function MatchPage() {
  const router = useRouter();
  const [myTeam, setMyTeam] = useState<TeamProfile | null>(null);
  const [results, setResults] = useState<MatchResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const team = loadTeam();
    if (team) {
      setMyTeam(team);
      setResults(rankTeams(team, mockTeams));
    }
    setLoading(false);
  }, []);

  if (loading) return null;

  if (!myTeam) {
    return (
      <>
        <AppHeader />
        <main className="py-20 px-4 text-center max-w-[480px] mx-auto">
          <p className="text-body mb-2">팀 프로필이 필요해요.</p>
          <p className="text-sm text-muted mb-6">
            성향 테스트를 완료하고 팀을 만들어야 추천을 볼 수 있어요.
          </p>
          <Button onClick={() => router.push("/")}>처음부터 시작하기</Button>
        </main>
      </>
    );
  }

  return (
    <>
      <AppHeader step={3} totalSteps={3} />
      <main className="py-10 px-4 bg-canvas-warm min-h-screen">
        <div className="max-w-[640px] mx-auto">
          <h1 className="text-2xl font-bold text-ink mb-1">추천 과팅 팀</h1>
          <p className="text-sm text-muted mb-8">
            <span className="font-semibold text-ink">{myTeam.teamName}</span>과 잘 어울릴 팀을 분위기·역할·조건 궁합으로 추천했어요.
          </p>
          <div className="flex flex-col gap-4">
            {results.map((result, i) => (
              <RecommendationTeamCard key={result.team.teamName} result={result} rank={i + 1} />
            ))}
          </div>
          <div className="mt-8">
            <Button
              variant="secondary"
              fullWidth
              onClick={() => router.push("/team/demo")}
            >
              우리 팀으로 돌아가기
            </Button>
          </div>
        </div>
      </main>
    </>
  );
}
```

- [ ] **Step 5: 브라우저에서 전체 플로우 확인**

```
/ → /test (10문항 완료) → /team/create (팀원 추가) → /team/demo → /match
```

Expected:
- `/match`에서 5개 팀이 점수 내림차순으로 표시됨
- 각 카드에 `%` 점수, 레이블, 이유 2~3개, 분위기 칩, 역할 이모지 표시
- 빈 localStorage 상태에서 `/match` 직접 접속 시 → 복구 화면 표시

- [ ] **Step 6: Commit**

```bash
cd .. && git add gwating-app/components/MatchScoreCard.tsx gwating-app/components/MatchReasonList.tsx gwating-app/components/RecommendationTeamCard.tsx gwating-app/app/match/
git commit -m "feat: add match page with ranked recommendations (score + reasons)"
```

---

## Task 14: Final Build & Polish

**Files:**
- Modify: `gwating-app/next.config.ts` (필요시)
- Modify: `gwating-app/app/globals.css` (모바일 확인)

- [ ] **Step 1: 프로덕션 빌드 확인**

```bash
cd gwating-app && npm run build
```

Expected: 오류 없이 완료. TypeScript 타입 에러 없음.

- [ ] **Step 2: 전체 테스트 통과 확인**

```bash
npm test
```

Expected: `Test Suites: 3 passed` / `Tests: 13 passed`

- [ ] **Step 3: 모바일 레이아웃 확인**

브라우저 DevTools에서 Mobile (375px) 설정:
- 홈: Hero CTA 버튼이 세로로 쌓임
- 테스트: 선택지가 전체 너비
- 팀 생성: 역할 버튼 2열 그리드
- 매칭: 카드가 1열

- [ ] **Step 4: Final Commit**

```bash
cd ..
git add gwating-app/
git commit -m "feat: complete gwating-app MVP (과팅 매칭 5-page demo for 부산대)"
```
