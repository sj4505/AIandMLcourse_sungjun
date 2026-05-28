# 학원 관리 시스템 - 1단계: 프로젝트 셋업 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Next.js 14 + TypeScript + Tailwind + Supabase 기반 학원 관리 시스템의 기초 인프라 구축 (DB 스키마, 클라이언트 설정, 라우트 보호 포함)

**Architecture:** Next.js 14 App Router 기반. Supabase를 BaaS로 활용 (Auth + PostgreSQL + Realtime). Server Components + Server Actions 패턴. 키오스크(/kiosk)는 anon 접근 허용, 데스크(/desk)는 인증 필요. RLS로 테이블별 접근 제어.

**Tech Stack:** Next.js 14, TypeScript, Tailwind CSS v3, @supabase/ssr, @supabase/supabase-js

---

## 파일 구조

```
academy-app/
├── app/
│   ├── layout.tsx              - 루트 레이아웃 (한국어, 메타데이터)
│   ├── page.tsx                - 루트 → /kiosk 리다이렉트
│   ├── kiosk/
│   │   └── page.tsx            - 키오스크 placeholder (3단계)
│   ├── login/
│   │   └── page.tsx            - 로그인 placeholder (2단계)
│   └── desk/
│       └── page.tsx            - 데스크 placeholder (4단계)
├── lib/
│   ├── supabase/
│   │   ├── client.ts           - 브라우저 Supabase 클라이언트
│   │   └── server.ts           - 서버 Supabase 클라이언트
│   └── types.ts                - 전체 DB 타입 정의
├── middleware.ts                - /desk 라우트 보호 (미인증 → /login)
├── supabase/
│   └── schema.sql              - 전체 DB 스키마 + RLS + periods 시드
├── .env.local                  - 환경변수 (git 미포함)
└── .env.local.example          - 환경변수 템플릿
```

---

## 사전 준비 (실행 전 필수)

Supabase 프로젝트가 없다면:
1. https://app.supabase.com → New Project 생성
2. Project Settings → API에서 `Project URL`과 `anon public` 키 복사
3. 아래 Task 2에서 사용

---

### Task 1: Next.js 프로젝트 생성 + 패키지 설치

**Files:**
- Create: `academy-app/` (프로젝트 루트)

- [ ] **Step 1: Next.js 14 프로젝트 생성**

현재 작업 디렉토리(`AIandMLcourse/`)에서 실행:
```bash
npx create-next-app@14 academy-app --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*"
```
프롬프트가 나오면 모두 기본값(Enter)으로 진행.

- [ ] **Step 2: Supabase 패키지 설치**

```bash
cd academy-app
npm install @supabase/supabase-js @supabase/ssr
```

- [ ] **Step 3: 빌드 확인**

```bash
npm run build
```
Expected: `✓ Compiled successfully` (에러 없음)

- [ ] **Step 4: 커밋**

```bash
cd ..
git add academy-app/
git commit -m "feat: initialize Next.js 14 academy app"
```

---

### Task 2: 환경변수 + Supabase 클라이언트 설정

**Files:**
- Create: `academy-app/.env.local`
- Create: `academy-app/.env.local.example`
- Create: `academy-app/lib/supabase/client.ts`
- Create: `academy-app/lib/supabase/server.ts`

- [ ] **Step 1: 환경변수 파일 작성**

`academy-app/.env.local` (실제 값 입력):
```
NEXT_PUBLIC_SUPABASE_URL=https://your-project-ref.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
```

`academy-app/.env.local.example` (템플릿, git 포함):
```
NEXT_PUBLIC_SUPABASE_URL=https://your-project-ref.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
```

- [ ] **Step 2: `lib/supabase/` 디렉토리 생성 확인**

```bash
mkdir -p academy-app/lib/supabase
```

- [ ] **Step 3: 브라우저 클라이언트 작성**

`academy-app/lib/supabase/client.ts`:
```typescript
import { createBrowserClient } from '@supabase/ssr'

export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  )
}
```

- [ ] **Step 4: 서버 클라이언트 작성**

`academy-app/lib/supabase/server.ts`:
```typescript
import { createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

export async function createClient() {
  const cookieStore = await cookies()

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll()
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            )
          } catch {}
        },
      },
    }
  )
}
```

- [ ] **Step 5: 타입 체크**

```bash
cd academy-app && npx tsc --noEmit
```
Expected: 에러 없음

- [ ] **Step 6: 커밋**

```bash
cd ..
git add academy-app/lib/ academy-app/.env.local.example
git commit -m "feat: add Supabase client configuration (browser + server)"
```

---

### Task 3: TypeScript 타입 정의

**Files:**
- Create: `academy-app/lib/types.ts`

- [ ] **Step 1: 타입 파일 작성**

`academy-app/lib/types.ts`:
```typescript
export type Role = 'principal' | 'teacher'
export type OutingType = 'toilet' | 'academy' | 'meal'
export type DisruptionType = 'distraction' | 'drowsiness'
export type ScheduleStatus = 'pending_teacher' | 'approved' | 'rejected'
export type ReportStatus = 'draft' | 'approved'

export interface Staff {
  id: string
  name: string
  role: Role
  created_at: string
}

export interface Student {
  id: string
  name: string
  phone: string
  parent_phone: string
  memo: string | null
  created_at: string
}

export interface Period {
  id: number
  name: string
  start_time: string  // 'HH:MM' 형식
  end_time: string    // 'HH:MM' 형식
}

export interface Schedule {
  id: string
  student_id: string
  day_of_week: number  // 0=일, 1=월, 2=화, 3=수, 4=목, 5=금, 6=토
  expected_in: string  // 'HH:MM' 형식
  expected_out: string // 'HH:MM' 형식
  title: string | null
  status: ScheduleStatus
  created_at: string
}

export interface Attendance {
  id: string
  student_id: string
  date: string           // 'YYYY-MM-DD' 형식
  check_in_at: string | null   // ISO 8601 타임스탬프
  check_out_at: string | null
  is_late: boolean
  late_minutes: number
  created_at: string
}

export interface Outing {
  id: string
  attendance_id: string
  student_id: string
  out_at: string         // ISO 8601 타임스탬프
  back_at: string | null
  outing_type: OutingType
  created_at: string
}

export interface Disruption {
  id: string
  student_id: string
  attendance_id: string
  period_id: number
  type: DisruptionType
  recorded_by: string | null
  created_at: string
}

export interface TaskCheck {
  id: string
  student_id: string
  period_id: number
  date: string           // 'YYYY-MM-DD' 형식
  content: string
  is_done: boolean
  created_at: string
}

export interface Report {
  id: string
  student_id: string
  month: string          // 'YYYY-MM' 형식
  attendance_rate: number | null
  late_count: number
  avg_late_minutes: number | null
  distraction_count: number
  drowsiness_count: number
  planner_achievement: number | null
  status: ReportStatus
  created_at: string
}

export interface Announcement {
  id: string
  title: string
  content: string
  is_active: boolean
  created_by: string | null
  created_at: string
}
```

- [ ] **Step 2: 타입 체크**

```bash
cd academy-app && npx tsc --noEmit
```
Expected: 에러 없음

- [ ] **Step 3: 커밋**

```bash
cd ..
git add academy-app/lib/types.ts
git commit -m "feat: add TypeScript type definitions for all DB entities"
```

---

### Task 4: DB 스키마 SQL 작성 + Supabase 적용

**Files:**
- Create: `academy-app/supabase/schema.sql`

- [ ] **Step 1: `supabase/` 디렉토리 생성**

```bash
mkdir -p academy-app/supabase
```

- [ ] **Step 2: 전체 스키마 SQL 작성**

`academy-app/supabase/schema.sql`:
```sql
-- ============================================================
-- 학원 관리 시스템 DB 스키마
-- Supabase SQL Editor에서 실행 (전체 복붙 후 Run)
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ========================
-- 테이블 생성
-- ========================

-- 교직원 (Supabase Auth와 1:1 연동)
CREATE TABLE IF NOT EXISTS public.staff (
  id         UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  name       TEXT NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('principal', 'teacher')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 학생
CREATE TABLE IF NOT EXISTS public.students (
  id           UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  name         TEXT        NOT NULL,
  phone        TEXT        NOT NULL UNIQUE,
  parent_phone TEXT        NOT NULL,
  memo         TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 교시 (1교시~4교시, 시드 데이터로 채움)
CREATE TABLE IF NOT EXISTS public.periods (
  id         SERIAL      PRIMARY KEY,
  name       TEXT        NOT NULL,
  start_time TIME        NOT NULL,
  end_time   TIME        NOT NULL
);

-- 정기 일정 (학생이 요청 → 선생님 승인)
CREATE TABLE IF NOT EXISTS public.schedules (
  id           UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id   UUID        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  day_of_week  SMALLINT    NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
  expected_in  TIME        NOT NULL,
  expected_out TIME        NOT NULL,
  title        TEXT,
  status       TEXT        NOT NULL DEFAULT 'pending_teacher'
                 CHECK (status IN ('pending_teacher', 'approved', 'rejected')),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 출결 (날짜별 1건)
CREATE TABLE IF NOT EXISTS public.attendance (
  id           UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id   UUID        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  date         DATE        NOT NULL DEFAULT CURRENT_DATE,
  check_in_at  TIMESTAMPTZ,
  check_out_at TIMESTAMPTZ,
  is_late      BOOLEAN     NOT NULL DEFAULT FALSE,
  late_minutes INT         NOT NULL DEFAULT 0,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (student_id, date)
);

-- 외출
CREATE TABLE IF NOT EXISTS public.outings (
  id           UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  attendance_id UUID       NOT NULL REFERENCES public.attendance(id) ON DELETE CASCADE,
  student_id   UUID        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  out_at       TIMESTAMPTZ NOT NULL,
  back_at      TIMESTAMPTZ,
  outing_type  TEXT        NOT NULL CHECK (outing_type IN ('toilet', 'academy', 'meal')),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 딴짓 / 졸음 기록
CREATE TABLE IF NOT EXISTS public.disruptions (
  id            UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id    UUID        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  attendance_id UUID        NOT NULL REFERENCES public.attendance(id) ON DELETE CASCADE,
  period_id     INT         NOT NULL REFERENCES public.periods(id),
  type          TEXT        NOT NULL CHECK (type IN ('distraction', 'drowsiness')),
  recorded_by   UUID        REFERENCES public.staff(id),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 플래너 체크
CREATE TABLE IF NOT EXISTS public.task_checks (
  id         UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id UUID        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  period_id  INT         NOT NULL REFERENCES public.periods(id),
  date       DATE        NOT NULL DEFAULT CURRENT_DATE,
  content    TEXT        NOT NULL,
  is_done    BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 리포트
CREATE TABLE IF NOT EXISTS public.reports (
  id                  UUID          NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  student_id          UUID          NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  month               TEXT          NOT NULL,  -- 'YYYY-MM' 형식
  attendance_rate     NUMERIC(5,2),
  late_count          INT           NOT NULL DEFAULT 0,
  avg_late_minutes    NUMERIC(5,2),
  distraction_count   INT           NOT NULL DEFAULT 0,
  drowsiness_count    INT           NOT NULL DEFAULT 0,
  planner_achievement NUMERIC(5,2),
  status              TEXT          NOT NULL DEFAULT 'draft'
                        CHECK (status IN ('draft', 'approved')),
  created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  UNIQUE (student_id, month)
);

-- 공지사항
CREATE TABLE IF NOT EXISTS public.announcements (
  id         UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  title      TEXT        NOT NULL,
  content    TEXT        NOT NULL,
  is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
  created_by UUID        REFERENCES public.staff(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ========================
-- Row Level Security
-- ========================

ALTER TABLE public.staff         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.students      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.periods       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.schedules     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.attendance    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.outings       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.disruptions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.task_checks   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reports       ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.announcements ENABLE ROW LEVEL SECURITY;

-- staff: 인증된 교직원만 읽기/등록
CREATE POLICY "staff_select" ON public.staff
  FOR SELECT TO authenticated USING (true);
CREATE POLICY "staff_insert" ON public.staff
  FOR INSERT TO authenticated WITH CHECK (true);

-- students: 인증된 교직원만 전체 권한
CREATE POLICY "students_all_auth" ON public.students
  FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- periods: 누구나 읽기 (키오스크 anon 필요)
CREATE POLICY "periods_select_all" ON public.periods
  FOR SELECT USING (true);

-- schedules: 인증된 교직원만
CREATE POLICY "schedules_all_auth" ON public.schedules
  FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- attendance: 키오스크(anon)가 입/퇴실 기록 → 읽기/쓰기/수정 anon 허용
--             삭제는 인증만
CREATE POLICY "attendance_select_all"  ON public.attendance FOR SELECT            USING (true);
CREATE POLICY "attendance_insert_all"  ON public.attendance FOR INSERT            WITH CHECK (true);
CREATE POLICY "attendance_update_all"  ON public.attendance FOR UPDATE            USING (true);
CREATE POLICY "attendance_delete_auth" ON public.attendance FOR DELETE TO authenticated USING (true);

-- outings: 키오스크(anon)가 외출/복귀 기록 → 전체 anon 허용
CREATE POLICY "outings_all" ON public.outings
  FOR ALL USING (true) WITH CHECK (true);

-- disruptions: 인증된 교직원만
CREATE POLICY "disruptions_all_auth" ON public.disruptions
  FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- task_checks: 인증된 교직원만
CREATE POLICY "task_checks_all_auth" ON public.task_checks
  FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- reports: 인증된 교직원만
CREATE POLICY "reports_all_auth" ON public.reports
  FOR ALL TO authenticated USING (true) WITH CHECK (true);

-- announcements: 읽기는 누구나, 쓰기/수정/삭제는 인증
CREATE POLICY "announcements_select_all"  ON public.announcements FOR SELECT            USING (true);
CREATE POLICY "announcements_write_auth"  ON public.announcements FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "announcements_update_auth" ON public.announcements FOR UPDATE TO authenticated USING (true);
CREATE POLICY "announcements_delete_auth" ON public.announcements FOR DELETE TO authenticated USING (true);

-- ========================
-- 교시 시드 데이터
-- ========================

INSERT INTO public.periods (name, start_time, end_time) VALUES
  ('1교시', '15:00', '16:30'),
  ('2교시', '16:30', '18:00'),
  ('3교시', '18:00', '19:30'),
  ('4교시', '19:30', '21:00')
ON CONFLICT DO NOTHING;
```

- [ ] **Step 3: Supabase SQL Editor에서 스키마 실행**

1. https://app.supabase.com → 해당 프로젝트 선택
2. 좌측 메뉴 → **SQL Editor** → New query
3. `academy-app/supabase/schema.sql` 전체 내용 붙여넣기
4. **Run** 클릭

Expected: "Success. No rows returned" 메시지

- [ ] **Step 4: 테이블 생성 + 시드 데이터 확인**

Supabase 대시보드 → **Table Editor**:
- 좌측 목록에 10개 테이블 확인: staff, students, periods, schedules, attendance, outings, disruptions, task_checks, reports, announcements
- periods 테이블 클릭 → 4개 행 (1교시~4교시) 확인

- [ ] **Step 5: 커밋**

```bash
cd ..
git add academy-app/supabase/schema.sql
git commit -m "feat: add complete DB schema with RLS policies and periods seed data"
```

---

### Task 5: 미들웨어 + placeholder 페이지 설정

**Files:**
- Create: `academy-app/middleware.ts`
- Modify: `academy-app/app/layout.tsx`
- Modify: `academy-app/app/page.tsx`
- Create: `academy-app/app/kiosk/page.tsx`
- Create: `academy-app/app/login/page.tsx`
- Create: `academy-app/app/desk/page.tsx`

- [ ] **Step 1: 라우트 보호 미들웨어 작성**

`academy-app/middleware.ts`:
```typescript
import { createServerClient } from '@supabase/ssr'
import { NextResponse, type NextRequest } from 'next/server'

export async function middleware(request: NextRequest) {
  let supabaseResponse = NextResponse.next({ request })

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return request.cookies.getAll()
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value }) =>
            request.cookies.set(name, value)
          )
          supabaseResponse = NextResponse.next({ request })
          cookiesToSet.forEach(({ name, value, options }) =>
            supabaseResponse.cookies.set(name, value, options)
          )
        },
      },
    }
  )

  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user && request.nextUrl.pathname.startsWith('/desk')) {
    const url = request.nextUrl.clone()
    url.pathname = '/login'
    return NextResponse.redirect(url)
  }

  return supabaseResponse
}

export const config = {
  matcher: ['/desk/:path*'],
}
```

- [ ] **Step 2: 루트 레이아웃 한국어 설정**

`academy-app/app/layout.tsx` 의 `<html lang="en">` 을 `<html lang="ko">` 로 수정하고
`metadata` 를 아래로 교체:
```typescript
export const metadata: Metadata = {
  title: '학원 관리 시스템',
  description: '학원 출결 및 학습 관리 시스템',
}
```

- [ ] **Step 3: 루트 페이지 → /kiosk 리다이렉트**

`academy-app/app/page.tsx` 전체를:
```typescript
import { redirect } from 'next/navigation'

export default function RootPage() {
  redirect('/kiosk')
}
```

- [ ] **Step 4: /kiosk placeholder 생성**

`academy-app/app/kiosk/page.tsx`:
```typescript
export default function KioskPage() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-50">
      <p className="text-2xl text-gray-400">키오스크 페이지 (3단계 구현 예정)</p>
    </main>
  )
}
```

- [ ] **Step 5: /login placeholder 생성**

`academy-app/app/login/page.tsx`:
```typescript
export default function LoginPage() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-50">
      <p className="text-2xl text-gray-400">로그인 페이지 (2단계 구현 예정)</p>
    </main>
  )
}
```

- [ ] **Step 6: /desk placeholder 생성**

`academy-app/app/desk/page.tsx`:
```typescript
export default function DeskPage() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-50">
      <p className="text-2xl text-gray-400">데스크 페이지 (4단계 구현 예정)</p>
    </main>
  )
}
```

- [ ] **Step 7: 빌드 + 동작 확인**

```bash
cd academy-app && npm run build
```
Expected: 빌드 성공

```bash
npm run dev
```
브라우저에서 http://localhost:3000 접속 → `/kiosk`로 리다이렉트 확인
http://localhost:3000/desk 접속 → `/login`으로 리다이렉트 확인 (미인증 상태)

- [ ] **Step 8: 커밋**

```bash
cd ..
git add academy-app/middleware.ts academy-app/app/
git commit -m "feat: add route protection middleware and placeholder pages"
```

---

## Self-Review

### 1. Spec 커버리지

| 스펙 항목 | 커버 Task |
|---|---|
| Next.js 14 + TypeScript + Tailwind 초기화 | Task 1 |
| Supabase 초기화 | Task 1 |
| 환경변수 구성 (.env.local) | Task 2 |
| Supabase 클라이언트 설정 | Task 2 |
| DB 스키마 전체 생성 SQL | Task 4 |
| periods 테이블 시드 데이터 | Task 4 |
| /desk 미인증 → /login 리다이렉트 | Task 5 |

1단계 스펙 항목 모두 커버됨.

### 2. Placeholder 스캔

- 모든 코드 스텝에 실제 코드 포함됨 ✓
- "TBD" 또는 "TODO" 없음 ✓
- placeholder 페이지는 이후 단계에서 교체 예정이며, 빌드가 통과되는 실제 코드임 ✓

### 3. 타입 일관성

- `Staff`, `Student`, `Period` 등 types.ts 타입명이 schema.sql 테이블명과 일치 ✓
- `OutingType` 값 (`'toilet' | 'academy' | 'meal'`)이 schema.sql CHECK 제약과 일치 ✓
- `ScheduleStatus`, `DisruptionType`, `ReportStatus` 모두 schema.sql CHECK 제약과 일치 ✓

---

## 다음 단계

이 계획이 완료되면 **2단계 (인증)** 계획을 별도로 작성:
- Supabase Auth 기반 이메일/비밀번호 로그인
- /login 페이지 구현
- role 기반 권한 훅 (`useStaffRole`)
- 원장/선생님 계정 초기 생성 방법
