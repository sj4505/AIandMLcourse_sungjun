# 학원 관리 시스템 - 2단계: 인증 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Supabase Auth 기반 이메일/비밀번호 로그인 구현, /desk 라우트 role 기반 권한 분기 훅 제공, 데스크 헤더(이름/역할/로그아웃) 추가

**Architecture:** Server Action으로 signIn/signOut 처리. `lib/auth.ts`에서 서버 컴포넌트용 `getStaff()` 제공, `lib/hooks/use-staff.ts`에서 클라이언트 컴포넌트용 `useStaff()` 제공. desk layout에서 getStaff()를 호출해 role 정보를 children에 필요 시 전달.

**Tech Stack:** Next.js 14 Server Actions, `useFormState` + `useFormStatus` (react-dom), Supabase Auth, Tailwind CSS

---

## 파일 구조

```
academy-app/
├── app/
│   ├── login/
│   │   ├── page.tsx          - 로그인 폼 (Client Component, useFormState)
│   │   └── actions.ts        - signIn / signOut Server Actions
│   └── desk/
│       └── layout.tsx        - 데스크 공통 레이아웃 (헤더 + 로그아웃)
└── lib/
    ├── auth.ts               - 서버용 getStaff() 헬퍼
    └── hooks/
        └── use-staff.ts      - 클라이언트용 useStaff() 훅
```

---

### Task 1: signIn / signOut Server Actions

**Files:**
- Create: `academy-app/app/login/actions.ts`

- [ ] **Step 1: actions.ts 작성**

`academy-app/app/login/actions.ts`:
```typescript
'use server'

import { redirect } from 'next/navigation'
import { createClient } from '@/lib/supabase/server'

export async function signIn(
  _prevState: { error: string | null },
  formData: FormData
): Promise<{ error: string | null }> {
  const email = formData.get('email') as string
  const password = formData.get('password') as string

  const supabase = await createClient()
  const { error } = await supabase.auth.signInWithPassword({ email, password })

  if (error) {
    return { error: '이메일 또는 비밀번호가 올바르지 않습니다.' }
  }

  redirect('/desk')
}

export async function signOut() {
  const supabase = await createClient()
  await supabase.auth.signOut()
  redirect('/login')
}
```

- [ ] **Step 2: tsc 확인**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse/academy-app"
npx tsc --noEmit
```
Expected: 에러 없음

- [ ] **Step 3: 커밋**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse"
git add academy-app/app/login/actions.ts
git commit -m "feat: add signIn/signOut Server Actions"
```

---

### Task 2: 로그인 페이지 구현

**Files:**
- Modify: `academy-app/app/login/page.tsx` (placeholder → 실제 폼)

- [ ] **Step 1: login/page.tsx 전체 교체**

`academy-app/app/login/page.tsx`:
```typescript
'use client'

import { useFormState, useFormStatus } from 'react-dom'
import { signIn } from './actions'

function SubmitButton() {
  const { pending } = useFormStatus()
  return (
    <button
      type="submit"
      disabled={pending}
      className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
    >
      {pending ? '로그인 중...' : '로그인'}
    </button>
  )
}

export default function LoginPage() {
  const [state, formAction] = useFormState(signIn, { error: null })

  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="bg-white p-8 rounded-xl shadow-sm w-full max-w-sm border border-gray-100">
        <h1 className="text-2xl font-bold text-center mb-2 text-gray-900">
          학원 관리 시스템
        </h1>
        <p className="text-sm text-center text-gray-400 mb-8">선생님 전용</p>

        <form action={formAction} className="space-y-4">
          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              이메일
            </label>
            <input
              id="email"
              name="email"
              type="email"
              required
              autoComplete="email"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              비밀번호
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              autoComplete="current-password"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {state?.error && (
            <p className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">
              {state.error}
            </p>
          )}

          <SubmitButton />
        </form>
      </div>
    </main>
  )
}
```

- [ ] **Step 2: tsc 확인**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse/academy-app"
npx tsc --noEmit
```
Expected: 에러 없음

- [ ] **Step 3: 커밋**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse"
git add academy-app/app/login/page.tsx
git commit -m "feat: implement login page with email/password form"
```

---

### Task 3: 서버 auth 헬퍼 + 클라이언트 훅

**Files:**
- Create: `academy-app/lib/auth.ts`
- Create: `academy-app/lib/hooks/use-staff.ts`

- [ ] **Step 1: lib/auth.ts 작성 (서버 컴포넌트용)**

`academy-app/lib/auth.ts`:
```typescript
import { createClient } from '@/lib/supabase/server'
import { Staff } from '@/lib/types'

export async function getStaff(): Promise<Staff | null> {
  const supabase = await createClient()

  const {
    data: { user },
  } = await supabase.auth.getUser()
  if (!user) return null

  const { data } = await supabase
    .from('staff')
    .select('*')
    .eq('id', user.id)
    .single()

  return data ?? null
}
```

- [ ] **Step 2: lib/hooks/use-staff.ts 작성 (클라이언트 컴포넌트용)**

`academy-app/lib/hooks/use-staff.ts`:
```typescript
'use client'

import { useEffect, useState } from 'react'
import { createClient } from '@/lib/supabase/client'
import { Staff } from '@/lib/types'

export function useStaff() {
  const [staff, setStaff] = useState<Staff | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const supabase = createClient()

    async function fetchStaff() {
      const {
        data: { user },
      } = await supabase.auth.getUser()

      if (!user) {
        setLoading(false)
        return
      }

      const { data } = await supabase
        .from('staff')
        .select('*')
        .eq('id', user.id)
        .single()

      setStaff(data ?? null)
      setLoading(false)
    }

    fetchStaff()
  }, [])

  return { staff, loading }
}
```

- [ ] **Step 3: tsc 확인**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse/academy-app"
npx tsc --noEmit
```
Expected: 에러 없음

- [ ] **Step 4: 커밋**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse"
git add academy-app/lib/auth.ts academy-app/lib/hooks/
git commit -m "feat: add getStaff server helper and useStaff client hook"
```

---

### Task 4: 데스크 레이아웃 (헤더 + 로그아웃)

**Files:**
- Create: `academy-app/app/desk/layout.tsx`

- [ ] **Step 1: desk/layout.tsx 작성**

`academy-app/app/desk/layout.tsx`:
```typescript
import { redirect } from 'next/navigation'
import { getStaff } from '@/lib/auth'
import { signOut } from '@/app/login/actions'

export default async function DeskLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const staff = await getStaff()

  if (!staff) {
    redirect('/login')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 h-14 flex items-center justify-between sticky top-0 z-10">
        <span className="text-base font-semibold text-gray-800">학원 관리 시스템</span>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600">
            {staff.name}
            <span className="ml-1.5 text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-500">
              {staff.role === 'principal' ? '원장' : '선생님'}
            </span>
          </span>
          <form action={signOut}>
            <button
              type="submit"
              className="text-sm text-gray-400 hover:text-gray-600 transition-colors"
            >
              로그아웃
            </button>
          </form>
        </div>
      </header>
      <div>{children}</div>
    </div>
  )
}
```

- [ ] **Step 2: 빌드 확인**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse/academy-app"
npm run build
```
Expected: 빌드 성공

- [ ] **Step 3: 커밋**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse"
git add academy-app/app/desk/layout.tsx
git commit -m "feat: add desk layout with header and logout"
```

---

### Task 5: 초기 계정 생성 SQL + 동작 확인

**Files:**
- Create: `academy-app/supabase/seed-staff.sql` (참고용, git 포함)

이 Task는 **수동 실행이 필요**합니다. 코드 작성 후 사용자가 직접 Supabase 대시보드에서 실행.

- [ ] **Step 1: seed-staff.sql 작성**

`academy-app/supabase/seed-staff.sql`:
```sql
-- ============================================================
-- 초기 교직원 계정 생성 가이드
-- 순서: Supabase Auth에서 먼저 유저 생성 → 아래 SQL 실행
-- ============================================================

-- 1. Supabase 대시보드 → Authentication → Users → "Add user" 클릭
--    이메일/비밀번호 입력 후 "Create user" → 생성된 user의 UUID 복사

-- 2. 아래 SQL에서 UUID 교체 후 SQL Editor에서 실행

-- 원장 계정 예시
INSERT INTO public.staff (id, name, role)
VALUES (
  'PASTE-AUTH-USER-UUID-HERE',  -- 위에서 복사한 UUID
  '원장님',                      -- 실제 이름으로 교체
  'principal'
);

-- 선생님 계정 예시 (추가 계정 필요 시 반복)
-- INSERT INTO public.staff (id, name, role)
-- VALUES (
--   'PASTE-TEACHER-UUID-HERE',
--   '선생님 이름',
--   'teacher'
-- );
```

- [ ] **Step 2: Supabase에서 계정 생성 (수동)**

1. https://app.supabase.com → 프로젝트 선택
2. **Authentication** → **Users** → "Add user" 버튼
3. 이메일 + 비밀번호 입력 → Create user
4. 생성된 유저의 UUID 복사 (User ID 열)
5. **SQL Editor** → `seed-staff.sql` 내용에서 UUID 교체 후 실행
6. Table Editor → staff 테이블에 레코드 확인

- [ ] **Step 3: 로컬 서버에서 로그인 테스트**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse/academy-app"
npm run dev
```

브라우저에서:
1. http://localhost:3000 → `/kiosk` 리다이렉트 확인
2. http://localhost:3000/desk → `/login` 리다이렉트 확인
3. `/login`에서 이메일/비밀번호 입력 → `/desk` 리다이렉트 확인
4. 헤더에 이름 + 역할(원장/선생님) 표시 확인
5. 로그아웃 버튼 → `/login` 이동 확인

- [ ] **Step 4: 커밋**

```powershell
cd "C:/Users/ansun/OneDrive/바탕 화면/AIandMLcourse"
git add academy-app/supabase/seed-staff.sql
git commit -m "docs: add initial staff account creation guide"
```

---

## Self-Review

### 1. Spec 커버리지

| 스펙 항목 | Task |
|---|---|
| Supabase Auth 기반 로그인 | Task 1, 2 |
| /desk 미인증 → /login 리다이렉트 | 1단계 middleware (완료) + Task 4 이중 보호 |
| role 기반 권한 분기 훅 | Task 3 (`getStaff`, `useStaff`) |
| 로그아웃 | Task 1 (`signOut`), Task 4 (버튼 UI) |
| 원장/선생님 계정 생성 방법 | Task 5 |

모든 스펙 항목 커버됨.

### 2. Placeholder 스캔

- 모든 코드 스텝에 실제 코드 포함 ✓
- "TBD" 없음 ✓
- Task 5 Step 2는 수동 단계이며 명확한 지시사항 포함 ✓

### 3. 타입 일관성

- `getStaff()` 반환 타입 `Staff | null` → `useStaff()`의 `staff: Staff | null`과 일치 ✓
- `signIn` 시그니처 `(_prevState, formData) => { error: string | null }` → `useFormState(signIn, { error: null })`과 일치 ✓
- `staff.role === 'principal'` → `lib/types.ts`의 `Role = 'principal' | 'teacher'`와 일치 ✓
