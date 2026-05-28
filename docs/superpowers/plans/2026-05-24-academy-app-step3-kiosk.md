# 학원 관리 시스템 - 3단계: 키오스크 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 전화번호 키패드로 학생 조회 → 상태 판단(미입실/입실중/외출중/퇴실) → 입실·외출·복귀·퇴실 처리하는 키오스크 페이지 구현

**Architecture:** `/kiosk` 는 인증 불필요(anon). 숫자 키패드 입력 → Server Action으로 상태 조회 → 상태별 액션 버튼 → Server Action으로 DB 업데이트. 3화면(키패드 → 액션패널 → 결과) 전환은 Client Component 상태로 관리. 지각 판단은 schedules.expected_in 기준.

**Tech Stack:** Next.js 14 Client Component + Server Actions, Supabase anon client, Tailwind CSS

---

## 파일 구조

```
academy-app/app/kiosk/
├── page.tsx                  - 메인 키오스크 페이지 (화면 전환 상태 관리)
├── actions.ts                - Server Actions (getStudentState, checkIn, startOuting, endOuting, checkOut)
└── components/
    ├── Keypad.tsx            - 숫자 키패드 UI
    └── ActionPanel.tsx       - 학생 상태 표시 + 액션 버튼
```

---

### Task 1: Server Actions

**Files:**
- Create: `academy-app/app/kiosk/actions.ts`

- [ ] **Step 1: actions.ts 작성**

`academy-app/app/kiosk/actions.ts`:
```typescript
'use server'

import { createClient } from '@/lib/supabase/server'
import { OutingType } from '@/lib/types'

function formatDate(date: Date): string {
  return date.toISOString().split('T')[0]
}

function timeToMinutes(time: string): number {
  const [h, m] = time.split(':').map(Number)
  return h * 60 + m
}

export type StudentState =
  | { status: 'not_found' }
  | { status: 'not_arrived'; studentId: string; studentName: string }
  | { status: 'checked_in'; studentId: string; studentName: string; attendanceId: string }
  | { status: 'on_outing'; studentId: string; studentName: string; attendanceId: string; outingId: string; outingType: OutingType; outAt: string }
  | { status: 'checked_out'; studentName: string }

export async function getStudentState(phone: string): Promise<StudentState> {
  const supabase = await createClient()
  const today = formatDate(new Date())

  const { data: student } = await supabase
    .from('students')
    .select('id, name')
    .eq('phone', phone)
    .single()

  if (!student) return { status: 'not_found' }

  const { data: attendance } = await supabase
    .from('attendance')
    .select('id, check_in_at, check_out_at')
    .eq('student_id', student.id)
    .eq('date', today)
    .single()

  if (!attendance || !attendance.check_in_at) {
    return { status: 'not_arrived', studentId: student.id, studentName: student.name }
  }

  if (attendance.check_out_at) {
    return { status: 'checked_out', studentName: student.name }
  }

  const { data: activeOuting } = await supabase
    .from('outings')
    .select('id, outing_type, out_at')
    .eq('attendance_id', attendance.id)
    .is('back_at', null)
    .order('out_at', { ascending: false })
    .limit(1)
    .single()

  if (activeOuting) {
    return {
      status: 'on_outing',
      studentId: student.id,
      studentName: student.name,
      attendanceId: attendance.id,
      outingId: activeOuting.id,
      outingType: activeOuting.outing_type as OutingType,
      outAt: activeOuting.out_at,
    }
  }

  return {
    status: 'checked_in',
    studentId: student.id,
    studentName: student.name,
    attendanceId: attendance.id,
  }
}

export async function checkIn(phone: string): Promise<{
  success: boolean
  studentName: string
  isLate: boolean
  lateMinutes: number
  error?: string
}> {
  const supabase = await createClient()
  const now = new Date()
  const today = formatDate(now)
  const dayOfWeek = now.getDay()

  const { data: student } = await supabase
    .from('students')
    .select('id, name')
    .eq('phone', phone)
    .single()

  if (!student) {
    return { success: false, studentName: '', isLate: false, lateMinutes: 0, error: '학생 없음' }
  }

  const { data: schedule } = await supabase
    .from('schedules')
    .select('expected_in')
    .eq('student_id', student.id)
    .eq('day_of_week', dayOfWeek)
    .eq('status', 'approved')
    .order('expected_in', { ascending: true })
    .limit(1)
    .single()

  let isLate = false
  let lateMinutes = 0

  if (schedule) {
    const nowMinutes = now.getHours() * 60 + now.getMinutes()
    const expectedMinutes = timeToMinutes(schedule.expected_in)
    if (nowMinutes > expectedMinutes) {
      isLate = true
      lateMinutes = nowMinutes - expectedMinutes
    }
  }

  const { error } = await supabase.from('attendance').upsert(
    {
      student_id: student.id,
      date: today,
      check_in_at: now.toISOString(),
      is_late: isLate,
      late_minutes: lateMinutes,
    },
    { onConflict: 'student_id,date' }
  )

  if (error) {
    return { success: false, studentName: student.name, isLate: false, lateMinutes: 0, error: error.message }
  }

  return { success: true, studentName: student.name, isLate, lateMinutes }
}

export async function startOuting(
  attendanceId: string,
  studentId: string,
  outingType: OutingType
): Promise<{ success: boolean; error?: string }> {
  const supabase = await createClient()

  const { error } = await supabase.from('outings').insert({
    attendance_id: attendanceId,
    student_id: studentId,
    out_at: new Date().toISOString(),
    outing_type: outingType,
  })

  if (error) return { success: false, error: error.message }
  return { success: true }
}

export async function endOuting(outingId: string): Promise<{ success: boolean; error?: string }> {
  const supabase = await createClient()

  const { error } = await supabase
    .from('outings')
    .update({ back_at: new Date().toISOString() })
    .eq('id', outingId)

  if (error) return { success: false, error: error.message }
  return { success: true }
}

export async function checkOut(attendanceId: string): Promise<{ success: boolean; error?: string }> {
  const supabase = await createClient()

  const { error } = await supabase
    .from('attendance')
    .update({ check_out_at: new Date().toISOString() })
    .eq('id', attendanceId)

  if (error) return { success: false, error: error.message }
  return { success: true }
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
git add academy-app/app/kiosk/actions.ts
git commit -m "feat: add kiosk Server Actions (state query + check-in/out/outing)"
```

---

### Task 2: Keypad 컴포넌트

**Files:**
- Create: `academy-app/app/kiosk/components/Keypad.tsx`

- [ ] **Step 1: Keypad.tsx 작성**

`academy-app/app/kiosk/components/Keypad.tsx`:
```typescript
'use client'

interface KeypadProps {
  value: string
  onChange: (value: string) => void
  onConfirm: () => void
  disabled?: boolean
}

export function Keypad({ value, onChange, onConfirm, disabled }: KeypadProps) {
  const formatted = value.replace(
    /(\d{3})(\d{0,4})(\d{0,4})/,
    (_, a, b, c) => (c ? `${a}-${b}-${c}` : b ? `${a}-${b}` : a)
  )

  function press(key: string) {
    if (disabled) return
    if (key === 'del') {
      onChange(value.slice(0, -1))
    } else if (key === 'confirm') {
      if (value.length >= 10) onConfirm()
    } else if (value.length < 11) {
      onChange(value + key)
    }
  }

  const keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'del', '0', 'confirm']

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="w-72 h-16 bg-gray-100 rounded-xl flex items-center justify-center">
        {formatted ? (
          <span className="text-3xl font-mono tracking-widest text-gray-800">{formatted}</span>
        ) : (
          <span className="text-gray-400 text-lg">전화번호를 입력하세요</span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        {keys.map((key) => (
          <button
            key={key}
            onClick={() => press(key)}
            disabled={disabled || (key === 'confirm' && value.length < 10)}
            className={[
              'w-24 h-20 rounded-2xl text-2xl font-semibold transition-all active:scale-95 select-none',
              key === 'confirm'
                ? 'bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-40'
                : key === 'del'
                ? 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                : 'bg-white border-2 border-gray-200 text-gray-800 hover:bg-gray-50 shadow-sm',
            ].join(' ')}
          >
            {key === 'del' ? '⌫' : key === 'confirm' ? '확인' : key}
          </button>
        ))}
      </div>
    </div>
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
git add academy-app/app/kiosk/components/Keypad.tsx
git commit -m "feat: add Keypad component for kiosk phone number input"
```

---

### Task 3: ActionPanel 컴포넌트

**Files:**
- Create: `academy-app/app/kiosk/components/ActionPanel.tsx`

- [ ] **Step 1: ActionPanel.tsx 작성**

`academy-app/app/kiosk/components/ActionPanel.tsx`:
```typescript
'use client'

import { StudentState } from '../actions'
import { OutingType } from '@/lib/types'

interface ActionPanelProps {
  state: StudentState
  onCheckIn: () => void
  onStartOuting: (type: OutingType) => void
  onEndOuting: () => void
  onCheckOut: () => void
  onBack: () => void
  loading: boolean
}

const OUTING_LABELS: Record<OutingType, string> = {
  toilet: '화장실',
  academy: '학원 외출',
  meal: '식사',
}

function getElapsedMinutes(outAt: string): number {
  return Math.floor((Date.now() - new Date(outAt).getTime()) / 60000)
}

function getStatusLabel(state: StudentState): string {
  switch (state.status) {
    case 'not_arrived': return '미입실'
    case 'checked_in': return '입실 중'
    case 'on_outing': return `외출 중 (${OUTING_LABELS[state.outingType]})`
    case 'checked_out': return '퇴실 완료'
    default: return ''
  }
}

function ActionButton({
  children,
  onClick,
  disabled,
  color,
}: {
  children: React.ReactNode
  onClick: () => void
  disabled: boolean
  color: 'blue' | 'green' | 'red' | 'gray'
}) {
  const colors: Record<string, string> = {
    blue:  'bg-blue-600 text-white hover:bg-blue-700',
    green: 'bg-green-600 text-white hover:bg-green-700',
    red:   'bg-red-500 text-white hover:bg-red-600',
    gray:  'bg-gray-100 text-gray-800 hover:bg-gray-200',
  }
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full py-4 rounded-xl text-xl font-semibold transition-all active:scale-95 disabled:opacity-50 ${colors[color]}`}
    >
      {children}
    </button>
  )
}

export function ActionPanel({
  state,
  onCheckIn,
  onStartOuting,
  onEndOuting,
  onCheckOut,
  onBack,
  loading,
}: ActionPanelProps) {
  if (state.status === 'not_found') {
    return (
      <div className="flex flex-col items-center gap-6 text-center">
        <p className="text-2xl text-red-500 font-medium">등록되지 않은 번호입니다</p>
        <button
          onClick={onBack}
          className="px-8 py-3 bg-gray-200 rounded-xl text-gray-700 font-medium hover:bg-gray-300"
        >
          다시 입력
        </button>
      </div>
    )
  }

  const name = 'studentName' in state ? state.studentName : ''

  return (
    <div className="flex flex-col items-center gap-6 text-center">
      <div>
        <p className="text-4xl font-bold text-gray-900">{name}</p>
        <p className="text-lg text-gray-400 mt-1">{getStatusLabel(state)}</p>
      </div>

      <div className="flex flex-col gap-3 w-72">
        {state.status === 'not_arrived' && (
          <ActionButton onClick={onCheckIn} disabled={loading} color="blue">
            입실
          </ActionButton>
        )}

        {state.status === 'checked_in' && (
          <>
            <ActionButton onClick={() => onStartOuting('toilet')} disabled={loading} color="gray">
              화장실 외출
            </ActionButton>
            <ActionButton onClick={() => onStartOuting('academy')} disabled={loading} color="gray">
              학원 외출
            </ActionButton>
            <ActionButton onClick={() => onStartOuting('meal')} disabled={loading} color="gray">
              식사
            </ActionButton>
            <ActionButton onClick={onCheckOut} disabled={loading} color="red">
              퇴실
            </ActionButton>
          </>
        )}

        {state.status === 'on_outing' && (
          <>
            <p className="text-sm text-gray-400">
              {OUTING_LABELS[state.outingType]} 중 · {getElapsedMinutes(state.outAt)}분 경과
            </p>
            <ActionButton onClick={onEndOuting} disabled={loading} color="green">
              복귀
            </ActionButton>
          </>
        )}

        {state.status === 'checked_out' && (
          <p className="text-lg text-gray-500">오늘 퇴실 완료</p>
        )}
      </div>

      <button
        onClick={onBack}
        className="text-sm text-gray-400 hover:text-gray-600 underline mt-2"
      >
        취소
      </button>
    </div>
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
git add academy-app/app/kiosk/components/ActionPanel.tsx
git commit -m "feat: add ActionPanel component for kiosk student actions"
```

---

### Task 4: 키오스크 메인 페이지

**Files:**
- Modify: `academy-app/app/kiosk/page.tsx` (placeholder 교체)

- [ ] **Step 1: page.tsx 전체 교체**

`academy-app/app/kiosk/page.tsx`:
```typescript
'use client'

import { useState } from 'react'
import { Keypad } from './components/Keypad'
import { ActionPanel } from './components/ActionPanel'
import {
  getStudentState,
  checkIn,
  startOuting,
  endOuting,
  checkOut,
  StudentState,
} from './actions'
import { OutingType } from '@/lib/types'

type Screen = 'keypad' | 'action' | 'result'

export default function KioskPage() {
  const [phone, setPhone] = useState('')
  const [screen, setScreen] = useState<Screen>('keypad')
  const [studentState, setStudentState] = useState<StudentState | null>(null)
  const [resultMessage, setResultMessage] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleConfirm() {
    setLoading(true)
    const state = await getStudentState(phone)
    setStudentState(state)
    setScreen('action')
    setLoading(false)
  }

  function showResult(message: string) {
    setResultMessage(message)
    setScreen('result')
    setTimeout(() => {
      setPhone('')
      setStudentState(null)
      setScreen('keypad')
    }, 3000)
  }

  function handleBack() {
    setPhone('')
    setStudentState(null)
    setScreen('keypad')
  }

  async function handleCheckIn() {
    setLoading(true)
    const result = await checkIn(phone)
    if (result.success) {
      const msg = result.isLate
        ? `${result.studentName}님 입실\n(${result.lateMinutes}분 지각)`
        : `${result.studentName}님\n입실 완료`
      showResult(msg)
    }
    setLoading(false)
  }

  async function handleStartOuting(type: OutingType) {
    if (studentState?.status !== 'checked_in') return
    setLoading(true)
    const result = await startOuting(studentState.attendanceId, studentState.studentId, type)
    if (result.success) {
      const labels: Record<OutingType, string> = { toilet: '화장실', academy: '학원 외출', meal: '식사' }
      showResult(`${studentState.studentName}님\n${labels[type]} 외출`)
    }
    setLoading(false)
  }

  async function handleEndOuting() {
    if (studentState?.status !== 'on_outing') return
    setLoading(true)
    const result = await endOuting(studentState.outingId)
    if (result.success) showResult(`${studentState.studentName}님\n복귀 완료`)
    setLoading(false)
  }

  async function handleCheckOut() {
    if (studentState?.status !== 'checked_in') return
    setLoading(true)
    const result = await checkOut(studentState.attendanceId)
    if (result.success) showResult(`${studentState.studentName}님\n퇴실 완료`)
    setLoading(false)
  }

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-8">
      <h1 className="text-2xl font-bold text-gray-600 mb-10">출결 체크</h1>

      {screen === 'keypad' && (
        <Keypad
          value={phone}
          onChange={setPhone}
          onConfirm={handleConfirm}
          disabled={loading}
        />
      )}

      {screen === 'action' && studentState && (
        <ActionPanel
          state={studentState}
          onCheckIn={handleCheckIn}
          onStartOuting={handleStartOuting}
          onEndOuting={handleEndOuting}
          onCheckOut={handleCheckOut}
          onBack={handleBack}
          loading={loading}
        />
      )}

      {screen === 'result' && (
        <div className="text-center animate-pulse">
          <p className="text-6xl mb-6">✓</p>
          <p className="text-3xl font-bold text-gray-800 whitespace-pre-line">{resultMessage}</p>
          <p className="text-gray-400 mt-6 text-sm">3초 후 자동으로 돌아갑니다</p>
        </div>
      )}
    </main>
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
git add academy-app/app/kiosk/page.tsx
git commit -m "feat: implement kiosk main page with keypad and action flow"
```

---

## Self-Review

### 1. Spec 커버리지

| 스펙 항목 | Task |
|---|---|
| 숫자 키패드 UI | Task 2 |
| 상태 판단 로직 (Server Action) | Task 1 (`getStudentState`) |
| 입실 처리 + 지각 판단 | Task 1 (`checkIn`) |
| 외출 처리 (화장실/학원/식사) | Task 1 (`startOuting`) |
| 복귀 처리 | Task 1 (`endOuting`) |
| 퇴실 처리 | Task 1 (`checkOut`) |
| 학원 외출 정기 일정 체크 | schedules 쿼리로 expected_in 조회 |
| 3초 후 자동 리셋 | Task 4 (`showResult` setTimeout) |

### 2. Placeholder 스캔

- 모든 코드 스텝에 실제 코드 포함 ✓
- "TBD" 없음 ✓

### 3. 타입 일관성

- `StudentState` 유니온 타입이 `ActionPanel` props와 일치 ✓
- `OutingType` (`'toilet' | 'academy' | 'meal'`)이 `lib/types.ts`, `actions.ts`, `ActionPanel.tsx` 모두 일치 ✓
- `startOuting(attendanceId, studentId, outingType)` 시그니처가 page.tsx 호출과 일치 ✓
