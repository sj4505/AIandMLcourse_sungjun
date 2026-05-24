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
        <div className="text-center">
          <p className="text-6xl mb-6">✓</p>
          <p className="text-3xl font-bold text-gray-800 whitespace-pre-line">{resultMessage}</p>
          <p className="text-gray-400 mt-6 text-sm">3초 후 자동으로 돌아갑니다</p>
        </div>
      )}
    </main>
  )
}
