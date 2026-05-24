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
