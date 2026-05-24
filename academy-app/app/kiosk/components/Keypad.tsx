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
