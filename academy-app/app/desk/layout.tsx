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
