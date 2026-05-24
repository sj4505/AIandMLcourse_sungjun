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
