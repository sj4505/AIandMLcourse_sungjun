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

export type PeriodType = 'class' | 'break' | 'meal'

export interface Period {
  id: number
  name: string
  type: PeriodType
  weekday_start: string
  weekday_end: string
  weekend_start: string
  weekend_end: string
  weekday_is_autonomous: boolean
  weekend_is_autonomous: boolean
  order_index: number
}

export interface Schedule {
  id: string
  student_id: string
  day_of_week: number
  expected_in: string
  expected_out: string
  title: string | null
  status: ScheduleStatus
  created_at: string
}

export interface Attendance {
  id: string
  student_id: string
  date: string
  check_in_at: string | null
  check_out_at: string | null
  is_late: boolean
  late_minutes: number
  created_at: string
}

export interface Outing {
  id: string
  attendance_id: string
  student_id: string
  out_at: string
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
  date: string
  content: string
  is_done: boolean
  created_at: string
}

export interface Report {
  id: string
  student_id: string
  month: string
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
