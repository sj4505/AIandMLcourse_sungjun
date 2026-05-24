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

-- 교시
CREATE TABLE IF NOT EXISTS public.periods (
  id                    SERIAL   PRIMARY KEY,
  name                  TEXT     NOT NULL,
  type                  TEXT     NOT NULL CHECK (type IN ('class', 'break', 'meal')),
  weekday_start         TIME     NOT NULL,
  weekday_end           TIME     NOT NULL,
  weekend_start         TIME     NOT NULL,
  weekend_end           TIME     NOT NULL,
  weekday_is_autonomous BOOLEAN  NOT NULL DEFAULT FALSE,
  weekend_is_autonomous BOOLEAN  NOT NULL DEFAULT FALSE,
  order_index           INT      NOT NULL
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
  id            UUID        NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,
  attendance_id UUID        NOT NULL REFERENCES public.attendance(id) ON DELETE CASCADE,
  student_id    UUID        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  out_at        TIMESTAMPTZ NOT NULL,
  back_at       TIMESTAMPTZ,
  outing_type   TEXT        NOT NULL CHECK (outing_type IN ('toilet', 'academy', 'meal')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
  month               TEXT          NOT NULL,
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

-- students: 읽기는 anon 허용 (키오스크 전화번호 조회), 쓰기는 인증만
CREATE POLICY "students_select_anon" ON public.students
  FOR SELECT USING (true);
CREATE POLICY "students_write_auth" ON public.students
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

INSERT INTO public.periods (name, type, weekday_start, weekday_end, weekend_start, weekend_end, weekday_is_autonomous, weekend_is_autonomous, order_index) VALUES
  ('등원',    'break', '07:50', '08:00', '07:50', '08:00', false, false,  0),
  ('1교시',   'class', '08:00', '08:50', '08:00', '08:50', false, false,  1),
  ('쉬는시간', 'break', '08:50', '09:00', '08:50', '09:00', false, false,  2),
  ('2교시',   'class', '09:00', '10:30', '09:00', '10:30', false, false,  3),
  ('쉬는시간', 'break', '10:30', '10:40', '10:30', '10:40', false, false,  4),
  ('3교시',   'class', '10:40', '12:10', '10:40', '12:10', false, false,  5),
  ('점심시간', 'meal',  '12:10', '13:10', '12:10', '13:10', false, false,  6),
  ('4교시',   'class', '13:10', '14:40', '13:10', '14:40', false, false,  7),
  ('쉬는시간', 'break', '14:40', '14:50', '14:40', '14:50', false, false,  8),
  ('5교시',   'class', '14:50', '16:20', '14:50', '16:20', false, false,  9),
  ('쉬는시간', 'break', '16:20', '16:30', '16:20', '16:30', false, false, 10),
  ('6교시',   'class', '16:30', '18:00', '16:30', '18:00', false, false, 11),
  ('저녁시간', 'meal',  '18:00', '19:00', '18:00', '19:00', false, false, 12),
  ('7교시',   'class', '19:00', '20:30', '19:00', '20:30', false, true,  13),
  ('쉬는시간', 'break', '20:30', '20:40', '20:30', '20:40', false, false, 14),
  ('8교시',   'class', '20:40', '22:00', '20:40', '22:00', false, true,  15),
  ('쉬는시간', 'break', '22:00', '22:10', '22:00', '22:10', false, false, 16),
  ('9교시',   'class', '22:10', '23:40', '22:10', '23:40', true,  true,  17)
ON CONFLICT DO NOTHING;
