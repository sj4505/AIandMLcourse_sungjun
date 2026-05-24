-- ============================================================
-- 초기 교직원 계정 생성 가이드
-- 순서: Supabase Auth에서 먼저 유저 생성 → 아래 SQL 실행
-- ============================================================

-- 1. Supabase 대시보드 → Authentication → Users → "Add user" 클릭
--    이메일/비밀번호 입력 후 "Create user" → 생성된 user의 UUID 복사

-- 2. 아래 SQL에서 UUID와 이름 교체 후 SQL Editor에서 실행

-- 원장 계정
INSERT INTO public.staff (id, name, role)
VALUES (
  'PASTE-AUTH-USER-UUID-HERE',
  '원장님',
  'principal'
);

-- 선생님 계정 (추가 시 주석 해제)
-- INSERT INTO public.staff (id, name, role)
-- VALUES (
--   'PASTE-TEACHER-UUID-HERE',
--   '선생님 이름',
--   'teacher'
-- );
