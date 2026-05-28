# 가계부 앱 — Google OAuth + Supabase + Polar.sh 설계

**날짜:** 2026-04-08  
**범위:** 기존 단일 HTML 가계부 앱에 인증, 클라우드 DB, 구독 결제 추가

---

## 1. 목표

- 여러 사용자가 각자 계정으로 앱을 사용할 수 있게 한다
- Google 계정으로 로그인 (Google OAuth via Supabase Auth)
- 데이터를 localStorage 대신 Supabase(PostgreSQL)에 저장 (사용자별 격리)
- Polar.sh 월 구독 결제 (1개월 무료 체험)
- 미결제 사용자는 앱 진입 차단

---

## 2. 기술 스택

| 역할 | 기술 |
|------|------|
| 프론트엔드 | 기존 `index.html` (Tailwind CDN) |
| 인증 | Supabase Auth (Google OAuth provider) |
| DB | Supabase (PostgreSQL) |
| 구독 결제 | Polar.sh (월정액, 1개월 무료 체험) |
| Webhook 처리 | Vercel Function (`/api/webhook/polar`) |
| 배포 | Vercel (GitHub 연동) |

---

## 3. 아키텍처

```
[index.html]
  ├── Supabase JS (CDN)   ← Google OAuth + DB read/write
  └── Polar checkout link ← 구독 결제 페이지 이동

[Vercel Function]
  └── /api/webhook/polar
        ← Polar.sh가 결제 완료/취소/갱신 이벤트 POST
        → Supabase users 테이블 subscription_status 업데이트
```

---

## 4. 사용자 흐름

1. **앱 진입** → Supabase 세션 확인
2. **미로그인** → Google OAuth 로그인 화면 표시
3. **로그인 완료** → Supabase `users` 테이블에서 `subscription_status` 조회
4. **미구독 / 만료** → Polar.sh 결제 페이지로 이동 (checkout URL + `?customer_email=` 파라미터)
5. **결제 완료** → Polar webhook → `/api/webhook/polar` → `subscription_status = 'active'` 업데이트
6. **구독 중** → 기존 앱 화면 표시, 모든 데이터 Supabase에서 로드/저장

---

## 5. Supabase DB 스키마

### `users` 테이블
```sql
id                  uuid  PRIMARY KEY  -- Supabase Auth user id
email               text  NOT NULL
subscription_status text  DEFAULT 'inactive'  -- 'active' | 'inactive' | 'trialing'
subscription_id     text  -- Polar.sh subscription id
trial_ends_at       timestamptz
created_at          timestamptz DEFAULT now()
```

### `user_data` 테이블
기존 localStorage 키들을 통합 저장:
```sql
id          uuid  PRIMARY KEY DEFAULT gen_random_uuid()
user_id     uuid  REFERENCES users(id) ON DELETE CASCADE
key         text  NOT NULL   -- 'settings' | 'work' | 'expenses' | ...
value       jsonb NOT NULL
updated_at  timestamptz DEFAULT now()
UNIQUE(user_id, key)
```

Row Level Security (RLS): 각 사용자는 자신의 `user_id` 행만 읽기/쓰기 가능.

---

## 6. 코드 변경 범위

### `index.html`
- Supabase JS SDK CDN 추가
- 앱 시작 시 auth/subscription 게이트 로직 추가 (로그인 화면, 결제 안내 화면)
- `store()` / `save()` 함수를 Supabase 비동기 호출로 교체
- 설정 탭에 로그아웃 버튼, 구독 상태 표시 추가

### `api/webhook/polar.js` (신규)
- Polar.sh webhook signature 검증
- 이벤트 타입별 처리:
  - `subscription.created` / `subscription.activated` → `status = 'active'`
  - `subscription.canceled` / `subscription.revoked` → `status = 'inactive'`

---

## 7. 환경 변수

```
# Vercel 환경 변수로 설정
SUPABASE_URL
SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY   # webhook에서 RLS 우회용
POLAR_WEBHOOK_SECRET
```

---

## 8. 범위 외 (이번 구현에 포함 안 함)

- Toss / 포트원 결제 내역 자동 import (마이데이터 허가 필요)
- 파일 분리 / 번들링 (index.html 단일 파일 유지)
- 관리자 대시보드
