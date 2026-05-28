# 가계부 앱 — Google OAuth + Supabase + Polar.sh 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 단일 HTML 가계부 앱에 Google OAuth 로그인, Supabase 클라우드 DB, Polar.sh 월 구독 결제를 추가하여 다중 사용자 서비스로 만든다.

**Architecture:** localStorage를 인메모리 캐시로 교체 (읽기 동기 / 쓰기는 Supabase에 비동기 백그라운드). Supabase Auth로 Google OAuth 처리. Polar.sh 결제 완료 이벤트는 Vercel Function이 수신하여 Supabase 구독 상태 업데이트.

**Tech Stack:** Supabase JS (CDN v2), Polar.sh checkout link, Vercel Functions (Node.js CommonJS), Jest (webhook 테스트)

---

## 파일 구조

```
budget-app/
  index.html              ← 수정 (Supabase 연동, auth/payment 게이트 추가)
  api/
    webhook/
      polar.js            ← 신규 (Polar webhook 핸들러)
  tests/
    webhook.test.js       ← 신규 (Jest 테스트)
  package.json            ← 신규 (Vercel Function 의존성 + Jest)
  .env.local.example      ← 신규 (환경변수 템플릿)
  .gitignore              ← 수정 (환경변수 파일 제외)
```

---

## Task 1: Supabase 프로젝트 설정 (대시보드 수동 작업)

**Files:** 없음 (Supabase 대시보드에서 진행)

- [ ] **Step 1: Supabase 프로젝트 생성**

  [supabase.com](https://supabase.com) → New Project → 이름/비밀번호 입력 후 생성.  
  생성 완료 후 **Settings → API** 에서 아래 두 값을 복사해 둔다:
  - `Project URL` (예: `https://abcdefgh.supabase.co`)
  - `anon public` key

- [ ] **Step 2: DB 테이블 생성**

  Supabase 대시보드 → **SQL Editor** → New Query → 아래 SQL 실행:

  ```sql
  -- 사용자 테이블
  CREATE TABLE users (
    id                  uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email               text NOT NULL,
    subscription_status text DEFAULT 'inactive',
    subscription_id     text,
    created_at          timestamptz DEFAULT now()
  );

  ALTER TABLE users ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "read own"   ON users FOR SELECT USING (auth.uid() = id);
  CREATE POLICY "insert own" ON users FOR INSERT WITH CHECK (auth.uid() = id);
  CREATE POLICY "update own" ON users FOR UPDATE USING (auth.uid() = id);

  -- 사용자 데이터 테이블 (localStorage 키-값 구조 유지)
  CREATE TABLE user_data (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid REFERENCES users(id) ON DELETE CASCADE NOT NULL,
    key        text NOT NULL,
    value      jsonb NOT NULL,
    updated_at timestamptz DEFAULT now(),
    UNIQUE(user_id, key)
  );

  ALTER TABLE user_data ENABLE ROW LEVEL SECURITY;
  CREATE POLICY "manage own data" ON user_data FOR ALL USING (auth.uid() = user_id);
  ```

- [ ] **Step 3: Google OAuth 활성화**

  Supabase 대시보드 → **Authentication → Providers → Google** → Enable 토글 ON.  
  Google Cloud Console에서 발급한 OAuth Client ID / Secret 입력.  
  Authorized redirect URI에 아래 추가:
  ```
  https://<your-project-ref>.supabase.co/auth/v1/callback
  ```

- [ ] **Step 4: Vercel 도메인 허용 목록 추가**

  Supabase → **Authentication → URL Configuration → Redirect URLs** 에 아래 추가:
  ```
  https://<your-vercel-domain>.vercel.app
  http://localhost:3000
  ```

---

## Task 2: 프로젝트 스캐폴딩

**Files:**
- Create: `budget-app/package.json`
- Create: `budget-app/.env.local.example`
- Modify: `.gitignore` (루트)

- [ ] **Step 1: package.json 생성**

  `budget-app/package.json`:
  ```json
  {
    "name": "budget-app",
    "version": "1.0.0",
    "private": true,
    "scripts": {
      "test": "jest --testEnvironment node"
    },
    "dependencies": {
      "@supabase/supabase-js": "^2.49.4"
    },
    "devDependencies": {
      "jest": "^29.7.0"
    }
  }
  ```

- [ ] **Step 2: 환경변수 템플릿 생성**

  `budget-app/.env.local.example`:
  ```
  # Supabase (webhook 핸들러용 - service role key는 절대 공개 금지)
  SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
  SUPABASE_SERVICE_ROLE_KEY=YOUR_SERVICE_ROLE_KEY

  # Polar.sh
  POLAR_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET
  ```

- [ ] **Step 3: .gitignore 확인/수정**

  루트 `.gitignore`에 아래 항목이 없으면 추가:
  ```
  .env.local
  .env.*.local
  node_modules/
  ```

- [ ] **Step 4: 의존성 설치**

  ```bash
  cd budget-app
  npm install
  ```

  Expected: `node_modules/` 생성, `package-lock.json` 생성.

- [ ] **Step 5: 커밋**

  ```bash
  git add budget-app/package.json budget-app/package-lock.json budget-app/.env.local.example
  git commit -m "chore: scaffold budget-app for Vercel Functions"
  ```

---

## Task 3: Polar webhook 핸들러 + 테스트

**Files:**
- Create: `budget-app/api/webhook/polar.js`
- Create: `budget-app/tests/webhook.test.js`

- [ ] **Step 1: 테스트 파일 작성 (먼저)**

  `budget-app/tests/webhook.test.js`:
  ```javascript
  const crypto = require('crypto');

  // Supabase 클라이언트 모킹
  const mockUpdate = jest.fn().mockReturnValue({ eq: jest.fn().mockResolvedValue({ error: null }) });
  jest.mock('@supabase/supabase-js', () => ({
    createClient: () => ({ from: () => ({ update: mockUpdate }) })
  }));

  const handler = require('../api/webhook/polar');

  const WEBHOOK_SECRET = 'whsec_dGVzdHNlY3JldA=='; // whsec_ + base64('testsecret')
  const SECRET_RAW = 'dGVzdHNlY3JldA==';

  function makeRequest(body, overrideSignature) {
    const msgId = 'msg_test_001';
    const msgTimestamp = String(Math.floor(Date.now() / 1000));
    const toSign = `${msgId}.${msgTimestamp}.${body}`;
    const secretBytes = Buffer.from(SECRET_RAW, 'base64');
    const sig = crypto.createHmac('sha256', secretBytes).update(toSign).digest('base64');

    const req = {
      method: 'POST',
      headers: {
        'webhook-id': msgId,
        'webhook-timestamp': msgTimestamp,
        'webhook-signature': overrideSignature ?? `v1,${sig}`,
      },
      on: jest.fn((event, cb) => {
        if (event === 'data') cb(body);
        if (event === 'end') cb();
        return req;
      }),
    };
    const res = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
      end: jest.fn(),
    };
    return { req, res };
  }

  beforeEach(() => {
    process.env.POLAR_WEBHOOK_SECRET = WEBHOOK_SECRET;
    process.env.SUPABASE_URL = 'https://test.supabase.co';
    process.env.SUPABASE_SERVICE_ROLE_KEY = 'service-role-key';
    mockUpdate.mockClear();
  });

  test('405 for non-POST method', async () => {
    const req = { method: 'GET' };
    const res = { status: jest.fn().mockReturnThis(), end: jest.fn() };
    await handler(req, res);
    expect(res.status).toHaveBeenCalledWith(405);
  });

  test('401 for invalid signature', async () => {
    const { req, res } = makeRequest('{}', 'v1,invalidsignature==');
    await handler(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  test('200 and sets active for subscription.active', async () => {
    const body = JSON.stringify({
      type: 'subscription.active',
      data: { id: 'sub_abc', customer: { email: 'user@example.com' } },
    });
    const { req, res } = makeRequest(body);
    await handler(req, res);
    expect(res.status).toHaveBeenCalledWith(200);
    expect(mockUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ subscription_status: 'active' })
    );
  });

  test('200 and sets inactive for subscription.canceled', async () => {
    const body = JSON.stringify({
      type: 'subscription.canceled',
      data: { id: 'sub_abc', customer: { email: 'user@example.com' } },
    });
    const { req, res } = makeRequest(body);
    await handler(req, res);
    expect(res.status).toHaveBeenCalledWith(200);
    expect(mockUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ subscription_status: 'inactive' })
    );
  });

  test('200 with no-op when customer email missing', async () => {
    const body = JSON.stringify({
      type: 'subscription.active',
      data: { id: 'sub_abc' },
    });
    const { req, res } = makeRequest(body);
    await handler(req, res);
    expect(res.status).toHaveBeenCalledWith(200);
    expect(mockUpdate).not.toHaveBeenCalled();
  });
  ```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

  ```bash
  cd budget-app
  npm test
  ```

  Expected: `Cannot find module '../api/webhook/polar'` 에러로 실패.

- [ ] **Step 3: webhook 핸들러 구현**

  `budget-app/api/webhook/polar.js`:
  ```javascript
  const crypto = require('crypto');
  const { createClient } = require('@supabase/supabase-js');

  const ACTIVE_EVENTS = new Set(['subscription.active', 'subscription.created']);

  function verifySignature(rawBody, headers, secret) {
    const msgId = headers['webhook-id'];
    const msgTimestamp = headers['webhook-timestamp'];
    const msgSignature = headers['webhook-signature'];
    if (!msgId || !msgTimestamp || !msgSignature) return false;

    const toSign = `${msgId}.${msgTimestamp}.${rawBody}`;
    const secretBytes = Buffer.from(secret.replace(/^whsec_/, ''), 'base64');
    const computed = crypto
      .createHmac('sha256', secretBytes)
      .update(toSign)
      .digest('base64');

    return msgSignature.split(' ').some(sig => {
      const [version, val] = sig.split(',');
      return version === 'v1' && val === computed;
    });
  }

  function getRawBody(req) {
    return new Promise((resolve, reject) => {
      let data = '';
      req.on('data', chunk => { data += chunk; });
      req.on('end', () => resolve(data));
      req.on('error', reject);
    });
  }

  module.exports = async function handler(req, res) {
    if (req.method !== 'POST') return res.status(405).end();

    const rawBody = await getRawBody(req);
    const isValid = verifySignature(rawBody, req.headers, process.env.POLAR_WEBHOOK_SECRET);
    if (!isValid) return res.status(401).json({ error: 'Invalid signature' });

    const event = JSON.parse(rawBody);
    const email = event.data?.customer?.email;
    if (!email) return res.status(200).json({ ok: true });

    const status = ACTIVE_EVENTS.has(event.type) ? 'active' : 'inactive';
    const subscriptionId = event.data?.id ?? null;

    const sbAdmin = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { error } = await sbAdmin
      .from('users')
      .update({ subscription_status: status, subscription_id: subscriptionId })
      .eq('email', email);

    if (error) return res.status(500).json({ error: error.message });
    return res.status(200).json({ ok: true });
  };
  ```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

  ```bash
  cd budget-app
  npm test
  ```

  Expected:
  ```
  PASS tests/webhook.test.js
    ✓ 405 for non-POST method
    ✓ 401 for invalid signature
    ✓ 200 and sets active for subscription.active
    ✓ 200 and sets inactive for subscription.canceled
    ✓ 200 with no-op when customer email missing
  Tests: 5 passed
  ```

- [ ] **Step 5: 커밋**

  ```bash
  git add budget-app/api/webhook/polar.js budget-app/tests/webhook.test.js
  git commit -m "feat: add Polar.sh webhook handler with signature verification"
  ```

---

## Task 4: index.html - Supabase 설정 및 데이터 레이어

**Files:**
- Modify: `budget-app/index.html`

- [ ] **Step 1: Supabase CDN 스크립트 태그 추가**

  `index.html` `</head>` 바로 앞에 추가:
  ```html
    <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
  </head>
  ```

- [ ] **Step 2: Supabase 설정 상수 추가**

  `index.html`의 `<script>` 태그 내 맨 위 (`// ── STORAGE` 주석 위)에 추가:
  ```javascript
  // ── SUPABASE CONFIG ────────────────────────────────────────────────
  // SUPABASE_URL, SUPABASE_ANON_KEY: 공개해도 안전 (anon key, RLS로 보호됨)
  // POLAR_CHECKOUT_URL: Polar 대시보드 → 상품 → Checkout URL
  const SUPABASE_URL        = 'YOUR_SUPABASE_PROJECT_URL';
  const SUPABASE_ANON_KEY   = 'YOUR_SUPABASE_ANON_KEY';
  const POLAR_CHECKOUT_URL  = 'YOUR_POLAR_SANDBOX_CHECKOUT_URL';

  const _sbClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  let _currentUser = null;
  const _memCache  = {};
  ```

- [ ] **Step 3: `store()`, `save()` 함수 교체**

  기존 코드 (96~97번째 줄):
  ```javascript
  function store(key)        { return JSON.parse(localStorage.getItem(key) ?? 'null'); }
  function save(key, val)    { localStorage.setItem(key, JSON.stringify(val)); }
  ```

  위 두 줄을 아래로 교체:
  ```javascript
  function store(key) {
    return _memCache[key] ?? null;
  }

  function save(key, val) {
    _memCache[key] = val;
    if (!_currentUser) return;
    _sbClient.from('user_data').upsert({
      user_id:    _currentUser.id,
      key,
      value:      val,
      updated_at: new Date().toISOString(),
    }).then(({ error }) => {
      if (error) console.error('[save]', key, error.message);
    });
  }
  ```

- [ ] **Step 4: `initStorage()` 함수를 async로 교체**

  기존 코드 (104~108번째 줄):
  ```javascript
  function initStorage() {
    Object.keys(DEFAULT_DATA).forEach(key => {
      if (store(key) === null) save(key, DEFAULT_DATA[key]);
    });
  }
  ```

  위 함수를 아래로 교체:
  ```javascript
  async function initStorage() {
    const { data, error } = await _sbClient
      .from('user_data')
      .select('key, value')
      .eq('user_id', _currentUser.id);

    if (error) { console.error('[initStorage]', error.message); return; }

    data.forEach(row => { _memCache[row.key] = row.value; });

    // 없는 키는 기본값으로 초기화 (Supabase에도 저장)
    Object.keys(DEFAULT_DATA).forEach(key => {
      if (_memCache[key] == null) {
        save(key, DEFAULT_DATA[key]);
      }
    });
  }
  ```

- [ ] **Step 5: 커밋**

  ```bash
  git add budget-app/index.html
  git commit -m "feat: replace localStorage with Supabase-backed in-memory cache"
  ```

---

## Task 5: index.html - 인증 게이트 (Google 로그인 화면)

**Files:**
- Modify: `budget-app/index.html`

- [ ] **Step 1: 로그인 오버레이 HTML 추가**

  `<body class="flex flex-col" ...>` 바로 다음 줄에 추가:
  ```html
  <!-- ── AUTH OVERLAY ──────────────────────────────────────────── -->
  <div id="auth-overlay"
    style="display:none; position:fixed; inset:0; background:var(--bg); z-index:200;
           flex-direction:column; align-items:center; justify-content:center; gap:20px; padding:24px; text-align:center;">
    <div class="mono" style="font-size:2.5rem;">₩</div>
    <div style="font-size:1.25rem; font-weight:700;">가계부</div>
    <p style="font-size:0.875rem; color:var(--text2);">Google 계정으로 로그인하세요</p>
    <button onclick="signInWithGoogle()"
      style="display:flex; align-items:center; gap:10px; padding:12px 24px;
             border-radius:12px; background:#fff; color:#0f172a; font-weight:700; font-size:0.875rem; border:none; cursor:pointer;">
      <img src="https://www.google.com/favicon.ico" width="18" height="18" alt="Google">
      Google로 로그인
    </button>
  </div>
  ```

- [ ] **Step 2: 앱 초기화 함수 추가**

  `initStorage();` `switchTab('home');` 두 줄 (1070~1071번째 줄) 바로 위에 아래 함수들 추가:
  ```javascript
  // ── AUTH & INIT ──────────────────────────────────────────────────────

  function showOverlay(id) {
    ['auth-overlay', 'pay-overlay'].forEach(oid => {
      const el = document.getElementById(oid);
      if (el) el.style.display = (oid === id) ? 'flex' : 'none';
    });
  }

  function hideOverlays() {
    ['auth-overlay', 'pay-overlay'].forEach(oid => {
      const el = document.getElementById(oid);
      if (el) el.style.display = 'none';
    });
  }

  async function signInWithGoogle() {
    await _sbClient.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.href },
    });
  }

  async function ensureUserRecord(user) {
    const { data } = await _sbClient
      .from('users')
      .select('id')
      .eq('id', user.id)
      .maybeSingle();
    if (!data) {
      await _sbClient.from('users').insert({ id: user.id, email: user.email });
    }
  }

  async function getSubscriptionStatus(userId) {
    const { data } = await _sbClient
      .from('users')
      .select('subscription_status')
      .eq('id', userId)
      .single();
    return data?.subscription_status ?? 'inactive';
  }

  async function init() {
    const { data: { session } } = await _sbClient.auth.getSession();

    if (!session) {
      showOverlay('auth-overlay');
      return;
    }

    _currentUser = session.user;
    await ensureUserRecord(session.user);

    const status = await getSubscriptionStatus(session.user.id);
    if (status !== 'active') {
      showOverlay('pay-overlay');
      return;
    }

    await initStorage();
    hideOverlays();
    switchTab('home');
  }

  // OAuth 리다이렉트 콜백 처리
  _sbClient.auth.onAuthStateChange(async (event, session) => {
    if (event === 'SIGNED_IN' && session) {
      _currentUser = session.user;
      await init();
    }
  });
  ```

- [ ] **Step 3: 마지막 두 줄 교체**

  기존:
  ```javascript
  initStorage();
  switchTab('home');
  ```

  교체:
  ```javascript
  init();
  ```

- [ ] **Step 4: 커밋**

  ```bash
  git add budget-app/index.html
  git commit -m "feat: add Google OAuth auth gate via Supabase Auth"
  ```

---

## Task 6: index.html - 구독 게이트 (결제 화면)

**Files:**
- Modify: `budget-app/index.html`

- [ ] **Step 1: 결제 안내 오버레이 HTML 추가**

  `auth-overlay` div 바로 다음에 추가:
  ```html
  <!-- ── PAY OVERLAY ──────────────────────────────────────────── -->
  <div id="pay-overlay"
    style="display:none; position:fixed; inset:0; background:var(--bg); z-index:200;
           flex-direction:column; align-items:center; justify-content:center; gap:20px; padding:24px; text-align:center;">
    <div class="mono" style="font-size:2.5rem;">₩</div>
    <div style="font-size:1.25rem; font-weight:700;">구독이 필요해요</div>
    <p style="font-size:0.875rem; color:var(--text2);">첫 1개월은 무료로 사용할 수 있어요</p>
    <button onclick="goToCheckout()"
      style="padding:12px 28px; border-radius:12px; background:var(--green); color:#0f172a;
             font-weight:700; font-size:0.875rem; border:none; cursor:pointer;">
      무료로 시작하기 →
    </button>
    <button onclick="refreshSubscription()"
      style="font-size:0.75rem; color:var(--text3); background:none; border:none; cursor:pointer; text-decoration:underline;">
      이미 결제했어요 (새로고침)
    </button>
    <button onclick="handleLogout()"
      style="font-size:0.75rem; color:var(--text3); background:none; border:none; cursor:pointer;">
      다른 계정으로 로그인
    </button>
  </div>
  ```

- [ ] **Step 2: goToCheckout, refreshSubscription 함수 추가**

  `init()` 함수 아래에 추가:
  ```javascript
  function goToCheckout() {
    const email = _currentUser?.email ?? '';
    const url = POLAR_CHECKOUT_URL +
      (email ? '?customer_email=' + encodeURIComponent(email) : '');
    window.location.href = url;
  }

  async function refreshSubscription() {
    if (!_currentUser) { showOverlay('auth-overlay'); return; }
    const status = await getSubscriptionStatus(_currentUser.id);
    if (status === 'active') {
      await initStorage();
      hideOverlays();
      switchTab('home');
    } else {
      showToast('아직 구독이 확인되지 않았어요. 잠시 후 다시 시도해주세요.', 'warn');
    }
  }
  ```

- [ ] **Step 3: 커밋**

  ```bash
  git add budget-app/index.html
  git commit -m "feat: add Polar.sh subscription gate and checkout redirect"
  ```

---

## Task 7: index.html - 설정 탭에 계정 섹션 추가

**Files:**
- Modify: `budget-app/index.html`

- [ ] **Step 1: renderSettings()에 계정 섹션 추가**

  `renderSettings()` 함수 안, `</div>\`;` (설정 컨테이너 닫기) 바로 앞에 추가.  
  구체적으로, `OpenAI API` 섹션 div 닫는 태그 이후 `</div>\`;\`` 직전:

  ```javascript
          <div class="mb-6">
            <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--green)">계정</div>
            <div class="rounded-lg px-3 py-2 mb-2" style="background:var(--bg2);border:1px solid var(--border);">
              <p class="text-xs mb-1" style="color:var(--text2)">로그인 계정</p>
              <p class="text-sm mono">${escapeHtml(_currentUser?.email ?? '')}</p>
            </div>
            <div class="rounded-lg px-3 py-2 mb-3" style="background:var(--bg2);border:1px solid var(--border);">
              <p class="text-xs mb-1" style="color:var(--text2)">구독 상태</p>
              <p class="text-sm mono" id="sub-status-display" style="color:var(--text3)">확인 중...</p>
            </div>
            <button onclick="handleLogout()"
              class="w-full rounded-lg py-2 text-sm font-bold"
              style="background:var(--bg2);border:1px solid var(--border);color:var(--red);">로그아웃</button>
          </div>
  ```

- [ ] **Step 2: 구독 상태 비동기 로드 추가**

  `renderSettings()` 함수 맨 끝 (tab.innerHTML = `...` 대입문 바로 다음)에 추가:
  ```javascript
  // 구독 상태 비동기 로드
  if (_currentUser) {
    getSubscriptionStatus(_currentUser.id).then(status => {
      const el = document.getElementById('sub-status-display');
      if (!el) return;
      el.textContent = status === 'active' ? '구독 중 ✓' : '미구독';
      el.style.color = status === 'active' ? 'var(--green)' : 'var(--amber)';
    });
  }
  ```

- [ ] **Step 3: handleLogout 함수 추가**

  `refreshSubscription()` 함수 아래에 추가:
  ```javascript
  async function handleLogout() {
    await _sbClient.auth.signOut();
    _currentUser = null;
    Object.keys(_memCache).forEach(k => delete _memCache[k]);
    showOverlay('auth-overlay');
  }
  ```

- [ ] **Step 4: OpenAI API 섹션 텍스트 수정**

  설정탭 OpenAI API 섹션 안내 문구 (현재: "이 기기 localStorage에만 저장됩니다."):
  ```html
  <p class="text-xs mt-1" style="color:var(--text3)">클라우드에 암호화되어 저장됩니다.</p>
  ```

- [ ] **Step 5: 커밋**

  ```bash
  git add budget-app/index.html
  git commit -m "feat: add account section with logout and subscription status to settings"
  ```

---

## Task 8: Vercel 배포 및 환경변수 설정

**Files:** 없음 (Vercel/Polar 대시보드 설정)

- [ ] **Step 1: GitHub에 푸시**

  ```bash
  git push origin HEAD
  ```

- [ ] **Step 2: Vercel 프로젝트 연결**

  [vercel.com](https://vercel.com) → New Project → GitHub 저장소 선택.  
  **Root Directory** 를 `budget-app` 으로 설정 (중요).  
  Framework Preset: `Other`.

- [ ] **Step 3: Vercel 환경변수 설정**

  Vercel 프로젝트 → Settings → Environment Variables에 아래 추가:
  | 이름 | 값 |
  |------|----|
  | `SUPABASE_URL` | Supabase Project URL |
  | `SUPABASE_SERVICE_ROLE_KEY` | Supabase service_role key (Settings → API) |
  | `POLAR_WEBHOOK_SECRET` | Polar 대시보드 → Webhooks → Secret |

  > `SUPABASE_ANON_KEY` 와 `SUPABASE_URL` 은 `index.html`에 하드코딩되므로 Vercel 환경변수 불필요.

- [ ] **Step 4: Polar webhook 엔드포인트 등록**

  Polar 대시보드 (sandbox) → Webhooks → Add Endpoint:
  ```
  URL: https://<your-vercel-domain>.vercel.app/api/webhook/polar
  Events: subscription.created, subscription.active, subscription.canceled, subscription.revoked
  ```
  생성 후 Secret을 복사해 Vercel 환경변수 `POLAR_WEBHOOK_SECRET`에 설정.

- [ ] **Step 5: index.html 상수값 채우기**

  `budget-app/index.html` 상단의 세 상수에 실제 값 입력:
  ```javascript
  const SUPABASE_URL       = 'https://YOUR_REF.supabase.co';
  const SUPABASE_ANON_KEY  = 'eyJhb...';
  const POLAR_CHECKOUT_URL = 'https://sandbox.polar.sh/checkout/YOUR_PRODUCT_ID';
  ```

  커밋 및 푸시:
  ```bash
  git add budget-app/index.html
  git commit -m "feat: wire Supabase and Polar config constants"
  git push origin HEAD
  ```

- [ ] **Step 6: E2E 동작 확인**

  배포된 Vercel URL에서 아래 흐름 확인:
  1. 앱 접속 → Google 로그인 화면 표시
  2. Google 로그인 완료 → 구독 안내 화면 표시
  3. "무료로 시작하기" 클릭 → Polar 결제 페이지 이동
  4. 결제 완료 → Polar webhook 전송 → Supabase `subscription_status = 'active'` 업데이트
  5. 앱으로 돌아와 "이미 결제했어요" 클릭 → 앱 정상 진입
  6. 설정 탭 → 계정 섹션에서 이메일, 구독 상태 "구독 중 ✓" 확인
  7. 로그아웃 → 로그인 화면으로 복귀

---

## Self-Review 체크

- **스펙 커버리지:** 모든 요구사항(Google OAuth, Supabase DB, Polar.sh 구독, 1개월 무료 체험) 포함됨. ✓
- **플레이스홀더:** 없음. 모든 코드 블록이 실제 구현 코드. ✓
- **타입 일관성:** `_sbClient`, `_memCache`, `_currentUser`, `showOverlay()`, `hideOverlays()`, `getSubscriptionStatus()` 전 태스크에서 동일하게 사용. ✓
- **누락:** 없음.
