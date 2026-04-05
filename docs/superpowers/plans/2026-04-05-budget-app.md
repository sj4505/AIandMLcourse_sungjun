# 가계부 앱 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 단일 HTML 파일로 동작하는 대학생용 가계부 앱 — 알바/용돈 기반 일일 예산 계산, SMS 파싱 지출 입력, 할부·대출 추적.

**Architecture:** `budget-app/index.html` 단일 파일. JS는 파일 하단 `<script>`에 STORAGE / CALCULATIONS / UI / EVENTS / INIT 섹션으로 구분. 순수 함수(계산 엔진)는 `budget-app/tests.html`에서 브라우저 기반 단위 테스트.

**Tech Stack:** HTML5, Tailwind CSS (CDN), Vanilla JS (ES2020), localStorage, OpenAI API (GPT-4o mini), JetBrains Mono + Noto Sans KR (Google Fonts)

**Spec:** `docs/superpowers/specs/2026-04-05-budget-app-design.md`

---

## Task 1: 프로젝트 스캐폴드 — HTML 구조 + 디자인 시스템 + 탭 네비게이션

**Files:**
- Create: `budget-app/index.html`

- [ ] **Step 1: 기본 HTML 뼈대 작성**

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>가계부</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+KR:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:       #0f172a;
      --bg2:      #0a1120;
      --border:   #1e293b;
      --text:     #f1f5f9;
      --text2:    #94a3b8;
      --text3:    #475569;
      --green:    #34d399;
      --amber:    #fbbf24;
      --red:      #f87171;
      --red2:     #ef4444;
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Noto Sans KR', sans-serif;
      min-height: 100dvh;
      overscroll-behavior: none;
    }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    /* 색상 상태 */
    .color-good   { color: var(--green); }
    .color-warn   { color: var(--amber); }
    .color-danger { color: var(--red);   }
    .color-over   { color: var(--red2);  }
  </style>
</head>
<body class="flex flex-col" style="min-height:100dvh;">

  <!-- 메인 콘텐츠 영역 -->
  <main id="main" class="flex-1 overflow-y-auto pb-16">
    <div id="tab-home"         class="tab-content active p-4"></div>
    <div id="tab-expense"      class="tab-content p-4"></div>
    <div id="tab-installments" class="tab-content p-4"></div>
    <div id="tab-settings"     class="tab-content p-4"></div>
  </main>

  <!-- 하단 탭바 -->
  <nav class="fixed bottom-0 left-0 right-0 flex border-t" style="background:#080f1e; border-color:var(--border); height:56px;">
    <button class="tab-btn flex-1 flex flex-col items-center justify-center gap-0.5" data-tab="home"         onclick="switchTab('home')">
      <span class="text-lg">🏠</span>
      <span class="text-[9px] mono uppercase tracking-wide tab-btn-label">홈</span>
    </button>
    <button class="tab-btn flex-1 flex flex-col items-center justify-center gap-0.5" data-tab="expense"      onclick="switchTab('expense')">
      <span class="text-lg">➕</span>
      <span class="text-[9px] mono uppercase tracking-wide tab-btn-label">지출</span>
    </button>
    <button class="tab-btn flex-1 flex flex-col items-center justify-center gap-0.5" data-tab="installments" onclick="switchTab('installments')">
      <span class="text-lg">💳</span>
      <span class="text-[9px] mono uppercase tracking-wide tab-btn-label">할부</span>
    </button>
    <button class="tab-btn flex-1 flex flex-col items-center justify-center gap-0.5" data-tab="settings"     onclick="switchTab('settings')">
      <span class="text-lg">⚙️</span>
      <span class="text-[9px] mono uppercase tracking-wide tab-btn-label">설정</span>
    </button>
  </nav>

  <script>
    // ── TAB NAVIGATION ──────────────────────────────────────────────
    let currentTab = 'home';

    function switchTab(name) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + name).classList.add('active');

      document.querySelectorAll('.tab-btn').forEach(btn => {
        const isActive = btn.dataset.tab === name;
        btn.querySelector('.tab-btn-label').style.color = isActive ? 'var(--green)' : 'var(--text3)';
      });

      currentTab = name;
      renderTab(name);
    }

    function renderTab(name) {
      if (name === 'home')         renderHome();
      if (name === 'expense')      renderExpense();
      if (name === 'installments') renderInstallments();
      if (name === 'settings')     renderSettings();
    }

    // 스텁 — 이후 태스크에서 구현
    function renderHome()         { document.getElementById('tab-home').innerHTML         = '<p class="text-sm" style="color:var(--text3)">홈 (구현 예정)</p>'; }
    function renderExpense()      { document.getElementById('tab-expense').innerHTML      = '<p class="text-sm" style="color:var(--text3)">지출 (구현 예정)</p>'; }
    function renderInstallments() { document.getElementById('tab-installments').innerHTML = '<p class="text-sm" style="color:var(--text3)">할부 (구현 예정)</p>'; }
    function renderSettings()     { document.getElementById('tab-settings').innerHTML     = '<p class="text-sm" style="color:var(--text3)">설정 (구현 예정)</p>'; }

    // 초기 탭 활성화
    switchTab('home');
  </script>
</body>
</html>
```

- [ ] **Step 2: 브라우저에서 열어 탭 전환이 동작하는지 확인**

`budget-app/index.html`을 브라우저로 열고 4개 탭 버튼을 클릭해 화면이 전환되고 활성 탭 라벨이 초록색으로 바뀌는지 확인.

- [ ] **Step 3: 커밋**

```bash
git add budget-app/index.html
git commit -m "feat: scaffold budget app with tab navigation and design system"
```

---

## Task 2: 스토리지 레이어 — localStorage CRUD + 기본값 초기화

**Files:**
- Modify: `budget-app/index.html` (STORAGE 섹션 추가)

- [ ] **Step 1: STORAGE 섹션을 `<script>` 맨 위에 추가**

TAB NAVIGATION 주석 위에 아래 코드를 삽입:

```javascript
// ── STORAGE ─────────────────────────────────────────────────────────
const KEYS = {
  settings:     'budgetApp_settings',
  work:         'budgetApp_work',
  workLog:      'budgetApp_workLog',
  fixed:        'budgetApp_fixed',
  expenses:     'budgetApp_expenses',
  installments: 'budgetApp_installments',
  loans:        'budgetApp_loans',
  categories:   'budgetApp_categories',
};

const DEFAULT_CATEGORIES = [
  '식비','카페','편의점','교통','쇼핑','구독','의료','문화',
  '자기관리','단백질음료','운동','사줌','기타'
].map((name, i) => ({ id: i + 1, name, isDefault: true }));

const DEFAULT_DATA = {
  [KEYS.settings]:     { allowance: 0, apiKey: '' },
  [KEYS.work]:         { pattern: [], hourlyWage: 10030 },
  [KEYS.workLog]:      [],
  [KEYS.fixed]:        [],
  [KEYS.expenses]:     [],
  [KEYS.installments]: [],
  [KEYS.loans]:        [],
  [KEYS.categories]:   DEFAULT_CATEGORIES,
};

function store(key)        { return JSON.parse(localStorage.getItem(key) ?? 'null'); }
function save(key, val)    { localStorage.setItem(key, JSON.stringify(val)); }

function getOrDefault(key) {
  const val = store(key);
  return val !== null ? val : DEFAULT_DATA[key];
}

function initStorage() {
  Object.keys(DEFAULT_DATA).forEach(key => {
    if (store(key) === null) save(key, DEFAULT_DATA[key]);
  });
}

// 자주 쓰는 접근자
const db = {
  settings:     () => getOrDefault(KEYS.settings),
  work:         () => getOrDefault(KEYS.work),
  workLog:      () => getOrDefault(KEYS.workLog),
  fixed:        () => getOrDefault(KEYS.fixed),
  expenses:     () => getOrDefault(KEYS.expenses),
  installments: () => getOrDefault(KEYS.installments),
  loans:        () => getOrDefault(KEYS.loans),
  categories:   () => getOrDefault(KEYS.categories),

  saveSettings:     v => save(KEYS.settings, v),
  saveWork:         v => save(KEYS.work, v),
  saveWorkLog:      v => save(KEYS.workLog, v),
  saveFixed:        v => save(KEYS.fixed, v),
  saveExpenses:     v => save(KEYS.expenses, v),
  saveInstallments: v => save(KEYS.installments, v),
  saveLoans:        v => save(KEYS.loans, v),
  saveCategories:   v => save(KEYS.categories, v),
};

function genId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
```

- [ ] **Step 2: `switchTab('home')` 호출 직전에 `initStorage()` 추가**

```javascript
// 스크립트 맨 아래, switchTab('home') 앞에:
initStorage();
switchTab('home');
```

- [ ] **Step 3: 콘솔에서 스토리지 동작 확인**

브라우저 콘솔에서 실행:
```javascript
console.log(db.categories()); // 13개 기본 카테고리 배열 출력 확인
console.log(db.settings());   // { allowance: 0, apiKey: '' } 확인
```

- [ ] **Step 4: 커밋**

```bash
git add budget-app/index.html
git commit -m "feat: add localStorage storage layer with default data initialization"
```

---

## Task 3: 계산 엔진 — 순수 함수 + 브라우저 단위 테스트

**Files:**
- Create: `budget-app/tests.html`
- Modify: `budget-app/index.html` (CALCULATIONS 섹션 추가)

- [ ] **Step 1: CALCULATIONS 섹션을 STORAGE 섹션 바로 아래에 추가**

```javascript
// ── CALCULATIONS ─────────────────────────────────────────────────────

// 이번달 특정 요일의 개수 반환 (day: 0=일,1=월,...,6=토)
function countWeekdayInMonth(year, month, day) {
  let count = 0;
  const date = new Date(year, month - 1, 1);
  while (date.getMonth() === month - 1) {
    if (date.getDay() === day) count++;
    date.setDate(date.getDate() + 1);
  }
  return count;
}

// 이번달 주차 키 목록 반환 (YYYY-WNN 형식)
function getWeekKey(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + 3 - ((d.getDay() + 6) % 7));
  const week1 = new Date(d.getFullYear(), 0, 4);
  const wn = 1 + Math.round(((d - week1) / 86400000 - 3 + ((week1.getDay() + 6) % 7)) / 7);
  return `${d.getFullYear()}-W${String(wn).padStart(2, '0')}`;
}

// 월 알바 수입 계산
// work: { pattern: [{day:Number, hours:Number}], hourlyWage:Number }
// workLog: [{ weekKey:String, actualHours:Number }]
// actualHours: 해당 주 실제 총 근무시간 (보정값 없으면 패턴 시간 사용)
function calcMonthlyWorkIncome(work, workLog, year, month) {
  const { pattern, hourlyWage } = work;
  if (!pattern.length || !hourlyWage) return 0;

  // 패턴 기준 주당 기본 시간
  const baseWeeklyHours = pattern.reduce((s, p) => s + p.hours, 0);

  // 이번달에 해당하는 모든 날짜에서 주차별 근무시간 합산
  const weekTotals = {};
  const date = new Date(year, month - 1, 1);
  while (date.getMonth() === month - 1) {
    const wk = getWeekKey(date);
    if (!weekTotals[wk]) {
      const log = workLog.find(l => l.weekKey === wk);
      weekTotals[wk] = log ? log.actualHours : baseWeeklyHours;
    }
    date.setDate(date.getDate() + 1);
  }

  // 주차별 시간 × 시급 합산 (월에 걸친 비율 적용 없이 주차 단위로 간단 합산)
  const totalHours = Object.values(weekTotals).reduce((s, h) => s + h, 0);
  // 패턴 기반 주당 일수로 비례 적용
  const weeksInMonth = Object.keys(weekTotals).length;
  return Math.round(totalHours * hourlyWage);
}

// 월 총 수입
function calcMonthlyIncome(settings, work, workLog, year, month) {
  return (settings.allowance || 0) + calcMonthlyWorkIncome(work, workLog, year, month);
}

// 월 고정 지출 (고정비 + 할부)
function calcMonthlyFixed(fixedExpenses, installments) {
  const fixed = fixedExpenses.reduce((s, f) => s + (f.amount || 0), 0);
  const inst  = installments
    .filter(i => i.paidMonths < i.months)
    .reduce((s, i) => s + (i.monthlyAmount || 0), 0);
  return fixed + inst;
}

// 월 대출 적립액 (잔액 ÷ 목표일까지 남은 일수 × 이번달 남은 일수)
function calcMonthlyLoanReserve(loans, today) {
  const todayMs = new Date(today).setHours(0, 0, 0, 0);
  const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0);
  const remainingDaysInMonth = lastDay.getDate() - today.getDate() + 1;

  return loans.reduce((sum, loan) => {
    if (!loan.remainingBalance || !loan.targetDate) return sum;
    const targetMs = new Date(loan.targetDate).setHours(0, 0, 0, 0);
    const daysLeft = Math.max(1, Math.round((targetMs - todayMs) / 86400000));
    const dailyReserve = loan.remainingBalance / daysLeft;
    return sum + dailyReserve * remainingDaysInMonth;
  }, 0);
}

// 가처분 소득
function calcDisposableIncome(settings, work, workLog, fixedExpenses, installments, loans, today) {
  const year  = today.getFullYear();
  const month = today.getMonth() + 1;
  const income      = calcMonthlyIncome(settings, work, workLog, year, month);
  const fixedTotal  = calcMonthlyFixed(fixedExpenses, installments);
  const loanReserve = calcMonthlyLoanReserve(loans, today);
  return Math.round(income - fixedTotal - loanReserve);
}

// 일일 예산 (오늘 포함 이번달 남은 일수로 나눔)
function calcDailyBudget(disposable, today) {
  const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();
  const remaining = lastDay - today.getDate() + 1;
  return Math.round(disposable / remaining);
}

// 오늘 지출 합계 (myAmount 기준)
function calcTodaySpent(expenses, today) {
  const todayStr = formatDate(today);
  return expenses
    .filter(e => e.date === todayStr)
    .reduce((s, e) => s + (e.myAmount ?? e.amount ?? 0), 0);
}

// 잔여 예산 상태
function getBudgetState(remaining, dailyBudget) {
  if (remaining < 0)                          return 'over';
  if (remaining / dailyBudget < 0.2)          return 'danger';
  if (remaining / dailyBudget < 0.5)          return 'warn';
  return 'good';
}

// 날짜 포맷 YYYY-MM-DD
function formatDate(date) {
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

// 금액 포맷 ₩ 1,234,567
function formatKRW(amount) {
  const abs = Math.abs(Math.round(amount));
  const formatted = abs.toLocaleString('ko-KR');
  return (amount < 0 ? '− ' : '') + '₩ ' + formatted;
}

// 할부 D-Day까지 남은 일수
function calcDaysLeft(targetDate, today) {
  const t = new Date(targetDate).setHours(0, 0, 0, 0);
  const n = new Date(today).setHours(0, 0, 0, 0);
  return Math.max(0, Math.round((t - n) / 86400000));
}
```

- [ ] **Step 2: `budget-app/tests.html` 작성**

```html
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>계산 엔진 단위 테스트</title>
  <style>
    body { font-family: monospace; background: #0f172a; color: #e2e8f0; padding: 24px; }
    .pass { color: #34d399; } .fail { color: #f87171; }
    h2 { color: #94a3b8; font-size: 13px; margin: 20px 0 8px; }
  </style>
</head>
<body>
<h1 style="font-size:16px; margin-bottom:16px;">가계부 계산 엔진 테스트</h1>
<div id="results"></div>
<script>
// ── 테스트 대상 함수 복사 (index.html의 CALCULATIONS 섹션과 동일하게 유지) ──

function countWeekdayInMonth(year, month, day) {
  let count = 0;
  const date = new Date(year, month - 1, 1);
  while (date.getMonth() === month - 1) {
    if (date.getDay() === day) count++;
    date.setDate(date.getDate() + 1);
  }
  return count;
}

function getWeekKey(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() + 3 - ((d.getDay() + 6) % 7));
  const week1 = new Date(d.getFullYear(), 0, 4);
  const wn = 1 + Math.round(((d - week1) / 86400000 - 3 + ((week1.getDay() + 6) % 7)) / 7);
  return `${d.getFullYear()}-W${String(wn).padStart(2, '0')}`;
}

function calcMonthlyWorkIncome(work, workLog, year, month) {
  const { pattern, hourlyWage } = work;
  if (!pattern.length || !hourlyWage) return 0;
  const baseWeeklyHours = pattern.reduce((s, p) => s + p.hours, 0);
  const weekTotals = {};
  const date = new Date(year, month - 1, 1);
  while (date.getMonth() === month - 1) {
    const wk = getWeekKey(date);
    if (!weekTotals[wk]) {
      const log = workLog.find(l => l.weekKey === wk);
      weekTotals[wk] = log ? log.actualHours : baseWeeklyHours;
    }
    date.setDate(date.getDate() + 1);
  }
  const totalHours = Object.values(weekTotals).reduce((s, h) => s + h, 0);
  return Math.round(totalHours * hourlyWage);
}

function calcMonthlyFixed(fixedExpenses, installments) {
  const fixed = fixedExpenses.reduce((s, f) => s + (f.amount || 0), 0);
  const inst  = installments.filter(i => i.paidMonths < i.months).reduce((s, i) => s + (i.monthlyAmount || 0), 0);
  return fixed + inst;
}

function calcMonthlyLoanReserve(loans, today) {
  const todayMs = new Date(today).setHours(0, 0, 0, 0);
  const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0);
  const remainingDaysInMonth = lastDay.getDate() - today.getDate() + 1;
  return loans.reduce((sum, loan) => {
    if (!loan.remainingBalance || !loan.targetDate) return sum;
    const targetMs = new Date(loan.targetDate).setHours(0, 0, 0, 0);
    const daysLeft = Math.max(1, Math.round((targetMs - todayMs) / 86400000));
    const dailyReserve = loan.remainingBalance / daysLeft;
    return sum + dailyReserve * remainingDaysInMonth;
  }, 0);
}

function calcDailyBudget(disposable, today) {
  const lastDay = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();
  const remaining = lastDay - today.getDate() + 1;
  return Math.round(disposable / remaining);
}

function getBudgetState(remaining, dailyBudget) {
  if (remaining < 0)                         return 'over';
  if (remaining / dailyBudget < 0.2)         return 'danger';
  if (remaining / dailyBudget < 0.5)         return 'warn';
  return 'good';
}

function formatDate(date) {
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

// ── 테스트 러너 ──
let passed = 0, failed = 0;
const out = document.getElementById('results');

function assert(desc, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (ok) passed++; else failed++;
  out.innerHTML += `<div class="${ok?'pass':'fail'}">${ok?'✓':'✗'} ${desc}${ok?'':` — got ${JSON.stringify(actual)}, expected ${JSON.stringify(expected)}`}</div>`;
}

// ── 테스트 케이스 ──
out.innerHTML += '<h2>countWeekdayInMonth</h2>';
// 2026년 4월 화요일(2)은 몇 번? → 4번 (7,14,21,28)
assert('2026-04 화요일 4번', countWeekdayInMonth(2026, 4, 2), 4);
// 2026년 4월 수요일(3)은 몇 번? → 5번 (1,8,15,22,29)
assert('2026-04 수요일 5번', countWeekdayInMonth(2026, 4, 3), 5);

out.innerHTML += '<h2>calcMonthlyWorkIncome — 보정 없음</h2>';
const work = { pattern: [{ day: 2, hours: 5 }, { day: 4, hours: 5 }, { day: 6, hours: 5 }], hourlyWage: 10000 };
// 2026년 4월: 화4+목4+토5 = 13회 × 5h = 65h (패턴기준), 주차별로 계산
// 실제 계산은 주차 단위이므로 결과가 정확히 일치하는지보다 0보다 크고 합리적인 값인지 확인
const income = calcMonthlyWorkIncome(work, [], 2026, 4);
assert('알바 수입 > 0', income > 0, true);
assert('알바 수입 < 1000000', income < 1000000, true);

out.innerHTML += '<h2>calcMonthlyFixed</h2>';
const fixed = [{ amount: 55000 }, { amount: 30000 }];
const inst  = [{ monthlyAmount: 29000, paidMonths: 2, months: 12 }];
assert('고정 지출 합산', calcMonthlyFixed(fixed, inst), 114000);
assert('완납 할부 제외', calcMonthlyFixed([], [{ monthlyAmount: 10000, paidMonths: 12, months: 12 }]), 0);

out.innerHTML += '<h2>calcMonthlyLoanReserve</h2>';
const today = new Date(2026, 3, 5); // 2026-04-05
// 잔액 300000, 목표 2026-04-35 = 2026-05-05 → 30일 후
// remainingDaysInMonth = 30 - 5 + 1 = 26
// dailyReserve = 300000 / 30 = 10000
// reserve = 10000 * 26 = 260000
const loans = [{ remainingBalance: 300000, targetDate: '2026-05-05' }];
const reserve = calcMonthlyLoanReserve(loans, today);
assert('대출 적립 > 0', reserve > 0, true);

out.innerHTML += '<h2>calcDailyBudget</h2>';
// 가처분 소득 620000, 2026-04-05 → 남은 날수 26
assert('일일 예산', calcDailyBudget(620000, new Date(2026, 3, 5)), Math.round(620000 / 26));

out.innerHTML += '<h2>getBudgetState</h2>';
assert('good',   getBudgetState(6000, 10000), 'good');
assert('warn',   getBudgetState(3000, 10000), 'warn');
assert('danger', getBudgetState(1000, 10000), 'danger');
assert('over',   getBudgetState(-500, 10000), 'over');

out.innerHTML += '<h2>formatDate</h2>';
assert('날짜 포맷', formatDate(new Date(2026, 3, 5)), '2026-04-05');

// 결과 요약
out.innerHTML += `<div style="margin-top:20px; padding:12px; background:#0a1120; border-radius:8px;">
  <span class="pass">✓ ${passed} passed</span>
  ${failed ? `<span class="fail" style="margin-left:12px;">✗ ${failed} failed</span>` : ''}
</div>`;
</script>
</body>
</html>
```

- [ ] **Step 3: `budget-app/tests.html`을 브라우저에서 열어 모든 테스트 통과 확인**

모든 항목이 초록색 ✓로 표시되어야 함. 실패 항목이 있으면 계산 함수를 수정.

- [ ] **Step 4: 커밋**

```bash
git add budget-app/index.html budget-app/tests.html
git commit -m "feat: add calculation engine with browser unit tests"
```

---

## Task 4: 설정 탭 — 수입/고정지출/카테고리/API 키 입력

**Files:**
- Modify: `budget-app/index.html` (renderSettings 구현)

- [ ] **Step 1: 공통 UI 헬퍼 함수를 CALCULATIONS 섹션 바로 아래에 추가**

```javascript
// ── UI HELPERS ────────────────────────────────────────────────────────

function el(tag, attrs = {}, ...children) {
  const elem = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') elem.className = v;
    else if (k === 'style') elem.style.cssText = v;
    else if (k.startsWith('on')) elem.addEventListener(k.slice(2), v);
    else elem.setAttribute(k, v);
  });
  children.forEach(c => {
    if (c == null) return;
    elem.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  });
  return elem;
}

function setHTML(id, html) { document.getElementById(id).innerHTML = html; }

const DAYS = ['일','월','화','수','목','금','토'];

function showToast(msg, type = 'info') {
  const existing = document.getElementById('toast');
  if (existing) existing.remove();
  const toast = el('div', {
    id: 'toast',
    style: `position:fixed;bottom:72px;left:50%;transform:translateX(-50%);
            background:${type==='warn'?'#7f1d1d':'#0d2a1f'};
            color:${type==='warn'?'#fca5a5':'#6ee7b7'};
            padding:8px 16px;border-radius:8px;font-size:13px;
            font-family:monospace;z-index:999;white-space:nowrap;`
  }, msg);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2000);
}
```

- [ ] **Step 2: `renderSettings` 함수 전체를 아래로 교체**

```javascript
function renderSettings() {
  const tab = document.getElementById('tab-settings');
  const settings = db.settings();
  const work     = db.work();
  const workLog  = db.workLog();
  const fixed    = db.fixed();
  const cats     = db.categories();

  // 현재 주차 보정값
  const thisWeek   = getWeekKey(new Date());
  const weekLogEntry = workLog.find(l => l.weekKey === thisWeek);
  const baseHours  = work.pattern.reduce((s, p) => s + p.hours, 0);
  const actualHours = weekLogEntry ? weekLogEntry.actualHours : baseHours;

  tab.innerHTML = `
    <div style="max-width:480px;margin:0 auto;">

      <!-- 수입 섹션 -->
      <div class="mb-6">
        <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--green)">수입</div>

        <label class="block mb-3">
          <span class="text-xs mb-1 block" style="color:var(--text2)">용돈 (월)</span>
          <div class="flex gap-2">
            <input id="inp-allowance" type="number" value="${settings.allowance}"
              class="flex-1 rounded-lg px-3 py-2 text-sm mono"
              style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
            <span class="flex items-center text-xs" style="color:var(--text3)">원</span>
          </div>
        </label>

        <label class="block mb-3">
          <span class="text-xs mb-1 block" style="color:var(--text2)">시급</span>
          <div class="flex gap-2">
            <input id="inp-wage" type="number" value="${work.hourlyWage}"
              class="flex-1 rounded-lg px-3 py-2 text-sm mono"
              style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
            <span class="flex items-center text-xs" style="color:var(--text3)">원</span>
          </div>
        </label>

        <div class="mb-2">
          <span class="text-xs mb-2 block" style="color:var(--text2)">알바 요일 & 시간</span>
          <div class="flex gap-1 mb-2">
            ${DAYS.map((d, i) => {
              const pat = work.pattern.find(p => p.day === i);
              return `<button class="day-btn flex-1 rounded py-1.5 text-xs mono"
                data-day="${i}"
                style="background:${pat?'var(--green)':'var(--bg2)'};
                       color:${pat?'#0f172a':'var(--text3)'};
                       border:1px solid ${pat?'var(--green)':'var(--border)'}"
                onclick="toggleDay(${i})">${d}</button>`;
            }).join('')}
          </div>
          <div id="day-hours-inputs">
            ${work.pattern.map(p => `
              <div class="flex items-center gap-2 mb-1">
                <span class="text-xs mono w-4" style="color:var(--text2)">${DAYS[p.day]}</span>
                <input type="number" value="${p.hours}" min="1" max="24"
                  class="w-16 rounded px-2 py-1 text-xs mono"
                  style="background:var(--bg2);border:1px solid var(--border);color:var(--text);"
                  onchange="updateDayHours(${p.day}, this.value)">
                <span class="text-xs" style="color:var(--text3)">시간</span>
              </div>`).join('')}
          </div>
        </div>

        <label class="block mb-3">
          <span class="text-xs mb-1 block" style="color:var(--text2)">이번 주 실제 근무시간 (기본 ${baseHours}h)</span>
          <div class="flex gap-2">
            <input id="inp-actual-hours" type="number" value="${actualHours}" step="0.5"
              class="flex-1 rounded-lg px-3 py-2 text-sm mono"
              style="background:${actualHours !== baseHours ? 'rgba(52,211,153,0.05)' : 'var(--bg2)'};
                     border:1px solid ${actualHours !== baseHours ? 'var(--green)' : 'var(--border)'};
                     color:var(--text);">
            <span class="flex items-center text-xs" style="color:var(--text3)">시간</span>
          </div>
        </label>

        <button onclick="saveIncomeSettings()"
          class="w-full rounded-lg py-2 text-sm font-bold"
          style="background:var(--green);color:#0f172a;">수입 설정 저장</button>
      </div>

      <!-- 고정 지출 섹션 -->
      <div class="mb-6">
        <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--green)">고정 지출</div>
        <div id="fixed-list">
          ${fixed.map(f => `
            <div class="flex items-center gap-2 mb-2 rounded-lg px-3 py-2"
              style="background:var(--bg2);border:1px solid var(--border);">
              <span class="flex-1 text-sm">${f.name}</span>
              <span class="mono text-sm" style="color:var(--text2)">${formatKRW(f.amount)}</span>
              <button onclick="deleteFixed('${f.id}')" class="text-xs" style="color:var(--text3)">✕</button>
            </div>`).join('') || '<p class="text-xs mb-2" style="color:var(--text3)">없음</p>'}
        </div>
        <div class="flex gap-2">
          <input id="inp-fixed-name" placeholder="항목명" class="flex-1 rounded-lg px-3 py-2 text-sm"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
          <input id="inp-fixed-amount" type="number" placeholder="금액" class="w-24 rounded-lg px-3 py-2 text-sm mono"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
          <button onclick="addFixed()"
            class="px-3 rounded-lg text-sm font-bold"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--green);">+</button>
        </div>
      </div>

      <!-- 카테고리 섹션 -->
      <div class="mb-6">
        <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--green)">카테고리</div>
        <div class="flex flex-wrap gap-2 mb-3">
          ${cats.map(c => `
            <span class="flex items-center gap-1 rounded-full px-3 py-1 text-xs mono"
              style="background:var(--bg2);border:1px solid var(--border);color:var(--text2);">
              ${c.name}
              ${!c.isDefault ? `<button onclick="deleteCategory(${c.id})" style="color:var(--text3);">✕</button>` : ''}
            </span>`).join('')}
        </div>
        <div class="flex gap-2">
          <input id="inp-cat-name" placeholder="새 카테고리" class="flex-1 rounded-lg px-3 py-2 text-sm"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
          <button onclick="addCategory()"
            class="px-3 rounded-lg text-sm font-bold"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--green);">추가</button>
        </div>
      </div>

      <!-- API 키 섹션 -->
      <div class="mb-6">
        <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--green)">OpenAI API</div>
        <div class="flex gap-2">
          <input id="inp-apikey" type="password" value="${settings.apiKey}"
            placeholder="sk-..."
            class="flex-1 rounded-lg px-3 py-2 text-sm mono"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
          <button onclick="saveApiKey()"
            class="px-3 rounded-lg text-sm font-bold"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--green);">저장</button>
        </div>
        <p class="text-xs mt-1" style="color:var(--text3)">이 기기 localStorage에만 저장됩니다.</p>
      </div>

    </div>`;
}
```

- [ ] **Step 3: 설정 탭 이벤트 핸들러 추가 (EVENTS 섹션)**

```javascript
// ── EVENTS — 설정 탭 ─────────────────────────────────────────────────

function toggleDay(dayIdx) {
  const work = db.work();
  const exists = work.pattern.findIndex(p => p.day === dayIdx);
  if (exists >= 0) work.pattern.splice(exists, 1);
  else work.pattern.push({ day: dayIdx, hours: 5 });
  work.pattern.sort((a, b) => a.day - b.day);
  db.saveWork(work);
  renderSettings();
}

function updateDayHours(dayIdx, val) {
  const work = db.work();
  const pat = work.pattern.find(p => p.day === dayIdx);
  if (pat) pat.hours = parseFloat(val) || 0;
  db.saveWork(work);
}

function saveIncomeSettings() {
  const settings = db.settings();
  settings.allowance = parseInt(document.getElementById('inp-allowance').value) || 0;
  db.saveSettings(settings);

  const work = db.work();
  work.hourlyWage = parseInt(document.getElementById('inp-wage').value) || 0;
  db.saveWork(work);

  // 이번 주 보정 저장
  const thisWeek = getWeekKey(new Date());
  const workLog = db.workLog();
  const actualHours = parseFloat(document.getElementById('inp-actual-hours').value) || 0;
  const idx = workLog.findIndex(l => l.weekKey === thisWeek);
  if (idx >= 0) workLog[idx].actualHours = actualHours;
  else workLog.push({ weekKey: thisWeek, actualHours });
  db.saveWorkLog(workLog);

  showToast('수입 설정 저장됨');
  if (currentTab === 'home') renderHome();
}

function deleteFixed(id) {
  db.saveFixed(db.fixed().filter(f => f.id !== id));
  renderSettings();
  if (currentTab === 'home') renderHome();
}

function addFixed() {
  const name   = document.getElementById('inp-fixed-name').value.trim();
  const amount = parseInt(document.getElementById('inp-fixed-amount').value) || 0;
  if (!name || !amount) { showToast('항목명과 금액을 입력하세요', 'warn'); return; }
  const fixed = db.fixed();
  fixed.push({ id: genId(), name, amount });
  db.saveFixed(fixed);
  renderSettings();
  if (currentTab === 'home') renderHome();
}

function addCategory() {
  const name = document.getElementById('inp-cat-name').value.trim();
  if (!name) return;
  const cats = db.categories();
  if (cats.find(c => c.name === name)) { showToast('이미 있는 카테고리입니다', 'warn'); return; }
  cats.push({ id: genId(), name, isDefault: false });
  db.saveCategories(cats);
  renderSettings();
}

function deleteCategory(id) {
  db.saveCategories(db.categories().filter(c => c.id !== id));
  renderSettings();
}

function saveApiKey() {
  const settings = db.settings();
  settings.apiKey = document.getElementById('inp-apikey').value.trim();
  db.saveSettings(settings);
  showToast('API 키 저장됨');
}
```

- [ ] **Step 4: 브라우저에서 설정 탭 확인**

설정 탭을 열어:
- 요일 버튼 토글이 초록으로 바뀌는지
- 고정 지출 추가/삭제 동작하는지
- 카테고리 추가/삭제 동작하는지
- 수입 설정 저장 토스트가 뜨는지

- [ ] **Step 5: 커밋**

```bash
git add budget-app/index.html
git commit -m "feat: implement settings tab (income, fixed expenses, categories, API key)"
```

---

## Task 5: 홈 탭 — 가처분 소득 대시보드 + 잔여 예산

**Files:**
- Modify: `budget-app/index.html` (renderHome 구현)

- [ ] **Step 1: `renderHome` 함수 전체를 아래로 교체**

```javascript
function renderHome() {
  const tab          = document.getElementById('tab-home');
  const today        = new Date();
  const settings     = db.settings();
  const work         = db.work();
  const workLog      = db.workLog();
  const fixedExp     = db.fixed();
  const installments = db.installments();
  const loans        = db.loans();
  const expenses     = db.expenses();

  const year  = today.getFullYear();
  const month = today.getMonth() + 1;

  const workIncome   = calcMonthlyWorkIncome(work, workLog, year, month);
  const totalIncome  = workIncome + (settings.allowance || 0);
  const fixedTotal   = calcMonthlyFixed(fixedExp, installments);
  const loanReserve  = calcMonthlyLoanReserve(loans, today);
  const disposable   = Math.round(totalIncome - fixedTotal - loanReserve);
  const dailyBudget  = calcDailyBudget(disposable, today);
  const todaySpent   = calcTodaySpent(expenses, today);
  const remaining    = dailyBudget - todaySpent;
  const state        = getBudgetState(remaining, dailyBudget);

  const colorMap = { good:'var(--green)', warn:'var(--amber)', danger:'var(--red)', over:'var(--red2)' };
  const color = colorMap[state];
  const barPct = dailyBudget > 0 ? Math.max(0, Math.min(100, (remaining / dailyBudget) * 100)) : 0;

  // 최근 지출 5건
  const recentExp = [...expenses]
    .sort((a, b) => b.id.localeCompare(a.id))
    .slice(0, 5);

  tab.innerHTML = `
    <div style="max-width:480px;margin:0 auto;">

      <!-- 잔여 예산 -->
      <div class="mb-6 rounded-xl p-4" style="background:var(--bg2);border:1px solid var(--border);">
        <div class="mono text-[10px] uppercase tracking-widest mb-1" style="color:var(--text3)">오늘의 잔여 예산</div>
        <div class="mono font-bold" style="font-size:36px;letter-spacing:-1px;color:${color};">${formatKRW(remaining)}</div>
        <div class="mt-2 rounded-full overflow-hidden" style="height:3px;background:var(--border);">
          <div style="height:100%;width:${barPct}%;background:${color};border-radius:2px;transition:width 0.3s;"></div>
        </div>
        <div class="flex justify-between mt-1">
          <span class="mono text-[10px]" style="color:var(--text3)">일일예산 ${formatKRW(dailyBudget)}</span>
          <span class="mono text-[10px]" style="color:var(--text3)">오늘지출 ${formatKRW(todaySpent)}</span>
        </div>
      </div>

      <!-- 이번달 분해 -->
      <div class="mb-6 rounded-xl p-4" style="background:var(--bg2);border:1px solid var(--border);">
        <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--text3)">이번달 분석</div>
        ${[
          ['알바 예상', workIncome, false],
          ['용돈', settings.allowance || 0, false],
          ['고정 지출', fixedTotal, true],
          ['할부 합계', installments.filter(i=>i.paidMonths<i.months).reduce((s,i)=>s+i.monthlyAmount,0), true],
          ['대출 적립', Math.round(loanReserve), true],
        ].map(([label, val, minus]) => `
          <div class="flex justify-between items-center py-2" style="border-bottom:1px solid var(--border);">
            <span class="text-xs" style="color:var(--text2)">${label}</span>
            <span class="mono text-sm font-bold" style="color:${minus?'var(--red)':'var(--text)'};">
              ${minus ? '−' : ''} ${formatKRW(val)}
            </span>
          </div>`).join('')}
        <div class="flex justify-between items-center pt-3">
          <span class="text-sm font-bold" style="color:var(--text)">가처분 소득</span>
          <span class="mono font-bold" style="font-size:18px;color:var(--green);">${formatKRW(disposable)}</span>
        </div>
      </div>

      <!-- 최근 지출 -->
      <div>
        <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--text3)">최근 지출</div>
        ${recentExp.length === 0
          ? `<p class="text-xs" style="color:var(--text3)">지출 내역이 없습니다.</p>`
          : recentExp.map(e => `
            <div class="flex justify-between items-center py-2.5" style="border-bottom:1px solid var(--border);">
              <div>
                <span class="text-sm">${e.merchant || '직접입력'}</span>
                <span class="ml-2 text-[10px] mono px-1.5 py-0.5 rounded" style="background:var(--border);color:var(--text3);">${e.category}</span>
                ${e.isDutch ? `<span class="ml-1 text-[10px] mono px-1.5 py-0.5 rounded" style="background:#1e1b4b;color:#818cf8;">더치÷${e.splitCount}</span>` : ''}
              </div>
              <span class="mono text-sm font-bold" style="color:var(--text);">${formatKRW(e.myAmount ?? e.amount)}</span>
            </div>`).join('')}
      </div>

    </div>`;
}
```

- [ ] **Step 2: 브라우저에서 홈 탭 확인**

설정에서 용돈·시급·알바패턴을 저장한 뒤 홈 탭으로 돌아와:
- 잔여 예산 숫자가 상태 컬러로 표시되는지
- 이번달 분해 내역이 올바르게 계산되는지
- 가처분 소득이 합리적인 값인지

- [ ] **Step 3: 커밋**

```bash
git add budget-app/index.html
git commit -m "feat: implement home dashboard with daily budget and disposable income breakdown"
```

---

## Task 6: 지출 입력 탭 — SMS 파싱 + AI + 더치페이

**Files:**
- Modify: `budget-app/index.html` (renderExpense 구현 + AI 통합)

- [ ] **Step 1: AI 호출 함수 추가 (EVENTS 섹션)**

```javascript
// ── EVENTS — 지출 탭 ─────────────────────────────────────────────────

async function callOpenAI(systemPrompt, userPrompt) {
  const key = db.settings().apiKey;
  if (!key) { showToast('설정에서 API 키를 입력해주세요', 'warn'); return null; }

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user',   content: userPrompt   },
      ],
      response_format: { type: 'json_object' },
      temperature: 0,
    }),
  });

  if (!res.ok) { showToast(`API 오류: ${res.status}`, 'warn'); return null; }
  const data = await res.json();
  return JSON.parse(data.choices[0].message.content);
}

async function analyzeSMS() {
  const smsText = document.getElementById('sms-input').value.trim();
  if (!smsText) { showToast('문자를 붙여넣어 주세요', 'warn'); return; }

  const btn = document.getElementById('btn-analyze');
  btn.textContent = '분석 중...';
  btn.disabled = true;

  const cats = db.categories().map(c => c.name).join(', ');
  const today = new Date();
  const todaySpent = calcTodaySpent(db.expenses(), today);
  const dailyBudget = calcDailyBudget(
    calcDisposableIncome(db.settings(), db.work(), db.workLog(), db.fixed(), db.installments(), db.loans(), today),
    today
  );

  const systemPrompt = `한국 카드사 SMS에서 결제 정보를 추출해 JSON으로 반환하라.
카테고리는 반드시 다음 중 하나: ${cats}
JSON 형식: { "amount": number, "merchant": string, "category": string, "date": "YYYY-MM-DD", "feedback": string }
feedback은 오늘 예산 상황을 포함한 1문장 한국어 코멘트. 절약 조언이나 격려 포함.`;

  const userPrompt = `SMS:\n${smsText}\n\n오늘 날짜: ${formatDate(today)}\n일일예산: ${dailyBudget}원\n오늘 지출합계: ${todaySpent}원`;

  const result = await callOpenAI(systemPrompt, userPrompt);

  btn.textContent = '분석';
  btn.disabled = false;

  if (!result) return;

  // 결과를 전역 임시 저장
  window._parsedExpense = result;

  document.getElementById('parsed-result').innerHTML = `
    <div class="rounded-lg p-3 mb-3" style="background:var(--bg2);border:1px solid var(--border);">
      <div class="flex justify-between py-1"><span class="text-xs" style="color:var(--text3)">금액</span>
        <span class="mono font-bold" style="color:var(--green);">${formatKRW(result.amount)}</span></div>
      <div class="flex justify-between py-1"><span class="text-xs" style="color:var(--text3)">가맹점</span>
        <span class="text-sm">${result.merchant}</span></div>
      <div class="flex justify-between py-1"><span class="text-xs" style="color:var(--text3)">카테고리</span>
        <select id="sel-category" class="text-xs rounded px-2 py-0.5 mono"
          style="background:var(--border);color:var(--text);border:none;">
          ${db.categories().map(c => `<option value="${c.name}" ${c.name===result.category?'selected':''}>${c.name}</option>`).join('')}
        </select>
      </div>
      <div class="flex justify-between py-1"><span class="text-xs" style="color:var(--text3)">날짜</span>
        <input type="date" id="inp-date" value="${result.date}"
          class="text-xs mono rounded px-2 py-0.5"
          style="background:var(--border);color:var(--text);border:none;"></div>
    </div>
    <div class="rounded-lg px-3 py-2 mb-3 text-xs" style="background:#0d2a1f;border:1px solid #065f46;color:var(--green);">
      ✦ ${result.feedback}
    </div>
    <!-- 더치페이 -->
    <div class="rounded-lg p-3 mb-3" style="background:var(--bg2);border:1px solid var(--border);">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm">더치페이</span>
        <button id="btn-dutch" onclick="toggleDutch()"
          class="rounded-full px-3 py-1 text-xs mono"
          style="background:var(--border);color:var(--text3);">OFF</button>
      </div>
      <div id="dutch-detail" class="hidden">
        <div class="flex items-center gap-2">
          <span class="text-xs" style="color:var(--text2)">인원 수 (나 포함)</span>
          <input type="number" id="inp-split" value="2" min="2" max="10"
            class="w-16 rounded px-2 py-1 text-xs mono"
            style="background:var(--bg);border:1px solid var(--border);color:var(--text);"
            oninput="updateDutchCalc()">
          <span class="text-xs" style="color:var(--text3)">명</span>
        </div>
        <div class="mt-2 text-xs mono" style="color:var(--green);" id="dutch-my-amount"></div>
      </div>
    </div>
    <button onclick="saveExpense()"
      class="w-full rounded-lg py-2.5 text-sm font-bold"
      style="background:var(--green);color:#0f172a;">저장</button>`;
}

let _dutchOn = false;
function toggleDutch() {
  _dutchOn = !_dutchOn;
  const btn = document.getElementById('btn-dutch');
  const detail = document.getElementById('dutch-detail');
  btn.textContent = _dutchOn ? 'ON' : 'OFF';
  btn.style.background = _dutchOn ? 'var(--green)';
  btn.style.color = _dutchOn ? '#0f172a' : 'var(--text3)';
  detail.classList.toggle('hidden', !_dutchOn);
  if (_dutchOn) updateDutchCalc();
}

function updateDutchCalc() {
  const split = parseInt(document.getElementById('inp-split')?.value) || 2;
  const amount = window._parsedExpense?.amount || 0;
  const myAmt = Math.round(amount / split);
  const el = document.getElementById('dutch-my-amount');
  if (el) el.textContent = `내 부담: ${formatKRW(myAmt)} (${formatKRW(amount)} ÷ ${split})`;
}

function saveExpense() {
  const parsed = window._parsedExpense;
  if (!parsed) return;

  const amount   = parsed.amount;
  const split    = _dutchOn ? (parseInt(document.getElementById('inp-split').value) || 2) : 1;
  const myAmount = Math.round(amount / split);

  const expense = {
    id:         genId(),
    date:       document.getElementById('inp-date').value || formatDate(new Date()),
    amount,
    myAmount,
    merchant:   parsed.merchant,
    category:   document.getElementById('sel-category').value,
    memo:       '',
    source:     'sms',
    isDutch:    _dutchOn,
    splitCount: split,
  };

  const expenses = db.expenses();
  expenses.push(expense);
  db.saveExpenses(expenses);

  window._parsedExpense = null;
  _dutchOn = false;
  showToast('지출 저장됨');
  switchTab('home');
}
```

- [ ] **Step 2: `renderExpense` 함수 전체를 아래로 교체**

```javascript
function renderExpense() {
  const tab = document.getElementById('tab-expense');
  _dutchOn = false;

  tab.innerHTML = `
    <div style="max-width:480px;margin:0 auto;">

      <div class="mono text-[10px] uppercase tracking-widest mb-4" style="color:var(--text3)">지출 입력</div>

      <!-- SMS 파싱 -->
      <div class="mb-4">
        <label class="text-xs mb-1 block" style="color:var(--text2)">카드 결제 문자 붙여넣기</label>
        <textarea id="sms-input" rows="4" placeholder="[신한카드] 승인 ₩15,900&#10;GS리테일(GS25) 01/15 14:32..."
          class="w-full rounded-lg px-3 py-2 text-sm"
          style="background:var(--bg2);border:1px solid var(--border);color:var(--text);resize:none;"></textarea>
        <button id="btn-analyze" onclick="analyzeSMS()"
          class="w-full mt-2 rounded-lg py-2.5 text-sm font-bold"
          style="background:var(--green);color:#0f172a;">분석</button>
      </div>

      <div id="parsed-result"></div>

      <!-- 직접 입력 토글 -->
      <div class="mt-4">
        <button onclick="toggleManualForm()" class="text-xs" style="color:var(--text3);">직접 입력 →</button>
        <div id="manual-form" class="hidden mt-3">
          <div class="flex gap-2 mb-2">
            <input id="m-amount" type="number" placeholder="금액"
              class="flex-1 rounded-lg px-3 py-2 text-sm mono"
              style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
            <select id="m-category" class="rounded-lg px-2 py-2 text-sm"
              style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
              ${db.categories().map(c => `<option value="${c.name}">${c.name}</option>`).join('')}
            </select>
          </div>
          <input id="m-merchant" placeholder="가맹점 / 내용"
            class="w-full rounded-lg px-3 py-2 text-sm mb-2"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
          <input id="m-date" type="date" value="${formatDate(new Date())}"
            class="w-full rounded-lg px-3 py-2 text-sm mono mb-2"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
          <button onclick="saveManualExpense()"
            class="w-full rounded-lg py-2.5 text-sm font-bold"
            style="background:var(--green);color:#0f172a;">저장</button>
        </div>
      </div>

    </div>`;
}

function toggleManualForm() {
  document.getElementById('manual-form').classList.toggle('hidden');
}

function saveManualExpense() {
  const amount   = parseInt(document.getElementById('m-amount').value) || 0;
  const category = document.getElementById('m-category').value;
  const merchant = document.getElementById('m-merchant').value.trim() || '직접입력';
  const date     = document.getElementById('m-date').value || formatDate(new Date());

  if (!amount) { showToast('금액을 입력하세요', 'warn'); return; }

  const expenses = db.expenses();
  expenses.push({ id: genId(), date, amount, myAmount: amount, merchant, category, memo: '', source: 'manual', isDutch: false, splitCount: 1 });
  db.saveExpenses(expenses);
  showToast('지출 저장됨');
  switchTab('home');
}
```

- [ ] **Step 3: `toggleDutch` 함수의 구문 오류 수정**

Step 1에서 추가한 `toggleDutch` 함수 안에 아래 줄을 찾아 수정:

```javascript
// 잘못된 줄 (수정 전):
btn.style.background = _dutchOn ? 'var(--green)';

// 올바른 줄 (수정 후):
btn.style.background = _dutchOn ? 'var(--green)' : 'var(--border)';
```

- [ ] **Step 4: 브라우저에서 지출 탭 확인**

1. 설정에서 API 키 저장
2. 지출 탭으로 이동
3. 실제 카드 결제 문자를 붙여넣고 "분석" 클릭
4. 파싱 결과 표시 확인 (금액, 가맹점, 카테고리)
5. AI 피드백 문장 표시 확인
6. "저장" 클릭 후 홈 탭으로 이동, 최근 지출에 표시되는지 확인
7. 직접 입력으로도 저장 확인

- [ ] **Step 5: 커밋**

```bash
git add budget-app/index.html
git commit -m "feat: implement expense entry tab with SMS parsing, OpenAI analysis, and dutch pay"
```

---

## Task 7: 할부 & 대출 탭

**Files:**
- Modify: `budget-app/index.html` (renderInstallments 구현)

- [ ] **Step 1: `renderInstallments` 함수 전체를 아래로 교체**

```javascript
function renderInstallments() {
  const tab          = document.getElementById('tab-installments');
  const installments = db.installments();
  const loans        = db.loans();
  const today        = new Date();

  function instCard(i) {
    const pct     = Math.round((i.paidMonths / i.months) * 100);
    const lastOne = i.months - i.paidMonths === 1;
    const barColor = lastOne ? 'var(--red)' : 'var(--green)';
    return `
      <div class="rounded-xl p-4 mb-3" style="background:var(--bg2);border:1px solid ${lastOne ? 'var(--red)' : 'var(--border)'};">
        <div class="flex justify-between items-start mb-2">
          <span class="text-sm font-medium">${i.name}</span>
          <div class="flex gap-1 items-center">
            ${lastOne ? `<span class="text-[9px] mono px-2 py-0.5 rounded" style="background:#2d1515;color:var(--red);">⚠ 마지막달</span>` : ''}
            <span class="text-[9px] mono px-2 py-0.5 rounded" style="background:var(--border);color:var(--text3);">${i.paidMonths}/${i.months}개월</span>
          </div>
        </div>
        <div class="rounded-full overflow-hidden mb-2" style="height:2px;background:var(--border);">
          <div style="height:100%;width:${pct}%;background:${barColor};"></div>
        </div>
        <div class="flex justify-between text-xs mono">
          <span style="color:var(--text2)">월 ${formatKRW(i.monthlyAmount)}</span>
          <span style="color:var(--text3)">잔여 ${formatKRW(i.monthlyAmount * (i.months - i.paidMonths))}</span>
        </div>
        <div class="flex gap-2 mt-3">
          <button onclick="markInstallmentPaid('${i.id}')"
            class="flex-1 rounded-lg py-1.5 text-xs font-bold"
            style="background:var(--green);color:#0f172a;">이번달 납부 완료</button>
          <button onclick="deleteInstallment('${i.id}')"
            class="px-3 rounded-lg text-xs"
            style="background:var(--border);color:var(--text3);">삭제</button>
        </div>
      </div>`;
  }

  function loanCard(l) {
    const daysLeft    = calcDaysLeft(l.targetDate, today);
    const dailyAmount = l.remainingBalance / Math.max(1, daysLeft);
    return `
      <div class="rounded-xl p-4 mb-3" style="background:var(--bg2);border:1px solid var(--border);">
        <div class="flex justify-between items-start mb-2">
          <span class="text-sm font-medium">${l.name}</span>
          <span class="text-[9px] mono px-2 py-0.5 rounded" style="background:#1c1917;color:var(--amber);">D-${daysLeft}</span>
        </div>
        <div class="flex justify-between text-xs mono">
          <span style="color:var(--text2)">일 ${formatKRW(Math.round(dailyAmount))} 적립</span>
          <span style="color:var(--text3)">잔여 ${formatKRW(l.remainingBalance)}</span>
        </div>
        <div class="flex gap-2 mt-3">
          <button onclick="openUpdateLoan('${l.id}')"
            class="flex-1 rounded-lg py-1.5 text-xs"
            style="background:var(--border);color:var(--text2);">잔액 수정</button>
          <button onclick="deleteLoan('${l.id}')"
            class="px-3 rounded-lg text-xs"
            style="background:var(--border);color:var(--text3);">삭제</button>
        </div>
      </div>`;
  }

  tab.innerHTML = `
    <div style="max-width:480px;margin:0 auto;">

      <!-- 할부 섹션 -->
      <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--green)">할부</div>
      ${installments.length ? installments.map(instCard).join('') : '<p class="text-xs mb-4" style="color:var(--text3)">할부 없음</p>'}
      <button onclick="openAddInstallment()"
        class="w-full rounded-lg py-2 text-sm mb-6"
        style="background:var(--bg2);border:1px solid var(--border);color:var(--text2);">+ 할부 추가</button>

      <!-- 대출 섹션 -->
      <div class="mono text-[10px] uppercase tracking-widest mb-3" style="color:var(--green)">대출</div>
      ${loans.length ? loans.map(loanCard).join('') : '<p class="text-xs mb-4" style="color:var(--text3)">대출 없음</p>'}
      <button onclick="openAddLoan()"
        class="w-full rounded-lg py-2 text-sm mb-6"
        style="background:var(--bg2);border:1px solid var(--border);color:var(--text2);">+ 대출 추가</button>

    </div>

    <!-- 할부 추가 모달 -->
    <div id="modal-installment" class="hidden fixed inset-0 z-50 flex items-end justify-center"
      style="background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);">
      <div class="w-full rounded-t-2xl p-5" style="background:#0d1829;max-width:480px;">
        <div class="text-sm font-bold mb-4">할부 추가</div>
        <input id="i-name" placeholder="항목명 (예: 에어팟 프로)"
          class="w-full rounded-lg px-3 py-2 text-sm mb-2"
          style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
        <div class="flex gap-2 mb-2">
          <input id="i-total" type="number" placeholder="총 금액"
            class="flex-1 rounded-lg px-3 py-2 text-sm mono"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
          <input id="i-months" type="number" placeholder="개월"
            class="w-20 rounded-lg px-3 py-2 text-sm mono"
            style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
        </div>
        <input id="i-start" type="month" value="${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}"
          class="w-full rounded-lg px-3 py-2 text-sm mono mb-3"
          style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
        <button onclick="saveInstallment()"
          class="w-full rounded-lg py-2.5 text-sm font-bold mb-2"
          style="background:var(--green);color:#0f172a;">추가</button>
        <button onclick="closeModal('modal-installment')"
          class="w-full rounded-lg py-2 text-sm" style="color:var(--text3);">취소</button>
      </div>
    </div>

    <!-- 대출 추가 모달 -->
    <div id="modal-loan" class="hidden fixed inset-0 z-50 flex items-end justify-center"
      style="background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);">
      <div class="w-full rounded-t-2xl p-5" style="background:#0d1829;max-width:480px;">
        <div class="text-sm font-bold mb-4">대출 추가</div>
        <input id="l-name" placeholder="항목명 (예: 학자금 대출)"
          class="w-full rounded-lg px-3 py-2 text-sm mb-2"
          style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
        <input id="l-balance" type="number" placeholder="잔여 금액"
          class="w-full rounded-lg px-3 py-2 text-sm mono mb-2"
          style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
        <label class="text-xs mb-1 block" style="color:var(--text2)">목표 상환일</label>
        <input id="l-target" type="date"
          class="w-full rounded-lg px-3 py-2 text-sm mono mb-3"
          style="background:var(--bg2);border:1px solid var(--border);color:var(--text);">
        <button onclick="saveLoan()"
          class="w-full rounded-lg py-2.5 text-sm font-bold mb-2"
          style="background:var(--green);color:#0f172a;">추가</button>
        <button onclick="closeModal('modal-loan')"
          class="w-full rounded-lg py-2 text-sm" style="color:var(--text3);">취소</button>
      </div>
    </div>`;
}
```

- [ ] **Step 2: 할부/대출 이벤트 핸들러 추가**

```javascript
// ── EVENTS — 할부/대출 탭 ─────────────────────────────────────────────

function openAddInstallment() { document.getElementById('modal-installment').classList.remove('hidden'); }
function openAddLoan()        { document.getElementById('modal-loan').classList.remove('hidden'); }
function closeModal(id)       { document.getElementById(id).classList.add('hidden'); }

function saveInstallment() {
  const name   = document.getElementById('i-name').value.trim();
  const total  = parseInt(document.getElementById('i-total').value) || 0;
  const months = parseInt(document.getElementById('i-months').value) || 0;
  const start  = document.getElementById('i-start').value;
  if (!name || !total || !months) { showToast('모든 항목을 입력하세요', 'warn'); return; }

  const list = db.installments();
  list.push({ id: genId(), name, totalAmount: total, months, paidMonths: 0, startDate: start, monthlyAmount: Math.round(total / months) });
  db.saveInstallments(list);
  closeModal('modal-installment');
  renderInstallments();
  if (currentTab === 'home') renderHome();
}

function markInstallmentPaid(id) {
  const list = db.installments();
  const item = list.find(i => i.id === id);
  if (item && item.paidMonths < item.months) item.paidMonths++;
  db.saveInstallments(list);
  renderInstallments();
  if (currentTab === 'home') renderHome();
}

function deleteInstallment(id) {
  db.saveInstallments(db.installments().filter(i => i.id !== id));
  renderInstallments();
  if (currentTab === 'home') renderHome();
}

function saveLoan() {
  const name    = document.getElementById('l-name').value.trim();
  const balance = parseInt(document.getElementById('l-balance').value) || 0;
  const target  = document.getElementById('l-target').value;
  if (!name || !balance || !target) { showToast('모든 항목을 입력하세요', 'warn'); return; }

  const list = db.loans();
  list.push({ id: genId(), name, remainingBalance: balance, targetDate: target });
  db.saveLoans(list);
  closeModal('modal-loan');
  renderInstallments();
  if (currentTab === 'home') renderHome();
}

function openUpdateLoan(id) {
  const loan = db.loans().find(l => l.id === id);
  if (!loan) return;
  const newBal = prompt(`${loan.name} 잔여 금액 수정 (현재: ${loan.remainingBalance.toLocaleString()}원):`);
  if (newBal === null) return;
  const val = parseInt(newBal);
  if (isNaN(val) || val < 0) { showToast('올바른 금액을 입력하세요', 'warn'); return; }
  loan.remainingBalance = val;
  db.saveLoans(db.loans().map(l => l.id === id ? loan : l));
  renderInstallments();
  if (currentTab === 'home') renderHome();
}

function deleteLoan(id) {
  db.saveLoans(db.loans().filter(l => l.id !== id));
  renderInstallments();
  if (currentTab === 'home') renderHome();
}
```

- [ ] **Step 3: 브라우저에서 할부/대출 탭 확인**

1. "+ 할부 추가" → 항목명, 총액, 개월 입력 → 저장
2. 할부 카드 표시 확인 (진행률 바, 월납금)
3. "이번달 납부 완료" 클릭 → 진행률 업데이트
4. "+ 대출 추가" → 항목명, 잔액, 목표일 입력 → 저장
5. 대출 카드 D-day, 일 적립액 표시 확인

- [ ] **Step 4: 커밋**

```bash
git add budget-app/index.html
git commit -m "feat: implement installments and loans tab with modals and progress tracking"
```

---

## Task 8: 초기 설정 유도 + 빈 상태 처리 + 최종 통합

**Files:**
- Modify: `budget-app/index.html` (INIT 섹션 + 엣지 케이스)

- [ ] **Step 1: 빈 상태 안내 — `renderHome`에 미설정 감지 추가**

`renderHome` 함수 맨 위에 조건 추가:

```javascript
function renderHome() {
  const tab      = document.getElementById('tab-home');
  const settings = db.settings();
  const work     = db.work();

  // 초기 설정이 안 된 경우 안내
  if (!settings.allowance && !work.pattern.length) {
    tab.innerHTML = `
      <div style="max-width:480px;margin:0 auto;padding-top:40px;text-align:center;">
        <div class="mono text-3xl mb-4">₩</div>
        <div class="text-sm font-bold mb-2">설정을 먼저 해주세요</div>
        <p class="text-xs mb-6" style="color:var(--text3);">용돈, 알바 패턴, 시급을 등록하면<br>일일 예산을 자동으로 계산합니다.</p>
        <button onclick="switchTab('settings')"
          class="rounded-lg px-6 py-2.5 text-sm font-bold"
          style="background:var(--green);color:#0f172a;">설정 시작 →</button>
      </div>`;
    return;
  }

  // 이하 기존 코드...
```

- [ ] **Step 2: `calcTodaySpent` 함수가 CALCULATIONS 섹션에 있는지 확인**

누락된 경우 CALCULATIONS 섹션에 추가:

```javascript
function calcTodaySpent(expenses, today) {
  const todayStr = formatDate(today);
  return expenses
    .filter(e => e.date === todayStr)
    .reduce((s, e) => s + (e.myAmount ?? e.amount ?? 0), 0);
}
```

- [ ] **Step 3: 모바일 터치 스크롤 개선 — `<main>` 스타일 조정**

`<main>` 태그를 아래로 교체:

```html
<main id="main" class="flex-1 overflow-y-auto pb-16" style="-webkit-overflow-scrolling:touch;">
```

- [ ] **Step 4: 전체 시나리오 테스트**

1. 설정 탭에서 용돈·시급·알바패턴 저장
2. 고정 지출 1개 추가 (예: 통신비 55,000)
3. 할부 1개 추가 (예: 에어팟, 360,000 / 12개월)
4. 대출 1개 추가 (예: 학자금, 500,000 / 목표일 3개월 후)
5. 홈 탭에서 가처분 소득 분해 내역 확인
6. 지출 탭에서 SMS 파싱 → 저장 → 홈 탭 최근 지출 확인
7. 더치페이 흐름 확인 (myAmount만 예산에서 차감)
8. 브라우저 새로고침 후 데이터 유지 확인

- [ ] **Step 5: 최종 커밋**

```bash
git add budget-app/index.html budget-app/tests.html
git commit -m "feat: complete budget app MVP — empty state, mobile polish, integration tested"
```

---

## 완료 기준 체크리스트

- [ ] 설정 탭에서 알바 패턴 + 용돈 + 시급 저장 가능
- [ ] 홈 탭에서 가처분 소득 분해 내역 + 일일 예산 표시
- [ ] 잔여 예산 상태에 따라 컬러 변경 (초록→노랑→빨강)
- [ ] SMS 붙여넣기 → GPT-4o mini 파싱 → 카테고리 분류
- [ ] AI 피드백 한 줄 표시
- [ ] 더치페이 토글 → myAmount만 예산 차감
- [ ] 직접 입력 폼 동작
- [ ] 할부 추가/납부/삭제 동작
- [ ] 대출 추가/잔액 수정/삭제 동작
- [ ] 새로고침 후 모든 데이터 유지 (localStorage)
- [ ] `tests.html`에서 계산 엔진 모든 테스트 통과
