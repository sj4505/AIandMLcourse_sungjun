"""
Week 1 과제 풀이
1. 01_hello_nn.py + 02_polynomial_fitting.py 실행 및 비교
2. y = 3x + 2 공식으로 세 방법 모두 테스트
3. 노이즈 크기 변경 실험 (scale = 0.1, 1.0, 5.0)
4. 결과를 한글 PDF로 저장
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
import reportlab

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# 한글 폰트 등록 (Windows 기본 폰트)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_bold_path = "C:/Windows/Fonts/malgunbd.ttf"
pdfmetrics.registerFont(TTFont('Malgun', font_path))
pdfmetrics.registerFont(TTFont('MalgunBold', font_bold_path))

# matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────
def train_nn(X, y, epochs=500, seed=42):
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    history = model.fit(X, y, epochs=epochs, verbose=0)
    w = model.get_weights()[0][0][0]
    b = model.get_weights()[1][0]
    return model, history, w, b

def polyfit_method(X, y):
    coef = np.polyfit(X, y, deg=1)
    return coef[0], coef[1]

def scipy_method(X, y):
    def linear_fn(x, w, b): return w * x + b
    popt, _ = curve_fit(linear_fn, X, y, p0=[0.5, 0.5])
    return popt[0], popt[1]

# ─────────────────────────────────────────────
# 과제 1: 원본 y=2x-1 세 방법 비교
# ─────────────────────────────────────────────
print("=== 과제 1: y = 2x - 1 원본 비교 ===")
X1 = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y1_clean = 2 * X1 - 1
np.random.seed(42)
y1 = y1_clean + np.random.normal(0, 1.0, len(X1))

model1, hist1, w1_nn, b1_nn = train_nn(X1, y1)
w1_poly, b1_poly = polyfit_method(X1, y1)
w1_scipy, b1_scipy = scipy_method(X1, y1)

fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle('과제 1: y = 2x - 1  |  세 가지 방법 비교', fontsize=13, fontweight='bold')

ax = axes[0]
x_range = np.linspace(-2, 5, 100)
ax.scatter(X1, y1, color='red', s=100, zorder=5, label='노이즈 데이터')
ax.plot(X1, y1_clean, 'k:', alpha=0.5, label='실제 함수 (y=2x-1)')
y_nn = model1.predict(x_range.reshape(-1, 1), verbose=0).flatten()
ax.plot(x_range, y_nn, 'b-', label=f'신경망: y={w1_nn:.2f}x+{b1_nn:.2f}')
ax.plot(x_range, w1_poly*x_range+b1_poly, 'g--', label=f'Polyfit: y={w1_poly:.2f}x+{b1_poly:.2f}')
ax.plot(x_range, w1_scipy*x_range+b1_scipy, 'm:', linewidth=2, label=f'SciPy: y={w1_scipy:.2f}x+{b1_scipy:.2f}')
ax.set_title('피팅 결과 비교'); ax.set_xlabel('X'); ax.set_ylabel('y')
ax.legend(fontsize=8); ax.grid(True)

ax2 = axes[1]
ax2.plot(hist1.history['loss'], color='blue')
ax2.set_title('신경망 훈련 손실 (Loss) 변화'); ax2.set_xlabel('에폭 (Epoch)'); ax2.set_ylabel('MSE 손실')
ax2.grid(True)

plt.tight_layout()
fig1.savefig(f'{output_dir}/hw_task1_comparison.png', dpi=100, bbox_inches='tight')
plt.close(fig1)
print(f"  NN:     y = {w1_nn:.4f}x + {b1_nn:.4f}  | x=10 → {w1_nn*10+b1_nn:.4f}")
print(f"  Poly:   y = {w1_poly:.4f}x + {b1_poly:.4f}")
print(f"  SciPy:  y = {w1_scipy:.4f}x + {b1_scipy:.4f}")

# ─────────────────────────────────────────────
# 과제 2: 다른 공식 y = 3x + 2
# ─────────────────────────────────────────────
print("\n=== 과제 2: y = 3x + 2 ===")
X2 = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y2_clean = 3 * X2 + 2
np.random.seed(42)
y2 = y2_clean + np.random.normal(0, 1.0, len(X2))

model2, hist2, w2_nn, b2_nn = train_nn(X2, y2)
w2_poly, b2_poly = polyfit_method(X2, y2)
w2_scipy, b2_scipy = scipy_method(X2, y2)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('과제 2: y = 3x + 2  |  세 가지 방법 모두 학습 가능한가?', fontsize=13, fontweight='bold')

ax = axes2[0]
x_range2 = np.linspace(-2, 5, 100)
ax.scatter(X2, y2, color='red', s=100, zorder=5, label='노이즈 데이터')
ax.plot(X2, y2_clean, 'k:', alpha=0.5, label='실제 함수 (y=3x+2)')
y_nn2 = model2.predict(x_range2.reshape(-1, 1), verbose=0).flatten()
ax.plot(x_range2, y_nn2, 'b-', label=f'신경망: y={w2_nn:.2f}x+{b2_nn:.2f}')
ax.plot(x_range2, w2_poly*x_range2+b2_poly, 'g--', label=f'Polyfit: y={w2_poly:.2f}x+{b2_poly:.2f}')
ax.plot(x_range2, w2_scipy*x_range2+b2_scipy, 'm:', linewidth=2, label=f'SciPy: y={w2_scipy:.2f}x+{b2_scipy:.2f}')
ax.set_title('피팅 결과 비교 (y=3x+2)'); ax.set_xlabel('X'); ax.set_ylabel('y')
ax.legend(fontsize=8); ax.grid(True)

ax2 = axes2[1]
ax2.plot(hist2.history['loss'], color='green')
ax2.set_title('신경망 훈련 손실 변화 (y=3x+2)'); ax2.set_xlabel('에폭 (Epoch)'); ax2.set_ylabel('MSE 손실')
ax2.grid(True)

plt.tight_layout()
fig2.savefig(f'{output_dir}/hw_task2_y3x2.png', dpi=100, bbox_inches='tight')
plt.close(fig2)
print(f"  NN:     y = {w2_nn:.4f}x + {b2_nn:.4f}  | x=10 → {w2_nn*10+b2_nn:.4f}")
print(f"  Poly:   y = {w2_poly:.4f}x + {b2_poly:.4f}")
print(f"  SciPy:  y = {w2_scipy:.4f}x + {b2_scipy:.4f}")

# ─────────────────────────────────────────────
# 과제 3: 노이즈 크기 변화 실험
# ─────────────────────────────────────────────
print("\n=== 과제 3: 노이즈 크기 변화 (0.1 / 1.0 / 5.0) ===")
scales = [0.1, 1.0, 5.0]
noise_results = {}

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
fig3.suptitle('과제 3: 노이즈 크기에 따른 각 방법의 결과 변화', fontsize=13, fontweight='bold')

for i, scale in enumerate(scales):
    np.random.seed(42)
    y_noisy = y1_clean + np.random.normal(0, scale, len(X1))
    _, _, w_nn, b_nn = train_nn(X1, y_noisy)
    w_p, b_p = polyfit_method(X1, y_noisy)
    w_s, b_s = scipy_method(X1, y_noisy)
    noise_results[scale] = {'nn': (w_nn, b_nn), 'poly': (w_p, b_p), 'scipy': (w_s, b_s)}

    ax = axes3[i]
    x_r = np.linspace(-2, 5, 100)
    ax.scatter(X1, y_noisy, color='red', s=80, zorder=5, label='노이즈 데이터')
    ax.plot(X1, y1_clean, 'k:', alpha=0.5, label='실제: y=2x-1')
    ax.plot(x_r, w_nn*x_r+b_nn, 'b-', label=f'신경망: {w_nn:.2f}x+{b_nn:.2f}')
    ax.plot(x_r, w_p*x_r+b_p, 'g--', label=f'Polyfit: {w_p:.2f}x+{b_p:.2f}')
    ax.plot(x_r, w_s*x_r+b_s, 'm:', linewidth=2, label=f'SciPy: {w_s:.2f}x+{b_s:.2f}')
    ax.set_title(f'노이즈 크기 = {scale}'); ax.set_xlabel('X'); ax.set_ylabel('y')
    ax.legend(fontsize=7); ax.grid(True)

    print(f"  scale={scale}: NN=({w_nn:.3f},{b_nn:.3f})  Poly=({w_p:.3f},{b_p:.3f})  SciPy=({w_s:.3f},{b_s:.3f})")

plt.tight_layout()
fig3.savefig(f'{output_dir}/hw_task3_noise.png', dpi=100, bbox_inches='tight')
plt.close(fig3)

# ─────────────────────────────────────────────
# PDF 생성 (한글)
# ─────────────────────────────────────────────
print("\n=== PDF 생성 중 ===")

pdf_path = f'{output_dir}/week1_homework_report.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                        rightMargin=2*cm, leftMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

# 스타일 정의 (한글 폰트 적용)
title_style = ParagraphStyle('Title', fontName='MalgunBold', fontSize=20,
                              spaceAfter=12, alignment=TA_CENTER, textColor=colors.HexColor('#1A237E'))
subtitle_style = ParagraphStyle('Sub', fontName='Malgun', fontSize=12,
                                 alignment=TA_CENTER, textColor=colors.grey, spaceAfter=6)
h1_style = ParagraphStyle('H1', fontName='MalgunBold', fontSize=14,
                           spaceBefore=16, spaceAfter=6, textColor=colors.HexColor('#1565C0'))
h2_style = ParagraphStyle('H2', fontName='MalgunBold', fontSize=11,
                           spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#37474F'))
body_style = ParagraphStyle('Body', fontName='Malgun', fontSize=10,
                             spaceAfter=4, leading=16)
bold_body = ParagraphStyle('BoldBody', fontName='MalgunBold', fontSize=10,
                            spaceAfter=4, leading=16)

def add_img(path, width=16*cm):
    if os.path.exists(path):
        return Image(path, width=width, height=width*0.38)
    return Paragraph(f"[이미지 없음: {path}]", body_style)

def make_table(data, col_widths, header_color):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), header_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'MalgunBold'),
        ('FONTNAME', (0,1), (-1,-1), 'Malgun'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#EEF2FF')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    return t

story = []

# ── 표지 ──
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph("Week 1 과제 보고서", title_style))
story.append(Paragraph("인공지능과 물리학 강의", subtitle_style))
story.append(Paragraph("신경망 vs 수치 해법 비교 실험", subtitle_style))
story.append(Spacer(1, 0.8*cm))

env_data = [
    ['패키지', '버전'],
    ['Python', '3.12.10'],
    ['NumPy', np.__version__],
    ['Matplotlib', matplotlib.__version__],
    ['TensorFlow', tf.__version__],
    ['ReportLab', reportlab.Version],
]
env_table = make_table(env_data, [8*cm, 8*cm], colors.HexColor('#1565C0'))
story.append(env_table)
story.append(Spacer(1, 1*cm))

story.append(Paragraph(
    "본 보고서는 Week 1 과제의 세 가지 실험 결과를 정리한 것입니다. "
    "신경망(TensorFlow SGD), 다항식 피팅(NumPy polyfit), 수치 최적화(SciPy curve_fit) "
    "세 가지 방법을 동일한 데이터에 적용하여 성능을 비교하였습니다.",
    body_style))

# ── 과제 1 ──
story.append(PageBreak())
story.append(Paragraph("과제 1. 세 가지 방법 실행 및 비교 (y = 2x - 1)", h1_style))
story.append(Paragraph(
    "01_hello_nn.py와 02_polynomial_fitting.py를 모두 실행하고 결과를 비교하였다. "
    "동일한 데이터셋(y = 2x - 1, 가우시안 노이즈 scale=1.0, seed=42)을 사용하여 "
    "세 가지 방법을 적용하였으며, 각 방법이 학습한 파라미터와 예측값을 아래 표에 정리하였다.",
    body_style))
story.append(Spacer(1, 0.3*cm))
story.append(add_img(f'{output_dir}/hw_task1_comparison.png'))
story.append(Spacer(1, 0.3*cm))

t1_data = [
    ['방법', '학습된 w', '학습된 b', 'x=10 예측값', '실제값 (x=10)'],
    ['신경망 (SGD)', f'{w1_nn:.4f}', f'{b1_nn:.4f}', f'{w1_nn*10+b1_nn:.4f}', '19.0000'],
    ['NumPy Polyfit', f'{w1_poly:.4f}', f'{b1_poly:.4f}', f'{w1_poly*10+b1_poly:.4f}', '19.0000'],
    ['SciPy Curve Fit', f'{w1_scipy:.4f}', f'{b1_scipy:.4f}', f'{w1_scipy*10+b1_scipy:.4f}', '19.0000'],
    ['이론값 (참값)', '2.0000', '-1.0000', '19.0000', '19.0000'],
]
story.append(make_table(t1_data, [4*cm, 3*cm, 3*cm, 3.5*cm, 3.5*cm], colors.HexColor('#1976D2')))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "【분석】 세 가지 방법 모두 w ≈ 1.91, b ≈ -0.52로 수렴하였다. "
    "참값(w=2.0, b=-1.0)과의 차이는 추가된 노이즈(scale=1.0)에 의한 것으로, "
    "노이즈가 없는 이상적인 데이터라면 세 방법 모두 정확히 w=2, b=-1을 학습한다. "
    "NumPy Polyfit과 SciPy Curve Fit은 동일한 결과를 출력하는데, "
    "이는 선형 문제에서 두 방법이 수학적으로 동등하기 때문이다. "
    "신경망은 500 에폭 반복 학습 후 유사한 값에 수렴하였다.",
    body_style))

# ── 과제 2 ──
story.append(PageBreak())
story.append(Paragraph("과제 2. 다른 공식 학습 실험 (y = 3x + 2)", h1_style))
story.append(Paragraph(
    "데이터의 X 값은 동일하게 유지하되 공식을 y = 3x + 2 (w=3, b=2)로 변경하여 "
    "세 가지 방법이 새로운 파라미터를 올바르게 학습할 수 있는지 검증하였다. "
    "노이즈 조건(scale=1.0, seed=42)도 동일하게 적용하였다.",
    body_style))
story.append(Spacer(1, 0.3*cm))
story.append(add_img(f'{output_dir}/hw_task2_y3x2.png'))
story.append(Spacer(1, 0.3*cm))

t2_data = [
    ['방법', '학습된 w', '학습된 b', 'x=10 예측값', '실제값 (x=10)'],
    ['신경망 (SGD)', f'{w2_nn:.4f}', f'{b2_nn:.4f}', f'{w2_nn*10+b2_nn:.4f}', '32.0000'],
    ['NumPy Polyfit', f'{w2_poly:.4f}', f'{b2_poly:.4f}', f'{w2_poly*10+b2_poly:.4f}', '32.0000'],
    ['SciPy Curve Fit', f'{w2_scipy:.4f}', f'{b2_scipy:.4f}', f'{w2_scipy*10+b2_scipy:.4f}', '32.0000'],
    ['이론값 (참값)', '3.0000', '2.0000', '32.0000', '32.0000'],
]
story.append(make_table(t2_data, [4*cm, 3*cm, 3*cm, 3.5*cm, 3.5*cm], colors.HexColor('#E65100')))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "【분석】 세 가지 방법 모두 y = 3x + 2를 성공적으로 학습하였다. "
    "학습된 w는 약 2.91~2.92, b는 약 2.47로 참값(w=3, b=2)에 근접하였으며, "
    "y = 2x - 1 실험과 마찬가지로 노이즈로 인한 편차가 발생하였다. "
    "이를 통해 세 가지 방법 모두 특정 공식에 종속되지 않고 임의의 선형 관계를 "
    "일반적으로 학습할 수 있음을 확인하였다.",
    body_style))

# ── 과제 3 ──
story.append(PageBreak())
story.append(Paragraph("과제 3. 노이즈 크기 변화에 따른 영향 분석", h1_style))
story.append(Paragraph(
    "노이즈 크기(scale)를 0.1, 1.0, 5.0으로 변경하여 각 방법의 학습 결과가 "
    "어떻게 달라지는지 실험하였다. 기준 공식은 y = 2x - 1이며, "
    "나머지 조건(데이터 범위, seed, 에폭 수)은 모두 동일하게 유지하였다.",
    body_style))
story.append(Spacer(1, 0.3*cm))
story.append(add_img(f'{output_dir}/hw_task3_noise.png'))
story.append(Spacer(1, 0.3*cm))

noise_header = ['노이즈 크기', '신경망  w / b', 'Polyfit  w / b', 'SciPy  w / b']
noise_rows = [noise_header]
for sc in scales:
    r = noise_results[sc]
    noise_rows.append([
        str(sc),
        f"{r['nn'][0]:.3f} / {r['nn'][1]:.3f}",
        f"{r['poly'][0]:.3f} / {r['poly'][1]:.3f}",
        f"{r['scipy'][0]:.3f} / {r['scipy'][1]:.3f}",
    ])
noise_rows.append(['참값', '2.000 / -1.000', '2.000 / -1.000', '2.000 / -1.000'])

story.append(make_table(noise_rows, [3.5*cm, 4.5*cm, 4.5*cm, 4.5*cm], colors.HexColor('#6A1B9A')))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "【분석】 노이즈가 작을수록(scale=0.1) 세 방법 모두 참값에 매우 근접한 결과를 보였다. "
    "반면 노이즈가 커질수록(scale=5.0) 추정 파라미터의 오차가 크게 증가하였는데, "
    "이는 데이터 포인트가 6개에 불과하여 노이즈의 영향을 상쇄하기에 표본이 부족하기 때문이다. "
    "NumPy Polyfit과 SciPy Curve Fit은 모든 노이즈 수준에서 동일한 결과를 보였고, "
    "신경망은 약간 다른 값으로 수렴하였다. "
    "이는 SGD 기반 최적화가 확률적 특성을 가지며, "
    "소규모 데이터에서는 해석적 방법보다 수렴이 불안정할 수 있음을 시사한다.",
    body_style))

# ── 종합 결론 ──
story.append(PageBreak())
story.append(Paragraph("종합 결론", h1_style))

summary_data = [
    ['비교 항목', '신경망 (SGD)', 'NumPy Polyfit', 'SciPy Curve Fit'],
    ['계산 속도', '느림 (500 에폭)', '매우 빠름', '빠름'],
    ['코드 복잡도', '복잡', '매우 단순 (1줄)', '단순'],
    ['적용 범위', '선형 · 비선형 모두', '다항식만', '임의 함수 가능'],
    ['노이즈 강건성', '보통', '좋음', '좋음'],
    ['소규모 데이터 정확도', '보통', '높음', '높음'],
    ['최적 활용 상황', '복잡한 패턴 학습', '단순 선형 회귀', '수식이 알려진 경우'],
]
story.append(make_table(summary_data,
    [4*cm, 3.5*cm, 3.5*cm, 4*cm], colors.HexColor('#263238')))
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph(
    "본 실험을 통해 다음과 같은 결론을 도출하였다. "
    "첫째, y = 2x - 1과 y = 3x + 2 모두에서 세 가지 방법이 성공적으로 파라미터를 학습하였으며, "
    "이는 각 방법의 범용성을 보여준다. "
    "둘째, 선형 문제에서는 NumPy Polyfit과 SciPy Curve Fit이 신경망보다 빠르고 정확한 경향이 있다. "
    "셋째, 노이즈가 커질수록 모든 방법의 정확도가 저하되며, "
    "이를 완화하기 위해서는 더 많은 데이터 포인트가 필요하다. "
    "신경망의 진가는 단순한 선형 회귀가 아닌, "
    "기존 수식으로 표현할 수 없는 복잡한 비선형 패턴을 학습하는 데 있다.",
    body_style))

doc.build(story)
print(f"PDF 저장 완료: {pdf_path}")
