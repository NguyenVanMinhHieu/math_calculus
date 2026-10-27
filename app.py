"""Web App Giải tích Hình học THPT - Nhãn tỉ lệ x:y bên phải và Ô điều chỉnh đơn vị vẽ."""

from __future__ import annotations

import io
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import streamlit as st
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from calculus_engine import CalculusGeometryEngine

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

st.set_page_config(page_title="Giải tích Hình học THPT", page_icon="📐", layout="wide")

st.title("📐 Ứng dụng Giải tích: Diện tích & Thể tích tròn xoay")
st.markdown("**Hỗ trợ nghiên cứu và giảng dạy Toán học - Thầy giáo Nguyễn Văn Minh Hiếu - THPT TRUNG PHÚ**")

engine = CalculusGeometryEngine()

def preprocess_math_str(s: str) -> str:
    """Tự động chuẩn hóa: 2x -> 2*x, x^3 -> x**3."""
    if not s or s.strip() == "0":
        return "0"
    s = s.strip().replace("^", "**")
    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'(\))(\()', r'\1*\2', s)
    return s

def parse_bounds(text: str) -> tuple[float, float] | None:
    """Bóc tách chuỗi [a; b], [-1; 2], -1; 2 hoặc [0; pi] thành cặp (a, b)."""
    if not text:
        return None
    cleaned = text.strip().strip("[]()")
    parts = [p.strip() for p in re.split(r"[;,]", cleaned) if p.strip()]
    if len(parts) == 2:
        try:
            ea = engine._parse_expression(preprocess_math_str(parts[0]))
            eb = engine._parse_expression(preprocess_math_str(parts[1]))
            va = float(ea.evalf())
            vb = float(eb.evalf())
            if va > vb:
                va, vb = vb, va
            return va, vb
        except Exception:
            return None
    return None

def draw_single_revolution_solid(
    ax: plt.Axes,
    h_expr: sp.Expr,
    x_var: sp.Symbol,
    x_a: float,
    x_b: float,
    label_a: str,
    label_b: str,
    title_str: str,
    main_color: str = "#005f73",
    hatch_color: str = "#c2185b",
    aspect_ratio: float = 1.0,
) -> None:
    """Dựng khối tròn xoay chuẩn SGK có nhãn tỉ lệ x:y bên phải."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    h_num = sp.lambdify(x_var, h_expr, modules=["numpy"])
    span = max(x_b - x_a, 1.0)
    x_min = x_a - span * 0.25
    x_max = x_b + span * 0.35

    xs = np.linspace(x_a, x_b, 350)
    ys_real = np.asarray(h_num(xs), dtype=float)
    if ys_real.ndim == 0:
        ys_real = np.full_like(xs, ys_real)
    ys_refl = -ys_real

    max_r = max(np.max(np.abs(ys_real)), 1.0)
    y_lim = max_r * 1.25
    y_top_arrow = y_lim * 1.15
    y_bot = -y_lim * 1.15

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_bot, y_top_arrow * 1.08)
    ax.set_aspect(aspect_ratio)

    # 1. Trục Ox xuyên tâm
    ax.plot([x_min, x_a], [0, 0], color="black", linewidth=1.2)
    ax.plot([x_a, x_b], [0, 0], color="black", linestyle="--", linewidth=1.2)
    ax.plot([x_b, x_max], [0, 0], color="black", linewidth=1.2)
    ax.plot(x_max, 0, ">k", clip_on=False, markersize=5)
    ax.text(x_max, -y_lim * 0.09, r"$x$", fontsize=11, style="italic")

    # Mốc a và b
    ax.plot(x_a, 0, "ko", markersize=4.0)
    ax.plot(x_b, 0, "ko", markersize=4.0)
    ax.text(x_a, -y_lim * 0.18, rf"${label_a}$", fontsize=11, ha="center")
    ax.text(x_b, -y_lim * 0.18, rf"${label_b}$", fontsize=11, ha="center")

    # 2. Trục tung Oy và gốc O
    if x_min <= 0 <= x_max:
        ax.plot([0, 0], [-y_lim * 0.95, y_top_arrow], color="black", linewidth=1.2)
        ax.plot(0, y_top_arrow, "^k", clip_on=False, markersize=5)
        dx = span * 0.02
        ax.text(-dx * 1.6, y_top_arrow, r"$y$", fontsize=11, style="italic", va="center")
        ax.text(-dx * 1.8, -y_lim * 0.14, r"$O$", fontsize=10, style="italic",
                bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85))

    # 3. Miền phẳng gốc (gạch sọc)
    ax.fill_between(xs, ys_real, 0, facecolor="#fce4ec", edgecolor=hatch_color, hatch="///", linewidth=0.6, alpha=0.9)

    ha = float(ys_real[0])
    hb = float(ys_real[-1])
    ax.plot([x_a, x_a], [0, ha], color=hatch_color, linestyle="--", linewidth=1.1)
    ax.plot([x_b, x_b], [0, hb], color=hatch_color, linestyle="--", linewidth=1.1)

    # 4. Đường đồ thị gốc và đối xứng
    ax.plot(xs, ys_real, color=main_color, linewidth=2.0)
    ax.plot(xs, ys_refl, color=main_color, linestyle="--", linewidth=1.5, alpha=0.75)

    # 5. Hai Elip tại a và b
    rx = span * 0.045
    r_a = abs(ha)
    r_b = abs(hb)

    if r_a > 0.06:
        t_in = np.linspace(-np.pi / 2, np.pi / 2, 60)
        t_out = np.linspace(np.pi / 2, 3 * np.pi / 2, 60)
        ax.plot(x_a + rx * np.cos(t_in), r_a * np.sin(t_in), color=main_color, linestyle="--", linewidth=1.1)
        ax.plot(x_a + rx * np.cos(t_out), r_a * np.sin(t_out), color=main_color, linestyle="-", linewidth=1.8)

    if r_b > 0.06:
        t_full = np.linspace(0, 2 * np.pi, 120)
        ax.plot(x_b + rx * np.cos(t_full), r_b * np.sin(t_full), color=main_color, linestyle="-", linewidth=1.8)

    # 6. NHÃN TỈ LỆ x:y BÊN PHẢI HÌNH VẼ
    scale_str = "x : y = 1 : 1" if abs(aspect_ratio - 1.0) < 1e-4 else f"x : y = 1 : {aspect_ratio:g}"
    ax.text(
        0.96, 0.94, scale_str,
        transform=ax.transAxes,
        fontsize=10.5,
        ha="right",
        va="top",
        weight="bold",
        color="#1d3557",
        bbox=dict(boxstyle="square,pad=0.25", facecolor="#f8f9fa", edgecolor="#b0bec5", alpha=0.92)
    )

    ax.set_title(title_str, fontsize=12, pad=12, weight="bold", color="#1d3557")

# ========================================================
# SIDEBAR NHẬP LIỆU & ĐIỀU CHỈNH TỈ LỆ ĐƠN VỊ VẼ
# ========================================================
st.sidebar.header("⚙️ Tùy chọn bài toán")
mode = st.sidebar.radio(
    "Dạng toán cần tính:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)
is_volume = "Thể tích" in mode

st.sidebar.markdown("---")
st.sidebar.header("📝 Dữ liệu đầu vào")

if "f_input" not in st.session_state:
    st.session_state["f_input"] = "x^3 - 5*x"
if "g_input" not in st.session_state:
    st.session_state["g_input"] = "2*x"
if "use_bounds" not in st.session_state:
    st.session_state["use_bounds"] = True
if "bounds_input" not in st.session_state:
    st.session_state["bounds_input"] = "[-1; 2]"

f_raw = st.sidebar.text_input("Hàm số f(x):", value=st.session_state["f_input"])
g_raw = st.sidebar.text_input("Hàm số g(x) (trục hoành đặt là 0):", value=st.session_state["g_input"])

try:
    f_clean = preprocess_math_str(f_raw)
    g_clean = preprocess_math_str(g_raw)
    f_preview = engine._parse_expression(f_clean)
    g_preview = engine._parse_expression(g_clean)
    st.sidebar.info(f"Xem trước: $f(x) = {sp.latex(f_preview)}$ và $g(x) = {sp.latex(g_preview)}$")
except Exception:
    f_clean, g_clean = f_raw, g_raw

use_custom_bounds = st.sidebar.checkbox("Chỉ định cận [a; b]", value=st.session_state["use_bounds"])
a_val: float | None = None
b_val: float | None = None

if use_custom_bounds:
    bounds_raw = st.sidebar.text_input(
        "[a; b] =",
        value=st.session_state["bounds_input"],
        placeholder="[-1; 2]",
        help="Nhập dạng [-1; 2] hoặc biểu thức như [0; pi], [-sqrt(3); 2]",
    )
    st.session_state["bounds_input"] = bounds_raw
    parsed = parse_bounds(bounds_raw)
    if parsed:
        a_val, b_val = parsed
    else:
        st.sidebar.error("Định dạng: [a; b], ví dụ: [-1; 2]")

# ĐIỀU CHỈNH TỈ LỆ ĐƠN VỊ VẼ HÌNH (DÙNG Ô NHẬP SỐ CHÍNH XÁC, MẶC ĐỊNH 1.0)
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Tỉ lệ đơn vị vẽ hình")
axis_ratio = st.sidebar.number_input(
    "Tỉ lệ trục y (so với x):",
    min_value=0.1,
    max_value=10.0,
    value=1.00,  # Mặc định chuẩn 1:1 theo ý Thầy
    step=0.1,
    format="%.2f",
    help="Mặc định là 1.00 (Tỉ lệ chuẩn x:y = 1:1). Giáo viên có thể điều chỉnh để tăng/giảm tỉ lệ vẽ theo ý muốn.",
)

btn_calc = st.sidebar.button("🚀 Bắt đầu tính toán & Dựng hình", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Ví dụ mẫu nhanh:**")
if st.sidebar.button("📌 y = x^3 - 5x và y = 2x"):
    st.session_state["f_input"] = "x^3 - 5*x"
    st.session_state["g_input"] = "2*x"
    st.session_state["use_bounds"] = True
    st.session_state["bounds_input"] = "[-1; 2]"
if st.sidebar.button("📌 Parabol y = 4 - x^2 và Ox"):
    st.session_state["f_input"] = "4 - x^2"
    st.session_state["g_input"] = "0"
    st.session_state["use_bounds"] = True
    st.session_state["bounds_input"] = "[0; 2]"

# ========================================================
# MÀN HÌNH CHÍNH
# ========================================================
title_res = "🎯 Phân tích Sư phạm Khối tròn xoay" if is_volume else "🎯 Hình phẳng chuẩn Đề thi"
st.subheader(title_res)

if btn_calc or "has_run" in st.session_state:
    st.session_state["has_run"] = True
    try:
        x_sym = engine.var
        f_expr = engine._parse_expression(f_clean)
        g_expr = engine._parse_expression(g_clean)
        has_g = str(g_expr) != "0"

        if not is_volume:
            res = engine.calculate_area(f_expr, g_expr, a=a_val, b=b_val)
            st.success("✅ Đã tính xong Diện tích hình phẳng!")
            st.latex(rf"S = {res.latex_str} \approx {res.numeric_value:.6f}")
        else:
            res = engine.calculate_rotation_volume_ox(f_expr, g_expr, a=a_val, b=b_val)
            st.success("✅ Đã tính xong Thể tích khối tròn xoay quanh Ox!")
            st.latex(rf"V_{{Ox}} = {res.latex_str} \approx {res.numeric_value:.6f}")

        all_pts_num = [pt for interval in res.intervals for pt in interval]
        x_a, x_b = min(all_pts_num), max(all_pts_num)
        label_a = f"{x_a:g}"
        label_b = f"{x_b:g}"
        if res.exact_intervals:
            label_a = sp.latex(res.exact_intervals[0][0])
            label_b = sp.latex(res.exact_intervals[-1][1])

        # Chi tiết tính toán Bản chất Sư phạm
        if is_volume and has_g:
            with st.container(border=True):
                st.markdown("### 💡 Chi tiết tính toán Bản chất Sư phạm: $V = |V_1 - V_2|$")

                sym_a = sp.nsimplify(x_a, rational=True)
                sym_b = sp.nsimplify(x_b, rational=True)

                v1_integral = sp.integrate(f_expr**2, (x_sym, sym_a, sym_b))
                v1_total = sp.simplify(sp.pi * v1_integral)
                st.markdown(f"**1. Khối 1 do $y = {sp.latex(f_expr)}$ quay quanh $Ox$ trên $[{label_a}; {label_b}]$:**")
                st.latex(
                    rf"V_1 = \pi \int_{{{label_a}}}^{{{label_b}}} \left({sp.latex(f_expr)}\right)^2 dx "
                    rf"= \pi \int_{{{label_a}}}^{{{label_b}}} \left({sp.latex(sp.expand(f_expr**2))}\right) dx "
                    rf"= {sp.latex(v1_total)} \approx {float(v1_total.evalf()):.4f}"
                )

                v2_integral = sp.integrate(g_expr**2, (x_sym, sym_a, sym_b))
                v2_total = sp.simplify(sp.pi * v2_integral)
                st.markdown(f"**2. Khối 2 do $y = {sp.latex(g_expr)}$ quay quanh $Ox$ trên $[{label_a}; {label_b}]$:**")
                st.latex(
                    rf"V_2 = \pi \int_{{{label_a}}}^{{{label_b}}} \left({sp.latex(g_expr)}\right)^2 dx "
                    rf"= \pi \int_{{{label_a}}}^{{{label_b}}} \left({sp.latex(sp.expand(g_expr**2))}\right) dx "
                    rf"= {sp.latex(v2_total)} \approx {float(v2_total.evalf()):.4f}"
                )

                st.markdown("**3. Kết luận thể tích khối tròn xoay (Lấy Khối 1 trừ Khối 2):**")
                exact_sub_intervals = res.exact_intervals if res.exact_intervals else [(sp.sympify(a), sp.sympify(b)) for a, b in res.intervals]

                if len(exact_sub_intervals) == 1:
                    st.latex(
                        rf"V = |V_1 - V_2| = \left| {sp.latex(v1_total)} - {sp.latex(v2_total)} \right| "
                        rf"= {res.latex_str} \approx {res.numeric_value:.6f}"
                    )
                else:
                    st.write("📌 *Vì hai đồ thị cắt nhau trên đoạn tính phân, ta phân rã và lấy hiệu trên từng khoảng:*")
                    for idx_inv, (sub_a, sub_b) in enumerate(exact_sub_intervals):
                        v1_sub = sp.simplify(sp.pi * sp.integrate(f_expr**2, (x_sym, sub_a, sub_b)))
                        v2_sub = sp.simplify(sp.pi * sp.integrate(g_expr**2, (x_sym, sub_a, sub_b)))
                        v_sub = sp.simplify(sp.Abs(v1_sub - v2_sub))
                        st.latex(
                            rf"\text{{Đoạn }} [{sp.latex(sub_a)}; {sp.latex(sub_b)}]:\quad "
                            rf"V^{{({idx_inv + 1})}} = \left| V_1^{{({idx_inv + 1})}} - V_2^{{({idx_inv + 1})}} \right| "
                            rf"= \left| {sp.latex(v1_sub)} - {sp.latex(v2_sub)} \right| = {sp.latex(v_sub)}"
                        )
                    st.latex(rf"\Rightarrow V_{{Ox}} = \sum V^{{(i)}} = {res.latex_str} \approx {res.numeric_value:.6f}")

        tikz_code = engine.generate_tikz(f_expr, g_expr, a=a_val, b=b_val, intervals=res.intervals, exact_intervals=res.exact_intervals, is_volume=is_volume)

        # DỰNG HÌNH VỚI NHÃN TỈ LỆ x : y BÊN PHẢI
        if is_volume:
            if not has_g:
                fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=130)
                draw_single_revolution_solid(
                    ax=ax,
                    h_expr=f_expr,
                    x_var=x_sym,
                    x_a=x_a,
                    x_b=x_b,
                    label_a=label_a,
                    label_b=label_b,
                    title_str=rf"Khối tròn xoay do $y = {sp.latex(f_expr)}$ quay quanh $Ox$",
                    main_color="#005f73",
                    hatch_color="#c2185b",
                    aspect_ratio=axis_ratio,
                )
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=130)
                draw_single_revolution_solid(
                    ax=ax1,
                    h_expr=f_expr,
                    x_var=x_sym,
                    x_a=x_a,
                    x_b=x_b,
                    label_a=label_a,
                    label_b=label_b,
                    title_str=rf"(1) Khối $V_1$ do $y = {sp.latex(f_expr)}$ quay",
                    main_color="#005f73",
                    hatch_color="#c2185b",
                    aspect_ratio=axis_ratio,
                )
                draw_single_revolution_solid(
                    ax=ax2,
                    h_expr=g_expr,
                    x_var=x_sym,
                    x_a=x_a,
                    x_b=x_b,
                    label_a=label_a,
                    label_b=label_b,
                    title_str=rf"(2) Khối $V_2$ do $y = {sp.latex(g_expr)}$ quay",
                    main_color="#ca6702",
                    hatch_color="#d97706",
                    aspect_ratio=axis_ratio,
                )
                fig.subplots_adjust(top=0.88, wspace=0.22)
        else:
            # DIỆN TÍCH HÌNH PHẲNG: LƯỚI TRỰC CHUẨN CÂN XỨNG (NVMH MATH-ARCH STANDARD)
            f_n = sp.lambdify(x_sym, f_expr, "numpy")
            g_n = sp.lambdify(x_sym, g_expr, "numpy")

            # 1. Xác định tâm đối xứng và bán kính bao của miền tích phân
            xs_core = np.linspace(x_a, x_b, 350)
            yf_core = np.asarray(f_n(xs_core), dtype=float)
            yg_core = np.asarray(g_n(xs_core), dtype=float)

            y_min_core = min(np.nanmin(yf_core), np.nanmin(yg_core), 0.0)
            y_max_core = max(np.nanmax(yf_core), np.nanmax(yg_core), 0.0)

            # Tính toạ độ tâm hình học (Center of Symmetry)
            cx = (x_a + x_b) / 2.0
            cy = (y_min_core + y_max_core) / 2.0

            # Bán kính cân đối toả đều từ tâm (symmetric padding)
            rx = max(abs(x_b - cx), abs(x_a - cx), abs(0.0 - cx)) + 2.0
            ry = max(abs(y_max_core - cy), abs(y_min_core - cy), abs(0.0 - cy)) + 2.0

            # Khóa mốc nguyên trực chuẩn
            x_plot_min = int(np.floor(cx - rx))
            x_plot_max = int(np.ceil(cx + rx))
            y_plot_min = int(np.floor(cy - ry))
            y_plot_max = int(np.ceil(cy + ry))

            # Giới hạn kích thước lưới trang nhã
            dx_units = x_plot_max - x_plot_min
            dy_units = y_plot_max - y_plot_min

            unit_px = 0.55  # Tỉ lệ vật lý 1:1 chuẩn
            fig_w = max(min(dx_units * unit_px, 10.0), 6.5)
            fig_h = max(min(dy_units * unit_px, 10.0), 5.5)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            # Khóa cứng tỉ lệ trực chuẩn 1:1
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_plot_min, x_plot_max)
            ax.set_ylim(y_plot_min, y_plot_max)

            # 2. Lưới ô vuông mờ nhã nhặn NVMH MATH-ARCH (mỗi ô là 1 đơn vị trực chuẩn)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.grid(True, which="major", color="#edf2f7", linestyle="-", linewidth=0.8, zorder=1)

            # 3. Trục toạ độ Oxy xuyên tâm đối xứng
            ax.spines["left"].set_position(("data", 0))
            ax.spines["bottom"].set_position(("data", 0))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#1e293b")
            ax.spines["bottom"].set_color("#1e293b")
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)
            ax.spines["left"].set_zorder(3)
            ax.spines["bottom"].set_zorder(3)

            # Mũi tên trục thanh thoát
            arrow_len_x = dx_units * 0.02
            arrow_len_y = dy_units * 0.02
            ax.plot(x_plot_max, 0, ">k", clip_on=False, markersize=5.5, zorder=5)
            ax.plot(0, y_plot_max, "^k", clip_on=False, markersize=5.5, zorder=5)

            # Nhãn tên trục x, y và gốc O
            ax.text(x_plot_max + 0.3, -0.35, r"$x$", fontsize=12, style="italic", ha="left", va="center")
            ax.text(-0.35, y_plot_max + 0.3, r"$y$", fontsize=12, style="italic", ha="center", va="bottom")
            ax.text(-0.35, -0.35, r"$O$", fontsize=11, style="italic", ha="right", va="top")

            # Ẩn nhãn số 0 tại gốc
            def _clean_zero_lbl(v, pos):
                iv = int(round(v))
                if abs(v) < 1e-4:
                    return ""
                if abs(v - iv) < 1e-4:
                    return f"{iv}"
                return ""

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_clean_zero_lbl))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(_clean_zero_lbl))
            ax.tick_params(axis="both", which="major", labelsize=8.5, length=3, width=0.8, colors="#334155")

            # 4. Tô miền diện tích tích phân
            ax.fill_between(
                xs_core, yf_core, yg_core,
                facecolor="#eff6ff",
                edgecolor="#1d4ed8",
                hatch="///",
                linewidth=0.8,
                alpha=0.85,
                zorder=2
            )

            # 5. Vẽ hai đường cong sắc nét
            xs_plot = np.linspace(x_plot_min, x_plot_max, 600)
            ys_plot_f = f_n(xs_plot)
            ax.plot(xs_plot, ys_plot_f, color="#2563eb", linewidth=1.8, zorder=4)

            # Gắn nhãn chuẩn SGK (C): y = f(x) trực tiếp trên nhánh đồ thị
            valid_f = np.where((ys_plot_f >= y_plot_min + 0.8) & (ys_plot_f <= y_plot_max - 0.8))[0]
            if len(valid_f) > 0:
                idx_f = valid_f[int(len(valid_f) * 0.2)]
                ax.text(
                    xs_plot[idx_f], ys_plot_f[idx_f] + 0.35,
                    rf"$(C): y = {sp.latex(f_expr)}$",
                    color="#1d4ed8", fontsize=10.5, weight="bold",
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
                    zorder=6
                )

            if has_g:
                ys_plot_g = g_n(xs_plot)
                ax.plot(xs_plot, ys_plot_g, color="#dc2626", linestyle="--", linewidth=1.6, zorder=4)

                # Xác định ký hiệu (d) cho đường thẳng hoặc (C') cho đường cong
                is_line = False
                try:
                    is_line = sp.Poly(g_expr, x_sym).degree() <= 1
                except Exception:
                    is_line = False
                name_g = "(d)" if is_line else r"(C')"

                valid_g = np.where((ys_plot_g >= y_plot_min + 0.8) & (ys_plot_g <= y_plot_max - 0.8))[0]
                if len(valid_g) > 0:
                    idx_g = valid_g[int(len(valid_g) * 0.85)]
                    ax.text(
                        xs_plot[idx_g] + 0.25, ys_plot_g[idx_g] - 0.2,
                        rf"${name_g}: y = {sp.latex(g_expr)}$",
                        color="#dc2626", fontsize=10.5, weight="bold",
                        ha="left", va="top",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
                        zorder=6
                    )

            # 6. Đường dóng vuông góc tại các cận
            ax.plot([x_a, x_a], [0, f_n(x_a)], color="#1d4ed8", linestyle=":", linewidth=1.1, zorder=3)
            ax.plot([x_b, x_b], [0, f_n(x_b)], color="#1d4ed8", linestyle=":", linewidth=1.1, zorder=3)
            if has_g:
                ax.plot([x_a, x_a], [0, g_n(x_a)], color="#dc2626", linestyle=":", linewidth=1.1, zorder=3)
                ax.plot([x_b, x_b], [0, g_n(x_b)], color="#dc2626", linestyle=":", linewidth=1.1, zorder=3)

        svg_buffer = io.StringIO()
        fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_data = svg_buffer.getvalue()

        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
        png_data = png_buffer.getvalue()
        plt.close(fig)

        tab_img, tab_tikz = st.tabs(["📊 Hình vẽ Sư phạm", "📋 Mã nguồn TikZ (LaTeX)"])

        with tab_img:
            _, col_mid, _ = st.columns([1, 6, 1])
            with col_mid:
                st.image(svg_data, width=620)
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                st.download_button("📥 Tải ảnh SVG (Vector sắc nét)", data=svg_data, file_name="hinh_su_pham.svg", mime="image/svg+xml", use_container_width=True)
            with c_btn2:
                st.download_button("📥 Tải ảnh PNG (300 DPI)", data=png_data, file_name="hinh_su_pham.png", mime="image/png", use_container_width=True)

        with tab_tikz:
            st.info("💡 Mã TikZ chuẩn SGK để Thầy copy vào đề thi:")
            st.code(tikz_code, language="latex")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
