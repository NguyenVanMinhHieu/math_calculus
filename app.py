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
            # DIỆN TÍCH HÌNH PHẲNG CHUẨN MỰC BỘ GD&ĐT
            f_n = sp.lambdify(x_sym, f_expr, "numpy")
            g_n = sp.lambdify(x_sym, g_expr, "numpy")

            # Mở rộng biên ngang để hình cân đối, không bị hẹp như thanh kẹp sách
            x_min_data = min(x_a, -1.0) - 1.2
            x_max_data = max(x_b, 1.0) + 1.2

            xs_2d = np.linspace(x_min_data, x_max_data, 500)
            yf = np.asarray(f_n(xs_2d), dtype=float)
            yg = np.asarray(g_n(xs_2d), dtype=float)

            # Cắt phạm vi hiển thị y hợp lý xung quanh miền tích phân
            xs_fill = np.linspace(x_a, x_b, 350)
            yf_fill = np.asarray(f_n(xs_fill), dtype=float)
            yg_fill = np.asarray(g_n(xs_fill), dtype=float)

            y_core_min = min(np.nanmin(yf_fill), np.nanmin(yg_fill), 0.0)
            y_core_max = max(np.nanmax(yf_fill), np.nanmax(yg_fill), 0.0)
            y_span = max(y_core_max - y_core_min, 2.0)
            
            y_plot_min = np.floor(y_core_min - y_span * 0.25)
            y_plot_max = np.ceil(y_core_max + y_span * 0.35)

            # Khung hình chuẩn tỉ lệ vàng sư phạm
            dx = x_max_data - x_min_data
            dy = y_plot_max - y_plot_min
            fig_w = 7.5
            fig_h = max(min(fig_w * (dy / dx), 9.0), 5.5)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
            ax.set_aspect("equal")
            ax.set_xlim(x_min_data, x_max_data)
            ax.set_ylim(y_plot_min, y_plot_max)
            ax.axis("off")

            # 1. Trục toạ độ thanh mảnh có mũi tên nhọn LaTeX
            arrow_props = dict(arrowstyle="-|>,head_width=0.22,head_length=0.4", color="black", lw=1.0, mutation_scale=11)
            ax.annotate("", xy=(x_max_data - 0.1, 0), xytext=(x_min_data + 0.1, 0), arrowprops=arrow_props)
            ax.annotate("", xy=(0, y_plot_max - 0.1), xytext=(0, y_plot_min + 0.1), arrowprops=arrow_props)

            # Nhãn trục x, y và gốc O
            ax.text(x_max_data - 0.05, -0.35, r"$x$", fontsize=12, style="italic", ha="right", va="top")
            ax.text(-0.3, y_plot_max - 0.05, r"$y$", fontsize=12, style="italic", ha="right", va="top")
            ax.text(-0.25, -0.32, r"$O$", fontsize=11, style="italic", ha="right", va="top")

            # 2. Vạch chia trục (Ticks) chuẩn SGK: bước 2 nếu trục quá cao, bước 1 nếu vừa phải
            step_y = 2 if dy > 7 else 1
            for vx in range(int(np.ceil(x_min_data)) + 1, int(np.floor(x_max_data))):
                if vx == 0:
                    continue
                ax.plot([vx, vx], [-0.07, 0.07], color="black", lw=0.8)
                ax.text(vx, -0.28, f"{vx}", fontsize=9.5, ha="center", va="top")

            for vy in range(int(np.ceil(y_plot_min)) + 1, int(np.floor(y_plot_max))):
                if vy == 0 or vy % step_y != 0:
                    continue
                ax.plot([-0.07, 0.07], [vy, vy], color="black", lw=0.8)
                ax.text(-0.2, vy, f"{vy}", fontsize=9.5, ha="right", va="center")

            # 3. Tô miền diện tích gạch sọc chuẩn mực SGK
            ax.fill_between(
                xs_fill, yf_fill, yg_fill,
                facecolor="#f4f7fb",
                edgecolor="#1d3557",
                hatch="///",
                linewidth=0.7,
                alpha=0.9
            )

            # 4. Vẽ hai đường cong thanh lịch
            line_f, = ax.plot(xs_2d, yf, color="#005f73", linewidth=1.8)
            if has_g:
                line_g, = ax.plot(xs_2d, yg, color="#c85a17", linestyle="--", linewidth=1.6)

            # 5. Viết tên đường cong trực tiếp lên nhánh đồ thị (Không dùng hộp Legend thô thiển)
            ax.text(xs_2d[-25], yf[-25] + 0.35, rf"$y = {sp.latex(f_expr)}$", color="#005f73", fontsize=11, weight="bold", ha="center")
            if has_g:
                idx_g_lbl = int(len(xs_2d) * 0.72)
                ax.text(xs_2d[idx_g_lbl], yg[idx_g_lbl] + 0.4, rf"$y = {sp.latex(g_expr)}$", color="#c85a17", fontsize=11, weight="bold", ha="center")

            # 6. Đường dóng vuông góc xuống Ox tại a và b
            ax.plot([x_a, x_a], [0, f_n(x_a)], color="#1d3557", linestyle=":", linewidth=1.1)
            ax.plot([x_b, x_b], [0, f_n(x_b)], color="#1d3557", linestyle=":", linewidth=1.1)
            if has_g:
                ax.plot([x_a, x_a], [0, g_n(x_a)], color="#c85a17", linestyle=":", linewidth=1.1)
                ax.plot([x_b, x_b], [0, g_n(x_b)], color="#c85a17", linestyle=":", linewidth=1.1)

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
