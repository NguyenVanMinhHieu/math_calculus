"""Web App Giải tích Hình học THPT - Ánh xạ 1-1 chuẩn mực SGK."""

from __future__ import annotations

import io
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
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

    ax.plot([x_min, x_a], [0, 0], color="black", linewidth=1.2)
    ax.plot([x_a, x_b], [0, 0], color="black", linestyle="--", linewidth=1.2)
    ax.plot([x_b, x_max], [0, 0], color="black", linewidth=1.2)
    ax.plot(x_max, 0, ">k", clip_on=False, markersize=5)
    ax.text(x_max, -y_lim * 0.09, r"$x$", fontsize=11, style="italic")

    ax.plot(x_a, 0, "ko", markersize=4.0)
    ax.plot(x_b, 0, "ko", markersize=4.0)
    ax.text(x_a, -y_lim * 0.18, rf"${label_a}$", fontsize=11, ha="center")
    ax.text(x_b, -y_lim * 0.18, rf"${label_b}$", fontsize=11, ha="center")

    if x_min <= 0 <= x_max:
        ax.plot([0, 0], [-y_lim * 0.95, y_top_arrow], color="black", linewidth=1.2)
        ax.plot(0, y_top_arrow, "^k", clip_on=False, markersize=5)
        dx = span * 0.02
        ax.text(-dx * 1.6, y_top_arrow, r"$y$", fontsize=11, style="italic", va="center")
        ax.text(-dx * 1.8, -y_lim * 0.14, r"$O$", fontsize=10, style="italic",
                bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.85))

    ax.fill_between(xs, ys_real, 0, facecolor="#fce4ec", edgecolor=hatch_color, hatch="///", linewidth=0.6, alpha=0.9)

    ha = float(ys_real[0])
    hb = float(ys_real[-1])
    ax.plot([x_a, x_a], [0, ha], color=hatch_color, linestyle="--", linewidth=1.1)
    ax.plot([x_b, x_b], [0, hb], color=hatch_color, linestyle="--", linewidth=1.1)

    ax.plot(xs, ys_real, color=main_color, linewidth=2.0)
    ax.plot(xs, ys_refl, color=main_color, linestyle="--", linewidth=1.5, alpha=0.75)

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

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Tỉ lệ đơn vị vẽ hình")
axis_ratio = st.sidebar.number_input(
    "Tỉ lệ trục y (so với x):",
    min_value=0.1,
    max_value=10.0,
    value=1.00,
    step=0.1,
    format="%.2f",
    help="Mặc định là 1.00 (Tỉ lệ chuẩn x:y = 1:1).",
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
            all_pts_num = [pt for interval in res.intervals for pt in interval]
            x_a, x_b = min(all_pts_num), max(all_pts_num)
            label_a = f"{x_a:g}"
            label_b = f"{x_b:g}"
            if res.exact_intervals:
                label_a = sp.latex(res.exact_intervals[0][0])
                label_b = sp.latex(res.exact_intervals[-1][1])
            st.success("✅ Đã tính xong Diện tích hình phẳng!")
            
            with st.container(border=True):
                st.markdown("### 💡 Các bước giải Toán Sư phạm: Diện tích Hình phẳng")
                diff_expr = sp.simplify(f_expr - g_expr)
                F_expr = sp.integrate(diff_expr, x_sym)

                def fmt_math(val) -> str:
                    try:
                        sym = sp.nsimplify(val, rational=True)
                        return sp.latex(sym)
                    except Exception:
                        v = float(val)
                        return f"{int(v)}" if v.is_integer() else f"{v:.4f}"

                st.markdown("**Bước 1. Phương trình hoành độ giao điểm:**")
                st.latex(rf"f(x) - g(x) = 0 \iff {sp.latex(diff_expr)} = 0")

                inner_roots = []
                if len(res.intervals) > 1:
                    for inv in res.intervals[:-1]:
                        inner_roots.append(fmt_math(inv[1]))

                lbl_a_str = fmt_math(x_a)
                lbl_b_str = fmt_math(x_b)
                if inner_roots:
                    roots_str = ", ".join([rf"x = {r}" for r in inner_roots])
                    st.markdown(rf"Trên khoảng $({lbl_a_str};\, {lbl_b_str})$, phương trình có nghiệm: ${roots_str}$.")
                else:
                    st.markdown(rf"Trên khoảng $({lbl_a_str};\, {lbl_b_str})$, phương trình không có nghiệm.")

                st.markdown("**Bước 2. Tìm một nguyên hàm của hiệu hai hàm số:**")
                st.latex(rf"F(x) = \int \left( {sp.latex(diff_expr)} \right) dx = {sp.latex(F_expr)} + C")

                st.markdown("**Bước 3. Diện tích hình phẳng cần tìm:**")
                if len(res.intervals) == 1:
                    sub_a, sub_b = res.intervals[0]
                    sa_str, sb_str = fmt_math(sub_a), fmt_math(sub_b)
                    st.latex(
                        rf"S = \int_{{{sa_str}}}^{{{sb_str}}} \left| f(x) - g(x) \right| dx "
                        rf"= \left| F({sb_str}) - F({sa_str}) \right| "
                        rf"= {res.latex_str} \approx {res.numeric_value:.4f}"
                    )
                else:
                    st.markdown("*Tách tích phân theo các nghiệm thuộc đoạn xác định:*")
                    sum_parts = []
                    val_parts = []
                    for idx_inv, (sub_a, sub_b) in enumerate(res.intervals):
                        sa_str, sb_str = fmt_math(sub_a), fmt_math(sub_b)
                        val_b = F_expr.subs(x_sym, sp.sympify(sub_b))
                        val_a = F_expr.subs(x_sym, sp.sympify(sub_a))
                        sub_area = sp.simplify(sp.Abs(val_b - val_a))
                        sub_area_str = fmt_math(sub_area)
                        sum_parts.append(rf"\left| F({sb_str}) - F({sa_str}) \right|")
                        val_parts.append(sub_area_str)
                        st.latex(
                            rf"S_{{{idx_inv + 1}}} = \int_{{{sa_str}}}^{{{sb_str}}} \left| f(x) - g(x) \right| dx "
                            rf"= \left| F({sb_str}) - F({sa_str}) \right| = {sub_area_str}"
                        )
                    st.latex(
                        rf"\Rightarrow S = " + " + ".join(sum_parts) + 
                        rf" = " + " + ".join(val_parts) + 
                        rf" = {res.latex_str} \approx {res.numeric_value:.4f}"
                    )
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

        if is_volume:
            with st.container(border=True):
                st.markdown("### 💡 Các bước giải Toán Sư phạm: Thể tích Khối tròn xoay (quanh Ox)")
                diff_sq = sp.simplify(f_expr**2 - g_expr**2) if has_g else sp.simplify(f_expr**2)
                
                # Bước 1
                st.markdown("**Bước 1. Phương trình hoành độ giao điểm bán kính:**")
                if has_g:
                    st.latex(rf"f^2(x) - g^2(x) = 0 \iff {sp.latex(diff_sq)} = 0")
                else:
                    st.latex(rf"f(x) = 0 \iff {sp.latex(f_expr)} = 0")
                
                inner_roots = []
                inner_roots = []
                if res.exact_intervals and len(res.exact_intervals) > 1:
                    for exact_inv in res.exact_intervals[:-1]:
                        exact_r = exact_inv[1]
                        inner_roots.append(sp.latex(exact_r))
                    st.markdown(rf"Trên khoảng $({label_a}; {label_b})$, phương trình có nghiệm: $x \in \{{{', '.join(inner_roots)}\}}$")
                else:
                    st.markdown(rf"Trên khoảng $({label_a}; {label_b})$, phương trình không có nghiệm đổi dấu.")

                # Bước 2
                st.markdown("**Bước 2. Công thức thể tích tổng quát (Phương pháp hình vành khăn):**")
                if has_g:
                    st.latex(rf"V_{{Ox}} = \pi \int_{{{label_a}}}^{{{label_b}}} \left| f^2(x) - g^2(x) \right| \, dx = \pi \int_{{{label_a}}}^{{{label_b}}} \left| {sp.latex(diff_sq)} \right| \, dx")
                else:
                    st.latex(rf"V_{{Ox}} = \pi \int_{{{label_a}}}^{{{label_b}}} [f(x)]^2 \, dx = \pi \int_{{{label_a}}}^{{{label_b}}} {sp.latex(diff_sq)} \, dx")

                # Bước 3
                st.markdown("**Bước 3. Tìm một nguyên hàm của biểu thức dưới dấu tích phân:**")
                H_expr = sp.integrate(diff_sq, x_sym)
                st.latex(rf"H(x) = \int \left( {sp.latex(diff_sq)} \right) \, dx = {sp.latex(H_expr)} + C")

                # Bước 4
                st.markdown("**Bước 4. Tách tích phân theo các nghiệm và tính thể tích:**")
                if len(res.exact_intervals) > 1:
                    sum_parts = []
                    val_parts = []
                    for idx_inv, (sub_a, sub_b) in enumerate(res.exact_intervals):
                        sa_str = sp.latex(sub_a)
                        sb_str = sp.latex(sub_b)
                        val_a = H_expr.subs(x_sym, sub_a)
                        val_b = H_expr.subs(x_sym, sub_b)
                        sub_vol = sp.simplify(sp.Abs(val_b - val_a))
                        sub_vol_latex = sp.latex(sub_vol)
                        sum_parts.append(rf"\left| H({sb_str}) - H({sa_str}) \right|")
                        val_parts.append(sub_vol_latex)
                        st.latex(
                            rf"V_{{{idx_inv + 1}}} = \pi \int_{{{sa_str}}}^{{{sb_str}}} \left| {sp.latex(diff_sq)} \right| dx "
                            rf"= \pi \left| H({sb_str}) - H({sa_str}) \right| = \pi \cdot {sub_vol_latex}"
                        )
                    st.latex(
                        rf"\Rightarrow V_{{Ox}} = \pi \left( " + " + ".join(sum_parts) + 
                        rf" \right) = {res.latex_str} \approx {res.numeric_value:.6f}"
                    )
                else:
                    st.latex(
                        rf"V_{{Ox}} = \pi \left| H({label_b}) - H({label_a}) \right| = {res.latex_str} \approx {res.numeric_value:.6f}"
                    )
        tikz_code = engine.generate_tikz(f_expr, g_expr, a=a_val, b=b_val, intervals=res.intervals, exact_intervals=res.exact_intervals, is_volume=is_volume)

        if is_volume:
            if not has_g:
                fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=130)
                draw_single_revolution_solid(
                    ax=ax, h_expr=f_expr, x_var=x_sym, x_a=x_a, x_b=x_b,
                    label_a=label_a, label_b=label_b,
                    title_str=rf"Khối tròn xoay do $y = {sp.latex(f_expr)}$ quay quanh $Ox$",
                    main_color="#005f73", hatch_color="#c2185b", aspect_ratio=axis_ratio,
                )
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=130)
                draw_single_revolution_solid(
                    ax=ax1, h_expr=f_expr, x_var=x_sym, x_a=x_a, x_b=x_b,
                    label_a=label_a, label_b=label_b,
                    title_str=rf"(1) Khối $V_1$ do $y = {sp.latex(f_expr)}$ quay",
                    main_color="#005f73", hatch_color="#c2185b", aspect_ratio=axis_ratio,
                )
                draw_single_revolution_solid(
                    ax=ax2, h_expr=g_expr, x_var=x_sym, x_a=x_a, x_b=x_b,
                    label_a=label_a, label_b=label_b,
                    title_str=rf"(2) Khối $V_2$ do $y = {sp.latex(g_expr)}$ quay",
                    main_color="#ca6702", hatch_color="#d97706", aspect_ratio=axis_ratio,
                )
                fig.subplots_adjust(top=0.88, wspace=0.22)
        else:
            # DIỆN TÍCH HÌNH PHẲNG: ÁNH XẠ HOÀN HẢO 1-1 VỚI TIKZ
            f_n = sp.lambdify(x_sym, f_expr, "numpy")
            g_n = sp.lambdify(x_sym, g_expr, "numpy")

            xs_core = np.linspace(x_a, x_b, 350)
            yf_core = np.asarray(f_n(xs_core), dtype=float)
            yg_core = np.asarray(g_n(xs_core), dtype=float)

            y_min_core = min(np.nanmin(yf_core), np.nanmin(yg_core), 0.0)
            y_max_core = max(np.nanmax(yf_core), np.nanmax(yg_core), 0.0)

            x_axis_min = int(np.floor(min(x_a, 0.0) - 2.0))
            x_axis_max = int(np.ceil(max(x_b, 0.0) + 2.0))
            y_axis_min = int(np.floor(min(y_min_core, 0.0) - 2.0))
            y_axis_max = int(np.ceil(max(y_max_core, 0.0) + 2.0))

            x_view_min = x_axis_min - 0.5
            x_view_max = x_axis_max + 0.5
            y_view_min = y_axis_min - 0.5
            y_view_max = y_axis_max + 0.5

            fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_view_min, x_view_max)
            ax.set_ylim(y_view_min, y_view_max)
            ax.axis("off")

            # Lưới ô vuông mờ chuẩn SGK
            for gx in range(x_axis_min, x_axis_max + 1):
                ax.plot([gx, gx], [y_axis_min, y_axis_max], color="#edf2f7", lw=0.7, zorder=1)
            for gy in range(y_axis_min, y_axis_max + 1):
                ax.plot([x_axis_min, x_axis_max], [gy, gy], color="#edf2f7", lw=0.7, zorder=1)

            # Khung viền Bounding Box (Clip Box)
            grid_clip = patches.Rectangle(
                (x_axis_min, y_axis_min),
                x_axis_max - x_axis_min,
                y_axis_max - y_axis_min,
                transform=ax.transData,
                fill=False, edgecolor="#cbd5e1", lw=0.8, zorder=2
            )
            ax.add_patch(grid_clip)

            # Trục toạ độ Oxy thanh mảnh
            arrow_style = dict(arrowstyle="-|>,head_width=0.22,head_length=0.45", color="#1e293b", lw=1.1, mutation_scale=12)
            ax.annotate("", xy=(x_axis_max + 0.4, 0), xytext=(x_axis_min - 0.2, 0), arrowprops=arrow_style, zorder=3)
            ax.annotate("", xy=(0, y_axis_max + 0.4), xytext=(0, y_axis_min - 0.2), arrowprops=arrow_style, zorder=3)

            ax.text(x_axis_max + 0.55, 0, r"$x$", fontsize=12, style="italic", ha="left", va="center")
            ax.text(0, y_axis_max + 0.55, r"$y$", fontsize=12, style="italic", ha="center", va="bottom")
            ax.text(-0.25, -0.28, r"$O$", fontsize=11, style="italic", ha="right", va="top")

            for vx in range(x_axis_min, x_axis_max + 1):
                if vx == 0:
                    continue
                ax.plot([vx, vx], [-0.07, 0.07], color="#1e293b", lw=0.8, zorder=3)
                ax.text(vx, -0.25, f"{vx}", fontsize=9, ha="center", va="top", color="#334155")

            for vy in range(y_axis_min, y_axis_max + 1):
                if vy == 0:
                    continue
                ax.plot([-0.07, 0.07], [vy, vy], color="#1e293b", lw=0.8, zorder=3)
                ax.text(-0.2, vy, f"{vy}", fontsize=9, ha="right", va="center", color="#334155")

            # Tô miền diện tích khoá trong clip path
            for sub_a, sub_b in res.intervals:
                xs_sub = np.linspace(sub_a, sub_b, 120)
                ys_sub_f = f_n(xs_sub)
                ys_sub_g = g_n(xs_sub) if has_g else np.zeros_like(xs_sub)
                
                poly_fill = ax.fill_between(
                    xs_sub, ys_sub_f, ys_sub_g,
                    facecolor="#bfdbfe", edgecolor="#2563eb",
                    hatch="///", linewidth=0.8, alpha=0.75, zorder=3
                )
                poly_fill.set_clip_path(grid_clip)

            # Vẽ đồ thị khoá trong clip path
            xs_plot = np.linspace(x_view_min, x_view_max, 600)
            line_f, = ax.plot(xs_plot, f_n(xs_plot), color="#2563eb", linewidth=1.8, zorder=4)
            line_f.set_clip_path(grid_clip)

            if has_g:
                line_g, = ax.plot(xs_plot, g_n(xs_plot), color="#dc2626", linewidth=1.7, zorder=4)
                line_g.set_clip_path(grid_clip)

            # Nhãn (C) và (d)
            ax.text(
                x_view_min + 0.3, y_view_max - 0.35,
                rf"$(C): y = {sp.latex(f_expr)}$",
                color="#2563eb", fontsize=11, weight="bold",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.7, alpha=0.95),
                zorder=7
            )
            if has_g:
                ax.text(
                    x_view_min + 0.25, y_view_min + 0.35,
                    rf"$(d): y = {sp.latex(g_expr)}$",
                    color="#dc2626", fontsize=11, weight="bold",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=0.7, alpha=0.95),
                    zorder=7
                )

            # Đường dóng vuông góc tại các cận a và b
            ax.plot([x_a, x_a], [0, f_n(x_a)], color="#334155", linestyle=":", linewidth=1.1, zorder=3)
            ax.plot([x_b, x_b], [0, f_n(x_b)], color="#334155", linestyle=":", linewidth=1.1, zorder=3)

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
