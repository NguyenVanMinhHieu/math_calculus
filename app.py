"""Web App Giải tích Hình học THPT - Mô hình hóa Tổng quát Phép trừ 2 Khối tròn xoay."""

from __future__ import annotations

import io
import re
import matplotlib.pyplot as plt
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
) -> None:
    """Dựng một khối tròn xoay đơn chuẩn SGK (Đối xứng qua Ox + 2 Elip tại a, b)."""
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

    xs = np.linspace(x_a, x_b, 300)
    ys_top = np.abs(np.asarray(h_num(xs), dtype=float))
    if ys_top.ndim == 0:
        ys_top = np.full_like(xs, ys_top)
    ys_bot = -ys_top

    max_r = max(np.max(ys_top), 1.0)
    y_lim = max_r * 1.45
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_lim, y_lim)

    # 1. Trục Ox (Đoạn [a, b] bên trong nét đứt)
    ax.plot([x_min, x_a], [0, 0], color="black", linewidth=1.2)
    ax.plot([x_a, x_b], [0, 0], color="black", linestyle="--", linewidth=1.2)
    ax.plot([x_b, x_max], [0, 0], color="black", linewidth=1.2)
    ax.plot(x_max, 0, ">k", clip_on=False, markersize=5)
    ax.text(x_max, -y_lim * 0.08, r"$x$", fontsize=11, style="italic")

    # Điểm mốc a và b
    ax.plot(x_a, 0, "ko", markersize=4.0)
    ax.plot(x_b, 0, "ko", markersize=4.0)
    ax.text(x_a, -y_lim * 0.16, rf"${label_a}$", fontsize=11, ha="center")
    ax.text(x_b, -y_lim * 0.16, rf"${label_b}$", fontsize=11, ha="center")

    # 2. Miền phẳng gốc quay quanh Ox (Gạch sọc)
    ax.fill_between(xs, ys_top, 0, facecolor="#fce4ec", edgecolor=hatch_color, hatch="///", linewidth=0.6, alpha=0.9)

    # Đoạn dóng thẳng đứng tại a và b
    fa = float(ys_top[0])
    fb = float(ys_top[-1])
    ax.plot([x_a, x_a], [0, fa], color=hatch_color, linestyle="--", linewidth=1.1)
    ax.plot([x_b, x_b], [0, fb], color=hatch_color, linestyle="--", linewidth=1.1)

    # 3. Đường trên h(x) và đường đối xứng dưới -h(x)
    ax.plot(xs, ys_top, color=main_color, linewidth=2.0)
    ax.plot(xs, ys_bot, color=main_color, linewidth=2.0)

    # 4. Hai Elip khép kín chuẩn tại a và b
    rx = span * 0.045
    # Tại a: Nửa sau nét đứt, nửa ngoài nét liền
    if fa > 0.08:
        t_in = np.linspace(-np.pi / 2, np.pi / 2, 60)
        t_out = np.linspace(np.pi / 2, 3 * np.pi / 2, 60)
        ax.plot(x_a + rx * np.cos(t_in), fa * np.sin(t_in), color=main_color, linestyle="--", linewidth=1.1)
        ax.plot(x_a + rx * np.cos(t_out), fa * np.sin(t_out), color=main_color, linestyle="-", linewidth=1.8)

    # Tại b: Toàn bộ Elip nét liền
    if fb > 0.08:
        t_full = np.linspace(0, 2 * np.pi, 120)
        ax.plot(x_b + rx * np.cos(t_full), fb * np.sin(t_full), color=main_color, linestyle="-", linewidth=1.8)

    ax.set_title(title_str, fontsize=12, pad=10, weight="bold", color="#1d3557")

st.sidebar.header("⚙️ Tùy chọn bài toán")
mode = st.sidebar.radio(
    "Dạng toán cần tính:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)
is_volume = "Thể tích" in mode

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Ví dụ mẫu SGK:**")
if st.sidebar.button("📌 y = x^3 và y = 2x"):
    st.session_state["f_input"] = "x^3"
    st.session_state["g_input"] = "2x"
    st.session_state["use_bounds"] = True
    st.session_state["a_val"] = -1.0
    st.session_state["b_val"] = 2.0
if st.sidebar.button("📌 Parabol y = 4 - x^2 và Ox"):
    st.session_state["f_input"] = "4 - x^2"
    st.session_state["g_input"] = "0"
    st.session_state["use_bounds"] = True
    st.session_state["a_val"] = 0.0
    st.session_state["b_val"] = 2.0

if "f_input" not in st.session_state:
    st.session_state["f_input"] = "x^3"
if "g_input" not in st.session_state:
    st.session_state["g_input"] = "2x"
if "use_bounds" not in st.session_state:
    st.session_state["use_bounds"] = True
if "a_val" not in st.session_state:
    st.session_state["a_val"] = -1.0
if "b_val" not in st.session_state:
    st.session_state["b_val"] = 2.0

col_input, col_view = st.columns([1, 1.45], gap="large")

with col_input:
    st.subheader("📝 Dữ liệu đầu vào")
    f_raw = st.text_input("Hàm số f(x):", value=st.session_state["f_input"])
    g_raw = st.text_input("Hàm số g(x) (trục hoành đặt là 0):", value=st.session_state["g_input"])

    try:
        f_clean = preprocess_math_str(f_raw)
        g_clean = preprocess_math_str(g_raw)
        f_preview = engine._parse_expression(f_clean)
        g_preview = engine._parse_expression(g_clean)
        st.info(f"Xem trước: $f(x) = {sp.latex(f_preview)}$ và $g(x) = {sp.latex(g_preview)}$")
    except Exception:
        f_clean, g_clean = f_raw, g_raw

    use_custom_bounds = st.checkbox("Chỉ định cận [a, b] thủ công", value=st.session_state["use_bounds"])
    a_val: float | None = None
    b_val: float | None = None

    if use_custom_bounds:
        c1, c2 = st.columns(2)
        with c1:
            a_val = st.number_input("Cận dưới (a):", value=float(st.session_state["a_val"]))
        with c2:
            b_val = st.number_input("Cận trên (b):", value=float(st.session_state["b_val"]))

    btn_calc = st.button("🚀 Bắt đầu tính toán & Dựng hình", type="primary", use_container_width=True)

with col_view:
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

            tikz_code = engine.generate_tikz(f_expr, g_expr, a=a_val, b=b_val, intervals=res.intervals, exact_intervals=res.exact_intervals, is_volume=is_volume)

            # ====================================================
            # DỰNG HÌNH THEO Ý TƯỞNG SƯ PHẠM CỦA THẦY
            # ====================================================
            if is_volume:
                if not has_g:
                    # TRƯỜNG HỢP 1: 1 hàm cong duy nhất quanh Ox (Khối đơn chuẩn SGK)
                    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)
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
                    )
                else:
                    # TRƯỜNG HỢP 2: 2 hàm cong -> Phân tích thành phép trừ 2 khối trực quan!
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=130)
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
                    )
                    fig.suptitle(r"$\mathbf{Bản\;chất\;Sư\;phạm:\;} V = |V_1 - V_2|$ (Lấy Khối 1 trừ Khối 2)", fontsize=13, y=1.02, color="#1d3557")
            else:
                # Diện tích hình phẳng 2D
                fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)
                ax.spines["left"].set_position(("data", 0))
                ax.spines["bottom"].set_position(("data", 0))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                xs_2d = np.linspace(x_a, x_b, 300)
                f_n = sp.lambdify(x_sym, f_expr, "numpy")
                g_n = sp.lambdify(x_sym, g_expr, "numpy")
                yf = np.asarray(f_n(xs_2d), dtype=float)
                yg = np.asarray(g_n(xs_2d), dtype=float)
                ax.plot(xs_2d, yf, color="#005f73", linewidth=2.0, label=rf"$y = {sp.latex(f_expr)}$")
                if has_g:
                    ax.plot(xs_2d, yg, color="#ca6702", linestyle="--", linewidth=1.8, label=rf"$y = {sp.latex(g_expr)}$")
                ax.fill_between(xs_2d, yf, yg, facecolor="#edf4f9", edgecolor="#1d3557", hatch="///", alpha=0.85)
                ax.legend(loc="upper left")

            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_data = svg_buffer.getvalue()

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
            png_data = png_buffer.getvalue()
            plt.close(fig)

            # Tabs hiển thị
            tab_img, tab_tikz = st.tabs(["📊 Hình vẽ Sư phạm", "📋 Mã nguồn TikZ (LaTeX)"])

            with tab_img:
                st.image(svg_data, use_container_width=True)
                if is_volume and has_g:
                    st.info("💡 **Giải thích sư phạm:** Thể tích phần rỗng được tạo thành bằng cách lấy **Khối $V_1$ (do $f$ quay)** trừ đi **Khối $V_2$ (do $g$ quay)**: $V = |V_1 - V_2|$.")
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.download_button("📥 Tải ảnh SVG (Vector sắc nét)", data=svg_data, file_name="khoi_tron_xoay_su_pham.svg", mime="image/svg+xml", use_container_width=True)
                with c_btn2:
                    st.download_button("📥 Tải ảnh PNG (300 DPI)", data=png_data, file_name="khoi_tron_xoay_su_pham.png", mime="image/png", use_container_width=True)

            with tab_tikz:
                st.info("💡 Mã TikZ chuẩn SGK để Thầy copy vào đề thi:")
                st.code(tikz_code, language="latex")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
