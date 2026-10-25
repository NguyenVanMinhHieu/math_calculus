"""Web App Giải tích Hình học THPT - Mô phỏng toàn diện |f(x) - g(x)| xoay quanh Ox."""

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
    """Tự động chuyển đổi: 2x -> 2*x, x^3 -> x**3."""
    if not s or s.strip() == "0":
        return "0"
    s = s.strip().replace("^", "**")
    s = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'(\))(\()', r'\1*\2', s)
    return s

st.sidebar.header("⚙️ Tùy chọn bài toán")
mode = st.sidebar.radio(
    "Dạng toán cần tính:",
    ["1. Diện tích hình phẳng", "2. Thể tích tròn xoay (quanh Ox)"],
)
is_volume = "Thể tích" in mode

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Ví dụ mẫu:**")
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

col_input, col_view = st.columns([1, 1.4], gap="large")

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
    title_res = "🎯 Khối tròn xoay của |f(x) - g(x)|" if is_volume else "🎯 Hình phẳng chuẩn Đề thi"
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

            all_pts = [pt for interval in res.intervals for pt in interval]
            x_a, x_b = min(all_pts), max(all_pts)
            label_a = f"{x_a:g}"
            label_b = f"{x_b:g}"
            if res.exact_intervals:
                label_a = sp.latex(res.exact_intervals[0][0])
                label_b = sp.latex(res.exact_intervals[-1][1])

            tikz_code = engine.generate_tikz(f_expr, g_expr, a=a_val, b=b_val, intervals=res.intervals, exact_intervals=res.exact_intervals, is_volume=is_volume)

            # DỰNG HÌNH CHUẨN XÁC
            fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            f_num = sp.lambdify(x_sym, f_expr, modules=["numpy"])
            g_num = sp.lambdify(x_sym, g_expr, modules=["numpy"])

            span = max(x_b - x_a, 1.0)
            x_min = x_a - span * 0.25
            x_max = x_b + span * 0.35

            xs_curve = np.linspace(x_a, x_b, 300)
            yf = np.asarray(f_num(xs_curve), dtype=float)
            yg = np.asarray(g_num(xs_curve), dtype=float)
            if yf.ndim == 0: yf = np.full_like(xs_curve, yf)
            if yg.ndim == 0: yg = np.full_like(xs_curve, yg)

            max_r = max(np.max(np.abs(yf)), np.max(np.abs(yg)), 1.0)
            y_lim = max_r * 1.35
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-y_lim, y_lim)

            if is_volume:
                # 1. Trục Ox xuyên tâm (đoạn trong [a, b] nét đứt)
                ax.plot([x_min, x_a], [0, 0], color="black", linewidth=1.2)
                ax.plot([x_a, x_b], [0, 0], color="black", linestyle="--", linewidth=1.2)
                ax.plot([x_b, x_max], [0, 0], color="black", linewidth=1.2)
                ax.plot(x_max, 0, ">k", clip_on=False, markersize=5)
                ax.text(x_max, -y_lim * 0.08, r"$x$", fontsize=12, style="italic")

                # Chấm mốc a và b
                ax.plot(x_a, 0, "ko", markersize=4.5)
                ax.plot(x_b, 0, "ko", markersize=4.5)
                ax.text(x_a, -y_lim * 0.14, rf"${label_a}$", fontsize=11, ha="center")
                ax.text(x_b, -y_lim * 0.14, rf"${label_b}$", fontsize=11, ha="center")

                # 2. Miền phẳng gốc (H) giữa f(x) và g(x) (Gạch sọc)
                ax.fill_between(xs_curve, yf, yg, facecolor="#fce4ec", edgecolor="#c2185b", hatch="///", linewidth=0.6, alpha=0.9)

                # Miền phản chiếu đối xứng qua Ox
                ax.fill_between(xs_curve, -yf, -yg, facecolor="#e1f5fe", edgecolor="#0288d1", hatch="\\\\\\", linewidth=0.5, alpha=0.45)

                # 3. Hai đường cong gốc và hai đường đối xứng qua Ox
                ax.plot(xs_curve, yf, color="#005f73", linewidth=2.0, label=rf"$y = {sp.latex(f_expr)}$")
                ax.plot(xs_curve, -yf, color="#005f73", linestyle="--", linewidth=1.2, alpha=0.7)
                if has_g:
                    ax.plot(xs_curve, yg, color="#ca6702", linewidth=1.8, label=rf"$y = {sp.latex(g_expr)}$")
                    ax.plot(xs_curve, -yg, color="#ca6702", linestyle="--", linewidth=1.2, alpha=0.7)

                # 4. Mặt cắt Elip tại a và b (Washer Ellipses)
                rx = span * 0.045
                fa, ga = abs(float(yf[0])), abs(float(yg[0]))
                fb, gb = abs(float(yf[-1])), abs(float(yg[-1]))
                r_out_a, r_in_a = max(fa, ga), min(fa, ga)
                r_out_b, r_in_b = max(fb, gb), min(fb, gb)

                # Tại a:
                t_in = np.linspace(-np.pi/2, np.pi/2, 60)
                t_out = np.linspace(np.pi/2, 3*np.pi/2, 60)
                if r_out_a > 0.1:
                    ax.plot(x_a + rx * np.cos(t_in), r_out_a * np.sin(t_in), color="#005f73", linestyle="--", linewidth=1.1)
                    ax.plot(x_a + rx * np.cos(t_out), r_out_a * np.sin(t_out), color="#005f73", linestyle="-", linewidth=1.8)
                if has_g and r_in_a > 0.15 and abs(r_out_a - r_in_a) > 0.1:
                    ax.plot(x_a + rx * 0.7 * np.cos(t_in), r_in_a * np.sin(t_in), color="#ca6702", linestyle="--", linewidth=0.9)
                    ax.plot(x_a + rx * 0.7 * np.cos(t_out), r_in_a * np.sin(t_out), color="#ca6702", linestyle="--", linewidth=1.2)

                # Tại b:
                t_full = np.linspace(0, 2*np.pi, 120)
                if r_out_b > 0.1:
                    ax.plot(x_b + rx * np.cos(t_full), r_out_b * np.sin(t_full), color="#005f73", linestyle="-", linewidth=1.8)
                if has_g and r_in_b > 0.15 and abs(r_out_b - r_in_b) > 0.1:
                    ax.plot(x_b + rx * 0.7 * np.cos(t_full), r_in_b * np.sin(t_full), color="#ca6702", linestyle="-", linewidth=1.3)

                ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=9)
            else:
                # 2D thông thường
                ax.spines["left"].set_visible(True)
                ax.spines["bottom"].set_visible(True)
                ax.spines["left"].set_position(("data", 0))
                ax.spines["bottom"].set_position(("data", 0))
                ax.plot(xs_curve, yf, color="#005f73", linewidth=2.0, label=rf"$y = {sp.latex(f_expr)}$")
                if has_g:
                    ax.plot(xs_curve, yg, color="#ca6702", linestyle="--", linewidth=1.8, label=rf"$y = {sp.latex(g_expr)}$")
                ax.fill_between(xs_curve, yf, yg, facecolor="#edf4f9", edgecolor="#1d3557", hatch="///", alpha=0.85)
                ax.legend(loc="upper left", frameon=True, fontsize=9)

            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_data = svg_buffer.getvalue()

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
            png_data = png_buffer.getvalue()
            plt.close(fig)

            tab_img, tab_tikz = st.tabs(["📊 Khối tròn xoay |f(x) - g(x)|", "📋 Mã nguồn TikZ (LaTeX)"])

            with tab_img:
                st.image(svg_data, use_container_width=True)
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.download_button("📥 Tải ảnh SVG (Vector)", data=svg_data, file_name="khoi_tron_xoay.svg", mime="image/svg+xml", use_container_width=True)
                with c_btn2:
                    st.download_button("📥 Tải ảnh PNG (300 DPI)", data=png_data, file_name="khoi_tron_xoay.png", mime="image/png", use_container_width=True)

            with tab_tikz:
                st.info("💡 Mã TikZ chuẩn xác cho bài toán quay |f(x) - g(x)| quanh Ox:")
                st.code(tikz_code, language="latex")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
