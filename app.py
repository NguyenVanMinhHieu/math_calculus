"""Web App Giải tích Hình học chuẩn Sư phạm Toán THPT - Hiển thị nhãn nghiệm giải tích."""

from __future__ import annotations

import io
import re
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp

from calculus_engine import CalculusGeometryEngine

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"

st.set_page_config(page_title="Giải tích Hình học THPT", page_icon="📐", layout="wide")

st.title("📐 Ứng dụng Giải tích: Diện tích & Thể tích tròn xoay")
st.markdown("**Hỗ trợ nghiên cứu và giảng dạy Toán học - Thầy giáo Nguyễn Văn Minh Hiếu - THPT TRUNG PHÚ**")

engine = CalculusGeometryEngine()

def preprocess_math_str(s: str) -> str:
    """Tự động chuyển đổi ký hiệu: 2x -> 2*x, x^3 -> x**3."""
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

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Ví dụ mẫu:**")
if st.sidebar.button("📌 y = x^3 và y = 2x"):
    st.session_state["f_input"] = "x^3"
    st.session_state["g_input"] = "2x"
    st.session_state["use_bounds"] = True
    st.session_state["a_val"] = -1.0
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
    g_raw = st.text_input("Hàm số g(x) (trục hoành = 0):", value=st.session_state["g_input"])

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
    st.subheader("🎯 Bản vẽ Sư phạm & Lời giải")
    if btn_calc or "has_run" in st.session_state:
        st.session_state["has_run"] = True
        try:
            x_sym = engine.var
            f_expr = engine._parse_expression(f_clean)
            g_expr = engine._parse_expression(g_clean)

            if "Diện tích" in mode:
                res = engine.calculate_area(f_expr, g_expr, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Diện tích hình phẳng!")
                st.latex(rf"S = {res.latex_str} \approx {res.numeric_value:.6f}")
            else:
                res = engine.calculate_rotation_volume_ox(f_expr, g_expr, a=a_val, b=b_val)
                st.success("✅ Đã tính xong Thể tích khối tròn xoay quanh Ox!")
                st.latex(rf"V_{{Ox}} = {res.latex_str} \approx {res.numeric_value:.6f}")

            # 1. Trình bày công thức phân đoạn với nhãn giải tích (\sqrt{2}, -1, 2)
            parts_latex = []
            exact_pairs = res.exact_intervals if res.exact_intervals else [(sp.sympify(a), sp.sympify(b)) for a, b in res.intervals]
            for sa, sb in exact_pairs:
                parts_latex.append(rf"\int_{{{sp.latex(sa)}}}^{{{sp.latex(sb)}}} |f(x) - g(x)| dx")
            st.markdown("**Công thức tích phân chuẩn giải tích:**")
            st.latex("S = " + " + ".join(parts_latex))

            # 2. Sinh mã TikZ
            tikz_code = engine.generate_tikz(f_expr, g_expr, a=a_val, b=b_val, intervals=res.intervals)

            # 3. Dựng đồ thị
            fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=130)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_position(("data", 0))
            ax.spines["bottom"].set_position(("data", 0))
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)

            # TRIỆT TIÊU ĐÈ CHỮ: Tắt toàn bộ nhãn số mặc định của trục Ox
            ax.set_xticks([])

            f_num = sp.lambdify(x_sym, f_expr, modules=["numpy"])
            g_num = sp.lambdify(x_sym, g_expr, modules=["numpy"])

            all_x = [pt for interval in res.intervals for pt in interval]
            span = max(all_x) - min(all_x)
            margin_x = max(span * 0.28, 0.8)
            x_min, x_max = min(all_x) - margin_x, max(all_x) + margin_x
            xs = np.linspace(x_min, x_max, 500)

            # Lấy nét trục Oy
            x_focus = np.linspace(min(all_x), max(all_x), 300)
            yf_focus = np.asarray(f_num(x_focus), dtype=float)
            yg_focus = np.asarray(g_num(x_focus), dtype=float)
            if yf_focus.ndim == 0: yf_focus = np.full_like(x_focus, yf_focus)
            if yg_focus.ndim == 0: yg_focus = np.full_like(x_focus, yg_focus)

            y_f_min, y_f_max = min(np.min(yf_focus), np.min(yg_focus)), max(np.max(yf_focus), np.max(yg_focus))
            margin_y = max((y_f_max - y_f_min) * 0.35, 1.2)
            y_min_plot = min(y_f_min - margin_y, -0.6)
            y_max_plot = max(y_f_max + margin_y, 1.0)

            # Thêm đệm 15% phía trên để mũi tên và chữ y không bao giờ đè lên số vạch cao nhất
            y_headroom = (y_max_plot - y_min_plot) * 0.12
            y_top_arrow = y_max_plot + y_headroom
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min_plot, y_top_arrow)

            # Lọc trục Oy: loại bỏ nhãn số 0 để không bao giờ đè chữ O
            current_yticks = ax.get_yticks()
            ax.set_yticks([y for y in current_yticks if abs(y) > 0.1])

            # Mũi tên trục
            ax.plot(x_max, 0, ">k", clip_on=False, markersize=5)
            ax.plot(0, y_top_arrow, "^k", clip_on=False, markersize=5.5)

            # Chữ O, x, y
            dx = (x_max - x_min) * 0.02
            dy = (y_max_plot - y_min_plot) * 0.035
            ax.text(x_max, -dy * 1.4, r"$x$", fontsize=12, style="italic", ha="center")
            ax.text(-dx * 1.8, y_top_arrow, r"$y$", fontsize=12, style="italic", va="center")
            ax.text(-dx * 1.8, -dy * 1.4, r"$O$", fontsize=11, style="italic", bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85))

            ax.grid(True, linestyle=":", alpha=0.35, color="#888888")

            # Tô miền diện tích gạch sọc
            for sub_a, sub_b in res.intervals:
                x_fill = np.linspace(sub_a, sub_b, 250)
                yf = np.asarray(f_num(x_fill), dtype=float)
                yg = np.asarray(g_num(x_fill), dtype=float)
                if yf.ndim == 0: yf = np.full_like(x_fill, yf)
                if yg.ndim == 0: yg = np.full_like(x_fill, yg)

                ax.fill_between(
                    x_fill, yf, yg,
                    facecolor="#edf4f9", edgecolor="#1d3557",
                    hatch="///", linewidth=0.5, alpha=0.9
                )

            # Đồ thị hai hàm số
            ys_f = np.asarray(f_num(xs), dtype=float)
            ys_g = np.asarray(g_num(xs), dtype=float)
            if ys_f.ndim == 0: ys_f = np.full_like(xs, ys_f)
            if ys_g.ndim == 0: ys_g = np.full_like(xs, ys_g)

            ax.plot(xs, ys_f, label=rf"$y = {sp.latex(f_expr)}$", color="#005f73", linewidth=2.0)
            ax.plot(xs, ys_g, label=rf"$y = {sp.latex(g_expr)}$", color="#ca6702", linestyle="--", linewidth=1.8)

            # 4. GHI NHÃN GIẢI TÍCH CHUẨN XÁC (\sqrt{2}, -1, 2)
            # Tập hợp các điểm mốc từ exact_intervals
            exact_points = []
            for sa, sb in exact_pairs:
                exact_points.extend([sa, sb])
            # Khử trùng lặp theo giá trị số nhưng giữ biểu thức Expr
            unique_exact = []
            seen_floats = []
            for ep in exact_points:
                val = round(float(ep.evalf()), 5)
                if val not in seen_floats:
                    seen_floats.append(val)
                    unique_exact.append(ep)

            # Sắp xếp từ trái sang phải
            unique_exact.sort(key=lambda item: float(item.evalf()))

            for ep in unique_exact:
                pt_num = float(ep.evalf())
                f_val = float(np.asarray(f_num(pt_num)).item())
                g_val = float(np.asarray(g_num(pt_num)).item())
                target_y = f_val if abs(f_val) > abs(g_val) else g_val

                # Đường dóng nét đứt
                ax.plot([pt_num, pt_num], [0, target_y], color="#555555", linestyle=":", linewidth=0.9)
                ax.plot(pt_num, target_y, marker="o", markersize=4, color="#1d3557")

                # Không in nhãn số tại gốc O(0,0) vì đã có chữ O
                if abs(pt_num) > 0.08:
                    label_latex = sp.latex(ep)
                    # ĐẢO CHIỀU THÔNG MINH: Nếu miền nằm phía dưới, đẩy nhãn lên TRÊN trục hoành
                    is_under = target_y < -0.1
                    y_pos = dy * 1.3 if is_under else -dy * 1.5
                    va_align = "bottom" if is_under else "top"

                    ax.text(
                        pt_num,
                        y_pos,
                        rf"${label_latex}$",
                        fontsize=11,
                        ha="center",
                        va=va_align,
                        color="#111111",
                        bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
                    )

            ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e0e0e0", fontsize=9.5)

            # Xuất buffers
            svg_buffer = io.StringIO()
            fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
            svg_data = svg_buffer.getvalue()

            png_buffer = io.BytesIO()
            fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
            png_data = png_buffer.getvalue()
            plt.close(fig)

            # Tabs hiển thị
            tab_graph, tab_tikz = st.tabs(["📊 Đồ thị SGK (Vector SVG)", "📋 Mã nguồn TikZ (LaTeX)"])

            with tab_graph:
                st.image(svg_data, use_container_width=True)
                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    st.download_button(
                        label="📥 Tải ảnh SVG (In đề sắc nét)",
                        data=svg_data,
                        file_name="hinh_sgk.svg",
                        mime="image/svg+xml",
                        use_container_width=True,
                    )
                with c_btn2:
                    st.download_button(
                        label="📥 Tải ảnh PNG (Chèn Word 300 DPI)",
                        data=png_data,
                        file_name="hinh_sgk.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            with tab_tikz:
                st.info("💡 Mã TikZ chuẩn SGK:")
                st.code(tikz_code, language="latex")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
