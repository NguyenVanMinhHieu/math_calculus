"""Module tính toán Giải tích Hình học: Diện tích hình phẳng và Thể tích tròn xoay.

Được thiết kế tối ưu cho giảng dạy toán học, hỗ trợ cả nghiệm giải tích (SymPy)
và xấp xỉ số học.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import sympy as sp
from sympy.core.expr import Expr

# Cấu hình logging chuẩn mực
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CalculusEngine")


@dataclass(frozen=True)
class MathResult:
    """Dataclass chứa kết quả tính toán hình học giải tích."""

    exact_value: Expr
    numeric_value: float
    latex_str: str
    intervals: List[Tuple[float, float]]
    exact_intervals: List[Tuple[Expr, Expr]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Kết quả chính xác: {self.exact_value}\n"
            f"Giá trị xấp xỉ   : {self.numeric_value:.6f}\n"
            f"Mã LaTeX         : {self.latex_str}"
        )


class CalculusGeometryEngine:
    """Engine xử lý các phép toán tích phân ứng dụng trong hình học phẳng và không gian."""

    def __init__(self, variable_name: str = "x") -> None:
        self.var: sp.Symbol = sp.Symbol(variable_name, real=True)
        logger.info("Khởi tạo CalculusGeometryEngine với biến độc lập: %s", self.var)

    def _parse_expression(self, expr_input: Union[str, Expr, float, int]) -> Expr:
        if isinstance(expr_input, Expr):
            return expr_input
        try:
            parsed = sp.sympify(str(expr_input), locals={str(self.var): self.var})
            return parsed
        except (sp.SympifyError, TypeError, SyntaxError) as err:
            logger.error("Lỗi phân tích cú pháp biểu thức: %s", expr_input)
            raise ValueError(f"Biểu thức không hợp lệ: {expr_input}") from err

    def _find_real_roots(
        self,
        diff_expr: Expr,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> List[Expr]:
        found_roots: List[Expr] = []
        numeric_roots: List[float] = []

        try:
            sol_set = sp.solveset(diff_expr, self.var, domain=sp.S.Reals)
            if isinstance(sol_set, sp.FiniteSet):
                for r in sol_set:
                    found_roots.append(r)
                    numeric_roots.append(float(r.evalf()))
        except Exception:
            pass

        try:
            poly = sp.Poly(diff_expr, self.var)
            for r in poly.real_roots():
                v = float(r.evalf())
                if not any(abs(v - nr) < 1e-5 for nr in numeric_roots):
                    found_roots.append(r)
                    numeric_roots.append(v)
        except Exception:
            pass

        try:
            sym_roots = sp.solve(diff_expr, self.var)
            for r in sym_roots:
                try:
                    val = complex(r.evalf())
                    if abs(val.imag) < 1e-7:
                        v = val.real
                        if not any(abs(v - nr) < 1e-5 for nr in numeric_roots):
                            found_roots.append(sp.Float(v))
                            numeric_roots.append(v)
                except Exception:
                    continue
        except Exception:
            pass

        if lower is not None and upper is not None and lower < upper:
            h_fn = sp.lambdify(self.var, diff_expr, modules=["numpy", "math"])
            num_samples = 400
            step = (upper - lower) / num_samples
            xs = [lower + i * step for i in range(num_samples + 1)]
            ys = []
            valid_xs = []
            for x in xs:
                try:
                    y = float(h_fn(x))
                    if not (sp.isnan(y) or sp.isinf(y)):
                        ys.append(y)
                        valid_xs.append(x)
                    else:
                        ys.append(None)
                        valid_xs.append(x)
                except Exception:
                    ys.append(None)
                    valid_xs.append(x)

            for j in range(len(valid_xs) - 1):
                y1, y2 = ys[j], ys[j + 1]
                x1, x2 = valid_xs[j], valid_xs[j + 1]
                if y1 is None or y2 is None:
                    continue
                if y1 * y2 <= 0:
                    x_mid = (x1 + x2) / 2.0
                    try:
                        root_num = sp.nsolve(diff_expr, self.var, x_mid)
                        v = float(root_num.evalf())
                        if not any(abs(v - nr) < 1e-4 for nr in numeric_roots):
                            found_roots.append(root_num)
                            numeric_roots.append(v)
                    except Exception:
                        if not any(abs(x_mid - nr) < 1e-3 for nr in numeric_roots):
                            r_sym = sp.Rational(str(round(x_mid, 5)))
                            found_roots.append(r_sym)
                            numeric_roots.append(x_mid)

        final_roots: List[Expr] = []
        for r in found_roots:
            try:
                val = float(r.evalf())
                if lower is not None and val < lower - 1e-7:
                    continue
                if upper is not None and val > upper + 1e-7:
                    continue
                final_roots.append(r)
            except Exception:
                continue

        final_roots.sort(key=lambda x: float(x.evalf()))
        return final_roots

    def calculate_area(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ) -> MathResult:
        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        diff = f - g

        roots = self._find_real_roots(diff, a, b)

        partition: List[Any] = []
        if a is not None:
            partition.append(sp.sympify(a))
        partition.extend([r for r in roots if (a is None or float(r.evalf()) > a) and (b is None or float(r.evalf()) < b)])
        if b is not None:
            partition.append(sp.sympify(b))

        unique_partition = []
        for p in sorted(partition, key=lambda x: float(x.evalf())):
            val_p = float(p.evalf())
            if not any(abs(val_p - float(u.evalf())) < 1e-5 for u in unique_partition):
                unique_partition.append(p)
        partition = unique_partition

        if len(partition) < 2:
            raise ValueError("Không đủ cận để tính tích phân.")

        total_area: Expr = sp.S.Zero
        intervals_used: List[Tuple[float, float]] = []

        for i in range(len(partition) - 1):
            sub_a, sub_b = partition[i], partition[i + 1]
            sub_integral = sp.integrate(diff, (self.var, sub_a, sub_b))
            total_area += sp.Abs(sub_integral)
            intervals_used.append((float(sub_a.evalf()), float(sub_b.evalf())))

        total_area = sp.nsimplify(total_area, constants=[sp.pi])
        total_area = sp.trigsimp(total_area)
        total_area = sp.simplify(total_area)
        numeric_val = float(total_area.evalf())

        exact_intervals_used = [(partition[i], partition[i+1]) for i in range(len(partition) - 1)]
        return MathResult(
            exact_value=total_area,
            numeric_value=numeric_val,
            latex_str=sp.latex(total_area),
            intervals=intervals_used,
            exact_intervals=exact_intervals_used,
        )

    def calculate_rotation_volume_ox(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ) -> MathResult:
        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        diff_sq = sp.simplify(f**2 - g**2)

        roots = self._find_real_roots(diff_sq, a, b)

        partition: List[Any] = []
        if a is not None:
            partition.append(sp.nsimplify(a, rational=True))
        partition.extend([r for r in roots if (a is None or float(r.evalf()) > a) and (b is None or float(r.evalf()) < b)])
        if b is not None:
            partition.append(sp.nsimplify(b, rational=True))
        unique_partition = []
        for p in sorted(partition, key=lambda x: float(x.evalf())):
            val_p = float(p.evalf())
            if not any(abs(val_p - float(u.evalf())) < 1e-5 for u in unique_partition):
                unique_partition.append(p)
        partition = unique_partition

        if len(partition) < 2:
            raise ValueError("Không đủ cận xác định để tính thể tích quay quanh Ox.")

        total_integral: Expr = sp.S.Zero
        intervals_used: List[Tuple[float, float]] = []

        for i in range(len(partition) - 1):
            sub_a, sub_b = partition[i], partition[i + 1]
            sub_val = sp.integrate(diff_sq, (self.var, sub_a, sub_b))
            total_integral += sp.Abs(sub_val)
            intervals_used.append((float(sub_a.evalf()), float(sub_b.evalf())))

        total_volume = sp.simplify(sp.pi * total_integral)
        numeric_val = float(total_volume.evalf())

        exact_intervals_used = [(partition[i], partition[i+1]) for i in range(len(partition) - 1)]
        return MathResult(
            exact_value=total_volume,
            numeric_value=numeric_val,
            latex_str=sp.latex(total_volume),
            intervals=intervals_used,
            exact_intervals=exact_intervals_used,
        )

    def calculate_rotation_volume_oy(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: float = 0.0,
        b: float = 1.0,
    ) -> MathResult:
        if a < 0 or b < 0:
            raise ValueError("Phương pháp Shell yêu cầu dải tích phân x >= 0.")

        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        diff = f - g

        roots = self._find_real_roots(diff, a, b)

        partition = [sp.sympify(a)]
        partition.extend([r for r in roots if float(r.evalf()) > a and float(r.evalf()) < b])
        partition.append(sp.sympify(b))
        
        unique_partition = []
        for p in sorted(partition, key=lambda x: float(x.evalf())):
            val_p = float(p.evalf())
            if not any(abs(val_p - float(u.evalf())) < 1e-5 for u in unique_partition):
                unique_partition.append(p)
        partition = unique_partition

        total_integral: Expr = sp.S.Zero
        intervals_used: List[Tuple[float, float]] = []

        for i in range(len(partition) - 1):
            sub_a, sub_b = partition[i], partition[i + 1]
            kernel = self.var * diff
            sub_val = sp.integrate(kernel, (self.var, sub_a, sub_b))
            total_integral += sp.Abs(sub_val)
            intervals_used.append((float(sub_a.evalf()), float(sub_b.evalf())))

        total_volume = sp.simplify(2 * sp.pi * total_integral)
        numeric_val = float(total_volume.evalf())

        exact_intervals_used = [(partition[i], partition[i+1]) for i in range(len(partition) - 1)]
        return MathResult(
            exact_value=total_volume,
            numeric_value=numeric_val,
            latex_str=sp.latex(total_volume),
            intervals=intervals_used,
            exact_intervals=exact_intervals_used,
        )

    def generate_tikz(
        self,
        f_expr: Union[str, Expr],
        g_expr: Union[str, Expr] = 0,
        a: Optional[float] = None,
        b: Optional[float] = None,
        intervals: Optional[List[Tuple[float, float]]] = None,
        exact_intervals: Optional[List[Tuple[Expr, Expr]]] = None,
        is_volume: bool = False,
    ) -> str:
        r"""Sinh mã TikZ chuẩn Sư phạm: Đồng bộ 1-1 tuyệt đối với SVG (Có lệnh \clip khoá viền)."""
        f = self._parse_expression(f_expr)
        g = self._parse_expression(g_expr)
        
        f_tikz = str(f).replace(str(self.var), r"\x").replace("**", "^")
        g_tikz = str(g).replace(str(self.var), r"\x").replace("**", "^")
        has_g = str(g) != "0"

        if intervals and len(intervals) > 0:
            all_pts = [p for inv in intervals for p in inv]
            x_a, x_b = min(all_pts), max(all_pts)
        else:
            x_a, x_b = (a if a is not None else 0.0), (b if b is not None else 2.0)

        label_a = str(x_a)
        label_b = str(x_b)
        if exact_intervals and len(exact_intervals) > 0:
            label_a = sp.latex(exact_intervals[0][0])
            label_b = sp.latex(exact_intervals[-1][1])

        x_min, x_max = -5, 5
        y_min, y_max = -5, 5

        if is_volume:
            return rf"""% === MÃ TIKZ KHỐI TRÒN XOAY CHUẨN SGK ===
\begin{{tikzpicture}}[xscale=1.0, yscale=1.0, >=stealth]
    \draw[thick] ({x_min:.2f}, 0) -- ({x_a:.2f}, 0);
    \draw[thick, dashed] ({x_a:.2f}, 0) -- ({x_b:.2f}, 0);
    \draw[->, thick] ({x_b:.2f}, 0) -- ({x_max:.2f}, 0) node[below] {{$x$}};
    \fill ({x_a:.2f}, 0) circle (1.5pt) node[below=2pt] {{${label_a}$}};
    \fill ({x_b:.2f}, 0) circle (1.5pt) node[below=2pt] {{${label_b}$}};
    \fill[pattern=north east lines, pattern color=magenta!80]
        plot[domain={x_a:.2f}:{x_b:.2f}, samples=100] (\x, {{abs({f_tikz})}}) --
        ({x_b:.2f}, 0) -- ({x_a:.2f}, 0) -- cycle;
    \draw[thick, blue!85!black, domain={x_a:.2f}:{x_b:.2f}, samples=120] plot (\x, {{abs({f_tikz})}});
    \draw[thick, blue!85!black, domain={x_a:.2f}:{x_b:.2f}, samples=120] plot (\x, {{-abs({f_tikz})}});
\end{{tikzpicture}}"""

        active_intervals = intervals if intervals else [(x_a, x_b)]
        fill_commands: List[str] = []
        for sub_a, sub_b in active_intervals:
            if has_g:
                fill_cmd = (
                    f"    \\fill[pattern=north east lines, pattern color=blue!40]\n"
                    f"        plot[domain={sub_a:.4f}:{sub_b:.4f}, samples=100] (\\x, {{{f_tikz}}}) --\n"
                    f"        plot[domain={sub_b:.4f}:{sub_a:.4f}, samples=100] (\\x, {{{g_tikz}}}) -- cycle;"
                )
            else:
                fill_cmd = (
                    f"    \\fill[pattern=north east lines, pattern color=blue!40]\n"
                    f"        plot[domain={sub_a:.4f}:{sub_b:.4f}, samples=100] (\\x, {{{f_tikz}}}) --\n"
                    f"        ({sub_b:.4f}, 0) -- ({sub_a:.4f}, 0) -- cycle;"
                )
            fill_commands.append(fill_cmd)

        fills_str = "\n".join(fill_commands)
        
        f_eval = sp.lambdify(self.var, f, modules=["numpy"])
        fa = float(f_eval(x_a))
        fb = float(f_eval(x_b))

        return rf"""% === MÃ TIKZ MIỀN PHẲNG (H) SONG ÁNH 1-1 VỚI SVG (CÓ \clip) ===
\begin{{tikzpicture}}[xscale=1.0, yscale=1.0, >=stealth]
    % Lệnh Clip khoá toàn bộ đồ thị không bị tràn viền (Giống Clip Path trong SVG)
    \clip ({x_min}, {y_min}) rectangle ({x_max}, {y_max});

    % Lưới ô vuông mờ nhã nhặn chuẩn SGK
    \draw[step=1cm, gray!20, very thin] ({x_min}, {y_min}) grid ({x_max}, {y_max});

    % Khung viền Bounding Box
    \draw[thin, gray] ({x_min}, {y_min}) rectangle ({x_max}, {y_max});

    % Hệ trục tọa độ trực chuẩn tỷ lệ 1:1
    \draw[->, thick] ({x_min}, 0) -- ({x_max}, 0) node[below] {{$x$}};
    \draw[->, thick] (0, {y_min}) -- (0, {y_max}) node[left] {{$y$}};
    \node at (-0.3, -0.3) {{$O$}};

    % Các miền diện tích gạch sọc tự động theo phân hoạch
{fills_str}

    % Đồ thị hàm số (C) và (d)
    \draw[thick, blue!85!black, domain={x_min}:{x_max}, samples=150] plot (\x, {{{f_tikz}}}) node[above right] {{$(C)$}};
    {"\\draw[thick, red!85!black, domain=" + f"{x_min}:{x_max}" + ", samples=150] plot (\\x, {" + g_tikz + "}) node[below right] {$(d)$};" if has_g else ""}

    % Đường dóng vuông góc tại các cận a và b
    \draw[dashed, thin, darkgray] ({x_a:.4f}, 0) -- ({x_a:.4f}, {fa:.4f});
    \draw[dashed, thin, darkgray] ({x_b:.4f}, 0) -- ({x_b:.4f}, {fb:.4f});
    
    \fill ({x_a:.4f}, 0) circle (1.3pt) node[below=3pt] {{${label_a}$}};
    \fill ({x_b:.4f}, 0) circle (1.3pt) node[below=3pt] {{${label_b}$}};
\end{{tikzpicture}}"""
