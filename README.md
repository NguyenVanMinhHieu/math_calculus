---

### 💻 Lệnh Terminal cập nhật 1-click vào file:

Thầy copy toàn bộ khối lệnh dưới đây dán vào terminal:

```bash
cat << 'EOF' > README.md
# Ứng dụng Giải tích Sư phạm: Diện tích & Thể tích tròn xoay

## Hệ thống hỗ trợ Nghiên cứu và Giảng dạy Toán học THPT

> **Tác giả:** Thầy giáo **Nguyễn Văn Minh Hiếu**  
> **Đơn vị công tác:** Trường THPT Trung Phú - TP. Hồ Chí Minh  
> **Phiên bản:** `v1.0.0` (Locked Production Baseline)  
> **Giấy phép:** MIT License

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)
![Package Manager](https://img.shields.io/badge/managed%20by-uv-blueviolet.svg)
![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Math Engine](https://img.shields.io/badge/CAS-SymPy-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Release-v1.0.0-success.svg)

---

## 1. Giới thiệu tổng quan

**Math Calculus Engine** là ứng dụng Giải tích chuyên sâu được phát triển nhằm chuẩn hóa và trực quan hóa các bài toán **Ứng dụng tích phân tính Diện tích hình phẳng và Thể tích khối tròn xoay** trong chương trình Toán 12 (Chương trình GDPT mới & Đề thi Tốt nghiệp THPT).

Khác biệt với các công cụ tính toán số học thuần túy, phần mềm vận hành trên nền tảng **Hệ đại số máy tính (CAS - Computer Algebra System)** kết hợp triết lý **Sư phạm giải tích**:

- **Bảo toàn tính chính xác tuyệt đối (Exact Symbolic):** Giữ nguyên các biểu thức phân số tối giản và căn thức vô tỉ (như $\sqrt{3}, \pi$), triệt tiêu hiện tượng làm tròn số thập phân.
- **Trình bày chuẩn mực 4 bước Sư phạm:** Mô phỏng hoàn hảo các bước lập luận, phá dấu giá trị tuyệt đối và tính tích phân như trong giáo án giảng dạy chính khóa.
- **Hỗ trợ giáo viên soạn giảng:** Tự động xuất biểu thức dạng mã nguồn **LaTeX** và mã đồ họa **TikZ** đồng bộ 1-1 để chèn trực tiếp vào đề thi hay tài liệu bài giảng.

---

## 2. Tính năng nổi bật

### 2.1. Tính Diện tích hình phẳng chuẩn Sư phạm
- Tự động nhận diện và giải phương trình hoành độ giao điểm $f(x) - g(x) = 0$.
- Phân hoạch đoạn tích phân $[a; b]$ thành các đoạn con không đổi dấu theo các nghiệm $x_k \in (a; b)$.
- Tìm một nguyên hàm đại số:
  $$F(x) = \int [f(x) - g(x)] \, dx$$
- Thế cận, khử dấu giá trị tuyệt đối từng phần và xuất kết quả chính xác dạng căn thức đại số và xấp xỉ số thập phân.

### 2.2. Tính Thể tích tròn xoay quanh trục Ox (Washer Method)
Khắc phục hoàn toàn các sai lầm kinh điển khi áp dụng máy móc công thức $|V_1 - V_2|$:
- **Bước 1 (Giao điểm bán kính):** Giải phương trình $f^2(x) - g^2(x) = 0 \iff [f(x) - g(x)][f(x) + g(x)] = 0$. Tự động nhận diện cả các nghiệm đối xứng qua trục hoành $f(x) = -g(x)$ (nơi hai đồ thị hoán đổi vị trí bán kính trong và bán kính ngoài khi quay).
- **Bước 2 (Công thức tổng quát):** Thiết lập tích phân hình vành khăn chuẩn mực:
  $$V_{Ox} = \pi \int_a^b \left| f^2(x) - g^2(x) \right| \, dx$$
- **Bước 3 (Nguyên hàm bình phương):** Xác định nguyên hàm của hiệu hai bình phương:
  $$H(x) = \int \left( f^2(x) - g^2(x) \right) \, dx$$
- **Bước 4 (Phân rã tích phân):** Tách tích phân theo các nghiệm giao điểm, nhân hệ số $\pi$ và rút gọn biểu thức đại số chính xác.

### 2.3. Trực quan hóa hình học & Đồng bộ TikZ
- Vẽ đồ thị 2D với hệ trục tọa độ $Oxy$ chuẩn tỉ lệ, gạch sọc và tô màu miền phẳng giới hạn.
- Vẽ khối tròn xoay trực quan khi quay quanh $Ox$.
- Trình tạo mã **TikZ (LaTeX)** chuẩn hóa với lệnh `\clip` khóa viền, cho phép giáo viên copy-paste trực tiếp vào TeXstudio hoặc Overleaf.

---

## 3. Cơ sở Toán học & Thuật toán Sư phạm

### A. Bài toán Diện tích hình phẳng ($S$)
- **Bản chất hình học:** Miền phẳng giới hạn bởi hai đường cong $y = f(x)$ và $y = g(x)$ trên đoạn $[a; b]$.
- **Phương trình nghiệm:** $f(x) - g(x) = 0$.
- **Công thức vi phân:**
  $$dS = |f(x) - g(x)| \, dx$$
- **Định lượng phân đoạn:**
  $$S = \sum_{k} |F(x_{k+1}) - F(x_k)|$$

### B. Bài toán Thể tích tròn xoay quanh Ox ($V_{Ox}$)
- **Bản chất hình học:** Mặt cắt hình vành khăn (Washer) bán kính ngoài $R(x)$ và bán kính trong $r(x)$.
- **Phương trình nghiệm bán kính:**
  $$f^2(x) - g^2(x) = 0 \iff f(x) = \pm g(x)$$
- **Công thức vi phân:**
  $$dV = \pi \left( R^2(x) - r^2(x) \right) dx = \pi \left| f^2(x) - g^2(x) \right| dx$$
- **Định lượng phân đoạn:**
  $$V_{Ox} = \pi \sum_{k} |H(x_{k+1}) - H(x_k)|$$

---

## 4. Cấu trúc dự án

```text
math-calculus/
├── app.py                  # Giao diện người dùng Streamlit & Điều hướng Sư phạm
├── calculus_engine.py      # Động cơ Giải tích Đại số máy tính (SymPy Core Engine)
├── test_calculus.py        # Bộ kiểm thử tự động (Unit Tests) với Pytest
├── pyproject.toml          # Quản lý cấu hình dự án & Metadata tác giả (PEP 621)
├── uv.lock                 # Khóa toàn vẹn phiên bản dependencies
├── LICENSE                 # Giấy phép bản quyền mã nguồn mở MIT
├── README.md               # Tài liệu hướng dẫn & Bản quyền tác giả
└── src/
    └── math_calculus/      # Module lõi mở rộng giải tích nâng cao
5. Hướng dẫn Cài đặt & Khởi chạy
Dự án được tối ưu hóa đặc biệt trên môi trường Arch Linux và quản lý phụ thuộc bằng công cụ uv.
5.1. Yêu cầu hệ thống
Python >= 3.10 (Khuyên dùng Python 3.12 hoặc 3.13)
Trình quản lý gói uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh
5.2. Khởi chạy ứng dụng
# 1. Clone repository về máy
git clone https://github.com/<tai-khoan-github-cua-thay>/math-calculus.git
cd math-calculus

# 2. Đồng bộ môi trường và dependencies
uv sync

# 3. Khởi chạy ứng dụng Streamlit
uv run streamlit run app.py
Ứng dụng sẽ hoạt động tại địa chỉ: http://localhost:8501.
5.3. Kiểm định chất lượng mã nguồn
# Kiểm tra tĩnh cú pháp toàn bộ hệ thống
uv run python -m py_compile app.py calculus_engine.py

# Chạy kiểm thử tự động với Pytest
uv run pytest
6. Bản quyền tác giả & Giấy phép
Dự án này được phát triển phục vụ mục đích nghiên cứu học thuật và giảng dạy Toán học.
Tác giả: Thầy giáo Nguyễn Văn Minh Hiếu
Đơn vị: Trường THPT Trung Phú - TP. Hồ Chí Minh
Giấy phép phát hành: MIT License — Bản quyền © 2026 Nguyễn Văn Minh Hiếu.