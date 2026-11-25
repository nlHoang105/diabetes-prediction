# 🩺 HỆ THỐNG CHẨN ĐOÁN SỚM NGUY CƠ TIỂU ĐƯỜNG DỰA TRÊN HỌC MÁY (MACHINE LEARNING)

> **Học phần:** Học máy (Machine Learning) — Bài kiểm tra kỹ năng số 1  
> **Trường:** Đại học Công nghệ Thông tin & Truyền thông — Đại học Thái Nguyên (ICTU)  
> **Khoa:** Công nghệ Thông tin  
> **Giảng viên hướng dẫn:** ThS. Hà Thị Thanh  
> **Nhóm thực hiện:** Nhóm 3 (Nguyễn Lê Hoàng, Ngô Xuân Đỉnh, Hà Đức Trung, Trần Đức Đạt)  

---

## 📌 1. Giới Thiệu & Mục Tiêu Đề Tài

Bệnh đái tháo đường (Diabetes Mellitus) là một trong những bệnh lý mãn tính không lây nhiễm nguy hiểm nhất trên thế giới. Mục tiêu của dự án là xây dựng một hệ thống học máy (Machine Learning) hỗ trợ chẩn đoán và sàng lọc sớm nguy cơ mắc bệnh từ các chỉ số sinh hiệu lâm sàng thông thường (Tuổi, Giới tính, BMI, Huyết áp, Bệnh tim, Thói quen hút thuốc, HbA1c và Đường huyết tức thời).

Hệ thống được thiết kế theo chuẩn quy trình **MLOps hiện đại**, bao gồm từ tiền xử lý dữ liệu, giải quyết mất cân bằng lớp bằng kỹ thuật **SMOTE**, huấn luyện & so sánh 7 mô hình học máy theo chuẩn *Giáo trình Học máy (ICTU, 2026)*, cho đến đóng gói mô hình tối ưu sang Web Application tương tác bằng **Streamlit** sẵn sàng triển khai lên môi trường Đám mây (Cloud).

---

## 🗂️ 2. Cấu Trúc Thư Mục Dự Án (Chuẩn Cloud & MLOps)

Dự án được tái cấu trúc theo mô hình phân tách độc lập (Separation of Concerns), đáp ứng tiêu chuẩn kiểm định học thuật và triển khai Cloud:

```text
dự án/
├── app/
│   └── app.py                      # Ứng dụng Web tương tác Streamlit chẩn đoán lâm sàng
├── data/
│   ├── raw/                        # Chứa tập dữ liệu thô ban đầu (100,000 dòng)
│   │   └── diabetes_prediction_dataset.csv
│   └── processed/                  # Dữ liệu sạch, vector đặc trưng sau tiền xử lý & SMOTE
│       ├── diabetes_cleaned.csv    # Dữ liệu sạch (96,128 dòng)
│       ├── X_train_raw.csv / X_test_raw.csv
│       ├── X_train_scaled.csv / X_test_scaled.csv
│       ├── y_train.csv / y_test.csv
│       ├── X_train_resampled.csv / y_train_resampled.csv (140,232 mẫu sau SMOTE)
│       └── preprocessing_summary.json
├── models/
│   ├── best_model.pkl              # Mô hình tối ưu nhất (Gradient Boosting)
│   ├── scaler.pkl                  # Bộ chuẩn hóa thống kê (Mean & Std Z-score)
│   └── model_evaluation_summary.json # Thống kê chi tiết kết quả 4 mô hình
├── figures/                        # 10 biểu đồ trực quan hóa dữ liệu & mô hình (PNG 300 DPI)
│   ├── 01_class_distribution.png
│   ├── 02_vital_signs_distribution.png
│   ├── 03_vital_signs_by_diabetes.png
│   ├── 04_categorical_risk.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_preprocessing_effects.png
│   ├── 07_model_comparison_metrics.png
│   ├── 08_confusion_matrices.png
│   ├── 09_roc_curves.png
│   └── 10_feature_importance.png
├── docs/                           # Báo cáo học thuật và bộ thuyết trình Nhóm 3
│   ├── Bao_Cao_Hoc_May_Nhom_3.docx / .pdf
│   └── Bo_Thuyet_Trinh_Nhom_3.docx / .pdf
├── diabetes_ml_pipeline.py         # Pipeline huấn luyện, đánh giá 7 mô hình & xuất biểu đồ
├── Bai_Kiem_Tra_Ky_Nang_1.ipynb    # Jupyter Notebook hoàn chỉnh có đầy đủ markdown giải thích
├── main.py                         # File chạy thực thi kiểm thử nhanh CLI
├── run.bat / run.ps1               # 1-Click chạy CLI Pipeline huấn luyện trên Windows
├── run_web.bat / run_web.ps1       # 1-Click khởi động Web App Streamlit tương tác
├── install_requirements.bat        # 1-Click cài đặt toàn bộ thư viện cần thiết
├── requirements.txt                # Danh sách thư viện phụ thuộc cho Local & Cloud
└── README.md                       # Hướng dẫn chi tiết sử dụng và triển khai Cloud
```

---

## 📊 3. Kết Quả Huấn Luyện & So Sánh Mô Hình

Mô hình được huấn luyện trên tập dữ liệu đã qua cân bằng lớp bằng **SMOTE** (140,232 mẫu) và đánh giá độc lập trên tập kiểm thử **19,227 mẫu** (không bị rò rỉ dữ liệu - Data Leakage):

| Mô Hình Thuật Toán | Giáo Trình | Accuracy | Recall | Precision | F1-Score | ROC-AUC | Thời Gian |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN (k=5)** | Ch. 5 | 90.67% | 79.38% | 48.28% | 60.04% | 91.23% | 0.43s |
| **Naïve Bayes (GaussianNB)** | Ch. 6 | 85.26% | 79.49% | 35.17% | 48.76% | 90.58% | 0.05s |
| **Decision Tree (CART)** | Ch. 8 | 85.72% | **92.40%** | 37.48% | 53.32% | 96.37% | 0.19s |
| **Logistic Regression** | Ch. 9 | 88.46% | 87.45% | 42.53% | 57.23% | 95.95% | 0.21s |
| **Linear SVM (Calibrated)** | Ch. 10 | 88.57% | 87.86% | 42.81% | 57.57% | 95.95% | 0.44s |
| **Random Forest (100 cây)** | Ch. 14 | 91.22% | 86.68% | 50.15% | 63.54% | 97.24% | 2.74s |
| **🏆 Gradient Boosting** | Ensemble | **96.27%** | 75.37% | **81.05%** | **78.11%** | **97.54%** | 21.18s |

> **Nhận định chuyên môn:**
> - **Gradient Boosting Classifier** được lựa chọn làm **mô hình chính thức để đóng gói lên ứng dụng web** nhờ chỉ số toàn diện vượt bậc: Accuracy đạt **96.27%**, ROC-AUC đạt **97.54%**, và tỷ lệ dự đoán dương tính chính xác (Precision) đạt **81.05%**, giảm thiểu tối đa các cảnh báo sai.
> - **Đặc trưng đóng góp lớn nhất (Feature Importance):** Lượng đường huyết tức thời (`blood_glucose_level`) và mức đường huyết hóa huyết sắc tố (`HbA1c_level`) chiếm tới hơn 70% tổng trọng số quyết định chẩn đoán.

---

## 🚀 4. Hướng Dẫn Cài Đặt & Chạy Tại Máy Cục Bộ (Local)

### 4.1. Cài đặt môi trường
Bạn có thể cài đặt tự động bằng cách nhấp đúp chuột vào file `install_requirements.bat`, hoặc chạy lệnh sau trong Terminal / PowerShell:
```bash
pip install -r requirements.txt
```

### 4.2. Khởi chạy Ứng dụng Web Tương tác (Streamlit Demo)
- **Cách 1 (Nhanh nhất trên Windows):** Nhấp đúp chuột vào file `run_web.bat`.
- **Cách 2 (Dùng lệnh):**
  ```bash
  streamlit run app/app.py
  ```
  Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

### 4.3. Khởi chạy Pipeline Huấn luyện Toàn Diện (CLI)
- **Cách 1 (Nhanh nhất trên Windows):** Nhấp đúp chuột vào file `run.bat`.
- **Cách 2 (Dùng lệnh):**
  ```bash
  python main.py
  # hoặc bỏ qua vẽ biểu đồ để chạy nhanh:
  python main.py --no-eda
  ```

### 4.4. Chạy trên Jupyter Notebook
Mở trình duyệt Jupyter Lab hoặc VS Code, mở tệp `Bai_Kiem_Tra_Ky_Nang_1.ipynb` và chọn `Run All Cells`. Toàn bộ dữ liệu, biểu đồ EDA và ma trận đánh giá sẽ hiển thị trực quan.

---

## ☁️ 5. Hướng Dẫn Triển Khai Lên Nền Tảng Đám Mây (Cloud Deployment)

Dự án đã sẵn sàng 100% để triển khai lên các dịch vụ Cloud miễn phí phổ biến hiện nay:

### 🌐 Cách 1: Triển khai lên Streamlit Community Cloud (Khuyến nghị số 1)
1. **Đẩy mã nguồn lên GitHub:**
   - Tạo một kho chứa (Repository) trên GitHub (ví dụ: `diabetes-prediction-ml`).
   - Đẩy toàn bộ source code của thư mục dự án lên nhánh `main`.
2. **Triển khai ứng dụng:**
   - Truy cập vào [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
   - Nhấn **New app**.
   - Chọn Repository vừa tạo, nhánh `main`.
   - Tại dòng **Main file path**, điền: `app/app.py`.
   - Nhấn nút **Deploy!**.
3. **Hoàn thành:** Hệ thống sẽ tự động cài các gói trong `requirements.txt` và cung cấp đường dẫn web công khai (ví dụ: `https://diabetes-prediction-ictu.streamlit.app`) để gửi cho giảng viên chấm điểm.

---

### 🐳 Cách 2: Triển khai lên Hugging Face Spaces (Streamlit SDK)
1. Đăng nhập [Hugging Face](https://huggingface.co/), nhấn **New Space**.
2. Đặt tên Space và chọn Space SDK là **Streamlit**.
3. Clone repository của Space về máy, copy các thư mục `app/`, `models/`, `figures/` và file `requirements.txt` vào.
4. Đổi tên `app/app.py` thành `app.py` ở thư mục gốc (hoặc cấu hình đường dẫn trong `README.md` của Space: `app_file: app/app.py`).
5. Git push lên Hugging Face, Space sẽ tự động build và chạy trực tiếp.

---

### 🚢 Cách 3: Đóng gói Docker Container (Render / Fly.io / AWS)
Nếu muốn đóng gói dưới dạng Docker image, bạn có thể tạo một file `Dockerfile` với nội dung sau:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 👥 6. Thông Tin Thành Viên Nhóm 3 (Lớp Học Máy)

| STT | Họ và Tên | Vai Trò & Đóng Góp Chính |
| :---: | :--- | :--- |
| 1 | **Nguyễn Lê Hoàng** | **Trưởng nhóm:** Thiết kế kiến trúc Pipeline, tối ưu hóa siêu tham số mô hình Gradient Boosting, xử lý SMOTE |
| 2 | **Ngô Xuân Đỉnh** | Tiền xử lý dữ liệu (loại bỏ trùng lặp, mã hóa One-Hot), xây dựng bộ biểu đồ thám hiểm dữ liệu EDA |
| 3 | **Hà Đức Trung** | Phát triển ứng dụng Web Streamlit, đóng gói mô hình nhị phân `.pkl`, cấu hình triển khai Cloud |
| 4 | **Trần Đức Đạt** | Đánh giá & so sánh chỉ số mô hình, biên soạn Báo cáo học thuật (.docx) và Kịch bản thuyết trình |

---
*Bản quyền học thuật thuộc về Nhóm 3 — Khoa Công nghệ Thông tin, Trường Đại học CNTT & TT (ICTU).*
