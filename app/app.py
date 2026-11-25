"""
========================================================================================
HỆ THỐNG CHẨN ĐOÁN SỚM NGUY CƠ TIỂU ĐƯỜNG DỰA TRÊN HỌC MÁY (MACHINE LEARNING)
Đơn vị: Trường Đại học Công nghệ Thông tin & Truyền thông (ICTU)
Học phần: Học máy (Machine Learning) - Bài kiểm tra kỹ năng số 1
Nhóm thực hiện: Nhóm 3 (Nguyễn Lê Hoàng, Ngô Xuân Đỉnh, Hà Đức Trung, Trần Đức Đạt)
Giảng viên hướng dẫn: ThS. Hà Thị Thanh
========================================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import streamlit as st

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Chẩn đoán sớm Nguy cơ Tiểu đường | AI & Machine Learning - ICTU",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện y tế nền sáng hiện đại & trang nhã
st.markdown("""
<style>
    /* Nền tổng thể sáng và êm dịu cho mắt */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Tiêu đề chính mang phong cách y tế lâm sàng */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0369a1;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    /* Khung thẻ Card y tế */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }

    /* Sidebar nền sáng chuyên nghiệp */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Tabs điều hướng */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0284c7 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
    }

    /* Huy hiệu trạng thái lâm sàng */
    .badge-safe {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #bbf7d0;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #fecaca;
    }

    /* Nút bấm chẩn đoán nổi bật */
    div[data-testid="stForm"] button[kind="secondary"],
    div[data-testid="stForm"] button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stForm"] button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 35px;
        border-top: 1px solid #e2e8f0;
        padding-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


class PortableGradientBoosting:
    """Bộ suy luận thuần NumPy cho GradientBoostingClassifier (không cần DLL scikit-learn)."""
    def __init__(self, payload):
        self.init_val = float(payload["init_val"])
        self.learning_rate = float(payload["learning_rate"])
        self.trees = payload["trees"]

    def _predict_raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        raw = np.full(X.shape[0], self.init_val, dtype=np.float64)
        for tree in self.trees:
            feat = tree["feature"]
            thresh = tree["threshold"]
            left = tree["children_left"]
            right = tree["children_right"]
            val = tree["value"]
            for i in range(X.shape[0]):
                node = 0
                while left[node] != -1:
                    if X[i, feat[node]] <= thresh[node]:
                        node = left[node]
                    else:
                        node = right[node]
                raw[i] += self.learning_rate * val[node]
        return raw

    def predict_proba(self, X):
        raw = self._predict_raw(X)
        prob1 = 1.0 / (1.0 + np.exp(-np.clip(raw, -500.0, 500.0)))
        return np.column_stack([1.0 - prob1, prob1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@st.cache_resource
def load_ml_components():
    """Tải mô hình tốt nhất và bộ chuẩn hóa từ thư mục dự án (tự động dùng bản portable nếu Windows chặn DLL)."""
    possible_portable_paths = [
        os.path.join("models", "best_model_portable.json"),
        os.path.join("data", "processed", "best_model_portable.json"),
        "best_model_portable.json",
        os.path.join("..", "models", "best_model_portable.json")
    ]
    possible_model_paths = [
        os.path.join("models", "best_model.pkl"),
        os.path.join("data", "processed", "best_model.pkl"),
        "best_model.pkl",
        os.path.join("..", "models", "best_model.pkl")
    ]
    possible_scaler_paths = [
        os.path.join("models", "scaler.pkl"),
        os.path.join("data", "processed", "scaler.pkl"),
        "scaler.pkl",
        os.path.join("..", "models", "scaler.pkl")
    ]
    possible_json_paths = [
        os.path.join("models", "model_evaluation_summary.json"),
        os.path.join("data", "processed", "model_evaluation_summary.json"),
        "model_evaluation_summary.json",
        os.path.join("..", "models", "model_evaluation_summary.json")
    ]

    model = None
    scaler = None
    eval_summary = {}

    for p in possible_model_paths:
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    model = pickle.load(f)
                break
            except (ImportError, OSError, Exception):
                model = None

    if model is None:
        for p in possible_portable_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    model = PortableGradientBoosting(json.load(f))
                break

    for p in possible_scaler_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                scaler = pickle.load(f)
            break

    for p in possible_json_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                eval_summary = json.load(f)
            break

    return model, scaler, eval_summary


# Nạp tài nguyên học máy
best_model, scaler_obj, eval_summary = load_ml_components()

# Khởi tạo session_state cho mẫu thử nghiệm nếu chưa có
if "sample_gender" not in st.session_state:
    st.session_state.sample_gender = "Nam"
    st.session_state.sample_age = 45
    st.session_state.sample_hyp = "Không"
    st.session_state.sample_heart = "Không"
    st.session_state.sample_smoking = "Chưa từng hút (never)"
    st.session_state.sample_bmi = 24.5
    st.session_state.sample_hba1c = 5.6
    st.session_state.sample_glucose = 100


def apply_sample(gender, age, hyp, heart, smoking, bmi, hba1c, glucose):
    st.session_state.sample_gender = gender
    st.session_state.sample_age = age
    st.session_state.sample_hyp = hyp
    st.session_state.sample_heart = heart
    st.session_state.sample_smoking = smoking
    st.session_state.sample_bmi = bmi
    st.session_state.sample_hba1c = hba1c
    st.session_state.sample_glucose = glucose


# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("Hệ Thống Trợ Giúp Chẩn Đoán")
    st.markdown("**Đề tài:** Dự đoán sớm nguy cơ tiểu đường từ chỉ số sinh hiệu lâm sàng.")
    st.info("""
    **Nhóm thực hiện:** Nhóm 3
    - Nguyễn Lê Hoàng
    - Ngô Xuân Đỉnh
    - Hà Đức Trung
    - Trần Đức Đạt
    
    **GVHD:** ThS. Hà Thị Thanh  
    **Đơn vị:** Khoa CNTT - ICTU
    """)

    st.markdown("---")
    st.subheader("📌 Thông số Mô hình Tối ưu")
    st.markdown("""
    - **Thuật toán:** Gradient Boosting Classifier
    - **Độ chính xác (Accuracy):** `96.27%`
    - **Diện tích ROC (AUC):** `97.54%`
    - **Độ thu hồi (Recall):** `75.37%`
    - **Điểm F1-Score:** `78.11%`
    - **Kỹ thuật lấy mẫu:** SMOTE Resampling
    """)

    st.markdown("---")
    st.caption("© 2026 Nhóm 3 - Học máy ứng dụng y tế. All rights reserved.")


# HEADER TRANG CHÍNH
st.markdown('<div class="main-header">🩺 HỆ THỐNG CHẨN ĐOÁN SỚM NGUY CƠ TIỂU ĐƯỜNG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ứng dụng Mô hình Học máy Gradient Boosting trên Tập dữ liệu Lâm sàng 100,000 Bệnh nhân</div>', unsafe_allow_html=True)

# TABS ĐIỀU HƯỚNG CHÍNH
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Chẩn Đoán Bệnh Nhân",
    "📊 So Sánh 7 Mô Hình ML",
    "📈 Phân Tích Dữ Liệu Lâm Sàng (EDA)",
    "ℹ️ Về Dự Án & Cloud Deploy"
])

# ======================================================================================
# TAB 1: CHẨN ĐOÁN BỆNH NHÂN
# ======================================================================================
with tab1:
    st.markdown("### Nhập thông tin & Chỉ số Sinh hiệu Bệnh nhân")
    st.write("Chọn bệnh nhân mẫu để thử nghiệm nhanh hoặc tự nhập các chỉ số lâm sàng bên dưới:")

    # Nút chọn mẫu thử nghiệm nhanh
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("🟢 Mẫu 1: Người trẻ khỏe mạnh", use_container_width=True):
            apply_sample("Nữ", 24, "Không", "Không", "Chưa từng hút (never)", 21.2, 5.0, 95)
            st.rerun()
    with col_s2:
        if st.button("🟡 Mẫu 2: Tiền đái tháo đường", use_container_width=True):
            apply_sample("Nam", 48, "Không", "Không", "Chưa từng hút (never)", 27.5, 6.1, 118)
            st.rerun()
    with col_s3:
        if st.button("🔴 Mẫu 3: Bệnh nhân nguy cơ cao", use_container_width=True):
            apply_sample("Nam", 55, "Có", "Không", "Đã từng hút (former)", 32.5, 7.5, 200)
            st.rerun()

    st.markdown("---")

    # Form nhập dữ liệu
    with st.form("diagnosis_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Thông tin Bệnh nhân & Tiền sử")

            gender_idx = 0 if st.session_state.sample_gender == "Nam" else 1
            gender_input = st.selectbox(
                "Giới tính sinh học:",
                ["Nam", "Nữ"],
                index=gender_idx,
                help="Nam (Male) hoặc Nữ (Female)"
            )

            age_input = st.slider(
                "Độ tuổi (năm):",
                min_value=1,
                max_value=100,
                value=int(st.session_state.sample_age),
                help="Tuổi bệnh nhân (0 - 100 tuổi)"
            )

            hyp_idx = 1 if st.session_state.sample_hyp == "Có" else 0
            hypertension_input = st.selectbox(
                "Tiền sử Tăng huyết áp:",
                ["Không", "Có"],
                index=hyp_idx,
                help="Bệnh nhân đã từng được chẩn đoán tăng huyết áp hay chưa"
            )

            heart_idx = 1 if st.session_state.sample_heart == "Có" else 0
            heart_disease_input = st.selectbox(
                "Tiền sử Bệnh lý Tim mạch:",
                ["Không", "Có"],
                index=heart_idx,
                help="Bệnh nhân có tiền sử bệnh tim mạch (nhồi máu, mạch vành...)"
            )

            smoking_options = [
                "Chưa từng hút (never)",
                "Đã từng hút (former)",
                "Đang hút thuốc (current)",
                "Từng có hút (ever)",
                "Hiện không hút (not current)",
                "Không có thông tin (No Info)"
            ]
            current_smoking_idx = 0
            for i, opt in enumerate(smoking_options):
                if opt == st.session_state.sample_smoking:
                    current_smoking_idx = i
                    break

            smoking_input = st.selectbox(
                "Tiền sử & Thói quen hút thuốc lá:",
                smoking_options,
                index=current_smoking_idx,
                help="Thói quen sử dụng thuốc lá của bệnh nhân"
            )

        with col2:
            st.subheader("2. Chỉ số Sinh hóa & Thể trạng")

            bmi_input = st.number_input(
                "Chỉ số khối cơ thể (BMI - kg/m²):",
                min_value=10.0,
                max_value=60.0,
                value=float(st.session_state.sample_bmi),
                step=0.1,
                format="%.1f",
                help="Trọng lượng (kg) / [Chiều cao (m)]^2"
            )
            # Phân loại BMI theo chuẩn người Châu Á (WPRO)
            if bmi_input < 18.5:
                st.caption("🩺 Phân loại thể trạng: :blue[Thiếu cân / Gầy (BMI < 18.5)]")
            elif bmi_input < 23.0:
                st.caption("🩺 Phân loại thể trạng: :green[Bình thường (18.5 - 22.9)]")
            elif bmi_input < 25.0:
                st.caption("🩺 Phân loại thể trạng: :orange[Thừa cân / Tiền béo phì (23.0 - 24.9)]")
            elif bmi_input < 30.0:
                st.caption("🩺 Phân loại thể trạng: :red[Béo phì Độ I (25.0 - 29.9)]")
            else:
                st.caption("🩺 Phân loại thể trạng: :red[Béo phì Độ II/III (>= 30.0)]")

            hba1c_input = st.number_input(
                "Chỉ số đường hóa huyết sắc tố (HbA1c - %):",
                min_value=3.5,
                max_value=9.0,
                value=float(st.session_state.sample_hba1c),
                step=0.1,
                format="%.1f",
                help="Chỉ số phản ánh lượng đường huyết trung bình trong 2 - 3 tháng gần nhất"
            )
            if hba1c_input < 5.7:
                st.caption("🩺 Tiêu chuẩn ADA: :green[Bình thường (< 5.7%)]")
            elif hba1c_input < 6.5:
                st.caption("🩺 Tiêu chuẩn ADA: :orange[Tiền đái tháo đường (5.7% - 6.4%)]")
            else:
                st.caption("🩺 Tiêu chuẩn ADA: :red[Ngưỡng đái tháo đường (>= 6.5%)]")

            glucose_input = st.number_input(
                "Mức đường huyết tức thời / lúc đói (mg/dL):",
                min_value=70,
                max_value=300,
                value=int(st.session_state.sample_glucose),
                step=1,
                help="Nồng độ Glucose trong máu được đo tại thời điểm kiểm tra"
            )
            if glucose_input < 100:
                st.caption("🩺 Ngưỡng lâm sàng: :green[Bình thường khi đói (< 100 mg/dL)]")
            elif glucose_input < 126:
                st.caption("🩺 Ngưỡng lâm sàng: :orange[Rối loạn đường huyết đói / Tiền đái tháo đường (100 - 125 mg/dL)]")
            else:
                st.caption("🩺 Ngưỡng lâm sàng: :red[Nghi ngờ đái tháo đường (>= 126 mg/dL)]")

        submitted = st.form_submit_button("🔍 THỰC HIỆN PHÂN TÍCH & CHẨN ĐOÁN NGUY CƠ", use_container_width=True)

    # XỬ LÝ DỰ ĐOÁN
    if submitted:
        if best_model is None or scaler_obj is None:
            st.error("Không tìm thấy tệp mô hình hoặc scaler! Vui lòng chạy file `diabetes_ml_pipeline.py` (hoặc `main.py`) để tạo tệp `models/best_model.pkl`.")
        else:
            # 1. Chuyển đổi giới tính: Nam -> 1, Nữ -> 0
            g_val = 1 if gender_input == "Nam" else 0

            # 2. Chuyển đổi bệnh lý nền
            hyp_val = 1 if hypertension_input == "Có" else 0
            heart_val = 1 if heart_disease_input == "Có" else 0

            # 3. Chuyển đổi smoking history
            smoking_code_map = {
                "Không có thông tin (No Info)": "No Info",
                "Đang hút thuốc (current)": "current",
                "Từng có hút (ever)": "ever",
                "Đã từng hút (former)": "former",
                "Chưa từng hút (never)": "never",
                "Hiện không hút (not current)": "not current"
            }
            raw_smoking_code = smoking_code_map.get(smoking_input, "never")
            smoking_cats = ['No Info', 'current', 'ever', 'former', 'never', 'not current']
            smk_features = [1 if raw_smoking_code == cat else 0 for cat in smoking_cats]

            # 4. Chuẩn hóa 4 biến liên tục (age, bmi, HbA1c, glucose) bằng Mean & Std đã lưu
            mean_vec = scaler_obj['mean']
            std_vec = scaler_obj['std']
            raw_num = np.array([float(age_input), float(bmi_input), float(hba1c_input), float(glucose_input)], dtype=np.float64)
            scaled_num = (raw_num - mean_vec) / std_vec

            # 5. Lập vector đặc trưng đầu vào (13 chiều)
            sample_vector = np.array([[
                g_val,
                scaled_num[0],  # age
                hyp_val,
                heart_val,
                scaled_num[1],  # bmi
                scaled_num[2],  # HbA1c_level
                scaled_num[3]   # blood_glucose_level
            ] + smk_features])

            # 6. Dự đoán
            pred = best_model.predict(sample_vector)[0]
            prob = best_model.predict_proba(sample_vector)[0, 1]
            prob_percent = prob * 100

            st.markdown("---")
            st.subheader("📋 KẾT QUẢ ĐÁNH GIÁ NGUY CƠ LÂM SÀNG")

            r_col1, r_col2 = st.columns([1.2, 1])

            with r_col1:
                if pred == 1 or prob_percent >= 50.0:
                    st.error(f"### ⚠️ KẾT LUẬN: NGUY CƠ CAO (MẮC BỆNH TIỂU ĐƯỜNG)")
                    st.markdown(f"""
                    Mô hình học máy xác định bệnh nhân có nguy cơ **tiểu đường rất cao** với độ tin cậy ước tính: **{prob_percent:.2f}%**.
                    """)
                else:
                    st.success(f"### ✅ KẾT LUẬN: NGUY CƠ THẤP (KHỎE MẠNH / BÌNH THƯỜNG)")
                    st.markdown(f"""
                    Mô hình học máy xác định bệnh nhân hiện ở mức **an toàn** với tỷ lệ nguy cơ tiểu đường: **{prob_percent:.2f}%**.
                    """)

                # Thanh đo tỷ lệ rủi ro
                st.write("**Thước đo xác suất nguy cơ (Probability Gauge):**")
                st.progress(min(max(prob, 0.0), 1.0))

                if prob_percent < 25.0:
                    st.markdown("🟢 **Phân tầng rủi ro:** Nguy cơ thấp (< 25%) - An toàn.")
                elif prob_percent < 50.0:
                    st.markdown("🟡 **Phân tầng rủi ro:** Nguy cơ trung bình (25% - 50%) - Cần theo dõi lối sống.")
                elif prob_percent < 75.0:
                    st.markdown("🟠 **Phân tầng rủi ro:** Nguy cơ cao (50% - 75%) - Cần can thiệp y tế.")
                else:
                    st.markdown("🔴 **Phân tầng rủi ro:** Nguy cơ rất cao (>= 75%) - Báo động lâm sàng.")

            with r_col2:
                st.markdown("#### 💡 Khuyến nghị Y tế Lâm sàng")
                if pred == 1 or prob_percent >= 50.0:
                    st.markdown("""
                    - **Khám chuyên khoa:** Cần đến cơ sở y tế chuyên khoa Nội tiết để thực hiện xét nghiệm định lượng đường huyết lúc đói hoặc nghiệm pháp dung nạp Glucose đường uống (OGTT).
                    - **Kiểm soát dinh dưỡng:** Hạn chế carbohydrate tinh chế, đường tự do, tăng cường rau xanh, chất xơ và kiểm soát khẩu phần calo.
                    - **Vận động:** Duy trì tối thiểu 150 phút thể dục nhịp điệu mức độ trung bình mỗi tuần (đi bộ nhanh, bơi lội).
                    - **Theo dõi định kỳ:** Đo lại chỉ số HbA1c sau mỗi 3 tháng.
                    """)
                else:
                    st.markdown("""
                    - **Duy trì lối sống lành mạnh:** Tiếp tục chế độ ăn cân bằng dinh dưỡng, hạn chế rượu bia và thuốc lá.
                    - **Kiểm soát cân nặng:** Giữ chỉ số BMI trong ngưỡng lý tưởng từ 18.5 - 22.9 kg/m².
                    - **Khám sức khỏe tổng quát:** Tầm soát xét nghiệm máu và đường huyết định kỳ mỗi 6 - 12 tháng/lần.
                    """)

            # Phân tích các yếu tố rủi ro chính của bệnh nhân này
            st.markdown("#### 🔍 Phân tích các yếu tố đóng góp rủi ro cá nhân:")
            risk_factors = []
            if float(hba1c_input) >= 6.5:
                risk_factors.append(f"❌ **Chỉ số HbA1c ({hba1c_input}%):** Đạt ngưỡng chẩn đoán đái tháo đường lâm sàng (>= 6.5%).")
            elif float(hba1c_input) >= 5.7:
                risk_factors.append(f"⚠️ **Chỉ số HbA1c ({hba1c_input}%):** Nằm trong vùng tiền đái tháo đường (5.7% - 6.4%).")

            if float(glucose_input) >= 126:
                risk_factors.append(f"❌ **Đường huyết ({glucose_input} mg/dL):** Vượt ngưỡng bình thường lúc đói (>= 126 mg/dL).")
            elif float(glucose_input) >= 100:
                risk_factors.append(f"⚠️ **Đường huyết ({glucose_input} mg/dL):** Rối loạn dung nạp nhẹ (100 - 125 mg/dL).")

            if float(bmi_input) >= 30.0:
                risk_factors.append(f"❌ **Chỉ số BMI ({bmi_input}):** Thể trạng béo phì độ II/III - tăng mạnh kháng insulin.")
            elif float(bmi_input) >= 25.0:
                risk_factors.append(f"⚠️ **Chỉ số BMI ({bmi_input}):** Béo phì độ I.")

            if hyp_val == 1:
                risk_factors.append("⚠️ **Tiền sử Tăng huyết áp:** Bệnh lý nền đồng mắc phổ biến ở bệnh nhân chuyển hóa.")
            if heart_val == 1:
                risk_factors.append("⚠️ **Tiền sử Tim mạch:** Yếu tố nguy cơ cao dẫn đến biến chứng mạch máu.")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(rf)
            else:
                st.markdown("✅ Các chỉ số sinh hiệu và tiền sử bệnh lý của bệnh nhân đều ở ngưỡng an toàn.")


# ======================================================================================
# TAB 2: SO SÁNH 7 MÔ HÌNH HỌC MÁY
# ======================================================================================
with tab2:
    st.subheader("📊 So Sánh & Đánh Giá Hiệu Năng 7 Thuật Toán Học Máy (Chuẩn Giáo trình ICTU 2026)")
    st.write("Đánh giá độc lập trên tập kiểm thử gồm **19,227 bệnh nhân** (20% sau khi phân tầng Stratified Split, không rò rỉ dữ liệu).")

    # Bảng số liệu tổng hợp 7 mô hình (HTML/CSS giao diện sáng, hiển thị sắc nét)
    table_html = """
    <div style="overflow-x: auto; margin-bottom: 20px;">
        <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 0.93rem; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
            <thead>
                <tr style="background-color: #0284c7; color: #ffffff;">
                    <th style="padding: 11px; border: 1px solid #0369a1; text-align: left; font-weight: 700;">Mô Hình Thuật Toán</th>
                    <th style="padding: 11px; border: 1px solid #0369a1; font-weight: 700;">Giáo Trình</th>
                    <th style="padding: 11px; border: 1px solid #0369a1; font-weight: 700;">Accuracy (%)</th>
                    <th style="padding: 11px; border: 1px solid #0369a1; font-weight: 700;">Recall (%)</th>
                    <th style="padding: 11px; border: 1px solid #0369a1; font-weight: 700;">Precision (%)</th>
                    <th style="padding: 11px; border: 1px solid #0369a1; font-weight: 700;">F1-Score (%)</th>
                    <th style="padding: 11px; border: 1px solid #0369a1; font-weight: 700;">ROC-AUC (%)</th>
                    <th style="padding: 11px; border: 1px solid #0369a1; font-weight: 700;">Thời Gian</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background-color: #ffffff; color: #0f172a; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 9px; border: 1px solid #e2e8f0; text-align: left; font-weight: 600;">KNN (k=5)</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0; color: #64748b;">Chương 5</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">90.67</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">79.38</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">48.28</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">60.04</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">91.23</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">0.38 s</td>
                </tr>
                <tr style="background-color: #f8fafc; color: #0f172a; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 9px; border: 1px solid #e2e8f0; text-align: left; font-weight: 600;">Naïve Bayes (GaussianNB)</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0; color: #64748b;">Chương 6</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">85.26</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">79.49</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">35.17</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">48.76</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">90.58</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">0.05 s</td>
                </tr>
                <tr style="background-color: #ffffff; color: #0f172a; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 9px; border: 1px solid #e2e8f0; text-align: left; font-weight: 600;">Decision Tree (CART)</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0; color: #64748b;">Chương 8</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">85.72</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0; color: #15803d; font-weight: 800;">92.40</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">37.48</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">53.32</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">96.37</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">0.17 s</td>
                </tr>
                <tr style="background-color: #f8fafc; color: #0f172a; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 9px; border: 1px solid #e2e8f0; text-align: left; font-weight: 600;">Logistic Regression</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0; color: #64748b;">Chương 9</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">88.46</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">87.45</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">42.53</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">57.23</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">95.95</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">0.14 s</td>
                </tr>
                <tr style="background-color: #ffffff; color: #0f172a; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 9px; border: 1px solid #e2e8f0; text-align: left; font-weight: 600;">Linear SVM (Calibrated)</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0; color: #64748b;">Chương 10</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">88.57</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">87.86</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">42.81</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">57.57</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">95.95</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">0.38 s</td>
                </tr>
                <tr style="background-color: #f8fafc; color: #0f172a; border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 9px; border: 1px solid #e2e8f0; text-align: left; font-weight: 600;">Random Forest (100 cây)</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0; color: #64748b;">Chương 14</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">91.22</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">86.68</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">50.15</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">63.54</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">97.24</td>
                    <td style="padding: 9px; border: 1px solid #e2e8f0;">2.34 s</td>
                </tr>
                <tr style="background-color: #ecfdf5; color: #15803d; font-weight: 800; border-left: 4px solid #10b981;">
                    <td style="padding: 9px; border: 1px solid #bbf7d0; text-align: left; color: #047857;">🏆 Gradient Boosting (Tối ưu)</td>
                    <td style="padding: 9px; border: 1px solid #bbf7d0; color: #047857;">Ensemble</td>
                    <td style="padding: 9px; border: 1px solid #bbf7d0; color: #047857;">96.27</td>
                    <td style="padding: 9px; border: 1px solid #bbf7d0; color: #047857;">75.37</td>
                    <td style="padding: 9px; border: 1px solid #bbf7d0; color: #047857;">81.05</td>
                    <td style="padding: 9px; border: 1px solid #bbf7d0; color: #047857;">78.11</td>
                    <td style="padding: 9px; border: 1px solid #bbf7d0; color: #047857;">97.54</td>
                    <td style="padding: 9px; border: 1px solid #bbf7d0; color: #047857;">18.66 s</td>
                </tr>
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("""
    > [!TIP] **Nhận xét chuyên sâu (7 mô hình theo Giáo trình Học máy ICTU 2026):**
    > - **Gradient Boosting Classifier** đạt hiệu năng tổng thể vượt trội nhất với **Accuracy 96.27%**, **ROC-AUC 97.54%**, **Precision 81.05%** và **F1-Score 78.11%**, cân bằng xuất sắc giữa phát hiện bệnh nhân thật và hạn chế báo động giả.
    > - **Decision Tree (CART)** đạt độ nhạy **Recall cao nhất (92.40%)**, phát hiện 1,568/1,697 ca bệnh thật (chỉ bỏ sót 129 ca), rất phù hợp cho sàng lọc sơ bộ diện rộng.
    > - **Random Forest (100 cây)** xếp thứ hai với **Accuracy 91.22%** và **ROC-AUC 97.24%**, vượt trội nhóm mô hình tuyến tính (**Linear SVM**, **Logistic Regression** ~95.95% AUC), **KNN** (90.67% Accuracy) và **Naïve Bayes** (90.58% AUC — huấn luyện nhanh nhất 0.05s).
    """)

    st.markdown("---")
    st.subheader("🖼️ Biểu Đồ So Sánh Trực Quan Hóa")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        f7_path = os.path.join("figures", "07_model_comparison_metrics.png")
        if os.path.exists(f7_path):
            st.image(f7_path, caption="Hình 7: So sánh 5 chỉ số Accuracy, Recall, Precision, F1-Score, ROC-AUC trên 7 mô hình", use_container_width=True)
        else:
            st.warning("Chưa tìm thấy biểu đồ 07_model_comparison_metrics.png")

    with col_fig2:
        f9_path = os.path.join("figures", "09_roc_curves.png")
        if os.path.exists(f9_path):
            st.image(f9_path, caption="Hình 9: Đường cong ROC và diện tích dưới đường cong (AUC) của 7 mô hình", use_container_width=True)
        else:
            st.warning("Chưa tìm thấy biểu đồ 09_roc_curves.png")

    col_fig3, col_fig4 = st.columns(2)
    with col_fig3:
        f8_path = os.path.join("figures", "08_confusion_matrices.png")
        if os.path.exists(f8_path):
            st.image(f8_path, caption="Hình 8: Ma trận nhầm lẫn (Confusion Matrix) của 7 mô hình trên 19,227 mẫu kiểm thử", use_container_width=True)
        else:
            st.warning("Chưa tìm thấy biểu đồ 08_confusion_matrices.png")

    with col_fig4:
        f10_path = os.path.join("figures", "10_feature_importance.png")
        if os.path.exists(f10_path):
            st.image(f10_path, caption="Hình 10: Mức độ quan trọng của các đặc trưng (Feature Importance)", use_container_width=True)
        else:
            st.warning("Chưa tìm thấy biểu đồ 10_feature_importance.png")


# ======================================================================================
# TAB 3: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)
# ======================================================================================
with tab3:
    st.subheader("📈 Phân Tích Thám Hiểm Dữ Liệu Lâm Sàng (EDA)")
    st.write("Tập dữ liệu gồm **100,000 bản ghi** ban đầu, sau khi làm sạch còn **96,128 bản ghi** hợp lệ.")

    c1, c2 = st.columns(2)
    with c1:
        f1_path = os.path.join("figures", "01_class_distribution.png")
        if os.path.exists(f1_path):
            st.image(f1_path, caption="Hình 1: Phân bố nhãn mục tiêu và hiện tượng mất cân bằng lớp (8.5% mắc bệnh)", use_container_width=True)

    with c2:
        f2_path = os.path.join("figures", "02_vital_signs_distribution.png")
        if os.path.exists(f2_path):
            st.image(f2_path, caption="Hình 2: Phân bố tần suất các chỉ số sinh hiệu liên tục (Tuổi, BMI, HbA1c, Glucose)", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        f3_path = os.path.join("figures", "03_vital_signs_by_diabetes.png")
        if os.path.exists(f3_path):
            st.image(f3_path, caption="Hình 3: Biểu đồ hộp so sánh chỉ số giữa nhóm bệnh nhân tiểu đường và bình thường", use_container_width=True)

    with c4:
        f4_path = os.path.join("figures", "04_categorical_risk.png")
        if os.path.exists(f4_path):
            st.image(f4_path, caption="Hình 4: Tỷ lệ mắc bệnh theo giới tính, tiền sử huyết áp, bệnh tim và thói quen hút thuốc", use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        f5_path = os.path.join("figures", "05_correlation_heatmap.png")
        if os.path.exists(f5_path):
            st.image(f5_path, caption="Hình 5: Ma trận hệ số tương quan tuyến tính Pearson giữa các biến", use_container_width=True)

    with c6:
        f6_path = os.path.join("figures", "06_preprocessing_effects.png")
        if os.path.exists(f6_path):
            st.image(f6_path, caption="Hình 6: Minh họa hiệu quả cân bằng dữ liệu bằng thuật toán SMOTE (tăng lên 140,232 mẫu)", use_container_width=True)


# ======================================================================================
# TAB 4: VỀ DỰ ÁN & HƯỚNG DẪN CLOUD DEPLOYMENT
# ======================================================================================
with tab4:
    st.subheader("ℹ️ Giới Thiệu Đề Tài & Hướng Dẫn Triển Khai Lên Đám Mây")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        ### 🎯 Mục Tiêu & Phạm Vi Đề Tài
        - **Học phần:** Học máy (Machine Learning) - Bài kiểm tra kỹ năng số 1
        - **Đề tài:** Ứng dụng các thuật toán phân loại có giám sát (Supervised Classification) để chẩn đoán sớm nguy cơ mắc bệnh tiểu đường dựa trên dữ liệu sinh hiệu lâm sàng.
        - **Quy trình chuẩn MLOps:**
          1. Thu thập & Thẩm định dữ liệu (100,000 dòng).
          2. Tiền xử lý: Lọc trùng lặp, mã hóa One-Hot, chuẩn hóa Z-Score.
          3. Xử lý mất cân bằng nhãn bằng kỹ thuật **SMOTE** trên tập huấn luyện.
          4. Huấn luyện & đánh giá 7 mô hình theo *Giáo trình Học máy (ICTU 2026)*: KNN, Naïve Bayes, Decision Tree, Logistic Regression, Linear SVM, Random Forest và Gradient Boosting.
          5. Đóng gói mô hình tối ưu sang tệp nhị phân `.pkl` và xây dựng Web App Streamlit tương tác.
        """)

    with col_info2:
        st.markdown("""
        ### 👥 Danh Sách Thành Viên Thực Hiện (Nhóm 3)
        1. **Nguyễn Lê Hoàng** (Trưởng nhóm - Xây dựng Pipeline & Mô hình Học máy)
        2. **Ngô Xuân Đỉnh** (Tiền xử lý dữ liệu, Trực quan hóa EDA & Đánh giá mô hình)
        3. **Hà Đức Trung** (Thiết kế Web App Streamlit & Đóng gói triển khai)
        4. **Trần Đức Đạt** (Biên soạn báo cáo học thuật Word & Kịch bản thuyết trình)

        **Giảng viên hướng dẫn:** ThS. Hà Thị Thanh  
        **Khoa:** Công nghệ Thông tin  
        **Trường:** Đại học Công nghệ Thông tin & Truyền thông - Đại học Thái Nguyên (ICTU)
        """)

    st.markdown("---")
    st.subheader("☁️ Hướng Dẫn Triển Khai Lên Streamlit Community Cloud (Miễn phí)")
    st.markdown("""
    Để đưa ứng dụng này lên mạng Internet cho giảng viên và hội đồng kiểm tra trực tiếp:

    1. **Bước 1: Đẩy mã nguồn lên GitHub**
       - Tạo một Repository mới trên GitHub (ví dụ: `diabetes-prediction-ictu`).
       - Commit toàn bộ thư mục dự án gồm: `app/app.py`, `models/`, `figures/`, `requirements.txt`, `README.md`.
    
    2. **Bước 2: Kết nối với Streamlit Community Cloud**
       - Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
       - Nhấn **New app**.
       - Chọn Repository của bạn, nhánh `main`.
       - Tại mục **Main file path**, điền: `app/app.py`.
       - Nhấn **Deploy!**

    3. **Bước 3: Nhận liên kết công khai**
       - Streamlit Cloud sẽ tự động đọc file `requirements.txt`, cài đặt môi trường và cấp một đường link công khai (ví dụ: `https://diabetes-prediction-ictu.streamlit.app/`).
    """)

# FOOTER
st.markdown('<div class="footer-text">Khoa Công nghệ Thông tin - Trường Đại học Công nghệ Thông tin & Truyền thông (ICTU) | Học phần Học máy 2026</div>', unsafe_allow_html=True)
