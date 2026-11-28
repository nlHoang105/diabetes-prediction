# ==============================================================================
# Dockerfile — Hệ thống Chẩn đoán sớm Nguy cơ Tiểu đường (Streamlit + ML)
# Phương án 1: Đóng gói ứng dụng Streamlit vào Docker Container triển khai trên
# Microsoft Azure (Azure Container Registry + Azure App Service / Container Apps)
# ==============================================================================

FROM python:3.11-slim

# Thiết lập biến môi trường tối ưu cho Python & Streamlit trong Container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Cài đặt công cụ tối thiểu cho HEALTHCHECK và dọn cache apt
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện phụ thuộc trước để tận dụng Docker Layer Cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng, cấu hình Streamlit, mô hình đã huấn luyện và biểu đồ
COPY .streamlit/ ./.streamlit/
COPY app/ ./app/
COPY models/ ./models/
COPY figures/ ./figures/

# Mở cổng 8501 chuẩn của Streamlit (khai báo WEBSITES_PORT=8501 trên Azure)
EXPOSE 8501

# Kiểm tra sức khỏe Container tự động
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Khởi chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
