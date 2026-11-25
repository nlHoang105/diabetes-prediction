

import os
import sys
import csv
import json
import pickle
import argparse

# Thiết lập UTF-8 cho console Windows để in tiếng Việt không bị lỗi charmap
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt

# Sử dụng engine thuần csv + numpy + matplotlib + scikit-learn để tương thích tối đa trên Windows
USE_PANDAS = False

# Thiết lập giao diện biểu đồ Matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

def get_raw_data_path():
    raw_path = os.path.join('data', 'raw', 'diabetes_prediction_dataset.csv')
    legacy_path = os.path.join('data', 'diabetes_prediction_dataset.csv')
    if os.path.exists(raw_path):
        return raw_path
    return legacy_path

DATA_PATH = get_raw_data_path()
PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = 'models'
FIGURES_DIR = 'figures'
DOCS_DIR = 'docs'


def load_raw_data(filepath=None):
    """Đọc dữ liệu thô từ file CSV."""
    if filepath is None:
        filepath = get_raw_data_path()
    print("\n" + "="*70)
    print("1. THU THẬP & KIỂM TRA DỮ LIỆU BAN ĐẦU")
    print("="*70)
    
    if not os.path.exists(filepath):
        print(f"LỖI: Không tìm thấy tệp dữ liệu tại '{filepath}'!")
        sys.exit(1)
        
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for r in reader:
            records.append(r)
            
    print(f"-> Đã đọc tệp dữ liệu thành công: {filepath}")
    print(f"-> Kích thước tập dữ liệu: {len(records):,} dòng x {len(header)} cột")
    print(f"-> Các cột: {header}")
    
    # Kiểm tra null
    null_counts = {col: sum(1 for r in records if r[col] == '' or r[col] is None) for col in header}
    print("\n--- Số lượng giá trị khuyết (Missing/Null values) ---")
    for k, v in null_counts.items():
        print(f"  {k:22}: {v} null")
        
    return records, header


def generate_all_eda_figures(records, output_dir=FIGURES_DIR):
    """Bước 2: Phân tích & Trực quan hóa dữ liệu EDA."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*70)
    print("2. PHÂN TÍCH & TRỰC QUAN HÓA DỮ LIỆU (EDA)")
    print("="*70)
    print(f"-> Đang kết xuất và lưu biểu đồ độ phân giải cao vào: '{output_dir}/'...")

    # Chuyển đổi dữ liệu số phục vụ vẽ biểu đồ
    y_vals = np.array([int(r['diabetes']) for r in records])
    ages = np.array([float(r['age']) for r in records])
    bmis = np.array([float(r['bmi']) for r in records])
    hba1cs = np.array([float(r['HbA1c_level']) for r in records])
    glucoses = np.array([float(r['blood_glucose_level']) for r in records])

    # 1. Phân phối nhãn mục tiêu
    plt.figure(figsize=(7, 5))
    c0 = np.sum(y_vals == 0)
    c1 = np.sum(y_vals == 1)
    bars = plt.bar(['Không bệnh (0)', 'Tiểu đường (1)'], [c0, c1], color=['#1f77b4', '#d62728'], width=0.55)
    plt.title('Phân phối nhãn mục tiêu (0: Không bệnh, 1: Mắc bệnh tiểu đường)', fontsize=12, pad=12)
    plt.ylabel('Số lượng bản ghi', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2., h/2., f'{h:,.0f}\n({h/len(y_vals)*100:.1f}%)', 
                 ha='center', va='center', color='white', weight='bold', fontsize=11)
    plt.tight_layout()
    f1_path = os.path.join(output_dir, '01_class_distribution.png')
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu: {f1_path}")

    # 2. Phân phối tần suất 4 chỉ số sinh hiệu
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    data_list = [ages, bmis, hba1cs, glucoses]
    titles = ['Độ tuổi (age)', 'Chỉ số khối cơ thể (bmi)', 'Chỉ số HbA1c (%)', 'Mức đường huyết tức thời (mg/dL)']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    for i in range(4):
        axes[i].hist(data_list[i], bins=30, color=colors[i], alpha=0.75, edgecolor='black', linewidth=0.5)
        axes[i].set_title(f'Phân phối {titles[i]}', fontsize=11)
        axes[i].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    f2_path = os.path.join(output_dir, '02_vital_signs_distribution.png')
    plt.savefig(f2_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu: {f2_path}")

    # 3. Biểu đồ hộp (Boxplot) so sánh 2 nhóm
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(4):
        d0 = data_list[i][y_vals == 0]
        d1 = data_list[i][y_vals == 1]
        # Matplotlib 3.9+ đổi labels thành tick_labels
        try:
            bp = axes[i].boxplot([d0, d1], patch_artist=True, tick_labels=['Không bệnh (0)', 'Tiểu đường (1)'],
                                 medianprops=dict(color='black', linewidth=1.5))
        except TypeError:
            bp = axes[i].boxplot([d0, d1], patch_artist=True, labels=['Không bệnh (0)', 'Tiểu đường (1)'],
                                 medianprops=dict(color='black', linewidth=1.5))
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][1].set_facecolor('#d62728')
        axes[i].set_title(f'{titles[i]} theo Trạng thái bệnh', fontsize=11)
        axes[i].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    f3_path = os.path.join(output_dir, '03_vital_signs_by_diabetes.png')
    plt.savefig(f3_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu: {f3_path}")

    # 4. Tác động của Bệnh nền, Giới tính và Thói quen sinh hoạt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    # Cao huyết áp
    hyp0 = np.mean([int(r['diabetes']) for r in records if r['hypertension'] == '0']) * 100
    hyp1 = np.mean([int(r['diabetes']) for r in records if r['hypertension'] == '1']) * 100
    axes[0, 0].bar(['Không bị (0)', 'Có bị (1)'], [hyp0, hyp1], color=['#9ecae1', '#2171b5'], width=0.5)
    axes[0, 0].set_title('Tỷ lệ mắc bệnh theo Tiền sử Cao huyết áp (%)', fontsize=11)
    axes[0, 0].set_ylabel('Tỷ lệ mắc bệnh (%)')
    for idx, v in enumerate([hyp0, hyp1]): axes[0, 0].text(idx, v + 1, f'{v:.1f}%', ha='center', weight='bold')

    # Bệnh tim
    hd0 = np.mean([int(r['diabetes']) for r in records if r['heart_disease'] == '0']) * 100
    hd1 = np.mean([int(r['diabetes']) for r in records if r['heart_disease'] == '1']) * 100
    axes[0, 1].bar(['Không bị (0)', 'Có bị (1)'], [hd0, hd1], color=['#fc9272', '#de2d26'], width=0.5)
    axes[0, 1].set_title('Tỷ lệ mắc bệnh theo Tiền sử Bệnh tim mạch (%)', fontsize=11)
    for idx, v in enumerate([hd0, hd1]): axes[0, 1].text(idx, v + 1, f'{v:.1f}%', ha='center', weight='bold')

    # Giới tính
    gf = np.mean([int(r['diabetes']) for r in records if r['gender'] == 'Female']) * 100
    gm = np.mean([int(r['diabetes']) for r in records if r['gender'] == 'Male']) * 100
    axes[1, 0].bar(['Female', 'Male'], [gf, gm], color=['#bcbddc', '#756bb1'], width=0.5)
    axes[1, 0].set_title('Tỷ lệ mắc bệnh theo Giới tính (%)', fontsize=11)
    for idx, v in enumerate([gf, gm]): axes[1, 0].text(idx, v + 0.5, f'{v:.1f}%', ha='center', weight='bold')

    # Hút thuốc
    smoking_types = sorted(list(set(r['smoking_history'] for r in records)))
    smk_rates = [np.mean([int(r['diabetes']) for r in records if r['smoking_history'] == st]) * 100 for st in smoking_types]
    axes[1, 1].bar(smoking_types, smk_rates, color='#fd8d3c', width=0.6)
    axes[1, 1].set_title('Tỷ lệ mắc bệnh theo Thói quen Hút thuốc (%)', fontsize=11)
    axes[1, 1].tick_params(axis='x', rotation=30)
    for idx, v in enumerate(smk_rates): axes[1, 1].text(idx, v + 0.5, f'{v:.1f}%', ha='center', weight='bold', fontsize=9)
    plt.tight_layout()
    f4_path = os.path.join(output_dir, '04_categorical_risk.png')
    plt.savefig(f4_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu: {f4_path}")

    # 5. Ma trận nhiệt hệ số tương quan Pearson
    corr_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    matrix_data = []
    for r in records:
        matrix_data.append([
            float(r['age']), float(r['hypertension']), float(r['heart_disease']),
            float(r['bmi']), float(r['HbA1c_level']), float(r['blood_glucose_level']),
            float(r['diabetes'])
        ])
    mat_arr = np.array(matrix_data)
    corr_mat = np.corrcoef(mat_arr, rowvar=False)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr_mat, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Hệ số tương quan Pearson')
    plt.xticks(range(7), corr_cols, rotation=35, ha='right')
    plt.yticks(range(7), corr_cols)
    for i in range(7):
        for j in range(7):
            plt.text(j, i, f'{corr_mat[i, j]:.3f}', ha='center', va='center',
                     color='white' if abs(corr_mat[i, j]) > 0.5 else 'black', fontsize=9)
    plt.title('Ma trận tương quan giữa các chỉ số sinh hiệu và Bệnh tiểu đường', fontsize=12, pad=12)
    plt.tight_layout()
    f5_path = os.path.join(output_dir, '05_correlation_heatmap.png')
    plt.savefig(f5_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu: {f5_path}")


def run_preprocessing_pipeline(records, header):
    """Bước 3: Quy trình Tiền xử lý dữ liệu chuẩn khoa học."""
    print("\n" + "="*70)
    print("3. TIỀN XỬ LÝ DỮ LIỆU CHUẨN KHOA HỌC")
    print("="*70)

    # 1. Lọc trùng lặp
    seen = set()
    unique_records = []
    for r in records:
        key = tuple(r[k] for k in header)
        if key not in seen:
            seen.add(key)
            unique_records.append(r)
    dup_count = len(records) - len(unique_records)
    print(f"-> Bước 1: Loại bỏ {dup_count:,} bản ghi trùng lặp 100%. Còn lại: {len(unique_records):,} dòng.")

    # 2. Lọc bỏ dị biệt gender == 'Other'
    clean_records = [r for r in unique_records if r['gender'] != 'Other']
    other_count = len(unique_records) - len(clean_records)
    print(f"-> Bước 2: Loại bỏ {other_count} mẫu gender='Other'. Còn lại {len(clean_records):,} bản ghi sạch.")

    # 3. Mã hóa biến phân loại
    print("-> Bước 3: Mã hóa biến phân loại:")
    print("   [+] gender: Female -> 0, Male -> 1")
    smoking_categories = ['No Info', 'current', 'ever', 'former', 'never', 'not current']
    print(f"   [+] smoking_history: One-Hot Encoding thành {len(smoking_categories)} cột nhị phân")

    feature_cols = [
        'gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'smoking_No Info', 'smoking_current', 'smoking_ever', 'smoking_former', 'smoking_never', 'smoking_not current'
    ]

    X_all = []
    y_all = []
    cleaned_rows = []

    for r in clean_records:
        g = 0 if r['gender'] == 'Female' else 1
        a = float(r['age'])
        hyp = int(r['hypertension'])
        hd = int(r['heart_disease'])
        bmi = float(r['bmi'])
        hba1c = float(r['HbA1c_level'])
        bg = float(r['blood_glucose_level'])
        diab = int(r['diabetes'])

        smk_onehot = [1 if r['smoking_history'] == cat else 0 for cat in smoking_categories]
        row_features = [g, a, hyp, hd, bmi, hba1c, bg] + smk_onehot
        
        X_all.append(row_features)
        y_all.append(diab)
        cleaned_rows.append(row_features + [diab])

    X_all = np.array(X_all, dtype=np.float64)
    y_all = np.array(y_all, dtype=np.int64)

    # 4. Phân chia tập dữ liệu Train/Test phân tầng (stratified 80/20)
    print("\n-> Bước 4: Phân chia tập dữ liệu 80% Train / 20% Test phân tầng (stratify=y)...")
    idx_0 = np.where(y_all == 0)[0]
    idx_1 = np.where(y_all == 1)[0]
    
    np.random.seed(42)
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    split_0 = int(len(idx_0) * 0.8)
    split_1 = int(len(idx_1) * 0.8)

    train_idx = np.concatenate([idx_0[:split_0], idx_1[:split_1]])
    test_idx = np.concatenate([idx_0[split_0:], idx_1[split_1:]])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    X_train_raw = X_all[train_idx].copy()
    y_train = y_all[train_idx].copy()
    X_test_raw = X_all[test_idx].copy()
    y_test = y_all[test_idx].copy()

    print(f"   Tập Huấn luyện (Train): {len(train_idx):,} mẫu | Ca bệnh: {np.sum(y_train==1):,} ({np.mean(y_train)*100:.2f}%)")
    print(f"   Tập Kiểm thử (Test)   : {len(test_idx):,} mẫu | Ca bệnh: {np.sum(y_test==1):,} ({np.mean(y_test)*100:.2f}%)")

    # 5. Chuẩn hóa thang đo đặc trưng (StandardScaler cho 4 cột số)
    print("\n-> Bước 5: Chuẩn hóa thang đo (StandardScaler) cho 4 chỉ số sinh hiệu liên tục...")
    num_indices = [1, 4, 5, 6] # age, bmi, HbA1c_level, blood_glucose_level
    mean_vec = np.mean(X_train_raw[:, num_indices], axis=0)
    std_vec = np.std(X_train_raw[:, num_indices], axis=0)

    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()

    X_train_scaled[:, num_indices] = (X_train_raw[:, num_indices] - mean_vec) / std_vec
    X_test_scaled[:, num_indices] = (X_test_raw[:, num_indices] - mean_vec) / std_vec
    print("   [+] Đã fit bộ chuẩn hóa trên Train và transform cho cả Train & Test.")

    # 6. Cân bằng dữ liệu bằng SMOTE trên tập Train
    print("\n-> Bước 6: Cân bằng lớp thiểu số bằng kỹ thuật SMOTE (CHỈ trên tập Train)...")
    idx_min = np.where(y_train == 1)[0]
    idx_maj = np.where(y_train == 0)[0]
    n_maj = len(idx_maj)
    n_min = len(idx_min)
    n_to_create = n_maj - n_min

    X_min = X_train_scaled[idx_min]

    # Vector hóa thuật toán SMOTE với NearestNeighbors siêu tốc
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=6).fit(X_min)
    _, nn_indices = nn.kneighbors(X_min)

    sample_idx = np.random.randint(0, n_min, size=n_to_create)
    neighbor_col = np.random.randint(1, 6, size=n_to_create)
    neighbor_idx = nn_indices[sample_idx, neighbor_col]

    samples = X_min[sample_idx]
    neighbors = X_min[neighbor_idx]
    gaps = np.random.rand(n_to_create, 1)
    synthetic_samples = samples + gaps * (neighbors - samples)

    X_train_res = np.vstack([X_train_scaled, synthetic_samples])
    y_train_res = np.concatenate([y_train, np.ones(n_to_create, dtype=np.int64)])

    print(f"   Tập Train trước SMOTE: {{0: {n_maj:,}, 1: {n_min:,}}}")
    print(f"   Tập Train sau SMOTE  : {{0: {n_maj:,}, 1: {len(y_train_res)-n_maj:,}}} (Tỷ lệ cân bằng 50% : 50%)")
    print(f"   Tập Test giữ nguyên vẹn 100% dữ liệu thực tế ({len(y_test):,} mẫu) để đánh giá khách quan.")

    # Biểu đồ tác động SMOTE
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(['Không bệnh (0)', 'Tiểu đường (1)'], [n_maj, n_min], color=['#1f77b4', '#d62728'], width=0.5)
    axes[0].set_title('Tập Train GỐC (Mất cân bằng 11:1)')
    axes[0].set_ylabel('Số lượng bản ghi')
    axes[0].text(0, n_maj/2, f'{n_maj:,}', ha='center', color='white', weight='bold')
    axes[0].text(1, n_min/2, f'{n_min:,}', ha='center', color='white', weight='bold')

    axes[1].bar(['Không bệnh (0)', 'Tiểu đường (1)'], [n_maj, n_maj], color=['#1f77b4', '#d62728'], width=0.5)
    axes[1].set_title('Tập Train SAU SMOTE (Cân bằng chuẩn 50:50)')
    axes[1].set_ylabel('Số lượng bản ghi')
    axes[1].text(0, n_maj/2, f'{n_maj:,}', ha='center', color='white', weight='bold')
    axes[1].text(1, n_maj/2, f'{n_maj:,}', ha='center', color='white', weight='bold')
    plt.tight_layout()
    f6_path = os.path.join(FIGURES_DIR, '06_preprocessing_effects.png')
    plt.savefig(f6_path, dpi=300)
    plt.close()
    print(f"   [+] Đã cập nhật biểu đồ minh họa: {f6_path}")

    # Tạo đối tượng scaler
    scaler_obj = {
        'mean': mean_vec,
        'std': std_vec,
        'features': ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'],
        'indices': num_indices
    }

    return {
        'feature_cols': feature_cols,
        'cleaned_rows': cleaned_rows,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_res': X_train_res,
        'y_train_res': y_train_res,
        'scaler_obj': scaler_obj,
        'raw_count': len(records),
        'dup_count': dup_count,
        'other_count': other_count
    }


def save_processed_csvs(data, output_dir=PROCESSED_DIR):
    """Bước 4: Xuất các file dữ liệu sạch vào data/processed/."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*70)
    print("4. XUẤT TẬP DỮ LIỆU ĐÃ XỬ LÝ VÀ MÔ HÌNH CHUẨN HÓA")
    print("="*70)

    # 1. diabetes_cleaned.csv
    f_clean = os.path.join(output_dir, 'diabetes_cleaned.csv')
    try:
        with open(f_clean, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data['feature_cols'] + ['diabetes'])
            writer.writerows(data['cleaned_rows'])
        print(f"   [+] Đã lưu: {f_clean:42} ({os.path.getsize(f_clean):,} bytes)")
    except PermissionError:
        print(f"   [!] File {f_clean} đang được mở trong trình soạn thảo, đã giữ nguyên phiên bản hiện tại.")

    # 2. X_train_raw, X_test_raw
    def write_matrix(filename, mat, header_cols):
        p = os.path.join(output_dir, filename)
        try:
            with open(p, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                w.writerow(header_cols)
                w.writerows(mat)
            print(f"   [+] Đã lưu: {p:42} ({os.path.getsize(p):,} bytes)")
        except PermissionError:
            print(f"   [!] File {p} đang mở trong chương trình khác, giữ nguyên.")

    def write_vector(filename, vec, header_col):
        p = os.path.join(output_dir, filename)
        try:
            with open(p, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                w.writerow([header_col])
                for v in vec: w.writerow([v])
            print(f"   [+] Đã lưu: {p:42} ({os.path.getsize(p):,} bytes)")
        except PermissionError:
            print(f"   [!] File {p} đang mở trong chương trình khác, giữ nguyên.")

    write_matrix('X_train_raw.csv', data['X_train_raw'], data['feature_cols'])
    write_matrix('X_test_raw.csv', data['X_test_raw'], data['feature_cols'])
    write_matrix('X_train_scaled.csv', data['X_train_scaled'], data['feature_cols'])
    write_matrix('X_test_scaled.csv', data['X_test_scaled'], data['feature_cols'])
    write_matrix('X_train_resampled.csv', data['X_train_res'], data['feature_cols'])

    write_vector('y_train.csv', data['y_train'], 'diabetes')
    write_vector('y_test.csv', data['y_test'], 'diabetes')
    write_vector('y_train_resampled.csv', data['y_train_res'], 'diabetes')

    # Lưu scaler.pkl
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(data['scaler_obj'], f)
        print(f"   [+] Đã lưu: {scaler_path:42} (Đối tượng Scaler)")
        
        # Đồng bộ lưu vào models/
        os.makedirs(MODELS_DIR, exist_ok=True)
        models_scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        with open(models_scaler_path, 'wb') as f:
            pickle.dump(data['scaler_obj'], f)
        print(f"   [+] Đã lưu: {models_scaler_path:42} (Đối tượng Scaler)")
    except PermissionError:
        print(f"   [!] File scaler.pkl đang mở, giữ nguyên.")

    # Lưu preprocessing_summary.json
    summary_path = os.path.join(output_dir, 'preprocessing_summary.json')
    summary_data = {
        "dataset_name": "Diabetes Prediction Dataset",
        "raw_samples": data['raw_count'],
        "duplicate_samples_removed": data['dup_count'],
        "invalid_gender_removed": data['other_count'],
        "clean_samples": len(data['cleaned_rows']),
        "train_samples": len(data['X_train_scaled']),
        "test_samples": len(data['X_test_scaled']),
        "smote_train_samples": len(data['X_train_res']),
        "features": data['feature_cols'],
        "scaled_numerical_features": ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'],
        "target_col": "diabetes",
        "class_counts_test": {
            "non_diabetic": int(np.sum(data['y_test'] == 0)),
            "diabetic": int(np.sum(data['y_test'] == 1))
        }
    }
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"   [+] Đã lưu: {summary_path:42} (Metadata tóm tắt)")
    except PermissionError:
        print(f"   [!] File {summary_path} đang mở, giữ nguyên.")


def train_and_evaluate_models(data, output_dir=PROCESSED_DIR, figures_dir=FIGURES_DIR):
    """
    Bước 5: Huấn luyện, Đánh giá và So sánh 7 mô hình Machine Learning
    (Theo chuẩn Giáo trình Học máy - ICTU 2026):
    1. KNN (Chương 5 - K lân cận gần nhất)
    2. Naïve Bayes (Chương 6 - Phân loại Bayes ngây thơ)
    3. Decision Tree (Chương 8 - Cây quyết định)
    4. Logistic Regression (Chương 9 - Hồi quy Logistic)
    5. Linear SVM (Chương 10 - Máy vector hỗ trợ)
    6. Random Forest (Chương 14 - Rừng ngẫu nhiên)
    7. Gradient Boosting (Học kết hợp tăng cường Gradient - Mô hình tối ưu)
    """
    import time
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
        confusion_matrix, roc_curve
    )

    print("\n" + "="*70)
    print("5. HUẤN LUYỆN, ĐÁNH GIÁ & SO SÁNH 7 MÔ HÌNH HỌC MÁY")
    print("="*70)

    X_train = data['X_train_res']
    y_train = data['y_train_res']
    X_test = data['X_test_scaled']
    y_test = data['y_test']
    feature_names = data['feature_cols']

    models = {
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Naïve Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000), cv=3),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }

    trained_models = {}
    eval_results = {}

    print(f"-> Tập huấn luyện (Sau SMOTE): {X_train.shape[0]:,} mẫu | 13 đặc trưng")
    print(f"-> Tập kiểm thử (Độc lập 100%):  {X_test.shape[0]:,} mẫu\n")

    print(f"{'Mô hình':<22} | {'Accuracy':<9} | {'Recall':<9} | {'Precision':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'Train Time'}")
    print("-" * 92)

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)

        eval_results[name] = {
            'Accuracy': acc,
            'Recall': rec,
            'Precision': prec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Confusion_Matrix': cm.tolist(),
            'y_prob': y_prob,
            'Train_Time': t_train
        }

        print(f"{name:<22} | {acc*100:>7.2f}% | {rec*100:>7.2f}% | {prec*100:>7.2f}% | {f1*100:>7.2f}% | {auc*100:>7.2f}% | {t_train:>6.2f}s")

    print("-" * 92)

    # 1. Vẽ biểu đồ so sánh các chỉ số (07_model_comparison_metrics.png)
    os.makedirs(figures_dir, exist_ok=True)
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'ROC-AUC']
    model_names = list(models.keys())
    n_models = len(model_names)
    x = np.arange(len(metrics))
    width = 0.11

    fig, ax = plt.subplots(figsize=(15, 6.8))
    colors = ['#6366f1', '#8b5cf6', '#f59e0b', '#0284c7', '#06b6d4', '#10b981', '#dc2626']

    for i, name in enumerate(model_names):
        vals = [eval_results[name][m] * 100 for m in metrics]
        offset = (i - (n_models - 1) / 2.0) * width
        rects = ax.bar(x + offset, vals, width * 0.92, label=name, color=colors[i])
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f'{h:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7.5, rotation=90
            )

    ax.set_ylabel('Điểm số (%)', fontsize=11)
    ax.set_title(f'So sánh hiệu năng 7 mô hình Học máy trên tập Test độc lập ({len(y_test):,} mẫu)', fontsize=13, pad=36)
    ax.set_xticks(x)
    ax.set_xticklabels(['Độ chính xác\n(Accuracy)', 'Độ nhạy\n(Recall)', 'Độ chuẩn xác\n(Precision)', 'F1-Score', 'ROC-AUC'], fontsize=10.5)
    ax.set_ylim(0, 118)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=4, fontsize=9.5, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    f7_path = os.path.join(figures_dir, '07_model_comparison_metrics.png')
    plt.savefig(f7_path, dpi=300)
    plt.close()
    print(f"\n   [+] Đã lưu biểu đồ so sánh: {f7_path}")

    # 2. Vẽ Ma trận nhầm lẫn 2x4 (7 mô hình, ẩn ô trống thứ 8) (08_confusion_matrices.png)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    axes = axes.flatten()
    for i, name in enumerate(model_names):
        cm = np.array(eval_results[name]['Confusion_Matrix'])
        axes[i].imshow(cm, cmap='Blues', interpolation='nearest')
        axes[i].set_title(f'{name}', fontsize=11.5, weight='bold')
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['Dự đoán Lành (0)', 'Dự đoán Bệnh (1)'], fontsize=8.5)
        axes[i].set_yticklabels(['Thực tế Lành (0)', 'Thực tế Bệnh (1)'], fontsize=8.5)

        thresh = cm.max() / 2.
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, f'{cm[r, c]:,}', ha='center', va='center',
                             color='white' if cm[r, c] > thresh else 'black', fontsize=11, weight='bold')
    for j in range(len(model_names), len(axes)):
        axes[j].axis('off')
    fig.suptitle(f'Ma trận nhầm lẫn (Confusion Matrix) của 7 mô hình trên tập Test ({len(y_test):,} mẫu)', fontsize=14, weight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    f8_path = os.path.join(figures_dir, '08_confusion_matrices.png')
    plt.savefig(f8_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu Ma trận nhầm lẫn: {f8_path}")

    # 3. Vẽ Đường cong ROC (09_roc_curves.png)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    linestyles = ['--', '-.', ':', '-', '--', '-', '-']
    for i, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(y_test, eval_results[name]['y_prob'])
        auc_val = eval_results[name]['ROC-AUC']
        lw = 2.6 if name == 'Gradient Boosting' else 1.8
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.4f})', color=colors[i], linestyle=linestyles[i], linewidth=lw)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.45, label='Ngẫu nhiên (AUC = 0.5000)')
    ax.set_xlabel('Tỷ lệ Dương tính giả (False Positive Rate)', fontsize=11)
    ax.set_ylabel('Tỷ lệ Dương tính thật - Recall (True Positive Rate)', fontsize=11)
    ax.set_title('Đường cong ROC (Receiver Operating Characteristic) của 7 mô hình Học máy', fontsize=12.5, pad=12)
    ax.legend(loc='lower right', fontsize=9.5, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    f9_path = os.path.join(figures_dir, '09_roc_curves.png')
    plt.savefig(f9_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu Đường cong ROC:   {f9_path}")

    # 4. Trực quan hóa Feature Importance (10_feature_importance.png)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (m_name, m_key) in enumerate([('Random Forest', 'Random Forest'), ('Gradient Boosting', 'Gradient Boosting')]):
        model_obj = trained_models[m_key]
        importances = model_obj.feature_importances_
        sorted_idx = np.argsort(importances)
        y_ticks = np.arange(len(feature_names))
        axes[idx].barh(y_ticks, importances[sorted_idx], color='#10b981' if idx==0 else '#dc2626', alpha=0.85)
        axes[idx].set_yticks(y_ticks)
        axes[idx].set_yticklabels(np.array(feature_names)[sorted_idx], fontsize=9)
        axes[idx].set_title(f'Độ quan trọng đặc trưng: {m_name}', fontsize=12)
        axes[idx].set_xlabel('Mức độ đóng góp tương đối (Importance)', fontsize=10)
        axes[idx].grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    f10_path = os.path.join(figures_dir, '10_feature_importance.png')
    plt.savefig(f10_path, dpi=300)
    plt.close()
    print(f"   [+] Đã lưu Độ quan trọng:    {f10_path}")

    # 5. Lưu mô hình tốt nhất (best_model.pkl)
    best_name = 'Gradient Boosting'
    best_model = trained_models[best_name]
    best_model_path = os.path.join(output_dir, 'best_model.pkl')
    try:
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"\n-> Đã lưu mô hình tối ưu ({best_name}) vào: {best_model_path}")
        
        # Đồng bộ lưu vào models/
        os.makedirs(MODELS_DIR, exist_ok=True)
        models_best_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        with open(models_best_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"-> Đã lưu mô hình tối ưu ({best_name}) vào: {models_best_path}")
    except PermissionError:
        print(f"   [!] File best_model.pkl đang mở, giữ nguyên.")

    # 6. Lưu file tóm tắt đánh giá JSON
    eval_json_path = os.path.join(output_dir, 'model_evaluation_summary.json')
    eval_summary = {
        name: {
            'Accuracy': round(res['Accuracy'] * 100, 2),
            'Recall': round(res['Recall'] * 100, 2),
            'Precision': round(res['Precision'] * 100, 2),
            'F1_Score': round(res['F1-Score'] * 100, 2),
            'ROC_AUC': round(res['ROC-AUC'] * 100, 2),
            'Train_Time_Seconds': round(res['Train_Time'], 2),
            'Confusion_Matrix': res['Confusion_Matrix']
        }
        for name, res in eval_results.items()
    }
    try:
        with open(eval_json_path, 'w', encoding='utf-8') as f:
            json.dump(eval_summary, f, indent=4, ensure_ascii=False)
        print(f"-> Đã lưu kết quả đánh giá chi tiết: {eval_json_path}")
        
        models_eval_path = os.path.join(MODELS_DIR, 'model_evaluation_summary.json')
        with open(models_eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_summary, f, indent=4, ensure_ascii=False)
        print(f"-> Đã lưu kết quả đánh giá chi tiết vào: {models_eval_path}")
    except PermissionError:
        print(f"   [!] File model_evaluation_summary.json đang mở, giữ nguyên.")

    return trained_models, eval_results


def predict_patient_risk(model, scaler_obj, gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c, glucose):
    """
    Hàm dự đoán nguy cơ mắc bệnh tiểu đường cho bệnh nhân mới.
    """
    g_val = 1 if str(gender).lower() in ['male', 'nam', '1'] else 0
    smoking_cats = ['No Info', 'current', 'ever', 'former', 'never', 'not current']
    smk_features = [1 if str(smoking_history).lower() == cat.lower() else 0 for cat in smoking_cats]

    # Chuẩn hóa Z-Score cho 4 biến số bằng mean & std đã lưu
    mean_vec = scaler_obj['mean']
    std_vec = scaler_obj['std']
    raw_num = np.array([age, bmi, HbA1c, glucose], dtype=np.float64)
    scaled_num = (raw_num - mean_vec) / std_vec

    sample_vector = np.array([[g_val, scaled_num[0], hypertension, heart_disease, scaled_num[1], scaled_num[2], scaled_num[3]] + smk_features])

    pred = model.predict(sample_vector)[0]
    prob = model.predict_proba(sample_vector)[0, 1]

    status = "NGUY CƠ CAO (MẮC BỆNH TIỂU ĐƯỜNG)" if pred == 1 else "NGUY CƠ THẤP (KHỎE MẠNH / BÌNH THƯỜNG)"
    print("\n" + "="*50)
    print("KẾT QUẢ SÀNG LỌC BỆNH NHÂN LÂM SÀNG")
    print("="*50)
    print(f"• Tuổi: {age} | Giới tính: {gender} | BMI: {bmi}")
    print(f"• HbA1c: {HbA1c}% | Đường huyết tức thời: {glucose} mg/dL")
    print(f"• Tiền sử huyết áp: {hypertension} | Tiền sử bệnh tim: {heart_disease}")
    print(f"• Thói quen hút thuốc: {smoking_history}")
    print(f"• KẾT LUẬN: {status}")
    print(f"• XÁC SUẤT NGUY CƠ: {prob*100:.2f}%\n")
    return pred, prob


def main():
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning hoàn chỉnh - Chẩn đoán tiểu đường.")
    parser.add_argument('--no-eda', action='store_true', help='Bỏ qua bước tạo biểu đồ EDA để chạy nhanh hơn.')
    args = parser.parse_args()

    print("*" * 75)
    print("*" + " "*73 + "*")
    print("*" + "    HỌC PHẦN: HỌC MÁY (MACHINE LEARNING) - BÀI KIỂM TRA KỸ NĂNG SỐ 1".center(73) + "*")
    print("*" + "    ĐỀ TÀI: CHẨN ĐOÁN SỚM NGUY CƠ TIỂU ĐƯỜNG TỪ CHỈ SỐ SINH HIỆU".center(73) + "*")
    print("*" + "    NHÓM 3: NGUYỄN LÊ HOÀNG, NGÔ XUÂN ĐỈNH, HÀ ĐỨC TRUNG, TRẦN ĐỨC ĐẠT".center(73) + "*")
    print("*" + " "*73 + "*")
    print("*" * 75)

    # 1. Thu thập dữ liệu
    records, header = load_raw_data()

    # 2. EDA
    if not args.no_eda:
        generate_all_eda_figures(records)
    else:
        print("\n-> Đã bỏ qua bước trực quan hóa EDA (--no-eda).")

    # 3. Tiền xử lý
    data = run_preprocessing_pipeline(records, header)

    # 4. Xuất file dữ liệu sạch
    save_processed_csvs(data)

    # 5. Huấn luyện, Đánh giá & So sánh các mô hình Machine Learning
    trained_models, eval_results = train_and_evaluate_models(data)

    # 6. Thử nghiệm Demo dự đoán bệnh nhân lâm sàng
    print("\n" + "="*70)
    print("6. THỬ NGHIỆM DỰ ĐOÁN LÂM SÀNG BỆNH NHÂN MỚI (INFERENCE DEMO)")
    print("="*70)
    best_model = trained_models['Gradient Boosting']
    scaler_obj = data['scaler_obj']

    print("--- [Ca 1]: Bệnh nhân nam 55 tuổi, béo phì BMI 32.5, HbA1c 7.5%, Đường huyết 200 mg/dL ---")
    predict_patient_risk(best_model, scaler_obj, gender='Male', age=55, hypertension=1,
                         heart_disease=0, smoking_history='former', bmi=32.5, HbA1c=7.5, glucose=200)

    print("--- [Ca 2]: Người trẻ nữ 24 tuổi, thể trạng cân đối BMI 21.2, HbA1c 5.0%, Đường huyết 95 mg/dL ---")
    predict_patient_risk(best_model, scaler_obj, gender='Female', age=24, hypertension=0,
                         heart_disease=0, smoking_history='never', bmi=21.2, HbA1c=5.0, glucose=95)

    print("="*70)
    print("HOÀN THÀNH TOÀN BỘ DỰ ÁN HỌC MÁY THÀNH CÔNG RỰC RỠ!")
    print(f"• Dữ liệu sạch đã lưu tại:    '{PROCESSED_DIR}/'")
    print(f"• Toàn bộ 10 biểu đồ đã lưu: '{FIGURES_DIR}/'")
    print(f"• Mô hình tối ưu đã lưu tại: '{PROCESSED_DIR}/best_model.pkl'")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
