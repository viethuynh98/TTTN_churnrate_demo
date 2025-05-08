import pandas as pd
import inflection
import streamlit as st
from joblib import load

# === Load các đối tượng đã lưu ===
model = load('models/lightgbm_churn_model_ver2.pkl')
selected_features = load('models/selected_features.pkl')
ohe = load('models/ohe_encoder.pkl')
le = load('models/label_encoder.pkl')
scaler = load('models/scaler.pkl')
selector = load('models/rfe_selector.pkl')

# === Các cột số và danh mục ===
column_numerical = ['tenure', 'monthly_charges', 'total_charges']

# === Hàm xử lý cơ bản + tạo biến tổng hợp ===
def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Chuẩn hóa tên cột
    df.columns = [inflection.underscore(col).replace(' ', '_') for col in df.columns]

    # Loại bỏ cột customer_id nếu tồn tại
    if 'customer_id' in df.columns:
        df.drop(columns=['customer_id'], inplace=True)

    # Xử lý dữ liệu thiếu hoặc không đúng kiểu
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0).astype(float)

    # Map các giá trị No service
    df.replace({'No phone service': 'No', 'No internet service': 'No'}, inplace=True)

    # Map giá trị senior_citizen
    if 'senior_citizen' in df.columns:
        df['senior_citizen'] = df['senior_citizen'].replace({0: 'No', 1: 'Yes'})

    return df


# === Hàm encode và scale ===
def transform_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Phân loại các cột
    all_cols = df.columns.tolist()
    column_categorical = [col for col in all_cols if col not in column_numerical + ['churn']]

    # st.text(f"Categorical columns: {column_categorical}")  # Debug thông tin các cột categorical

    # # Kiểm tra các cột categorical trong bộ dữ liệu mới và đã huấn luyện
    # st.text(f"Columns in new data: {df.columns.tolist()}")
    # st.text(f"Columns in model (trained): {ohe.get_feature_names_out().tolist()}")  # Tên các cột đã được huấn luyện với OneHotEncoder

    # One-hot encoding
    try:
        cat_encoded = ohe.transform(df[column_categorical])
        # st.text("✅ OneHotEncoding completed successfully.")
        
        # Gán lại column_ohe sau khi OneHotEncoder đã transform
        column_ohe = [inflection.underscore(col).replace(' ', '_').replace('_(automatic)', '') for col in ohe.get_feature_names_out()]
        # st.text(f"Encoded column names (normalized): {column_ohe}")  # Debug danh sách cột đã mã hóa

    except Exception as e:
        # st.text(f"❌ Error during OneHotEncoding: {str(e)}")
        return df  # Return dataframe gốc nếu có lỗi để dễ dàng debug

    df_cat = pd.DataFrame(cat_encoded, columns=column_ohe, index=df.index)

    # Scale numerical
    try:
        df_num = pd.DataFrame(scaler.transform(df[column_numerical]), columns=column_numerical, index=df.index)
        # st.text("✅ Scaling completed successfully.")
    except Exception as e:
        # st.text(f"❌ Error during scaling: {str(e)}")
        return df  # Return dataframe gốc nếu có lỗi

    # Kết hợp lại
    df_encoded = pd.concat([df_num, df_cat], axis=1)

    # Debug thông tin về dataframe sau khi kết hợp
    # st.text(f"Columns after encoding and scaling: {df_encoded.columns.tolist()}")

    # Chọn feature
    try:
        selected_cols = selector.get_support(indices=True)
        selected_feature_names = df_encoded.columns[selected_cols]
        df_selected = pd.DataFrame(selector.transform(df_encoded), columns=selected_feature_names, index=df.index)

        # st.text("✅ Feature selection completed successfully.")
    except Exception as e:
        # st.text(f"❌ Error during feature selection: {str(e)}")
        return df  # Return dataframe gốc nếu có lỗi

    return df_selected


def predict_churn(df_transformed: pd.DataFrame) -> pd.DataFrame:
    st.text("🚀 Bắt đầu dự đoán churn (dữ liệu đã được transform)...")

    try:
        y_pred_proba = model.predict_proba(df_transformed)[:, 1]
        st.text("✅ Dự đoán thành công.")
    except Exception as e:
        st.text(f"❌ Lỗi khi dự đoán: {str(e)}")
        return None

    # Trả kết quả dưới dạng DataFrame
    df_result = pd.DataFrame(df_transformed.copy())
    df_result['churn_probability'] = y_pred_proba
    st.text("🎯 Hoàn tất thêm xác suất churn vào kết quả.")

    return df_result



