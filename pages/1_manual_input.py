import streamlit as st
import pandas as pd
from pipeline.pipepline_module import (
    preprocess_input_data,
    transform_input_data,
    predict_churn
)

st.set_page_config(page_title="Telco Churn Prediction - Manual Input", layout="wide")
st.title("✍️ Dự đoán churn từ dữ liệu nhập tay")

st.markdown("""
Hãy điền thông tin của một khách hàng bên dưới để dự đoán **khả năng rời bỏ (churn probability)**.
""")

with st.form("manual_input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Giới tính", ["Male", "Female"], index=0)
        senior_citizen = st.selectbox("Khách hàng cao tuổi?", [0, 1], index=0)
        partner = st.selectbox("Có vợ/chồng?", ["Yes", "No"], index=1)
        dependents = st.selectbox("Có người phụ thuộc?", ["Yes", "No"], index=1)
        tenure = st.number_input("Thời gian sử dụng (tháng)", min_value=0, max_value=100, step=1, value=24)
        phone_service = st.selectbox("Sử dụng điện thoại?", ["Yes", "No"], index=0)
        multiple_lines = st.selectbox("Nhiều đường dây?", ["Yes", "No", "No phone service"], index=1)

    with col2:
        internet_service = st.selectbox("Loại Internet", ["DSL", "Fiber optic", "No"], index=0)
        online_security = st.selectbox("Bảo mật online?", ["Yes", "No", "No internet service"], index=1)
        online_backup = st.selectbox("Sao lưu online?", ["Yes", "No", "No internet service"], index=1)
        device_protection = st.selectbox("Bảo vệ thiết bị?", ["Yes", "No", "No internet service"], index=1)
        tech_support = st.selectbox("Hỗ trợ kỹ thuật?", ["Yes", "No", "No internet service"], index=1)
        streaming_tv = st.selectbox("Xem TV online?", ["Yes", "No", "No internet service"], index=1)
        streaming_movies = st.selectbox("Xem phim online?", ["Yes", "No", "No internet service"], index=1)

    with col3:
        contract = st.selectbox("Loại hợp đồng", ["Month-to-month", "One year", "Two year"], index=0)
        paperless_billing = st.selectbox("Hóa đơn điện tử?", ["Yes", "No"], index=0)
        payment_method = st.selectbox("Hình thức thanh toán", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], index=0)
        monthly_charges = st.number_input("Phí hàng tháng ($)", min_value=0.0, step=0.1, value=70.0)
        total_charges = st.number_input("Tổng chi tiêu ($)", min_value=0.0, step=0.1, value=1500.0)

    submitted = st.form_submit_button("🚀 Dự đoán")

if submitted:
    try:
        input_dict = {
            "gender": [gender],
            "senior_citizen": [senior_citizen],
            "partner": [partner],
            "dependents": [dependents],
            "tenure": [tenure],
            "phone_service": [phone_service],
            "multiple_lines": [multiple_lines],
            "internet_service": [internet_service],
            "online_security": [online_security],
            "online_backup": [online_backup],
            "device_protection": [device_protection],
            "tech_support": [tech_support],
            "streaming_tv": [streaming_tv],
            "streaming_movies": [streaming_movies],
            "contract": [contract],
            "paperless_billing": [paperless_billing],
            "payment_method": [payment_method],
            "monthly_charges": [monthly_charges],
            "total_charges": [total_charges]
        }

        df_input = pd.DataFrame(input_dict)

        # Tiền xử lý    
        df_processed = preprocess_input_data(df_input)
        df_transformed = transform_input_data(df_processed)
        df_result = predict_churn(df_transformed)

        # ✅ In kết quả df_result
        st.subheader("📄 Kết quả đầy đủ:")
        st.dataframe(df_result, use_container_width=True)


       # Hiển thị xác suất churn
        prob = df_result['churn_probability'].values[0]
        st.success(f"🔮 Xác suất churn của khách hàng này là: **{prob:.2%}**")

        if prob > 0.5:
            st.warning("⚠️ Khách hàng có nguy cơ **rời bỏ cao**!")
        else:
            st.info("✅ Khách hàng có **nguy cơ rời bỏ thấp**.")

        # === Giải thích bằng SHAP ===
        import shap
        import joblib
        import matplotlib.pyplot as plt

        st.subheader("🧠 Giải thích bằng SHAP")

        # Load lại model, scaler, feature list
        model = joblib.load('models/lightgbm_churn_model_ver2.pkl')
        selected_features = joblib.load('models/selected_features.pkl')

        input_for_shap = df_result[selected_features].copy()

        # SHAP explain
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_for_shap)
        shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values

        # Vẽ biểu đồ waterfall
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=input_for_shap.iloc[0],
            feature_names=selected_features),
            max_display=13, show=False)
        st.pyplot(fig)

        # Giải thích tự động
        top_features = pd.Series(shap_vals[0], index=selected_features).sort_values(key=abs, ascending=False)
        top_contribs = top_features.head(3)
        reasons = []
        for feat, val in top_contribs.items():
            direction = "tăng" if val > 0 else "giảm"
            reasons.append(f"- `{feat}` có xu hướng {direction} xác suất rời bỏ (SHAP = {val:.3f})")

        st.markdown("**📝 Giải thích tự động:**")
        if prob > 0.5:
            st.write("Khách hàng có khả năng rời bỏ chủ yếu do:")
        else:
            st.write("Khách hàng có khả năng ở lại vì:")
        st.markdown("\n".join(reasons))

    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
