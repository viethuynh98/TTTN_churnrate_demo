import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth

# === Load model và thông tin ===
model = joblib.load('lightgbm_churn_model.pkl')
selected_features = joblib.load('selected_features.pkl')
comparison_test_result = joblib.load('comparison_test_result.pkl')
scaler = joblib.load('scaler.pkl')

numeric_features = ['tenure', 'monthly_charges', 'total_charges']

st.set_page_config(layout='wide')
st.title('📉 Customer Churn Prediction Dashboard')
# st.markdown('Chọn chế độ nhập dữ liệu hoặc kiểm thử với tập test có sẵn để xem khả năng rời bỏ và giải thích mô hình.')

# === Tab điều hướng ===
tab1, tab2 = st.tabs(["🧍 Nhập dữ liệu thủ công", "📁 Dữ liệu test có sẵn"])

# === TAB 1: NHẬP DỮ LIỆU THỦ CÔNG ===
with tab1:
    with st.sidebar:
        st.header("🎛️ Nhập thông tin khách hàng")

        tenure = st.slider('📅 Thời gian sử dụng (tháng)', 0, 72, 12)
        monthly_charges = st.number_input('💵 Cước hàng tháng ($)', min_value=0.0, value=70.0)
        total_charges = st.number_input('💰 Tổng chi tiêu ($)', min_value=0.0, value=1000.0)

        dependents_yes = 1 if st.selectbox('👨‍👩‍👧 Có người phụ thuộc?', ['Không', 'Có']) == 'Có' else 0
        phone_service_yes = 1 if st.selectbox('📞 Dịch vụ điện thoại?', ['Không', 'Có']) == 'Có' else 0
        internet_service_fiber_optic = 1 if st.selectbox('🌐 Sử dụng cáp quang?', ['Không', 'Có']) == 'Có' else 0
        online_security_yes = 1 if st.selectbox('🔐 Online Security?', ['Không', 'Có']) == 'Có' else 0
        online_backup_yes = 1 if st.selectbox('💾 Online Backup?', ['Không', 'Có']) == 'Có' else 0
        tech_support_yes = 1 if st.selectbox('🛠️ Tech Support?', ['Không', 'Có']) == 'Có' else 0

        contract_type = st.selectbox('📄 Loại hợp đồng', ['Month-to-month', 'Two year', 'Khác'])
        contract_month_to_month = 1 if contract_type == 'Month-to-month' else 0
        contract_two_year = 1 if contract_type == 'Two year' else 0

        paperless_billing_yes = 1 if st.selectbox('🧾 Hóa đơn điện tử?', ['Không', 'Có']) == 'Có' else 0
        payment_method_electronic_check = 1 if st.selectbox('💳 Thanh toán bằng electronic check?', ['Không', 'Có']) == 'Có' else 0

    input_data = np.array([
        tenure,
        monthly_charges,
        total_charges,
        dependents_yes,
        phone_service_yes,
        internet_service_fiber_optic,
        online_security_yes,
        online_backup_yes,
        tech_support_yes,
        contract_month_to_month,
        contract_two_year,
        paperless_billing_yes,
        payment_method_electronic_check
    ]).reshape(1, -1)

    input_df = pd.DataFrame(input_data, columns=selected_features)
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    if st.button('🔍 Dự đoán'):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader('📊 Kết quả dự đoán:')
        if prediction == 1:
            st.error(f'⚠️ Khách hàng có thể sẽ rời bỏ.\n\nXác suất: **{proba:.2%}**')
        else:
            st.success(f'✅ Khách hàng có khả năng ở lại.\n\nXác suất rời bỏ: **{proba:.2%}**')

        st.subheader("📉 Dữ liệu đầu vào sau xử lý")
        st.dataframe(input_df.T, use_container_width=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values

        st.subheader("🧠 Giải thích bằng SHAP")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=selected_features),
            max_display=13, show=False)
        st.pyplot(fig)

        top_features = pd.Series(shap_vals[0], index=selected_features).sort_values(key=abs, ascending=False)
        top_contribs = top_features.head(3)
        reasons = []
        for feat, val in top_contribs.items():
            direction = "tăng" if val > 0 else "giảm"
            reasons.append(f"- `{feat}` có xu hướng {direction} xác suất rời bỏ (SHAP = {val:.3f})")

        st.markdown("**📝 Giải thích tự động:**")
        if prediction == 1:
            st.write("Khách hàng có khả năng rời bỏ chủ yếu do:")
        else:
            st.write("Khách hàng có khả năng ở lại vì:")
        st.markdown("\n".join(reasons))

# === TAB 2: KIỂM THỬ TẬP DỮ LIỆU CÓ SẴN ===
# === TAB 2: KIỂM THỬ TẬP DỮ LIỆU CÓ SẴN ===
with tab2:
    st.subheader("🔁 Kiểm thử lại với dữ liệu từ tập test")

    test_idx = st.number_input("Chọn chỉ số bản ghi để kiểm thử lại", 
                               min_value=0, 
                               max_value=len(comparison_test_result) - 1, 
                               value=0, 
                               step=1)

    test_sample = comparison_test_result.drop(columns=['true_label', 'predicted_label', 'correct']).iloc[[test_idx]]
    true_label = comparison_test_result.iloc[test_idx]['true_label']

    # Hiển thị chi tiết feature input lên trước
    with st.expander("🧾 Xem chi tiết feature input của bản ghi", expanded=True):
        st.dataframe(test_sample.T.rename(columns={test_sample.index[0]: f"Bản ghi #{test_idx}"}), use_container_width=True)

    # Dự đoán mới
    pred_new = model.predict(test_sample)[0]
    proba_new = model.predict_proba(test_sample)[0][1]

    st.markdown(f"""
    **🎯 Bản ghi #{test_idx}**

    - Nhãn thật: **{true_label}**
    - Dự đoán mới: **{pred_new}**
    - Xác suất rời bỏ: **{proba_new:.2%}**
    """)

    if pred_new == true_label:
        st.success("✅ Dự đoán  đúng như nhãn thật.")
    else:
        st.error("❌ Dự đoán KHÔNG khớp với nhãn thật.")

    # Biểu đồ SHAP
    st.subheader("🧠 Biểu đồ SHAP cho bản ghi")
    explainer = shap.TreeExplainer(model)
    shap_values_test = explainer.shap_values(test_sample)
    shap_vals_test = shap_values_test[0] if isinstance(shap_values_test, list) else shap_values_test

    fig2, ax2 = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_vals_test[0],
        base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=test_sample.iloc[0],
        feature_names=selected_features),
        max_display=13, show=False)
    st.pyplot(fig2)

    # Giải thích tự động
    top_features_test = pd.Series(shap_vals_test[0], index=selected_features).sort_values(key=abs, ascending=False)
    top_contribs_test = top_features_test.head(3)
    reasons_test = []
    for feat, val in top_contribs_test.items():
        direction = "tăng" if val > 0 else "giảm"
        reasons_test.append(f"- `{feat}` có xu hướng {direction} xác suất rời bỏ (SHAP = {val:.3f})")

    st.markdown("**📝 Giải thích tự động:**")
    if pred_new == 1:
        st.write("Khách hàng này có thể rời bỏ chủ yếu do:")
    else:
        st.write("Khách hàng này có khả năng ở lại vì:")
    st.markdown("\n".join(reasons_test))
