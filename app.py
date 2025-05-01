import streamlit as st
import numpy as np
import joblib

# Load model và feature selector
model = joblib.load('lightgbm_churn_model.pkl')
feature_selector = joblib.load('feature_selector.pkl')  # nếu cần, hoặc bỏ nếu đã lọc từ trước

st.title('Customer Churn Prediction App 📉')
st.markdown('Nhập thông tin khách hàng bên dưới để dự đoán khả năng rời bỏ.')

# ======= Form nhập liệu =======
tenure = st.slider('Thời gian sử dụng (tháng)', 0, 72, 12)
monthly_charges = st.number_input('Cước hàng tháng ($)', min_value=0.0, value=70.0)
total_charges = st.number_input('Tổng chi tiêu ($)', min_value=0.0, value=1000.0)

dependents_yes = st.selectbox('Có người phụ thuộc?', ['Không', 'Có']) == 'Có'
internet_service = st.selectbox('Loại dịch vụ Internet', ['Fiber optic', 'DSL/Khác', 'Không sử dụng'])
online_security_yes = st.selectbox('Có sử dụng Online Security?', ['Không', 'Có']) == 'Có'
tech_support_yes = st.selectbox('Có sử dụng Tech Support?', ['Không', 'Có']) == 'Có'

contract = st.selectbox('Loại hợp đồng', ['Month-to-month', 'One year', 'Two year'])
paperless_billing_yes = st.selectbox('Có sử dụng hóa đơn điện tử?', ['Không', 'Có']) == 'Có'
payment_method = st.selectbox('Phương thức thanh toán', ['Electronic check', 'Khác'])

# ======= Chuyển dữ liệu thành vector input =======
input_data = np.array([
    tenure,
    monthly_charges,
    total_charges,
    int(dependents_yes),
    int(internet_service == 'Fiber optic'),
    int(internet_service == 'Không sử dụng'),
    int(online_security_yes),
    int(tech_support_yes),
    int(contract == 'Month-to-month'),
    int(contract == 'One year'),
    int(contract == 'Two year'),
    int(paperless_billing_yes),
    int(payment_method == 'Electronic check')
]).reshape(1, -1)

# ======= Dự đoán =======
if st.button('Dự đoán'):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader('Kết quả dự đoán:')
    if prediction == 1:
        st.error(f'⚠️ Khách hàng có thể sẽ rời bỏ. Xác suất: {proba:.2%}')
    else:
        st.success(f'✅ Khách hàng có khả năng sẽ ở lại. Xác suất rời bỏ: {proba:.2%}')
