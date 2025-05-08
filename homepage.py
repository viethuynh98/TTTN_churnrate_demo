# import streamlit as st

# st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )

# st.write("# Welcome to Streamlit! 👋")

# st.sidebar.success("Select a demo above.")

# st.markdown(
#     """
#     Streamlit is an open-source app framework built specifically for
#     Machine Learning and Data Science projects.
#     **👈 Select a demo from the sidebar** to see some examples
#     of what Streamlit can do!
#     ### Want to learn more?
#     - Check out [streamlit.io](https://streamlit.io)
#     - Jump into our [documentation](https://docs.streamlit.io)
#     - Ask a question in our [community
#         forums](https://discuss.streamlit.io)
#     ### See more complex demos
#     - Use a neural net to [analyze the Udacity Self-driving Car Image
#         Dataset](https://github.com/streamlit/demo-self-driving)
#     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
# """
# )

import streamlit as st

# Cấu hình trang
st.set_page_config(page_title="Telco Churn Prediction - Home", layout="wide")

# Header
st.title("📊 Telco Customer Churn Prediction App")
st.markdown("""
Chào mừng bạn đến với ứng dụng dự đoán khả năng **rời bỏ của khách hàng** ngành viễn thông!

Ứng dụng sử dụng mô hình **AI tiên tiến** để phân tích dữ liệu và đưa ra **xác suất churn (rời bỏ)** cho từng khách hàng.

---
""")

# Sidebar hướng dẫn
st.sidebar.success("👉 Chọn một chức năng từ menu bên trái")

# Nội dung chính
st.markdown("""
### 🧭 Các chức năng chính:

- 📂 **Dự đoán từ file CSV**: Tải lên danh sách khách hàng để dự đoán hàng loạt.
- ✍️ **Nhập tay 1 khách hàng**: Thử nghiệm dự đoán trực tiếp cho một khách hàng cụ thể.
- 📈 **Xem báo cáo tổng quan**: Tổng hợp kết quả và biểu đồ trực quan từ các lần dự đoán.

---

### ❓ Câu hỏi thường gặp

**📁 Tôi cần chuẩn bị gì để upload dữ liệu?**  
- File CSV phải chứa các cột thông tin khách hàng như: `tenure`, `monthly_charges`, `internet_service`, `online_security`, v.v.
- Có thể xem [📘 file mẫu tại đây](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

**⚙️ Dữ liệu sẽ được xử lý như thế nào?**  
- Hệ thống tự động xử lý missing data, encoding, scaling, feature selection và dự đoán bằng mô hình LightGBM đã huấn luyện.

**🧪 Mục tiêu là gì?**  
- Phát hiện khách hàng có **nguy cơ rời bỏ cao** để xây dựng chiến lược giữ chân phù hợp.
""")

# Footer
st.markdown("---")
st.caption("👨‍💻 Developed by [VietHQ] | Data source: Telco Customer Churn - Kaggle")
