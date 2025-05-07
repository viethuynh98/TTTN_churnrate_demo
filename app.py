import streamlit as st
import pymysql
import pandas as pd

# Lấy thông tin từ secrets
mysql_config = st.secrets["mysql"]

# Hàm kết nối
@st.cache_resource
def connect_db():
    return pymysql.connect(
        host=mysql_config["host"],
        user=mysql_config["user"],
        password=mysql_config["password"],
        database=mysql_config["database"],
        port=mysql_config["port"]
    )

# Load dữ liệu từ bảng
def load_data(table_name):
    conn = connect_db()
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# Giao diện
st.title("🔍 Xem dữ liệu từ MySQL")

table = st.text_input("Nhập tên bảng cần xem:", "city")

if st.button("Tải dữ liệu"):
    try:
        df = load_data(table)
        st.success(f"✅ Đã tải {len(df)} dòng từ bảng `{table}`")
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Lỗi: {e}")
