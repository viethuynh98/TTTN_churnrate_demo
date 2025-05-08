import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="📊 Churn Dashboard", layout="wide")
st.title("📊 Dashboard Phân Tích Churn Khách Hàng")

# ================== Load session data ==================
if "df_dashboard" not in st.session_state:
    st.warning("❗ Hãy tải dữ liệu ở trang 'CSV Upload' trước.")
    st.stop()

df = st.session_state.df_dashboard.copy()

# ================== Sidebar Filters ==================
st.sidebar.header("🔍 Bộ lọc dữ liệu")

# Filter by churn prediction
churn_options = ["Tất cả", "Churn", "Không churn"]
churn_filter = st.sidebar.radio("Churn dự đoán", churn_options)
if churn_filter == "Churn":
    df = df[df["churn_predicted"] == 1]
elif churn_filter == "Không churn":
    df = df[df["churn_predicted"] == 0]

# Filter by tenure
tenure_range = st.sidebar.slider("Khoảng thời gian sử dụng (tenure)", 
                                 int(df["tenure"].min()), int(df["tenure"].max()), 
                                 (0, 72))
df = df[df["tenure"].between(*tenure_range)]

# Filter by internet type
internet_types = df["internet_service"].dropna().unique().tolist()
selected_internet = st.sidebar.multiselect("Loại internet", internet_types, default=internet_types)
df = df[df["internet_service"].isin(selected_internet)]

# Filter by contract type
contract_types = df["contract"].dropna().unique().tolist()
selected_contracts = st.sidebar.multiselect("Loại hợp đồng", contract_types, default=contract_types)
df = df[df["contract"].isin(selected_contracts)]

# ================== Tổng quan ==================
st.subheader("📌 Tổng quan")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tổng số khách hàng", len(df))
col2.metric("Tỷ lệ churn", f"{df['churn_predicted'].mean()*100:.2f}%")
col3.metric("Doanh thu trung bình", f"${df['monthly_charges'].mean():.2f}")
col4.metric("Số khách hàng nguy cơ cao", (df['prob_churn'] > 0.8).sum())

# ================== Tabs phân tích ==================
st.subheader("📊 Phân tích trực quan")
tab1, tab2, tab3 = st.tabs(["Dịch vụ", "Phân phối xác suất", "Theo hợp đồng"])

with tab1:
    fig1 = px.histogram(df, x="amt_internet_services", color="churn_predicted",
                        barmode="group", labels={"churn_predicted": "Churn"}, 
                        title="Số dịch vụ Internet vs Churn")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, x="churn_predicted", y="monthly_charges", 
                  points="all", labels={"churn_predicted": "Churn"}, 
                  title="Chi phí hàng tháng theo churn")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig3 = px.histogram(df, x="prob_churn", nbins=30, 
                        title="Phân phối xác suất churn")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    fig4 = px.bar(df.groupby("contract")["churn_predicted"].mean().reset_index(), 
                  x="contract", y="churn_predicted", 
                  labels={"churn_predicted": "Tỷ lệ churn"}, 
                  title="Tỷ lệ churn theo loại hợp đồng")
    st.plotly_chart(fig4, use_container_width=True)

# ================== Phân khúc churn ==================
st.subheader("🔍 Phân khúc khách hàng theo xác suất churn")
high_risk = df[df["prob_churn"] > 0.8]
medium_risk = df[(df["prob_churn"] > 0.5) & (df["prob_churn"] <= 0.8)]
low_risk = df[df["prob_churn"] <= 0.5]

with st.expander("🟥 Nguy cơ cao (> 0.8)"):
    st.write(f"Số lượng: {len(high_risk)}")
    st.dataframe(high_risk.head(50), use_container_width=True)

with st.expander("🟨 Nguy cơ trung bình (0.5 - 0.8)"):
    st.write(f"Số lượng: {len(medium_risk)}")
    st.dataframe(medium_risk.head(50), use_container_width=True)

with st.expander("🟩 An toàn (<= 0.5)"):
    st.write(f"Số lượng: {len(low_risk)}")
    st.dataframe(low_risk.head(50), use_container_width=True)

# ================== Xem chi tiết & Tải xuống ==================
st.subheader("🧾 Bảng dữ liệu chi tiết")
st.dataframe(df.head(100), use_container_width=True)

csv_out = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Tải dữ liệu đã lọc", data=csv_out,
                   file_name="filtered_churn_data.csv", mime="text/csv")
