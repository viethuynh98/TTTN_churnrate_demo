import streamlit as st
import pandas as pd
from pipeline.pipepline_module import (
    preprocess_input_data,
    transform_input_data,
    predict_churn
)

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction from CSV Upload")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # read_csv
        df_raw = pd.read_csv(uploaded_file)

        # print_data
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df_raw.head())

        # Step 1:
        df_for_prediction = df_raw.copy()
        df_for_dashboard = df_raw.copy()

        # Step 2: agg_columns
        internet_sub_services = ['online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
        df_for_dashboard['amt_internet_services'] = df_for_dashboard[internet_sub_services].apply(lambda row: sum(row == 'Yes'), axis=1)

        digital_cols = ['online_security', 'online_backup', 'device_protection', 'tech_support']
        df_for_dashboard['amt_digital_services'] = df_for_dashboard[digital_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

        streaming_cols = ['streaming_tv', 'streaming_movies']
        df_for_dashboard['amt_streaming_services'] = df_for_dashboard[streaming_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

        # Step 3: preprocessing + predicttion
        st.text("🚀 Step 1: Tiền xử lý dữ liệu (Preprocessing)...")
        df_for_prediction_processed = preprocess_input_data(df_for_prediction)
        st.text("✅ Data preprocessing completed!")

        st.text("🚀 Step 2: Chuyển đổi dữ liệu (Data Transformation)...")
        df_for_prediction_transformed = transform_input_data(df_for_prediction_processed)
        st.text("✅ Data transformation completed!")

        st.text("🚀 Step 3: Dự đoán churn (Prediction)...")
        df_with_predictions = predict_churn(df_for_prediction_transformed)
        st.text("✅ Churn prediction completed!")


        # dashboard
        df_for_dashboard['prob_churn'] = df_with_predictions['churn_probability']
        df_for_dashboard['churn_predicted'] = (df_for_dashboard['prob_churn'] > 0.5).astype(int)
        
        # data for dashboard
        st.session_state['df_dashboard'] = df_for_dashboard


        st.subheader("📈 Prediction Results")
        st.dataframe(df_for_dashboard.head())

        # Step 4: download
        csv_out = df_for_dashboard.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Results", data=csv_out,
                        file_name="churn_predictions_with_aggregates.csv", mime="text/csv")


    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
