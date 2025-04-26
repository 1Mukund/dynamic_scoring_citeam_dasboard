import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Dynamic Lead Scoring", layout="wide")
st.title("Dynamic Lead Scoring & Engagement System")

st.header("Checklist Progress")

uploaded_file = st.file_uploader("Step 1: Upload your Excel file", type=["xlsx"])

if uploaded_file:
    st.success("Step 1 Completed: File Uploaded")
    
    # Read and clean data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)

    feature_cols = [
        'CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits',
        'WhatsappInbound', 'WhatsappOutbound',
        'daysSinceLastWebActivity', 'daysSinceLastInbound', 'daysSinceLastOutbound',
        'HighValuePageViews', 'DownloadedFilesCount'
    ]

    # Step 2: Feature Scaling
    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(df[feature_cols])
    feature_df = pd.DataFrame(feature_data, columns=feature_cols)
    st.success("Step 2 Completed: Feature Scaling Done")

    # Step 3: Feature Correlation Analysis
    corrs = pd.DataFrame()
    corrs['feature'] = feature_cols
    corrs['correlation_to_inbound'] = [df[f].corr(df['WhatsappInbound']) for f in feature_cols]
    corrs['abs_corr'] = corrs['correlation_to_inbound'].abs()
    st.success("Step 3 Completed: Correlation Analysis Done")

    # Step 4: Dynamic Weight Calculation
    corrs['weight'] = corrs['abs_corr'] / corrs['abs_corr'].sum()
    weights = corrs.set_index('feature')['weight'].to_dict()
    st.success("Step 4 Completed: Dynamic Weights Generated")

    # Step 5: Lead Scoring Computation
    df['lead_score'] = feature_df.dot(pd.Series(weights))
    df['score_percentile'] = df['lead_score'].rank(pct=True) * 100
    st.success("Step 5 Completed: Lead Scoring Done")

    # Step 6: Lead Categorization
    def bucketize(p):
        if p >= 90:
            return 'Hot'
        elif p >= 75:
            return 'Engaged'
        elif p >= 50:
            return 'Warm'
        elif p >= 30:
            return 'Curious'
        elif p > 0:
            return 'Cold'
        else:
            return 'Dormant'

    df['lead_bucket'] = df['score_percentile'].apply(bucketize)

    def suggest_message(bucket):
        return {
            'Hot': "You're close! Let's schedule your site visit.",
            'Engaged': "Interested in pricing or EMI details?",
            'Warm': "Here's a project walkthrough.",
            'Curious': "See why our project stands out!",
            'Cold': "Questions? We're here.",
            'Dormant': "Special offers available!"
        }.get(bucket, "")

    df['recommended_message'] = df['lead_bucket'].apply(suggest_message)
    st.success("Step 6 Completed: Leads Categorized")

    # Step 7: Scoring Transparency Display
    st.subheader("Scoring Logic Transparency")
    st.dataframe(corrs)

    st.subheader("Leads Scored")
    st.dataframe(df[['LeadId', 'lead_score', 'lead_bucket', 'recommended_message']])

    st.subheader("Per-Lead Feature Contributions")
    contributions_df = feature_df.copy()
    for col in feature_cols:
        contributions_df[col] = contributions_df[col] * weights[col]
    contributions_df['LeadId'] = df['LeadId']
    st.dataframe(contributions_df.set_index('LeadId'))

    st.success("Step 7 Completed: Transparency Tables Shown")

    # Step 8: Export Results
    buffer_leads = io.BytesIO()
    with pd.ExcelWriter(buffer_leads, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer_leads.seek(0)

    st.download_button("Download Leads with Scores", buffer_leads, "scored_leads.xlsx")

    buffer_logic = io.BytesIO()
    with pd.ExcelWriter(buffer_logic, engine='openpyxl') as writer:
        corrs.to_excel(writer, index=False)
    buffer_logic.seek(0)

    st.download_button("Download Scoring Logic Report", buffer_logic, "scoring_logic.xlsx")

    st.success("Step 8 Completed: Files Ready for Download")

else:
    st.info("Please upload an Excel file to proceed with Dynamic Lead Scoring.")
