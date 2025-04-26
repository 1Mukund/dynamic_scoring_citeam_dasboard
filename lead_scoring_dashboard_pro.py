import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Dynamic Lead Scoring", layout="wide")
st.title("Dynamic Lead Scoring & Engagement System")

# Initialize checklist states
checklist = {
    "Upload Engine": False,
    "Auto-Clean Columns": False,
    "Feature Extraction": False,
    "Auto-Weight Learning": False,
    "Dynamic Scoring Engine": False,
    "Dynamic Bucketing": False,
    "Transparent Scoring Math Display": False,
    "Per-Lead Behavior Contribution": False,
    "Dynamic Content Suggestion": False,
    "Visualizations": False,
    "Download Updated Excel": False,
    "Download Scoring Logic Report": False
}

st.header("Checklist Progress")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    checklist["Upload Engine"] = True

    # Read and clean data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)
    checklist["Auto-Clean Columns"] = True

    feature_cols = [
        'CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits',
        'WhatsappInbound', 'WhatsappOutbound',
        'daysSinceLastWebActivity', 'daysSinceLastInbound', 'daysSinceLastOutbound',
        'HighValuePageViews', 'DownloadedFilesCount'
    ]
    checklist["Feature Extraction"] = True

    # Feature Scaling
    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(df[feature_cols])
    feature_df = pd.DataFrame(feature_data, columns=feature_cols)

    # Feature Correlation and Weight Calculation
    corrs = pd.DataFrame()
    corrs['feature'] = feature_cols
    corrs['correlation_to_inbound'] = [df[f].corr(df['WhatsappInbound']) for f in feature_cols]
    corrs['abs_corr'] = corrs['correlation_to_inbound'].abs()
    corrs['weight'] = corrs['abs_corr'] / corrs['abs_corr'].sum()
    weights = corrs.set_index('feature')['weight'].to_dict()
    checklist["Auto-Weight Learning"] = True

    # Lead Scoring Computation
    df['lead_score'] = feature_df.dot(pd.Series(weights))
    df['score_percentile'] = df['lead_score'].rank(pct=True) * 100
    checklist["Dynamic Scoring Engine"] = True

    # Lead Categorization (Dynamic Bucketing)
    num_buckets = 5
    df['lead_bucket'] = pd.qcut(df['score_percentile'], q=num_buckets, labels=[f'Bucket_{i+1}' for i in range(num_buckets)])
    checklist["Dynamic Bucketing"] = True

    def suggest_message(bucket):
        return {
            'Bucket_5': "You're close! Let's schedule your site visit.",
            'Bucket_4': "Interested in pricing or EMI details?",
            'Bucket_3': "Here's a project walkthrough.",
            'Bucket_2': "See why our project stands out!",
            'Bucket_1': "Questions? We're here."
        }.get(bucket, "Special offers available!")

    df['recommended_message'] = df['lead_bucket'].apply(suggest_message)
    checklist["Dynamic Content Suggestion"] = True

    # Scoring Transparency Display
    st.subheader("Scoring Logic Transparency")
    st.dataframe(corrs)
    checklist["Transparent Scoring Math Display"] = True

    # Leads Table
    st.subheader("Leads Scored")
    st.dataframe(df[['LeadId', 'lead_score', 'lead_bucket', 'recommended_message']])

    # Per-Lead Contribution Table
    st.subheader("Per-Lead Feature Contributions")
    contributions_df = feature_df.copy()
    for col in feature_cols:
        contributions_df[col] = contributions_df[col] * weights[col]
    contributions_df['LeadId'] = df['LeadId']
    st.dataframe(contributions_df.set_index('LeadId'))
    checklist["Per-Lead Behavior Contribution"] = True

    # Visualizations
    st.subheader("Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['lead_score'], bins=20, ax=ax)
    st.pyplot(fig)

    st.subheader("Bucket Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='lead_bucket', data=df, order=[f'Bucket_{i+1}' for i in range(num_buckets)], ax=ax2)
    st.pyplot(fig2)
    checklist["Visualizations"] = True

    # Export Results
    buffer_leads = io.BytesIO()
    with pd.ExcelWriter(buffer_leads, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer_leads.seek(0)

    st.download_button("Download Leads with Scores", buffer_leads, "scored_leads.xlsx")
    checklist["Download Updated Excel"] = True

    buffer_logic = io.BytesIO()
    with pd.ExcelWriter(buffer_logic, engine='openpyxl') as writer:
        corrs.to_excel(writer, index=False)
    buffer_logic.seek(0)

    st.download_button("Download Scoring Logic Report", buffer_logic, "scoring_logic.xlsx")
    checklist["Download Scoring Logic Report"] = True

    # Display final checklist
    st.header("✅ Final Progress Tracker")
    for task, status in checklist.items():
        st.write(f"{'✅' if status else '❌'} {task}")

else:
    st.info("Please upload an Excel file to proceed with Dynamic Lead Scoring.")
