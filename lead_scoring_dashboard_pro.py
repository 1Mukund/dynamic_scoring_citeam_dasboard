
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Dynamic Lead Scoring", layout="wide")
st.title("Dynamic Lead Scoring & Engagement System")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)

    feature_cols = [
        'CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits',
        'WhatsappInbound', 'WhatsappOutbound',
        'daysSinceLastWebActivity', 'daysSinceLastInbound', 'daysSinceLastOutbound',
        'HighValuePageViews', 'DownloadedFilesCount'
    ]

    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(df[feature_cols])
    feature_df = pd.DataFrame(feature_data, columns=feature_cols)

    corrs = pd.DataFrame()
    corrs['feature'] = feature_cols
    corrs['correlation_to_inbound'] = [df[f].corr(df['WhatsappInbound']) for f in feature_cols]
    corrs['abs_corr'] = corrs['correlation_to_inbound'].abs()
    corrs['weight'] = corrs['abs_corr'] / corrs['abs_corr'].sum()

    weights = corrs.set_index('feature')['weight'].to_dict()
    df['lead_score'] = feature_df.dot(pd.Series(weights))
    df['score_percentile'] = df['lead_score'].rank(pct=True) * 100

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

    st.subheader("Leads Scored")
    st.dataframe(df[['LeadId', 'lead_score', 'lead_bucket', 'recommended_message']])

    fig, ax = plt.subplots()
    sns.histplot(df['lead_score'], bins=20, ax=ax)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    sns.countplot(x='lead_bucket', data=df, order=['Hot', 'Engaged', 'Warm', 'Curious', 'Cold', 'Dormant'], ax=ax2)
    st.pyplot(fig2)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    st.download_button("Download Leads with Scores", buffer, "scored_leads.xlsx")

else:
    st.info("Upload an Excel file to get started.")
