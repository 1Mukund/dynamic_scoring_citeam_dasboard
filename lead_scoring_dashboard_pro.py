# Step 1: Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Streamlit Page Setup
st.set_page_config(page_title="Intelligent Lead Scoring Dashboard", layout="wide")
st.title("📊 Intelligent Lead Scoring & Opportunity Detection Dashboard")

# Step 3: Uploading Data
uploaded_file = st.file_uploader("Upload Lead Excel File", type=["xlsx"])

if uploaded_file:
    # Step 4: Read Data
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")

    # Step 5: Basic Cleaning
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df['daysSinceLastInbound'] = df.get('daysSinceLastInbound', pd.Series(999, index=df.index)).fillna(999)
    df['daysSinceLastOutbound'] = df.get('daysSinceLastOutbound', pd.Series(999, index=df.index)).fillna(999)

    # Step 6: Intelligent Recency Calculations
    if 'LastVisitTimestamp' in df.columns:
        today = pd.Timestamp.now()
        df['LastVisitTimestamp'] = pd.to_datetime(df['LastVisitTimestamp'], errors='coerce')
        df['daysSinceLastWebActivity'] = (today - df['LastVisitTimestamp']).dt.days.fillna(999)

    if 'Inbound_Message_time' in df.columns:
        df['Inbound_Message_time'] = pd.to_datetime(df['Inbound_Message_time'], errors='coerce')
        df['daysSinceLastInbound'] = (today - df['Inbound_Message_time']).dt.days.fillna(999)

    if 'Outbound_Message_time' in df.columns:
        df['Outbound_Message_time'] = pd.to_datetime(df['Outbound_Message_time'], errors='coerce')
        df['daysSinceLastOutbound'] = (today - df['Outbound_Message_time']).dt.days.fillna(999)

    # Step 7: Feature Engineering
    behavior_features = ['CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits',
                          'HighValuePageViews', 'DownloadedFilesCount',
                          'WhatsappInbound', 'WhatsappOutbound',
                          'daysSinceLastWebActivity', 'daysSinceLastInbound']

    available_features = [feat for feat in behavior_features if feat in df.columns]

    # Step 8: Dynamic Correlation Analysis
    correlation_matrix = df[available_features].corr()
    engagement_features = ['CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits', 'HighValuePageViews']
    intent_features = ['DownloadedFilesCount', 'WhatsappInbound']
    recency_features = ['daysSinceLastWebActivity', 'daysSinceLastInbound']

    # Step 9: Scoring Calculation
    def calculate_score(row):
        score = 0

        # Engagement Score
        if all(feat in row for feat in engagement_features):
            page_per_visit = row['Number_of_Page_Visited'] / (row['Unqiue_Visits'] + 1e-5)
            if page_per_visit > 3:
                score += 10
            if row['CumulativeTime'] > df['CumulativeTime'].median():
                score += 10
            if row['HighValuePageViews'] > df['HighValuePageViews'].median():
                score += 15

        # Intent Score
        if all(feat in row for feat in intent_features):
            score += row['DownloadedFilesCount'] * 15
            score += row['WhatsappInbound'] * 20

        # Recency Score
        if 'daysSinceLastWebActivity' in row:
            if row['daysSinceLastWebActivity'] < df['daysSinceLastWebActivity'].median():
                score += 10
        if 'daysSinceLastInbound' in row:
            if row['daysSinceLastInbound'] < df['daysSinceLastInbound'].median():
                score += 10

        # Communication Adjustment
        if 'WhatsappOutbound' in row:
            score += row['WhatsappOutbound'] * 2

        return score

    df['lead_score'] = df.apply(calculate_score, axis=1)

    # Step 10: Dynamic Bucketing
    df['lead_percentile'] = df['lead_score'].rank(pct=True) * 100

    def categorize_lead(p):
        if p >= 90:
            return 'Hot'
        elif p >= 70:
            return 'Engaged'
        elif p >= 40:
            return 'Warm'
        elif p >= 20:
            return 'Curious'
        else:
            return 'Cold'

    df['lead_bucket'] = df['lead_percentile'].apply(categorize_lead)

    # Step 11: Opportunity Detection
    df['opportunity_tag'] = np.where(
        (df['lead_bucket'].isin(['Engaged', 'Warm'])) & (df['WhatsappInbound'] == 0),
        'High Web Activity, No WhatsApp',
        np.where(
            (df['WhatsappInbound'] >= 1) & (df['daysSinceLastWebActivity'] > 30),
            'Needs Immediate Closure',
            ''
        )
    )

    # Step 12: Dashboard Outputs
    st.subheader("🧠 Scoring Summary")
    display_columns = ['LeadId', 'lead_score', 'lead_bucket', 'opportunity_tag']
    if 'CurrentStage' in df.columns:
        display_columns.append('CurrentStage')
    st.dataframe(df[display_columns])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Bucket Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='lead_bucket', data=df, order=df['lead_bucket'].value_counts().index, palette="viridis", ax=ax)
        ax.set_title("Lead Buckets")
        st.pyplot(fig)

    with col2:
        st.markdown("### 📈 Lead Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['lead_score'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title("Lead Score Histogram")
        st.pyplot(fig)

    st.markdown("### 🎯 Opportunity Summary")
    st.dataframe(df[['LeadId', 'lead_bucket', 'opportunity_tag']][df['opportunity_tag'] != ''])

else:
    st.info("Upload an Excel file with columns like: `LeadId`, `CumulativeTime`, `WhatsappInbound`, `WhatsappOutbound`, `Unqiue_Visits`, etc.")
