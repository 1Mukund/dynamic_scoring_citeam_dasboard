import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Lead Scoring Dashboard", layout="wide")

st.title("📊 Lead Scoring & Engagement Strategy Dashboard")
st.markdown("Upload your lead data (.xlsx format). The app will score and bucket leads automatically.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

def dynamic_weights(df, features):
    variances = df[features].var()
    weights = variances / variances.sum()
    return weights

def calculate_dynamic_score(row, weights):
    score = 0
    for feature, weight in weights.items():
        score += row.get(feature, 0) * weight
    return score * 100  # scale up

def categorize_lead(row):
    if row['lead_score'] >= 150 and row.get('WhatsappInbound', 0) >= 2:
        return 'Hot'
    elif 100 <= row['lead_score'] < 150 and row.get('WhatsappInbound', 0) >= 1:
        return 'Engaged'
    elif 60 <= row['lead_score'] < 100:
        return 'Warm'
    elif 30 <= row['lead_score'] < 60:
        return 'Curious'
    elif row.get('daysSinceLastWebActivity', 999) > 25 or row.get('daysSinceLastInbound', 999) > 25:
        return 'Dormant'
    else:
        return 'Cold'

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")

    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df['daysSinceLastInbound'] = df.get('daysSinceLastInbound', pd.Series(999, index=df.index)).fillna(999)
    df['daysSinceLastOutbound'] = df.get('daysSinceLastOutbound', pd.Series(999, index=df.index)).fillna(999)

    features_to_use = ['CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits', 'HighValuePageViews', 'DownloadedFilesCount', 'WhatsappOutbound', 'WhatsappInbound']
    features_present = [feat for feat in features_to_use if feat in df.columns]

    weights = dynamic_weights(df, features_present)

    df['lead_score'] = df.apply(lambda row: calculate_dynamic_score(row, weights), axis=1)
    df['lead_bucket'] = df.apply(categorize_lead, axis=1)

    st.subheader("🧠 Scoring Summary")
    st.dataframe(df[['LeadId'] + ['lead_score', 'lead_bucket', 'CurrentStage'] if 'CurrentStage' in df.columns else ['lead_score', 'lead_bucket']])

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

    st.markdown("### 🎯 Content & Engagement Recommendations")
    strategies = {
        'Hot': '✅ Immediate personal WhatsApp/call + Send cost sheet + Site visit push',
        'Engaged': '🟡 Send walkthrough videos + clear pricing + interest form',
        'Warm': '🔵 Testimonials, short videos, ROI case studies',
        'Curious': '🟠 Light educational content, blog posts, project USP reels',
        'Cold': '⚪ Monthly newsletters, occasional offers',
        'Dormant': '🔴 Targeted reactivation message with limited-time incentive',
    }
    for bucket, recommendation in strategies.items():
        st.markdown(f"**{bucket}** → {recommendation}")

    st.markdown("---")
    st.markdown("### 🔍 Scoring Logic Explained")

    scoring_explanation = ""
    for feature, weight in weights.items():
        scoring_explanation += f"- **{feature}** ➔ Weight: {round(weight * 100, 2)}%\n"
    st.markdown(scoring_explanation)

else:
    st.info("Upload an Excel file with columns like: `LeadId`, `CumulativeTime`, `WhatsappInbound`, `WhatsappOutbound`, `Unqiue_Visits`, etc.")
