import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Super AI Lead Intelligence", layout="wide")
st.title("Super AI-Based Dynamic Lead Scoring & Cluster Behavior Analysis")

st.sidebar.header("Settings")
lead_selection = st.sidebar.selectbox("Lead Selection", ["Top 5 per Cluster", "Top 10% Overall"])
content_strategy = st.sidebar.selectbox("Content Sensitivity", ["Aggressive", "Nurture"])

uploaded_file = st.file_uploader("Upload your Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)

    feature_types = {}
    for col in df.columns:
        if df[col].nunique() > 0.9 * len(df):
            feature_types[col] = 'id'
        elif 'date' in col.lower() or 'timestamp' in col.lower():
            feature_types[col] = 'date'
        elif df[col].dtype in [np.int64, np.float64]:
            feature_types[col] = 'numeric'
        else:
            feature_types[col] = 'categorical'

    behavior_features = [col for col, typ in feature_types.items() if typ == 'numeric']

    if not behavior_features:
        st.error("No valid numeric behavioral features detected.")
        st.stop()

    X = df[behavior_features]

    feature_variances = X.var()
    feature_weights = feature_variances / feature_variances.sum()
    df['lead_score'] = X.mul(feature_weights, axis=1).sum(axis=1)
    df['lead_score_normalized'] = (df['lead_score'] - df['lead_score'].min()) / (df['lead_score'].max() - df['lead_score'].min())

    silhouette_scores = []
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    best_k = np.argmax(silhouette_scores) + 2
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = final_model.fit_predict(X)

    cluster_behavior_profiles = {}
    for cid in sorted(df['cluster'].unique()):
        cluster_means = df[df['cluster'] == cid][behavior_features].mean()
        strong_behaviors = cluster_means[cluster_means > cluster_means.mean()].sort_values(ascending=False)
        if strong_behaviors.empty:
            cluster_behavior_profiles[cid] = "Dormant Behavior"
        else:
            top_behaviors = strong_behaviors.head(2).index.tolist()
            cluster_behavior_profiles[cid] = " & ".join([f"Strong {tb}" for tb in top_behaviors])

    df['cluster_profile'] = df['cluster'].map(cluster_behavior_profiles)

    def generate_dynamic_content(profile):
        if 'Whatsapp' in profile or 'Inbound' in profile:
            return "Start Personal Conversations"
        elif 'Page' in profile or 'Visit' in profile:
            return "Invite to Webinar or Site Visit"
        elif 'Download' in profile or 'HighValue' in profile:
            return "Offer Detailed Guides or Pricing"
        else:
            return "Send Educational Content to Nurture"

    df['content_recommendation'] = df['cluster_profile'].apply(generate_dynamic_content)

    st.subheader("Lead Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['lead_score_normalized'], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Cluster Behavior Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='cluster_profile', data=df, order=df['cluster_profile'].value_counts().index, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

    st.subheader("Download Leads")
    if lead_selection == "Top 5 per Cluster":
        top_leads = df.groupby('cluster').apply(lambda x: x.nlargest(5, 'lead_score')).reset_index(drop=True)
    else:
        threshold = np.percentile(df['lead_score_normalized'], 90)
        top_leads = df[df['lead_score_normalized'] >= threshold]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        top_leads.to_excel(writer, index=False)
    buffer.seek(0)
    st.download_button("Download Leads", buffer, file_name="leads_scored.xlsx")

    st.subheader("Cluster Insights")
    for cid, profile in cluster_behavior_profiles.items():
        cluster_data = df[df['cluster'] == cid]
        avg_score = cluster_data['lead_score_normalized'].mean()
        st.markdown(f"**Cluster {cid}: {profile}**")
        st.markdown(f"- Average Score: {avg_score:.2f}")
        st.markdown(f"- Top Suggested Action: {cluster_data['content_recommendation'].mode()[0]}")

    st.subheader("Opportunity Detection")
    activity_cols = [col for col in df.columns if 'daysSince' in col]
    if activity_cols:
        for col in activity_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'daysSinceLastWebActivity' in df.columns:
            dormant = df[(df['lead_score_normalized'] > 0.7) & (df['daysSinceLastWebActivity'] > 30)]
            st.markdown(f"**Dormant but High Potential Leads: {len(dormant)}**")
            st.dataframe(dormant[['lead_score_normalized', 'cluster_profile', 'content_recommendation']])
else:
    st.info("Please upload an Excel file to proceed.")
