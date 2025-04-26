import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="AI-Driven Lead Intelligence", layout="wide")
st.title("AI-Driven Dynamic Lead Scoring & Cluster Intelligence")

st.sidebar.header("Settings")
lead_selection = st.sidebar.selectbox("Lead Selection", ["Top 5 per Cluster", "Top 10% Overall"])
content_strategy = st.sidebar.selectbox("Content Sensitivity", ["Aggressive", "Nurture"])

def interpret_features(df):
    feature_types = {}
    for col in df.columns:
        if df[col].nunique() > 0.95 * len(df):
            feature_types[col] = 'id'
        elif df[col].dtype == 'object' and df[col].nunique() < 0.3 * len(df):
            feature_types[col] = 'categorical'
        elif 'date' in col.lower() or 'timestamp' in col.lower():
            feature_types[col] = 'date'
        elif df[col].dtype in [np.float64, np.int64]:
            feature_types[col] = 'numeric'
        else:
            feature_types[col] = 'unknown'
    return feature_types

def classify_behavior(cluster_data, selected_features):
    feature_means = cluster_data[selected_features].mean()
    if feature_means.mean() == 0:
        return "Dormant Leads"
    top_feats = feature_means.sort_values(ascending=False).head(2).index.tolist()
    return " & ".join([f"High {feat}" for feat in top_feats])

def recommend_action(profile):
    if 'Whatsapp' in profile or 'Inbound' in profile:
        return "Push Conversational Followups"
    elif 'Page' in profile or 'Visit' in profile:
        return "Promote Site Visits or Webinars"
    elif 'Download' in profile or 'HighValuePageViews' in profile:
        return "Offer Knowledge Packs / Pricing Info"
    else:
        return "Run Education Campaigns"

uploaded_file = st.file_uploader("Upload your Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)

    feature_types = interpret_features(df)

    behavior_features = [col for col, typ in feature_types.items() if typ == 'numeric']
    if not behavior_features:
        st.error("No meaningful numeric behavioral features detected.")
        st.stop()

    X = df[behavior_features]

    # Dynamic Scoring
    feature_variances = X.var()
    feature_weights = feature_variances / feature_variances.sum()
    df['lead_score'] = X.mul(feature_weights, axis=1).sum(axis=1)
    df['lead_score_normalized'] = (df['lead_score'] - df['lead_score'].min()) / (df['lead_score'].max() - df['lead_score'].min())

    # Clustering
    silhouette_scores = []
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    best_k = np.argmax(silhouette_scores) + 2
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = final_model.fit_predict(X)

    # Cluster Profiles
    cluster_profiles = {}
    for cid in sorted(df['cluster'].unique()):
        cluster_profiles[cid] = classify_behavior(df[df['cluster'] == cid], behavior_features)

    df['cluster_profile'] = df['cluster'].map(cluster_profiles)
    df['content_recommendation'] = df['cluster_profile'].apply(recommend_action)

    # Visual Insights
    st.subheader("Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['lead_score_normalized'], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Cluster Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='cluster_profile', data=df, order=df['cluster_profile'].value_counts().index, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

    # Opportunity Detection
    st.subheader("Opportunity Detection")
    if any('daysSince' in col for col in df.columns):
        days_cols = [col for col in df.columns if 'daysSince' in col]
        for col in days_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if 'daysSinceLastWebActivity' in df.columns:
            dormant = df[(df['lead_score_normalized'] > 0.7) & (df['daysSinceLastWebActivity'] > 30)]
            st.markdown(f"**Dormant but High Potential Leads Detected:** {len(dormant)}")
            st.dataframe(dormant[['lead_score_normalized', 'cluster_profile', 'content_recommendation']])

    # Download Option
    st.subheader("Download Scored Leads")
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

    # Cluster Insights
    st.subheader("Auto Insights per Cluster")
    for cid, profile in cluster_profiles.items():
        avg_score = df[df['cluster'] == cid]['lead_score_normalized'].mean()
        st.markdown(f"**Cluster {cid} ({profile})**")
        st.markdown(f"- Average Score: {avg_score:.2f}")
        st.markdown(f"- Suggested Action: {df[df['cluster']==cid]['content_recommendation'].mode()[0]}")
else:
    st.info("Please upload an Excel file to proceed.")
