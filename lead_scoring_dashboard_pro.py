import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Super Intelligent Lead Scoring", layout="wide")
st.title("Super Intelligent Dynamic Lead Scoring & Opportunity Detection")

# Sidebar controls
st.sidebar.header("Settings")
lead_selection = st.sidebar.selectbox("Lead Selection Strategy", ["Top 5 per Cluster", "Top 10% Overall"])
content_strategy = st.sidebar.selectbox("Content Sensitivity", ["Aggressive", "Nurture"])
download_option = st.sidebar.selectbox("Download File Type", ["Full Export", "Top Leads Export"])

uploaded_file = st.file_uploader("Step 1: Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)

    high_card_cols = [col for col in df.columns if df[col].nunique() > 0.9 * len(df)]
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
    categorical_cols = df.select_dtypes(include=['object']).columns.difference(high_card_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(high_card_cols)

    for col in categorical_cols:
        if df[col].nunique() < 0.3 * len(df):
            df[col] = LabelEncoder().fit_transform(df[col])
            numeric_cols = numeric_cols.append(pd.Index([col]))

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    selected_features = [col for col in numeric_cols if col not in to_drop]

    X = df[selected_features]

    # Scoring
    variances = X.var()
    feature_weights = variances / variances.sum()
    df['lead_score'] = X.mul(feature_weights, axis=1).sum(axis=1)
    df['lead_score_normalized'] = (df['lead_score'] - df['lead_score'].min()) / (df['lead_score'].max() - df['lead_score'].min())

    # Clustering
    st.subheader("Optimal Cluster Selection")
    silhouette_scores = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    best_k = range(2, 8)[np.argmax(silhouette_scores)]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    fig, ax = plt.subplots()
    ax.plot(range(2, 8), silhouette_scores, marker='o')
    ax.set_title('Silhouette Score vs K')
    st.pyplot(fig)

    # Intelligent Cluster Behavior Analysis
    cluster_profiles = {}
    cluster_summaries = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        feature_means = cluster_data[selected_features].mean()

        profile = []
        if feature_means.mean() > 0:
            top_features = feature_means.sort_values(ascending=False).head(3).index.tolist()
            profile = [f"High {feat}" for feat in top_features]
        else:
            profile = ["Dormant Behavior"]

        cluster_profiles[cluster_id] = ' & '.join(profile)
        cluster_summaries[cluster_id] = feature_means.to_dict()

    df['cluster_profile'] = df['cluster'].map(cluster_profiles)

    def dynamic_content_strategy(row):
        text = row['cluster_profile']
        if 'Whatsapp' in text or 'Inbound' in text:
            return "Push WhatsApp & Immediate Followup"
        elif 'Page' in text or 'Visit' in text:
            return "Invite for Virtual Tour / Webinar"
        elif 'Download' in text or 'HighValuePageViews' in text:
            return "Send Deep Dive Ebooks / Pricing Kits"
        else:
            return "Soft Nurturing and Education"

    df['content_recommendation'] = df.apply(dynamic_content_strategy, axis=1)

    # Visuals
    st.subheader("Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['lead_score_normalized'], bins=20, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Cluster Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='cluster_profile', data=df, order=df['cluster_profile'].value_counts().index, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Smart Download
    st.subheader("Download Scored Leads")
    if lead_selection == "Top 5 per Cluster":
        top_leads = df.groupby('cluster').apply(lambda x: x.nlargest(5, 'lead_score')).reset_index(drop=True)
    else:
        threshold = np.percentile(df['lead_score_normalized'], 90)
        top_leads = df[df['lead_score_normalized'] >= threshold]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        if download_option == "Top Leads Export":
            top_leads.to_excel(writer, index=False)
        else:
            df.to_excel(writer, index=False)
    buffer.seek(0)
    st.download_button("Download Leads Excel", buffer, file_name="scored_leads.xlsx")

    # Intelligent Opportunity Detection
    st.subheader("Opportunity Detection")
    if 'daysSinceLastWebActivity' in df.columns:
        high_score_low_activity = df[(df['lead_score_normalized'] > 0.75) & (df['daysSinceLastWebActivity'] > 30)]
        st.markdown(f"**High Potential but Dormant Leads:** {len(high_score_low_activity)}")
        st.dataframe(high_score_low_activity[['lead_score_normalized', 'daysSinceLastWebActivity', 'cluster_profile', 'content_recommendation']])
    else:
        st.info("No recent activity fields found to run opportunity detection.")

    st.subheader("Auto Cluster Insights")
    for cluster_id, profile in cluster_profiles.items():
        st.markdown(f"**Cluster {cluster_id} ({profile}) Insights:**")
        st.markdown(f"- Average Lead Score: {df[df['cluster']==cluster_id]['lead_score_normalized'].mean():.2f}")
        st.markdown(f"- Top Content Recommendation: {df[df['cluster']==cluster_id]['content_recommendation'].mode()[0]}")
else:
    st.info("Please upload an Excel file to proceed.")
