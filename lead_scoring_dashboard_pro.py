import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Advanced Dynamic Lead Scoring", layout="wide")
st.title("Advanced Dynamic Lead Scoring & Engagement System")

# Sidebar controls
st.sidebar.header("Settings")
lead_selection = st.sidebar.selectbox("Lead Selection Strategy", ["Top 5 per Cluster", "Top 10% Overall"])
content_strategy = st.sidebar.selectbox("Content Sensitivity", ["Aggressive", "Nurture"])
download_option = st.sidebar.selectbox("Download File Type", ["Full Export", "Top Leads Export"])

# Initialize checklist states
checklist = {k: False for k in [
    "Upload Engine", "Smart Auto-Cleaning", "Feature Extraction", "Feature Standardization",
    "Dynamic Scoring Engine", "Dynamic Clustering",
    "Cluster Analysis & Labeling", "Dynamic Content Recommendation", "Visualizations",
    "Download Results", "Executive Summary"
]}

st.header("Checklist Progress")

uploaded_file = st.file_uploader("Step 1: Upload your Excel file", type=["xlsx"])

if uploaded_file:
    checklist["Upload Engine"] = True

    # Read and auto-clean data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)
    checklist["Smart Auto-Cleaning"] = True

    # Intelligent feature analysis
    high_cardinality_cols = [col for col in df.columns if df[col].nunique() > 0.9 * len(df)]
    date_like_cols = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
    categorical_cols = df.select_dtypes(include=['object']).columns.difference(high_cardinality_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(high_cardinality_cols)

    for col in categorical_cols:
        if df[col].nunique() < len(df) * 0.3:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            numeric_cols = numeric_cols.append(pd.Index([col]))

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    selected_features = [col for col in numeric_cols if col not in to_drop]

    X = df[selected_features]
    checklist["Feature Extraction"] = True

    # Dynamic Scoring
    feature_variances = X.var()
    feature_weights = feature_variances / feature_variances.sum()
    df['lead_score'] = X.mul(feature_weights, axis=1).sum(axis=1)
    df['lead_score_normalized'] = (df['lead_score'] - df['lead_score'].min()) / (df['lead_score'].max() - df['lead_score'].min())
    checklist["Dynamic Scoring Engine"] = True

    # Dynamic Clustering
    st.subheader("Optimal Cluster (K) Selection")
    silhouette_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)

    best_k = K_range[np.argmax(silhouette_scores)]

    fig_k, ax_k = plt.subplots()
    ax_k.plot(K_range, silhouette_scores, marker='o')
    ax_k.set_xlabel('Number of Clusters K')
    ax_k.set_ylabel('Silhouette Score')
    ax_k.set_title('Optimal K using Silhouette Score')
    st.pyplot(fig_k)

    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = final_kmeans.fit_predict(X)
    checklist["Dynamic Clustering"] = True

    # Cluster Profiling and Dynamic Content Suggestion
    st.subheader("Cluster Profiles")
    cluster_profiles = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_center = cluster_data[selected_features].mean()
        top_features = cluster_center.sort_values(ascending=False).head(2).index.tolist()
        profile_name = f"High {top_features[0]} & High {top_features[1]}"
        cluster_profiles[cluster_id] = profile_name

    df['cluster_profile'] = df['cluster'].map(cluster_profiles)

    def generate_content(row):
        if content_strategy == "Aggressive":
            if 'Whatsapp' in row['cluster_profile']:
                return "Send Immediate WhatsApp Promotions"
            elif 'Page' in row['cluster_profile'] or 'Visit' in row['cluster_profile']:
                return "Push Webinar Invites or Site Visit Invitations"
            else:
                return "Send Direct Demo Invitations"
        else:
            if 'Download' in row['cluster_profile'] or 'HighValuePageViews' in row['cluster_profile']:
                return "Nurture with Ebooks or Case Studies"
            else:
                return "Send Educational Newsletters and Soft Nurturing Content"

    df['content_recommendation'] = df.apply(generate_content, axis=1)
    checklist["Cluster Analysis & Labeling"] = True
    checklist["Dynamic Content Recommendation"] = True

    # Visualizations and Executive Dashboard
    st.subheader("Lead Score Distribution")
    fig_score, ax_score = plt.subplots()
    sns.histplot(df['lead_score_normalized'], bins=30, kde=True, ax=ax_score)
    ax_score.set_title('Lead Score Normalized Distribution')
    st.pyplot(fig_score)

    st.subheader("Cluster Counts")
    fig_cluster, ax_cluster = plt.subplots()
    sns.countplot(x='cluster_profile', data=df, order=df['cluster_profile'].value_counts().index, ax=ax_cluster)
    ax_cluster.set_title('Cluster Profile Distribution')
    ax_cluster.set_xticklabels(ax_cluster.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig_cluster)

    st.subheader("Cluster Behavior Heatmaps")
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id][selected_features]
        fig_heat, ax_heat = plt.subplots(figsize=(10, 1))
        sns.heatmap(cluster_data.mean().to_frame().T, cmap="coolwarm", annot=True, fmt=".2f", cbar=False, ax=ax_heat)
        ax_heat.set_title(f'Cluster {cluster_id}: {cluster_profiles[cluster_id]}')
        st.pyplot(fig_heat)

    # Smart Download & Top Leads
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

    st.download_button("Download Scored Leads Excel", buffer, file_name="scored_leads.xlsx")
    checklist["Download Results"] = True

    # Auto Cluster Insights
    st.subheader("Auto-Generated Cluster Insights")
    for cluster_id, profile in cluster_profiles.items():
        st.markdown(f"**Cluster {cluster_id} ({profile}) Insights:**")
        size = len(df[df['cluster'] == cluster_id])
        avg_score = df[df['cluster'] == cluster_id]['lead_score_normalized'].mean()
        st.markdown(f"- Leads: {size}")
        st.markdown(f"- Average Score: {avg_score:.2f}")
        st.markdown(f"- Top Action: {df[df['cluster'] == cluster_id]['content_recommendation'].mode()[0]}")

else:
    st.info("Please upload an Excel file to proceed.")
