import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Advanced Dynamic Lead Scoring", layout="wide")
st.title("Advanced Dynamic Lead Scoring & Engagement System")

# Initialize checklist states
checklist = {
    "Upload Engine": False,
    "Smart Auto-Cleaning": False,
    "Feature Extraction": False,
    "Feature Standardization": False,
    "PCA Feature Compression": False,
    "Dynamic Scoring Engine": False,
    "Dynamic Clustering": False,
    "Cluster Analysis & Labeling": False,
    "Dynamic Content Recommendation": False,
    "Visualizations": False,
    "Download Results": False
}

st.header("Checklist Progress")

uploaded_file = st.file_uploader("Step 1: Upload your Excel file", type=["xlsx"])

if uploaded_file:
    checklist["Upload Engine"] = True

    # Read and auto-clean data
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    df.fillna(0, inplace=True)
    checklist["Smart Auto-Cleaning"] = True

    # Feature selection: pick only numeric columns automatically
    feature_cols = [col for col in df.columns if col not in ['LeadId'] and np.issubdtype(df[col].dtype, np.number)]
    X = df[feature_cols]
    checklist["Feature Extraction"] = True

    # Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    checklist["Feature Standardization"] = True

    # PCA for feature compression
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    checklist["PCA Feature Compression"] = True

    # Scree Plot
    st.subheader("PCA Variance Explained")
    fig_scree, ax_scree = plt.subplots()
    ax_scree.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    ax_scree.set_xlabel('Number of Components')
    ax_scree.set_ylabel('Cumulative Explained Variance')
    ax_scree.set_title('PCA Scree Plot')
    st.pyplot(fig_scree)

    # Dynamic Scoring using PC1
    df['lead_score'] = X_pca[:, 0]
    df['score_percentile'] = pd.qcut(df['lead_score'], 100, labels=False)
    checklist["Dynamic Scoring Engine"] = True

    # Dynamic Clustering (Auto K detection)
    best_k = 2
    best_score = -1
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, cluster_labels)
        if score > best_score:
            best_score = score
            best_k = k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_pca)
    checklist["Dynamic Clustering"] = True

    # Cluster Analysis & Labeling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cluster_summary = df.groupby('cluster')[numeric_cols].mean()
    checklist["Cluster Analysis & Labeling"] = True

    # Smart Cluster Naming
    cluster_profiles = {}
    for cluster_id in cluster_summary.index:
        top_features = cluster_summary.loc[cluster_id].sort_values(ascending=False).head(2).index.tolist()
        cluster_profiles[cluster_id] = f"High {top_features[0]} & {top_features[1]}"

    df['cluster_profile'] = df['cluster'].map(cluster_profiles)

    # Dynamic Content Recommendation
    def recommend_content(cluster):
        if cluster_summary.loc[cluster, 'lead_score'] > df['lead_score'].mean():
            return "Send demo invitation"
        else:
            return "Send case study or nurture content"

    df['content_recommendation'] = df['cluster'].apply(recommend_content)
    checklist["Dynamic Content Recommendation"] = True

    # Visualizations
    st.subheader("Lead Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['lead_score'], bins=20, ax=ax)
    st.pyplot(fig)

    st.subheader("Cluster Count")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='cluster_profile', data=df, order=df['cluster_profile'].value_counts().index, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
    checklist["Visualizations"] = True

    # Download Results
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)

    st.download_button("Download Leads with Scores, Clusters & Profiles", buffer, "scored_leads_with_profiles.xlsx")
    checklist["Download Results"] = True

    # Display final checklist
    st.header("✅ Final Progress Tracker")
    for task, status in checklist.items():
        st.write(f"{'✅' if status else '❌'} {task}")

else:
    st.info("Please upload an Excel file to proceed.")
