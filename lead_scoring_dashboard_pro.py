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
    cluster_summary = df.groupby('cluster').mean()
    checklist["Cluster Analysis & Labeling"] = True

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
    sns.countplot(x='cluster', data=df, ax=ax2)
    st.pyplot(fig2)
    checklist["Visualizations"] = True

    # Download Results
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)

    st.download_button("Download Leads with Scores & Clusters", buffer, "scored_leads_with_clusters.xlsx")
    checklist["Download Results"] = True

    # Display final checklist
    st.header("✅ Final Progress Tracker")
    for task, status in checklist.items():
        st.write(f"{'✅' if status else '❌'} {task}")

else:
    st.info("Please upload an Excel file to proceed.")
