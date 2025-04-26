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

# Sidebar controls
st.sidebar.header("Settings")
lead_selection = st.sidebar.selectbox("Lead Selection Strategy", ["Top 5 per Cluster", "Top 10% Overall"])
content_strategy = st.sidebar.selectbox("Content Sensitivity", ["Aggressive", "Nurture"])
download_option = st.sidebar.selectbox("Download File Type", ["Full Export", "Top Leads Export"])

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
    "Download Results": False,
    "Executive Summary": False
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

    # Correct Feature Selection based on meaningful behavioral features
    selected_features = [
        'CumulativeTime', 'Number_of_Page_Visited', 'Unqiue_Visits',
        'WhatsappInbound', 'WhatsappOutbound',
        'daysSinceLastWebActivity', 'daysSinceLastInbound', 'daysSinceLastOutbound',
        'HighValuePageViews', 'DownloadedFilesCount'
    ]

    available_features = [col for col in selected_features if col in df.columns]
    X = df[available_features]
    checklist["Feature Extraction"] = True

    # Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    checklist["Feature Standardization"] = True

    # PCA for feature compression
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    checklist["PCA Feature Compression"] = True

    # Show Scoring Logic (Feature Contributions)
    st.subheader("Scoring Logic Transparency")
    feature_importance = pd.Series(pca.components_[0], index=available_features)
    feature_importance = feature_importance.abs().sort_values(ascending=False)
    st.markdown("**Lead Score = Weighted combination of below features:**")
    st.dataframe(feature_importance.reset_index().rename(columns={0: 'Contribution', 'index': 'Feature'}))

    fig_logic, ax_logic = plt.subplots()
    feature_importance.plot(kind='bar', ax=ax_logic)
    ax_logic.set_ylabel('Absolute Contribution Weight')
    ax_logic.set_title('Feature Contributions to Lead Score (PC1)')
    st.pyplot(fig_logic)

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

    # Dynamic Clustering (Auto K detection with visualization)
    st.subheader("Optimal Cluster (K) Selection")
    silhouette_scores = []
    K_range = range(2, 8)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        silhouette_scores.append(silhouette_score(X_pca, cluster_labels))
    best_k = K_range[np.argmax(silhouette_scores)]

    fig_k, ax_k = plt.subplots()
    ax_k.plot(K_range, silhouette_scores, marker='o')
    ax_k.set_xlabel('Number of Clusters K')
    ax_k.set_ylabel('Silhouette Score')
    ax_k.set_title('Optimal K using Silhouette Score')
    st.pyplot(fig_k)

    # Final Clustering
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
    def dynamic_message(row):
        if content_strategy == "Aggressive":
            if row['lead_score'] > df['lead_score'].mean():
                return "Act now! Schedule a personalized demo."
            else:
                return "Limited time offer: Learn more today."
        else:
            if row['lead_score'] > df['lead_score'].mean():
                return "We'd love to show you more when you're ready."
            else:
                return "Explore helpful resources at your own pace."

    df['content_recommendation'] = df.apply(dynamic_message, axis=1)
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

    # Executive Summary
    st.header("Executive Summary")
    st.markdown(f"**Total Leads Processed:** {len(df)}")
    st.markdown(f"**Optimal Clusters Found:** {best_k}")
    st.markdown(f"**PCA Components Used:** {X_pca.shape[1]}")
    st.markdown(f"**Average Lead Score:** {df['lead_score'].mean():.2f}")
    checklist["Executive Summary"] = True

    # Cluster Mini Summary Cards
    st.header("Cluster Profiles Summary")
    for cluster_id, profile in cluster_profiles.items():
        cluster_data = df[df['cluster'] == cluster_id]
        avg_score = cluster_data['lead_score'].mean()
        st.subheader(f"Cluster {cluster_id}: {profile}")
        st.markdown(f"- **Number of Leads:** {len(cluster_data)}")
        st.markdown(f"- **Average Lead Score:** {avg_score:.2f}")
        st.markdown(f"- **Top Features:** {profile}")
        if content_strategy == "Aggressive":
            st.markdown(f"- **Recommended Action:** Prioritize aggressive outreach and demos.")
        else:
            st.markdown(f"- **Recommended Action:** Nurture through educational and value-driven content.")

    # Lead Selection Logic
    if lead_selection == "Top 5 per Cluster":
        top_leads = df.groupby('cluster').apply(lambda x: x.nlargest(5, 'lead_score')).reset_index(drop=True)
    else:
        threshold = np.percentile(df['lead_score'], 90)
        top_leads = df[df['lead_score'] >= threshold]

    # Download Results
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        if download_option == "Top Leads Export":
            top_leads.to_excel(writer, index=False)
        else:
            df.to_excel(writer, index=False)
    buffer.seek(0)

    st.download_button("Download Selected Leads", buffer, "dynamic_lead_scoring_results.xlsx")
    checklist["Download Results"] = True

    # Display final checklist
    st.header("✅ Final Progress Tracker")
    for task, status in checklist.items():
        st.write(f"{'✅' if status else '❌'} {task}")

else:
    st.info("Please upload an Excel file to proceed.")
