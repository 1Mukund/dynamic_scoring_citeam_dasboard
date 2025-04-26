# Step 1: Importing Libraries and Explaining Why

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1 Complete!

# --------------------------------------------------

# Step 2: Setting Streamlit Page Configuration

st.set_page_config(page_title="Intelligent Lead Scoring Dashboard", layout="wide")

st.title("📊 Intelligent Lead Scoring & Opportunity Detection Dashboard")

# Step 2 Complete!

# --------------------------------------------------

# Step 3: Creating File Upload UI

uploaded_file = st.file_uploader("📂 Upload your Lead Excel File", type=["xlsx"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully! Now processing...")

    # Step 4: Data Cleaning and Feature Preprocessing

    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.lower()

    days_since_cols = ['dayssincelastwebactivity', 'dayssincelastinbound', 'dayssincelastoutbound']
    for col in days_since_cols:
        if col in df.columns:
            df[col] = df[col].fillna(999)

    if 'lastvisittimestamp' in df.columns:
        df['lastvisittimestamp'] = pd.to_datetime(df['lastvisittimestamp'], errors='coerce')
        today = pd.Timestamp.today()
        df['dayssincelastwebactivity'] = (today - df['lastvisittimestamp']).dt.days.fillna(999)

    if 'inbound_message_time' in df.columns:
        df['inbound_message_time'] = pd.to_datetime(df['inbound_message_time'], errors='coerce')
        df['dayssincelastinbound'] = (today - df['inbound_message_time']).dt.days.fillna(999)

    if 'outbound_message_time' in df.columns:
        df['outbound_message_time'] = pd.to_datetime(df['outbound_message_time'], errors='coerce')
        df['dayssincelastoutbound'] = (today - df['outbound_message_time']).dt.days.fillna(999)

    st.success("✅ Data cleaning and feature preprocessing completed!")

    # --------------------------------------------------

    # Step 5: Defining Behavior Grouping

    st.subheader("🧠 Step 5: Behavior Feature Grouping")

    web_engagement_features = ['cumulativetime', 'number_of_page_visited', 'unqiue_visits', 'highvaluepageviews']
    intent_action_features = ['downloadedfilescount', 'whatsappinbound']
    communication_features = ['whatsappoutbound']
    recency_features = ['dayssincelastwebactivity', 'dayssincelastinbound']

    st.markdown("**Feature Groups:**")
    st.markdown("- **Web Engagement**: " + ", ".join(web_engagement_features))
    st.markdown("- **Intent Actions**: " + ", ".join(intent_action_features))
    st.markdown("- **Communication**: " + ", ".join(communication_features))
    st.markdown("- **Recency**: " + ", ".join(recency_features))

    st.success("✅ Behavior features grouped successfully!")

    # --------------------------------------------------

    # Step 6: Performing Correlation Analysis

    st.subheader("🔎 Step 6: Feature Correlation Analysis")

    feature_corr = df[web_engagement_features + intent_action_features + communication_features + recency_features].corr()

    st.write("✅ Correlation Matrix:")
    st.dataframe(feature_corr)

    focus_features = intent_action_features

    importance_scores = {}
    for feature in web_engagement_features + communication_features + recency_features:
        score = 0
        for intent_feat in focus_features:
            score += abs(feature_corr.get(feature, {}).get(intent_feat, 0))
        importance_scores[feature] = score / len(focus_features)

    importance_df = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['importance_score']).sort_values(by='importance_score', ascending=False)

    st.markdown("### 📋 Calculated Importance of Each Feature:")
    st.dataframe(importance_df)

    st.success("✅ Correlation Analysis Completed!")

    # --------------------------------------------------

    # Step 7: Dynamic Scoring Based on Importance Scores

    st.subheader("🔎 Step 7: Dynamic Lead Scoring Formula")

    st.markdown("**Scoring Logic:**")
    st.markdown("- Positive behaviors (like engagement, downloads) **increase score** proportional to importance.")
    st.markdown("- Higher 'days since last action' **decreases score** (recency penalty).")

    def dynamic_lead_score(row, importance_scores):
        score = 0
        for feature, weight in importance_scores.items():
            if feature in row:
                if 'dayssince' in feature:
                    score -= row[feature] * weight
                else:
                    score += row[feature] * weight
        return score

    df['lead_score'] = df.apply(lambda x: dynamic_lead_score(x, importance_scores), axis=1)

    st.success("✅ Dynamic lead scores calculated!")

    # --------------------------------------------------

    # Step 8: Bucketing Leads Based on Score Percentiles

    st.subheader("🔎 Step 8: Dynamic Bucketing Logic")

    st.markdown("**Bucketing Rules (based on score percentile):**")
    st.markdown("- Top 10% → **Hot 🔥**")
    st.markdown("- 70%–90% → **Engaged 🟡**")
    st.markdown("- 40%–70% → **Warm 🔵**")
    st.markdown("- 20%–40% → **Curious 🟠**")
    st.markdown("- Bottom 20% → **Cold ⚪**")

    df['score_percentile'] = df['lead_score'].rank(pct=True) * 100

    def assign_bucket(percentile):
        if percentile >= 90:
            return 'Hot'
        elif percentile >= 70:
            return 'Engaged'
        elif percentile >= 40:
            return 'Warm'
        elif percentile >= 20:
            return 'Curious'
        else:
            return 'Cold'

    df['lead_bucket'] = df['score_percentile'].apply(assign_bucket)

    st.success("✅ Leads bucketed dynamically!")

    # --------------------------------------------------

    # Step 9: Opportunity Detection for Special Cases

    st.subheader("🎯 Step 9: Opportunity Detection Rules")

    st.markdown("**Special Opportunity Tags:**")
    st.markdown("- 'High Web Activity, No WhatsApp' → Active on website but no inbound communication.")
    st.markdown("- 'Needs Immediate Closure' → WhatsApp inbound happened but no recent website activity.")

    def detect_opportunity(row):
        if row['lead_bucket'] in ['Engaged', 'Warm'] and row['whatsappinbound'] == 0:
            return 'High Web Activity, No WhatsApp'
        elif row['whatsappinbound'] >= 1 and row['dayssincelastwebactivity'] > 30:
            return 'Needs Immediate Closure'
        else:
            return ''

    df['opportunity_tag'] = df.apply(detect_opportunity, axis=1)

    st.success("✅ Opportunities detected!")

    # --------------------------------------------------

    # Step 10: Dashboard Visualization and Outputs

    st.subheader("📊 Step 10: Final Dashboard")

    st.markdown("### 🎯 Leads Scoring Table")
    st.dataframe(df[['leadid', 'lead_score', 'lead_bucket', 'opportunity_tag']])

    st.markdown("### 📈 Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['lead_score'], bins=20, kde=True)
    st.pyplot(fig)

    st.markdown("### 📌 Bucket Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='lead_bucket', data=df, order=['Hot', 'Engaged', 'Warm', 'Curious', 'Cold'])
    st.pyplot(fig2)

    st.markdown("### 📋 Opportunity Leads")
    st.dataframe(df[df['opportunity_tag'] != ''][['leadid', 'lead_bucket', 'opportunity_tag']])

    st.success("✅ Dashboard fully ready with all logics explained!")

else:
    st.info("ℹ️ Please upload an Excel (.xlsx) file with leads data to proceed.")

# 🧠 Now users can see every logic, feature grouping, scoring formula, bucketing rule, and opportunity detection inside the dashboard itself!
