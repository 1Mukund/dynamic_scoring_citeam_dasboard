# asbl_dashboard/app.py

import streamlit as st
import pandas as pd

# --- Page Config --- #
st.set_page_config(page_title="ASBL Dynamic Lead Dashboard", layout="wide")

st.title("\U0001F3E1 ASBL Dynamic Lead Scoring Dashboard")

# --- Upload Data --- #
st.sidebar.header("Upload Section")
uploaded_file = st.sidebar.file_uploader("Upload Leads Data (Excel)", type=["xlsx"])

# --- Define Features --- #
features = [
    'CumulativeTime', 'Number of Pages Visited', 'Unique Visits',
    'HighValuePageViews', 'DownloadedFilesCount',
    'WhatsApp Inbound', 'WhatsApp Outbound',
    'Days Since Last Visit', 'Days Since Last Inbound', 'Days Since Last Outbound'
]

# --- Archetype Mapping Logic --- #
def map_archetype(row):
    if row.get('HighValuePageViews', 0) >= 3 and row.get('CumulativeTime', 0) > 5:
        return 'Ruler', "Visited premium pages + spent >5 minutes (luxury/status-driven)."
    elif row.get('DownloadedFilesCount', 0) >= 1:
        return 'Creator', "Downloaded layouts/floor plans (creative customization interest)."
    elif row.get('Unique Visits', 0) >= 3 and row.get('Number of Pages Visited', 0) >= 5:
        return 'Explorer', "Visited 5+ pages across 3+ sessions (discovery behavior)."
    elif row.get('WhatsApp Inbound', 0) >= 1 and row.get('Number of Pages Visited', 0) < 3:
        return 'Caregiver', "WhatsApp replies but low browsing (emotional/family-driven)."
    else:
        return 'Everyman', "Focused on pricing/offers, practical buying behavior."

# --- Bucket Assignment Logic --- #
def assign_bucket(row):
    prob = row.get('Boosted Conversion %', 0)
    inbound = row.get('WhatsApp Inbound', 0)
    if prob >= 80:
        return "Hot"
    elif 50 <= prob < 80:
        return "Engaged" if inbound > 0 else "Warm"
    elif 20 <= prob < 50:
        return "Curious"
    elif prob < 20:
        return "Cold"
    return "Unknown"

# --- Simulated Churn Risk Logic --- #
def churn_risk(days):
    if pd.isna(days):
        return "Unknown"
    if days <= 30:
        return "Low"
    elif 30 < days <= 60:
        return "Medium"
    else:
        return "High"

# --- Boosted Conversion Logic Simulation --- #
def boosted_conversion_logic(row):
    reasons = []
    if row.get('Unique Visits', 0) >= 3:
        reasons.append("Multiple visits (+18%)")
    if row.get('HighValuePageViews', 0) >= 2:
        reasons.append("Viewed key pages (+15%)")
    if row.get('DownloadedFilesCount', 0) >= 1:
        reasons.append("Downloaded brochure (+8%)")
    if row.get('WhatsApp Inbound', 0) >= 1:
        reasons.append("WhatsApp reply (+22%)")
    if row.get('CumulativeTime', 0) >= 5:
        reasons.append("Spent >5 mins (+10%)")
    if not reasons:
        reasons.append("Low engagement (-15%)")
    return "; ".join(reasons)

# --- Simulated Boosted Conversion % --- #
def simulate_boosted_conversion(row):
    score = 10
    if row.get('Unique Visits', 0) >= 3:
        score += 18
    if row.get('HighValuePageViews', 0) >= 2:
        score += 15
    if row.get('DownloadedFilesCount', 0) >= 1:
        score += 8
    if row.get('WhatsApp Inbound', 0) >= 1:
        score += 22
    if row.get('CumulativeTime', 0) >= 5:
        score += 10
    return min(score, 100)

# --- Process Leads --- #
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0

    st.write("### Sample Leads Data", df.head())

    df[['Archetype', 'Archetype Logic']] = df.apply(lambda x: pd.Series(map_archetype(x)), axis=1)
    df['Boosted Conversion %'] = df.apply(simulate_boosted_conversion, axis=1)
    df['Lead Bucket'] = df.apply(assign_bucket, axis=1)
    df['Churn Risk'] = df['Days Since Last Visit'].apply(churn_risk)
    df['Boosted Conversion Logic'] = df.apply(boosted_conversion_logic, axis=1)

    st.success("\U0001F389 Predictions & Intelligence Generated!")

    # Show Filter Options
    st.sidebar.header("Filters")
    bucket_filter = st.sidebar.multiselect("Select Lead Bucket", options=df['Lead Bucket'].unique())
    archetype_filter = st.sidebar.multiselect("Select Archetype", options=df['Archetype'].unique())

    filtered_df = df.copy()
    if bucket_filter:
        filtered_df = filtered_df[filtered_df['Lead Bucket'].isin(bucket_filter)]
    if archetype_filter:
        filtered_df = filtered_df[filtered_df['Archetype'].isin(archetype_filter)]

    # Display Leads
    st.write("### Processed Leads with Full ML Insights")
    st.dataframe(filtered_df[[
        'LeadId', 'Lead Bucket', 'Boosted Conversion %', 'Boosted Conversion Logic',
        'Archetype', 'Archetype Logic', 'Churn Risk'
    ]])

    # Download Option
    st.download_button("Download Processed Leads", data=filtered_df.to_csv(index=False), file_name="processed_leads.csv", mime="text/csv")
else:
    st.warning("Please upload a leads Excel file to proceed.")
