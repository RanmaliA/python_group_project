import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ML & Pipeline Architecture matching internal core methodologies
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="MedCore Operations Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load data
df = pd.read_csv("department_analysis.csv")

risk_model = joblib.load("models/risk_model.pkl")
risk_columns = joblib.load("models/risk_columns.pkl")

cluster_model = joblib.load("models/cluster_model.pkl")
scaler = joblib.load("models/scaler.pkl")
cluster_columns = joblib.load("models/cluster_columns.pkl")


# Generate clusters
cluster_X = df[cluster_columns]
cluster_scaled = scaler.transform(cluster_X)

df["Cluster"] = cluster_model.predict(cluster_scaled)


cluster_labels = {
    0: "High Volume / Stable Operations",
    1: "Resource Constrained Departments",
    2: "Efficient Low Burden Departments"
}

df["Cluster Name"] = df["Cluster"].map(cluster_labels)


# Page title
st.title("MedCore AI Operations Dashboard")


# Sidebar
st.sidebar.title("MedCore Systems")

page = st.sidebar.radio(
    "test",
    [
        "Executive Overview",
        "Operational Error Drivers",
        "Actionable Recommendations"
    ]
)

# page one
if page == "Executive Overview":

    st.header("Executive Summary")

    # main metrics
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Departments",
        len(df)
    )

    lowest_risk = df["Predicted Risk"].min()
    lowest_risk_depts = ", ".join(df.loc[df["Predicted Risk"] == lowest_risk, "department_name"])


    highest_risk = df["Predicted Risk"].max()
    highest_risk_depts = ", ".join(df.loc[df["Predicted Risk"] == highest_risk, "department_name"])

    c2.metric(
        "Lowest Risk",
        lowest_risk_depts,
        round(lowest_risk, 2),
        delta_color="inverse",
        delta_arrow="off"
    )

    c3.metric(
        "Highest Risk",
        highest_risk_depts,
        round(highest_risk, 2),
        delta_color="inverse",
        delta_arrow="off"
    )

    # main overview chart

    fig = px.bar(
        df.sort_values("Predicted Risk", ascending=False),
        x="department_name",
        y="Predicted Risk",
        title="Predicted Operational Risk by Department",
        labels={"department_name": "Department",
                "Predicted Risk": "Operational Risk Score"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # main overview table

    display_df = (
    df[
        [
            "department_name",
            "Predicted Risk",
            "Cluster Name"
        ]
    ]
    .sort_values(
        "Predicted Risk",
        ascending=False
    )
    .rename(
        columns={
            "department_name": "Department",
            "Predicted Risk": "Operational Risk Score",
            "Cluster Name": "Operational Profile"
        }
    )
    )

    display_df["Operational Risk Score"] = display_df["Operational Risk Score"].round(2)

    st.dataframe(display_df)

    # interactive risk calculator
    st.subheader("Risk Calculator")

    dept = st.selectbox(
    "Choose Department",
    df["department_name"]
)

    row = df[df["department_name"] == dept].iloc[0]

    manual_multiplier = st.slider(
        "Manual Workload Multiplier",
        0.5,
        3.0,
        float(row.manual_workload_multiplier),
        0.1
    )

    denial_risk = st.slider(
    "Claims Processing Error Risk",
    0.0,
    1.0,
    float(row.claim_denial_risk),
    0.05
    )

    # create modified input using selected department values
    updated_features = row[
        [
            "annual_budget",
            "base_wait_days",
            "manual_workload_multiplier",
            "claim_denial_risk",
            "lab_delay_risk",
            "provider_count",
            "location"
        ]
    ].to_frame().T

    # replace workload with slider value
    updated_features["manual_workload_multiplier"] = manual_multiplier
    updated_features["claim_denial_risk"] = denial_risk

    # encode location exactly like your original model input
    updated_features = pd.get_dummies(
        updated_features,
        columns=["location"],
        drop_first=True
    )

    # ensure same columns as training
    updated_features = updated_features.reindex(
        columns=risk_columns,
        fill_value=0
    )

    # predict new risk
    updated_risk = risk_model.predict(updated_features)[0]

    m1, m2 = st. columns(2)
    m1.metric(
        "Predicted Risk",
        round(updated_risk, 2)
    )

    m2.metric(
        "Cluster",
        row["Cluster Name"]
    )

# page 2
if page == "Operational Error Drivers":

    st.header("Department Explorer")

    dept = st.selectbox(
        "Choose Department",
        df["department_name"]
    )
    row = df[df["department_name"] == dept].iloc[0]
    st.metric(
        "Predicted Risk",
        round(row["Predicted Risk"],2)
    )

    st.metric(
        "Cluster",
        int(row["Cluster"])
    )
    st.subheader("Operational Characteristics")

    st.write(f"Provider Count: {row.provider_count}")

    st.write(f"Average Wait Days: {row.avg_wait_days}")

    st.write(f"Manual Minutes: {round(row.total_manual_minutes)}")

    st.write(f"Error Rate: {round(row.error_rate,3)}")

    st.subheader("AI Recommendation")
    cluster = row["Cluster"]
    if cluster == 0:

        st.success("""
    ### Workload Heavy Department

    Recommended AI tools

    - Workflow automation

    - Smart documentation

    - Automated claims processing
    """)
    elif cluster == 1:

        st.warning("""
    ### Scheduling Bottleneck

    Recommended AI

    - Appointment optimization

    - Waitlist automation

    - No-show prediction
    """)
    else:

        st.info("""
    ### High Performing

    Recommendation

    Continue monitoring

    Use as benchmark
    """)

# page 3
if page == "Actionable Recommendations":

    st.header("Department Segments")
    # placeholder page

