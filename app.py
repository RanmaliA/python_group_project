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

# -----------------------------------------------------------------------------
# 1. page config & theme styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="MedCore Operations Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title
st.title("MedCore AI Operations Dashboard")

# Sidebar
st.sidebar.title("MedCore Systems")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Operational Error Drivers",
        "Operational Profiles",
        "Executive Action Plan"
    ]
)

# Set high-quality professional presentation overrides for Matplotlib/Seaborn
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fafbfc',
    'xtick.color': '#4a5568',
    'ytick.color': '#4a5568'
})

# Custom CSS styling for dashboard metrics and badges
st.markdown("""
    <style>
    .main { background-color: #f7fafc; }
    div[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 700; color: #1a365d; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; font-weight: 600; color: #4a5568; }
    h1, h2, h3 { color: #1a365d; font-family: 'Arial', sans-serif; font-weight: 700; }
    .insight-card { background-color: #ffffff; padding: 20px; border-radius: 6px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; border-top: 6px solid #2b6cb0; }
    .rec-card { background-color: #ffffff; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2b6cb0; min-height: 380px; }
    .rec-card-small { background-color: #ffffff; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2b6cb0; }
    .map-card { background-color: #ffffff; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2b6cb0; min-height: 420px; }
    .cluster-badge { padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }
    .dept-chip { background-color: #edf2f7; color: #2d3748; padding: 6px 12px; border-radius: 16px; font-size: 0.85rem; font-weight: 500; display: inline-block; margin: 4px; border: 1px solid #e2e8f0; }
    .id-badge { background-color: #4a5568; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-right: 6px; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# s. data & models setup
# -----------------------------------------------------------------------------

# load data
df = pd.read_csv("department_analysis.csv")

risk_model = joblib.load("models/risk_model.pkl")
risk_columns = joblib.load("models/risk_columns.pkl")

cluster_model = joblib.load("models/cluster_model.pkl")
scaler = joblib.load("models/scaler.pkl")
cluster_columns = joblib.load("models/cluster_columns.pkl")


# generate clusters
cluster_X = df[cluster_columns]
cluster_scaled = scaler.transform(cluster_X)

df["Cluster"] = cluster_model.predict(cluster_scaled)


cluster_labels = {
    0: "Standard Operations",
    1: "High Processing Friction",
    2: "High Patient Demand"
}

cluster_interpretations = {
    0: {"name": "Cluster 1: Standard Operations", "desc": "Operating normally with a steady workload and typical error rates. No immediate changes needed.", "color": "#440154"},
    1: {"name": "Cluster 2: High Processing Friction", "desc": "Teams struggling with slow manual tasks (~97 minutes per day) and high data entry error rates.", "color": "#21918c"},
    2: {"name": "Cluster 3: High Patient Demand", "desc": "Teams overwhelmed by sheer patient volume, causing long appointment wait times (33.6 days) and overall inefficiency.", "color": "#fde725"}
}

df["Cluster Name"] = df["Cluster"].map(cluster_labels)

# generate risk scores
risk_features = df[
    [
        "annual_budget",
        "base_wait_days",
        "manual_workload_multiplier",
        "claim_denial_risk",
        "lab_delay_risk",
        "provider_count",
        "location"
    ]
]

risk_features = pd.get_dummies(
    risk_features,
    columns=["location"],
    drop_first=True
)

risk_features = risk_features.reindex(
    columns=risk_columns,
    fill_value=0
)

df["Predicted Risk"] = risk_model.predict(risk_features)

# generate composite strain score
risk_cols = ['total_manual_minutes', 'avg_overbooking', 'avg_wait_days', 'noshow_rate', 'error_rate']
z = df[risk_cols].apply(lambda c: (c - c.mean()) / c.std())
df['composite_strain_score'] = z.mean(axis=1)

# manual workflow data
error_model = joblib.load("models/error_model.pkl")
error_importance = pd.read_csv("models/error_feature_importance.csv")

# page one
if page == "Executive Overview":

    st.header("Executive Summary")

    st.markdown("""
    Hello MedCore Executive. 
    Welcome to the MedCore Operations Dashboard. This overview highlights each department's operational risk, helping you quickly identify which teams are operating efficiently and which may benefit from targeted process improvements or AI support.
    """)

    sizes = df["Cluster"].value_counts().to_dict()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='insight-card' style='border-top-color: {cluster_interpretations[0]['color']};'>"
                    f"<h3>{cluster_interpretations[0]['name']}</h3>"
                    f"<span class='cluster-badge' style='background-color:#ebdcf0; color:{cluster_interpretations[0]['color']};'>{sizes.get(0, 0)} Departments</span>"
                    f"<p style='margin-top:10px; font-size:0.9rem; color:#4a5568;'>{cluster_interpretations[0]['desc']}</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='insight-card' style='border-top-color: {cluster_interpretations[1]['color']};'>"
                    f"<h3>{cluster_interpretations[1]['name']}</h3>"
                    f"<span class='cluster-badge' style='background-color:#d2f4f2; color:{cluster_interpretations[1]['color']};'>{sizes.get(1, 0)} Departments</span>"
                    f"<p style='margin-top:10px; font-size:0.9rem; color:#4a5568;'>{cluster_interpretations[1]['desc']}</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='insight-card' style='border-top-color: {cluster_interpretations[2]['color']};'>"
                    f"<h3>{cluster_interpretations[2]['name']}</h3>"
                    f"<span class='cluster-badge' style='background-color:#fef9c3; color:#854d0e;'>{sizes.get(2, 0)} Departments</span>"
                    f"<p style='margin-top:10px; font-size:0.9rem; color:#4a5568;'>{cluster_interpretations[2]['desc']}</p></div>", unsafe_allow_html=True)


    # main metrics
    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Departments",
        len(df)
    )

    lowest_risk = df["Predicted Risk"].min()
    lowest_risk_depts = ", ".join(df.loc[df["Predicted Risk"] == lowest_risk, "department_name"])


    highest_risk = df["Predicted Risk"].max()
    highest_risk_depts = ", ".join(df.loc[df["Predicted Risk"] == highest_risk, "department_name"])

    d2.metric(
        "Lowest Risk",
        lowest_risk_depts,
        round(lowest_risk, 2),
        delta_color="inverse",
        delta_arrow="off"
    )

    d3.metric(
        "Highest Risk",
        highest_risk_depts,
        round(highest_risk, 2),
        delta_color="inverse",
        delta_arrow="off"
    )

    # main overview

    tab1, tab2 = st.tabs(["Chart View", "Table View"])

    with tab1:
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

    with tab2:
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

    # cluster summary table
    cluster_profile = df.groupby('Cluster').agg(
        department_count=('department_id', 'count'),
        avg_annual_budget=('annual_budget', 'mean'),
        avg_provider_count=('provider_count', 'mean'),
        avg_task_count=('task_count', 'mean'),
        avg_appt_count=('appt_count', 'mean'),
        avg_manual_minutes=('total_manual_minutes', 'mean'),
        avg_overbooking=('avg_overbooking', 'mean'),
        avg_wait_days=('avg_wait_days', 'mean'),
        avg_noshow_rate=('noshow_rate', 'mean'),
        avg_error_rate=('error_rate', 'mean'),
        avg_claim_denial_risk=('claim_denial_risk', 'mean'),
        avg_lab_delay_risk=('lab_delay_risk', 'mean'),
        avg_strain_rate=('strain_rate', 'mean'),
        average_composite_strain=('composite_strain_score', 'mean')
    ).reset_index()

    st.subheader("Operational Cluster Registry Lookup")
    st.markdown("Expand any of the groups below to see the specific departments assigned to that performance profile.")
    
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_units = df[df['Cluster'] == cluster_id].sort_values(by='department_name')
        title_string = f"View Departments in {cluster_labels[cluster_id]} ({len(cluster_units)} Units)"
        
        with st.expander(title_string):
            cols = st.columns(3)
            for loop_idx, (_, row) in enumerate(cluster_units.iterrows()):
                target_col = cols[loop_idx % 3]
                card_html = (
                    f"<div class='dept-chip'>"
                    f"<span class='id-badge'>{row['department_id']}</span>"
                    f"{row['department_name']}"
                    f"</div>"
                )
                target_col.markdown(card_html, unsafe_allow_html=True)


    
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

    st.header("Administrative Workflows Driving Errors")
    st.markdown("Discover which administrative activities contribute most to operational errors. These insights highlight the highest-impact opportunities for workflow automation and help prioritize AI initiatives that reduce manual effort and improve consistency.")
    
    st.markdown(f"""<div class='insight-card' style='border-top-color: #2b6cb0;'>
        <h3>💡 AI Insight</h3>
        <span class='cluster-badge' style='background-color:#ebdcf0; color:#2b6cb0;'>Admin Burden</span>
        <p style='margin-top:10px; font-size:0.9rem; color:#4a5568;'>
            This analysis identifies the administrative activities and systems most strongly associated with operational errors. 
            By highlighting where mistakes are most likely to occur, MedCore can prioritize AI automation and workflow improvements where they will have the greatest operational impact.
        </p>
    </div>
    """,
    unsafe_allow_html=True
    )

    tab_f1, tab_f2 = st.tabs(["Workflow Error Drivers", "System Error Drivers"])

    # Separate before cleaning names
    workflow_features = error_importance[error_importance["Feature"].str.contains("task_type|Manual|Claim|Workload")]

    system_features = error_importance[error_importance["Feature"].str.contains("system_used")]


    # Clean labels AFTER splitting

    for data in [workflow_features, system_features]:

        data["Feature"] = (
            data["Feature"]
            .str.replace("task_type_", "")
            .str.replace("system_used_", "")
            .str.replace("_", " ")
            .str.title()
        )

    feature_display = {
    "Emr": "EMR",
    "Excel": "Excel",
    "Email": "Email",
    "Billing System": "Billing System",
    "Paper Form": "Paper Forms",
    "Manual Minutes": "Manual Processing Time",
    "Manual Workload Multiplier": "Workload Pressure",
    "Claim Denial Risk": "Claim Denial Risk"
    }

    system_features["Feature"] = (
    system_features["Feature"]
    .replace(feature_display)
    )

    workflow_features["Feature"] = (
    workflow_features["Feature"]
    .replace(feature_display)
    )

    with tab_f1:
        top_features = workflow_features.head(10)

        fig4 = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Workflow Audit: Top Operational Error Drivers",
            labels={
                "Importance": "Relative Impact Score on Errors",
                "Feature": ""
            }
        )

        fig4.update_layout(
            yaxis={"categoryorder": "total ascending"},
            title_font=dict(
                size=16,
                color="#1a365d"
            )
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )


    with tab_f2:
        st.subheader("Systems Associated With Operational Errors")

        fig_system = px.bar(
            system_features,
            x="Importance",
            y="Feature",
            orientation="h",
            title="System Factors Associated With Errors",
            labels={
                "Importance": "Relative Impact Score",
                "Feature": ""
            }
        )

        fig_system.update_layout(
            yaxis={"categoryorder": "total ascending"},
            title_font=dict(
                size=16,
                color="#1a365d"
            )
        )

        st.plotly_chart(
            fig_system,
            use_container_width=True
        )
        

    st.header("Department Explorer")

    dept_task_summary = pd.read_csv("models/dept_task_summary.csv")

    selected_dept = st.selectbox(
    "Select Department",
    dept_task_summary["department_name"].unique()
)

    dept_tasks = (
        dept_task_summary[
            dept_task_summary["department_name"] == selected_dept
        ]
        .sort_values(
            "total_manual_minutes",
            ascending=False
        )
        )
    
    fig = px.bar(
    dept_tasks,
    x="total_manual_minutes",
    y="task_type",
    orientation="h",
    title=f"Workflow Burden in {selected_dept}",
    labels={
        "total_manual_minutes": "Total Manual Minutes",
        "task_type": "Task Type"
    }
)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 AI Recommendation")
    row = df[df["department_name"] == selected_dept].iloc[0]
    cluster = row["Cluster"]
    if cluster == 0:

        st.success("""
        #### Standard Operations

        ##### Recommendation
                   
        - Consider AI support for Excel

        - Continue monitoring

        - Use as benchmark
        """)
    elif cluster == 1:

        st.warning("""
        #### High Processing Friction

        ##### Recommended AI

        - Excel AI Support
                   
        - AI-Powered Claims Processing
        
        - Automated Follow-Up Agent
        """)
    else:

        st.error("""
        #### High Patient Demand

        ##### Recommendation

        - Appointment optimization

        - Waitlist automation

        - No-show prediction
        """)

# page 3
if page == "Operational Profiles":

    st.header("Operational Profiles")
    # placeholder page

    st.markdown("Departments have been grouped into operational profiles based on shared workload, efficiency, and patient demand characteristics." \
    "Use these profiles to understand where targeted interventions will have the greatest organizational impact.")

    c_breakdown_count = len(df[df['Cluster'] == 1])
    c_bottleneck_count = len(df[df['Cluster'] == 2])

    st.header("Organization-Wide Fixes")
    
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(label="Departments Needing Process Fixes", value=f"{c_breakdown_count} Units")
    with k2:
        st.metric(label="Departments Needing Schedule Help", value=f"{c_bottleneck_count} Units")
    with k3:
        st.metric(label="Highest Strain Department Group", value="Cluster 3")

    st.markdown("---")
    
    st.markdown("<div class='rec-card-small' style='border-left-color: #21918c;'> "
                "<h3>1. Fix Slow Workflows in Cluster 2 (High Processing Friction)</h3>"
                "<p><strong>Where help is needed:</strong> These departments spend way too much time on manual admin work (~97 minutes), which is causing a high rate of mistakes.</p>"
                "<ul>"
                "<li><strong>Simple Solution:</strong> Prioritize <strong>Department E (Business Operations)</strong> first. Introduce basic data entry automation or pre-filled software templates for high-frequency admin tasks.</li>"
                "<li><strong>Why it works:</strong> The data proves that long manual hours directly lead to input errors. Cutting down human data typing drops error rates instantly.</li>"
                "</ul></div>", unsafe_allow_html=True)
                
    st.markdown("<div class='rec-card-small' style='border-left-color: #fde725;'> "
                "<h3>2. Optimize Scheduling for Cluster 3 (High Patient Demand)</h3>"
                "<p><strong>Where help is needed:</strong> These teams are actually very efficient. They don't have broken systems, but they are completely buried under patient bookings, leading to 33.6-day wait times.</p>"
                "<ul>"
                "<li><strong>Simple Solution:</strong> Do not change their day-to-day procedures. Instead, add an automated text/email reminder setup to handle cancellations and automatically pull in patients from a waitlist to fill empty slots.</li>"
                "<li><strong>Why it works:</strong> Since these departments are already doing good work, fixing internal steps won't help. Optimizing the existing calendar reduces patient wait times without needing to hire more staff.</li>"
                "</ul></div>", unsafe_allow_html=True)

    st.markdown("<div class='rec-card-small' style='border-left-color: #440154;'> "
                "<h3>3. Maintain the Baseline for Cluster 1 (Standard Operations)</h3>"
                "<p><strong>Where help is needed:</strong> These departments are meeting expectations and operating smoothly.</p>"
                "<ul>"
                "<li><strong>Simple Solution:</strong> Leave them as they are, but use their performance numbers as a template or target goal for struggling teams.</li>"
                "<li><strong>Why it works:</strong> It ensures management focus is applied exclusively to high-risk areas instead of over-engineering healthy areas of the hospital.</li>"
                "</ul></div>", unsafe_allow_html=True)
        # Copy this block to paste into the locations below
    st.markdown("---")
    st.subheader("Operational Cluster Registry Lookup")
    st.markdown("Select an operational profile to identify departments requiring action.")
    
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_units = df[df['Cluster'] == cluster_id].sort_values(by='department_name')
        title_string = f"View Departments in {cluster_labels[cluster_id]} ({len(cluster_units)} Units)"
        
        with st.expander(title_string):
            cols = st.columns(3)
            for loop_idx, (_, row) in enumerate(cluster_units.iterrows()):
                target_col = cols[loop_idx % 3]
                card_html = (
                    f"<div class='dept-chip'>"
                    f"<span class='id-badge'>{row['department_id']}</span>"
                    f"{row['department_name']}"
                    f"</div>"
                )
                target_col.markdown(card_html, unsafe_allow_html=True)

# page 4
if page == "Executive Action Plan":
    st.header("Executive Action Plan")
    
    st.markdown("""
    Translate AI insights into action. This page prioritizes recommended AI initiatives, outlines their expected business impact, and provides a phased roadmap for implementation across MedCore.
    """)

    st.markdown("""
    <div class='insight-card' style='border-top-color:#2b6cb0;'>

    <h3>💡 AI Recommendation Summary</h3>

    <p style='font-size:0.95rem;color:#4a5568;'>
    AI analysis identified that operational inefficiency is primarily driven by 
    high-volume manual workflows, particularly reporting, claims processing, 
    and repetitive follow-up activities. Targeting these processes provides 
    the greatest opportunity for improving productivity and consistency.
    </p>

    </div>
    """, unsafe_allow_html=True)
    
    recommendations = {

    "Excel Report": {
        "problem": "High manual reporting burden",
        "solution": "AI Reporting Assistant",
        "impact": "Reduce manual reporting effort and improve reporting consistency",
        "priority": "High"
    },

    "Claim Review": {
        "problem": "Manual claims validation creates delays",
        "solution": "AI Claims Processing Assistant",
        "impact": "Reduce claim processing time while improving consistency and accuracy",
        "priority": "High"
    },

    "Patient Follow-Up": {
        "problem": "Repeated communication tasks consume staff time",
        "solution": "AI Follow-Up Agent",
        "impact": "Automate reminders and patient communication workflows",
        "priority": "Medium"
    },

    "Data Entry": {
        "problem": "Duplicate administrative entry",
        "solution": "Intelligent Data Extraction",
        "impact": "Reduce repetitive data entry and improve documentation accuracy",
        "priority": "Medium"
    },

    "Scheduling": {
        "problem": "Appointment coordination inefficiencies",
        "solution": "AI Scheduling Optimization",
        "impact": "Improve capacity utilization and reduce delays",
        "priority": "Medium"
    }
    }

    recommendation_items = list(recommendations.items())

    cols = st.columns(2)

    for idx, (task, rec) in enumerate(recommendation_items):

        color = (
            "#dc2626" if rec["priority"] == "High"
            else "#ea580c"
        )

        with cols[idx % 2]:
            st.markdown(
            f"""
            <div class='rec-card' style='border-left:6px solid {color}; height:320px;'>

            <h3>{task}</h3>

            <b>Current Challenge:</b>
            <p>{rec["problem"]}</p>

            <b>💡 Recommended AI Solution:</b>
            <p>{rec["solution"]}</p>

            <b>Expected Impact:</b>
            <p>{rec["impact"]}</p>

            <span class='cluster-badge'
            style='background:#fee2e2;color:{color};'>
            {rec["priority"]} Priority
            </span>

            </div>
            """,
            unsafe_allow_html=True
            )

    st.subheader("Recommended Implementation Roadmap")

    st.markdown(
    "A phased implementation approach enables MedCore to deliver early operational improvements while building toward organization-wide AI adoption."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class='map-card'>
        <h3>Phase 1</h3>
        <h4>0–3 Months</h4>

        <b>Priority</b><br>
        Automate high-volume reporting workflows
        <br><br>
        <b>Expected Outcome</b><br>
        • Reduce repetitive administrative work<br>
        • Free staff time for patient care
                    
        <span class='cluster-badge'
        style='background:#fee2e2;color:#dc2626;'>
        Excel Report
        </span>
                    
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='map-card'>
        <h3>Phase 2</h3>
        <h4>3–6 Months</h4>

        <b>Priority</b><br>
        Deploy AI assistants for claims processing and patient follow-up
        <br><br>
        <b>Expected Outcome</b><br>
        • Improve processing consistency<br>
        • Reduce turnaround times
        
        <span class='cluster-badge'
        style='background:#fee2e2;color:#dc2626;'>
        Claim Review
        </span>            
        
        <span class='cluster-badge'
        style='background:#fee2e2;color:#dc2626;'>
        Patient Follow-Up
        </span>            
        
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='map-card'>
        <h3>Phase 3</h3>
        <h4>6–12 Months</h4>

        <b>Priority</b><br>
        Expand predictive monitoring across all departments
        <br><br>
        <b>Expected Outcome</b><br>
        • Organization-wide operational visibility<br>
        • Continuous AI-driven improvement
                    
        <span class='cluster-badge'
        style='background:#fee2e2;color:#ea580c;'>
        Progress Check
        </span> 
                    
        </div>
        """, unsafe_allow_html=True)