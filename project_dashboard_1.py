import streamlit as st
import pandas as pd
import numpy as np
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
# 1. PAGE CONFIGURATION & EXECUTIVE THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MedCore Operations Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .insight-card { background-color: #ffffff; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; border-top: 4px solid #2b6cb0; }
    .rec-card { background-color: #ffffff; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2b6cb0; }
    .cluster-badge { padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; }
    .dept-chip { background-color: #edf2f7; color: #2d3748; padding: 6px 12px; border-radius: 16px; font-size: 0.85rem; font-weight: 500; display: inline-block; margin: 4px; border: 1px solid #e2e8f0; }
    .id-badge { background-color: #4a5568; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; margin-right: 6px; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & ENGINE COMPUTATION
# -----------------------------------------------------------------------------
@st.cache_data
def run_analytics_engine():
    # A. Load the core department analysis dataset
    df = pd.read_csv("department_analysis.csv")
    
    cluster_features = [
        "task_count", "appt_count", "provider_count", "total_manual_minutes",
        "avg_manual_minutes", "manual_workload_multiplier", "avg_wait_days",
        "error_rate", "avg_overbooking", "noshow_rate", "strain_rate",
        "claim_denial_risk", "lab_delay_risk"
    ]
    
    X = df[cluster_features]
    
    # Scale and Cluster identically to baseline specifications
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Composite Strain Score Calculation Workflow
    risk_cols = ['total_manual_minutes', 'avg_overbooking', 'avg_wait_days', 'noshow_rate', 'error_rate']
    z = df[risk_cols].apply(lambda c: (c - c.mean()) / c.std())
    df['composite_strain_score'] = z.mean(axis=1)
    
    # Dimensionality Reduction for Visuals
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]
    
    # B. Load Admin Data & Fit Random Forest
    mw_df = pd.read_csv("manual_workflow_clean.csv")
    dept_df = pd.read_csv("departments.csv")
    
    admin_data = pd.merge(mw_df, dept_df, on='department_id', how='left')
    admin_data['error_target'] = admin_data['error_flag'].map({'Y': 1, 'N': 0})
    admin_data = admin_data.dropna(subset=['error_target'])
    
    X_rf = admin_data[['task_type', 'system_used', 'manual_minutes', 'manual_workload_multiplier', 'claim_denial_risk']]
    y_rf = admin_data['error_target']
    
    preprocessor_opt1 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['manual_minutes', 'manual_workload_multiplier', 'claim_denial_risk']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['task_type', 'system_used'])
        ])
        
    pipe_model_1b = Pipeline(steps=[
        ('preprocessor', preprocessor_opt1),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1))
    ])
    pipe_model_1b.fit(X_rf, y_rf)
    
    feature_names = pipe_model_1b.named_steps['preprocessor'].get_feature_names_out()
    importances = pipe_model_1b.named_steps['classifier'].feature_importances_
    
    df_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_importances = df_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    df_importances['Feature'] = df_importances['Feature'].str.replace('num__', '').str.replace('cat__', '')
    
    return df, df_importances, cluster_features

# Run the analysis pipeline on the live datasets
try:
    df, df_importances, cluster_features = run_analytics_engine()
except FileNotFoundError as e:
    st.error(f"Required data file missing from directory: {e.filename}. Please verify that department_analysis.csv, manual_workflow_clean.csv, and departments.csv are available.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR LAYOUT
# -----------------------------------------------------------------------------
st.sidebar.title("MedCore Systems")
st.sidebar.caption("Operational Performance Dashboard")
st.sidebar.markdown("---")
view_selection = st.sidebar.radio(
    "Navigation Menu",
    ["Executive Profiles", "Strategic Grouping Visualizer", "Operational Error Drivers", "Actionable Recommendations"]
)

cluster_interpretations = {
    0: {"name": "Cluster 0: Standard Operations", "desc": "Operating normally with a steady workload and typical error rates. No immediate changes needed.", "color": "#440154"},
    1: {"name": "Cluster 1: High Processing Friction", "desc": "Teams struggling with slow manual tasks (~97 minutes per day) and high data entry error rates.", "color": "#21918c"},
    2: {"name": "Cluster 2: High Patient Demand", "desc": "Efficient teams that are overwhelmed by sheer patient volume, causing long appointment wait times (33.6 days).", "color": "#fde725"}
}

# -----------------------------------------------------------------------------
# 4. RENDER PERSPECTIVES WITH EXECUTIVE CONTEXT
# -----------------------------------------------------------------------------

# --- PERSPECTIVE 1: EXECUTIVE PROFILES ---
if view_selection == "Executive Profiles":
    st.title("🏥 Executive Summary: MedCore Operational Profiling")
    st.markdown("""
    This breakdown groups MedCore Health's departments based on operational efficiency. 
    It separates teams that have internal process issues from teams that are simply overwhelmed by high patient volume, making it easier to see where help is needed most.
    """)
    
    # Render Cluster Sizing Cards Dynamically from Live Data
    sizes = df["cluster"].value_counts().to_dict()
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

    st.markdown("### Department Performance Comparison Matrix")
    
    # Groupby summary table
    cluster_profile = df.groupby('cluster').agg(
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

    clean_summary = cluster_profile.T
    clean_summary.columns = [f'Cluster {int(i)}' for i in clean_summary.loc['cluster']]
    clean_summary = clean_summary.drop('cluster')
    
    st.dataframe(clean_summary.style.format("{:.2f}").background_gradient(cmap="Blues", axis=1), use_container_width=True)
    # Copy this block to paste into the locations below
    st.markdown("---")
    st.subheader("📋 Operational Cluster Registry Lookup")
    st.markdown("Expand any of the groups below to see the specific departments assigned to that performance profile.")
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_units = df[df['cluster'] == cluster_id].sort_values(by='department_name')
        title_string = f"🔍 View Departments in {cluster_interpretations[cluster_id]['name']} ({len(cluster_units)} Units)"
        
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


# --- PERSPECTIVE 2: STRATEGIC GROUPING VISUALIZER (WITH LOOKUP ADDED) ---
elif view_selection == "Strategic Grouping Visualizer":
    st.title("📊 Enterprise Performance Visualizations")
    st.markdown("These charts map out our departments visually to show how they separate into distinct groups based on performance trends.")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        fig1, ax1 = plt.subplots(figsize=(6.5, 5))
        scatter1 = ax1.scatter(df["pca1"], df["pca2"], c=df["cluster"], cmap="viridis", alpha=0.75, edgecolor='w', s=55)
        ax1.set_xlabel("PCA Dimension 1", fontsize=10)
        ax1.set_ylabel("PCA Dimension 2", fontsize=10)
        ax1.set_title("Clustering Analysis of Department Characteristics", fontsize=11, fontweight='bold', color='#1a365d')
        cbar1 = fig1.colorbar(scatter1, ax=ax1, ticks=[0, 1, 2])
        cbar1.set_label("Assigned Group Index", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption("**Insight:** Clear spacing between groups highlights that each category faces fundamentally different challenges.")
        
    with col_p2:
        fig2, ax2 = plt.subplots(figsize=(6.5, 5))
        scatter2 = ax2.scatter(df["avg_wait_days"], df["avg_manual_minutes"], c=df["cluster"], cmap="viridis", alpha=0.75, edgecolor='w', s=55)
        ax2.set_xlabel("Average Appointment Wait Time (Days)", fontsize=10)
        ax2.set_ylabel("Average Manual Processing Time (Minutes)", fontsize=10)
        ax2.set_title("Department Clusters by Appointment Wait Time and Manual Processing Time", fontsize=11, fontweight='bold', color='#1a365d')
        cbar2 = fig2.colorbar(scatter2, ax=ax2, ticks=[0, 1, 2])
        cbar2.set_label("Assigned Group Index", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption("**Insight:** This map isolates the problem clearly. Cluster 2 has access issues (high wait times), while Cluster 1 has internal system delays (high manual time).")

    st.markdown("---")
    st.subheader("Composite Risk/Strain Comparisons")
    st.markdown("This graph shows the composite operational strain/risk scores, calculated as an average of normalized Z-scores across five key risk metrics:  total manual minutes, average overbooking, average wait days, no-show rate, and error rate.")
    
    fig3, ax3 = plt.subplots(figsize=(11, 4.5))
    sns.boxplot(data=df, x='cluster', y='composite_strain_score', palette='Set2', hue='cluster', legend=False, ax=ax3)
    ax3.set_title('Overall Operational Risk/Strain Across Clusters', fontsize=11, fontweight='bold', color='#1a365d')
    ax3.set_xlabel('Assigned Department Category', fontsize=10)
    ax3.set_ylabel('Process Strain Score', fontsize=10)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig3)

    # MOVED SECTION: CLEAN REGISTRY LOOKUP NOW AT THE BOTTOM OF VISUALIZER
    st.markdown("---")
    st.subheader("📋 Operational Cluster Registry Lookup")
    st.markdown("Expand any of the groups below to see the specific departments assigned to that performance profile.")
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_units = df[df['cluster'] == cluster_id].sort_values(by='department_name')
        title_string = f"🔍 View Departments in {cluster_interpretations[cluster_id]['name']} ({len(cluster_units)} Units)"
        
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


# --- PERSPECTIVE 3: OPERATIONAL ERROR DRIVERS ---
elif view_selection == "Operational Error Drivers":
    st.title("🤖 Administrative Workflows Driving Errors")
    st.markdown("This dashboard pinpoints exactly which types of tasks and system issues correlate most heavily with manual workflow errors, as identified by the feature importances of a Random Forest Classification model")
    
    col_f1, col_f2 = st.columns([4, 6])
    
    with col_f1:
        st.subheader("Top Drivers of Operational Errors")
        st.markdown("Statistical scores showing which specific activities lead to mistakes most often.")
        st.dataframe(
            df_importances.style.background_gradient(cmap='Blues', subset=['Importance']).format({'Importance': '{:.4f}'}),
            use_container_width=True
        )
        
    with col_f2:
        fig4, ax4 = plt.subplots(figsize=(7, 5.2))
        sns.barplot(data=df_importances.head(10), x='Importance', y='Feature', palette='Blues_r', ax=ax4, hue='Feature', legend=False)
        ax4.set_title("Workflow Audit: Top Operational Error Drivers", fontsize=11, fontweight='bold', color='#1a365d')
        ax4.set_xlabel("Relative Impact Score on Errors")
        ax4.set_ylabel("")
        ax4.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig4)


# --- PERSPECTIVE 4: SIMPLIFIED RECOMMENDATIONS SECTION ---
elif view_selection == "Actionable Recommendations":
    st.title("🎯 Practical Improvements for Management")
    st.markdown("We have skipped the complicated consultant jargon. Here is a simple, direct look at exactly where improvements are needed and the easiest steps you can take right away.")

    c_breakdown_count = len(df[df['cluster'] == 1])
    c_bottleneck_count = len(df[df['cluster'] == 2])
    
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(label="Departments Needing Process Fixes", value=f"{c_breakdown_count} Units")
    with k2:
        st.metric(label="Departments Needing Schedule Help", value=f"{c_bottleneck_count} Units")
    with k3:
        st.metric(label="Highest Strain Department Group", value="Cluster 1")

    st.markdown("---")
    
    st.markdown("<div class='rec-card' style='border-left-color: #21918c;'> "
                "<h3>1. Fix Slow Workflows in Cluster 1 (Process Breakdown)</h3>"
                "<p><strong>Where help is needed:</strong> These departments spend way too much time on manual admin work (~97 minutes), which is causing a high rate of mistakes.</p>"
                "<ul>"
                "<li><strong>Simple Solution:</strong> Prioritize <strong>Department E (Business Operations)</strong> first. Introduce basic data entry automation or pre-filled software templates for high-frequency admin tasks.</li>"
                "<li><strong>Why it works:</strong> The data proves that long manual hours directly lead to input errors. Cutting down human data typing drops error rates instantly.</li>"
                "</ul></div>", unsafe_allow_html=True)
                
    st.markdown("<div class='rec-card' style='border-left-color: #fde725;'> "
                "<h3>2. Optimize Scheduling for Cluster 2 (High Patient Demand)</h3>"
                "<p><strong>Where help is needed:</strong> These teams are actually very efficient. They don't have broken systems, but they are completely buried under patient bookings, leading to 33.6-day wait times.</p>"
                "<ul>"
                "<li><strong>Simple Solution:</strong> Do not change their day-to-day procedures. Instead, add an automated text/email reminder setup to handle cancellations and automatically pull in patients from a waitlist to fill empty slots.</li>"
                "<li><strong>Why it works:</strong> Since these departments are already doing good work, fixing internal steps won't help. Optimizing the existing calendar reduces patient wait times without needing to hire more staff.</li>"
                "</ul></div>", unsafe_allow_html=True)

    st.markdown("<div class='rec-card' style='border-left-color: #440154;'> "
                "<h3>3. Maintain the Baseline for Cluster 0 (Standard Operations)</h3>"
                "<p><strong>Where help is needed:</strong> These departments are meeting expectations and operating smoothly.</p>"
                "<ul>"
                "<li><strong>Simple Solution:</strong> Leave them as they are, but use their performance numbers as a template or target goal for struggling teams.</li>"
                "<li><strong>Why it works:</strong> It ensures management focus is applied exclusively to high-risk areas instead of over-engineering healthy areas of the hospital.</li>"
                "</ul></div>", unsafe_allow_html=True)
        # Copy this block to paste into the locations below
    st.markdown("---")
    st.subheader("📋 Operational Cluster Registry Lookup")
    st.markdown("Expand any of the groups below to see the specific departments assigned to that performance profile.")
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_units = df[df['cluster'] == cluster_id].sort_values(by='department_name')
        title_string = f"🔍 View Departments in {cluster_interpretations[cluster_id]['name']} ({len(cluster_units)} Units)"
        
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