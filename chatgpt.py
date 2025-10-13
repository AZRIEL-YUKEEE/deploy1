import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---- Streamlit Page Setup ----
st.set_page_config(page_title="K-Means Cluster Maps", layout="wide")
st.title("🎯 K-Means Cluster Maps for Sex and Course vs Aptitude Scores")

# ---- File Upload ----
uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # ---- Read file ----
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # ---- Clean columns ----
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # ---- Required columns ----
    required_cols = [
        "sex",
        "course",
        "general_ability",
        "verbal_aptitude",
        "numerical_aptitude",
        "spatial_aptitude",
        "perceptual_aptitude",
        "manual_dexterity"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        st.write("Your file contains:", list(df.columns))
        st.stop()

    df = df.dropna(subset=required_cols)
    df["course"] = df["course"].astype(int)

    # ---- Helper function for plotting ----
    def cluster_and_plot(x_col, y_col, title, y_label):
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(df[[y_col, x_col]])
        centers = kmeans.cluster_centers_

        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            df[x_col], df[y_col],
            c=df["cluster"], cmap='viridis', s=80, edgecolors='w', linewidth=0.5
        )
        ax.scatter(
            centers[:, 1], centers[:, 0],
            c="red", s=200, marker="X", label="Cluster Centers"
        )
        ax.set_title(title)
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_label)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)

    # ============================
    # 📊 SEX VS APTITUDE CLUSTERS
    # ============================
    st.header("📌 SEX vs Aptitude Scores")
    cluster_and_plot("general_ability", "sex", "K-Means Cluster Map (Sex vs General Ability)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("verbal_aptitude", "sex", "K-Means Cluster Map (Sex vs Verbal Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("numerical_aptitude", "sex", "K-Means Cluster Map (Sex vs Numerical Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("spatial_aptitude", "sex", "K-Means Cluster Map (Sex vs Spatial Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("perceptual_aptitude", "sex", "K-Means Cluster Map (Sex vs Perceptual Aptitude)", "Sex (1=Female, 2=Male)")
    cluster_and_plot("manual_dexterity", "sex", "K-Means Cluster Map (Sex vs Manual Dexterity)", "Sex (1=Female, 2=Male)")

    # ============================
    # 📚 COURSE VS APTITUDE CLUSTERS
    # ============================
    st.header("📌 COURSE vs Aptitude Scores")
    cluster_and_plot("general_ability", "course", "K-Means Cluster Map (Course vs General Ability)", "Course Code")
    cluster_and_plot("verbal_aptitude", "course", "K-Means Cluster Map (Course vs Verbal Aptitude)", "Course Code")
    cluster_and_plot("numerical_aptitude", "course", "K-Means Cluster Map (Course vs Numerical Aptitude)", "Course Code")
    cluster_and_plot("spatial_aptitude", "course", "K-Means Cluster Map (Course vs Spatial Aptitude)", "Course Code")
    cluster_and_plot("perceptual_aptitude", "course", "K-Means Cluster Map (Course vs Perceptual Aptitude)", "Course Code")
    cluster_and_plot("manual_dexterity", "course", "K-Means Cluster Map (Course vs Manual Dexterity)", "Course Code")

    st.success("✅ All cluster maps generated successfully!")

else:
    st.info("Please upload your CSV or Excel file to generate the cluster maps.")
