import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Streamlit page setup
st.set_page_config(page_title="K-Means Clustering App", layout="centered")
st.title("🎯 K-Means Clustering (Sex vs General Ability)")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Display dataset preview
    st.subheader("📋 Uploaded Data")
    st.dataframe(df.head())

    # Validate required columns
    if "Sex" in df.columns and "General_Ability" in df.columns:
        # Step 1: Standardize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[["Sex", "General_Ability"]])

        # Step 2: Apply KMeans (k=2)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        df["Cluster"] = kmeans.fit_predict(scaled)

        # Step 3: Display cluster centers
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(centers, columns=["Sex", "General_Ability"])
        st.subheader("📊 Cluster Centers")
        st.dataframe(centers_df.style.format({"Sex": "{:.2f}", "General_Ability": "{:.2f}"}))

        # Step 4: Plot
        st.subheader("🗺️ Cluster Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(df["Sex"], df["General_Ability"], c=df["Cluster"], cmap="viridis")
        plt.xlabel("Sex (1=Female, 2=Male)")
        plt.ylabel("General Ability")
        plt.title("K-Means Cluster Map (Sex vs General Ability)")
        st.pyplot(fig)

        # Step 5: Downloadable results
        st.subheader("⬇️ Download Clustered Data")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="clustered_data.csv", mime="text/csv")

    else:
        st.error("❌ The uploaded file must contain 'Sex' and 'General_Ability' columns.")

else:
    st.info("📂 Please upload an Excel file to begin.")
