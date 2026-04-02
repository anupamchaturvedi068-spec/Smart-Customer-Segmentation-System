import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("kmeans_model.pkl", "rb"))
df = pd.read_csv("clustered_data.csv")

st.title("🔥 Smart Customer Segmentation System")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Input sliders
income = st.slider("Annual Income (k$)", 10, 150)
score = st.slider("Spending Score", 1, 100)

# Prediction
if st.button("Predict Customer Type"):
    data = np.array([[income, score]])
    cluster = model.predict(data)[0]

    st.success(f"Cluster: {cluster}")

    # Meaning of clusters
    if cluster == 0:
        st.info("Low Income - High Spending 💸")
    elif cluster == 1:
        st.info("High Income - High Spending 🤑")
    elif cluster == 2:
        st.info("Low Income - Low Spending")
    elif cluster == 3:
        st.info("High Income - Low Spending")
    else:
        st.info("Average Customer")

# Visualization
st.subheader("Cluster Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'])
ax.set_xlabel("Income")
ax.set_ylabel("Score")
st.pyplot(fig)