import streamlit as st
import numpy as np
import joblib

# Load model K-Means
model = joblib.load('kmeans_model.pkl')

# Judul aplikasi
st.title("Prediksi Cluster Obesitas dengan K-Means")
st.write("Masukkan data pada fitur di bawah ini untuk memprediksi cluster obesitas.")

# Input fitur dengan slider atau pilihan
st.header("Input Data Baru:")
age = st.slider("Age", min_value=5, max_value=100, step=1, value=25)
gender = st.selectbox("Gender (0 = Female, 1 = Male)", options=[0, 1], index=0)
height = st.slider("Height (cm)", min_value=50.0, max_value=250.0, step=0.1, value=170.0)
weight = st.slider("Weight (kg)", min_value=10.0, max_value=200.0, step=0.1, value=70.0)
favc = st.selectbox("Frequent Consumption of High Calorie Food (FAVC)", options=[0, 1], index=0)
ch2o = st.slider("Daily Water Intake (liters)", min_value=0.1, max_value=10.0, step=0.1, value=2.0)
family_history = st.selectbox("Family History with Overweight", options=[0, 1], index=0)

# Tombol prediksi
if st.button("Prediksi"):
    # Format input untuk model
    new_data = np.array([[age, gender, height, weight, favc, ch2o, family_history]])
    
    # Prediksi cluster
    predicted_cluster = model.predict(new_data)
    
    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(f"Data baru masuk ke cluster: *{predicted_cluster[0]}*")

