import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_rf.pkl")
fitur = joblib.load("fitur.pkl")   #

st.title("Prediksi Harga Rumah Seattle")

st.write("Masukkan karakteristik rumah, lalu aplikasi akan memprediksi harga jualnya.")

# 2. Input user (sesuaikan dengan fitur yang kamu pakai)
sqft_living   = st.number_input("Luas bangunan (sqft_living)",  min_value=300,  max_value=15000, value=1800)
sqft_basement = st.number_input("Luas basement (sqft_basement)",min_value=0,    max_value=5000,  value=0)
floors        = st.number_input("Jumlah lantai (floors)",        min_value=1.0, max_value=3.5,   value=1.0, step=0.5)
sqft_above    = st.number_input("Luas di atas tanah (sqft_above)",min_value=300,max_value=15000, value=1800)
bedrooms      = st.number_input("Jumlah kamar tidur (bedrooms)", min_value=0,   max_value=10,    value=3)
bathrooms     = st.number_input("Jumlah kamar mandi (bathrooms)",min_value=0.0, max_value=8.0,   value=2.0, step=0.25)
condition     = st.number_input("Kondisi (1–5)",                 min_value=1,   max_value=5,     value=3)
yr_built      = st.number_input("Tahun bangun (yr_built)",       min_value=1900,max_value=2015,  value=1995)
yr_renovated  = st.number_input("Tahun renovasi (0 jika belum)", min_value=0,   max_value=2015,  value=0)
sqft_lot      = st.number_input("Luas tanah (sqft_lot)",         min_value=400, max_value=1100000, value=5000)
zipcode       = st.number_input("Zipcode",                        min_value=98000,max_value=99999, value=98133)
view          = st.number_input("View (0–4)",                     min_value=0,    max_value=4,     value=0)

# 3. Susun ke DataFrame sesuai urutan fitur
data_input = pd.DataFrame([{
    "sqft_living": sqft_living,
    "sqft_basement": sqft_basement,
    "floors": floors,
    "sqft_above": sqft_above,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "condition": condition,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,
    "sqft_lot": sqft_lot,
    "zipcode": zipcode,
    "view": view,
}])

# jika di training kamu juga membuat fitur turunan, hitung lagi di sini
data_input["ratio_liv_lot"] = data_input["sqft_living"] / data_input["sqft_lot"]

# pastikan kolomnya diurutkan sama seperti waktu training
data_input = data_input[fitur]

# 4. Tombol prediksi
if st.button("Prediksi Harga"):
    pred = model.predict(data_input)[0]
    st.success(f"Perkiraan harga rumah: ${pred:,.0f}")
