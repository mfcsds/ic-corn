import streamlit as st
from keras.models import load_model
from keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image

# Load model yang telah disimpan
model = load_model('./model/mdj.h5')

# Fungsi untuk memproses gambar dan melakukan prediksi
def predict_image(img):
    # Mengolah gambar
    img = img.resize((150, 150))  # Sesuaikan dengan ukuran input model
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Membuat batch yang berisi satu gambar
    img_array /= 255.0  # Normalisasi gambar

    # Melakukan prediksi
    prediction = model.predict(img_array)
    return prediction

# Membuat antarmuka pengguna dengan Streamlit
st.header('Aplikasi Prediksi Tipe Jagung')

st.write("""
Unggah gambar jagung dan aplikasi ini akan memprediksi tipenya menggunakan model CNN.
""")

# Unggah gambar
uploaded_file = st.file_uploader("Pilih gambar jagung", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True, width=200)
    prediction = predict_image(image)


if st.button("Prediksi"):
     # Tampilkan hasil prediksi
    st.write("Hasil Prediksi:")
    
    # Contoh: Jika Anda memiliki 3 kelas, misalnya Tipe A, Tipe B, Tipe C
    classes = ['Jagung Muda', 'Jagung Matang', 'Jagung Rusak']
    predicted_class = np.argmax(prediction, axis=1)  # Mendapatkan indeks kelas dengan probabilitas tertinggi
    st.success(f"Jenis Jagung: {classes[predicted_class[0]]}")

# Menyediakan informasi tambahan
st.sidebar.write("Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini menggunakan model deep learning untuk mengklasifikasikan tipe jagung.
""")

# Informasi pengembang aplikasi di sidebar
st.sidebar.header("Informasi Pengembang")
st.sidebar.write("Dikembangkan oleh [Nama Anda atau Organisasi].")
