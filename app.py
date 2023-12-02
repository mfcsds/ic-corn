import streamlit as st
from st_pages import Page, show_pages, add_page_title

# Judul aplikasi
st.title('Aplikasi Klasifikasi Tipe Jagung')

# Penjelasan aplikasi
st.write("""
Selamat datang di Aplikasi Klasifikasi Tipe Jagung. Aplikasi ini menggunakan teknologi 
machine learning untuk mengidentifikasi berbagai tipe jagung dari gambar yang diunggah. 
Dengan menggunakan algoritma klasifikasi canggih, aplikasi ini dapat membantu petani, 
peneliti, dan penggemar tanaman dalam mengklasifikasikan tipe jagung dengan mudah dan cepat.
""")

# Menambahkan beberapa penjelasan tambahan jika diperlukan
st.header('Cara Penggunaan')
st.write("""
- Unggah gambar jagung yang ingin Anda klasifikasikan.
- Tunggu hingga sistem memproses gambar.
- Lihat hasil klasifikasi yang ditampilkan oleh aplikasi.
""")

st.divider()
st.header('Tentang Kami')
st.write("""
Aplikasi ini dikembangkan oleh tim yang berdedikasi dalam bidang pertanian dan teknologi. 
Kami berkomitmen untuk menyediakan solusi inovatif yang dapat mendukung kegiatan pertanian 
dan penelitian tanaman.
""")

# Optional -- adds the title and icon to the current page
add_page_title()

# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be
show_pages(
    [
        Page("app.py", "Home", "🏠"),
         Page("pages/3_📊_Model_Klasifikasi.py", "Model Klasifikasi"),
    ]
)