import streamlit as st
import pandas as pd
import math

# ============================
# SIDEBAR
# ============================

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard dengan Sidebar",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load data CSV
@st.cache_data
def load_data():
    return pd.read_csv("wondr_scrapped.csv")  # Pastikan file ini ada di folder yang sama

data = load_data()

# Inisialisasi session state untuk tombol menu
if "menu1" not in st.session_state:
    st.session_state.menu1 = None
if "menu4" not in st.session_state:
    st.session_state.menu4 = None
if "menu5" not in st.session_state:
    st.session_state.menu5 = None

# Sidebar
st.sidebar.title("ANALISIS SENTIMEN")

# Menu 1: Data Awal (dengan tombol)
st.sidebar.markdown("### 📊 Data Awal")
if st.sidebar.button("Dataset Asli"):
    st.session_state.menu1 = "Dataset Asli"
    st.session_state.menu4 = None
    st.session_state.menu5 = None
if st.sidebar.button("Informasi Fitur"):
    st.session_state.menu1 = "Informasi Fitur"
    st.session_state.menu4 = None
    st.session_state.menu5 = None

# Menu 2: Dropdown Preprocessing
menu2 = st.sidebar.selectbox("⚙️ Preprocessing", ["Pilih...", "Case Folding", "Cleaning", "Tokenizing", "Stopword Removal", "Stemming", "Normalisasi"])

# Menu 3: Dropdown Pemodelan
menu3 = st.sidebar.selectbox("📈 Pemodelan", ["Pilih...", "Split Data", "Training", "Evaluasi"])

# Menu 4: Visualisasi (dengan tombol)
st.sidebar.markdown("### 📊 Visualisasi")
if st.sidebar.button("Grafik Label"):
    st.session_state.menu1 = None
    st.session_state.menu4 = "Grafik Label"
    st.session_state.menu5 = None
if st.sidebar.button("Hasil Akurasi"):
    st.session_state.menu1 = None
    st.session_state.menu4 = "Hasil Akurasi"
    st.session_state.menu5 = None
if st.sidebar.button("Confusion Matrix"):
    st.session_state.menu1 = None
    st.session_state.menu4 = "Confusion Matrix"
    st.session_state.menu5 = None

# Menu 5: Pembahasan (dengan tombol)
st.sidebar.markdown("### 📝 Pembahasan")
if st.sidebar.button("Analisis Hasil"):
    st.session_state.menu1 = None
    st.session_state.menu4 = None
    st.session_state.menu5 = "Analisis Hasil"
if st.sidebar.button("Kesimpulan"):
    st.session_state.menu1 = None
    st.session_state.menu4 = None
    st.session_state.menu5 = "Kesimpulan"

# ============================
# KONTEN UTAMA
# ============================

st.title("📋 Halaman Dashboard Responsif")
st.markdown("Selamat datang di dashboard interaktif dengan sidebar menu.")

# Menampilkan konten sesuai tombol yang diklik
if st.session_state.menu1 == "Dataset Asli":
    st.subheader("📊 Data Awal: Dataset Asli")
    
    # Pagination
    rows_per_page = 10
    total_rows = len(data)
    total_pages = math.ceil(total_rows / rows_per_page)

    # Buat nomor halaman
    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1)

    # Hitung indeks baris untuk halaman saat ini
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    paginated_data = data.iloc[start_idx:end_idx]

    st.dataframe(paginated_data, use_container_width=True)

    st.markdown(f"Menampilkan **{start_idx + 1} - {min(end_idx, total_rows)}** dari **{total_rows}** data.")

if menu2 != "Pilih...":
    st.subheader(f"⚙️ Preprocessing: {menu2}")
    st.write(f"Menampilkan proses preprocessing: **{menu2}**.")

if menu3 != "Pilih...":
    st.subheader(f"📈 Pemodelan: {menu3}")
    st.write(f"Menampilkan hasil pemodelan: **{menu3}**.")

if st.session_state.menu4:
    st.subheader(f"📊 Visualisasi: {st.session_state.menu4}")
    st.write(f"Menampilkan visualisasi: **{st.session_state.menu4}**.")

if st.session_state.menu5:
    st.subheader(f"📝 Pembahasan: {st.session_state.menu5}")
    st.write(f"Menampilkan pembahasan: **{st.session_state.menu5}**.")

# Footer
st.markdown("---")
st.markdown("© 2025 Aisyah Wulandari. All rights reserved.")
