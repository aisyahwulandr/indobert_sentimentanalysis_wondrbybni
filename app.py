import streamlit as st
import pandas as pd
import math

# ============================
# KONFIGURASI HALAMAN
# ============================

st.set_page_config(
    page_title="Dashboard dengan Sidebar",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================
# MEMUAT DATA
# ============================

@st.cache_data
def load_data():
    return pd.read_csv("wondr_scrapped.csv")

data = load_data()

# ============================
# INISIALISASI SESSION STATE
# ============================

if "menu1" not in st.session_state:
    st.session_state.menu1 = None
if "menu2" not in st.session_state:
    st.session_state.menu2 = "Pilih..."
if "menu3" not in st.session_state:
    st.session_state.menu3 = "Pilih..."
if "menu4" not in st.session_state:
    st.session_state.menu4 = None
if "menu5" not in st.session_state:
    st.session_state.menu5 = None

# ============================
# SIDEBAR MENU
# ============================

st.sidebar.title("ANALISIS SENTIMEN")

# Menu 1: Data Awal
st.sidebar.markdown("### 📊 Data Awal")
if st.sidebar.button("Hasil Scraping"):
    st.session_state.menu1 = "Dataset Asli"
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Pilih..."
    st.session_state.menu4 = None
    st.session_state.menu5 = None
    st.rerun()


# Menu 2: Preprocessing
selected_menu2 = st.sidebar.selectbox("⚙️ Preprocessing", [
    "Pilih...", "Case Folding", "Cleaning", "Tokenizing",
    "Stopword Removal", "Stemming", "Normalisasi"
], index=["Pilih...", "Case Folding", "Cleaning", "Tokenizing",
          "Stopword Removal", "Stemming", "Normalisasi"].index(st.session_state.menu2))

if selected_menu2 != st.session_state.menu2:
    st.session_state.menu2 = selected_menu2
    st.session_state.menu1 = None
    st.session_state.menu3 = "Pilih..."
    st.session_state.menu4 = None
    st.session_state.menu5 = None

# Menu 3: Pemodelan
selected_menu3 = st.sidebar.selectbox("📈 Pemodelan", [
    "Pilih...", "Split Data", "Training", "Evaluasi"
], index=["Pilih...", "Split Data", "Training", "Evaluasi"].index(st.session_state.menu3))

if selected_menu3 != st.session_state.menu3:
    st.session_state.menu3 = selected_menu3
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu4 = None
    st.session_state.menu5 = None

# Menu 4: Visualisasi
st.sidebar.markdown("### 📊 Visualisasi")
if st.sidebar.button("Grafik Label"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Pilih..."
    st.session_state.menu4 = "Grafik Label"
    st.session_state.menu5 = None
if st.sidebar.button("Hasil Akurasi"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Pilih..."
    st.session_state.menu4 = "Hasil Akurasi"
    st.session_state.menu5 = None
if st.sidebar.button("Confusion Matrix"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Pilih..."
    st.session_state.menu4 = "Confusion Matrix"
    st.session_state.menu5 = None

# Menu 5: Pembahasan
st.sidebar.markdown("### 📝 Pembahasan")
if st.sidebar.button("Analisis Hasil"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Pilih..."
    st.session_state.menu4 = None
    st.session_state.menu5 = "Analisis Hasil"
if st.sidebar.button("Kesimpulan"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Pilih..."
    st.session_state.menu4 = None
    st.session_state.menu5 = "Kesimpulan"

# ============================
# FUNGSI BANTUAN
# ============================

def paginate_dataframe(df, rows_per_page=10):
    total_rows = len(df)
    total_pages = math.ceil(total_rows / rows_per_page)

    page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    paginated_df = df.iloc[start_idx:end_idx].copy()
    paginated_df.insert(0, 'No', range(start_idx + 1, min(end_idx, total_rows) + 1))

    return paginated_df, start_idx, end_idx, total_rows, page, total_pages

# ============================
# KONTEN UTAMA
# ============================

st.title("Analisis Sentimen Pada Aplikasi Wondr By BNI Menggunakan Metode IndoBERT")

# Menu 1: Data Awal
if st.session_state.menu1 == "Dataset Asli":
    st.subheader("📊 Data Awal: Hasil Scraping Google Play")
    paginated_data, start_idx, end_idx, total_rows, page, total_pages = paginate_dataframe(data)
    st.dataframe(paginated_data.set_index("No"), use_container_width=True)
    st.markdown(
        f"Menampilkan halaman **{page}** dari **{total_pages}** halaman | "
        f"Total data: **{total_rows}**"
    )

    st.markdown("### ℹ️ Informasi Dataset")
    fitur_keterangan = {
        "reviewId": "ID unik untuk setiap ulasan yang diberikan oleh pengguna.",
        "userName": "Nama pengguna yang memberikan ulasan terhadap aplikasi.",
        "userImage": "URL gambar profil pengguna (jika tersedia).",
        "content": "Isi ulasan yang dituliskan oleh pengguna.",
        "score": "Nilai rating dari pengguna terhadap aplikasi (1-5).",
        "thumbsUpCount": "Jumlah pengguna lain yang menyukai ulasan tersebut.",
        "reviewCreatedVersion": "Versi aplikasi yang digunakan saat pengguna menulis ulasan.",
        "at": "Tanggal ulasan dibuat oleh pengguna.",
        "replyContent": "Isi balasan dari pengembang terhadap ulasan pengguna (jika ada).",
        "repliedAt": "Tanggal balasan dari pengembang terhadap ulasan.",
        "appVersion": "Versi aplikasi saat ini ketika ulasan ditampilkan."
    }

    for kolom, deskripsi in fitur_keterangan.items():
        if kolom in data.columns:
            st.markdown(f"- **{kolom}**: {deskripsi}")

# Menu 2: Preprocessing
if st.session_state.menu2 == "Case Folding":
    st.subheader("⚙️ Preprocessing: Case Folding")

    try:
        data_sebelum = pd.read_csv("wondr_balanced.csv")
        data_sesudah = pd.read_csv("wondr_pp_casefolded.csv")

        data_sebelum = data_sebelum.loc[:, ~data_sebelum.columns.str.contains('^Unnamed')]
        data_sesudah = data_sesudah.loc[:, ~data_sesudah.columns.str.contains('^Unnamed')]

        if "content" in data_sebelum.columns and "content" in data_sesudah.columns:
            df_perbandingan = pd.DataFrame({
                "Sebelum": data_sebelum["content"],
                "Sesudah": data_sesudah["content"]
            })

            paginated_df, start_idx, end_idx, total_rows, page, total_pages = paginate_dataframe(df_perbandingan)
            st.dataframe(paginated_df.set_index("No"), use_container_width=True)
            st.markdown(
                f"Menampilkan halaman **{page}** dari **{total_pages}** halaman | "
                f"Total data: **{total_rows}**"
            )

            st.markdown("### ℹ️ Informasi Dataset Case Folding")
            st.markdown("Dataset ini telah melalui tahap *Case Folding*, yaitu proses mengubah seluruh huruf pada teks ulasan menjadi huruf kecil (lowercase).")
            st.markdown("Hal ini bertujuan untuk menyamakan representasi kata seperti 'Bagus' dan 'bagus' agar dihitung sebagai kata yang sama.")
            st.markdown("""
            - **Sebelum**: Isi ulasan asli sebelum dilakukan proses *Case Folding*.
            - **Sesudah**: Isi ulasan setelah diubah seluruh hurufnya menjadi huruf kecil.
            """)
        else:
            st.warning("Kolom 'content' tidak ditemukan di salah satu file.")
    except FileNotFoundError:
        st.error("File wondr_balanced.csv atau wondr_pp_casefolded.csv tidak ditemukan. Pastikan file tersedia.")

# Menu 3: Pemodelan
if st.session_state.menu3 != "Pilih...":
    st.subheader(f"📈 Pemodelan: {st.session_state.menu3}")
    st.write(f"Menampilkan hasil pemodelan: **{st.session_state.menu3}**.")

# Menu 4: Visualisasi
if st.session_state.menu4:
    st.subheader(f"📊 Visualisasi: {st.session_state.menu4}")
    st.write(f"Menampilkan visualisasi: **{st.session_state.menu4}**.")

# Menu 5: Pembahasan
if st.session_state.menu5:
    st.subheader(f"📝 Pembahasan: {st.session_state.menu5}")
    st.write(f"Menampilkan pembahasan: **{st.session_state.menu5}**.")

# ============================
# FOOTER
# ============================

st.markdown("---")
st.markdown("© 2025 Aisyah Wulandari. All rights reserved.")
