import streamlit as st            # Untuk membuat antarmuka web interaktif
import pandas as pd               # Untuk manipulasi dan analisis data
import math                       # Untuk perhitungan matematika (mis. pembagian halaman)
import plotly.express as px       # Untuk membuat diagram batang interaktif (plotly)
from wordcloud import WordCloud   # Untuk membentuk Word Cloud dari teks
import matplotlib.pyplot as plt   # Untuk menampilkan Word Cloud dengan Matplotlib


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
    st.session_state.menu3 = None
if "menu4" not in st.session_state:
    st.session_state.menu4 = None

# ============================
# SIDEBAR MENU
# ============================

st.sidebar.title("ANALISIS SENTIMEN")

# Menu 1: Data Awal
st.sidebar.markdown("### 📊 Data Awal")
if st.sidebar.button("Hasil Scraping"):
    st.session_state.menu1 = "Dataset Asli"
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = None
    st.session_state.menu4 = None
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
    st.session_state.menu3 = None
    st.session_state.menu4 = None
    st.rerun()

# Menu 3: Visualisasi
st.sidebar.markdown("### 📊 Visualisasi")
if st.sidebar.button("Hasil Klasifikasi Sentimen"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Hasil Klasifikasi Sentimen"
    st.session_state.menu4 = None
if st.sidebar.button("Diagram Batang"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Diagram Batang"
    st.session_state.menu4 = None
if st.sidebar.button("Word Cloud"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = "Word Cloud"
    st.session_state.menu4 = None

# Menu 4: Pembahasan
st.sidebar.markdown("### 📝 Pembahasan")
if st.sidebar.button("Analisis Hasil"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = None
    st.session_state.menu4 = "Analisis Hasil"
if st.sidebar.button("Kesimpulan"):
    st.session_state.menu1 = None
    st.session_state.menu2 = "Pilih..."
    st.session_state.menu3 = None
    st.session_state.menu4 = "Kesimpulan"

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

elif st.session_state.menu2 == "Cleaning":
    st.subheader("⚙️ Preprocessing: Cleaning")

    try:
        data_sebelum = pd.read_csv("wondr_pp_casefolded.csv")
        data_sesudah = pd.read_csv("wondr_pp_cleaned.csv")

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

            st.markdown("### ℹ️ Informasi Dataset Cleaning")
            st.markdown("Dataset ini telah melalui tahap *Cleaning*, yaitu proses pembersihan teks dari elemen-elemen yang tidak relevan dalam analisis sentimen. Tujuan dari proses ini adalah untuk meningkatkan kualitas data sebelum dilakukan pemrosesan lanjutan seperti tokenisasi atau stemming.")
            st.markdown("""
            - **Sebelum**: Teks ulasan setelah dilakukan *Case Folding* (huruf kecil semua).
            - **Sesudah**: Teks ulasan setelah dibersihkan dari elemen-elemen yang tidak relevan.

            Proses pembersihan meliputi:
            - Menghapus tag HTML: `re.sub(r'<.*?>', '', text)`
            - Menghapus URL: `re.sub(r'http\\S+|www\\S+', '', text)`
            - Menghapus angka: `re.sub(r'\\d+', '', text)`
            - Menghapus tanda baca: `re.sub(r'[^\\w\\s]', '', text)`
            - Menghapus emotikon dan karakter non-ASCII: `re.sub(r'[^\\x00-\\x7f]', '', text)`
            - Menghapus karakter berulang (lebih dari 2x): `re.sub(r'(.)\\1{2,}', r'\\1', text)`
            - Menghapus spasi berlebih: `re.sub(r'\\s+', ' ', text).strip()`
            - Menyisipkan spasi setelah dan sebelum tanda baca yang langsung menempel ke kata:  
              `re.sub(r"([.,;:!?()\\[\\]{}\\"'/])(\\w)", r"\\1 \\2", text)`  
              `re.sub(r"(\\w)([.,;:!?()\\[\\]{}\\"'/])", r"\\1 \\2", text)`
            - Menghapus karakter non-alfanumerik selain spasi: `re.sub(r"[^a-zA-Z0-9\\s]", "", text)`
            """)
        else:
            st.warning("Kolom 'content' tidak ditemukan di salah satu file.")
    except FileNotFoundError:
        st.error("File wondr_pp_casefolded.csv atau wondr_pp_cleaned.csv tidak ditemukan. Pastikan file tersedia.")

elif st.session_state.menu2 == "Tokenizing":
    st.subheader("⚙️ Preprocessing: Tokenizing")

    try:
        data_sebelum = pd.read_csv("wondr_pp_cleaned.csv")
        data_sesudah = pd.read_csv("wondr_pp_tokenized.csv")

        data_sebelum = data_sebelum.loc[:, ~data_sebelum.columns.str.contains('^Unnamed')]
        data_sesudah = data_sesudah.loc[:, ~data_sesudah.columns.str.contains('^Unnamed')]

        if "content" in data_sebelum.columns and "tokenisasi" in data_sesudah.columns:
            df_perbandingan = pd.DataFrame({
                "Sebelum": data_sebelum["content"],
                "Sesudah": data_sesudah["tokenisasi"]
            })

            paginated_df, start_idx, end_idx, total_rows, page, total_pages = paginate_dataframe(df_perbandingan)
            st.dataframe(paginated_df.set_index("No"), use_container_width=True)
            st.markdown(
                f"Menampilkan halaman **{page}** dari **{total_pages}** halaman | "
                f"Total data: **{total_rows}**"
            )

            st.markdown("### ℹ️ Informasi Dataset Tokenizing")
            st.markdown("Dataset ini telah melalui tahap *Tokenizing*, yaitu proses memecah teks menjadi unit yang lebih kecil seperti kata atau token.")
            st.markdown("Tokenizing sangat penting dalam analisis teks karena memungkinkan sistem memproses setiap kata secara individual. Tokenizing dilakukan dengan `word_tokenize` dari libarary NLTK.")
            st.markdown("""
            - **Sebelum**: Isi ulasan setelah tahap *Cleaning* (teks bersih dari noise).
            - **Sesudah**: Hasil pemecahan setiap ulasan menjadi token/kata.

            """)
        else:
            st.warning("Kolom 'content' tidak ditemukan di salah satu file.")
    except FileNotFoundError:
        st.error("File wondr_pp_cleaned.csv atau wondr_pp_tokenized.csv tidak ditemukan. Pastikan file tersedia.")

elif st.session_state.menu2 == "Stopword Removal":
    st.subheader("⚙️ Preprocessing: Stopword Removal")

    try:
        data_sebelum = pd.read_csv("wondr_pp_tokenized.csv")
        data_sesudah = pd.read_csv("wondr_pp_stopwords_removed.csv")

        # Hilangkan kolom Unnamed jika ada
        data_sebelum = data_sebelum.loc[:, ~data_sebelum.columns.str.contains('^Unnamed')]
        data_sesudah = data_sesudah.loc[:, ~data_sesudah.columns.str.contains('^Unnamed')]

        if "tokenisasi" in data_sebelum.columns and "stopwords_removal" in data_sesudah.columns:
            df_perbandingan = pd.DataFrame({
                "Sebelum": data_sebelum["tokenisasi"],
                "Sesudah": data_sesudah["stopwords_removal"]
            })

            paginated_df, start_idx, end_idx, total_rows, page, total_pages = paginate_dataframe(df_perbandingan)
            st.dataframe(paginated_df.set_index("No"), use_container_width=True)
            st.markdown(
                f"Menampilkan halaman **{page}** dari **{total_pages}** halaman | "
                f"Total data: **{total_rows}**"
            )

            st.markdown("### ℹ️ Informasi Dataset Stopword Removal")
            st.markdown("Dataset ini telah melalui tahap *Stopword Removal*, yaitu proses penghapusan kata-kata umum yang tidak memiliki makna signifikan dalam analisis sentimen.")
            st.markdown("Proses ini menggunakan daftar stopwords dari pustaka `nltk` dalam Bahasa Indonesia yang kemudian ditambahkan dengan stopwords informal khas ulasan aplikasi di Indonesia.")
            st.markdown("""
            - **Sebelum**: Hasil tokenisasi yang masih mengandung kata-kata umum (*stopwords*) seperti `yang`, `saya`, `dan`, `gk`, `yg`, dll.  
            - **Sesudah**: Token yang telah dibersihkan dari kata-kata tidak penting, hanya menyisakan kata-kata bermakna untuk analisis.

            Metode dalam tahap ini:
            - Menggunakan `nltk.corpus.stopwords` Bahasa Indonesia.
            - Penambahan stopwords informal seperti `yg`, `nya`, `gk`, `aja`, `ok`, `gw`, dll.
            - Stopwords dikumpulkan ke dalam satu set dan digunakan untuk memfilter setiap token.
            """)
        else:
            st.warning("Kolom 'tokenisasi' atau 'stopwords_removal' tidak ditemukan di salah satu file.")
    except FileNotFoundError:
        st.error("File wondr_pp_tokenized.csv atau wondr_pp_stopwords_removed.csv tidak ditemukan. Pastikan file tersedia.")
        
elif st.session_state.menu2 == "Stemming":
    st.subheader("⚙️ Preprocessing: Stemming")

    try:
        data_sebelum = pd.read_csv("wondr_pp_stopwords_removed.csv")
        data_sesudah = pd.read_csv("wondr_pp_stemmed.csv")

        # Hilangkan kolom Unnamed
        data_sebelum = data_sebelum.loc[:, ~data_sebelum.columns.str.contains('^Unnamed')]
        data_sesudah = data_sesudah.loc[:, ~data_sesudah.columns.str.contains('^Unnamed')]

        if "stopwords_removal" in data_sebelum.columns and "stemmed" in data_sesudah.columns:
            df_perbandingan = pd.DataFrame({
                "Sebelum": data_sebelum["stopwords_removal"],
                "Sesudah": data_sesudah["stemmed"]
            })

            paginated_df, start_idx, end_idx, total_rows, page, total_pages = paginate_dataframe(df_perbandingan)
            st.dataframe(paginated_df.set_index("No"), use_container_width=True)
            st.markdown(
                f"Menampilkan halaman **{page}** dari **{total_pages}** halaman | "
                f"Total data: **{total_rows}**"
            )

            st.markdown("### ℹ️ Informasi Dataset Stemming")
            st.markdown("Dataset ini telah melalui tahap *Stemming*, yaitu proses mengubah kata menjadi bentuk dasarnya (*root word*) agar analisis menjadi lebih efisien dan seragam.")
            st.markdown("Stemming sangat berguna dalam NLP untuk menyederhanakan berbagai bentuk kata menjadi satu representasi umum.")
            st.markdown("""
            - **Sebelum**: Token hasil *Stopword Removal* yang masih dalam bentuk kata turunan. Contohnya: `bermain`, `makanan`, `diberikan`.
            - **Sesudah**: Kata-kata yang telah direduksi ke bentuk dasarnya. Contohnya: `main`, `makan`, `beri`.

            Tahap stemming ini menggunakan library [**Sastrawi**](https://github.com/har07/PySastrawi), yaitu library stemming Bahasa Indonesia yang telah banyak digunakan pada aplikasi teks lokal.
            """)
        else:
            st.warning("Kolom 'stopwords_removal' atau 'stemmed' tidak ditemukan di file.")
    except FileNotFoundError:
        st.error("File wondr_pp_stopwords_removed.csv atau wondr_pp_stemmed.csv tidak ditemukan. Pastikan file tersedia.")
        
elif st.session_state.menu2 == "Normalisasi":
    st.subheader("⚙️ Preprocessing: Normalisasi")

    try:
        data_sebelum = pd.read_csv("wondr_pp_stemmed.csv")
        data_sesudah = pd.read_csv("wondr_pp_normalized.csv")

        # Hilangkan kolom Unnamed jika ada
        data_sebelum = data_sebelum.loc[:, ~data_sebelum.columns.str.contains('^Unnamed')]
        data_sesudah = data_sesudah.loc[:, ~data_sesudah.columns.str.contains('^Unnamed')]

        if "stemmed" in data_sebelum.columns and "normalized" in data_sesudah.columns:
            df_perbandingan = pd.DataFrame({
                "Sebelum": data_sebelum["stemmed"],
                "Sesudah": data_sesudah["normalized"]
            })

            paginated_df, start_idx, end_idx, total_rows, page, total_pages = paginate_dataframe(df_perbandingan)
            st.dataframe(paginated_df.set_index("No"), use_container_width=True)
            st.markdown(
                f"Menampilkan halaman **{page}** dari **{total_pages}** halaman | "
                f"Total data: **{total_rows}**"
            )

            st.markdown("### ℹ️ Informasi Dataset Normalisasi")
            st.markdown("""
            Dataset ini telah melalui tahap *Normalisasi*, yaitu proses mengganti kata-kata tidak baku atau slang ke dalam bentuk formal/baku sesuai Bahasa Indonesia.

            Proses normalisasi dilakukan dengan mencocokkan kata-kata dalam dokumen dengan kamus kolokial Indonesia yang berasal dari file:
            **`colloquial-indonesian-lexicon.csv`**.
            
            - **Sebelum**: Token hasil *stemming* yang masih mengandung kata slang.
            - **Sesudah**: Token telah digantikan dengan bentuk formal jika tersedia dalam kamus kolokial.

            Kamus ini memuat pasangan kata:
            - Kolom `slang`: berisi kata tidak baku atau kata gaul (contoh: `gk`, `bgt`, `trs`)
            - Kolom `formal`: berisi bentuk baku dari kata tersebut (contoh: `tidak`, `banget`, `terus`)
            """)
        else:
            st.warning("Kolom 'stemmed' atau 'normalized' tidak ditemukan di file.")
    except FileNotFoundError:
        st.error("File wondr_pp_stemmed.csv atau wondr_pp_normalized.csv tidak ditemukan. Pastikan file tersedia.")

# Menu 3: Visualisasi
if st.session_state.menu3 == "Hasil Klasifikasi Sentimen":
    st.subheader("📊 Visualisasi: Hasil Klasifikasi Sentimen")

    try:
        df = pd.read_csv("wondr_test_result_70_30_16_3e6.csv")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if "review_text" in df.columns and "pred" in df.columns:
            df_display = pd.DataFrame({
                "Ulasan": df["review_text"],
                "Kategori": df["pred"]
            })

            paginated_df, start_idx, end_idx, total_rows, page, total_pages = paginate_dataframe(df_display)
            st.dataframe(paginated_df.set_index("No"), use_container_width=True)

            st.markdown(
                f"Menampilkan halaman **{page}** dari **{total_pages}** halaman | "
                f"Total data: **{total_rows}**"
            )

            st.markdown("### ℹ️ Informasi Hasil Klasifikasi")
            st.markdown("""
                Tabel di atas menampilkan hasil klasifikasi sentimen berdasarkan model **IndoBERT** yang telah dilatih sebelumnya:

                - **Ulasan**: Merupakan kalimat atau teks dari pengguna aplikasi *Wondr*.
                - **Kategori**: Hasil prediksi model terhadap masing-masing ulasan, dikategorikan menjadi:
                - 🟢 `positive`
                - 🟡 `neutral`
                - 🔴 `negative`

                Hasil ini berguna untuk mengevaluasi performa model dalam memahami konteks dan emosi dari ulasan pengguna.

            """)

        else:
            st.warning("Kolom 'ulasan' atau 'prediksi' tidak ditemukan dalam file wondr_test_result.csv")
    except FileNotFoundError:
        st.error("File wondr_test_result.csv tidak ditemukan. Pastikan file tersedia.")


elif st.session_state.menu3 == "Diagram Batang":
    st.subheader("📊 Visualisasi: Hasil Klasifikasi Sentimen")

    try:
        df = pd.read_csv("wondr_labeled.csv")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if "category" in df.columns:
            sentiment_counts = df["category"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentimen", "Jumlah"]

            fig = px.bar(
                sentiment_counts,
                x="Sentimen",
                y="Jumlah",
                color="Sentimen",
                color_discrete_map={
                    "positive": "#28a745",
                    "neutral": "#ffc107",
                    "negative": "#dc3545"
                },
                title="Distribusi Klasifikasi Sentimen pada Ulasan Aplikasi Wondr",
                labels={"Sentimen": "Kategori Sentimen", "Jumlah": "Jumlah Ulasan"},
                text_auto=True
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### ℹ️ Keterangan")
            st.markdown("""
            Diagram batang di atas menunjukkan jumlah ulasan untuk masing-masing kategori sentimen:
            - 🟢 **Positif**: ulasan dengan skor > 3.
            - 🟡 **Netral**: ulasan dengan skor = 3.
            - 🔴 **Negatif**: ulasan dengan skor < 3.

            """)
        else:
            st.warning("Kolom 'category' tidak ditemukan dalam file wondr_labeled.csv")
    except FileNotFoundError:
        st.error("File wondr_labeled.csv tidak ditemukan. Pastikan file tersedia.")


elif st.session_state.menu3 == "Word Cloud":
    st.subheader("📊 Visualisasi: Word Cloud per Sentimen")

    try:
        df = pd.read_csv("wondr_pp_normalized.csv")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if "normalized" in df.columns and "category" in df.columns:
            # Warna berbeda untuk setiap sentimen
            color_map_dict = {
                "positive": "Greens",
                "neutral": "Blues",
                "negative": "Reds"
            }

            # Buat WordCloud untuk setiap sentimen
            for sentiment_label, sentiment_name in [("positive", "Positif"), ("neutral", "Netral"), ("negative", "Negatif")]:
                st.markdown(f"### 💬 Word Cloud Sentimen **{sentiment_name}**")

                # Filter data berdasarkan kategori
                text_data = df[df["category"] == sentiment_label]["normalized"].dropna().astype(str)

                # Gabungkan menjadi satu string
                text = " ".join(text_data)

                if not text.strip():
                    st.info(f"Tidak ada data untuk sentimen **{sentiment_name}**.")
                    continue

                # Buat dan tampilkan WordCloud dengan colormap sesuai sentimen
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap=color_map_dict[sentiment_label],  # pakai warna sesuai sentimen
                    max_words=200,
                    contour_width=1,
                    contour_color='gray'
                ).generate(text)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

            st.markdown("### ℹ️ Informasi Word Cloud")
            st.markdown("""
            Word Cloud di atas menunjukkan kata-kata yang paling sering muncul untuk masing-masing kategori sentimen:
            - 🟢 **Positif**: Ditampilkan dengan warna hijau (colormap `Greens`).
            - 🔵 **Netral**: Ditampilkan dengan warna biru (colormap `Blues`).
            - 🔴 **Negatif**: Ditampilkan dengan warna merah (colormap `Reds`).

            Warna membantu memperjelas nuansa emosi dalam visualisasi Word Cloud.
            """)
        else:
            st.warning("Kolom 'normalized' atau 'category' tidak ditemukan dalam file wondr_pp_normalized.csv.")
    except FileNotFoundError:
        st.error("File wondr_pp_normalized.csv tidak ditemukan. Pastikan file tersedia.")

# Menu 4: Analisis Hasil dan Kesimpulan
if st.session_state.menu4 == "Analisis Hasil":
    st.subheader("📝 Analisis Hasil")

    st.markdown("""
    Pada proyek ini, dilakukan analisis sentimen terhadap ulasan pengguna aplikasi **Wondr by BNI** menggunakan pendekatan berbasis **Deep Learning IndoBERT**. Proses dimulai dari pengumpulan data, dilanjutkan dengan tahap-tahap *preprocessing*, hingga pemodelan dan evaluasi model.
    
    #### 🔄 Preprocessing
    Tahapan preprocessing bertujuan untuk membersihkan dan menyiapkan data teks agar dapat digunakan oleh model. Tahapan meliputi:
    - **Case Folding**: Mengubah seluruh huruf menjadi huruf kecil.
    - **Cleaning**: Menghapus HTML, URL, angka, tanda baca, karakter non-alfabet, dan karakter berulang.
    - **Tokenizing**: Memecah teks menjadi token atau kata-kata.
    - **Stopword Removal**: Menghapus kata-kata tidak penting (seperti "yang", "dan", "saya").
    - **Stemming**: Mengubah kata ke bentuk dasarnya.
    - **Normalisasi**: Mengganti kata tidak baku/slang menjadi kata formal berdasarkan kamus kolokial.

    #### 📊 Visualisasi
    Visualisasi dilakukan melalui:
    - **Diagram Batang**: Untuk melihat distribusi data sentimen.
    - **Word Cloud**: Untuk masing-masing sentimen (`positive`, `neutral`, `negative`) yang menggambarkan kata-kata yang paling sering muncul.

    **Hasil Word Cloud menunjukkan**:
    - **Sentimen Positif**: Didominasi kata seperti *mudah*, *bagus*, *transaksi*, *fitur*, dan *praktis*, menandakan kepuasan pengguna terhadap fitur dan kemudahan penggunaan aplikasi.
    - **Sentimen Netral**: Kata seperti *mobile*, *apk*, *verifikasi*, dan *token* mendominasi. Menandakan pengalaman pengguna yang biasa saja atau netral, namun tetap mengandung masukan tentang fungsi login dan akses.
    - **Sentimen Negatif**: Kata seperti *gagal*, *masuk*, *enggak*, *verifikasi*, dan *saldo* mendominasi. Ini menunjukkan bahwa keluhan utama pengguna berpusat pada kegagalan login/verifikasi serta masalah saldo/transaksi.

    #### 📈 Evaluasi
    - Hasil klasifikasi ditampilkan dalam bentuk tabel yang berisi prediksi sentimen untuk setiap ulasan pengguna.
    - Evaluasi ini membantu menilai sejauh mana model memahami konteks ulasan dan mampu mengkategorikan dengan akurat.
    """)

elif st.session_state.menu4 == "Kesimpulan":
    st.subheader("📝 Kesimpulan")

    st.markdown("""
    Berdasarkan analisis dan implementasi yang telah dilakukan, dapat disimpulkan beberapa hal penting berikut:

    1. **Proses preprocessing** berperan penting dalam meningkatkan kualitas data teks.
    2. **IndoBERT** melakukan klasifikasi sentimen dengan baik terhadap ulasan berbahasa Indonesia, termasuk dalam konteks informal seperti ulasan aplikasi mobile.
    3. Visualisasi menggunakan **Word Cloud** dan **Diagram Batang** membantu dalam memahami sentimen secara cepat dan intuitif.
    4. Berdasarkan hasil Word Cloud:
       - **Pengguna merasa puas** terhadap kemudahan penggunaan aplikasi, kelengkapan fitur, dan kecepatan transaksi.
       - **Isu yang sering muncul** dalam ulasan netral hingga negatif adalah tentang proses verifikasi (wajah/email), kendala login, serta masalah saldo/transaksi.
       - **Rekomendasi**: Pemilik aplikasi disarankan untuk fokus pada perbaikan dan optimasi sistem login dan verifikasi agar pengalaman pengguna lebih lancar dan minim gangguan.
    """)

# ============================
# FOOTER
# ============================

st.markdown("---")
st.markdown("© 2025 Aisyah Wulandari. All rights reserved.")
