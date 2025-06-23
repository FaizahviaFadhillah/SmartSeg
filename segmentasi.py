import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- MEMUAT DATASET CONTOH ---
try:
    df_sample = pd.read_csv("shopping_trends.csv")
except FileNotFoundError:
    st.error("File 'shopping_trends.csv' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
    df_sample = pd.DataFrame({
        'Customer ID': [], 'Age': [], 'Gender': [], 'Item Purchased': [], 'Category': [],
        'Purchase Amount (USD)': [], 'Location': [], 'Size': [], 'Color': [], 'Season': [],
        'Review Rating': [], 'Subscription Status': [], 'Payment Method': [],
        'Shipping Type': [], 'Discount Applied': [], 'Promo Code Used': [],
        'Previous Purchases': [], 'Preferred Payment Method': [], 'Frequency of Purchases': []
    })

# --- CSS STYLING ---
st.markdown("""
<style>
    /* Ganti warna background seluruh halaman */
    [data-testid="stAppViewContainer"] {
        background-color: #F0F8FF;
    }
    /* Ubah warna font default di seluruh halaman */
    [data-testid="stAppViewContainer"] {
        color: #1B2631 !important;
    }
    /* Ubah warna font sidebar */
    [data-testid="stSidebar"] {
        color: #154360 !important;
    }
    /* Ubah warna font label dan teks di uploader file */
    section[data-testid="stFileUploader"] label {
        color: #AED6F1 !important;
    }
    /* Opsional: warna background sidebar */
    [data-testid="stSidebar"] {
        background-color: #D6EAF8;
    }
    /* Supaya teks di header dan paragraf tidak putih */
    h1, h2, h3, p, label, div, span {
        color: #154360 !important;
    }
    h1, h3 {
        color: #2E86C1;
        text-align: center;
    }
    footer {
        text-align: center;
        color: #999;
        margin-top: 30px;
    }
    .cluster-info {
        background-color: #D6EAF8;
        border-left: 5px solid #2980B9;
        padding: 15px;
        margin-top: 20px;
        font-family: sans-serif;
        border-radius: 5px;
    }
    .interpretation-box {
        background-color: #EBF5FB;
        border: 1px solid #AED6F1;
        padding: 15px;
        margin-top: 10px;
        border-radius: 5px;
    }
    /* Gaya upload file custom */
    section[data-testid="stFileUploader"] > div {
        background-color: #1C2833;
        border: 2px dashed #3498DB;
        border-radius: 10px;
        padding: 20px;
        transition: 0.3s ease;
    }
    section[data-testid="stFileUploader"]:hover > div {
        border-color: #5DADE2;
        background-color: #212F3D;
    }
    section[data-testid="stFileUploader"] label {
        font-size: 16px;
        font-weight: bold;
        color: #AED6F1;
    }
    section[data-testid="stFileUploader"] svg {
        stroke: #5DADE2;
    }
</style>
""", unsafe_allow_html=True)

# --- JUDUL APLIKASI  ---
st.markdown("""
<style>
.app-title {
    font-size: 36px;
    font-weight: bold;
    color: #154360;
    text-align: center;
    background: linear-gradient(to right, #D6EAF8, #AED6F1);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    font-family: 'Segoe UI', sans-serif;
}
</style>

<div class='app-title'>
    SmartSeg: Aplikasi Segmentasi Pelanggan Cerdas Berbasis Machine Learning
</div>
""", unsafe_allow_html=True)


# --- TAB NAVIGASI ---
tab1, tab2, tab3 = st.tabs(["📊 Segmentasi", "📂 Dataset", "ℹ️ Tentang"])

with tab1:
    st.header("Segmentasi Pelanggan")

    uploaded_file = st.file_uploader("Upload dataset CSV pelanggan:", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("**Menampilkan analisis menggunakan contoh dataset 'shopping_trends.csv'. Anda dapat mengunggah file Anda sendiri di atas.**")
        df = df_sample.copy()

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Dataset harus memiliki minimal 2 kolom numerik untuk clustering.")
    else:
        # Bagian pemilihan fitur dan algoritma 
        default_index_col1 = 0
        default_index_col2 = 1 if len(numeric_cols) > 1 else 0
        if 'Purchase Amount (USD)' in numeric_cols: default_index_col1 = numeric_cols.index('Purchase Amount (USD)')
        if 'Review Rating' in numeric_cols: default_index_col2 = numeric_cols.index('Review Rating')
        if default_index_col1 == default_index_col2: default_index_col2 = (default_index_col1 + 1) % len(numeric_cols)
        col1 = st.selectbox("Pilih fitur X (sumbu horizontal)", options=numeric_cols, index=default_index_col1)
        available_cols_for_col2 = [c for c in numeric_cols if c != col1]
        
        if not available_cols_for_col2:
            st.error("Tidak ada kolom lain selain kolom X yang bisa dipilih untuk fitur Y.")
        else:
            default_index_col2_in_available = 0
            if 'Review Rating' in available_cols_for_col2: default_index_col2_in_available = available_cols_for_col2.index('Review Rating')
            col2 = st.selectbox("Pilih fitur Y (sumbu vertikal)", options=available_cols_for_col2, index=default_index_col2_in_available)
            X = df[[col1, col2]].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            algo = st.selectbox("Pilih algoritma segmentasi", ["K-Means", "DBSCAN", "Hierarchical Clustering"])

            # Logika clustering 
            if algo == "K-Means":
                n_clusters = st.slider("Jumlah cluster", 2, 10, 3)
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
            elif algo == "DBSCAN":
                eps = st.slider("Nilai eps (radius)", 0.1, 5.0, 0.5)
                min_samples = st.slider("Minimal samples", 3, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
            else:
                n_clusters = st.slider("Jumlah cluster", 2, 10, 3)
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_scaled)
            
            # Visualisasi 
            fig, ax = plt.subplots(figsize=(8, 5))
            if algo == "DBSCAN":
                unique_labels_for_palette = np.unique(labels)
                palette_size = len(unique_labels_for_palette) - (1 if -1 in unique_labels_for_palette else 0)
                palette = sns.color_palette("Set2", palette_size)
                colors = [palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in labels]
                ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors)
            else:
                sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette="Set2", ax=ax, legend='full')
            ax.set_title(f"Hasil Segmentasi {algo}")
            ax.set_xlabel(f"Scaled {col1}")
            ax.set_ylabel(f"Scaled {col2}")
            st.pyplot(fig)

            df["Cluster"] = labels
            
            # --- ANALISIS & INTERPRETASI CLUSTER ---

            st.markdown("<h3>Analisis & Interpretasi Cluster</h3>", unsafe_allow_html=True)

            # Hanya proses cluster yang valid (bukan noise -1 dari DBSCAN)
            valid_clusters = sorted([c for c in df['Cluster'].unique() if c != -1])
            
            if not valid_clusters:
                st.warning("Tidak ada cluster yang terbentuk. Coba sesuaikan parameter DBSCAN.")
            else:
                # Pilih kolom numerik yang relevan untuk dianalisis
                cols_to_analyze = [col for col in ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases'] if col in df.columns]

                # Hitung rata-rata untuk setiap cluster
                cluster_summary = df.groupby('Cluster')[cols_to_analyze].mean().round(2)
                
                # Tampilkan tabel ringkasan
                st.subheader("Ringkasan Karakteristik Cluster")
                st.markdown("Tabel berikut menunjukkan nilai rata-rata dari fitur numerik utama untuk setiap cluster yang terbentuk.")
                st.dataframe(cluster_summary.loc[valid_clusters])

                # Buat interpretasi dinamis
                st.subheader("Persona Setiap Cluster")
                st.markdown("Berikut adalah deskripsi persona atau karakteristik dari setiap kelompok pelanggan:")
                
                for i in valid_clusters:
                    summary = cluster_summary.loc[i]
                    # Deskripsi berdasarkan fitur yang dipilih untuk clustering (col1 dan col2)
                    desc_col1_val = summary[col1]
                    desc_col2_val = summary[col2]
                    
                    # Membandingkan dengan rata-rata keseluruhan untuk memberikan konteks
                    overall_mean_col1 = df[col1].mean()
                    overall_mean_col2 = df[col2].mean()

                    pos_col1 = "di atas rata-rata" if desc_col1_val > overall_mean_col1 else "di bawah rata-rata"
                    pos_col2 = "di atas rata-rata" if desc_col2_val > overall_mean_col2 else "di bawah rata-rata"

                    # Membuat deskripsi persona
                    persona_desc = f"""
                    <div class='interpretation-box'>
                    <b>Cluster {i}:</b>
                    <ul>
                        <li>Kelompok ini dicirikan oleh <b>{col1}</b> yang <b>{pos_col1}</b> (rata-rata: {desc_col1_val:.2f}).</li>
                        <li>Mereka juga memiliki <b>{col2}</b> yang <b>{pos_col2}</b> (rata-rata: {desc_col2_val:.2f}).</li>
                    </ul>
                    <b>Saran:</b> Pelanggan di segmen ini mungkin merespon baik terhadap... (Contoh: penawaran produk premium jika belanja tinggi, atau program loyalitas jika rating tinggi).
                    </div>
                    """
                    st.markdown(persona_desc, unsafe_allow_html=True)
            
            # Tampilkan hasil cluster di tabel akhir
            st.subheader("Hasil Data + Cluster")
            st.dataframe(df)

            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='openpyxl')
            st.download_button("📥 Download hasil ke Excel", data=buffer.getvalue(), file_name="hasil_segmentasi.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab2:
    st.header("📂 Contoh Dataset")
    st.markdown("""
    Anda dapat menggunakan contoh dataset di bawah ini untuk mencoba aplikasi. 
    Dataset ini berisi informasi tren belanja pelanggan.
    """)
    if not df_sample.empty:
        st.dataframe(df_sample)
        st.download_button(
           label="📥 Download Contoh Dataset (CSV)",
           data=df_sample.to_csv(index=False).encode('utf-8'),
           file_name='shopping_trends.csv',
           mime='text/csv',
        )
        st.markdown("---") # Garis pemisah
        st.subheader("Keterangan Kolom")
        # Deskripsi kolom disesuaikan dengan dataset shopping_trends.csv
        st.markdown("""
        - **Customer ID**: ID unik untuk setiap pelanggan.
        - **Age**: Usia pelanggan.
        - **Gender**: Jenis kelamin pelanggan.
        - **Item Purchased**: Barang yang dibeli.
        - **Category**: Kategori barang yang dibeli.
        - **Purchase Amount (USD)**: Jumlah pembelian dalam USD.
        - **Location**: Lokasi pelanggan.
        - **Size**: Ukuran barang.
        - **Color**: Warna barang.
        - **Season**: Musim saat pembelian dilakukan.
        - **Review Rating**: Rating ulasan yang diberikan pelanggan (1-5).
        - **Previous Purchases**: Jumlah pembelian sebelumnya.
        - **Frequency of Purchases**: Seberapa sering pelanggan melakukan pembelian.
        """)
    else:
        st.warning("Gagal memuat contoh dataset. Silakan unggah dataset Anda sendiri di tab 'Segmentasi'.")

with tab3:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk membantu bisnis dalam melakukan *segmentasi pelanggan* dengan berbagai algoritma machine learning:
    | Algoritma                  | Tujuan                                                        |
    |----------------------------|---------------------------------------------------------------|
    | *K-Means* | Mengelompokkan pelanggan berdasarkan kemiripan data numerik.  |
    | *DBSCAN* | Mengelompokkan berdasarkan kepadatan, cocok untuk data tidak beraturan. |
    | *Hierarchical Clustering* | Mengelompokkan secara bertingkat, ideal untuk visualisasi hirarki. |
    
    Anggota Kelompok:*- Faizah Via Fadhillah*
                     *- Anggita Risqi Nur Clarita*
                     *- Zahra Nurhaliza*""")

st.markdown("<footer>© 2025 - Segmentasi Pelanggan App by Kelompok 1</footer>", unsafe_allow_html=True)
