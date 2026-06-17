# Deep Instance Segmentation

**Implementasi Instance Segmentation menggunakan Pretrained Mask R-CNN dan OpenCV DNN**

Proyek ini adalah aplikasi web berbasis Streamlit yang mendemonstrasikan kemampuan *Deep Instance Segmentation*. Aplikasi ini menggunakan model *pretrained* Mask R-CNN (Inception V2 backbone) yang dilatih pada dataset COCO untuk mendeteksi dan melakukan segmentasi piksel pada 90 kelas objek yang berbeda.

Proyek ini diajukan untuk memenuhi tugas mata kuliah **PACD (Pengolahan Analisis Citra Digital)**.

## Informasi Mahasiswa

- **Nama**: Lukas Austin
- **NPM**: 140810230011

## Sumber Referensi

Proyek ini dibangun dan dikembangkan berdasarkan referensi utama dari buku:
**"Python Image Processing Cookbook" (2020)** - *Chapter 6: Image Segmentation, Deep instance segmentation recipe*.

## Fitur Utama

- **Antarmuka Pengguna Interaktif**: Dibangun dengan Streamlit, menyediakan antarmuka yang bersih dan profesional.
- **Deteksi Otomatis (Mask R-CNN)**: Menggunakan arsitektur Mask R-CNN yang menghasilkan *bounding box*, label kelas, dan *mask* tingkat piksel (instance segmentation).
- **OpenCV DNN Module**: Inferensi model dijalankan menggunakan modul OpenCV DNN, tanpa memerlukan instalasi *heavy-framework* seperti TensorFlow atau PyTorch. Proses inferensi dapat berjalan efisien di CPU.
- **Visualisasi Alur Pemrosesan (Pipeline)**: Aplikasi ini secara transparan menampilkan tahapan proses dari gambar input, deteksi *bounding box* (sebelum *masking*), hingga output akhir yang telah diberi *mask* dan *blend* warna.
- **Informasi Objek Terdeteksi**: Menampilkan metrik utama seperti jumlah objek yang terdeteksi, jumlah kelas unik, waktu inferensi, dan tabel detail mengenai label beserta *confidence score*.

## Persyaratan Sistem (Prerequisites)

Pastikan Python telah terinstal di sistem Anda (direkomendasikan Python 3.8+). Anda memerlukan pustaka berikut:
- `numpy`
- `opencv-python`
- `matplotlib`
- `streamlit>=1.32`

## Instalasi dan Menjalankan Aplikasi

1. **Clone repositori atau unduh kode sumber.**
2. **Instal dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Siapkan Model Pretrained:**
   File model yang diperlukan untuk menjalankan aplikasi sudah tersedia di dalam direktori `models/`. Namun, jika Anda perlu mengunduh ulang file-file tersebut (karena file mentah `.tar.gz` tidak disertakan di repository), Anda dapat menjalankan skrip berikut:
   ```bash
   python utils/download_models.py
   ```
   Skrip ini secara otomatis akan mengunduh file model, mengekstrak *frozen inference graph* (`.pb`), mengunduh file konfigurasi (`.pbtxt`) beserta daftar kelas COCO, dan kemudian menghapus file *archive* `.tar.gz` untuk menghemat ruang.
4. **Jalankan Aplikasi:**
   ```bash
   streamlit run app.py
   ```
5. Buka tautan lokal yang diberikan oleh Streamlit di peramban web (biasanya `http://localhost:8501`).

## Konsep Utama: Instance vs Semantic Segmentation

- **Semantic Segmentation**: Mengklasifikasikan setiap piksel ke dalam kelas tertentu. Jika ada dua objek dari kelas yang sama berdekatan, mereka digabung sebagai satu *blob*.
- **Instance Segmentation**: Tidak hanya mengenali kelas, tapi membedakan *individu* dari kelas tersebut. Misalnya, jika ada 3 anjing, model dapat mengenali Anjing 1, Anjing 2, dan Anjing 3 sebagai instansi yang terpisah. Proyek ini mendemonstrasikan pendekatan ini.

## Struktur Direktori

```
deep-instance-segmentation/
├── app.py                      # File utama aplikasi Streamlit
├── requirements.txt            # Daftar pustaka Python yang dibutuhkan
├── README.md                   # Dokumentasi proyek (file ini)
├── images/                     # Direktori untuk gambar contoh (placeholder)
├── models/                     # Tempat model dan label yang telah diunduh
└── utils/                      # Modul utilitas
    ├── rcnn_segmentation.py    # Logika inferensi Mask R-CNN dan pemrosesan gambar
    └── download_models.py      # Skrip otomatis untuk mengunduh model OpenCV DNN
```
