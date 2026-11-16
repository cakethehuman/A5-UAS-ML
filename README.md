# Cenayang Tulisan Tangan

Aplikasi klasifikasi tulisan tangan untuk mengenali identitas mahasiswa (A-NIM & Nama) menggunakan **Vision Transformer (DINOv2)**.

## Fitur

- Mengunggah citra tulisan tangan (PNG, JPG, JPEG)
- Mengenali identitas berdasarkan hasil klasifikasi model
- Menampilkan kode NIM dan nama mahasiswa
- Menampilkan tingkat keyakinan prediksi
- Menampilkan 5 probabilitas kelas tertinggi
- Menyaring prediksi dengan confidence di bawah 30%
- Memeriksa koneksi ke backend

## Teknologi

| Bagian | Teknologi |
|---|---|
| Frontend | Python, Streamlit, Pandas, Altair |
| Backend | Python, FastAPI, Uvicorn |
| Model | TensorFlow / Keras, DINOv2 |

## Instalasi dan Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/cakethehuman/A5-UAS-ML
```

### 2. Menjalankan Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend berjalan di `http://127.0.0.1:8000`.

### 3. Menjalankan Frontend

Buka terminal baru:

```bash
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
```

Frontend berjalan di `http://localhost:8501`.

## Struktur Data

File `nim_nama.txt` digunakan untuk mencocokkan indeks kelas model dengan identitas mahasiswa.

Urutan indeks harus sesuai dengan kelas yang digunakan saat pelatihan model.

## Alur Aplikasi

1. Pengguna mengunggah citra tulisan tangan.
2. Frontend mengirim citra ke backend melalui API.
3. Model melakukan klasifikasi terhadap citra.
4. Backend mengembalikan hasil prediksi dan probabilitas.
5. Sistem menampilkan identitas jika confidence memenuhi batas minimum 30%.
6. Sistem menampilkan 5 probabilitas kelas tertinggi.

© 2026 | UAS Machine Learning | Kelompok A5