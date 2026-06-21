# Analisis Sentimen Ulasan Gojek

## Deskripsi

Proyek ini merupakan aplikasi analisis sentimen untuk ulasan aplikasi Gojek menggunakan pendekatan Machine Learning dan TF-IDF.

Sistem mengklasifikasikan ulasan pengguna ke dalam dua kategori:

* Positif
* Negatif

Model yang diuji dalam penelitian ini meliputi:

* Naive Bayes
* Support Vector Machine (SVM)
* Logistic Regression
* Random Forest

Selain proses pelatihan model, proyek ini juga menyediakan aplikasi berbasis Streamlit untuk melakukan prediksi sentimen secara interaktif.

---

## Dataset

Dataset yang digunakan dapat diunduh melalui tautan berikut:

**Dataset Link:**
(https://www.kaggle.com/datasets/ucupsedaya/gojek-app-reviews-bahasa-indonesia)

Pastikan file dataset disimpan dengan nama:

```text
gojek_reviews.csv
```

---

## Struktur Project

```text
Analisis-Sentiment-Gojek/
│
├── app.py
├── train_model.py
├── requirements.txt
├── gojek_reviews.csv
├── README.md
└── .gitignore
```

---

## Instalasi

Clone repository:

```bash
git clone https://github.com/SamuelKendrik/Analisis-Sentiment-Gojek.git
cd Analisis-Sentiment-Gojek
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Model

Untuk melatih seluruh model:

```bash
python train_model.py
```

Setelah proses training selesai, model dan TF-IDF vectorizer akan disimpan secara otomatis.

---

## Menjalankan Aplikasi

Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

Kemudian buka browser pada alamat yang ditampilkan oleh Streamlit.

---

## Preprocessing

Tahapan preprocessing yang digunakan:

1. Mengubah teks menjadi huruf kecil (lowercase)
2. Menghapus URL
3. Menghapus karakter non-alfabet
4. Menghapus spasi berlebih

---

## Feature Extraction

Feature extraction dilakukan menggunakan TF-IDF dengan konfigurasi:

* Maximum features: 5000
* N-gram range: (1, 2)
* Minimum document frequency: 2

---

## Evaluation Metrics

Model dievaluasi menggunakan:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Author

Samuel Kendrik

Binus University
