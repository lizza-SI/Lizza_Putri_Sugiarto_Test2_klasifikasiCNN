# Oceanic Life CNN

## Klasifikasi Citra Organisme Laut Menggunakan CNN MobileNetV2

Oceanic Life CNN merupakan proyek klasifikasi citra yang menggunakan
Convolutional Neural Network (CNN) dengan arsitektur MobileNetV2
untuk mengenali gambar organisme laut ke dalam tujuh kategori.

Proyek ini mencakup pengolahan dataset, pelatihan model,
evaluasi performa, serta implementasi dashboard menggunakan Streamlit.

## 1. Kategori Klasifikasi

Model mengenali tujuh kategori organisme laut:

1. Coral
2. Crustaceans
3. Fish
4. Octopuses
5. Other marine organisms
6. Shark
7. Sponges

## 2. Dataset

Dataset diperoleh dari Kaggle:

https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset

Dataset akhir yang digunakan terdiri dari 840 gambar unik.

Pembagian dataset:
- Training: 588 gambar (70%)
- Validation: 126 gambar (15%)
- Testing: 126 gambar (15%)

Dataset tidak disertakan dalam repository. Pengguna dapat
mengunduhnya melalui tautan Kaggle di atas.

## 3. Arsitektur dan Pelatihan Model

Model menggunakan MobileNetV2 dengan pretrained weights ImageNet.

Proses pelatihan dilakukan melalui dua tahap:

- Stage 1: Transfer Learning
- Stage 2: Fine-Tuning

Ukuran input gambar adalah 224 x 224 piksel dengan tujuh
neuron keluaran menggunakan fungsi aktivasi Softmax.

## 4. Hasil Evaluasi Model

Hasil evaluasi validation menggunakan 126 gambar:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 81.75% |
| Macro Precision | 0.7908 |
| Macro Recall | 0.7280 |
| Macro F1-score | 0.7279 |
| Weighted F1-score | 0.8132 |

Jumlah prediksi benar: 103 gambar.
Jumlah prediksi salah: 23 gambar.

## 5. Struktur Repository

```text
OceanicLife_GitHub/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── Laporan_Praktikum_Oceanic_Life.docx
│
├── notebook/
│   └── Oceanic_Life_CNN_Notebook_Final.ipynb
│
├── output_final/
│   └── models/
│       └── mobilenetv2_stage2_best.keras
│
└── hasil_eksperimen/
    ├── OceanicLife_Final_Confusion_Matrix.png
    ├── OceanicLife_Final_Classification_Report.csv
    ├── OceanicLife_Final_Test_Performance.csv
    ├── OceanicLife_Stage1_Training_History.csv
    ├── OceanicLife_Stage2_Training_History.csv
    └── OceanicLife_Class_Distribution_Deduplicated.png
```

## 6. Instalasi

Pastikan Python sudah terinstal.

Clone repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

Buat virtual environment:

```bash
python -m venv venv
```

Aktifkan virtual environment pada Windows:

```bash
venv\Scripts\activate
```

Instal library:

```bash
pip install -r requirements.txt
```

## 7. Menjalankan Dashboard

Jalankan perintah:

```bash
streamlit run app.py
```

Dashboard akan menampilkan aplikasi klasifikasi citra organisme laut.

Pengguna dapat mengunggah gambar berformat JPG, JPEG,
atau PNG untuk mendapatkan hasil prediksi model.

## 8. Teknologi yang Digunakan

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit
- Jupyter Notebook

## 9. Dokumentasi

Notebook, laporan praktikum, model final, dan hasil evaluasi
tersedia di dalam repository ini.

Proyek ini dibuat untuk memenuhi kegiatan praktikum
klasifikasi citra menggunakan metode Deep Learning.

