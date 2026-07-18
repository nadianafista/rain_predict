# Rain Prediction — Weather AUS Dataset
**PT AIGRA EON INDONESIA — Coding Test AI Engineer**

Prediksi apakah akan turun hujan esok hari (`RainTomorrow`) menggunakan data cuaca historis Australia (Weather AUS, 145.460 baris, 23 kolom).

---

## 1. Struktur Project

```
├── Nadia_Test_AI_Engineer_Intern.ipynb   # Notebook utama (EDA -> preprocessing -> modeling -> evaluasi)
├── weatherAUS.csv                         # Dataset (tidak disertakan, lihat sumber data)
├── requirements.txt                       # Daftar dependency Python
└── README.md
```

**Tools:** Python 3, Jupyter Notebook (Google Colab)
**Library:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

### Cara Menjalankan
```bash
pip install -r requirements.txt
jupyter notebook Nadia_Test_AI_Engineer_Intern.ipynb
```
Pastikan file `weatherAUS.csv` berada di path yang sesuai dengan yang dirujuk pada notebook (`/content/weatherAUS.csv` — path default Google Colab; sesuaikan jika dijalankan secara lokal).

---

## 2. Exploratory Data Analysis (EDA)

- **Info dataset**: `df.info()` untuk mengecek tipe data tiap kolom dan jumlah data non-null.
- **Missing values**: dihitung jumlah dan persentase missing values per kolom (`df.isnull().sum()`).
- **Distribusi target (`RainTomorrow`)**: divisualisasikan dengan bar chart (`sns.countplot`). Hasilnya menunjukkan data **imbalanced** — kelas `No` jauh lebih dominan dibanding `Yes` (rasio sekitar 78% vs 22%). Temuan ini menjadi dasar keputusan penggunaan `class_weight='balanced'` dan `scale_pos_weight` pada tahap modeling.

---

## 3. Handling Missing Values

Strategi imputasi dibedakan berdasarkan **persentase missing** per kolom, bukan diperlakukan sama rata:

| Kategori | Kolom | Perlakuan |
|---|---|---|
| **Missing < 10%** | MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am/3pm, Humidity9am/3pm, Pressure9am/3pm, Temp9am/3pm | **Interpolasi linear per Location** (data diurutkan berdasarkan `Location` + `Date` lebih dulu agar interpolasi mengikuti pola waktu-lokasi yang wajar), dengan fallback **mean/median per Location** dipilih berdasarkan **skewness** distribusi tiap kolom (mean untuk distribusi mendekati simetris, median untuk distribusi skewed) |
| **Missing 5–10%** (kategorikal) | WindGustDir, WindDir9am | Diisi berdasarkan pola dominan per lokasi |
| **Missing > 35%** | Evaporation, Sunshine, Cloud9am, Cloud3pm | Dianalisis pola missing-nya terhadap target (missing rate per kelas, crosstab missing-indicator vs target, distribusi nilai per kelas) — hasilnya **tidak menunjukkan perbedaan berarti** antara kelas `Yes`/`No`, sehingga kolom asli **di-drop**. Sebagai gantinya dibuat **missing indicator flag** (`_missing`) untuk menyimpan jejak informasi "data ini awalnya kosong atau tidak", lalu diuji lagi relevansinya di tahap feature selection |
| **RainToday** | Direkonstruksi konsisten dengan `Rainfall` | Termasuk kategori missing rendah, ikut diproses di alur yang sama |

**Prinsip penting yang diikuti:**
- Semua statistik imputasi (mean, median, skewness, frekuensi) **dihitung hanya dari `X_train`**, lalu diterapkan ke `X_test` — untuk mencegah *data leakage*.
- Baris dengan `RainTomorrow` kosong **di-drop**, bukan diimputasi, karena ini adalah target/label — mengisi label dengan nilai buatan akan mencemari proses training dan evaluasi.

---

## 4. Feature Selection (Minimal 5 Kolom + Alasan)

Pendekatan yang digunakan: **korelasi terhadap target + eliminasi multikolinearitas**, dengan langkah:

1. Seluruh fitur kategorikal di-**encode** (Frequency Encoding) agar bisa dihitung korelasinya secara numerik terhadap target.
2. **Korelasi tiap fitur terhadap `RainTomorrow`** dihitung; fitur dengan korelasi sangat lemah (`|r| < 0.03`) dibuang.
3. Dari fitur yang tersisa, dicek **korelasi antar-fitur** (heatmap + pairwise check, threshold `|r| > 0.85`). Jika dua fitur saling berkorelasi tinggi (redundan), fitur dengan korelasi lebih lemah terhadap target dibuang — contoh: `MaxTemp` dan `Pressure3pm` dibuang karena masing-masing sangat berkorelasi dengan `Temp3pm` dan `Pressure9am`.

**Hasil: 13 fitur terpilih** (dari 21 fitur hasil encoding):

| Fitur | Korelasi ke RainTomorrow | Alasan relevansi |
|---|---|---|
| `Humidity3pm` | **+0.44** (terkuat) | Kelembapan sore hari adalah indikator paling langsung dari potensi kondensasi/hujan |
| `RainToday` | −0.31 | Pola cuaca cenderung persisten dari hari ke hari (autokorelasi temporal) |
| `Humidity9am` | +0.25 | Sama seperti Humidity3pm, kelembapan pagi juga berkontribusi pada potensi hujan |
| `Rainfall` | +0.24 | Curah hujan hari ini berhubungan dengan sistem cuaca yang mungkin berlanjut esok |
| `Pressure9am` | −0.23 | Tekanan udara rendah secara meteorologis berasosiasi dengan sistem cuaca basah |
| `WindGustSpeed` | +0.22 | Angin kencang sering menyertai sistem cuaca yang membawa hujan |
| `Temp3pm` | −0.19 | Suhu sore berkaitan dengan pola pembentukan awan/hujan |
| `WindSpeed9am`, `WindSpeed3pm`, `MinTemp` | 0.08–0.09 | Kontribusi lebih kecil namun tetap informatif dan tidak redundan dengan fitur lain |
| `Cloud3pm_missing` | −0.04 | Missing indicator yang tetap dipertahankan karena lolos ambang korelasi |
| `WindDir9am`, `WindGustDir` | ~0.03 | Arah angin berkaitan dengan asal massa udara (basah/kering) |

Fitur seperti `Evaporation`, `Sunshine`, `Cloud9am`, `Cloud3pm` (missing >35%) sudah dieliminasi di tahap sebelumnya karena analisis distribusinya terhadap target tidak menunjukkan sinyal prediktif yang berarti.

---

## 5. Modeling & Evaluasi

Target di-encode: `No -> 0`, `Yes -> 1`. Split data **80% train / 20% test**, `stratify=y` untuk menjaga proporsi kelas.

### Model utama: Random Forest
- Hyperparameter tuning dengan `RandomizedSearchCV` (30 kombinasi, 5-fold `StratifiedKFold`, scoring `f1` — dipilih F1 alih-alih accuracy karena data imbalanced).
- `class_weight='balanced'` digunakan untuk mengatasi ketimpangan kelas tanpa resampling manual.

**Best params:** `n_estimators=300, max_depth=None, min_samples_split=15, min_samples_leaf=6, max_features='sqrt', bootstrap=False`

**Hasil evaluasi (test set, threshold default 0.5):**

| Metrik | Skor |
|---|---|
| Accuracy | 0.832 |
| Precision (Yes) | 0.617 |
| Recall (Yes) | 0.662 |
| F1-score (Yes) | 0.639 |
| ROC-AUC | 0.870 |

**Confusion Matrix** ditampilkan dalam notebook (Cell evaluasi Random Forest).

---

## 6. Kesimpulan Singkat

- **Kualitas dan struktur data sangat memengaruhi hasil model.** Missing values yang tidak ditangani secara hati-hati (misalnya index yang tidak sinkron antara fitur dan target setelah proses sorting) dapat membuat korelasi antar variabel menjadi tidak bermakna — ini pelajaran penting bahwa validasi integritas data harus dilakukan di setiap tahap pipeline, bukan hanya di awal.
- **Kelembapan (Humidity3pm/9am) dan status hujan hari ini (RainToday)** adalah prediktor paling kuat terhadap hujan esok hari — sejalan dengan intuisi meteorologis bahwa kondisi atmosfer memiliki persistensi jangka pendek.
- **Fitur dengan missing value sangat tinggi (>35%) tidak selalu perlu dipaksakan diimputasi** — setelah dianalisis, kolom seperti `Sunshine` dan `Cloud9am/3pm` tidak menunjukkan pola prediktif yang cukup kuat sehingga lebih aman di-drop daripada mengisi hampir separuh data dengan nilai buatan yang berisiko bias.
- **Data imbalanced menuntut pendekatan evaluasi dan tuning yang berbeda** — accuracy saja tidak cukup untuk menilai model pada kasus seperti ini; F1-score dan recall pada kelas minoritas (`Yes`) lebih relevan karena konsekuensi gagal mendeteksi hujan (false negative) lebih besar dampaknya dibanding salah memprediksi hujan (false positive).
- Random Forest dengan tuning menghasilkan **keseimbangan terbaik** antara precision dan recall dibanding XGBoost dan Logistic Regression pada eksperimen ini (lihat bagian Bonus).

---

## 7. Bonus / Requirement Opsional

### a. Perbandingan 3 Model

| Model | Accuracy | Precision (Yes) | Recall (Yes) | F1 (Yes) | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** (tuned) | **0.832** | **0.617** | 0.662 | **0.639** | 0.870 |
| XGBoost | 0.804 | 0.546 | **0.754** | 0.633 | **0.872** |
| Logistic Regression | 0.773 | 0.496 | 0.761 | 0.601 | 0.848 |

**Insight pemilihan model:**
- **Random Forest** memberikan **F1-score dan accuracy terbaik**, dengan keseimbangan precision-recall paling baik di antara ketiganya — dipilih sebagai model utama.
- **XGBoost** memiliki **recall tertinggi** (mendeteksi lebih banyak kejadian hujan) dan ROC-AUC sedikit lebih tinggi, namun precision-nya lebih rendah (lebih banyak false alarm) — kandidat baik jika prioritas bisnis adalah *meminimalkan kejadian hujan yang tidak terdeteksi*, meskipun konsekuensinya lebih banyak false alarm.
- **Logistic Regression** sebagai baseline linear menunjukkan performa paling rendah di semua metrik, mengindikasikan hubungan fitur-target tidak sepenuhnya linear — mendukung penggunaan model berbasis tree (RF/XGBoost).

### b. Feature Engineering Sederhana
- **Missing indicator flags** (`Evaporation_missing`, `Sunshine_missing`, `Cloud9am_missing`, `Cloud3pm_missing`) dibuat sebagai fitur biner baru untuk menangkap sinyal dari pola *missingness* itu sendiri, bukan sekadar dibuang begitu saja. Salah satunya (`Cloud3pm_missing`) lolos seleksi fitur dan digunakan dalam model final.

### c. Visualisasi Feature Importance
- Ditampilkan pada notebook menggunakan `feature_importances_` dari Random Forest, memvisualisasikan kontribusi relatif tiap fitur terhadap prediksi model.

### d. Handling Class Imbalance
- Ditangani menggunakan:
  - `class_weight='balanced'` pada Random Forest & Logistic Regression
  - `scale_pos_weight` (rasio kelas negatif/positif) pada XGBoost
  - Pemilihan metrik evaluasi utama berupa **F1-score**, bukan accuracy, baik untuk scoring hyperparameter tuning maupun perbandingan model
- (Opsional lanjutan yang bisa dieksplor: threshold tuning pada `predict_proba` untuk mengoptimalkan F1/recall sesuai kebutuhan bisnis.)

---

## 8. Catatan Reproducibility
- `random_state=42` digunakan konsisten di seluruh proses (split data, model, cross-validation) untuk memastikan hasil dapat direproduksi.
- Seluruh statistik preprocessing (mean, median, skewness, frequency encoding map) dihitung dari data training saja dan diterapkan ke data test, untuk mencegah data leakage.
