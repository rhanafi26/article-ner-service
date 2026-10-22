# Article Named-Entity Recognition (NER) Service

Prototipe layanan web dan REST API untuk ekstraksi entitas bernama (*Named-Entity Recognition*) pada artikel berita (kliping media perbankan daring) menggunakan model NLP teradaptasi.

---

## Daftar Isi
- [Struktur Direktori](#struktur-direktori)
- [Data Profiling & Splitting](#data-profiling--splitting)
- [Model NLP & Evaluasi](#model-nlp--evaluasi)
- [Pengujian Otomatis (Automated Tests)](#pengujian-otomatis-automated-tests)
- [Jawaban Pertanyaan Studi Kasus](#jawaban-pertanyaan-studi-kasus)
- [Batasan Sistem & Rencana Perbaikan](#batasan-sistem--rencana-perbaikan)

---

## Struktur Direktori

```text
article-ner-service/
├── README.md                      # Dokumentasi komprehensif proyek
├── requirements.txt               # Daftar dependensi library Python
├── .gitignore                     # Mengabaikan venv dan file dataset CSV mentah
├── data/
│   ├── raw/                       # 8 berkas CSV mentah (tidak di-commit ke Git)
│   ├── processed/                
│   │   ├── train_split.json           # Metadata partisi data latih (70%)
│   │   ├── val_split.json             # Metadata partisi data validasi (15%)
│   │   ├── test_split.json            # Metadata partisi data uji (15%)
│   │   └── evaluation_examples.json   # 10 artikel uji berlabel manual (gold standard)
├── src/
│   ├── __init__.py
│   ├── app.py                     # Entry point REST API dan web panel
│   ├── data_pipeline.py           # Skrip profiling, deduplikasi, dan data splitting
│   ├── ner_model.py               # Logika loading artefak model dan ekstraksi entitas
│   └── static/
│       └── index.html             # Antarmuka web panel interaktif
└── tests/
    ├── __init__.py
    └── test_api.py                # Minimal 5 unit tests otomatis (pytest)
```

---

## Data Profiling & Splitting

Berdasarkan analisis 8 berkas CSV kliping media daring perbankan (`data/raw/`):

1. **Jumlah Data:** Seluruh artikel dimuat dengan delimiter titik koma (`;`) dan penanganan kuotasi teks (`quotechar='"'`).
2. **Duplikasi:** Ditemukan baris duplikat yang memiliki kesamaan judul dan isi artikel akibat publikasi sindikasi media; data duplikat ini dideduplikasi sebelum pembagian data.
3. **Data Kosong:** Baris dengan `title` atau `body` bernilai null disaring keluar.
4. **Distribusi Bahasa:** Sebagian besar artikel menggunakan bahasa Inggris (`en`) dan bahasa Melayu (`ms`).
5. **Analisis Kolom `entities` Bawaan:** Kolom `entities` pada CSV asli tidak memuat indeks rentang karakter (*character offsets / spans*) dan provenance/asal-usulnya tidak terverifikasi. Oleh karena itu, kolom tersebut tidak dapat dijadikan *ground truth* tanpa anotasi manual.
6. **Data Partitioning:** Menggunakan pemisahan berdasarkan artikel unik (*split by article identity*) dengan `random_state=42`:
   - **Training Set (70%):** Digunakan untuk kalibrasi dan adaptasi model.
   - **Validation Set (15%):** Digunakan untuk tuning parameter dan aturan pemetaan label.
   - **Test Set (15%):** Disimpan tanpa diubah untuk evaluasi akhir.

---

## Model NLP & Evaluasi

### Pemilihan Model
- **Baseline:** Model pra-latih spaCy (`en_core_web_sm` / pipeline multilingual) untuk deteksi awal entitas standar.
- **Model Teradaptasi:** Model yang disesuaikan (*fine-tuned* / *calibrated*) menggunakan data latih domain perbankan untuk mengenali singkatan institusi finansial (misal: *CIMB, RHB, MDV*) dan entitas lokal (*Kuala Lumpur, Bank Negara*).

### Metrik Evaluasi (Diuji pada `evaluation_examples.json`)
Evaluasi dihitung berbasis *exact-span matching* dan *entity-type matching*:

| Model | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Baseline (Pretrained) | ... | ... | ... |
| Final Adapted Pipeline | ... | ... | ... |

### Analisis Kesalahan (Error Analysis)
1. **False Positive (Kategori Umum sebagai Organisasi):** Model dasar mendeteksi kata generik "Bank" atau "Central Bank" sebagai entitas tunggal `ORG` terpisah, padahal merupakan bagian dari frasa umum.
2. **False Negative (Singkatan Lembaga Keuangan Khusus):** Entitas akronim seperti "MDV" (Malaysia Debt Ventures) terlewat pada model *baseline*, tetapi berhasil diatasi setelah adaptasi data perbankan.
3. **Span Boundary Error (Gelar dan Nama Orang):** Terdapat pergeseran batas span saat mendeteksi nama bertitel seperti "Datuk Khairussaleh Ramli", di mana gelar kehormatan kadang ikut terbawa ke dalam entitas `PERSON`.

---

## Pengujian Otomatis (Automated Tests)

Pengujian unit otomatis diimplementasikan pada direktori `tests/test_api.py` dan dapat dijalankan dengan:
```bash
pytest tests/
```

Skenario pengujian mencakup minimal 5 kasus:
1. `test_extract_entities_valid_article`: Verifikasi response status 200, struktur JSON, dan keakuratan kalkulasi rentang offset.
2. `test_extract_entities_empty_input`: Verifikasi bahwa judul atau body kosong mengembalikan HTTP 400.
3. `test_extract_entities_whitespace_only`: Verifikasi input yang hanya memuat spasi/karakter kosong ditolak.
4. `test_repeated_entities_distinct_offsets`: Memastikan entitas yang muncul berulang kali di satu artikel memiliki indeks *start* dan *end* yang berbeda.
5. `test_special_characters_and_punctuation`: Memastikan sistem tidak rusak saat memproses tanda baca kompleks, tanda petik, dan karakter non-ASCII.

---

## Jawaban Pertanyaan Studi Kasus

### 1. What makes a text span a named entity in this problem?
Rentang teks (*text span*) dianggap sebagai entitas bernama jika teks tersebut merujuk secara spesifik pada individu di dunia nyata (orang), instansi/organisasi formal (perusahaan, bank, lembaga pemerintah), lokasi geografis/wilayah administrasi, maupun ekspresi tanggal tertentu yang relevan bagi pemantauan berita industri perbankan.

### 2. Why should title and body be combined before analysis? What information could be lost if they are processed independently?
Judul berita sering kali padat dan minim konteks sintaksis lengkap (misal: "Maybank Catat Pertumbuhan Kuartal II"). Jika diproses terpisah:
- Model dapat kehilangan resolusi ko-referensi (*coreference resolution*) dan konteks semantik antara subjek di judul dan penjelasan rincinya di paragraf awal isi berita.
- Menggabungkan keduanya memberikan konteks wacana (*discourse context*) yang utuh kepada model dan memastikan indeks karakter dihitung dari satu dokumen terpadu.

### 3. What are the supported labels of the chosen model, and how do they map to the categories shown to users?
Model mendukung label ontologi spaCy / CoNLL:
- `PERSON`: Tokoh, eksekutif, narasumber.
- `ORG`: Bank, korporasi, institusi keuangan, asosiasi.
- `GPE` / `LOC`: Negara, kota, lokasi regional (ditampilkan sebagai "Location").
- `DATE`: Penanggalan kuartal, hari, atau periode rilis berita.
Pada antarmuka pengguna, label teknis ini disederhanakan menjadi kategori yang mudah dipahami: *Person*, *Organization*, *Location*, dan *Date*.

### 4. How does tokenization, punctuation, capitalization, or language affect the result?
- **Kapitalisasi:** Model NER berbasis konteks sangat bergantung pada huruf kapital (*title case*) untuk mencurigai adanya entitas nama diri. Judul artikel yang ditulis dengan *ALL CAPS* dapat membingungkan model.
- **Tanda Baca:** Tanda kutip, titik koma, dan tanda hubung dapat memotong tokenisasi span jika tokenizer tidak dikonfigurasi dengan benar.
- **Bahasa:** Campuran bahasa Inggris dan Melayu pada media Malaysia dapat menurunkan akurasi jika tokenizer tidak menangani imbuhan atau pola part-of-speech lokal.

### 5. What is the difference between an entity's text, label, and character offsets?
- **Text:** Teks substring aktual yang muncul di artikel (misal: `"Kuala Lumpur"`).
- **Label:** Kategori semantik dari entitas tersebut (misal: `GPE`).
- **Character Offsets (`start`, `end`):** Indeks posisi karakter berbasis nol (*zero-based indexing*) di mana entitas tersebut dimulai dan berakhir secara eksklusif dalam teks dokumen, memungkinkan penandaan koordinat yang presisi tanpa bergantung pada tokenisasi.

### 6. What happens when the same entity appears multiple times with different spelling or capitalization?
Model akan mendeteksi setiap kemunculan sebagai entitas terpisah dengan offset karakter yang unik. Jika terdapat variasi ejaan atau kapitalisasi, model probabilistik mungkin berhasil mendeteksi yang berhuruf kapital namun melewatkan versi huruf kecil. Sistem dasar tidak menyatukan entitas ini; proses normalisasi atau *Entity Linking* diperlukan untuk mengelompokkannya ke satu identitas kanonikal.

### 7. Which errors are most important for a news-monitoring user, and how could the system be improved?
- Bagi tim pemantau berita, kesalahan **False Negative** (melewatkan nama bank atau krisis yang sedang diberitakan) jauh lebih berisiko daripada *False Positive*, karena tim dapat kehilangan berita penting (*missed alert*).
- Sistem dapat ditingkatkan dengan menambahkan modul kamus entitas finansial (*gazetteer / entity ruler*), penyesuaian model dwibahasa (Inggris-Melayu), dan fitur normalisasi entitas ekuivalen.

### 8. What privacy, copyright, bias, or responsible-use concerns apply to processing article text?
- **Hak Cipta:** Konten artikel berita daring memiliki hak cipta penerbit; penyimpanan teks lengkap harus dibatasi untuk keperluan pemrosesan internal analitik tanpa mendistribusikan ulang teks mentah secara publik.
- **Privasi (PII):** Penyebutan nama orang perseorangan dalam berita perlu diproses secara bertanggung jawab dan mematuhi regulasi privasi data.
- **Bias Representasi:** Model yang dilatih pada media perbankan korporat besar rentan mengalami bias terhadap entitas institusi besar dan berpotensi memiliki performa lebih rendah saat mengekstrak nama organisasi non-pemerintah atau usaha kecil.

---

## Batasan Sistem & Rencana Perbaikan

### Batasan Saat Ini
- Pemrosesan dibatasi maksimal 10.000 karakter per artikel untuk mencegah beban memori berlebih.
- Belum mendukung deduplikasi tingkat entitas (*coreference resolution*, misal menghubungkan "Khairussaleh" dengan "Datuk Khairussaleh Ramli").

### Rencana Pengembangan
- [ ] Implementasi *Entity Linking* berbasis Wikidata/basis data perbankan.
- [ ] Penambahan fitur visualisasi penandaan entitas langsung di dalam teks (*span highlighter*).
- [ ] Pembuatan *batch processing endpoint* untuk mengekstrak banyak artikel sekaligus.