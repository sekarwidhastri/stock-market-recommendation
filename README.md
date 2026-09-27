# 📈 Stock Market Recommendation: AI & Quant Platform

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FIfan-Apres%2Fstock-market-recommendation)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Ifan--Apres%2Fstock--market--recommendation-blue?logo=github)](https://github.com/Ifan-Apres/stock-market-recommendation)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Platform Rekomendasi Saham Kuantitatif Berbasis Multi-Engine Machine Learning (GBDT + LSTM + ARIMA + GARCH), Institutional Morning Brief AI (Gemini 3.6 Flash), dan Kalkulator Lot & Modal Ramah Pemula untuk Bursa Efek Indonesia (BEI / IDX).**
> **Dikembangkan dan Dikelola oleh TIM New York: Ifan Apres & Sekar Widhastri.**

---

## 🌟 Ikhtisar Proyek (Project Overview)

**Stock Market Recommendation** adalah platform investasi saham dan riset kuantitatif yang dirancang dengan antarmuka yang bersih, ramah pengguna (*user-friendly*), dan bertenaga mesin komputasi kuantitatif tingkat institusional. Platform ini memadukan model machine learning modern (**GBDT + Deep Learning LSTM + ARIMA**), pemodelan volatilitas ekonometrika (**GARCH 1,1**), optimasi alokasi portofolio (**Sharpe & Risk-Parity dengan 20% Kas Siaga**), serta generator **Institutional Morning Brief IHSG harian otomatis** yang ditenagai oleh **Google Gemini AI**.

Platform ini menyajikan rekomendasi yang actionable dengan level teknikal presisi (*Entry Price, Target Price, Stop Loss, Risk-to-Reward Ratio*) dan kalkulator alokasi modal nominal (Rupiah & Lot) yang langsung dapat dieksekusi oleh investor maupun trader.

---

## 🛡️ Kepatuhan Hukum & Clean-Room Protocol (Legal Compliance)

Platform ini dibangun di bawah protokol ketat **"Clean-Room Engineering"** untuk memastikan kepatuhan penuh terhadap perjanjian kerja (PKWTT) dan peraturan perundangan (UU ITE & HAKI):

| Prinsip Kepatuhan | Implementasi di Stock Market Recommendation |
| :--- | :--- |
| **Bebas Scraping Ilegal** | **Tidak ada** *web scraping*, injeksi session cookie, ataupun pemanfaatan token rahasia dari portal riset tertutup (*Infovesta, Stockbit, dll.*). |
| **Sumber Data 100% Terbuka** | Data pasar resmi diperoleh melalui feed publik Yahoo Finance API (`yfinance` dengan sufiks `.JK`), serta data makro resmi Bank Indonesia. |
| **Standar Matematika Publik** | Semua formula (Sharpe Ratio, Beta Pasar, GARCH, Floor Pivots, Black-Litterman/Markowitz) merupakan metodologi domain publik internasional (*CFA Institute, Modern Portfolio Theory*), bukan formula rahasia institusi manapun. |
| **Integritas Lisensi** | Menggunakan arsitektur modular open-source standar Python kuantitatif (*Scikit-Learn, PyTorch, Statsmodels, Arch*). |

---

## 🚀 Fitur Unggulan (Core Capabilities)

### 1. 📰 Automated Institutional Morning Brief IHSG
- **Analisis Sentimen Global Semalam**: Mengagregasi penutupan Wall Street (*S&P 500, Nasdaq, Dow Jones*), *yield* US Treasury 10-Tahun, harga minyak mentah (*Brent/WTI*), emas dunia, dan kurs Rupiah (USD/IDR).
- **Pemetaan Level Kunci IHSG**: Menghitung secara otomatis area *Support 1, Support 2, Pivot Point, Resistance 1, Resistance 2*, serta posisi teknikal relatif terhadap Moving Average (MA20).
- **Editorial Bernada Institusional**: Menghasilkan narasi editorial harian otomatis menggunakan **Google Gemini 3.6 Flash** dengan gaya riset analis riset pasar terkemuka.

### 2. 🧠 Multi-Engine Hybrid Ensemble Modeling
Platform Stock Market Recommendation tidak mengandalkan satu model saja, melainkan menggabungkan tiga mesin inferensi yang saling melengkapi:
- **Gradient Boosted Decision Trees (GBDT - 50% Bobot)**: Mengekstraksi non-linear interaction antara 60+ faktor teknikal, likuiditas, dan fundamental.
- **Deep Learning LSTM (PyTorch - 30% Bobot)**: Neural network sekuensial untuk menangkap pola temporal dan memori historis pergerakan harga.
- **ARIMA (20% Bobot)**: Model deret waktu ekonometrika stasioner untuk mendeteksi *mean-reversion* jangka pendek.

### 3. 📉 Pemodelan Risiko Lanjutan (GARCH & VaR)
- **GARCH(1,1) Conditional Volatility**: Mengukur *volatility clustering* harian dan tahunan untuk mendeteksi lonjakan risiko pasar secara dinamis.
- **Value at Risk (VaR 95% 1-Hari)**: Mengestimasi batas kerugian maksimal harian pada tingkat keyakinan 95%.
- **Metrik Kinerja Investasi**: *Beta terhadap IHSG*, *Sharpe Ratio Historis*, *Maximum Drawdown 1-Tahun*, dan *Annualized Return*.

### 4. 💼 Dynamic Portfolio Allocation Optimizer
- **Alokasi Berdasarkan Risk-Adjusted Return**: Menghitung bobot optimal tiap saham menggunakan kombinasi rasio Sharpe dan invers volatilitas GARCH.
- **Kebijakan Kas Siaga (20% Cash Reserve)**: Menyisihkan 20% modal dalam bentuk tunai untuk manajemen likuiditas dan mitigasi *market shock*.
- **Kalkulasi Lot Bursa Efek Indonesia**: Mengonversi nominal Rupiah ke dalam satuan lot resmi IDX (1 lot = 100 lembar saham) secara presisi.

### 5. 🎯 Rekomendasi Taktis & Level Trading Presisi
Mengklasifikasikan saham ke dalam aksi yang jelas dan terarah:
- **Buy on Weakness (BoW)**: Akumulasi bertahap saat harga menguji area support kuat.
- **Buy on Breakout (BoB)**: Pembelian agresif saat harga menembus level resisten dengan konfirmasi volume.
- **Trading Buy**: Momentum buy dengan target jangka pendek.
- **Sell on Strength (SoS) / Hold**: Realisasi profit atau proteksi modal saat tren melandai.
- Lengkap dengan **Harga Masuk (Entry)**, **Target Harga 1 (TP1)**, **Stop Loss (SL)**, dan **Risk-to-Reward Ratio (RRR)**.

### 6. 🏛️ Cakupan 11 Sektor Resmi Bursa Efek Indonesia (IDX-IC)
Menyajikan klasifikasi dan filter rekomendasi untuk 11 sektor resmi:
1. *Financials* (Keuangan)
2. *Energy* (Energi)
3. *Basic Materials* (Barang Baku)
4. *Consumer Non-Cyclicals* (Konsumer Primer)
5. *Consumer Cyclicals* (Konsumer Non-Primer)
6. *Healthcare* (Kesehatan)
7. *Technology* (Teknologi)
8. *Infrastructures* (Infrastruktur & Telekomunikasi)
9. *Properties & Real Estate* (Properti & Real Estat)
10. *Industrials* (Perindustrian)
11. *Transportation & Logistics* (Transportasi & Logistik)

---

## 📂 Struktur Direktori & File (Directory Architecture)

Struktur proyek disusun secara terorganisir, modular, dan bersih untuk kemudahan pemeliharaan dan skalabilitas:

```text
Stock Market Recomendation/
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml         # Otomasi GitHub Actions harian (cron 05:00 WIB)
├── api/
│   ├── index.py                       # Serverless handler untuk Vercel
│   └── main.py                        # FastAPI REST API, routing, and dashboard server
├── data/
│   ├── README.md                      # Dokumentasi arsitektur data & siklus hidup
│   ├── raw/                           # Ingestion data mentah
│   │   ├── raw_market_data.csv        # Data OHLCV saham (2018 - sekarang)
│   │   ├── benchmark_market_data.csv  # Data historis IHSG (^JKSE)
│   │   ├── fundamental_financial_data.csv # Snapshot valuasi rasio fundamental
│   │   └── financial_statements/      # FOLDER KHUSUS LAPORAN KEUANGAN PER EMITEN
│   │       ├── README.md              # Penjelasan format multi-sheet & metrik laporan
│   │       ├── BBCA_financial_statements.xlsx  # Multi-sheet: Income, Balance, Cash Flow
│   │       ├── BBCA_income_statement.csv
│   │       ├── BBRI_financial_statements.xlsx
│   │       ├── BBRI_income_statement.csv
│   │       └── ... (laporan emiten lainnya)
│   └── processed/                     # Hasil pemrosesan kuantitatif
│       ├── processed_market_features.csv      # Matriks 60+ fitur teknikal & risiko
│       ├── advanced_quant_metrics.csv         # Metrik Sharpe, Beta, VaR, Pivots
│       ├── alpha_model.joblib                 # Bobot model GBDT tersimpan
│       ├── lstm_model.pth                     # Bobot neural network PyTorch LSTM
│       ├── latest_morning_brief.json          # Hasil editorial Morning Brief dari Gemini
│       ├── latest_portfolio_allocation.csv    # Rekomendasi bobot alokasi modal
│       ├── latest_alpha_recommendations.csv   # Rekomendasi konsolidasi
│       ├── latest_alpha_recommendations_swing.csv     # Filter Swing Trader
│       ├── latest_alpha_recommendations_dividend.csv  # Filter Dividend & Value
│       └── latest_alpha_recommendations_favorites.csv # Filter Portfolio Pilihan
├── src/
│   ├── __init__.py
│   ├── config.py                      # Konfigurasi semesta saham, sektor, & parameter
│   ├── 01_data_ingestion.py           # Engine penarikan OHLCV, macro & laporan keuangan
│   ├── 02_feature_eng.py              # Ekstraksi fitur, GARCH(1,1), Pivots, Sharpe, Beta
│   ├── 03_model_inference.py          # Ensemble GBDT+LSTM+ARIMA & Optimizer Alokasi
│   └── morning_brief.py               # Generator Institutional Morning Brief via Gemini AI
├── index.html                         # Dashboard web modern responsif (Dark Mode & Glassmorphism)
├── poster.html                        # Presentasi visual ringkasan platform
├── main.py                            # Master runner pipeline 4 langkah
├── run.bat                            # Skrip eksekusi satu klik untuk Windows
├── run_daily.ps1                      # Skrip eksekusi harian untuk PowerShell
├── run_daily.sh                       # Skrip eksekusi harian untuk Linux / macOS
├── requirements.txt                   # Daftar pustaka dependensi Python
├── vercel.json                        # Konfigurasi deployment ke Vercel Serverless
├── .env                               # Kunci API lokal (GEMINI_API_KEY)
└── README.md                          # Dokumentasi resmi proyek
```

---

## 🛠️ Instalasi & Menjalankan Lokal (Quickstart Guide)

### 1. Prasyarat Sistem
- **Python 3.10** atau versi lebih baru (diuji pada Python 3.10 – 3.13 di Windows & Linux).
- Akses internet untuk penarikan data bursa terkini.

### 2. Kloning & Pemasangan Dependensi
```bash
# Masuk ke direktori proyek
cd "Stock Market Recomendation"

# Buat virtual environment (opsional namun disarankan)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Pasang seluruh dependensi pustaka
pip install -r requirements.txt
```

### 3. Konfigurasi Lingkungan (`.env`)
Buat atau pastikan file `.env` di direktori utama berisi API key Gemini Anda:
```ini
GEMINI_API_KEY=AIzaSy... (kunci API Gemini Anda)
```

### 4. Menjalankan Master Pipeline Kuantitatif
Untuk menjalankan seluruh tahapan dari awal (Penarikan Data $\rightarrow$ Rekayasa Fitur & GARCH $\rightarrow$ Pelatihan Model & Ensemble $\rightarrow$ Generator Morning Brief):

**Opsi A (Menggunakan Master Script Python):**
```bash
python main.py
```

**Opsi B (Satu Klik di Windows):**
Cukup klik ganda pada file `run.bat` atau jalankan via PowerShell:
```powershell
.\run_daily.ps1
```

### 5. Menjalankan Dashboard & REST API
Setelah pipeline selesai menghasilkan data rekomendasi, jalankan server web FastAPI:
```bash
uvicorn api.main:app --reload --port 8000
```
Buka peramban (*browser*) Anda di:
- **Dashboard Web Interaktif**: [http://localhost:8000/](http://localhost:8000/)
- **Dokumentasi Interaktif Swagger API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternatif Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🌐 Dokumentasi Endpoint API (RESTful Endpoints)

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Menampilkan antarmuka Dashboard Web interaktif. |
| `GET` | `/morning-brief` | Mengembalikan narasi editorial Morning Brief IHSG terkini beserta snapshot pasar global. |
| `GET` | `/portfolio/allocate` | Menghitung alokasi modal optimal berdasarkan input nominal modal (`?capital=50000000`). |
| `GET` | `/recommendations` | Mengembalikan daftar rekomendasi saham (filter `?universe=swing`, `dividend`, `favorites`, atau `sector=Financials`). |
| `GET` | `/models/compare/{ticker}` | Menampilkan perbandingan probabilitas prediksi antar model (GBDT vs LSTM vs ARIMA) untuk emiten tertentu. |
| `GET` | `/sectors` | Daftar 11 sektor resmi IDX dan kode emiten yang terdaftar. |
| `GET` | `/status` | Informasi status kesehatan sistem dan waktu pembaruan pipeline terakhir. |
| `POST`| `/pipeline/run` | Menjalankan ulang seluruh pipeline kuantitatif di background secara asinkron. |

---

## ☁️ Panduan Deployment ke Vercel (Vercel Serverless)

Proyek ini telah dikonfigurasi secara *native* agar dapat dideploy ke platform **Vercel** dengan konfigurasi serverless:

1. **Pastikan File Konfigurasi Tersedia**:
   - `vercel.json` sudah terkonfigurasi mengarahkan rute `/(.*)` ke `api/index.py`.
   - `api/index.py` mengekspos instance FastAPI `app`.
2. **Deploy Menggunakan Vercel CLI**:
   ```bash
   npm i -g vercel
   vercel login
   vercel
   ```
3. **Atur Environment Variable di Vercel Dashboard**:
   - Tambahkan variabel `GEMINI_API_KEY` di pengaturan proyek Vercel (*Settings $\rightarrow$ Environment Variables*).
4. **Data Statis**: File hasil pemrosesan di folder `data/processed/` akan otomatis ter-bundle ke dalam container Vercel sehingga API dapat langsung menyajikan data rekomendasi secara instan tanpa *cold start delay*.

---

## ⏰ Otomasi Harian (GitHub Actions CI/CD)

Proyek ini dilengkapi dengan skrip otomatis `.github/workflows/daily_pipeline.yml`:
- **Jadwal Eksekusi**: Setiap hari kerja (Senin–Jumat) pukul 22:00 UTC (pukul 05:00 WIB pagi berikutnya).
- **Alur Kerja**:
  1. Melakukan checkout repositori dan menyiapkan Python.
  2. Menjalankan `src/01_data_ingestion.py` untuk menarik data penutupan pasar terbaru.
  3. Menjalankan `src/02_feature_eng.py` untuk mengkalkulasi ulang GARCH dan indikator.
  4. Menjalankan `src/03_model_inference.py` untuk memperbarui sinyal dan optimasi portofolio.
  5. Menjalankan `src/morning_brief.py` untuk menghasilkan Morning Brief hari bersangkutan.
  6. Melakukan *auto-commit* dan *push* pembaruan ke cabang utama (*main branch*).

---

## 📊 Metrik Finansial & Ekonometrika yang Digunakan

1. **Conditional Volatility (GARCH 1,1)**:
   $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
   Digunakan untuk mendeteksi rezim volatilitas pasar dan penyesuaian bobot portofolio dinamis.
2. **Value at Risk (Parametrik 95%)**:
   $$\text{VaR}_{95} = 1.645 \times \sigma_{\text{daily}}$$
3. **Sharpe Ratio Tahunan**:
   $$\text{Sharpe} = \frac{R_{\text{annual}} - R_f}{\sigma_{\text{annual}}}$$
   (dengan $R_f = 6.00\%$ BI-Rate).
4. **Floor Pivots Klasik**:
   $$\text{Pivot} = \frac{H + L + C}{3}, \quad R_1 = 2P - L, \quad S_1 = 2P - H, \quad R_2 = P + (H - L), \quad S_2 = P - (H - L)$$

---

## 👥 Tim Riset Kuantitatif & Rekayasa Sistem (TIM New York)

Platform ini dikembangkan dan dikelola secara kolaboratif oleh **TIM New York**:

| Nama Kontributor | Peran & Tanggung Jawab Utama | Kontak / Profil |
| :--- | :--- | :--- |
| **Ifan Apres** | *Lead Quantitative Engineer & Fullstack Systems Architect* — Bertanggung jawab atas arsitektur komputasi, pipeline data kuantitatif, integrasi model ensemble (GBDT + LSTM), deployment Vercel & CI/CD automation. | [GitHub @Ifan-Apres](https://github.com/Ifan-Apres) |
| **Sekar Widhastri** | *Senior Market & Research Analyst* — Bertanggung jawab atas formulasi strategi analisis pasar ekonometrika, metodologi risk-parity alokasi portofolio, evaluasi sinyal teknikal BEI, dan kurasi editorial riset pasar. | [GitHub @sekarwidhastri](https://github.com/sekarwidhastri) |

---

## ⚖️ Disclaimer & Batasan Tanggung Jawab

*Aplikasi ini dikembangkan untuk tujuan riset kuantitatif, analisis data, dan edukasi finansial. Seluruh rekomendasi yang dihasilkan oleh model machine learning dan kecerdasan buatan merupakan indikator probabilitas statistik pasar dan bukan merupakan anjuran mutlak untuk membeli atau menjual efek tertentu. Keputusan investasi dan manajemen risiko sepenuhnya berada di tangan investor masing-masing.*

