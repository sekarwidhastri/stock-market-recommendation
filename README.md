# 📈 Stock Market Recommendation: AI & Quant Platform

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FIfan-Apres%2Fstock-market-recommendation)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Ifan--Apres%2Fstock--market--recommendation-blue?logo=github)](https://github.com/Ifan-Apres/stock-market-recommendation)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Platform Rekomendasi Saham Kuantitatif Berbasis Multi-Engine Machine Learning (GBDT + Macro-Aware PyTorch LSTM + ARIMA + GARCH), Target Excess Alpha Relatif, Doktrin AI Morning Brief (Gemini 3.8 Flash), dan Dynamic Universe Manager untuk Bursa Efek Indonesia (BEI / IDX).**  
> **Dikembangkan dan Dikelola oleh TIM New York: Ifan Apres & Sekar Widhastri.**

---

## 🌟 Ikhtisar Proyek (Project Overview)

**Stock Market Recommendation** adalah platform investasi saham dan riset kuantitatif tingkat institusional yang dirancang dengan antarmuka modern yang bersih, cepat, dan berbasis komputasi matematis presisi. Platform ini memadukan:
1. **Target Excess Return (Alpha Relatif)**: Mengisolasi keunggulan saham murni terhadap indeks acuan IHSG ($R_{\text{saham}, 5D} - R_{\text{IHSG}, 5D} > 0$).
2. **Macro-Aware PyTorch LSTM Sequence Model (12-Dimensi)**: Membaca dinamika mikro teknikal saham, pergerakan makro lintas aset (US 10Y Yield, DXY, Minyak Brent), serta fitur interaksi sensitivitas sektoral (*Sector-Macro Interaction*).
3. **Cross-Sectional Top Decile & High-Conviction Thresholding**: Menyaring sinyal spekulatif (*noise*) dengan hanya mengeksekusi sinyal BUY pada keyakinan $\ge 0.55$ atau 10% saham terbaik di bursa.
4. **Dynamic Universe Manager (Kapasitas N=66 Emiten)**: Manajemen semesta saham dinamis dengan sirkuit pengaman (*circuit-breaker*) otomatis dan substitusi cadangan *standby*.
5. **Institutional Morning Brief AI (Gemini 3.8 Flash)**: Narasi riset pasar terkurasi dengan scraping berita makro real-time, *Executive Key Takeaways* 10-detik, dan doktrin riset **AlphaTech**.
6. **Ekonometrika Risiko Lanjutan**: Pemodelan volatilitas kondisional Student-t GARCH(1,1), Value at Risk (VaR 95% & 99%), Expected Shortfall (ES), dan alokasi portofolio dengan **20% Kas Siaga**.

---

## 📊 Hasil Uji Validasi & Benchmark Performa Kuantitatif

Evaluasi model dilakukan secara ketat pada data uji independen di luar sampel (*Out-of-Sample Test Split*, ~14.600 observasi pasar historis) untuk menguji akurasi prediksi masa depan:

| Metrik Evaluasi Kuantitatif | Sebelum (Baseline) | Sesudah (Arsitektur Baru) | Peningkatan / Keunggulan | Dampak Praktis bagi Investor |
| :--- | :---: | :---: | :---: | :--- |
| **Metodologi Target** | Return Absolut ($R > 0$) | **Excess Return ($\text{Alpha} > 0$)** | **Target Alpha Murni** | Mengukur keunggulan riil saham terhadap IHSG; tetap mendeteksi saham defensif saat bursa sedang koreksi. |
| **Dimensi Fitur Sekuensial LSTM** | 5 Dimensi Mikro | **12 Dimensi (+ Makro & Sektor)** | **+7 Variabel Konteks** | LSTM memahami apakah tren saham didukung atau terhambat oleh likuiditas global dan harga komoditas. |
| **Win-Rate / Precision Sinyal BUY** | **48.12%** | **52.95%** | <span style="color:green">**+4.83%**</span> | **Membalikkan probabilitas kerugian!** Sebelumnya >51% sinyal BUY kalah. Sekarang mayoritas sinyal BUY menghasilkan profit riil. |
| **Top Decile Precision (Top 10% Teratas)** | *Tidak ada* | **53.53%** | <span style="color:green">**+5.41% vs Baseline**</span> | Saham-saham dengan keyakinan model tertinggi menghasilkan akurasi outperformance paling solid. |
| **GBDT ROC-AUC (Daya Bedah Kelas)** | 0.5276 | **0.5364** | <span style="color:green">**+0.0088**</span> | Pemisahan probabilitas antara saham pemenang dan saham tertinggal semakin tajam. |
| **Disiplin di Pasar Konsolidasi** | Ceroboh (*False BUY* di pasar lesu) | **Proteksi Modal (100% Kas Siaga)** | **Capital Preservation** | Menolak memberikan sinyal beli spekulatif saat rezim makro tertekan, mengamankan modal trader. |

---

## 🚀 Fitur Unggulan (Core Architecture)

### 1. 🎯 Target Excess Return (Alpha Relatif vs IHSG)
Alih-alih menebak pergerakan harga absolut yang sering kali hanya membonceng arus indeks pasar umum (*Beta effect*), model dilatih menggunakan target Alpha relatif:
$$\text{Alpha}_{5D} = R_{\text{Saham}, 5D} - R_{\text{IHSG}, 5D}$$
$$\text{Target\_Class}_{5D} = \begin{cases} 1, & \text{jika } \text{Alpha}_{5D} > 0 \\ 0, & \text{lainnya} \end{cases}$$
Model difokuskan menemukan saham-saham *outperformer* di segala siklus pasar (Bullish, Sideways, maupun Bearish).

### 2. 🧠 Macro-Aware PyTorch LSTM dengan Interaksi Sektoral (12-Dimensi)
Tensor input jaringan saraf tiruan sekuensial PyTorch LSTM membaca jendela waktu *lookback* 30 hari perdagangan secara terintegrasi:
* **Fitur Mikro Internal (5 Variabel)**: `Return_1D`, `RSI_14`, `Dist_SMA_20`, `CMF_20` (Chaikin Money Flow), `Volume_Ratio`.
* **Katalis Makro Global & Domestik (4 Variabel)**: `IHSG_Return_1D`, `US_10Y_Yield_Delta` (Yield Obligasi US Treasury 10Y), `USD_IDR_Return_1D` (Kurs Rupiah), `Brent_Oil_Return_1D` (Minyak Mentah ICE Brent).
* **Interaksi Sensitivitas Sektor (3 Variabel)**:
  * `Oil_Energy_Tailwind`: Dampak kenaikan minyak mentah aktif khusus untuk emiten sektor **Energy**.
  * `Rate_Bank_Sensitivity`: Delta yield obligasi aktif khusus untuk mengukur margin perbankan sektor **Financials**.
  * `FX_Consumer_Headwind`: Pelemahan Rupiah terhadap USD aktif sebagai beban biaya impor emiten sektor **Consumer Goods & Healthcare**.

### 3. 🛡️ Ambang Eksekusi Probabilitas & Cross-Sectional Top Decile
Untuk mengeliminasi sinyal palsu pada zona abu-abu (probabilitas 50%–52%), sistem menerapkan aturan eksekusi ketat:
* Sinyal `BUY ON WEAKNESS`, `BUY ON BREAKOUT`, dan `TRADING BUY` hanya dieksekusi jika:
  $$\text{Bullish Probability} \ge 0.55 \quad \text{ATAU masuk } \text{Top Decile (Top 10\% teratas)}$$
* Saham di luar kriteria ini secara disiplin diberi label **`HOLD`**, mencegah *overtrading* dan menjaga disiplin alokasi modal.

### 4. 🌐 Dynamic Universe Manager (Kapasitas N=66 Emiten)
* **Kapasitas Semesta Dinamis**: Mengelola 66 saham berlikuiditas tinggi di BEI yang terbagi rata ke dalam 11 sektor IDX-IC.
* **Automated Health-Check**: Mendeteksi anomali data penutupan, suspensi bursa, atau volatilitas ekstrem dengan *circuit-breaker* otomatis.
* **Standby Reserve Substitution**: Jika saham utama mengalami suspensi, sistem secara mulus (*zero-downtime*) mempromosikan saham cadangan (*standby reserve*) yang setara sektornya.

### 5. 📰 Institutional Morning Brief AI (Gemini 3.8 Flash & AlphaTech Doctrine)
* **Real-Time Macro Scraping**: Mengambil otomatis berita dan katalis terkini semalam dari bursa Wall Street (S&P 500), geopolitik minyak Brent, dan indeks Dolar AS (DXY).
* **AI System Doctrine (AlphaTech)**: Mengindoktrinasi model Google Gemini 3.8 Flash dengan kaidah riset quant New York, membedah korelasi sebab-akibat lintas pasar (*cross-market causality*).
* **Executive Key Takeaways**: Kartu ringkasan 10-detik mencakup Arah Indeks, Katalis Global, Risiko Makro, dan Panduan Taktis Alokasi Kas.
* **Live Session Date Synchronization**: Menyelaraskan tanggal analisis secara otomatis dengan sesi perdagangan bursa aktif BEI.

### 6. 📉 Pemodelan Risiko Ekonometrika (GARCH & VaR)
* **Student-t GARCH(1,1)**: Memodelkan *fat-tailed conditional volatility* untuk mengantisipasi *Black Swan events*.
* **Value at Risk (VaR 95% & 99% 1-Hari)** & **Expected Shortfall (ES)**: Menghitung potensi kerugian terburuk harian dengan cadangan fallback RiskMetrics EWMA ($\lambda=0.94$).
* **Institutional Metrics**: Perhitungan otomatis *Beta terhadap IHSG*, *Sharpe Ratio Historis*, *Maximum Drawdown 1-Tahun*, serta *Support/Resistance Floor Pivots*.

### 7. 💼 Portfolio Allocation Optimizer & Kas Siaga
* **20% Kas Siaga Wajib**: Mengunci alokasi kas minimal 20% untuk bantalan likuiditas (*liquidity buffer*).
* **Cap-and-Redistribute Algorithm**: Membatasi bobot maksimal saham tunggal ($\le 25\%$) demi mencegah konsentrasi risiko berlebih.
* **IDX Tick Size Rounding**: Mengonversi harga masuk (*Entry*), target laba (*TP*), dan batas rugi (*Stop Loss*) mengikuti fraksi harga resmi BEI (Rp 1, Rp 2, Rp 5, Rp 10, Rp 25).

---

## 🏛️ Cakupan 11 Sektor Resmi Bursa Efek Indonesia (IDX-IC)

1. **Financials (Keuangan)**: BBCA, BBRI, BMRI, BBNI, BRIS, BBTN
2. **Energy (Energi)**: ADRO, PTBA, ITMG, PGAS, MEDC, AKRA, HRUM, INDY, AADI, ADMR
3. **Basic Materials (Barang Baku)**: ANTM, MDKA, INCO, TPIA, BRPT, INKP, AMMN, MBMA
4. **Consumer Non-Cyclicals (Konsumer Primer)**: ICBP, INDF, UNVR, AMRT, MYOR, CPIN, SIDO
5. **Consumer Cyclicals (Konsumer Non-Primer)**: ASII, ACES, MAPI, ERAA
6. **Healthcare (Kesehatan)**: KLBF, MIKA, HEAL, SILO
7. **Technology (Teknologi)**: GOTO, EMTK, BUKA
8. **Infrastructures (Infrastruktur & Telekomunikasi)**: TLKM, ISAT, EXCL, TOWR, TBIG, PGEO, BREN
9. **Properties & Real Estate (Properti)**: BSDE, CTRA, PWON, SMRA
10. **Industrials (Perindustrian)**: UNTR, HEXA, AUTO, SMSM
11. **Transportation & Logistics (Transportasi & Logistik)**: BIRD, SMDR, ASSA

---

## 📂 Struktur Direktori & File (Directory Architecture)

```text
stock-market-recommendation/
├── .agents/rules/alphatech.md         # Kaidah doktrin sistem AI AlphaTech
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml         # Otomasi GitHub Actions harian (05:00 WIB)
├── api/
│   ├── index.py                       # Serverless handler untuk Vercel
│   └── main.py                        # FastAPI REST API, routing, and dashboard server
├── data/
│   ├── raw/                           # Ingestion data pasar mentah
│   │   ├── raw_market_data.csv        # Data OHLCV 66 saham aktif (2020 - sekarang)
│   │   ├── benchmark_market_data.csv  # Data historis IHSG (^JKSE)
│   │   ├── global_macro_data.csv      # Snapshot penutupan makro harian
│   │   ├── historical_macro_data.csv  # Time-series harian multi-tahun TNX, USD/IDR, Brent, SP500
│   │   ├── fundamental_financial_data.csv # Snapshot rasio keuangan fundamental
│   │   └── financial_statements/      # Laporan keuangan per emiten
│   └── processed/                     # Hasil komputasi kuantitatif
│       ├── processed_market_features.csv      # Matriks 60+ fitur teknikal & makro (gitignored)
│       ├── advanced_quant_metrics.csv         # Metrik Sharpe, Beta, VaR, Pivots
│       ├── active_universe.json               # State semesta dinamis N=66 emiten
│       ├── alpha_model.joblib                 # Bobot model GBDT tersimpan
│       ├── lstm_model.pth                     # Bobot PyTorch LSTM 12-dimensi
│       ├── latest_morning_brief.json          # Hasil riset editorial Morning Brief Gemini AI
│       ├── latest_portfolio_allocation.csv    # Rekomendasi bobot alokasi modal & kas
│       ├── latest_alpha_recommendations_swing.csv     # Rekomendasi Swing Trader (LQ45)
│       ├── latest_alpha_recommendations_dividend.csv  # Rekomendasi Dividend & Value
│       └── latest_alpha_recommendations_favorites.csv # Rekomendasi Portofolio Pilihan
├── src/
│   ├── __init__.py
│   ├── config.py                      # Konfigurasi semesta saham, sektor, & parameter
│   ├── universe_manager.py            # Dynamic Universe Manager & Circuit Breaker
│   ├── 01_data_ingestion.py           # Engine penarikan OHLCV, macro & fundamental
│   ├── 02_feature_eng.py              # Ekstraksi fitur, GARCH(1,1), Excess Alpha, Makro
│   ├── 03_model_inference.py          # Ensemble GBDT+LSTM 12D+ARIMA & Top Decile
│   └── morning_brief.py               # Generator Morning Brief Gemini 3.8 Flash
├── index.html                         # Dashboard web modern responsif (TradingView & Analytics)
├── login.html                         # Halaman login antarmuka pengguna
├── main.py                            # Master runner pipeline 4 langkah
├── run.bat                            # Skrip eksekusi satu klik untuk Windows
├── run_daily.ps1                      # Skrip eksekusi harian untuk PowerShell
├── run_daily.sh                       # Skrip eksekusi harian untuk Linux / macOS
├── requirements.txt                   # Dependensi pustaka Python
├── vercel.json                        # Konfigurasi deployment serverless Vercel
├── .env                               # Kunci API lokal (GEMINI_API_KEY)
└── README.md                          # Dokumentasi resmi proyek
```

---

## 🛠️ Instalasi & Menjalankan Lokal (Quickstart Guide)

### 1. Prasyarat Sistem
* **Python 3.10** atau versi lebih baru (diuji pada Python 3.10 – 3.13 di Windows, Linux, dan macOS).
* Koneksi internet aktif untuk penarikan data bursa terkini via Yahoo Finance API.

### 2. Kloning & Pemasangan Dependensi
```bash
# Masuk ke direktori repositori
cd stock-market-recommendation

# Buat virtual environment (disarankan)
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Pasang dependensi pustaka
pip install -r requirements.txt
```

### 3. Konfigurasi Kunci API (`.env`)
Buat file `.env` di root direktori proyek:
```ini
GEMINI_API_KEY=AIzaSy... (Kunci API Google Gemini Anda)
```

### 4. Menjalankan Master Pipeline Kuantitatif
Untuk menjalankan seluruh tahapan komputasi (*Ingestion $\rightarrow$ Feature Engineering $\rightarrow$ Model Inference & Top Decile $\rightarrow$ Morning Brief AI*):

```bash
python main.py
```
*Atau di Windows cukup klik ganda file `run.bat` atau jalankan via PowerShell `.\run_daily.ps1`.*

### 5. Menjalankan Server Dashboard & REST API
```bash
uvicorn api.main:app --reload --port 8000
```
Buka peramban (*browser*) Anda di:
* **Dashboard Web Interaktif**: [http://localhost:8000/](http://localhost:8000/)
* **Dokumentasi Interaktif Swagger API**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Dokumentasi Alternatif Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🌐 Dokumentasi Endpoint REST API

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Menampilkan antarmuka Dashboard Web interaktif. |
| `GET` | `/morning-brief` | Mengembalikan editorial Morning Brief IHSG terkini, berita makro, dan snapshot pasar global. |
| `GET` | `/portfolio/allocate` | Menghitung alokasi modal optimal berdasarkan input nominal modal (`?capital=50000000`). |
| `GET` | `/recommendations` | Mengembalikan daftar rekomendasi saham (`?universe=swing`, `dividend`, `favorites`, atau `sector=Financials`). |
| `GET` | `/models/compare/{ticker}` | Menampilkan perbandingan probabilitas model (GBDT vs LSTM vs ARIMA) untuk emiten tertentu. |
| `GET` | `/sectors` | Daftar 11 sektor resmi IDX dan daftar kode saham aktif. |
| `GET` | `/status` | Informasi status kesehatan sistem dan waktu pembaruan pipeline terakhir. |
| `POST`| `/pipeline/run` | Menjalankan ulang seluruh pipeline kuantitatif di background secara asinkron. |

---

## 👥 Tim Riset Kuantitatif & Rekayasa Sistem (TIM New York)

Platform ini dikembangkan dan dikelola secara kolaboratif oleh **TIM New York**:

| Nama Kontributor | Peran & Tanggung Jawab Utama | Profil GitHub |
| :--- | :--- | :--- |
| **Ifan Apres** | *Lead Quantitative Engineer & Fullstack Systems Architect* — Bertanggung jawab atas arsitektur komputasi, pipeline data kuantitatif, model PyTorch LSTM multi-dimensi, integrasi target excess alpha, sistem deployment Vercel & CI/CD automation. | [@Ifan-Apres](https://github.com/Ifan-Apres) |
| **Sekar Widhastri** | *Senior Market & Research Analyst* — Bertanggung jawab atas formulasi strategi analisis pasar ekonometrika, metodologi risk-parity alokasi portofolio, evaluasi sinyal teknikal BEI, dan kurasi editorial riset pasar. | [@sekarwidhastri](https://github.com/sekarwidhastri) |

---

## ⚖️ Disclaimer & Batasan Tanggung Jawab

*Aplikasi ini dikembangkan untuk tujuan riset kuantitatif, analisis data, dan edukasi finansial. Seluruh rekomendasi yang dihasilkan oleh model machine learning dan kecerdasan buatan merupakan indikator probabilitas statistik pasar dan bukan merupakan anjuran mutlak untuk membeli atau menjual efek tertentu. Keputusan investasi dan manajemen risiko sepenuhnya berada di tangan investor masing-masing.*
