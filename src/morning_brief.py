import importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv  # type: ignore # pyrefly: ignore [missing-import]
import google.generativeai as genai  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    ADVANCED_METRICS_FILE,
    BENCHMARK_DATA_FILE,
    GLOBAL_MACRO_FILE,
    LOG_FORMAT,
    MORNING_BRIEF_FILE,
    PROCESSED_DATA_FILE,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("MorningBriefGenerator")


class MorningBriefGenerator:
    """
    Automated Institutional Market Morning Brief Generator.
    Aggregates Wall Street closes, US Treasury yields, global oil prices, Rupiah exchange rate,
    and IHSG technical levels, augmented with real-time news scraping and synthesized via Gemini 3.8 Flash.
    """

    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.doctrine = self.load_alphatech_doctrine()

    def load_alphatech_doctrine(self) -> str:
        """
        Loads AlphaTech.antigravityrules quantitative doctrine to instruct Gemini 3.8 Flash
        as 'New York' - an elite Quantitative Investment Manager, ML Engineer, and Business Strategist.
        """
        rules_path = ROOT_DIR / "AlphaTech.antigravityrules"
        if rules_path.exists():
            try:
                content = rules_path.read_text(encoding="utf-8").strip()
                if content:
                    logger.info("AlphaTech.antigravityrules successfully loaded as AI System Doctrine.")
                    return content
            except Exception as e:
                logger.warning(f"Could not read AlphaTech rules file: {e}")

        return (
            "You are 'New York', an elite AI agent combining the expertise of a Quantitative Investment Manager, "
            "a Senior Fullstack/Machine Learning Engineer, and a Business Strategist. "
            "Your primary objective is to assist in analyzing financial markets, evaluating macroeconomic transmissions, "
            "and ensuring all insights deliver measurable commercial value and risk-adjusted alpha for institutional fund managers. "
            "Frameworks: Commercial Acumen & Business Strategy, Quantitative Investment & Market Analysis "
            "(Sharpe ratio, max drawdown, risk-to-reward, IHSG equities, dividend yields, order book dynamics, and econophysics crowd behavior / statistical mechanics distributions for collective sentiment), "
            "Machine Learning & Data Engineering, and Fullstack Scalability. "
            "Tone: Analytical, sharp, pragmatic, heavily data-driven, precise financial terminology, concise and direct."
        )

    def collect_macro_news(self) -> Dict[str, List[str]]:
        """
        Scrapes real-time headlines and summaries for global macro catalysts:
        - S&P 500 (^GSPC): Wall Street market drivers, Tech earnings, Fed sentiment
        - Brent Crude Oil (BZ=F): Geopolitics, OPEC+ supply, global energy tensions
        - DXY & US 10Y Yield (DX-Y.NYB, ^TNX): Dollar Index, Treasury yields, Forex drivers
        """
        news_data: Dict[str, List[str]] = {"sp500": [], "oil": [], "forex": []}
        try:
            import yfinance as yf

            def fetch_ticker_news(ticker_symbol: str, limit: int = 4) -> List[str]:
                results: List[str] = []
                try:
                    t = yf.Ticker(ticker_symbol)
                    raw_news = t.news or []
                    for item in raw_news[:limit]:
                        content = item.get("content", {}) if isinstance(item, dict) else {}
                        title = content.get("title") or item.get("title")
                        summary = content.get("summary") or content.get("description") or item.get("summary") or ""
                        if title:
                            clean_t = str(title).strip()
                            clean_s = str(summary).strip()[:180]
                            entry = f"{clean_t} - {clean_s}" if clean_s else clean_t
                            results.append(entry)
                except Exception as ex:
                    logger.debug(f"Could not fetch news feed for {ticker_symbol}: {ex}")
                return results

            news_data["sp500"] = fetch_ticker_news("^GSPC", limit=4)
            news_data["oil"] = fetch_ticker_news("BZ=F", limit=4)
            dxy_news = fetch_ticker_news("DX-Y.NYB", limit=2)
            tnx_news = fetch_ticker_news("^TNX", limit=2)
            news_data["forex"] = dxy_news + tnx_news

            logger.info(
                f"Scraped real-time macro news: S&P 500 ({len(news_data['sp500'])} items), "
                f"Brent Oil ({len(news_data['oil'])} items), Forex/DXY ({len(news_data['forex'])} items)."
            )
        except Exception as e:
            logger.warning(f"Error scraping real-time macro news: {e}")

        return news_data

    def collect_market_snapshot(self) -> Dict[str, Any]:
        """
        Gathers overnight global market closes and IHSG technical status using Floor Pivots and real data.
        """
        if not BENCHMARK_DATA_FILE.exists():
            raise FileNotFoundError(f"Benchmark file {BENCHMARK_DATA_FILE} not found. Ingestion must run first.")

        b_df = pd.read_csv(BENCHMARK_DATA_FILE)
        if b_df.empty:
            raise ValueError(f"Benchmark file {BENCHMARK_DATA_FILE} is empty.")

        last_row = b_df.iloc[-1]
        prev_row = b_df.iloc[-2] if len(b_df) > 1 else last_row
        close = round(float(last_row["Close"]), 2)
        prev_close = float(prev_row["Close"])
        chg_pct = round(((close - prev_close) / prev_close) * 100.0, 2)
        date_str = str(last_row["Date"])[:10]

        high = float(last_row.get("High", close))
        low = float(last_row.get("Low", close))

        # Standard Institutional Floor Pivots (P, S1, R1, S2, R2)
        pivot = (high + low + close) / 3.0
        s1 = (2.0 * pivot) - high
        r1 = (2.0 * pivot) - low
        s2 = pivot - (high - low)
        r2 = pivot + (high - low)

        ma20 = round(float(b_df["Close"].rolling(20, min_periods=5).mean().iloc[-1]), 2)

        snapshot: Dict[str, Any] = {
            "date": date_str,
            "ihsg_close": close,
            "ihsg_change_pct": chg_pct,
            "ihsg_ma20": ma20,
            "ihsg_pivot": round(pivot, 2),
            "ihsg_support": f"{int(round(s1)):,} - {int(round(pivot)):,}",
            "ihsg_resistance": f"{int(round(pivot)):,} - {int(round(r1)):,}",
            "ihsg_s1": int(round(s1)),
            "ihsg_r1": int(round(r1)),
            "ihsg_s2": int(round(s2)),
            "ihsg_r2": int(round(r2)),
            "sp500_change": "Data Belum Tersedia",
            "nasdaq_change": "Data Belum Tersedia",
            "dow_change": "Data Belum Tersedia",
            "ust_10y_yield": "Data Belum Tersedia",
            "brent_oil": "Data Belum Tersedia",
            "usd_idr": "Data Belum Tersedia",
            "sp500_close": None,
            "brent_oil_close": None,
            "usd_idr_close": None,
        }

        # Read actual global macro data if available
        macro_loaded = False
        if GLOBAL_MACRO_FILE.exists():
            try:
                m_df = pd.read_csv(GLOBAL_MACRO_FILE)
                if not m_df.empty:
                    for asset in ["SP500", "Nasdaq", "DowJones"]:
                        sub = m_df[m_df["Asset_Name"] == asset]
                        if len(sub) >= 2:
                            c1 = float(sub.iloc[-1]["Close"])
                            c0 = float(sub.iloc[-2]["Close"])
                            pct = round(((c1 - c0) / c0) * 100.0, 2)
                            snapshot[f"{asset.lower()}_change"] = f"{'+' if pct > 0 else ''}{pct}%"
                            if asset == "SP500":
                                snapshot["sp500_close"] = c1

                    oil_sub = m_df[m_df["Asset_Name"] == "Oil_Brent"]
                    if not oil_sub.empty:
                        oil_val = float(oil_sub.iloc[-1]["Close"])
                        snapshot["brent_oil"] = f"US$ {oil_val:.2f} / barel"
                        snapshot["brent_oil_close"] = oil_val

                    usd_sub = m_df[m_df["Asset_Name"] == "USD_IDR"]
                    if not usd_sub.empty:
                        usd_val = float(usd_sub.iloc[-1]["Close"])
                        snapshot["usd_idr"] = f"Rp {usd_val:,.0f} / US$"
                        snapshot["usd_idr_close"] = usd_val

                    ust_sub = m_df[m_df["Asset_Name"] == "US_Treasury_10Y"]
                    if not ust_sub.empty:
                        snapshot["ust_10y_yield"] = f"{float(ust_sub.iloc[-1]['Close']):.2f}%"
                    macro_loaded = True
            except Exception as e:
                logger.warning(f"Error reading macro file: {str(e)}")

        # If macro dataset was absent or missing change data, attempt on-the-fly fetch using yfinance
        if not macro_loaded or snapshot["sp500_change"] == "Data Belum Tersedia":
            try:
                import yfinance as yf

                macro_map = {
                    "SP500": "^GSPC",
                    "Nasdaq": "^IXIC",
                    "DowJones": "^DJI",
                    "US_Treasury_10Y": "^TNX",
                    "Oil_Brent": "BZ=F",
                    "USD_IDR": "USDIDR=X",
                }
                live_records = []
                for name, symbol in macro_map.items():
                    try:
                        ticker_data = yf.download(symbol, period="5d", progress=False)
                        if not ticker_data.empty:
                            closes = ticker_data["Close"].dropna().values.flatten()
                            if len(closes) >= 2:
                                c1 = float(closes[-1])
                                c0 = float(closes[-2])
                                pct = round(((c1 - c0) / c0) * 100.0, 2)
                                if name in ["SP500", "Nasdaq"]:
                                    snapshot[f"{name.lower()}_change"] = f"{'+' if pct > 0 else ''}{pct}%"
                                    if name == "SP500":
                                        snapshot["sp500_close"] = c1
                                elif name == "DowJones":
                                    snapshot["dow_change"] = f"{'+' if pct > 0 else ''}{pct}%"
                                elif name == "Oil_Brent":
                                    snapshot["brent_oil"] = f"US$ {c1:.2f} / barel"
                                    snapshot["brent_oil_close"] = c1
                                elif name == "USD_IDR":
                                    snapshot["usd_idr"] = f"Rp {c1:,.0f} / US$"
                                    snapshot["usd_idr_close"] = c1
                                elif name == "US_Treasury_10Y":
                                    snapshot["ust_10y_yield"] = f"{c1:.2f}%"

                                live_records.append(
                                    {
                                        "Date": str(ticker_data.index[-1])[:10],
                                        "Asset_Name": name,
                                        "Symbol": symbol,
                                        "Close": c1,
                                        "Volume": 0,
                                    }
                                )
                    except Exception as ex:
                        logger.warning(f"Could not live-fetch {name}: {str(ex)}")

                if live_records:
                    saved_mdf = pd.DataFrame(live_records)
                    GLOBAL_MACRO_FILE.parent.mkdir(parents=True, exist_ok=True)
                    saved_mdf.to_csv(GLOBAL_MACRO_FILE, index=False)
            except ImportError:
                pass

        # Collect scraped news headlines
        snapshot["news"] = self.collect_macro_news()

        return snapshot

    def _generate_fallback_content(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates robust, dynamic, institutional-grade fallback cards and narrative if Gemini API is offline.
        """
        sp_change_str = str(snapshot.get("sp500_change", ""))
        is_sp_bullish = "+" in sp_change_str or (not sp_change_str.startswith("-") and "%" in sp_change_str)
        oil_str = str(snapshot.get("brent_oil", ""))
        usd_str = str(snapshot.get("usd_idr", ""))

        # 1. S&P 500 dynamic fallback
        if is_sp_bullish:
            sp500_badge = "Katalis Risk-On Wall Street"
            sp500_desc = (
                f"Penguatan Wall Street ({snapshot['sp500_change']}) terdorong performa emiten teknologi AS, "
                f"membuka ruang sentimen positif dan potensi aliran modal asing (inflow) ke saham blue-chip IHSG."
            )
        else:
            sp500_badge = "Sikap Waspada Wall Street"
            sp500_desc = (
                f"Koreksi Wall Street ({snapshot['sp500_change']}) mencerminkan antisipasi pasar terhadap kebijakan The Fed, "
                f"berpotensi memicu volatilitas jangka pendek pada saham berkapitalisasi besar di BEI."
            )

        # 2. Brent Oil dynamic fallback
        brent_badge = "Dinamika Geopolitik Pasokan"
        brent_desc = (
            f"Minyak mentah Brent berada di {oil_str}. Isu tensi geopolitik Timur Tengah dan kebijakan produksi OPEC+ "
            f"menopang harga komoditas energi, memberikan katalis bagi sektor tambang/migas (MEDC, ENRG) namun menambah beban biaya manufaktur."
        )

        # 3. USD/IDR dynamic fallback
        usd_idr_badge = "Stabilitas Moneter Terjaga"
        usd_idr_desc = (
            f"Kurs Rupiah berada pada kisaran {usd_str}. Pergerakan dipengaruhi indeks Dolar AS (DXY) dan yield obligasi AS ({snapshot['ust_10y_yield']}), "
            f"dengan intervensi Bank Indonesia menopang likuiditas sektor perbankan dan pasar SBN domestik."
        )

        # 4. Comprehensive 4-5 paragraph narrative
        brief_body = (
            f"Untuk sesi perdagangan hari ini, Indeks Harga Saham Gabungan (IHSG) diperkirakan bergerak fluktuatif "
            f"dengan kecenderungan konsolidasi menguat pada rentang support {snapshot['ihsg_support']} hingga resistance {snapshot['ihsg_resistance']}. "
            f"Secara teknikal, posisi indeks berada di atas pivot harian {snapshot['ihsg_pivot']}, mencerminkan momentum akumulasi yang relatif terjaga.\n\n"
            f"Dari panggung global, Wall Street mencatatkan pergerakan {snapshot['sp500_change']} pada indeks S&P 500 dan {snapshot['nasdaq_change']} pada Nasdaq. "
            f"Katalis utama bersumber dari dinamika rilis laporan keuangan korporasi AS serta evaluasi pelaku pasar terhadap proyeksi arah suku bunga The Fed. "
            f"Sentimen ini memberikan efek rambatan (spillover effect) langsung terhadap minat risiko investor global di pasar negara berkembang, khususnya bursa domestik.\n\n"
            f"Di pasar komoditas, harga minyak mentah Brent bertengger di {oil_str}. Fluktuasi ini dipengaruhi ketegangan geopolitik internasional serta pembatasan kuota suplai global. "
            f"Bagi pasar modal Indonesia, level harga energi ini menjadi pisau bermata dua: menguntungkan pendapatan emiten energi dan migas, namun tetap perlu diwaspadai terhadap potensi tekanan inflasi impor bagi emiten konsumer dan transportasi.\n\n"
            f"Sementara itu dari sisi moneter, kurs Rupiah tercatat di level {usd_str} dengan yield US Treasury 10-Tahun berada di {snapshot['ust_10y_yield']}. "
            f"Kekuatan indeks dolar AS (DXY) memicu kehati-hatian pada arus dana portofolio, meski langkah stabilisasi Bank Indonesia diperkirakan menjaga stabilitas fundamental perbankan (BBCA, BBRI, BMRI).\n\n"
            f"Sebagai pertimbangan taktis, investor institusi dan profesional disarankan menerapkan strategi Selective Buy on Weakness pada saham-saham likuid berfundamental solid, "
            f"dengan tetap mematuhi disiplin level trailing stop loss pada rentang support krusial."
        )

        return {
            "sp500_badge": sp500_badge,
            "sp500_desc": sp500_desc,
            "brent_badge": brent_badge,
            "brent_desc": brent_desc,
            "usd_idr_badge": usd_idr_badge,
            "usd_idr_desc": usd_idr_desc,
            "brief_content": brief_body,
        }

    def generate_brief_text(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates editorial Morning Brief text and structured dynamic cards using Gemini 3.8 Flash.
        Incorporates scraped news for S&P 500, Brent Oil geopolitics, and USD/IDR drivers.
        """
        title = f"Morning Brief IHSG: Katalis Wall Street dan Arah Pasar Hari Ini ({snapshot['date']})"
        news_dict = snapshot.get("news", {})
        sp500_news = "\n".join([f"- {h}" for h in news_dict.get("sp500", [])[:4]]) or "- Sentimen umum bursa Wall Street"
        oil_news = "\n".join([f"- {h}" for h in news_dict.get("oil", [])[:4]]) or "- Dinamika pasar komoditas energi internasional"
        forex_news = "\n".join([f"- {h}" for h in news_dict.get("forex", [])[:4]]) or "- Pergerakan US Dollar Index dan Yield US Treasury"

        prompt = f"""
Bertindaklah sebagai "New York" (Senior Quantitative Investment Manager & Strategist) sesuai doktrin AlphaTech.
Susun laporan Morning Market Brief harian berstandar institusi untuk para manajer portofolio dan pelaku pasar profesional sebelum bel pembukaan Bursa Efek Indonesia (BEI) pagi ini.

Terapkan kerangka kerja AlphaTech secara konsisten:
1. Evaluasi profil risiko ketat (Risk-to-Reward ratio, volatilitas pasar, rentang support-resistance teknikal IHSG).
2. Analisis dinamika pasar modal Indonesia secara mendalam (likuiditas emiten Big Cap, arus dana asing/foreign flow, yield obligasi pemerintah SBN, dan transmisi nilai tukar).
3. Pendekatan makro & econophysics dalam membaca transmisi sentimen massa global (Wall Street, pasar energi minyak mentah, dan indeks dolar DXY) ke pasar domestik.
4. Gaya bahasa tajam, analitis, pragmatis, berbasis data, tanpa kalimat basa-basi atau asterisk ganda berlebihan.

Data Angka Pasar Hari Ini:
- Tanggal: {snapshot['date']}
- IHSG Terakhir: {snapshot['ihsg_close']} (Perubahan harian: {snapshot['ihsg_change_pct']}%)
- Estimasi Range Hari Ini: Support {snapshot['ihsg_support']} | Resistance {snapshot['ihsg_resistance']}
- Wall Street: S&P 500 ({snapshot['sp500_change']}), Nasdaq ({snapshot['nasdaq_change']}), Dow Jones ({snapshot['dow_change']})
- Yield US Treasury 10 Tahun: {snapshot['ust_10y_yield']}
- Minyak Mentah Brent: {snapshot['brent_oil']}
- Kurs Rupiah: {snapshot['usd_idr']}

Headline Berita & Isu Terkini Pasar Global (Hasil Scraping Real-Time):
- Isu S&P 500 & Wall Street:
{sp500_news}
- Isu Politik Luar Negeri & Pasar Minyak Brent:
{oil_news}
- Isu Indeks Dolar AS (DXY) & Yield Pasar Uang:
{forex_news}

INSTRUKSI WAJIB:
Analisis secara tajam hubungan sebab-akibat (causality) antara isu global tersebut dengan pasar modal Indonesia (IHSG).
Kembalikan respon HANYA dalam format JSON valid (tanpa teks pembuka atau markdown wrap ```json) dengan struktur:
{{
  "sp500_badge": "Label status singkat 2-4 kata (misal: Rally Big Tech AS / Reaksi Data Inflasi / Tekanan The Fed)",
  "sp500_desc": "Analisis isu terkini yang mempengaruhi nilai S&P 500 semalam dan mekanisme pengaruh langsungnya ke IHSG serta saham Big Cap (1-2 kalimat padat)",
  "brent_badge": "Label status geopolitik 2-4 kata (misal: Premi Risiko Geopolitik / Ketatnya Pasokan OPEC+ / Ekspektasi Permintaan)",
  "brent_desc": "Analisis isu politik luar negeri/geopolitik yang mempengaruhi harga minyak Brent dan pengaruhnya ke beban energi fiskal & emiten migas Indonesia (1-2 kalimat padat)",
  "usd_idr_badge": "Label status nilai tukar 2-4 kata (misal: Dolar AS Menguat Terbatas / Intervensi BI Terjaga / Tekanan DXY Kuat)",
  "usd_idr_desc": "Penyebab fluktuasi nilai tukar USD (DXY/Yield/Suku Bunga) dan pengaruhnya ke likuiditas pasar modal, obligasi, dan perbankan Indonesia (1-2 kalimat padat)",
  "brief_content": "Ulasan narasi riset pasar komprehensif (4 sampai 5 paragraf mengalir alami, membedah arah pembukaan IHSG, katalis Wall Street, isu geopolitik minyak, dinamika kurs USD/IDR, serta panduan taktis portofolio sebelum bel pembukaan BEI. JANGAN gunakan tanda bintang tebal ganda ** berlebihan)."
}}
"""

        parsed_data = None
        if self.api_key:
            try:
                # Menggunakan model Gemini 3.8 Flash yang didoktrin AlphaTech (New York)
                logger.info("Generating Morning Brief using Gemini 3.8 Flash indoctrinated with AlphaTech rules...")
                model = genai.GenerativeModel(
                    model_name="gemini-3.8-flash",
                    system_instruction=self.doctrine,
                )
                response = model.generate_content(prompt)
                raw_text = response.text.strip()

                # Clean markdown JSON wraps if present
                clean_json_str = raw_text
                if clean_json_str.startswith("```json"):
                    clean_json_str = clean_json_str[7:]
                if clean_json_str.startswith("```"):
                    clean_json_str = clean_json_str[3:]
                if clean_json_str.endswith("```"):
                    clean_json_str = clean_json_str[:-3]
                clean_json_str = clean_json_str.strip()

                parsed_data = json.loads(clean_json_str)
                logger.info("Successfully received and parsed structured brief from Gemini 3.8 Flash (AlphaTech Indoctrinated).")
            except Exception as e:
                logger.error(f"Failed to generate brief via Gemini 3.8 Flash API: {str(e)}")

        if not parsed_data:
            logger.info("Falling back to deterministic quantitative macro brief generator.")
            parsed_data = self._generate_fallback_content(snapshot)

        # Sanitize narrative content
        brief_body = str(parsed_data.get("brief_content", "")).strip()
        lines = brief_body.split("\n")
        cleaned_lines = []
        for line in lines:
            l_strip = line.strip()
            if l_strip.startswith("**INSTITUTIONAL") or l_strip.startswith("INSTITUTIONAL") or l_strip.startswith("**Tanggal"):
                continue
            cleaned_lines.append(line)
        brief_body = "\n".join(cleaned_lines).replace("**", "").replace("*", "").replace(" & ", " dan ").strip()

        sp500_badge = str(parsed_data.get("sp500_badge", "Katalis Pasar AS")).replace("**", "").strip()
        sp500_desc = str(parsed_data.get("sp500_desc", "Sentimen bursa AS memberikan dorongan awal bagi IHSG.")).replace("**", "").strip()
        brent_badge = str(parsed_data.get("brent_badge", "Dinamika Komoditas")).replace("**", "").strip()
        brent_desc = str(parsed_data.get("brent_desc", "Biaya energi dan inflasi manufaktur domestik terjaga.")).replace("**", "").strip()
        usd_idr_badge = str(parsed_data.get("usd_idr_badge", "Stabilitas Valuta")).replace("**", "").strip()
        usd_idr_desc = str(parsed_data.get("usd_idr_desc", "Stabilitas nilai tukar menopang arus modal asing.")).replace("**", "").strip()

        # Update snapshot with dynamic card content for full backward compatibility
        snapshot["sp500_badge"] = sp500_badge
        snapshot["sp500_desc"] = sp500_desc
        snapshot["brent_badge"] = brent_badge
        snapshot["brent_desc"] = brent_desc
        snapshot["usd_idr_badge"] = usd_idr_badge
        snapshot["usd_idr_desc"] = usd_idr_desc

        cards_data = {
            "sp500": {
                "value": snapshot["sp500_change"],
                "badge": sp500_badge,
                "desc": sp500_desc,
            },
            "brent": {
                "value": snapshot["brent_oil"],
                "badge": brent_badge,
                "desc": brent_desc,
            },
            "usd_idr": {
                "value": snapshot["usd_idr"],
                "badge": usd_idr_badge,
                "desc": usd_idr_desc,
            },
            "ihsg": {
                "value": f"{snapshot['ihsg_close']} ({'+' if snapshot['ihsg_change_pct'] >= 0 else ''}{snapshot['ihsg_change_pct']}%)",
                "range": f"Support {snapshot['ihsg_support']} • Resistance {snapshot['ihsg_resistance']}",
                "desc": "Fokus akumulasi pada saham likuid berkapitalisasi besar dengan disiplin level stop loss.",
            },
        }

        return {
            "date": snapshot["date"],
            "headline": title.replace("&", "dan"),
            "cards": cards_data,
            "brief_content": brief_body,
            "snapshot": snapshot,
        }

    def execute_and_save(self) -> Dict[str, Any]:
        snapshot = self.collect_market_snapshot()
        brief_data = self.generate_brief_text(snapshot)

        MORNING_BRIEF_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MORNING_BRIEF_FILE, "w", encoding="utf-8") as f:
            json.dump(brief_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Morning Brief successfully saved to {MORNING_BRIEF_FILE}")
        return brief_data


def run_morning_brief_pipeline() -> Dict[str, Any]:
    generator = MorningBriefGenerator()
    return generator.execute_and_save()


if __name__ == "__main__":
    run_morning_brief_pipeline()
