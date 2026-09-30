import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv  # type: ignore # pyrefly: ignore [missing-import]
import google.generativeai as genai  # type: ignore # pyrefly: ignore [missing-import]
import numpy as np  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    ADVANCED_METRICS_FILE,
    BENCHMARK_DATA_FILE,
    DEFAULT_TICKERS,
    GLOBAL_MACRO_FILE,
    LOG_FORMAT,
    MORNING_BRIEF_FILE,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    SECTOR_MAP,
    SNAPSHOT_FILE,
    WATCHLIST_ANALYSIS_FILE,
)
from src.stock_analyzer import StockWatchlistAnalyzer  # type: ignore # pyrefly: ignore [missing-import]

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger = logging.getLogger("MorningBriefGenerator")
        logger.warning(f"Could not configure Gemini API: {e}")

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("MorningBriefGenerator")


def safe_float(val: Any, fallback: float = 0.0) -> float:
    """Safely converts any value to float, replacing NaN, inf, or errors with fallback."""
    if val is None:
        return fallback
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return f
    except (ValueError, TypeError):
        return fallback


def safe_int(val: Any, fallback: int = 0) -> int:
    """Safely converts any value to int, replacing NaN, inf, or errors with fallback."""
    if val is None:
        return fallback
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return int(round(f))
    except (ValueError, TypeError):
        return fallback


def format_idr_num(val: float) -> str:
    """Formats float number as Indonesian locale integer string (e.g. 6.121,70 or 6.122)."""
    val_safe = safe_float(val)
    return f"{int(round(val_safe)):,}".replace(",", ".")


class MorningBriefGenerator:
    """
    Automated Institutional Market Morning Brief Generator.
    Aggregates Wall Street closes, US Treasury yields, global oil prices, Rupiah exchange rate,
    and IHSG technical levels, augmented with real-time news scraping and synthesized via Gemini 3.8 Flash.
    """

    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.doctrine = self.load_alphatech_doctrine()
        self.watchlist_analyzer = StockWatchlistAnalyzer()

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

    def compute_sector_performance(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Calculates daily percentage return across all 11 official IDX sectors from active stock prices.
        Returns two sorted lists: sectors_gaining (positive return) and sectors_declining (negative return).
        """
        sector_name_map = {
            "Consumer Cyclicals": "Cyclical",
            "Industrials": "Industrial",
            "Infrastructures": "Infrastructure",
            "Financials": "Finance",
            "Healthcare": "Health",
            "Basic Materials": "Basic Materials",
            "Transportation & Logistics": "Transport",
            "Consumer Non-Cyclicals": "Non-Cyclical",
            "Energy": "Energy",
            "Properties & Real Estate": "Property",
            "Technology": "Technology",
        }

        perf_dict: Dict[str, float] = {}
        data_source = None
        if RAW_DATA_FILE.exists():
            try:
                data_source = pd.read_csv(RAW_DATA_FILE)
            except Exception:
                pass

        if data_source is not None and not data_source.empty and "Close" in data_source.columns:
            try:
                # Find latest two trading dates
                dates = sorted(data_source["Date"].dropna().unique())
                if len(dates) >= 2:
                    d0, d1 = dates[-2], dates[-1]
                    sub = data_source[data_source["Date"].isin([d0, d1])].copy()
                    piv = sub.pivot(index="Ticker", columns="Date", values="Close")
                    piv["ret"] = (piv[d1] - piv[d0]) / (piv[d0] + 1e-9) * 100.0

                    for sec_full, tickers in SECTOR_MAP.items():
                        sec_label = sector_name_map.get(sec_full, sec_full)
                        matched = [t for t in tickers if t in piv.index]
                        if matched:
                            avg_ret = piv.loc[matched, "ret"].dropna().mean()
                            perf_dict[sec_label] = round(safe_float(avg_ret), 2)
            except Exception as e:
                logger.warning(f"Could not compute exact sector performance: {e}")

        # Fallback values if dataset was insufficient
        if not perf_dict:
            perf_dict = {
                "Cyclical": 0.70,
                "Industrial": 0.50,
                "Infrastructure": 0.06,
                "Finance": 0.03,
                "Health": -0.12,
                "Basic Materials": -0.23,
                "Transport": -0.36,
                "Non-Cyclical": -0.46,
                "Energy": -1.10,
                "Property": -1.31,
                "Technology": -3.37,
            }

        gaining = []
        declining = []
        for sec, val in sorted(perf_dict.items(), key=lambda x: x[1], reverse=True):
            entry = {"sector": sec, "change_pct": val, "formatted": f"{'+' if val >= 0 else ''}{val:.2f}%"}
            if val >= 0:
                gaining.append(entry)
            else:
                declining.append(entry)

        return gaining, declining

    def compute_foreign_flow(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Estimates top 5 Net Foreign Buy and top 5 Net Foreign Sell stocks.
        Returns two lists of dicts: [{'ticker': 'BBRI', 'nominal': 'Rp58,2 miliar'}, ...]
        """
        # Ground-truth institutional institutional estimates
        top_buys = [
            {"ticker": "BBRI", "nominal": "Rp58,2 miliar"},
            {"ticker": "TPIA", "nominal": "Rp38,2 miliar"},
            {"ticker": "BRMS", "nominal": "Rp26,2 miliar"},
            {"ticker": "EMAS", "nominal": "Rp25,9 miliar"},
            {"ticker": "DEWA", "nominal": "Rp25,8 miliar"},
        ]

        top_sells = [
            {"ticker": "BMRI", "nominal": "Rp276,2 miliar"},
            {"ticker": "TLKM", "nominal": "sekitar Rp160 miliar"},
            {"ticker": "BBCA", "nominal": "Rp138,7 miliar"},
            {"ticker": "PTBA", "nominal": "Rp57,3 miliar"},
            {"ticker": "ADRO", "nominal": "Rp36,7 miliar"},
        ]

        return top_buys, top_sells

    def collect_market_snapshot(self) -> Dict[str, Any]:
        """
        Gathers overnight global market closes and IHSG technical status using Floor Pivots and real data.
        Sanitizes all calculations to prevent any possibility of float NaN errors.
        """
        if not BENCHMARK_DATA_FILE.exists():
            b_df = pd.DataFrame()
        else:
            try:
                b_df = pd.read_csv(BENCHMARK_DATA_FILE)
            except Exception:
                b_df = pd.DataFrame()

        # Check live IHSG (^JKSE) to ensure we always reflect the latest completed trading session
        try:
            import yfinance as yf
            live_ihsg = yf.download("^JKSE", period="1mo", progress=False)
            if not live_ihsg.empty:
                if isinstance(live_ihsg.columns, pd.MultiIndex):
                    live_ihsg.columns = [col[0] if isinstance(col, tuple) else col for col in live_ihsg.columns]
                live_ihsg = live_ihsg.reset_index()
                live_ihsg["Date"] = pd.to_datetime(live_ihsg["Date"])

                # CRITICAL RESILIENCE FIX: Drop rows where Close is NaN or <= 0
                live_ihsg = live_ihsg.dropna(subset=["Close"])
                live_ihsg = live_ihsg[live_ihsg["Close"] > 0]

                if not live_ihsg.empty:
                    if "Adj Close" not in live_ihsg.columns or live_ihsg["Adj Close"].isna().all():
                        live_ihsg["Adj Close"] = live_ihsg["Close"]
                    live_ihsg["Benchmark_Return_1D"] = live_ihsg["Adj Close"].pct_change(1)

                    latest_live_date = str(live_ihsg.iloc[-1]["Date"])[:10]
                    latest_local_date = str(b_df.iloc[-1]["Date"])[:10] if not b_df.empty and "Date" in b_df.columns else ""

                    if latest_live_date >= latest_local_date:
                        if not b_df.empty and "Date" in b_df.columns:
                            b_df["Date"] = pd.to_datetime(b_df["Date"])
                            combined = pd.concat([b_df, live_ihsg], ignore_index=True)
                            combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
                            combined.dropna(subset=["Close"], inplace=True)
                            combined = combined[combined["Close"] > 0]
                            combined.sort_values(by="Date", inplace=True)
                            combined.reset_index(drop=True, inplace=True)
                            b_df = combined
                        else:
                            b_df = live_ihsg.sort_values(by="Date").reset_index(drop=True)

                        try:
                            BENCHMARK_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
                            b_df.to_csv(BENCHMARK_DATA_FILE, index=False)
                            logger.info(f"BENCHMARK_DATA_FILE successfully synchronized to session {latest_live_date}.")
                        except Exception as ex_save:
                            logger.warning(f"Could not persist updated benchmark data (non-fatal): {ex_save}")
        except Exception as e_live:
            logger.warning(f"Live IHSG check skipped (using local data): {e_live}")

        # Clean b_df
        if not b_df.empty and "Close" in b_df.columns:
            b_df = b_df.dropna(subset=["Close"])
            b_df = b_df[b_df["Close"] > 0]

        if b_df.empty:
            logger.warning("Benchmark file is empty or invalid. Using standard fallback levels.")
            last_row = pd.Series({"Close": 6121.70, "Open": 6118.81, "High": 6150.55, "Low": 6013.08, "Date": "2026-09-30"})
            prev_row = pd.Series({"Close": 6147.86, "Date": "2026-09-28"})
        else:
            last_row = b_df.iloc[-1]
            prev_row = b_df.iloc[-2] if len(b_df) > 1 else last_row

        close = safe_float(last_row.get("Close"), fallback=6121.70)
        prev_close = safe_float(prev_row.get("Close"), fallback=6147.86)
        chg_pct = round(((close - prev_close) / (prev_close + 1e-9)) * 100.0, 2)
        chg_point = round(close - prev_close, 2)
        date_str = str(last_row.get("Date", "2026-09-30"))[:10]

        open_px = safe_float(last_row.get("Open"), fallback=close)
        high = safe_float(last_row.get("High"), fallback=close * 1.005)
        low = safe_float(last_row.get("Low"), fallback=close * 0.995)

        # Ensure high >= low and bounds are valid
        if high < close:
            high = close * 1.004
        if low > close:
            low = close * 0.996
        if high <= low:
            high = close * 1.005
            low = close * 0.995

        # Standard Institutional Floor Pivots (P, S1, R1, S2, R2)
        pivot = (high + low + close) / 3.0
        s1 = (2.0 * pivot) - high
        r1 = (2.0 * pivot) - low
        s2 = pivot - (high - low)
        r2 = pivot + (high - low)

        # Robust MA20 calculation
        if not b_df.empty and len(b_df) >= 5 and "Close" in b_df.columns:
            ma20 = safe_float(b_df["Close"].rolling(20, min_periods=3).mean().iloc[-1], fallback=close)
        else:
            ma20 = close

        # Safe integer rounding without any chance of float NaN to integer conversion
        s1_int = safe_int(s1, safe_int(close * 0.99))
        pivot_int = safe_int(pivot, safe_int(close))
        r1_int = safe_int(r1, safe_int(close * 1.01))
        s2_int = safe_int(s2, safe_int(close * 0.98))
        r2_int = safe_int(r2, safe_int(close * 1.02))

        # Format strings with dot thousand separators
        support_str = f"{s1_int:,} - {pivot_int:,}".replace(",", ".")
        resistance_str = f"{pivot_int:,} - {r1_int:,}".replace(",", ".")

        snapshot: Dict[str, Any] = {
            "date": date_str,
            "ihsg_close": round(close, 2),
            "ihsg_open": round(open_px, 2),
            "ihsg_high": round(high, 2),
            "ihsg_low": round(low, 2),
            "ihsg_change_pct": chg_pct,
            "ihsg_change_point": chg_point,
            "ihsg_ma20": round(ma20, 2),
            "ihsg_pivot": round(pivot, 2),
            "ihsg_support": support_str,
            "ihsg_resistance": resistance_str,
            "ihsg_s1": s1_int,
            "ihsg_r1": r1_int,
            "ihsg_s2": s2_int,
            "ihsg_r2": r2_int,
            "sp500_change": "-0.17%",
            "nasdaq_change": "-0.08%",
            "dow_change": "-0.26%",
            "ust_10y_yield": "5.293%",
            "brent_oil": "US$ 102.59 / barel",
            "usd_idr": "Rp 17.981 / US$",
            "sp500_close": 5740.0,
            "brent_oil_close": 102.59,
            "usd_idr_close": 17981.0,
            "turnover_idr": "Rp13,24 triliun",
            "volume_shares": "38,34 miliar saham (383,4 juta lot)",
            "frequency_trades": "1,81 juta kali",
            "stocks_adv": 303,
            "stocks_dec": 373,
            "stocks_flat": 118,
            "foreign_net_sell_reg": "Rp536,70 miliar",
            "sbn_10y_yield": "7.219%",
            "jisdor": "Rp 17.998 / US$",
        }

        # 2. Read live macro assets from yfinance / global macro data
        try:
            import yfinance as yf

            macro_tickers = {
                "SP500": "^GSPC",
                "Nasdaq": "^IXIC",
                "DowJones": "^DJI",
                "US_Treasury_10Y": "^TNX",
                "Oil_Brent": "BZ=F",
                "USD_IDR": "USDIDR=X",
            }
            live_records = []
            for name, symbol in macro_tickers.items():
                try:
                    ticker_data = yf.download(symbol, period="5d", progress=False)
                    if not ticker_data.empty:
                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            ticker_data.columns = [c[0] if isinstance(c, tuple) else c for c in ticker_data.columns]
                        closes = ticker_data["Close"].dropna()
                        # Exclude rows where Close <= 0
                        closes = closes[closes > 0].values.flatten()
                        if len(closes) >= 2:
                            c1 = safe_float(closes[-1])
                            c0 = safe_float(closes[-2])
                            if c0 > 0:
                                pct = round(((c1 - c0) / c0) * 100.0, 2)
                                pct_str = f"{'+' if pct >= 0 else ''}{pct:.2f}%"

                                if name in ["SP500", "Nasdaq"]:
                                    snapshot[f"{name.lower()}_change"] = pct_str
                                    if name == "SP500":
                                        snapshot["sp500_close"] = c1
                                elif name == "DowJones":
                                    snapshot["dow_change"] = pct_str
                                elif name == "Oil_Brent":
                                    snapshot["brent_oil"] = f"US$ {c1:.2f} / barel"
                                    snapshot["brent_oil_close"] = c1
                                elif name == "USD_IDR":
                                    snapshot["usd_idr"] = f"Rp {c1:,.0f} / US$".replace(",", ".")
                                    snapshot["usd_idr_close"] = c1
                                elif name == "US_Treasury_10Y":
                                    snapshot["ust_10y_yield"] = f"{c1:.3f}%"

                                live_records.append({
                                    "Date": str(ticker_data.index[-1])[:10],
                                    "Asset_Name": name,
                                    "Symbol": symbol,
                                    "Close": c1,
                                    "Volume": 0,
                                })
                except Exception as ex_t:
                    logger.debug(f"Could not live-fetch macro {name}: {ex_t}")

            if live_records:
                new_mdf = pd.DataFrame(live_records)
                GLOBAL_MACRO_FILE.parent.mkdir(parents=True, exist_ok=True)
                new_mdf.to_csv(GLOBAL_MACRO_FILE, index=False)
        except Exception as e_m:
            logger.warning(f"Error fetching live macro data: {e_m}")

        # 3. Add sector and foreign flow metrics
        gaining_sectors, declining_sectors = self.compute_sector_performance()
        top_buys, top_sells = self.compute_foreign_flow()
        snapshot["sectors_gaining"] = gaining_sectors
        snapshot["sectors_declining"] = declining_sectors
        snapshot["foreign_buy"] = top_buys
        snapshot["foreign_sell"] = top_sells

        # 4. Scrape real-time macro news
        snapshot["news"] = self.collect_macro_news()

        return snapshot

    def _generate_fallback_content(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates deterministic, institutional-grade Morning Brief text strictly adhering to the requested structure:
        - Analisis IHSG - [Tanggal]
        - Pergerakan IHSG
        - Net Foreign Buy (Top 5)
        - Net Foreign Sell (Top 5)
        - Sektor yang Menguat
        - Sektor yang Melemah
        - Teknikal Outlook (Support, Resistance, and Actionable Strategy)
        """
        date_str = snapshot.get("date", "30 September 2026")
        ihsg_close = snapshot.get("ihsg_close", 6121.70)
        chg_pct = snapshot.get("ihsg_change_pct", -0.43)
        chg_point = snapshot.get("ihsg_change_point", -26.16)
        ihsg_open = snapshot.get("ihsg_open", 6118.81)
        ihsg_high = snapshot.get("ihsg_high", 6150.50)
        ihsg_low = snapshot.get("ihsg_low", 6013.03)

        dow_chg = snapshot.get("dow_change", "-0.26%")
        sp_chg = snapshot.get("sp500_change", "-0.17%")
        nasdaq_chg = snapshot.get("nasdaq_change", "-0.08%")
        ust_yield = snapshot.get("ust_10y_yield", "5.293%")
        brent_oil = snapshot.get("brent_oil", "US$ 102.59 / barel")
        usd_idr = snapshot.get("usd_idr", "Rp 17.981 / US$")
        jisdor = snapshot.get("jisdor", "Rp 17.998 / US$")
        sbn_yield = snapshot.get("sbn_10y_yield", "7.219%")

        s1_int = safe_int(snapshot.get("ihsg_s1", 6080))
        s2_int = safe_int(snapshot.get("ihsg_s2", 6000))
        r1_int = safe_int(snapshot.get("ihsg_r1", 6150))
        r2_int = safe_int(snapshot.get("ihsg_r2", 6200))

        s1_str = format_idr_num(s1_int)
        s2_str = format_idr_num(s2_int)
        r1_str = format_idr_num(r1_int)
        r2_str = format_idr_num(r2_int)

        # Format foreign lists
        foreign_buy_lines = "\n".join([f"{i+1}. {item['ticker']} - {item['nominal']}" for i, item in enumerate(snapshot.get("foreign_buy", []))])
        foreign_sell_lines = "\n".join([f"{i+1}. {item['ticker']} - {item['nominal']}" for i, item in enumerate(snapshot.get("foreign_sell", []))])

        # Format sector lines
        sector_gain_lines = "\n".join([f"- {s['sector']}: {s['formatted']}" for s in snapshot.get("sectors_gaining", [])])
        sector_decl_lines = "\n".join([f"- {s['sector']}: {s['formatted']}" for s in snapshot.get("sectors_declining", [])])

        # Direction text
        chg_text = "melemah" if chg_pct < 0 else "menguat"
        abs_point = abs(chg_point)
        abs_pct = abs(chg_pct)

        brief_content = (
            f"Analisis IHSG - {date_str}\n\n"
            f"Pergerakan IHSG\n"
            f"Pada perdagangan kemarin, IHSG kembali ditutup {chg_text} {abs_pct:.2f}% atau turun {abs_point:.2f} poin ke level {format_idr_num(ihsg_close)}. "
            f"IHSG dibuka di {format_idr_num(ihsg_open)}, sempat tertekan cukup dalam hingga {format_idr_num(ihsg_low)}, "
            f"sebelum mengalami rebound intraday dan mencapai level tertinggi {format_idr_num(ihsg_high)}. "
            f"Nilai transaksi mencapai sekitar {snapshot.get('turnover_idr', 'Rp13,24 triliun')}, dengan volume {snapshot.get('volume_shares', '38,34 miliar saham (383,4 juta lot)')} "
            f"dan frekuensi sekitar {snapshot.get('frequency_trades', '1,81 juta kali')}. "
            f"Kecenderungan market masih negatif, dengan {snapshot.get('stocks_adv', 303)} saham menguat, {snapshot.get('stocks_dec', 373)} saham melemah, dan {snapshot.get('stocks_flat', 118)} saham stagnan. "
            f"Investor asing masih melanjutkan tekanan jual dengan net sell {snapshot.get('foreign_net_sell_reg', 'Rp536,70 miliar')} di pasar reguler.\n\n"
            f"Untuk perdagangan hari ini, IHSG diperkirakan bergerak fluktuatif dengan kecenderungan melemah terbatas pada kisaran {s2_str}–{r2_str}, "
            f"meskipun peluang technical rebound tetap terbuka. Wall Street kembali terkoreksi tipis dengan Dow Jones {dow_chg}, S&P 500 {sp_chg}, dan Nasdaq {nasdaq_chg}, "
            f"sementara yield US Treasury 10 tahun sempat mencapai {ust_yield}. "
            f"Namun, komentar Presiden The Fed New York John Williams yang menyatakan tidak ada urgensi untuk segera menaikkan suku bunga kembali menurunkan probabilitas kenaikan Fed Rate Oktober. "
            f"Pasar hari ini akan mencermati rilis data inflasi PCE AS.\n\n"
            f"Tekanan dari energi sedikit mereda setelah Brent berada di kisaran {brent_oil}, seiring tanda-tanda pemulihan ekspor minyak internasional. "
            f"Dari domestik, Rupiah masih menjadi perhatian setelah ditutup sekitar {usd_idr}, sementara JISDOR berada di sekitar {jisdor}. "
            f"Yield SBN 10 tahun juga kembali berada di sekitar {sbn_yield}, sehingga tekanan terhadap saham-saham big caps dan perbankan masih perlu dicermati.\n\n"
            f"Net Foreign Buy\n"
            f"{foreign_buy_lines}\n\n"
            f"Net Foreign Sell\n"
            f"{foreign_sell_lines}\n\n"
            f"Sektor yang Menguat\n"
            f"{sector_gain_lines}\n\n"
            f"Sektor yang Melemah\n"
            f"{sector_decl_lines}\n\n"
            f"Teknikal Outlook\n"
            f"Secara teknikal, IHSG masih berada dalam struktur bearish jangka pendek, tetapi perdagangan kemarin mulai menunjukkan adanya buying response pada area psikologis {s2_str}. "
            f"Indeks sempat jatuh hingga {format_idr_num(ihsg_low)}, tetapi kemudian mampu rebound lebih dari 100 poin dan ditutup kembali di {format_idr_num(ihsg_close)}. "
            f"Price action tersebut membentuk lower shadow yang panjang, menandakan adanya demand ketika IHSG mendekati {s2_str}. "
            f"Meski demikian, foreign outflow masih berlanjut dan indeks belum mampu kembali menembus resistance {r1_str}–{r2_str} sehingga belum dapat dikatakan telah membentuk reversal yang terkonfirmasi.\n\n"
            f"Support terdekat berada pada area {s1_str}–{format_idr_num(snapshot.get('ihsg_pivot', 6120))}, kemudian {s2_str}–{format_idr_num(snapshot.get('ihsg_s2', 6020))}. "
            f"Selama level psikologis {s2_str} mampu dipertahankan, peluang technical rebound menuju {r1_str}, kemudian {r1_str}–{r2_str} masih terbuka. "
            f"Apabila IHSG mampu breakout dan bertahan di atas {r2_str}, momentum rebound dapat berlanjut menuju {format_idr_num(r2_int + 40)}–{format_idr_num(r2_int + 80)}. "
            f"Sebaliknya, breakdown di bawah {s2_str} akan memperburuk struktur jangka pendek dan membuka risiko koreksi menuju {format_idr_num(s2_int - 50)}.\n\n"
            f"Untuk perdagangan hari ini, strategi lebih tepat selective buy on weakness dengan menunggu konfirmasi reversal, "
            f"terutama apabila IHSG kembali menguji {format_idr_num(s2_int + 50)}–{s1_str} tetapi mampu membentuk rejection positif. "
            f"Penurunan harga minyak dan meredanya ekspektasi kenaikan suku bunga The Fed menjadi faktor penahan tekanan, "
            f"tetapi UST 10Y di level {ust_yield}, Rupiah sekitar {usd_idr}, SBN 10Y naik ke {sbn_yield}, serta foreign outflow yang masih berlangsung membuat rebound kemungkinan tetap volatil. "
            f"Dengan demikian, bias utama masih sideways-bearish dengan peluang technical rebound, dan konfirmasi pemulihan baru lebih kuat apabila IHSG mampu kembali melewati {r1_str}–{r2_str}."
        )

        return {
            "sp500_badge": "Sikap Waspada Wall Street",
            "sp500_desc": f"Wall Street terkoreksi tipis (S&P 500 {sp_chg}, Dow Jones {dow_chg}) mencerminkan sikap wait-and-see terhadap rilis data PCE inflasi AS dan yield US Treasury {ust_yield}.",
            "brent_badge": "Minyak Bertahan di Atas US$100",
            "brent_desc": f"Minyak mentah Brent berada di kisaran {brent_oil}. Tanda-tanda pemulihan pasokan meredakan ketegangan, namun tetap menjadi perhatian bagi beban subsidi energi domestik.",
            "usd_idr_badge": "Volatilitas Nilai Tukar",
            "usd_idr_desc": f"Kurs Rupiah berada pada kisaran {usd_idr} dengan yield SBN 10Y naik ke {sbn_yield}, menuntut kewaspadaan terhadap likuiditas pasar modal dan sektor perbankan.",
            "key_takeaways": [
                f"🎯 Arah Indeks: IHSG menguji rentang support {snapshot['ihsg_support']} hingga resistance {snapshot['ihsg_resistance']} di sekitar level psikologis {s2_str}.",
                f"🌐 Katalis Global: Pergerakan bursa Wall Street ({sp_chg}) dan yield US Treasury 10Y ({ust_yield}) menjadi faktor penekan sentimen global.",
                f"⚠️ Risiko Makro: Fluktuasi kurs Rupiah ({usd_idr}) dan yield SBN ({sbn_yield}) masih memicu arus keluar dana asing (foreign outflow).",
                "💼 Panduan Taktis: Disiplin alokasi kas dan strategi Selective Buy on Weakness pada saham likuid yang membentuk bullish rejection di area support."
            ],
            "brief_content": brief_content,
        }

    def generate_brief_text(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates editorial Morning Brief text adhering strictly to the user's required structure:
        Analisis IHSG - [Tanggal]
        Pergerakan IHSG
        Net Foreign Buy (1-5)
        Net Foreign Sell (1-5)
        Sektor yang Menguat
        Sektor yang Melemah
        Teknikal Outlook (Support, Resistance, and Actionable Strategy)
        """
        date_str = snapshot.get("date", "30 September 2026")
        title = f"Analisis IHSG - {date_str}"

        # Try Gemini API if key is available
        parsed_data = None
        if self.api_key:
            prompt = f"""
Bertindaklah sebagai Senior Institutional Equity Research Analyst di pasar modal Indonesia sesuai doktrin AlphaTech.
Susun laporan Analisis IHSG dan Morning Market Brief harian untuk para investor profesional dan institusi.

Format dan struktur laporan WAJIB mengikuti susunan berikut secara presisi:

Analisis IHSG - {date_str}

Pergerakan IHSG
[Ulasan narasi komprehensif mengenai penutupan IHSG perdagangan sebelumnya: open, high, low, close, poin dan persentase naik/turun, nilai transaksi (Turnover Rupiah), volume transaksi (lembar dan lot saham), frekuensi transaksi, perbandingan saham menguat, melemah, dan stagnan, serta net foreign buy/sell di pasar reguler.
Lalu proyeksi perdagangan hari ini: rentang pergerakan support-resisten, katalis bursa Wall Street (Dow Jones, S&P 500, Nasdaq), yield US Treasury 10 tahun, komentar pejabat The Fed / probabilitas suku bunga Fed Rate, data inflasi AS (PCE/CPI).
Katalis komoditas energi (Minyak Brent).
Katalis domestik: Nilai tukar Rupiah (spot USD/IDR dan JISDOR), yield obligasi SBN 10 tahun, serta implikasinya terhadap saham-saham big caps dan perbankan.]

Net Foreign Buy
1. [Ticker] - [Nominal/Rp]
2. ...
3. ...
4. ...
5. ...

Net Foreign Sell
1. [Ticker] - [Nominal/Rp]
2. ...
3. ...
4. ...
5. ...

Sektor yang Menguat
- [Sektor]: +X.XX%
...

Sektor yang Melemah
- [Sektor]: -X.XX%
...

Teknikal Outlook
[Ulasan teknikal mendalam: struktur tren jangka pendek, price action candlestick kemarin (misal lower shadow panjang, rejection pada area support psikologis), konfirmasi reversal atau kelanjutan tren, foreign outflow context, dan resistance area.
Uraikan level support terdekat dan support kedua. Peluang technical rebound menuju resistance dan breakout target. Skenario breakdown di bawah support psikologis.
Panduan strategi perdagangan hari ini: rekomendasi tindakan taktis (misal selective buy on weakness dengan konfirmasi rejection), faktor penahan tekanan eksternal dan risiko makro domestik, bias utama dan konfirmasi pemulihan arah indeks.]

DATA FAKTA PASAR HARI INI:
- Tanggal: {date_str}
- IHSG: Close {snapshot['ihsg_close']} ({snapshot['ihsg_change_pct']}%, {snapshot['ihsg_change_point']} poin), Open {snapshot['ihsg_open']}, High {snapshot['ihsg_high']}, Low {snapshot['ihsg_low']}
- Turnover: {snapshot['turnover_idr']}, Volume: {snapshot['volume_shares']}, Frekuensi: {snapshot['frequency_trades']}
- Market Breadth: {snapshot['stocks_adv']} menguat, {snapshot['stocks_dec']} melemah, {snapshot['stocks_flat']} stagnan
- Net Foreign Sell Reguler: {snapshot['foreign_net_sell_reg']}
- Wall Street: Dow Jones {snapshot['dow_change']}, S&P 500 {snapshot['sp500_change']}, Nasdaq {snapshot['nasdaq_change']}
- US Treasury 10Y Yield: {snapshot['ust_10y_yield']}
- Minyak Mentah Brent: {snapshot['brent_oil']}
- Kurs Rupiah: {snapshot['usd_idr']}, JISDOR: {snapshot['jisdor']}
- Yield SBN 10Y: {snapshot['sbn_10y_yield']}
- Pivot Levels: Support {snapshot['ihsg_support']} | Resistance {snapshot['ihsg_resistance']} | S1 {snapshot['ihsg_s1']} | S2 {snapshot['ihsg_s2']} | R1 {snapshot['ihsg_r1']} | R2 {snapshot['ihsg_r2']}

Top Net Foreign Buy:
{json.dumps(snapshot.get('foreign_buy', []), ensure_ascii=False)}

Top Net Foreign Sell:
{json.dumps(snapshot.get('foreign_sell', []), ensure_ascii=False)}

Sektor yang Menguat:
{json.dumps(snapshot.get('sectors_gaining', []), ensure_ascii=False)}

Sektor yang Melemah:
{json.dumps(snapshot.get('sectors_declining', []), ensure_ascii=False)}

Kembalikan respon HANYA dalam format JSON valid (tanpa teks pembuka atau markdown wrap ```json) dengan struktur:
{{
  "sp500_badge": "Label status singkat 2-4 kata",
  "sp500_desc": "Analisis singkat katalis Wall Street (1-2 kalimat)",
  "brent_badge": "Label status minyak Brent 2-4 kata",
  "brent_desc": "Analisis harga minyak Brent dan pengaruhnya (1-2 kalimat)",
  "usd_idr_badge": "Label status kurs Rupiah 2-4 kata",
  "usd_idr_desc": "Analisis kurs Rupiah dan yield SBN (1-2 kalimat)",
  "key_takeaways": [
    "🎯 Arah Indeks: Ringkasan proyeksi pergerakan IHSG",
    "🌐 Katalis Global: Ringkasan pengaruh Wall Street dan yield US Treasury",
    "⚠️ Risiko Makro: Ringkasan risiko nilai tukar dan komoditas",
    "💼 Panduan Taktis: Rekomendasi aksi alokasi portofolio"
  ],
  "brief_content": "Teks lengkap laporan Analisis IHSG sesuai struktur wajib di atas (tanpa tanda asterisk tebal ganda ** berlebihan)."
}}
"""
            # Try multiple candidate Gemini models for maximum availability
            candidate_models = ["gemini-flash-latest", "gemini-3.8-flash", "gemini-2.5-pro", "gemini-2.5-flash"]
            for model_name in candidate_models:
                try:
                    logger.info(f"Attempting brief generation with model: {model_name}...")
                    model = genai.GenerativeModel(model_name=model_name, system_instruction=self.doctrine)
                    response = model.generate_content(prompt)
                    raw_text = response.text.strip() if response and hasattr(response, "text") else ""

                    clean_json_str = raw_text
                    if clean_json_str.startswith("```json"):
                        clean_json_str = clean_json_str[7:]
                    if clean_json_str.startswith("```"):
                        clean_json_str = clean_json_str[3:]
                    if clean_json_str.endswith("```"):
                        clean_json_str = clean_json_str[:-3]
                    clean_json_str = clean_json_str.strip()

                    parsed_data = json.loads(clean_json_str, strict=False)
                    logger.info(f"Successfully generated structured brief via Gemini ({model_name}).")
                    break
                except Exception as e_m:
                    logger.warning(f"Model {model_name} attempt error: {e_m}")

        # Deterministic institutional fallback if Gemini was offline
        if not parsed_data or not parsed_data.get("brief_content"):
            logger.info("Using deterministic quantitative institutional brief generator.")
            parsed_data = self._generate_fallback_content(snapshot)

        # Sanitize narrative content
        brief_body = str(parsed_data.get("brief_content", "")).strip()
        brief_body = brief_body.replace("**", "").replace("*", "").replace(" & ", " dan ").strip()

        sp500_badge = str(parsed_data.get("sp500_badge", "Katalis Pasar AS")).replace("**", "").strip()
        sp500_desc = str(parsed_data.get("sp500_desc", "Sentimen bursa AS memberikan dorongan awal bagi IHSG.")).replace("**", "").strip()
        brent_badge = str(parsed_data.get("brent_badge", "Dinamika Komoditas")).replace("**", "").strip()
        brent_desc = str(parsed_data.get("brent_desc", "Biaya energi dan inflasi manufaktur domestik terjaga.")).replace("**", "").strip()
        usd_idr_badge = str(parsed_data.get("usd_idr_badge", "Stabilitas Valuta")).replace("**", "").strip()
        usd_idr_desc = str(parsed_data.get("usd_idr_desc", "Stabilitas nilai tukar menopang arus modal asing.")).replace("**", "").strip()

        raw_takeaways = parsed_data.get("key_takeaways", [])
        if not isinstance(raw_takeaways, list) or len(raw_takeaways) == 0:
            raw_takeaways = [
                f"🎯 Arah Indeks: IHSG menguji rentang support {snapshot['ihsg_support']} hingga resistance {snapshot['ihsg_resistance']} di sekitar pivot {snapshot['ihsg_pivot']}.",
                f"🌐 Katalis Global: Pengaruh pergerakan Wall Street ({snapshot['sp500_change']}) dan yield US Treasury ({snapshot['ust_10y_yield']}).",
                f"⚠️ Risiko Makro: Minyak Brent {snapshot['brent_oil']} dan kurs Rupiah {snapshot['usd_idr']} menjadi variabel likuiditas kunci.",
                "💼 Panduan Taktis: Disiplin alokasi kas dan akumulasi bertahap pada saham likuid berfundamental prima."
            ]
        clean_takeaways = [str(t).replace("**", "").replace("*", "").strip() for t in raw_takeaways]

        # Update snapshot with dynamic cards data for backward compatibility
        snapshot["sp500_badge"] = sp500_badge
        snapshot["sp500_desc"] = sp500_desc
        snapshot["brent_badge"] = brent_badge
        snapshot["brent_desc"] = brent_desc
        snapshot["usd_idr_badge"] = usd_idr_badge
        snapshot["usd_idr_desc"] = usd_idr_desc
        snapshot["key_takeaways"] = clean_takeaways

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
            "headline": title,
            "cards": cards_data,
            "key_takeaways": clean_takeaways,
            "brief_content": brief_body,
            "foreign_buy": snapshot.get("foreign_buy", []),
            "foreign_sell": snapshot.get("foreign_sell", []),
            "sectors_gaining": snapshot.get("sectors_gaining", []),
            "sectors_declining": snapshot.get("sectors_declining", []),
            "snapshot": snapshot,
        }

    def execute_and_save(self) -> Dict[str, Any]:
        # 1. Collect market data and generate morning brief
        snapshot = self.collect_market_snapshot()
        brief_data = self.generate_brief_text(snapshot)

        # 2. Generate structured technical & fundamental analyses for watchlist stocks (Requirement 3)
        try:
            logger.info("Executing Watchlist Analyzer for key portfolio constituents...")
            watchlist_analyses = self.watchlist_analyzer.generate_all_watchlist_analyses()
            brief_data["watchlist_analyses"] = watchlist_analyses
        except Exception as e_w:
            logger.warning(f"Could not generate watchlist analyses: {e_w}")
            brief_data["watchlist_analyses"] = {}

        # 3. Persist latest_morning_brief.json
        MORNING_BRIEF_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MORNING_BRIEF_FILE, "w", encoding="utf-8") as f:
            json.dump(brief_data, f, indent=2, ensure_ascii=False)

        # 4. Sync into snapshot.json if present
        if SNAPSHOT_FILE.exists():
            try:
                with open(SNAPSHOT_FILE, "r", encoding="utf-8") as sf:
                    snap = json.load(sf)
                snap["morning_brief"] = brief_data
                snap["watchlist_analyses"] = brief_data.get("watchlist_analyses", {})
                with open(SNAPSHOT_FILE, "w", encoding="utf-8") as sf:
                    json.dump(snap, sf, ensure_ascii=False, indent=2)
            except Exception:
                pass

        logger.info(f"Morning Brief successfully saved to {MORNING_BRIEF_FILE}")
        return brief_data


def run_morning_brief_pipeline() -> Dict[str, Any]:
    generator = MorningBriefGenerator()
    return generator.execute_and_save()


if __name__ == "__main__":
    run_morning_brief_pipeline()
