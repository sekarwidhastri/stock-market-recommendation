import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
import numpy as np  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]
import yfinance as yf  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    BENCHMARK_DATA_FILE,
    BENCHMARK_TICKER,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TICKERS,
    FINANCIAL_STATEMENTS_DIR,
    FUNDAMENTAL_DATA_FILE,
    GLOBAL_MACRO_FILE,
    LOG_FORMAT,
    MACRO_TICKERS,
    RAW_DATA_FILE,
)

# Configure module logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("DataIngestion")


class MarketDataIngestor:
    """
    Automated Quantitative Data Ingestion Engine.
    Fetches raw daily OHLCV market data, market benchmark (^JKSE), global macro indicators,
    and individual multi-year financial statements per emiten into dedicated folders.
    """

    def __init__(
        self,
        tickers: List[str] = DEFAULT_TICKERS,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        benchmark_ticker: str = BENCHMARK_TICKER,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime("%Y-%m-%d")
        self.benchmark_ticker = benchmark_ticker

    def fetch_data(self) -> pd.DataFrame:
        """
        Downloads historical price & volume data for specified tickers.
        """
        logger.info(
            f"Fetching market data for {len(self.tickers)} emiten | Range: {self.start_date} to {self.end_date}"
        )
        try:
            cleaned_records = []

            for ticker in self.tickers:
                try:
                    df_ticker = yf.download(
                        tickers=ticker,
                        start=self.start_date,
                        end=self.end_date,
                        auto_adjust=False,
                        progress=False,
                    )

                    if not df_ticker.empty:
                        df_ticker = df_ticker.reset_index()
                        if isinstance(df_ticker.columns, pd.MultiIndex):
                            df_ticker.columns = [
                                col[0] if isinstance(col, tuple) else col for col in df_ticker.columns
                            ]

                        df_ticker["Ticker"] = ticker
                        cleaned_records.append(df_ticker)
                except Exception as ex:
                    logger.error(f"Failed fetching {ticker}: {str(ex)}")

            if not cleaned_records:
                logger.error("No valid records extracted from download output.")
                return pd.DataFrame()

            combined_df = pd.concat(cleaned_records, ignore_index=True)

            expected_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if col not in combined_df.columns:
                    combined_df[col] = pd.NA

            combined_df = combined_df[expected_cols]
            combined_df["Date"] = pd.to_datetime(combined_df["Date"])
            combined_df.sort_values(by=["Ticker", "Date"], inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            logger.info(f"Market data ingestion complete. Total rows: {len(combined_df)}")
            return combined_df

        except Exception as e:
            logger.error(f"Failed to fetch market data: {str(e)}", exc_info=True)
            raise e

    def fetch_benchmark_data(self) -> pd.DataFrame:
        """
        Downloads historical market index benchmark (^JKSE / IHSG).
        """
        logger.info(f"Fetching market benchmark data for {self.benchmark_ticker}...")
        try:
            df_bench = yf.download(
                tickers=self.benchmark_ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=False,
            )
            if df_bench.empty:
                return pd.DataFrame()

            df_bench = df_bench.reset_index()
            if isinstance(df_bench.columns, pd.MultiIndex):
                df_bench.columns = [col[0] if isinstance(col, tuple) else col for col in df_bench.columns]

            expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for col in expected_cols:
                if col not in df_bench.columns:
                    df_bench[col] = pd.NA

            df_bench = df_bench[expected_cols].copy()
            df_bench["Date"] = pd.to_datetime(df_bench["Date"])
            df_bench["Benchmark_Return_1D"] = df_bench["Adj Close"].pct_change(1)
            df_bench.sort_values(by="Date", inplace=True)
            df_bench.reset_index(drop=True, inplace=True)
            return df_bench
        except Exception as e:
            logger.error(f"Failed fetching benchmark data: {str(e)}")
            return pd.DataFrame()

    def fetch_global_macro_data(self) -> pd.DataFrame:
        """
        Downloads global market catalysts: S&P500, Nasdaq, Dow Jones, US 10Y Yield, Oil, Gold, USD/IDR.
        """
        logger.info("Fetching global macroeconomic & cross-asset catalysts...")
        macro_records = []
        for name, symbol in MACRO_TICKERS.items():
            try:
                df = yf.download(
                    tickers=symbol,
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=False,
                    progress=False,
                )
                if not df.empty:
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    df["Asset_Name"] = name
                    df["Symbol"] = symbol
                    macro_records.append(df[["Date", "Asset_Name", "Symbol", "Close", "Volume"]])
            except Exception as e:
                logger.warning(f"Could not fetch macro asset {name} ({symbol}): {str(e)}")

        if macro_records:
            combined_macro = pd.concat(macro_records, ignore_index=True)
            combined_macro["Date"] = pd.to_datetime(combined_macro["Date"])
            logger.info(f"Global macro dataset ingested: {len(combined_macro)} rows.")
            return combined_macro
        return pd.DataFrame()

    def fetch_and_save_financial_statements(self) -> pd.DataFrame:
        """
        Fetches fundamental financial ratios and balance sheet statements per emiten.
        Saves individual multi-year financial statements into data/raw/financial_statements/{TICKER}_financials.csv.
        """
        logger.info(f"Extracting financial statements and ratios for {len(self.tickers)} emiten...")
        fundamental_summary = []

        FINANCIAL_STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)

        for ticker in self.tickers:
            clean_name = ticker.replace(".JK", "")
            try:
                t = yf.Ticker(ticker)
                info = t.info or {}

                # 1. Capture multi-year Income Statement & Balance Sheet
                annual_financials = t.financials
                if annual_financials is not None and not annual_financials.empty:
                    # Save individual emiten statement to dedicated clean folder
                    statement_file = FINANCIAL_STATEMENTS_DIR / f"{clean_name}_financials.csv"
                    annual_financials.to_csv(statement_file)

                # 2. Extract Key Ratios for Cross-Sectional Analysis
                div_yield = info.get("dividendYield")
                if div_yield is not None:
                    try:
                        div_yield = float(div_yield)
                        if 0 < div_yield <= 1.0:
                            div_yield = div_yield * 100.0
                    except (ValueError, TypeError):
                        div_yield = None

                pb_ratio = info.get("priceToBook")
                if pb_ratio is not None:
                    try:
                        pb_ratio = float(pb_ratio)
                        if pb_ratio > 500.0:
                            pb_ratio = pb_ratio / 10000.0
                    except (ValueError, TypeError):
                        pb_ratio = None

                pe_ratio = info.get("trailingPE")
                roe = info.get("returnOnEquity")
                roa = info.get("returnOnAssets")
                profit_margin = info.get("profitMargins")
                operating_margin = info.get("operatingMargins")
                earnings_growth = info.get("earningsGrowth")

                debt_to_equity = info.get("debtToEquity")
                if debt_to_equity is not None:
                    try:
                        debt_to_equity = float(debt_to_equity) / 100.0
                    except (ValueError, TypeError):
                        debt_to_equity = None

                current_ratio = info.get("currentRatio")
                quick_ratio = info.get("quickRatio")
                market_cap = info.get("marketCap")
                shares_outstanding = info.get("sharesOutstanding")
                float_shares = info.get("floatShares")
                free_float_ratio = None
                if float_shares and shares_outstanding and shares_outstanding > 0:
                    free_float_ratio = round(float_shares / shares_outstanding, 4)

                record = {
                    "Ticker": ticker,
                    "PE_Ratio": pe_ratio,
                    "PB_Ratio": pb_ratio,
                    "ROE": roe,
                    "ROA": roa,
                    "Profit_Margin": profit_margin,
                    "Operating_Margin": operating_margin,
                    "Dividend_Yield": div_yield,
                    "Market_Cap": market_cap,
                    "Earnings_Growth": earnings_growth,
                    "Debt_to_Equity": debt_to_equity,
                    "Current_Ratio": current_ratio,
                    "Quick_Ratio": quick_ratio,
                    "Free_Float_Ratio": free_float_ratio,
                    "Free_Cashflow": info.get("freeCashflow"),
                    "Operating_Cashflow": info.get("operatingCashflow"),
                }
                fundamental_summary.append(record)
            except Exception as e:
                logger.warning(f"Could not extract fundamentals for {ticker}: {str(e)}")
                fundamental_summary.append({
                    "Ticker": ticker,
                    "PE_Ratio": None,
                    "PB_Ratio": None,
                    "ROE": None,
                    "ROA": None,
                    "Profit_Margin": None,
                    "Operating_Margin": None,
                    "Dividend_Yield": None,
                    "Market_Cap": None,
                    "Earnings_Growth": None,
                    "Debt_to_Equity": None,
                    "Current_Ratio": None,
                    "Quick_Ratio": None,
                    "Free_Float_Ratio": None,
                    "Free_Cashflow": None,
                    "Operating_Cashflow": None,
                })

        summary_df = pd.DataFrame(fundamental_summary)
        return summary_df

    def save_raw_data(
        self,
        df: pd.DataFrame,
        fundamental_df: Optional[pd.DataFrame] = None,
        benchmark_df: Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Persists all raw datasets to disk.
        """
        if not df.empty:
            RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_DATA_FILE, index=False)
            logger.info(f"Raw market data safely persisted to {RAW_DATA_FILE}")

        if fundamental_df is not None and not fundamental_df.empty:
            FUNDAMENTAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            fundamental_df.to_csv(FUNDAMENTAL_DATA_FILE, index=False)
            logger.info(f"Fundamental summary persisted to {FUNDAMENTAL_DATA_FILE}")

        if benchmark_df is not None and not benchmark_df.empty:
            BENCHMARK_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            benchmark_df.to_csv(BENCHMARK_DATA_FILE, index=False)
            logger.info(f"Benchmark market data persisted to {BENCHMARK_DATA_FILE}")

        if macro_df is not None and not macro_df.empty:
            GLOBAL_MACRO_FILE.parent.mkdir(parents=True, exist_ok=True)
            macro_df.to_csv(GLOBAL_MACRO_FILE, index=False)
            logger.info(f"Global macro dataset persisted to {GLOBAL_MACRO_FILE}")


def run_ingestion_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main runner for 01_data_ingestion step.
    Orchestrates dynamic universe health validation and market data ingestion.
    """
    # 1. Evaluate Dynamic Universe Rebalancing & Active Tickers
    active_tickers = DEFAULT_TICKERS
    try:
        from src.universe_manager import get_universe_manager
        univ_mgr = get_universe_manager()
        univ_mgr.check_and_run_rebalance()
        active_tickers = univ_mgr.get_active_tickers()
    except Exception as e:
        logger.warning(f"Universe manager check bypassed: {e}")

    # 2. Ingest Active Market Data
    ingestor = MarketDataIngestor(tickers=active_tickers)
    df_raw = ingestor.fetch_data()
    df_bench = ingestor.fetch_benchmark_data()
    df_macro = ingestor.fetch_global_macro_data()
    df_fundamentals = ingestor.fetch_and_save_financial_statements()

    # 3. Post-Ingestion Automated Health-Check & Standby Reserve Substitution
    try:
        from src.universe_manager import get_universe_manager
        univ_mgr = get_universe_manager()
        health_res = univ_mgr.run_health_check(df_raw)
        
        if health_res.get("replacements_count", 0) > 0:
            reps = health_res.get("replacements", [])
            logger.info(f"Health check swapped {len(reps)} tickers. Backfilling data for promoted standby reserves...")
            promoted_tickers = [r["promoted_ticker"] for r in reps]
            degraded_tickers = [r["degraded_ticker"] for r in reps]
            
            backfill_ingestor = MarketDataIngestor(tickers=promoted_tickers)
            df_promoted = backfill_ingestor.fetch_data()
            if not df_promoted.empty and not df_raw.empty:
                df_raw = pd.concat([df_raw[~df_raw["Ticker"].isin(degraded_tickers)], df_promoted], ignore_index=True)
                df_raw.sort_values(by=["Ticker", "Date"], inplace=True)
                df_raw.reset_index(drop=True, inplace=True)
    except Exception as e:
        logger.warning(f"Post-ingestion health-check exception: {e}")

    # 4. Safely persist all validated datasets
    ingestor.save_raw_data(df_raw, df_fundamentals, df_bench, df_macro)
    return df_raw, df_fundamentals, df_bench, df_macro


if __name__ == "__main__":
    run_ingestion_pipeline()
