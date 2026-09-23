import os
from pathlib import Path
from typing import Dict, List

# Base Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINANCIAL_STATEMENTS_DIR = RAW_DATA_DIR / "financial_statements"

# Ensure directories exist safely (avoid failure on read-only serverless environments)
try:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINANCIAL_STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# 1. 11 Sektor Resmi Bursa Efek Indonesia (IDX-IC Universe)
SECTOR_MAP: Dict[str, List[str]] = {
    "Financials": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BBTN.JK"],
    "Energy": ["ADRO.JK", "PTBA.JK", "ITMG.JK", "PGAS.JK", "MEDC.JK", "AKRA.JK", "HRUM.JK", "INDY.JK"],
    "Basic Materials": ["ANTM.JK", "MDKA.JK", "INCO.JK", "TPIA.JK", "BRPT.JK", "INKP.JK", "AMMN.JK", "MBMA.JK"],
    "Consumer Non-Cyclicals": ["ICBP.JK", "INDF.JK", "UNVR.JK", "AMRT.JK", "MYOR.JK", "CPIN.JK", "SIDO.JK"],
    "Consumer Cyclicals": ["ASII.JK", "ACES.JK", "MAPI.JK", "ERAA.JK"],
    "Healthcare": ["KLBF.JK", "MIKA.JK", "HEAL.JK", "SILO.JK"],
    "Technology": ["GOTO.JK", "EMTK.JK", "BUKA.JK"],
    "Infrastructures": ["TLKM.JK", "ISAT.JK", "EXCL.JK", "TOWR.JK", "TBIG.JK", "PGEO.JK", "BREN.JK"],
    "Properties & Real Estate": ["BSDE.JK", "CTRA.JK", "PWON.JK", "SMRA.JK"],
    "Industrials": ["UNTR.JK", "HEXA.JK", "AUTO.JK", "SMSM.JK"],
    "Transportation & Logistics": ["BIRD.JK", "SMDR.JK", "ASSA.JK"],
}

# 2. Swing Trader Universe (LQ45 Liquid Emiten)
SWING_TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "AMRT.JK",
    "ADRO.JK", "PTBA.JK", "PGAS.JK", "GOTO.JK", "BRPT.JK",
    "INKP.JK", "CPIN.JK", "KLBF.JK", "MEDC.JK", "MDKA.JK",
    "ANTM.JK", "AKRA.JK", "ITMG.JK", "TPIA.JK", "BREN.JK",
    "AMMN.JK", "EMTK.JK", "EXCL.JK", "HRUM.JK", "INDY.JK",
    "ISAT.JK", "MBMA.JK", "PGEO.JK", "SMGR.JK", "TOWR.JK",
    "UNTR.JK", "BRIS.JK", "ACES.JK", "MIKA.JK", "HEAL.JK",
]

# 3. Dividend & Value Investor Universe
DIVIDEND_TICKERS = [
    "ITMG.JK", "PTBA.JK", "ADRO.JK", "UNTR.JK", "HEXA.JK",
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK",
    "ASII.JK", "BJBR.JK", "BJTM.JK", "MPMX.JK", "POWR.JK",
    "SMSM.JK", "AUTO.JK", "NRCA.JK", "ACES.JK", "SIDO.JK",
    "UNVR.JK", "ICBP.JK", "INDF.JK", "AMRT.JK",
]

# 4. 10 Favorite Emiten Portfolio
FAVORITE_TICKERS = [
    "AADI.JK", "ADMR.JK", "ADRO.JK", "BBCA.JK", "UNTR.JK",
    "ISAT.JK", "BBRI.JK", "ASII.JK", "ANTM.JK", "HEXA.JK",
]

# Flatten all sectors into fallback unique universe
ALL_SECTOR_TICKERS = [t for tickers in SECTOR_MAP.values() for t in tickers]
DEFAULT_TICKERS = sorted(list(set(SWING_TICKERS + DIVIDEND_TICKERS + FAVORITE_TICKERS + ALL_SECTOR_TICKERS)))

# Connect Dynamic Universe Manager (preserving capacity N=66)
try:
    from src.universe_manager import get_universe_manager
    _univ_mgr = get_universe_manager()
    DEFAULT_TICKERS = _univ_mgr.get_active_tickers()
    SECTOR_MAP = _univ_mgr.get_sector_map()
except Exception:
    pass

# 5. Global Macro & Market Catalysts
MACRO_TICKERS = {
    "IHSG": "^JKSE",
    "SP500": "^GSPC",
    "Nasdaq": "^IXIC",
    "DowJones": "^DJI",
    "US_Treasury_10Y": "^TNX",
    "Oil_Brent": "BZ=F",
    "Oil_WTI": "CL=F",
    "Gold": "GC=F",
    "USD_IDR": "USDIDR=X",
    "Dollar_Index": "DX-Y.NYB",
}

BENCHMARK_TICKER = "^JKSE"
RISK_FREE_RATE = 0.06  # Bank Indonesia BI-Rate (6.00% p.a.)

# Date Range Configuration
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = None

# Model Hyperparameters
LSTM_LOOKBACK = 30
ARIMA_FORECAST_STEPS = 5

# Data File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "raw_market_data.csv"
BENCHMARK_DATA_FILE = RAW_DATA_DIR / "benchmark_market_data.csv"
GLOBAL_MACRO_FILE = RAW_DATA_DIR / "global_macro_data.csv"
FUNDAMENTAL_DATA_FILE = RAW_DATA_DIR / "fundamental_financial_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_market_features.csv"
ADVANCED_METRICS_FILE = PROCESSED_DATA_DIR / "advanced_quant_metrics.csv"

# Recommendations & Morning Brief Outputs
SWING_RECOMMENDATION_FILE = PROCESSED_DATA_DIR / "latest_alpha_recommendations_swing.csv"
DIVIDEND_RECOMMENDATION_FILE = PROCESSED_DATA_DIR / "latest_alpha_recommendations_dividend.csv"
FAVORITES_RECOMMENDATION_FILE = PROCESSED_DATA_DIR / "latest_alpha_recommendations_favorites.csv"
MORNING_BRIEF_FILE = PROCESSED_DATA_DIR / "latest_morning_brief.json"
PORTFOLIO_ALLOCATION_FILE = PROCESSED_DATA_DIR / "latest_portfolio_allocation.csv"

# Logging Format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
