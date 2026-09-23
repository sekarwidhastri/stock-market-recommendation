import importlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv  # type: ignore # pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query  # type: ignore # pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore # pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse  # type: ignore # pyrefly: ignore [missing-import]
import google.generativeai as genai  # type: ignore # pyrefly: ignore [missing-import]
import numpy as np  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]
from pydantic import BaseModel  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    DEFAULT_TICKERS,
    DIVIDEND_RECOMMENDATION_FILE,
    FAVORITE_TICKERS,
    FAVORITES_RECOMMENDATION_FILE,
    LOG_FORMAT,
    MORNING_BRIEF_FILE,
    PORTFOLIO_ALLOCATION_FILE,
    SECTOR_MAP,
    SWING_RECOMMENDATION_FILE,
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Lazy pipeline imports for serverless compatibility
def run_ingestion_pipeline():
    mod = importlib.import_module("src.01_data_ingestion")
    return mod.run_ingestion_pipeline()

def run_feature_engineering_pipeline():
    mod = importlib.import_module("src.02_feature_eng")
    return mod.run_feature_engineering_pipeline()

def run_model_inference_pipeline():
    mod = importlib.import_module("src.03_model_inference")
    return mod.run_model_inference_pipeline()

def run_morning_brief_pipeline():
    mod = importlib.import_module("src.morning_brief")
    return mod.run_morning_brief_pipeline()

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("StockMarketRecommendationAPI")

RECOMMENDATION_FILE = SWING_RECOMMENDATION_FILE

app = FastAPI(
    title="Stock Market Recommendation API",
    description="Multi-Engine Algorithmic Recommendation Engine (GBDT + LSTM + ARIMA + GARCH), Institutional Morning Brief & Portfolio Allocator",
    version="4.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class RecommendationItem(BaseModel):
    Date: str
    Ticker: str
    Sector: Optional[str] = "General"
    Close: float
    Recommendation: str
    Entry_Price: Optional[float] = None
    Target_Price: Optional[float] = None
    Stop_Loss: Optional[float] = None
    Risk_Reward_Ratio: Optional[float] = None
    Bullish_Probability: float
    GBDT_Prob: Optional[float] = None
    ARIMA_Prob: Optional[float] = None
    LSTM_Prob: Optional[float] = None
    Beta_IHSG: Optional[float] = None
    Sharpe_Ratio: Optional[float] = None
    Max_Drawdown_1Y: Optional[float] = None
    Annualized_Return_1Y: Optional[float] = None
    GARCH_Vol: Optional[float] = None
    VaR_95_1D: Optional[float] = None
    Debt_to_Equity: Optional[float] = None
    Current_Ratio: Optional[float] = None
    RSI_14: Optional[float] = None
    MACD_Hist: Optional[float] = None
    MFI_14: Optional[float] = None
    CMF_20: Optional[float] = None
    PE_Ratio: Optional[float] = None
    PB_Ratio: Optional[float] = None
    ROE: Optional[float] = None
    Dividend_Yield: Optional[float] = None


class AllocationItem(BaseModel):
    Ticker: str
    Sector: str
    Recommendation: str
    Allocation_Pct: float
    Nominal_IDR: float
    Shares_Lot: int
    Close_Price: float
    Target_Price: float
    Stop_Loss: float
    Bullish_Probability: float


class PipelineResponse(BaseModel):
    status: str
    message: str


def execute_full_pipeline():
    logger.info("Executing automated end-to-end quant pipeline...")
    try:
        run_ingestion_pipeline()
        run_feature_engineering_pipeline()
        run_model_inference_pipeline()
        run_morning_brief_pipeline()
        logger.info("Full quant pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error during daily pipeline execution: {str(e)}", exc_info=True)


def _clean_record(record: Any) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    if not isinstance(record, dict):
        return cleaned
    for k, v in record.items():
        key = str(k)
        if pd.isna(v) or v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
            cleaned[key] = None
        elif isinstance(v, (np.floating, float)):
            cleaned[key] = round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            cleaned[key] = int(v)
        else:
            cleaned[key] = v
    return cleaned


@app.get("/status", tags=["Status"])
def get_pipeline_status() -> Dict[str, Any]:
    last_update = "Not Executed"
    if RECOMMENDATION_FILE.exists():
        mtime = os.path.getmtime(RECOMMENDATION_FILE)
        last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "online",
        "engine": "Hybrid Quant Suite (GBDT + ARIMA + PyTorch LSTM + GARCH)",
        "last_pipeline_update": last_update,
        "sectors_covered": list(SECTOR_MAP.keys()),
        "recommendation_file": str(RECOMMENDATION_FILE),
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def web_dashboard():
    index_file = Path(__file__).resolve().parent.parent / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Stock Market Recommendation Platform is running.</h1>")


@app.get("/login", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/login.html", response_class=HTMLResponse, tags=["Dashboard"])
def web_login():
    login_file = Path(__file__).resolve().parent.parent / "login.html"
    if login_file.exists():
        return HTMLResponse(content=login_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Halaman Login Tidak Ditemukan</h1>")


class AuthLoginRequest(BaseModel):
    username: str
    password: Optional[str] = None
    remember_me: Optional[bool] = True


@app.post("/api/v1/auth/login", tags=["Auth"])
def auth_login(req: AuthLoginRequest) -> Dict[str, Any]:
    username = req.username.strip().lower()
    if "sekar" in username or "analyst" in username:
        user_data = {
            "name": "Sekar Widhastri",
            "role": "Research Analyst",
            "email": "sekar.widhastri@stockmarket.id",
            "initials": "SW",
            "access_level": "Senior Analyst (Full Access)",
        }
    elif "portfolio" in username or "manager" in username:
        user_data = {
            "name": "Portfolio Manager",
            "role": "Asset Management",
            "email": "portfolio.manager@stockmarket.id",
            "initials": "PM",
            "access_level": "Portfolio Management Access",
        }
    else:
        clean_name = req.username.strip().split("@")[0].title() or "Tamu Pengunjung"
        user_data = {
            "name": clean_name,
            "role": "Market Explorer",
            "email": req.username.strip() if "@" in req.username else f"{req.username.strip()}@stockmarket.id",
            "initials": clean_name[:2].upper() if len(clean_name) >= 2 else "TM",
            "access_level": "Public Research Access",
        }
    return {
        "status": "success",
        "token": "bearer-jwt-idx-quant-2026",
        "user": user_data,
    }


@app.get("/morning-brief", tags=["Morning Market Brief"])
def get_morning_brief() -> Dict[str, Any]:
    """
    Returns the latest institutional Morning Brief IHSG narrative.
    """
    if MORNING_BRIEF_FILE.exists():
        try:
            with open(MORNING_BRIEF_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Generate fresh if not available
    return run_morning_brief_pipeline()


@app.get("/sectors", tags=["Sectors"])
def get_sectors() -> Dict[str, List[str]]:
    """
    Returns the 11 official IDX sectors and their constituent tickers.
    """
    return SECTOR_MAP


@app.get("/universe", tags=["Universe Management"])
@app.get("/api/universe", tags=["Universe Management"])
def get_universe_telemetry() -> Dict[str, Any]:
    """
    Returns diagnostic telemetry for the 66-stock dynamic universe,
    including sector distribution, health statuses, and standby replacements.
    """
    try:
        from src.universe_manager import get_universe_manager
        mgr = get_universe_manager()
        return mgr.get_universe_diagnostics()
    except Exception as e:
        return {
            "capacity": 66,
            "active_tickers_count": len(DEFAULT_TICKERS),
            "active_tickers": DEFAULT_TICKERS,
            "sector_breakdown": {s: len(t) for s, t in SECTOR_MAP.items()},
            "error": str(e),
        }


@app.get("/portfolio/allocate", response_model=List[AllocationItem], tags=["Portfolio Optimizer"])
def get_portfolio_allocation(capital: float = Query(50000000.0, description="Total capital in IDR (Rupiah)")) -> List[Dict]:
    """
    Computes optimal portfolio allocation weights and nominal IDR for user's capital.
    """
    if PORTFOLIO_ALLOCATION_FILE.exists():
        try:
            df = pd.read_csv(PORTFOLIO_ALLOCATION_FILE)
            # Re-scale nominal if capital differs from default
            records = []
            for _, row in df.iterrows():
                r = row.to_dict()
                pct = r["Allocation_Pct"] / 100.0
                nominal = round(capital * pct, 0)
                close = r["Close_Price"]
                lot = int((nominal / close) // 100) if close > 1.0 else 0
                r["Nominal_IDR"] = nominal
                r["Shares_Lot"] = lot
                records.append(_clean_record(r))
            return records
        except Exception as e:
            logger.error(f"Error reading portfolio allocation: {str(e)}")

    raise HTTPException(status_code=404, detail="Portfolio allocation data not generated yet. Run pipeline first.")


@app.get("/recommendations", response_model=List[RecommendationItem], tags=["Recommendations"])
def get_latest_recommendations(
    mode: str = Query("all", description="Strategy mode: 'all', 'swing', 'dividend', or 'favorites'"),
    universe: Optional[str] = Query(None, description="Alias for mode parameter"),
    sector: Optional[str] = Query(None, description="Optional sector filter (e.g. 'Financials', 'Energy')"),
) -> List[Dict]:
    active_mode = (universe or mode or "all").lower()

    try:
        if active_mode == "dividend":
            df = pd.read_csv(DIVIDEND_RECOMMENDATION_FILE)
        elif active_mode == "favorites":
            df = pd.read_csv(FAVORITES_RECOMMENDATION_FILE)
        elif active_mode == "swing":
            df = pd.read_csv(SWING_RECOMMENDATION_FILE)
        else:
            # Mode "all": gabungkan semua universe (Favorit + Dividen + Swing) tanpa duplikasi
            dfs = []
            for f in [FAVORITES_RECOMMENDATION_FILE, DIVIDEND_RECOMMENDATION_FILE, SWING_RECOMMENDATION_FILE]:
                if f.exists():
                    dfs.append(pd.read_csv(f))
            if dfs:
                df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Ticker"])
            elif SWING_RECOMMENDATION_FILE.exists():
                df = pd.read_csv(SWING_RECOMMENDATION_FILE)
            else:
                raise HTTPException(status_code=404, detail="Recommendations dataset not found. Run pipeline first.")

        if sector and sector.lower() != "all":
            if "Sector" in df.columns:
                df = df[df["Sector"].str.lower() == sector.lower()]

        # Ensure all schema columns exist
        all_cols = [
            "Sector", "Entry_Price", "Target_Price", "Stop_Loss", "Risk_Reward_Ratio",
            "Beta_IHSG", "Sharpe_Ratio", "Max_Drawdown_1Y", "Annualized_Return_1Y",
            "GARCH_Vol", "VaR_95_1D", "Debt_to_Equity", "Current_Ratio",
            "GBDT_Prob", "ARIMA_Prob", "LSTM_Prob", "PE_Ratio", "PB_Ratio", "ROE",
            "Dividend_Yield", "MFI_14", "CMF_20", "Volatility_20D", "RSI_14", "MACD_Hist"
        ]
        for c in all_cols:
            if c not in df.columns:
                df[c] = None

        return [_clean_record(r) for r in df.to_dict(orient="records")]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reading recommendation artifacts.")


@app.get("/models/compare/{ticker}", tags=["Model Architecture"])
def compare_models(ticker: str) -> Dict[str, Any]:
    clean_ticker = ticker.upper()
    if not clean_ticker.endswith(".JK"):
        clean_ticker = f"{clean_ticker}.JK"

    for file_path in [SWING_RECOMMENDATION_FILE, DIVIDEND_RECOMMENDATION_FILE, FAVORITES_RECOMMENDATION_FILE]:
        if file_path.exists():
            df = pd.read_csv(file_path)
            matched = df[df["Ticker"] == clean_ticker]
            if not matched.empty:
                row = _clean_record(matched.iloc[0].to_dict())
                return {
                    "Ticker": clean_ticker,
                    "Sector": row.get("Sector", "General"),
                    "Date": row.get("Date"),
                    "Close": row.get("Close"),
                    "Technical_Action": row.get("Recommendation"),
                    "Entry_Price": row.get("Entry_Price"),
                    "Target_Price": row.get("Target_Price"),
                    "Stop_Loss": row.get("Stop_Loss"),
                    "Composite_Bullish_Probability": row.get("Bullish_Probability"),
                    "Model_Consensus": {
                        "GBDT_Tabular": {"Probability": row.get("GBDT_Prob"), "Weight": "50%"},
                        "PyTorch_LSTM": {"Probability": row.get("LSTM_Prob"), "Weight": "30%"},
                        "ARIMA_TimeSeries": {"Probability": row.get("ARIMA_Prob"), "Weight": "20%"},
                    },
                    "Risk_and_Volatility": {
                        "Beta_IHSG": row.get("Beta_IHSG"),
                        "Sharpe_Ratio": row.get("Sharpe_Ratio"),
                        "GARCH_Annual_Vol": row.get("GARCH_Vol"),
                        "VaR_95_1D": f"{row.get('VaR_95_1D')}%" if row.get("VaR_95_1D") else "N/A",
                        "Max_Drawdown_1Y": row.get("Max_Drawdown_1Y"),
                    },
                }

    raise HTTPException(status_code=404, detail=f"Ticker {clean_ticker} not found in current recommendations.")


@app.get("/api/v1/analyze-favorite", tags=["Favorite Emiten"])
def analyze_favorite_ticker(ticker: str) -> Dict[str, str]:
    clean_ticker = ticker.strip().upper()
    if not clean_ticker.endswith(".JK"):
        clean_ticker = f"{clean_ticker}.JK"

    # Whitelist guard: ensure ticker belongs to known IDX universe
    if clean_ticker not in DEFAULT_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker '{ticker}' is not supported. Must be a valid constituent of the IDX universe.",
        )

    # Fetch factual quantitative data for this ticker from recommendation files to ground the LLM
    quant_context = ""
    for file_path in [FAVORITES_RECOMMENDATION_FILE, SWING_RECOMMENDATION_FILE, DIVIDEND_RECOMMENDATION_FILE]:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                match = df[df["Ticker"] == clean_ticker]
                if not match.empty:
                    row = match.iloc[0]
                    quant_context = (
                        f"Data pasar terkini {clean_ticker}: Close Rp {row.get('Close')}, "
                        f"Rekomendasi Algoritma: {row.get('Recommendation')}, "
                        f"Entry: Rp {row.get('Entry_Price')}, Target: Rp {row.get('Target_Price')}, "
                        f"Stop Loss: Rp {row.get('Stop_Loss')}, RSI-14: {row.get('RSI_14')}, "
                        f"Sharpe 1Y: {row.get('Sharpe_Ratio')}, Beta IHSG: {row.get('Beta_IHSG')}, "
                        f"VaR 95%: {row.get('VaR_95_1D')}%."
                    )
                    break
            except Exception:
                pass

    if not GEMINI_API_KEY:
        if quant_context:
            return {
                "ticker": clean_ticker,
                "analysis": f"Sinyal kuantitatif {clean_ticker}: {quant_context}",
            }
        return {
            "ticker": clean_ticker,
            "analysis": f"Saham {clean_ticker} berada dalam radar pemantauan kuantitatif dengan sinyal teknikal terpantau pada bursa IHSG.",
        }

    prompt = (
        f"Sebagai Senior Quant Analyst pasar modal Indonesia, gunakan data kuantitatif faktual berikut:\n"
        f"{quant_context}\n\n"
        f"Berikan analisis teknikal ringkas (maksimal 2 kalimat) untuk saham {clean_ticker}. "
        f"Sebutkan target harga dan stop loss sesuai data di atas tanpa mengarang angka tambahan. "
        f"Gunakan kata 'dan' bukan simbol ampersand, serta jangan gunakan tanda asterisk tebal ganda."
    )

    try:
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)
        text = response.text.strip() if response and hasattr(response, "text") else "Analisis tidak dapat dihasilkan."
        clean_text = text.replace("**", "").replace("*", "").replace(" & ", " dan ").strip()
        return {"ticker": clean_ticker, "analysis": clean_text}
    except Exception as e:
        logger.error(f"Error calling Gemini: {str(e)}")
        fallback_text = (
            f"Analisis kuantitatif {clean_ticker}: {quant_context}" if quant_context
            else f"Analisis kuantitatif {clean_ticker}: Menunjukkan konsolidasi harga dengan indikator likuiditas terkontrol."
        )
        return {"ticker": clean_ticker, "analysis": fallback_text}


@app.post("/pipeline/run", response_model=PipelineResponse, tags=["Pipeline Automation"])
def trigger_pipeline(
    background_tasks: BackgroundTasks,
    admin_token: Optional[str] = Query(None, description="Admin secret token for triggering pipeline"),
    x_admin_secret: Optional[str] = Header(None, alias="X-Admin-Secret"),
) -> Dict[str, str]:
    # Check if running in serverless environment (e.g. Vercel)
    if os.environ.get("VERCEL"):
        raise HTTPException(
            status_code=403,
            detail="Pipeline execution is disabled on Vercel Serverless runtime. The compute plane runs automatically via GitHub Actions CI/CD daily.",
        )

    expected_secret = os.getenv("ADMIN_PIPELINE_SECRET", "alphatech-secure-pipeline-trigger-2026")
    provided_token = admin_token or x_admin_secret
    if not provided_token or provided_token != expected_secret:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid admin pipeline secret token is required to trigger model training.",
        )

    background_tasks.add_task(execute_full_pipeline)
    return {
        "status": "accepted",
        "message": "Full quantitative pipeline (Ingestion, GARCH, ML Ensemble, Morning Brief) triggered in background.",
    }
