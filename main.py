import importlib
import logging
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
from src.config import LOG_FORMAT  # type: ignore # pyrefly: ignore [missing-import]

# Configure root logger
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("StockMarketRecommendationMain")


def run_all():
    """
    Executes the entire quantitative investment pipeline end-to-end.
    """
    start_time = time.time()
    logger.info("=========================================================")
    logger.info("  STOCK MARKET RECOMMENDATION PIPELINE - MASTER RUNNER   ")
    logger.info("=========================================================")

    # Dynamic imports for quantitative pipeline modules
    universe_mgr_mod = importlib.import_module("src.universe_manager")
    data_ingestion = importlib.import_module("src.01_data_ingestion")
    feature_eng = importlib.import_module("src.02_feature_eng")
    model_inference = importlib.import_module("src.03_model_inference")
    morning_brief = importlib.import_module("src.morning_brief")

    # Step 0: Universe Health & Capacity Invariance
    logger.info("[Step 0/4] Verifying Dynamic Universe Health & Capacity Invariance (N=66)...")
    univ_mgr = universe_mgr_mod.get_universe_manager()
    diag = univ_mgr.get_universe_diagnostics()
    logger.info(
        f"Active Universe: {diag['active_tickers_count']} emiten across 11 sectors | "
        f"Healthy: {diag['health_summary']['healthy_count']} | Replacements: {diag['health_summary']['replaced_count']}"
    )

    # Step 1: Data Ingestion
    logger.info("[Step 1/4] Ingesting Market OHLCV, Macro & Financial Data...")
    data_ingestion.run_ingestion_pipeline()

    # Step 2: Feature Engineering
    logger.info("[Step 2/4] Extracting Technical, GARCH Volatility & Alpha Features...")
    feature_eng.run_feature_engineering_pipeline()

    # Step 3: Model Training & Inference
    logger.info("[Step 3/4] Training Multi-Engine Ensemble & Optimizing Portfolio...")
    res = model_inference.run_model_inference_pipeline()
    metrics: dict = res[0] if isinstance(res, (tuple, list)) and len(res) > 0 and isinstance(res[0], dict) else {}
    recommendations = res[1] if isinstance(res, (tuple, list)) and len(res) > 1 else None

    # Step 4: Morning Market Brief
    logger.info("[Step 4/4] Generating Institutional Daily Morning Brief via Gemini AI...")
    try:
        morning_brief.run_morning_brief_pipeline()
    except Exception as e:
        logger.warning(f"Morning Brief generation encountered an issue (non-fatal): {e}")

    elapsed = round(time.time() - start_time, 2)
    logger.info("=========================================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed} SECONDS!")
    roc_auc = metrics.get('GBDT_ROC_AUC', metrics.get('ROC_AUC', 'N/A'))
    precision = metrics.get('Precision_Top5', 'N/A')
    logger.info(f"GBDT ROC-AUC Score: {roc_auc} | Precision@Top5: {precision}")
    logger.info("=========================================================")
    if recommendations is not None and hasattr(recommendations, "to_string"):
        print("\n--- LATEST ALPHA RECOMMENDATIONS ---")
        print(recommendations.to_string(index=False))


if __name__ == "__main__":
    run_all()
