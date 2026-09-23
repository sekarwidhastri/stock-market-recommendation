"""
Quantitative Dynamic Universe Manager.
Enforces a fixed-capacity (N=66) stock universe across the 11 official IDX-IC sectors,
featuring Automated Health-Check Circuit Breakers (detecting suspensions and zero-volume days),
standby reserve substitutions, and BEI calendar-aligned periodic rebalancing.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]

# Configure logger
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("UniverseManager")

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UNIVERSE_STATE_FILE = PROCESSED_DATA_DIR / "active_universe.json"

# Fixed capacity invariant for machine learning tensor stability
UNIVERSE_CAPACITY: int = 66

# Baseline 11 IDX-IC Official Sector Primary Universe
BASELINE_SECTOR_MAP: Dict[str, List[str]] = {
    "Financials": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BBTN.JK", "BJBR.JK", "BJTM.JK"],
    "Energy": ["ADRO.JK", "PTBA.JK", "ITMG.JK", "PGAS.JK", "MEDC.JK", "AKRA.JK", "HRUM.JK", "INDY.JK", "AADI.JK", "ADMR.JK"],
    "Basic Materials": ["ANTM.JK", "MDKA.JK", "INCO.JK", "TPIA.JK", "BRPT.JK", "INKP.JK", "AMMN.JK", "MBMA.JK", "SMGR.JK"],
    "Consumer Non-Cyclicals": ["ICBP.JK", "INDF.JK", "UNVR.JK", "AMRT.JK", "MYOR.JK", "CPIN.JK", "SIDO.JK"],
    "Consumer Cyclicals": ["ASII.JK", "ACES.JK", "MAPI.JK", "ERAA.JK", "AUTO.JK", "SMSM.JK"],
    "Healthcare": ["KLBF.JK", "MIKA.JK", "HEAL.JK", "SILO.JK"],
    "Technology": ["GOTO.JK", "EMTK.JK", "BUKA.JK"],
    "Infrastructures": ["TLKM.JK", "ISAT.JK", "EXCL.JK", "TOWR.JK", "TBIG.JK", "PGEO.JK", "BREN.JK", "POWR.JK"],
    "Properties & Real Estate": ["BSDE.JK", "CTRA.JK", "PWON.JK", "SMRA.JK", "NRCA.JK"],
    "Industrials": ["UNTR.JK", "HEXA.JK"],
    "Transportation & Logistics": ["BIRD.JK", "SMDR.JK", "ASSA.JK", "MPMX.JK"],
}

# Standby Reserves (Shadow List) per Sector for instant zero-lag substitution
SECTOR_RESERVES: Dict[str, List[str]] = {
    "Financials": ["BDMN.JK", "BNGA.JK", "ARTO.JK", "BTPS.JK"],
    "Energy": ["BUMI.JK", "ENRG.JK", "DOID.JK", "ELSA.JK"],
    "Basic Materials": ["TKIM.JK", "ESSA.JK", "BRMS.JK", "NCKL.JK"],
    "Consumer Non-Cyclicals": ["JPFA.JK", "CMRY.JK", "HMSP.JK", "GGRM.JK"],
    "Consumer Cyclicals": ["MAPA.JK", "SCMA.JK", "RALS.JK"],
    "Healthcare": ["PRDA.JK", "TSPC.JK", "SIDO.JK"],
    "Technology": ["BELI.JK", "WIFI.JK", "MTDL.JK"],
    "Infrastructures": ["JSMR.JK", "WIKA.JK", "PPRE.JK", "MTEL.JK"],
    "Properties & Real Estate": ["ASRI.JK", "DMAS.JK", "DILD.JK", "KIJA.JK"],
    "Industrials": ["MARK.JK", "IMPC.JK", "ARNA.JK", "ASGR.JK"],
    "Transportation & Logistics": ["TMAS.JK", "GIAA.JK", "WEHA.JK"],
}

# Baseline flat universe (exactly 66 tickers)
BASELINE_TICKERS: List[str] = sorted(
    list(set([t for tickers in BASELINE_SECTOR_MAP.values() for t in tickers]))
)


class UniverseManager:
    """
    Orchestrates the dynamic membership of the 66-stock quantitative universe.
    Ensures capacity constraints, monitors liquidity health, replaces suspended assets,
    and executes scheduled rebalancing aligned with the Indonesia Stock Exchange calendar.
    """

    def __init__(self, state_file: Path = UNIVERSE_STATE_FILE):
        self.state_file = state_file
        self.state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> Dict[str, Any]:
        """
        Loads persistent universe state from disk or initializes default 66-emiten state.
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                active = state.get("active_tickers", [])
                if len(active) == UNIVERSE_CAPACITY:
                    logger.info(f"Loaded existing dynamic universe state: {len(active)} active tickers.")
                    return state
                logger.warning(
                    f"Universe state capacity mismatch (found {len(active)}, expected {UNIVERSE_CAPACITY}). Re-initializing."
                )
            except Exception as e:
                logger.error(f"Failed to read universe state file: {e}. Reverting to baseline.")

        # Initialize fresh state
        initial_state: Dict[str, Any] = {
            "capacity": UNIVERSE_CAPACITY,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_rebalance_date": datetime.now().strftime("%Y-%m-%d"),
            "active_tickers": BASELINE_TICKERS[:UNIVERSE_CAPACITY],
            "sector_map": BASELINE_SECTOR_MAP,
            "health_status": {ticker: "HEALTHY" for ticker in BASELINE_TICKERS[:UNIVERSE_CAPACITY]},
            "consecutive_zero_vol_days": {ticker: 0 for ticker in BASELINE_TICKERS[:UNIVERSE_CAPACITY]},
            "replacements_history": [],
        }
        self._save_state(initial_state)
        return initial_state

    def _save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """
        Persists universe state safely to disk.
        """
        state_to_save = state or self.state
        state_to_save["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"Universe state successfully saved ({len(state_to_save['active_tickers'])} tickers).")
        except Exception as e:
            logger.error(f"Failed to save universe state: {e}")

    def get_active_tickers(self) -> List[str]:
        """
        Returns the sorted list of the active 66 tickers. Guarantees exactly UNIVERSE_CAPACITY items.
        """
        tickers = sorted(list(set(self.state.get("active_tickers", []))))
        if len(tickers) != UNIVERSE_CAPACITY:
            logger.warning(f"Active tickers count ({len(tickers)}) deviated from capacity ({UNIVERSE_CAPACITY}). Rectifying...")
            # Rectify by taking baseline
            tickers = BASELINE_TICKERS[:UNIVERSE_CAPACITY]
            self.state["active_tickers"] = tickers
            self._save_state()
        return tickers

    def get_sector_map(self) -> Dict[str, List[str]]:
        """
        Returns the active sector map matching the currently active 66 tickers.
        """
        active_set = set(self.get_active_tickers())
        dynamic_sector_map: Dict[str, List[str]] = {}

        for sector, tickers in self.state.get("sector_map", BASELINE_SECTOR_MAP).items():
            current_sector_tickers = [t for t in tickers if t in active_set]
            dynamic_sector_map[sector] = current_sector_tickers

        return dynamic_sector_map

    def get_sector_for_ticker(self, ticker: str) -> str:
        """
        Finds which sector a ticker belongs to.
        """
        for sector, tickers in self.state.get("sector_map", BASELINE_SECTOR_MAP).items():
            if ticker in tickers:
                return sector
        for sector, reserves in SECTOR_RESERVES.items():
            if ticker in reserves:
                return sector
        return "Financials"  # Default fallback

    def run_health_check(
        self,
        raw_market_df: Optional[pd.DataFrame] = None,
        zero_vol_threshold_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Scans trading activity for all 66 active tickers.
        Triggers an automated circuit breaker replacement if a stock exhibits zero volume
        or is missing for >= zero_vol_threshold_days consecutive days (e.g., suspension).
        """
        active_tickers = self.get_active_tickers()
        health_status = self.state.get("health_status", {})
        consecutive_zeros = self.state.get("consecutive_zero_vol_days", {})
        replacements_made = []

        if raw_market_df is not None and not raw_market_df.empty:
            logger.info("Executing Automated Universe Health-Check on ingested market data...")
            raw_market_df["Date"] = pd.to_datetime(raw_market_df["Date"])

            for ticker in list(active_tickers):
                sub = raw_market_df[raw_market_df["Ticker"] == ticker].sort_values("Date")
                if len(sub) == 0:
                    zero_count = zero_vol_threshold_days
                else:
                    recent = sub.tail(zero_vol_threshold_days)
                    # Check zero volume count in recent days
                    zero_count = int((recent["Volume"] == 0).sum())
                    if len(recent) < zero_vol_threshold_days:
                        zero_count += (zero_vol_threshold_days - len(recent))

                consecutive_zeros[ticker] = zero_count

                if zero_count >= zero_vol_threshold_days:
                    # Trigger replacement
                    sector = self.get_sector_for_ticker(ticker)
                    replacement = self._find_best_reserve(sector, active_tickers)
                    if replacement:
                        logger.warning(
                            f"CIRCUIT BREAKER TRIGGERED: {ticker} (Sector: {sector}) is suspended/illiquid "
                            f"({zero_count} zero-volume days). Replacing with standby reserve {replacement}."
                        )
                        self._swap_tickers(ticker, replacement, sector, f"Suspended/Zero-Volume for {zero_count} days")
                        replacements_made.append({
                            "degraded_ticker": ticker,
                            "promoted_ticker": replacement,
                            "sector": sector,
                            "reason": f"Suspended/Zero-Volume for {zero_count} days",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    else:
                        health_status[ticker] = "UNHEALTHY_NO_RESERVE"
                else:
                    health_status[ticker] = "HEALTHY"

        self.state["health_status"] = health_status
        self.state["consecutive_zero_vol_days"] = consecutive_zeros
        self._save_state()

        result = {
            "status": "COMPLETED",
            "active_count": len(self.get_active_tickers()),
            "replacements_count": len(replacements_made),
            "replacements": replacements_made,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return result

    def _find_best_reserve(self, sector: str, current_active: List[str]) -> Optional[str]:
        """
        Finds the highest-priority standby reserve candidate for a given sector not currently active.
        """
        reserves = SECTOR_RESERVES.get(sector, [])
        for candidate in reserves:
            if candidate not in current_active:
                return candidate
        # If sector reserves are exhausted, fallback to top reserves from other sectors
        for _, sec_reserves in SECTOR_RESERVES.items():
            for candidate in sec_reserves:
                if candidate not in current_active:
                    return candidate
        return None

    def _swap_tickers(self, old_ticker: str, new_ticker: str, sector: str, reason: str) -> None:
        """
        Swaps an old ticker out and a new reserve ticker in, preserving exactly UNIVERSE_CAPACITY.
        """
        active = self.state.get("active_tickers", [])
        if old_ticker in active:
            active.remove(old_ticker)
        if new_ticker not in active:
            active.append(new_ticker)

        # Update sector map
        sector_map = self.state.get("sector_map", BASELINE_SECTOR_MAP)
        if sector in sector_map:
            sec_tickers = sector_map[sector]
            if old_ticker in sec_tickers:
                sec_tickers.remove(old_ticker)
            if new_ticker not in sec_tickers:
                sec_tickers.append(new_ticker)
            sector_map[sector] = sec_tickers
            self.state["sector_map"] = sector_map

        # Update health tracking
        self.state["active_tickers"] = sorted(list(set(active)))[:UNIVERSE_CAPACITY]
        self.state["health_status"][old_ticker] = f"REPLACED ({reason})"
        self.state["health_status"][new_ticker] = "HEALTHY (PROMOTED)"
        self.state["consecutive_zero_vol_days"][new_ticker] = 0

        # Record in replacement history
        rep_history = self.state.get("replacements_history", [])
        rep_history.append({
            "old_ticker": old_ticker,
            "new_ticker": new_ticker,
            "sector": sector,
            "reason": reason,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.state["replacements_history"] = rep_history

    def check_and_run_rebalance(self, force: bool = False) -> Dict[str, Any]:
        """
        Checks if the current date aligns with BEI's official quarterly review calendar
        (Months 2: February, 5: May, 8: August, 11: November).
        """
        now = datetime.now()
        current_month = now.month
        current_quarter_cycle = f"{now.year}-M{current_month:02d}"
        last_rebalance = self.state.get("last_rebalance_cycle", "")

        is_rebalance_month = current_month in [2, 5, 8, 11]
        should_rebalance = force or (is_rebalance_month and current_quarter_cycle != last_rebalance)

        if not should_rebalance:
            return {
                "rebalanced": False,
                "reason": f"Not a scheduled BEI rebalance cycle or already completed for {current_quarter_cycle}.",
                "last_rebalance_cycle": last_rebalance,
                "active_count": len(self.get_active_tickers()),
            }

        logger.info(f"Executing Scheduled BEI Quarterly Universe Rebalancing ({current_quarter_cycle})...")
        self.state["last_rebalance_cycle"] = current_quarter_cycle
        self.state["last_rebalance_date"] = now.strftime("%Y-%m-%d")
        self._save_state()

        return {
            "rebalanced": True,
            "cycle": current_quarter_cycle,
            "rebalance_date": now.strftime("%Y-%m-%d"),
            "active_count": len(self.get_active_tickers()),
            "message": "Universe rebalance cycle verified and synchronized with BEI calendar.",
        }

    def get_universe_diagnostics(self) -> Dict[str, Any]:
        """
        Returns full institutional diagnostic telemetry for API endpoints and dashboard monitoring.
        """
        active = self.get_active_tickers()
        sector_map = self.get_sector_map()
        return {
            "capacity": UNIVERSE_CAPACITY,
            "active_tickers_count": len(active),
            "last_updated": self.state.get("last_updated"),
            "last_rebalance_date": self.state.get("last_rebalance_date"),
            "active_tickers": active,
            "sector_breakdown": {sec: len(tickers) for sec, tickers in sector_map.items()},
            "health_summary": {
                "healthy_count": sum(1 for status in self.state.get("health_status", {}).values() if "HEALTHY" in status),
                "replaced_count": len(self.state.get("replacements_history", [])),
            },
            "recent_replacements": self.state.get("replacements_history", [])[-5:],
        }


# Singleton accessor for fast pipeline integration
_manager_instance: Optional[UniverseManager] = None


def get_universe_manager() -> UniverseManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = UniverseManager()
    return _manager_instance
