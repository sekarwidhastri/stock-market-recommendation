from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
import joblib  # type: ignore # pyrefly: ignore [missing-import]
import numpy as np  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore # pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score  # type: ignore # pyrefly: ignore [missing-import]
from statsmodels.tsa.arima.model import ARIMA  # type: ignore # pyrefly: ignore [missing-import]
import torch  # type: ignore # pyrefly: ignore [missing-import]
import torch.nn as nn  # type: ignore # pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, TensorDataset  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    ARIMA_FORECAST_STEPS,
    DATA_DIR,
    DEFAULT_TICKERS,
    DIVIDEND_RECOMMENDATION_FILE,
    DIVIDEND_TICKERS,
    FAVORITE_TICKERS,
    FAVORITES_RECOMMENDATION_FILE,
    LOG_FORMAT,
    LSTM_LOOKBACK,
    MORNING_BRIEF_FILE,
    PORTFOLIO_ALLOCATION_FILE,
    PROCESSED_DATA_FILE,
    SECTOR_MAP,
    SNAPSHOT_FILE,
    SWING_RECOMMENDATION_FILE,
    SWING_TICKERS,
)


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ModelInference")

FEATURE_COLUMNS = [
    "Log_Return",
    "Dist_SMA_200",
    "Dist_SMA_20",
    "Volatility_20D",
    "EMA_12",
    "CMF_20",
    "Profit_Margin",
    "Bollinger_Upper",
    "Volume_Ratio",
    "Dividend_Yield",
    "MACD_Hist",
    "Volume_SMA_20",
    "Market_Cap",
    "PB_Ratio",
    "MACD_Signal",
    "IHSG_Return_1D",
    "US_10Y_Yield_Delta",
    "USD_IDR_Return_1D",
    "Brent_Oil_Return_1D",
    "Oil_Energy_Tailwind",
    "Rate_Bank_Sensitivity",
    "FX_Consumer_Headwind",
]

LSTM_FEATURE_COLS = [
    "Return_1D",
    "RSI_14",
    "Dist_SMA_20",
    "CMF_20",
    "Volume_Ratio",
    "IHSG_Return_1D",
    "US_10Y_Yield_Delta",
    "USD_IDR_Return_1D",
    "Brent_Oil_Return_1D",
    "Oil_Energy_Tailwind",
    "Rate_Bank_Sensitivity",
    "FX_Consumer_Headwind",
]
TARGET_COLUMN = "Target_Class_5D"
MODEL_PATH = DATA_DIR / "processed" / "alpha_model.joblib"
LSTM_MODEL_PATH = DATA_DIR / "processed" / "lstm_model.pth"


# ==============================================================================
# 1. ARIMA Statistical Time-Series Predictor
# ==============================================================================
class ARIMAPredictor:
    @staticmethod
    def predict_ticker(close_series: pd.Series, steps: int = ARIMA_FORECAST_STEPS) -> Tuple[float, float]:
        clean_prices = close_series.dropna().values
        if len(clean_prices) < 30:
            return 0.0, 0.50

        recent_prices = clean_prices[-120:]
        current_price = recent_prices[-1]

        try:
            model = ARIMA(recent_prices, order=(1, 1, 1))
            fit_res = model.fit()
            forecast = fit_res.forecast(steps=steps)
            expected_future_price = float(forecast[-1])
            expected_return = (expected_future_price - current_price) / (current_price + 1e-9)
            prob = float(1.0 / (1.0 + np.exp(-25.0 * expected_return)))
            return round(expected_return, 4), round(np.clip(prob, 0.10, 0.90), 4)
        except Exception:
            try:
                model = ARIMA(recent_prices, order=(1, 0, 0))
                fit_res = model.fit()
                forecast = fit_res.forecast(steps=steps)
                expected_future_price = float(forecast[-1])
                expected_return = (expected_future_price - current_price) / (current_price + 1e-9)
                prob = float(1.0 / (1.0 + np.exp(-25.0 * expected_return)))
                return round(expected_return, 4), round(np.clip(prob, 0.10, 0.90), 4)
            except Exception:
                return 0.0, 0.50


# ==============================================================================
# 2. PyTorch Deep Learning LSTM Sequence Model
# ==============================================================================
class PyTorchLSTMNet(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.sigmoid(self.fc(last_step))


class PyTorchLSTMTrainer:
    def __init__(self, lookback: int = LSTM_LOOKBACK):
        self.lookback = lookback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PyTorchLSTMNet(input_dim=len(LSTM_FEATURE_COLS)).to(self.device)

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences, labels = [], []
        for _, group in df.groupby("Ticker"):
            grp = group.sort_values(by="Date").copy()
            for col in LSTM_FEATURE_COLS:
                grp[col] = grp[col].fillna(0.0)

            features = grp[LSTM_FEATURE_COLS].values
            target = grp[TARGET_COLUMN].values

            if "RSI_14" in LSTM_FEATURE_COLS:
                rsi_idx = LSTM_FEATURE_COLS.index("RSI_14")
                features[:, rsi_idx] = features[:, rsi_idx] / 100.0

            for i in range(len(features) - self.lookback):
                seq_x = features[i : i + self.lookback]
                label_y = target[i + self.lookback]
                if not np.isnan(label_y):
                    sequences.append(seq_x)
                    labels.append(label_y)

        if not sequences:
            return torch.empty(0), torch.empty(0)

        return (
            torch.tensor(np.array(sequences), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1),
        )

    def train_lstm(self, df: pd.DataFrame, epochs: int = 6, batch_size: int = 256) -> float:
        X, y = self.prepare_sequences(df)
        if len(X) == 0:
            return 0.50

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)

        self.model.train()
        total_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), LSTM_MODEL_PATH)
        return total_loss / len(loader)

    def predict_latest_ticker(self, ticker_group: pd.DataFrame) -> float:
        self.model.eval()
        grp = ticker_group.sort_values(by="Date").copy()
        for col in LSTM_FEATURE_COLS:
            grp[col] = grp[col].fillna(0.0)

        features = grp[LSTM_FEATURE_COLS].values
        if len(features) < self.lookback:
            return 0.50

        if "RSI_14" in LSTM_FEATURE_COLS:
            rsi_idx = LSTM_FEATURE_COLS.index("RSI_14")
            features[:, rsi_idx] = features[:, rsi_idx] / 100.0

        latest_window = features[-self.lookback :]
        input_tensor = torch.tensor(np.expand_dims(latest_window, axis=0), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            prob = self.model(input_tensor).item()
        return round(float(prob), 4)


def round_to_idx_tick(price: float) -> int:
    """
    Rounds a stock price to official Indonesia Stock Exchange (IDX / BEI) tick size rules:
    - Price < 200: tick = 1 (minimum price = 50 on regular board)
    - 200 <= Price < 500: tick = 2
    - 500 <= Price < 2000: tick = 5
    - 2000 <= Price < 5000: tick = 10
    - Price >= 5000: tick = 25
    """
    p = max(50.0, float(price))
    if p < 200.0:
        tick = 1.0
    elif p < 500.0:
        tick = 2.0
    elif p < 2000.0:
        tick = 5.0
    elif p < 5000.0:
        tick = 10.0
    else:
        tick = 25.0
    rounded = int(round(p / tick) * tick)
    return max(50, rounded)


def capped_weights(score: np.ndarray, cap: float = 0.25, budget: float = 0.80) -> np.ndarray:
    """
    Convex cap-and-redistribute algorithm guaranteeing that:
    1. No asset exceeds single-stock cap (default 25%)
    2. Sum of equity weights never exceeds budget (default 80%),
       strictly reserving at least 20% for cash reserve (Kas Siaga).
    """
    w = np.maximum(np.asarray(score, dtype=float), 0.0)
    total_score = np.sum(w)
    if total_score <= 1e-12:
        return np.zeros_like(w)

    w = (w / total_score) * budget
    for _ in range(50):
        over = w > (cap + 1e-12)
        if not np.any(over):
            break
        excess = np.sum(w[over] - cap)
        w[over] = cap
        free = (~over) & (w > 0) & (w < (cap - 1e-12))
        if not np.any(free):
            break
        w[free] += excess * (w[free] / np.sum(w[free]))

    # Final guard to strictly respect budget
    current_sum = np.sum(w)
    if current_sum > budget + 1e-6:
        w = (w / current_sum) * budget
    return w


# ==============================================================================
# 3. Multi-Engine Quantitative Alpha Model & Portfolio Optimizer
# ==============================================================================
class QuantitativeAlphaModel:
    """
    Multi-Engine Alpha Model with Actionable Technical Recommendations
    (Buy on Weakness, Buy on Breakout, Sell on Strength) & Portfolio Allocation Optimizer.
    """

    def __init__(self, processed_data_path: str = str(PROCESSED_DATA_FILE)):
        self.processed_data_path = processed_data_path
        self.gbdt_model = HistGradientBoostingClassifier(
            learning_rate=0.02,
            max_iter=150,
            max_depth=4,
            min_samples_leaf=40,
            l2_regularization=3.0,
            random_state=42,
        )
        self.lstm_trainer = PyTorchLSTMTrainer()
        self.arima_predictor = ARIMAPredictor()
        self.ticker_to_sector = {t: sector for sector, tickers in SECTOR_MAP.items() for t in tickers}
        self._cached_latest_rows: Optional[pd.DataFrame] = None

    def load_processed_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.processed_data_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(by=["Date", "Ticker"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        valid_df = df.dropna(subset=[TARGET_COLUMN]).copy()
        valid_df.sort_values(by="Date", inplace=True)

        for c in FEATURE_COLUMNS:
            if c not in valid_df.columns:
                valid_df[c] = 0.0

        split_idx = int(len(valid_df) * 0.8)
        train_df = valid_df.iloc[:split_idx]
        test_df = valid_df.iloc[split_idx:]

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN].astype(int)
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN].astype(int)

        logger.info("Fitting Tabular GBDT Model with Macro Context...")
        self.gbdt_model.fit(X_train, y_train)

        test_probs = self.gbdt_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        prec = precision_score(y_test, test_preds, zero_division=0)
        auc = roc_auc_score(y_test, test_probs)

        # High-Conviction (Threshold >= 0.55)
        high_conv_mask = (test_probs >= 0.55)
        if np.sum(high_conv_mask) > 0:
            high_conv_prec = precision_score(y_test[high_conv_mask], (test_probs[high_conv_mask] >= 0.5).astype(int), zero_division=0)
        else:
            high_conv_prec = prec

        # Top Decile (Top 10% highest conviction predictions)
        top_decile_cutoff = np.percentile(test_probs, 90)
        top_decile_mask = (test_probs >= top_decile_cutoff)
        top_decile_prec = precision_score(y_test[top_decile_mask], (test_probs[top_decile_mask] >= 0.5).astype(int), zero_division=0)

        logger.info("Fitting Macro-Aware PyTorch LSTM Sequence Model...")
        lstm_train_loss = self.lstm_trainer.train_lstm(train_df, epochs=6, batch_size=256)

        # Evaluate LSTM on test split sequences
        X_lstm_test, y_lstm_test = self.lstm_trainer.prepare_sequences(test_df)
        if len(X_lstm_test) > 0:
            self.lstm_trainer.model.eval()
            with torch.no_grad():
                lstm_test_probs = self.lstm_trainer.model(X_lstm_test.to(self.lstm_trainer.device)).cpu().numpy().flatten()
            y_lstm_true = y_lstm_test.numpy().flatten()
            lstm_preds = (lstm_test_probs >= 0.5).astype(int)
            lstm_acc = accuracy_score(y_lstm_true, lstm_preds)
            lstm_prec = precision_score(y_lstm_true, lstm_preds, zero_division=0)
            lstm_auc = roc_auc_score(y_lstm_true, lstm_test_probs)
        else:
            lstm_acc, lstm_prec, lstm_auc = 0.5, 0.5, 0.5

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.gbdt_model, MODEL_PATH)

        return {
            "GBDT_Accuracy": round(float(acc), 4),
            "GBDT_Precision": round(float(prec), 4),
            "GBDT_ROC_AUC": round(float(auc), 4),
            "High_Conviction_Precision_55": round(float(high_conv_prec), 4),
            "Top_Decile_Precision": round(float(top_decile_prec), 4),
            "LSTM_Accuracy": round(float(lstm_acc), 4),
            "LSTM_Precision": round(float(lstm_prec), 4),
            "LSTM_ROC_AUC": round(float(lstm_auc), 4),
            "LSTM_Train_Loss": round(float(lstm_train_loss), 4),
        }

    def get_or_extract_latest_rows(self, df: pd.DataFrame, ticker_filter: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Caches and extracts latest rows across the default universe to avoid redundant multi-model computation.
        """
        if self._cached_latest_rows is None:
            self._cached_latest_rows = self._extract_latest_rows_with_ensemble(df, DEFAULT_TICKERS)
        if ticker_filter is None:
            return self._cached_latest_rows.copy()
        return self._cached_latest_rows[self._cached_latest_rows["Ticker"].isin(ticker_filter)].copy().reset_index(drop=True)

    def _extract_latest_rows_with_ensemble(self, df: pd.DataFrame, ticker_filter: List[str]) -> pd.DataFrame:
        filtered_df = df[df["Ticker"].isin(ticker_filter)].copy()
        pre_records = []

        for ticker, group in filtered_df.groupby("Ticker"):
            grp = group.sort_values(by="Date")
            latest_row = grp.iloc[-1].to_dict()

            feat_vector = pd.DataFrame([latest_row])[FEATURE_COLUMNS]
            gbdt_prob = float(self.gbdt_model.predict_proba(feat_vector)[:, 1][0])
            arima_ret, arima_prob = self.arima_predictor.predict_ticker(grp["Close"], steps=5)
            lstm_prob = self.lstm_trainer.predict_latest_ticker(grp)

            blended_prob = round(0.50 * gbdt_prob + 0.30 * lstm_prob + 0.20 * arima_prob, 4)

            latest_row["GBDT_Prob"] = round(gbdt_prob, 4)
            latest_row["ARIMA_Prob"] = arima_prob
            latest_row["ARIMA_Expected_Return"] = arima_ret
            latest_row["LSTM_Prob"] = lstm_prob
            latest_row["Bullish_Probability"] = blended_prob
            latest_row["Conviction_Score"] = np.abs(blended_prob - 0.50)
            latest_row["_grp"] = grp
            pre_records.append(latest_row)

        if not pre_records:
            return pd.DataFrame()

        # Cross-Sectional Ranking: Top Decile (top 10% highest conviction picks across active universe)
        all_probs = [r["Bullish_Probability"] for r in pre_records]
        top_decile_cutoff = float(np.percentile(all_probs, 90)) if len(all_probs) >= 10 else 0.55

        latest_records = []
        for latest_row in pre_records:
            grp = latest_row.pop("_grp")
            ticker = latest_row["Ticker"]
            blended_prob = latest_row["Bullish_Probability"]
            is_top_decile = bool(blended_prob >= top_decile_cutoff)
            latest_row["Is_Top_Decile"] = is_top_decile

            # Calculate 14-day Average True Range (ATR)
            high_low = grp["High"] - grp["Low"]
            high_close = (grp["High"] - grp["Close"].shift(1)).abs()
            low_close = (grp["Low"] - grp["Close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = float(tr.rolling(14).mean().iloc[-1])
            close = float(latest_row.get("Close", 1000.0))
            if np.isnan(atr_14) or atr_14 <= 0:
                atr_14 = max(close * 0.02, 1.0)

            # Technical Action Signals
            rsi = float(latest_row.get("RSI_14", 50.0) or 50.0)
            vol_ratio = float(latest_row.get("Volume_Ratio", 1.0) or 1.0)
            res1 = float(latest_row.get("Resistance_1", close * 1.03) or (close * 1.03))
            sup1 = float(latest_row.get("Support_1", close * 0.97) or (close * 0.97))

            # High-Conviction Ambang Eksekusi (>= 0.55 atau Top Decile dengan floor >= 0.52)
            is_buy_eligible = (blended_prob >= 0.55) or (is_top_decile and blended_prob >= 0.52)

            if is_buy_eligible and (rsi <= 45.0 or close <= sup1 * 1.01):
                action = "BUY ON WEAKNESS"
            elif is_buy_eligible and vol_ratio >= 1.25 and close >= res1 * 0.99:
                action = "BUY ON BREAKOUT"
            elif is_buy_eligible:
                action = "TRADING BUY"
            elif blended_prob <= 0.45 or rsi >= 70.0:
                action = "SELL ON STRENGTH"
            else:
                action = "HOLD"

            # Entry, Target Price (TP), Stop Loss (SL), Risk-Reward with IDX tick size rounding
            entry_price = round_to_idx_tick(close)

            if action in ["BUY ON WEAKNESS", "BUY ON BREAKOUT", "TRADING BUY"]:
                raw_sl = max(sup1, close - (1.5 * atr_14))
                raw_sl = min(raw_sl, close - (0.8 * atr_14))
                stop_loss = round_to_idx_tick(raw_sl)
                if stop_loss >= entry_price:
                    stop_loss = round_to_idx_tick(close - (1.2 * atr_14))
                stop_loss = max(50, stop_loss)

                raw_tp = min(res1, close + (2.0 * atr_14))
                raw_tp = max(raw_tp, close + (1.2 * atr_14))
                target_price = round_to_idx_tick(raw_tp)
                if target_price <= entry_price:
                    target_price = round_to_idx_tick(close + (1.5 * atr_14))
            elif action == "SELL ON STRENGTH":
                target_price = round_to_idx_tick(max(res1, close + atr_14))
                stop_loss = round_to_idx_tick(min(sup1, close - atr_14))
            else:  # HOLD
                target_price = round_to_idx_tick(max(res1, close * 1.03))
                stop_loss = round_to_idx_tick(min(sup1, close * 0.97))

            potential_gain = max(1.0, float(target_price - entry_price))
            potential_risk = max(1.0, float(abs(entry_price - stop_loss)))
            rr_ratio = round(potential_gain / potential_risk, 2)

            latest_row["Recommendation"] = action
            latest_row["Sector"] = self.ticker_to_sector.get(ticker, "General")
            latest_row["Entry_Price"] = entry_price
            latest_row["Target_Price"] = target_price
            latest_row["Stop_Loss"] = stop_loss
            latest_row["Risk_Reward_Ratio"] = rr_ratio
            latest_row["ATR_14"] = round(atr_14, 2)

            latest_records.append(latest_row)

        if not latest_records:
            return pd.DataFrame()

        latest_df = pd.DataFrame(latest_records)
        all_expected_cols = [
            "Beta_IHSG", "Sharpe_Ratio", "Max_Drawdown_1Y", "Annualized_Return_1Y",
            "Debt_to_Equity", "Current_Ratio", "RSI_14", "MACD_Hist", "MFI_14", "CMF_20",
            "Volatility_20D", "PE_Ratio", "PB_Ratio", "ROE", "Dividend_Yield",
            "GARCH_Vol", "VaR_95_1D", "VaR_99_1D", "ES_95_1D", "Sector"
        ]
        for c in all_expected_cols:
            if c not in latest_df.columns:
                latest_df[c] = None

        return latest_df

    def generate_swing_recommendations(self, df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
        logger.info("Generating Strategy 1: Daily Swing Trader (LQ45)...")
        latest_df = self.get_or_extract_latest_rows(df, SWING_TICKERS)
        priority_map = {
            "BUY ON BREAKOUT": 3,
            "BUY ON WEAKNESS": 2,
            "TRADING BUY": 1,
            "HOLD": 0,
            "SELL ON STRENGTH": -1,
        }
        latest_df["Action_Priority"] = latest_df["Recommendation"].map(priority_map).fillna(0)
        ranked_df = latest_df.sort_values(
            by=["Action_Priority", "Bullish_Probability"], ascending=[False, False]
        ).head(top_n).reset_index(drop=True)
        ranked_df.drop(columns=["Action_Priority"], inplace=True)

        ranked_df["Date"] = pd.to_datetime(ranked_df["Date"]).dt.strftime("%Y-%m-%d")
        SWING_RECOMMENDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(SWING_RECOMMENDATION_FILE, index=False)
        return ranked_df

    def generate_dividend_recommendations(self, df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
        logger.info("Generating Strategy 2: Dividend & Value Investor...")
        latest_df = self.get_or_extract_latest_rows(df, DIVIDEND_TICKERS)
        div_yield = latest_df["Dividend_Yield"].fillna(0.0) if "Dividend_Yield" in latest_df.columns else 0.0
        roe = latest_df["ROE"].fillna(0.0) if "ROE" in latest_df.columns else 0.0
        der = latest_df["Debt_to_Equity"].fillna(1.0) if "Debt_to_Equity" in latest_df.columns else 1.0

        latest_df["Dividend_Score"] = (
            div_yield * 0.4 + roe * 100.0 * 0.25 + latest_df["Bullish_Probability"] * 25.0 - der * 5.0
        )
        ranked_df = latest_df.sort_values(by=["Dividend_Score", "Dividend_Yield"], ascending=[False, False]).head(top_n).reset_index(drop=True)
        ranked_df["Date"] = pd.to_datetime(ranked_df["Date"]).dt.strftime("%Y-%m-%d")
        DIVIDEND_RECOMMENDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(DIVIDEND_RECOMMENDATION_FILE, index=False)
        return ranked_df

    def generate_favorites_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating Strategy 3: 10 Favorite Stocks Portfolio...")
        latest_df = self.get_or_extract_latest_rows(df, FAVORITE_TICKERS)
        ranked_df = latest_df.sort_values(by="Bullish_Probability", ascending=False).reset_index(drop=True)
        ranked_df["Date"] = pd.to_datetime(ranked_df["Date"]).dt.strftime("%Y-%m-%d")
        FAVORITES_RECOMMENDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(FAVORITES_RECOMMENDATION_FILE, index=False)
        return ranked_df

    def optimize_portfolio_allocation(self, df: pd.DataFrame, capital_idr: float = 50000000.0) -> pd.DataFrame:
        """
        Optimal Portfolio Weighting Algorithm (Sharpe-Weighted Convex Capped Weights with Cash Reserve).
        Guarantees:
        1. Candidates must be active BUY signals with positive Sharpe ratio.
        2. Maximum single-stock equity cap = 25%.
        3. Maximum total equity allocation = 80%, strictly guaranteeing Cash Reserve >= 20%.
        4. Exact unspent IDR from 100-shares lot rounding is added back to Cash Reserve.
        """
        logger.info(f"Computing optimal portfolio allocation for total capital: Rp {capital_idr:,.0f}...")
        latest_df = self.get_or_extract_latest_rows(df, DEFAULT_TICKERS)

        # 1. Primary candidates: active tactical BUY recommendations with positive Sharpe
        tactical_buys = latest_df[
            latest_df["Recommendation"].isin(["BUY ON WEAKNESS", "BUY ON BREAKOUT", "TRADING BUY"])
            & (latest_df["Sharpe_Ratio"].fillna(0.0) > 0.0)
        ].copy()

        # 2. Institutional Core-Satellite: Supplement with top defensive Dividend/Value stocks
        # if tactical buys with positive Sharpe are fewer than 3
        if len(tactical_buys) < 3:
            div_candidates = latest_df[
                latest_df["Ticker"].isin(DIVIDEND_TICKERS)
                & (~latest_df["Ticker"].isin(tactical_buys["Ticker"]))
                & (latest_df["Recommendation"] != "SELL ON STRENGTH")
                & (latest_df["Sharpe_Ratio"].fillna(0.0) > 0.0)
            ].copy()
            div_candidates = div_candidates.sort_values(
                by=["Sharpe_Ratio", "Dividend_Yield"], ascending=[False, False]
            )
            needed = 4 - len(tactical_buys)
            supplement = div_candidates.head(needed)
            buy_candidates = pd.concat([tactical_buys, supplement], ignore_index=True)
        else:
            buy_candidates = tactical_buys.copy()

        # If zero candidates pass the quantitative gate across universe, reserve 100% in Cash
        if len(buy_candidates) == 0:
            logger.warning("No candidates with positive Sharpe ratio found. Holding 100% Cash Reserve.")
            allocation_records = [{
                "Ticker": "KAS SIAGA (CASH)",
                "Sector": "Money Market / Risk Reserve",
                "Recommendation": "RESERVE",
                "Allocation_Pct": 100.0,
                "Nominal_IDR": round(capital_idr, 0),
                "Shares_Lot": 0,
                "Close_Price": 1.0,
                "Target_Price": 1.0,
                "Stop_Loss": 1.0,
                "Bullish_Probability": 0.50,
            }]
            alloc_df = pd.DataFrame(allocation_records)
            PORTFOLIO_ALLOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            alloc_df.to_csv(PORTFOLIO_ALLOCATION_FILE, index=False)
            logger.info(f"Portfolio allocation (100% Cash Reserve) saved to {PORTFOLIO_ALLOCATION_FILE}")
            return alloc_df

        # Sort top candidates by probability and Sharpe ratio (max 5)
        buy_candidates = buy_candidates.sort_values(
            by=["Bullish_Probability", "Sharpe_Ratio"], ascending=[False, False]
        ).head(5)

        sharpe_raw = buy_candidates["Sharpe_Ratio"].fillna(0.1).to_numpy()
        sharpe_safe = np.maximum(sharpe_raw, 0.05)
        garch_vol = buy_candidates["GARCH_Vol"].fillna(0.25).to_numpy()

        # Score = Sharpe / (Volatility + 0.05)
        raw_scores = sharpe_safe / (garch_vol + 0.05)
        weights = capped_weights(raw_scores, cap=0.25, budget=0.80)

        total_equity_spent = 0.0
        allocation_records = []
        for i, (_, row) in enumerate(buy_candidates.iterrows()):
            w = round(float(weights[i]), 4)
            target_nominal = capital_idr * w
            close = float(row["Close"])
            lot_price = close * 100.0
            num_lots = int(target_nominal // lot_price) if lot_price > 0 else 0
            actual_nominal = num_lots * lot_price

            if num_lots > 0:
                total_equity_spent += actual_nominal
                actual_pct = round((actual_nominal / capital_idr) * 100.0, 1)
                allocation_records.append({
                    "Ticker": row["Ticker"],
                    "Sector": row.get("Sector", "General"),
                    "Recommendation": row["Recommendation"],
                    "Allocation_Pct": actual_pct,
                    "Nominal_IDR": round(actual_nominal, 0),
                    "Shares_Lot": num_lots,
                    "Close_Price": close,
                    "Target_Price": row["Target_Price"],
                    "Stop_Loss": row["Stop_Loss"],
                    "Bullish_Probability": row["Bullish_Probability"],
                })

        # Remainder returned to Cash Reserve (including unspent fraction from 100-share lot rounding)
        actual_cash_idr = max(0.0, capital_idr - total_equity_spent)
        actual_cash_pct = round((actual_cash_idr / capital_idr) * 100.0, 1)

        allocation_records.append({
            "Ticker": "KAS SIAGA (CASH)",
            "Sector": "Money Market / Risk Reserve",
            "Recommendation": "RESERVE",
            "Allocation_Pct": actual_cash_pct,
            "Nominal_IDR": round(actual_cash_idr, 0),
            "Shares_Lot": 0,
            "Close_Price": 1.0,
            "Target_Price": 1.0,
            "Stop_Loss": 1.0,
            "Bullish_Probability": 0.50,
        })

        alloc_df = pd.DataFrame(allocation_records)
        PORTFOLIO_ALLOCATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        alloc_df.to_csv(PORTFOLIO_ALLOCATION_FILE, index=False)
        logger.info(f"Portfolio allocation saved to {PORTFOLIO_ALLOCATION_FILE}")
        return alloc_df


def export_snapshot_json(
    swing_df: pd.DataFrame,
    div_df: pd.DataFrame,
    fav_df: pd.DataFrame,
    alloc_df: pd.DataFrame,
    metrics: Dict[str, float],
    all_df: Optional[pd.DataFrame] = None,
) -> None:
    brief_data = {}
    if MORNING_BRIEF_FILE.exists():
        try:
            with open(MORNING_BRIEF_FILE, "r", encoding="utf-8") as f:
                brief_data = json.load(f)
        except Exception:
            pass

    def clean_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
        recs = df.to_dict(orient="records")
        for r in recs:
            for k, v in list(r.items()):
                if pd.isna(v) or v is None:
                    r[k] = None
                elif isinstance(v, (np.floating, float)):
                    r[k] = round(float(v), 4)
                elif isinstance(v, (np.integer, int)):
                    r[k] = int(v)
        return recs

    if all_df is not None and not all_df.empty:
        all_recs = all_df.copy()
    else:
        all_recs = pd.concat([fav_df, div_df, swing_df], ignore_index=True).drop_duplicates(subset=["Ticker"])

    if "Date" in all_recs.columns:
        all_recs["Date"] = pd.to_datetime(all_recs["Date"]).dt.strftime("%Y-%m-%d")

    snapshot = {
        "status": "online",
        "data_asof": str(swing_df["Date"].iloc[0]) if not swing_df.empty else datetime.today().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "morning_brief": brief_data,
        "portfolio_allocation": clean_records(alloc_df),
        "recommendations_swing": clean_records(swing_df),
        "recommendations_dividend": clean_records(div_df),
        "recommendations_favorites": clean_records(fav_df),
        "all_recommendations": clean_records(all_recs),
    }

    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    logger.info(f"Consolidated snapshot safely written to {SNAPSHOT_FILE}")


def run_model_inference_pipeline() -> Tuple[Dict[str, float], pd.DataFrame]:
    model_engine = QuantitativeAlphaModel()
    df = model_engine.load_processed_data()
    metrics = model_engine.train_and_evaluate(df)

    swing_recs = model_engine.generate_swing_recommendations(df)
    div_recs = model_engine.generate_dividend_recommendations(df)
    fav_recs = model_engine.generate_favorites_recommendations(df)
    alloc_recs = model_engine.optimize_portfolio_allocation(df, capital_idr=50000000.0)

    all_universe_df = model_engine.get_or_extract_latest_rows(df, DEFAULT_TICKERS)
    export_snapshot_json(swing_recs, div_recs, fav_recs, alloc_recs, metrics, all_df=all_universe_df)

    return metrics, swing_recs


if __name__ == "__main__":
    run_model_inference_pipeline()

