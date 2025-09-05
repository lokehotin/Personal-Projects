"""
Statistical Arbitrage Pipeline
================================

USAGE (examples)
----------------
# Basic run with WRDS fundamentals (Compustat) + 2 tickers
python stat_arb_ff_v2.py --wrds --tickers AAPL,MSFT \
    --start 2020-01-01 --end 2021-12-31 --horizon 5 \
    --gross 1.0 --tc_bps 5 --out_dir results/aapl_msft

# Run with 5-day horizon, Ridge/RandomForest ensemble, top-3 long/short
python stat_arb_ff_v2.py --wrds --tickers TSLA,GOOGL,NVDA,AAPL,MSFT \
    --start 2019-01-01 --end 2020-12-31 --horizon 5 \
    --gross 1.0 --topk 3 --tc_bps 10 \
    --out_dir results/topk3

# Quiet run (suppress logs), small window test for quick debugging
python stat_arb_ff_v2.py --wrds --tickers AAPL,MSFT \
    --start 2022-01-01 --end 2022-06-30 --horizon 5 \
    --min_train 50 --quiet

# Run without WRDS fundamentals (momentum + market factors only)
python stat_arb_ff_v2.py --tickers AAPL,MSFT,NVDA \
    --start 2020-01-01 --end 2020-12-31 --horizon 10 \
    --gross 1.5 --tc_bps 5 \
    --out_dir results/no_wrds

Notes
-----
- `--wrds' pulls point-in-time fundamentals directly from WRDS Compustat.
- '--horizon' sets the forward return window (e.g. 5 = 5-day returns).
- '--gross' is target gross exposure (sum of abs weights).
- '--topk' selects equal-weighted top-k longs and shorts instead of z-score weights.
- '--tc_bps' applies transaction cost per unit turnover, in basis points.
- Outputs include:
    * signals_latest.csv
    * expected_returns_timeseries.csv
    * probabilities_timeseries.csv
    * weights_timeseries.csv
    * backtest_pnl.csv
    * equity_curve.png / equity_curve_series.csv
"""

import argparse
import logging
import warnings
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import yfinance as yf
except Exception:
    raise RuntimeError("yfinance not installed. `pip install yfinance`")

try:
    import wrds  # pip install wrds
except Exception:
    wrds = None

# ======== Quiet logging / warnings ========


def setup_quiet_logging(quiet: bool):
    if quiet:
        warnings.filterwarnings("ignore")
        for name in ("yfinance", "urllib3", "matplotlib", "pandas", "PIL"):
            logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
    else:
        warnings.filterwarnings("ignore", category=FutureWarning)


# ======== Configuration ========


DEFAULT_TICKERS = [
    "TSLA",  # Tesla
    "GOOGL",  # Alphabet
    "NVDA",  # Nvidia
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "META",  # Meta
    "ORCL",  # Oracle
    "AMZN",  # Amazon
    "INTC",  # Intel
    "AMD",  # AMD
]
MARKET_TICKER = "^GSPC"  # S&P 500 index


# ======== Utilities ========


def cross_sectional_zscore(df):
    mean = df.mean(axis=1)
    standard_deviation = df.std(axis=1).replace(0, np.nan)
    z_score = df.sub(mean, axis=0).div(standard_deviation, axis=0)
    return z_score.fillna(0.0)


# ======== Data Loading ========


def load_prices(tickers, start, end):
    all_data = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    if isinstance(all_data.columns, pd.MultiIndex):
        prices = all_data["Close"].copy()
    else:
        prices = all_data[["Close"]].copy()
        prices.columns = [tickers[0]]
    prices = prices.dropna(how="all")
    return prices


def load_market(start, end):
    market_data = yf.download(
        MARKET_TICKER, start=start, end=end, auto_adjust=True, progress=False
    )
    market_price = market_data["Close"]
    if isinstance(market_price, pd.DataFrame):
        if market_price.shape[1] == 1:
            market_price = market_price.iloc[:, 0]
        elif MARKET_TICKER in getattr(market_price, "columns", []):
            market_price = market_price[MARKET_TICKER]
        else:
            raise ValueError(
                f"Unexpected Close columns for {MARKET_TICKER}: {list(market_price.columns)}"
            )
    return market_price.rename("MARKET").dropna()


def load_metadata(tickers):
    rows = []
    for t in tickers:
        mc = pe = pb = profit_margin = np.nan
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "fast_info", {}) or {}
            mc = fi.get("market_cap", np.nan)
            pe = fi.get("trailing_pe", np.nan)
            try:
                info = tk.get_info()
                pb = info.get("priceToBook", np.nan)
                profit_margin = info.get("profitMargins", np.nan)
            except Exception:
                pass
        except Exception:
            pass
        rows.append(
            {
                "ticker": t,
                "market_cap": mc,
                "pe": pe,
                "pb": pb,
                "profit_margin": profit_margin,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def load_macro_from_csv(path):
    """Optional macro: first column is the date, the rest are features."""
    if not path:
        return None
    df = pd.read_csv(path, parse_dates=[0])
    df = df.set_index(df.columns[0]).sort_index()
    return df


def wrds_connect():
    """
    Create a WRDS connection if the package is available.
    Uses interactive prompt or .pgpass/.wrds if configured.
    """
    if wrds is None:
        raise RuntimeError("wrds package not installed. `pip install wrds`")
    return wrds.Connection()


def load_fundamentals_from_wrds(
    db,
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """
    Pull quarterly fundamentals from comp.fundq using PIT availability date (rdq).
    Minimal set of vars to compute SIZE, VAL, PROFIT, LEV.

    Returns dict { "SIZE": Date×Ticker, "VAL": ..., "PROFIT": ..., "LEV": ... }.
    """

    # SQL: only what we need, quarterly
    in_tics = ",".join([f"'{t}'" for t in tickers])
    q = f"""
        select gvkey, tic, datadate, rdq,
               atq, ltq, ceqq, niq, cshoq, prccq
        from comp.fundq
        where indfmt='INDL'
          and datafmt='STD'
          and consol='C'
          and popsrc='D'
          and tic in ({in_tics})
          and datadate between '{start}' and '{end}'
    """
    df = db.raw_sql(q)

    # Clean
    df = df.copy()
    df["tic"] = df["tic"].str.upper().str.strip()
    # Availability date: use rdq; if missing, fallback to datadate (conservative users may prefer to drop)
    df["date"] = pd.to_datetime(df["rdq"]).fillna(pd.to_datetime(df["datadate"]))
    df = df.dropna(subset=["date"])
    # Numerics
    for c in ["atq", "ltq", "ceqq", "niq", "cshoq", "prccq"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Factors
    mktcap = (df["cshoq"] * df["prccq"]).replace({0: np.nan})
    size = np.log(mktcap)
    val = (df["ceqq"] / mktcap).replace([np.inf, -np.inf], np.nan)
    profit = (df["niq"] / df["atq"]).replace([np.inf, -np.inf], np.nan)
    lev = (df["ltq"] / df["atq"]).replace([np.inf, -np.inf], np.nan)

    ff = pd.DataFrame(
        {
            "date": df["date"],
            "ticker": df["tic"],
            "SIZE": size,
            "VAL": val,
            "PROFIT": profit,
            "LEV": lev,
        }
    ).dropna(subset=["ticker"])

    # Pivot to Date × Ticker for each factor
    out: Dict[str, pd.DataFrame] = {}
    for fac in ["SIZE", "VAL", "PROFIT", "LEV"]:
        wide = ff.pivot(index="date", columns="ticker", values=fac).sort_index()
        out[fac] = wide
    return out


def load_macro_from_wrds(db, start: str, end: str) -> pd.DataFrame:
    """
    Pull 3 common macro series from WRDS/FRED:
      - CPIAUCSL  -> CPI
      - UNRATE    -> UNRATE
      - FEDFUNDS  -> FEDFUNDS

    This function auto-detects the series-data table name in your WRDS 'fred' schema.
    It then returns a daily-indexed DataFrame (values forward-filled).
    """
    # 1) Discover table names available to you in the 'fred' library
    try:
        fred_tables = set(db.list_tables("fred"))
    except Exception as e:
        raise RuntimeError(f"Unable to list FRED tables on WRDS: {e}")

    # Common variants for the per-observation time series table:
    candidates = [
        "data",  # some sites expose fred.data
        "series_data",  # fred.series_data
        "fred_series_data",  # fred.fred_series_data
        "mdata",  # fred.mdata (monthly data)
        "monthly",  # fred.monthly
    ]

    table = next((t for t in candidates if t in fred_tables), None)
    if table is None:
        # As a fallback, show user what *is* available to help debugging
        raise RuntimeError(
            "Could not find a FRED series-data table in your WRDS account.\n"
            f"Found tables: {sorted(fred_tables)}\n"
            "Ask your library which FRED table contains (series_id, date, value), "
            "or share the list above and I’ll adapt the loader."
        )

    series_map = {
        "CPIAUCSL": "CPI",
        "UNRATE": "UNRATE",
        "FEDFUNDS": "FEDFUNDS",
    }

    frames = []
    for code, alias in series_map.items():
        # Build a generic SQL that works for all the candidate tables (they tend to share columns)
        q = f"""
            select date, value
            from fred.{table}
            where series_id = '{code}'
              and date between '{start}' and '{end}'
            order by date
        """
        df = db.raw_sql(q)
        if df.empty:
            # Try an alternative column naming seen at some sites
            # (rare, but keeps this resilient)
            q_alt = f"""
                select date, val as value
                from fred.{table}
                where series_id = '{code}'
                  and date between '{start}' and '{end}'
                order by date
            """
            df = db.raw_sql(q_alt)

        if df.empty:
            raise RuntimeError(
                f"No data returned for {code} from fred.{table}. "
                "Your site may use a different table or column names."
            )

        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"value": alias}).set_index("date").sort_index()
        frames.append(df)

    macro_m = pd.concat(frames, axis=1).sort_index()
    # Monthly → daily index; we’ll reindex again to trading days later
    macro_d = macro_m.resample("D").ffill()
    return macro_d


# ======== Feature Engineering ========


def factor_momentum(prices, lookback=126, gap=21):
    """Momentum: return over `lookback` days excluding most recent `gap` days (≈6–1)."""
    momentum = prices.shift(gap) / prices.shift(lookback + gap) - 1.0
    return momentum.add_prefix("MOM_")


def factor_values_from_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Compute SIZE, VAL, QUAL from current metadata (NOT point-in-time; disabled below)."""
    out = pd.DataFrame(index=metadata.index)
    out["SIZE"] = np.log(metadata["market_cap"].replace({0: np.nan}))
    inv_pe = 1.0 / metadata["pe"].replace({0: np.nan})
    inv_pb = 1.0 / metadata["pb"].replace({0: np.nan})
    out["VAL"] = pd.concat([inv_pe, inv_pb], axis=1).mean(axis=1, skipna=True)
    out["QUAL"] = metadata["profit_margin"]
    return out


def factor_market(market_prices: pd.Series, window: int = 21) -> pd.Series:
    """Market factor: recent market return (e.g., ~1 month)."""
    return market_prices.pct_change(window).rename("MKT_RET")


@dataclass
class FeatureConfiguration:
    lookahead: int = 5
    momentum_lb: int = 126
    momentum_gap: int = 21
    market_window: int = 21


def build_feature_panel(
    prices: pd.DataFrame,
    market: pd.Series,
    metadata: Optional[Dict[str, pd.DataFrame]],  # PIT fundamentals dict
    macro: Optional[pd.DataFrame],
    configuration: FeatureConfiguration,
) -> pd.DataFrame:
    """Build per-ticker features with consistent column naming."""
    # Momentum
    ticker_momentum = factor_momentum(
        prices, configuration.momentum_lb, configuration.momentum_gap
    )

    # Market factor (broadcast)
    mkt = factor_market(market, configuration.market_window).to_frame()
    feature_panel = ticker_momentum.copy()

    for t in prices.columns:
        feature_panel[f"MKT_RET_{t}"] = mkt["MKT_RET"]

    # === PIT fundamentals (SIZE, VAL, PROFIT, LEV), if provided ===
    if metadata is not None and isinstance(metadata, dict):
        for fac_name, fac_panel in metadata.items():
            fac_aligned = fac_panel.reindex(prices.index).ffill()
            fac_z = cross_sectional_zscore(fac_aligned)
            for t in prices.columns:
                if t in fac_z.columns:
                    feature_panel[f"{fac_name}_{t}"] = fac_z[t]

    # Macro (if provided): align to price index & broadcast per ticker
    if macro is not None and not macro.empty:
        mac = macro.reindex(prices.index).ffill()
        for col in mac.columns:
            for t in prices.columns:
                feature_panel[f"MACRO_{col}_{t}"] = mac[col]

    feature_panel = feature_panel.reindex(prices.index).ffill()
    return feature_panel


# ======== Targets ========


def target_return(prices, horizon):
    """Forward return at date t for t to t+horizon."""
    return prices.pct_change(horizon).shift(-horizon)


# ======== Models ========


@dataclass
class ModelConfig:
    ridge_alpha: float = 1.0
    rf_estimators: int = 400
    rf_max_depth: Optional[int] = None
    min_train_points: int = 250  # ~1 trading year


class RegressorWrap:
    def __init__(self, name, estimator):
        self.name = name
        self.pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("model", estimator),
            ]
        )
        self.resid_std_ = {}

    def fit(self, X, y, ticker):
        self.pipe.fit(X, y)
        predictions = self.pipe.predict(X)
        residuals = y - predictions
        sigma = float(np.nanstd(residuals))
        if not math.isfinite(sigma) or sigma == 0:
            sigma = 1e-6
        self.resid_std_[ticker] = sigma

    def predict(self, X, ticker):
        predictions = self.pipe.predict(X)
        sigma = self.resid_std_.get(ticker, float(np.nanstd(predictions)) or 1e-6)
        return predictions, sigma


class RidgeModel(RegressorWrap):
    def __init__(self, alpha):
        super().__init__("Ridge", Ridge(alpha=alpha))


class RFModel(RegressorWrap):
    def __init__(self, n_estimators, max_depth):
        super().__init__(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            ),
        )


# ======== Walk-forward prediction ========


def walk_forward_predict(X, y, model, min_train, horizon):
    ## <-- added to enforce train_end = t_idx - horizon
    """Expanding window: train up to t-horizon, predict at t."""
    predictions = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)

    for t_idx in range(min_train, len(X) - 1):  ##careful
        train_end = t_idx - horizon
        if train_end < min_train:
            continue

        train_idx = X.index[:train_end]
        test_idx = X.index[t_idx : t_idx + 1]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]

        for ticker in y.columns:
            cols = [c for c in X.columns if c.endswith(f"_{ticker}")]
            if not cols:
                continue
            Xtr, Xte = X_train[cols], X_test[cols]
            ytr = y.loc[train_idx, ticker]
            if ytr.dropna().shape[0] < max(25, int(0.25 * len(train_idx))):
                continue
            try:
                model.fit(Xtr, ytr, ticker)
                pred, _ = model.predict(Xte, ticker)
                predictions.loc[test_idx, ticker] = float(pred[0])
            except Exception:
                continue

    return predictions


# ======== Signals ========


def positive_return_probability(expected, realised_sigma):
    P = expected.copy() * np.nan
    for t in expected.columns:
        sigma = realised_sigma.get(t, float(expected[t].std()) or 1e-6)
        P[t] = 1.0 - norm.cdf(0, loc=expected[t], scale=sigma)
    return P


def latest_signals(expected_returns, positive_return_prob):
    if expected_returns.dropna(how="all").empty:
        return pd.DataFrame(
            columns=["expected_return", "prob_up", "rank_by_expected", "rank_by_prob"]
        )
    last_idx = expected_returns.dropna(how="all").index.max()
    e = expected_returns.loc[last_idx]
    p = positive_return_prob.loc[last_idx]
    df = (
        pd.DataFrame({"expected_return": e, "prob_up": p})
        .dropna()
        .sort_values("expected_return", ascending=False)
    )
    df["rank_by_expected"] = (
        df["expected_return"].rank(ascending=False, method="first").astype(int)
    )
    df["rank_by_prob"] = df["prob_up"].rank(ascending=False, method="first").astype(int)
    return df


# ======== Backtest and reporting ========


def cs_z(s):
    mean, standard_deviation = s.mean(), s.std()
    if not math.isfinite(standard_deviation) or standard_deviation == 0:
        return s * 0
    return (s - mean) / standard_deviation


def build_weights_timeseries(expected, method="zscore", gross=1.0, topk=None):
    Weights = pd.DataFrame(index=expected.index, columns=expected.columns, dtype=float)
    for date, row in expected.iterrows():
        r = row.dropna()
        if r.empty:
            continue
        if topk and topk > 0:
            longs = r.nlargest(topk).index
            shorts = r.nsmallest(topk).index
            weights = pd.Series(0.0, index=r.index)
            if len(longs) > 0:
                weights.loc[longs] = (gross / 2.0) / max(len(longs), 1)
            if len(shorts) > 0:
                weights.loc[shorts] = -(gross / 2.0) / max(len(shorts), 1)
        else:
            z = cs_z(r)
            z = z - z.mean()
            s = z.abs().sum()
            weights = (z / (s if s != 0 else 1.0)) * gross
        Weights.loc[date, r.index] = weights
    return Weights


def backtest_portfolio_holding_h(weights, prices, h=5, tc_bps=0.0):
    """
    h-day holding backtest using staggered books:
    - Effective weights = rolling mean of the last h target books.
    - PnL uses 1-day realised returns.
    """
    # 1-day realised returns
    y1 = prices.pct_change(1).shift(-1)

    # Align and fill
    idx = weights.index.intersection(y1.index)
    W = weights.reindex(idx).fillna(0)
    Y = y1.reindex(idx).fillna(0)

    # Effective weights: average of the last h target books
    W_eff = W.rolling(h, min_periods=1).mean()

    # Daily portfolio return (gross)
    port_ret = (W_eff * Y).sum(axis=1)

    # Turnover/costs
    W_eff_prev = W_eff.shift(1).fillna(0)
    turnover = 0.5 * (W_eff - W_eff_prev).abs().sum(axis=1)
    costs = turnover * (tc_bps / 1e4)

    # Net return
    net_ret = port_ret - costs

    df = pd.DataFrame(
        {
            "gross_ret": port_ret,
            "turnover": turnover,
            "costs": costs,
            "net_ret": net_ret,
        }
    )
    df["cum_gross"] = (1 + df["gross_ret"]).cumprod()
    df["cum_net"] = (1 + df["net_ret"]).cumprod()

    # Annualised stats
    periods_per_year = 252
    annual_return = df["net_ret"].mean() * periods_per_year
    volatility = df["net_ret"].std(ddof=0) * math.sqrt(periods_per_year)
    sharpe = annual_return / volatility if volatility > 0 else np.nan
    drawdown = (df["cum_net"].cummax() - df["cum_net"]).div(
        df["cum_net"].cummax().replace(0, np.nan)
    )
    max_drawdown = drawdown.max()

    df.attrs["summary"] = {
        "periods_per_year": periods_per_year,
        "ann_return": float(annual_return),
        "ann_vol": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
    }
    return df


def print_performance_summary(backtest_data):
    s = backtest_data.attrs.get("summary", {})
    if not s:
        return
    print("\n=== Backtest Performance (net) ===")
    print(f"Annualised Return: {s['ann_return']:.4f}")
    print(f"Annualised Vol   : {s['ann_vol']:.4f}")
    print(f"Sharpe Ratio     : {s['sharpe']:.3f}")
    print(f"Max Drawdown     : {s['max_drawdown']:.3f}")


# ======== Main ========


def main(
    tickers,
    start,
    end,
    horizon,
    macro_csv,
    ridge_alpha,
    rf_estimators,
    rf_max_depth,
    gross,
    topk,
    tc_bps,
    make_plot,
    min_train,
    quiet,
    use_wrds: bool,
):
    setup_quiet_logging(quiet)

    prices = load_prices(tickers, start, end)
    market = load_market(start, end)

    fundamentals = None
    macro = None
    if use_wrds:
        db = wrds_connect()
        fundamentals = load_fundamentals_from_wrds(db, tickers, start, end)

    # Align indices
    idx = prices.index.intersection(market.index)
    prices = prices.reindex(idx).ffill()
    market = market.reindex(idx).ffill()

    # Features
    fcfg = FeatureConfiguration(lookahead=horizon)
    X = build_feature_panel(prices, market, fundamentals, macro, fcfg)

    # Targets (h-day forward returns)
    y = target_return(prices, horizon)

    # Models
    model_cfg = ModelConfig(
        ridge_alpha=ridge_alpha,
        rf_estimators=rf_estimators,
        rf_max_depth=rf_max_depth,
        min_train_points=min_train,
    )
    ridge = RidgeModel(alpha=model_cfg.ridge_alpha)
    rf = RFModel(n_estimators=model_cfg.rf_estimators, max_depth=model_cfg.rf_max_depth)

    # Predictions (walk-forward) -- **FIX** pass horizon to avoid label leak
    preds_ridge = walk_forward_predict(
        X, y, ridge, min_train=model_cfg.min_train_points, horizon=horizon
    )
    preds_rf = walk_forward_predict(
        X, y, rf, min_train=model_cfg.min_train_points, horizon=horizon
    )

    # Ensemble
    expected = (preds_ridge + preds_rf) / 2.0

    # Residual sigma per ticker (for probability calc)
    resid_sigma: Dict[str, float] = {}
    for t in prices.columns:
        vals = [
            v
            for v in (ridge.resid_std_.get(t), rf.resid_std_.get(t))
            if v is not None and math.isfinite(v)
        ]
        s = float(np.nanmean(vals)) if len(vals) else float(y[t].std())
        if not math.isfinite(s) or s == 0:
            s = 1e-6
        resid_sigma[t] = s

    probs = positive_return_probability(expected, resid_sigma)
    sigs = latest_signals(expected, probs)

    print(f"\n=== Latest Signals (horizon = {horizon} days) ===")
    if not sigs.empty:
        print(sigs.to_string(float_format=lambda x: f"{x:,.4f}"))
    else:
        print("No signals available (not enough training data in chosen window).")

    # Backtest (staggered holdings with effective-weight costs)
    W = build_weights_timeseries(expected, method="zscore", gross=gross, topk=topk)
    bt = backtest_portfolio_holding_h(W, prices, h=horizon, tc_bps=tc_bps)
    print_performance_summary(bt)

    # Save outputs
    sigs.to_csv("signals_latest.csv")
    expected.to_csv("expected_returns_timeseries.csv")
    probs.to_csv("probabilities_timeseries.csv")
    W.to_csv("weights_timeseries.csv")
    bt.to_csv("backtest_pnl.csv")

    # Always save equity curve PNG + series CSV
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = bt["cum_net"].plot(title="Equity Curve (Net)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Value")
        fig.tight_layout()
        fig.savefig("equity_curve.png", dpi=150)
        plt.close(fig)
        bt[["cum_net"]].to_csv("equity_curve_series.csv")
        print("Saved equity_curve.png and equity_curve_series.csv")
        if make_plot:
            import matplotlib.pyplot as plt  # fresh figure manager

            plt.figure()
            bt["cum_net"].plot(title="Equity Curve (Net)")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Value")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(
        "\nSaved: signals_latest.csv, expected_returns_timeseries.csv, probabilities_timeseries.csv, weights_timeseries.csv, backtest_pnl.csv, equity_curve_series.csv, equity_curve.png"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers",
    )
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default="2025-08-27")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--macro_csv", type=str, default=None)
    p.add_argument(
        "--wrds",
        action="store_true",
        help="Pull fundamentals and macro directly from WRDS (Compustat/FRED).",
    )

    # Model params (+ min_train via ModelConfig)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--rf_estimators", type=int, default=400)
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument(
        "--min_train",
        type=int,
        default=250,
        help="Minimum training points before predictions",
    )
    # Portfolio / costs
    p.add_argument(
        "--gross", type=float, default=1.0, help="Gross exposure target (sum |w|)"
    )
    p.add_argument(
        "--topk",
        type=int,
        default=None,
        help="If set, equal-weight top-k longs & bottom-k shorts",
    )
    p.add_argument(
        "--tc_bps",
        type=float,
        default=5.0,
        help="Transaction cost per unit turnover in bps",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Show an interactive equity curve window (PNG always saved)",
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress non-essential warnings/logging"
    )

    args = p.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    main(
        tickers=tickers,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        macro_csv=args.macro_csv,
        ridge_alpha=args.ridge_alpha,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
        gross=args.gross,
        topk=args.topk,
        tc_bps=args.tc_bps,
        make_plot=bool(args.plot),
        min_train=args.min_train,
        quiet=bool(args.quiet),
        use_wrds=bool(args.wrds),
    )
