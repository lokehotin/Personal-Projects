"""
Simple Trading Backtester
================================

USAGE (examples)
----------------

# Basic run with 2 tickers and default settings
python backtester.py --tickers AAPL,MSFT \
    --start 2020-01-01 --end 2021-12-31 \
    --fast_ma 10 --slow_ma 30 \
    --out_dir results/aapl_msft

# Run with stop-loss/take-profit (2% SL, 4% TP)
python backtester.py --tickers TSLA,GOOGL \
    --start 2019-01-01 --end 2020-12-31 \
    --fast_ma 5 --slow_ma 20 \
    --sl 0.02 --tp 0.04 \
    --out_dir results/sl_tp

# Run parameter sweep across moving average windows
python backtester.py --tickers NVDA,AAPL,MSFT \
    --start 2020-01-01 --end 2020-12-31 \
    --sweep_fast 5 10 20 \
    --sweep_slow 30 60 120 \
    --out_dir results/sweep

# Run bootstrap Sharpe ratio test with 1000 resamples
python backtester.py --tickers AAPL,MSFT \
    --start 2021-01-01 --end 2022-12-31 \
    --fast_ma 10 --slow_ma 50 \
    --bootstrap 1000 \
    --out_dir results/bootstrap

# Quiet mode for debugging (suppress logs)
python backtester.py --tickers AAPL,MSFT \
    --start 2022-01-01 --end 2022-06-30 \
    --fast_ma 5 --slow_ma 15 \
    --quiet

Notes
-----
* '--tickers' sets the list of assets to backtest.
* '--start', '--end' define the backtest date range.
* '--fast_ma', '--slow_ma' are moving average window lengths.
* '--sl', '--tp' enable stop-loss / take-profit as decimal returns.
* '--sweep_fast', '--sweep_slow' run grid search over MA windows.
* '--bootstrap' runs bootstrap sampling of Sharpe ratios.
* '--quiet' suppresses console output.
* '--out_dir' specifies where to save results.

- "sl" and "tp" are *percent* thresholds applied to each individual trade
  (e.g., --sl 0.08 means stop out at -8% from entry; --tp 0.15 means take
  profit at +15%).

Outputs include:
* trades.csv            – trade entries and exits
* trade_stats.csv       – per-trade statistics
* pnl_timeseries.csv    – daily equity curve
* equity_curve.png      – plot of portfolio equity
* heatmap.png           – parameter sweep heatmap
* bootstrap_sharpe.csv  – bootstrap Sharpe results


- "sl" and "tp" are *percent* thresholds applied to each individual trade
  (e.g., --sl 0.08 means stop out at -8% from entry; --tp 0.15 means take
  profit at +15%).
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

TRADING_DAYS = 252


@dataclass
class Metrics:
    cagr: float
    sharpe: float
    max_dd: float
    hit_rate: float
    vol_annual: float
    total_return: float
    profit_factor: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    win_rate: Optional[float] = None
    n_trades: Optional[int] = None


# ======== metrics: annualised return, annualised volatility, sharpe ratio, maximum drawdown, hit rate ========


def annualised_return(equity):
    start_val, end_val = equity.iloc[0], equity.iloc[-1]
    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days <= 0 or start_val <= 0:
        return np.nan
    years = n_days / 365.25
    return (end_val / start_val) ** (1 / years) - 1


def annualised_volatility(returns):
    return returns.std() * math.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, risk_free_rate=0.0):
    if returns.std() == 0:
        return np.nan
    return (returns.mean() * TRADING_DAYS - risk_free_rate) / annualised_volatility(
        returns
    )


def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return drawdown.min()


def daily_hit_rate(returns):
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return np.nan
    return (nonzero > 0).mean()


# ======== data loading ========


def load_data(tickers, start=None, end=None):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        close_price = df["Close"].copy()
        volume = df["Volume"].copy()
        close_price.columns = [str(c) for c in close_price.columns]
        volume.columns = [str(c) for c in volume.columns]
    else:
        close_price = df["Close"].to_frame(tickers[0])
        volume = df["Volume"].to_frame(tickers[0])
    close_price = close_price.ffill().dropna(how="all")
    volume = volume.reindex_like(close_price).fillna(0)
    return close_price, volume


# ======== strategies: simple moving average, mean reversion ========


def sma_crossover_positions(prices, fast=20, slow=50):
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    positions = (sma_fast > sma_slow).astype(int)
    positions.iloc[:slow] = 0
    return positions


def mean_reversion_positions(prices, lookback=20, x_sigma=1.0):
    rolling_moving_average = prices.rolling(lookback).mean()
    rolling_volatility = prices.rolling(lookback).std()
    z_score = (prices - rolling_moving_average) / rolling_volatility
    long_signal = (z_score < -x_sigma).astype(int)
    flat_signal = (z_score >= 0).astype(int)

    positions = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)
    for col in prices.columns:
        s_long = long_signal[col]
        s_flat = flat_signal[col]
        position = 0
        for i, idx in enumerate(prices.index):
            if i < lookback:
                positions.at[idx, col] = 0
                continue
            if s_long.iloc[i] == 1:
                position = 1
            elif s_flat.iloc[i] == 1:
                position = 0
            positions.at[idx, col] = position
    return positions


# ======== stop-loss / take-profit function ========


def apply_sl_tp(prices, positions, sl=None, tp=None):
    """Apply per-trade stop-loss/take-profit on *long-only* positions*, without zeroing the future.
    - sl / tp are fractions, e.g. 0.08 for 8% stop, 0.15 for 15% take-profit.
    - Allows re-entry on future 0->1 signals after an SL/TP or base exit.
    """
    if sl is None and tp is None:
        return positions

    base = positions.fillna(0).astype(int)
    output_positions = pd.DataFrame(
        0, index=base.index, columns=base.columns, dtype=int
    )

    for col in base.columns:
        in_trade = False
        entry_price: Optional[float] = None
        prev_signal = 0

        for dt in base.index:
            signal = int(base.at[dt, col])
            close_price = float(prices.at[dt, col])

            if not in_trade:
                if prev_signal == 0 and signal == 1:
                    in_trade = True
                    entry_price = close_price
                    output_positions.at[dt, col] = 1
                else:
                    output_positions.at[dt, col] = 0
            else:
                assert entry_price is not None
                return_since_entry = close_price / entry_price - 1.0
                hit_sl = (sl is not None) and (return_since_entry <= -sl)
                hit_tp = (tp is not None) and (return_since_entry >= tp)
                base_exit = signal == 0

                if hit_sl or hit_tp or base_exit:
                    in_trade = False
                    entry_price = None
                    output_positions.at[dt, col] = 0
                else:
                    output_positions.at[dt, col] = 1

            prev_signal = signal

    return output_positions


# ======== trade extraction ========


def extract_trades(prices, positions):
    """Turn a 0/1 positions DataFrame into a table of trades per ticker.
    Columns: ticker, entry_date, exit_date, entry_px, exit_px, ret, hold_days
    """
    rows = []
    for col in positions.columns:
        position = positions[col].fillna(0).astype(int)

        entry_dates = position.index[
            (position.shift(1, fill_value=0) == 0) & (position == 1)
        ]
        exit_dates = position.index[
            (position.shift(1, fill_value=0) == 1) & (position == 0)
        ]

        if len(exit_dates) < len(entry_dates):
            exit_dates = pd.Index([*exit_dates, position.index[-1]])
        for entry, exit in zip(entry_dates, exit_dates):
            entry_price = prices.at[entry, col]
            exit_price = prices.at[exit, col]
            return_percentage = (exit_price / entry_price) - 1.0
            hold = (exit - entry).days
            rows.append(
                {
                    "ticker": col,
                    "entry_date": entry,
                    "exit_date": exit,
                    "entry_px": float(entry_price),
                    "exit_px": float(exit_price),
                    "ret": float(return_percentage),
                    "hold_days": int(hold),
                }
            )
    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades.sort_values(["ticker", "entry_date"], inplace=True)
    return trades


def trade_stats(trades):
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "avg_hold_days": np.nan,
        }
    wins = trades.loc[trades.ret > 0, "ret"]
    losses = trades.loc[trades.ret <= 0, "ret"]
    total_profit = wins.sum()
    total_loss = -losses.sum()
    profit_factor = (total_profit / total_loss) if total_loss > 0 else np.inf
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((len(wins) / len(trades)) if len(trades) else np.nan),
        "avg_win": float(wins.mean() if len(wins) else np.nan),
        "avg_loss": float(losses.mean() if len(losses) else np.nan),
        "profit_factor": float(profit_factor),
        "avg_hold_days": float(trades.hold_days.mean()),
    }


# ======== backtest ========


def backtest(prices, positions, cost_bps=5.0):
    returns = prices.pct_change().fillna(0.0)
    positions = positions.reindex_like(prices).fillna(0).astype(float)
    turnover = positions.diff().abs().fillna(0.0)
    gross_asset_returns = positions.shift(1) * returns
    per_trade_cost = cost_bps / 1e4
    asset_transaction_costs = turnover * per_trade_cost
    net_asset_returns = gross_asset_returns - asset_transaction_costs

    if net_asset_returns.shape[1] == 1:
        portforlio_returns = net_asset_returns.iloc[:, 0]
        benchmark_returns = returns.iloc[:, 0]
    else:
        portforlio_returns = net_asset_returns.mean(axis=1)
        benchmark_returns = returns.mean(axis=1)

    equity = (1.0 + portforlio_returns).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()

    # Trade-level outputs & stats
    trades = extract_trades(prices, positions)
    trade_statistics = trade_stats(trades)

    metrics = Metrics(
        cagr=annualised_return(equity),
        sharpe=sharpe_ratio(portforlio_returns),
        max_dd=max_drawdown(equity),
        hit_rate=daily_hit_rate(portforlio_returns),
        vol_annual=annualised_volatility(portforlio_returns),
        total_return=equity.iloc[-1] - 1.0,
        profit_factor=trade_statistics["profit_factor"],
        avg_win=trade_statistics["avg_win"],
        avg_loss=trade_statistics["avg_loss"],
        win_rate=trade_statistics["win_rate"],
        n_trades=trade_statistics["n_trades"],
    )
    return equity, benchmark_equity, portforlio_returns, metrics, trades


# ======== plotting ========


def plot_equity(equity, benchmark_equity, outpath=None, title=""):
    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values, label="Strategy")
    plt.plot(benchmark_equity.index, benchmark_equity.values, label="Buy & Hold")
    plt.legend()
    plt.title(title or "Equity Curve (vs Buy & Hold)")
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_drawdown(equity, outpath: Optional[str] = None):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    plt.figure(figsize=(10, 2.5))
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.ylabel("DD")
    plt.xlabel("Date")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_sma_heatmap(
    prices,
    fast_list,
    slow_list,
    out_png,
    cost_bps=5.0,
    sl=None,
    tp=None,
):
    fast_list = sorted(fast_list)
    slow_list = sorted(slow_list)
    heat = np.full((len(fast_list), len(slow_list)), np.nan)
    for i, f in enumerate(fast_list):
        for j, s in enumerate(slow_list):
            if f >= s:  # only sensible if fast < slow
                continue
            pos = sma_crossover_positions(prices, fast=f, slow=s)
            if sl is not None or tp is not None:
                pos = apply_sl_tp(prices, pos, sl=sl, tp=tp)
            equity, benchmark_equity, portfolio_returns, m, trades = backtest(
                prices, pos, cost_bps=cost_bps
            )
            heat[i, j] = m.sharpe
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(slow_list)))
    ax.set_xticklabels(slow_list)
    ax.set_yticks(range(len(fast_list)))
    ax.set_yticklabels(fast_list)
    ax.set_xlabel("Slow")
    ax.set_ylabel("Fast")
    ax.set_title("Sharpe Heatmap (SMA)")
    fig.colorbar(im, ax=ax, label="Sharpe")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ======== bootstrap sampling ========


def bootstrap_sharpe(returns, n_boot=1000, seed: Optional[int] = 42):
    rng = np.random.default_rng(seed)
    rets = returns.dropna().values
    if len(rets) == 0:
        return {"median": np.nan, "p05": np.nan, "p95": np.nan}
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(rets, size=len(rets), replace=True)
        s = (
            (sample.mean() * TRADING_DAYS)
            / (sample.std(ddof=1) * math.sqrt(TRADING_DAYS))
            if sample.std(ddof=1) != 0
            else np.nan
        )
        sharpes[b] = s
    sharpes = sharpes[~np.isnan(sharpes)]
    if len(sharpes) == 0:
        return {"median": np.nan, "p05": np.nan, "p95": np.nan}
    return {
        "median": float(np.median(sharpes)),
        "p05": float(np.percentile(sharpes, 5)),
        "p95": float(np.percentile(sharpes, 95)),
    }


# ======== exporting ========


def export_results(
    output_directory,
    equity,
    benchmark_equity,
    portfolio_returns,
    trades,
    metrics,
    extra=None,
):
    os.makedirs(output_directory, exist_ok=True)

    df_equity = pd.DataFrame(
        {
            "equity": equity,
            "benchmark_equity": benchmark_equity,
            "portfolio_returns": portfolio_returns,
        }
    )
    df_equity.to_csv(os.path.join(output_directory, "equity_and_returns.csv"))
    trades.to_csv(os.path.join(output_directory, "trades.csv"), index=False)

    m_dict = metrics.__dict__.copy()
    if extra:
        m_dict.update({f"bootstrap_{k}": v for k, v in extra.items()})
    with open(os.path.join(output_directory, "metrics.json"), "w") as f:
        json.dump(m_dict, f, indent=2)


# ======== arguments ========


def parse_args():
    p = argparse.ArgumentParser(description="Backtester")
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"], help="1–3 tickers")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--strategy", choices=["sma", "meanrev"], default="sma")

    # SMA params
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)

    # Mean reversion params
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--x_sigma", type=float, default=1.0)

    # Risk controls
    p.add_argument(
        "--sl",
        type=float,
        default=None,
        help="Stop-loss as fraction, e.g., 0.08 for 8%",
    )
    p.add_argument(
        "--tp",
        type=float,
        default=None,
        help="Take-profit as fraction, e.g., 0.15 for 15%",
    )

    # Costs
    p.add_argument("--cost_bps", type=float, default=5.0)

    # Sweep
    p.add_argument(
        "--sweep_fast",
        nargs="*",
        type=int,
        default=None,
        help="List of fast windows for SMA sweep",
    )
    p.add_argument(
        "--sweep_slow",
        nargs="*",
        type=int,
        default=None,
        help="List of slow windows for SMA sweep",
    )

    # Bootstrap
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="number of bootstrap samples for Sharpe (0=off)",
    )

    # Output
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument(
        "--plot", action="store_true", help="Show plots interactively instead of saving"
    )
    return p.parse_args()


# ======== main ========


def main():
    args = parse_args()
    tickers = args.tickers
    if len(tickers) < 1 or len(tickers) > 3:
        raise ValueError("Please provide between 1 and 3 tickers.")

    prices, volume = load_data(tickers, start=args.start, end=args.end)

    # Strategy positions
    if args.strategy == "sma":
        positions = sma_crossover_positions(prices, fast=args.fast, slow=args.slow)
        title = f"SMA {args.fast}/{args.slow} on {','.join(tickers)}"
    else:
        positions = mean_reversion_positions(
            prices, lookback=args.lookback, x_sigma=args.x_sigma
        )
        title = f"MeanRev Lb={args.lookback} Xσ={args.x_sigma} on {','.join(tickers)}"

    # Stop-loss / Take-profit
    positions = apply_sl_tp(prices, positions, sl=args.sl, tp=args.tp)

    # Backtest
    equity, benchmark_equity, portfolio_returns, metrics, trades = backtest(
        prices, positions, cost_bps=args.cost_bps
    )

    # Bootstrap
    boot = {}
    if args.bootstrap and args.bootstrap > 0:
        boot = bootstrap_sharpe(portfolio_returns, n_boot=args.bootstrap)

    # Export
    export_results(
        args.out_dir,
        equity,
        benchmark_equity,
        portfolio_returns,
        trades,
        metrics,
        extra=boot,
    )

    # Plots
    os.makedirs(args.out_dir, exist_ok=True)
    if args.plot:
        plot_equity(equity, benchmark_equity, title=title)
        plot_drawdown(equity)
    else:
        plot_equity(
            equity,
            benchmark_equity,
            outpath=os.path.join(args.out_dir, "equity.png"),
            title=title,
        )
        plot_drawdown(equity, outpath=os.path.join(args.out_dir, "drawdown.png"))

    # Parameter sweep (SMA only)
    if args.strategy == "sma" and args.sweep_fast and args.sweep_slow:
        plot_sma_heatmap(
            prices,
            args.sweep_fast,
            args.sweep_slow,
            out_png=os.path.join(args.out_dir, "sma_heatmap.png"),
            cost_bps=args.cost_bps,
            sl=args.sl,
            tp=args.tp,
        )
        # CSV of grid
        grid = []
        for f in sorted(set(args.sweep_fast)):
            for s in sorted(set(args.sweep_slow)):
                if f >= s:
                    continue
                pos = sma_crossover_positions(prices, fast=f, slow=s)
                if args.sl is not None or args.tp is not None:
                    pos = apply_sl_tp(prices, pos, sl=args.sl, tp=args.tp)
                equity_, bh_, port_, m_, trades_ = backtest(
                    prices, pos, cost_bps=args.cost_bps
                )
                grid.append(
                    {
                        "fast": f,
                        "slow": s,
                        "sharpe": m_.sharpe,
                        "cagr": m_.cagr,
                        "max_dd": m_.max_dd,
                    }
                )
        pd.DataFrame(grid).to_csv(
            os.path.join(args.out_dir, "sma_sweep.csv"), index=False
        )

    # Console summary
    print(title)
    print("-" * len(title))
    print(
        json.dumps(
            {
                k: (float(v) if v is not None and not isinstance(v, str) else v)
                for k, v in metrics.__dict__.items()
            },
            indent=2,
        )
    )
    if boot:
        print("Bootstrap Sharpe (median, p05, p95):", boot)
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
