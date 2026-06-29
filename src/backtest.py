from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

project_root = project_root = Path(__file__).resolve().parent.parent
RAW_DIR = project_root / "data" / "raw"
PROCESSED_DIR = project_root / "data" / "processed"
REPORT_DIR = project_root / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


import pandas as pd


def ma_cross_backtest(
    df: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 60,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    ) -> pd.DataFrame:
    """
    双均线择时策略回测。

    策略规则：
    1. fast_ma > slow_ma 时，目标仓位为 1；
    2. fast_ma <= slow_ma 时，目标仓位为 0；
    3. 当天收盘后产生信号，下一交易日使用该仓位；
    4. 买入和卖出都扣除手续费与滑点。

    Parameters
    ----------
    df:
        单只股票日线数据，至少包含 date、close 两列。
    fast_window:
        短期均线窗口，默认 20 个交易日。
    slow_window:
        长期均线窗口，默认 60 个交易日。
    commission_rate:
        单边手续费率，默认 0.03%。
    slippage_rate:
        单边滑点率，默认 0.02%。

    Returns
    -------
    pd.DataFrame
        包含均线、信号、仓位、收益、净值和回撤的完整回测表。
    """

    # ---------- 1. 检查参数 ----------

    if not isinstance(fast_window, int) or not isinstance(slow_window, int):
        raise TypeError("均线窗口必须是整数")

    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("均线窗口必须大于 0")

    if fast_window >= slow_window:
        raise ValueError("fast_window 必须小于 slow_window")

    if commission_rate < 0 or slippage_rate < 0:
        raise ValueError("手续费率和滑点率不能为负数")

    # ---------- 2. 检查必要字段 ----------

    required_cols = {"date", "close"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"缺少必要字段：{sorted(missing_cols)}"
        )

    # ---------- 3. 复制和整理数据 ----------

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data["close"] = pd.to_numeric(
        data["close"],
        errors="coerce",
    )

    if data["date"].isna().any():
        raise ValueError("date 列存在无法转换的日期")

    if data["close"].isna().any():
        raise ValueError("close 列存在缺失值或非数字数据")

    if (data["close"] <= 0).any():
        raise ValueError("close 列必须全部大于 0")

    data = (
        data
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    # ---------- 4. 计算股票日收益率 ----------

    data["asset_return"] = data["close"].pct_change()

    # ---------- 5. 计算快慢均线 ----------

    data["fast_ma"] = (
        data["close"]
        .rolling(
            window=fast_window,
            min_periods=fast_window,
        )
        .mean()
    )

    data["slow_ma"] = (
        data["close"]
        .rolling(
            window=slow_window,
            min_periods=slow_window,
        )
        .mean()
    )

    # ---------- 6. 生成当日收盘后的信号 ----------

    data["signal"] = (
        data["fast_ma"] > data["slow_ma"]
    ).astype(float)

    # 60 日均线尚未形成时保持空仓
    data.loc[
        data["slow_ma"].isna(),
        "signal",
    ] = 0.0

    # ---------- 7. 信号延迟一天形成实际仓位 ----------

    data["position"] = (
        data["signal"]
        .shift(1)
        .fillna(0.0)
    )

    # ---------- 8. 计算策略仓位变化 ----------

    data["position_change"] = (
        data["position"]
        .diff()
        .fillna(data["position"])
    )

    # ---------- 9. 计算扣费前策略收益 ----------

    data["gross_strategy_return"] = (
        data["position"]
        * data["asset_return"].fillna(0.0)
    )

    # ---------- 10. 计算交易成本 ----------

    one_way_cost = (
        commission_rate + slippage_rate
    )

    data["transaction_cost"] = (
        data["position_change"].abs()
        * one_way_cost
    )

    # ---------- 11. 计算扣费后策略收益 ----------

    data["strategy_return"] = (
        data["gross_strategy_return"]
        - data["transaction_cost"]
    )

    # ---------- 12. 计算策略和基准净值 ----------

    data["strategy_nav"] = (
        1 + data["strategy_return"]
    ).cumprod()

    data["benchmark_nav"] = (
        1 + data["asset_return"].fillna(0.0)
    ).cumprod()

    # ---------- 13. 计算策略回撤 ----------

    data["strategy_rolling_max"] = (
        data["strategy_nav"].cummax()
    )

    data["strategy_drawdown"] = (
        data["strategy_nav"]
        / data["strategy_rolling_max"]
        - 1
    )
    
    return data


def check_if_valid(result: pd.DataFrame):
    # 1. 前 19 条记录不应该有 20 日均线
    assert result["fast_ma"].iloc[:19].isna().all()
    # 第 20 条记录应该有 20 日均线
    assert pd.notna(result["fast_ma"].iloc[19])
    # 2. 前 59 条记录不应该有 60 日均线
    assert result["slow_ma"].iloc[:59].isna().all()
    # 第 60 条记录应该有 60 日均线
    assert pd.notna(result["slow_ma"].iloc[59])
    # 3. 60 日均线形成以前，信号必须为 0
    assert (
        result.loc[
            result["slow_ma"].isna(),
            "signal",
        ] == 0
    ).all()
    # 4. position 必须等于 signal 延迟一天
    expected_position = (
        result["signal"]
        .shift(1)
        .fillna(0.0)
    )
    assert result["position"].equals(
        expected_position
    )
    # 5. 没有仓位变化时，交易成本必须为 0
    assert (
        result.loc[
            result["position_change"] == 0,
            "transaction_cost",
        ] == 0
    ).all()
    # 6. 发生满仓买入或卖出时，默认成本为 0.0005
    expected_cost = 0.0003 + 0.0002
    assert (
        result.loc[
            result["position_change"].abs() == 1,
            "transaction_cost",
        ] == expected_cost
    ).all()
    # 7. 净值必须保持为正数
    assert (result["strategy_nav"] > 0).all()
    assert (result["benchmark_nav"] > 0).all()
    # 8. 回撤不应该大于 0
    assert (result["strategy_drawdown"] <= 1e-12).all()
    print("全部基础检查通过")


def backtest_plot_sbpoints(result: pd.DataFrame):
    
    if result.empty:
        raise ValueError("result 不能为空")
    
    required_cols = {
        "symbol",
        "date",
        "close",
        "fast_ma",
        "slow_ma",
        "position",
    }
    missing_cols = required_cols - set(result.columns)
    if missing_cols:
        raise ValueError(f"缺少必要字段：{sorted(missing_cols)}")
    
    symbol = result["symbol"].iloc[0]
    buy_points = result.loc[
        result["position_change"] > 0
    ].copy()

    sell_points = result.loc[
        result["position_change"] < 0
    ].copy()
    
    plt.figure(figsize=(14, 7))

    plt.plot(
        result["date"],
        result["close"],
        label="Close",
        linewidth=1,
        alpha=0.7,
    )

    plt.plot(
        result["date"],
        result["fast_ma"],
        label="MA20",
        linewidth=1.2,
    )

    plt.plot(
        result["date"],
        result["slow_ma"],
        label="MA60",
        linewidth=1.2,
    )

    plt.scatter(
        buy_points["date"],
        buy_points["close"],
        marker="$B$",
        s=80,
        label="Buy",
        zorder=3,
    )

    plt.scatter(
        sell_points["date"],
        sell_points["close"],
        marker="$S$",
        s=80,
        label="Sell",
        zorder=3,
    )

    plt.title(f"{symbol} 20/60 MA Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def backtest_plot_nav(result: pd.DataFrame, strategy_nav: str = "strategy_nav", benchmark_nav: str  = "benchmark_nav"):
    if result.empty:
        raise ValueError("result 不能为空")
    
    required_cols = {
        "symbol",
        "date",
        "close",
        "fast_ma",
        "slow_ma",
        "position",
    }
    missing_cols = required_cols - set(result.columns)
    if missing_cols:
        raise ValueError(f"缺少必要字段：{sorted(missing_cols)}")
    
    symbol = result["symbol"].iloc[0]
    plt.figure(figsize=(14, 6))

    plt.plot(
        result["date"],
        result[strategy_nav],
        label="MA Strategy",
        linewidth=1.5,
    )

    plt.plot(
        result["date"],
        result[benchmark_nav],
        label="Buy and Hold",
        linewidth=1.5,
    )

    plt.axhline(
        y=1,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    plt.title(f"{symbol} Strategy vs Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Net Asset Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def backtest_plot_drawdown(result: pd.DataFrame, strategy_drawdown: str = "strategy_drawdown"):
    if result.empty:
        raise ValueError("result 不能为空")
    
    required_cols = {
        "symbol",
        "date",
        "close",
        "fast_ma",
        "slow_ma",
        "position",
    }
    missing_cols = required_cols - set(result.columns)
    if missing_cols:
        raise ValueError(f"缺少必要字段：{sorted(missing_cols)}")
    
    symbol = result["symbol"].iloc[0]
    plt.figure(figsize=(14, 5))

    plt.plot(
        result["date"],
        result[strategy_drawdown] * 100,
        linewidth=1.2,
    )

    plt.fill_between(
        result["date"],
        result[strategy_drawdown] * 100,
        0,
        alpha=0.3,
    )

    plt.axhline(
        y=0,
        linewidth=1,
    )

    plt.title(f"{symbol} Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_performance(
    returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, float]:
    """
    根据日收益率序列计算绩效指标。

    Parameters
    ----------
    returns:
        日收益率序列。例如 strategy_return 或 asset_return。
    annual_risk_free_rate:
        年化无风险利率，默认使用 0。
    trading_days:
        每年的交易日数量，默认使用 252。

    Returns
    -------
    dict
        包含累计收益、年化收益、年化波动率、
        夏普比率、最大回撤和卡玛比率。
    """

    if trading_days <= 0:
        raise ValueError("trading_days 必须大于 0")

    if annual_risk_free_rate <= -1:
        raise ValueError("annual_risk_free_rate 必须大于 -1")

    # 转换为数值序列，并清理异常值
    clean_returns = (
        pd.to_numeric(
            pd.Series(returns),
            errors="coerce",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    if clean_returns.empty:
        raise ValueError("收益率序列为空")

    # 单日收益小于等于 -100% 时，净值会归零或变成负数
    if (clean_returns <= -1).any():
        raise ValueError(
            "收益率序列中存在小于等于 -100% 的数据"
        )

    # 1. 净值序列
    nav = (1 + clean_returns).cumprod()

    # 2. 累计收益率
    cumulative_return = nav.iloc[-1] - 1

    # 3. 年化收益率
    periods = len(clean_returns)

    annual_return = (
        nav.iloc[-1] ** (trading_days / periods)
        - 1
    )

    # 4. 年化波动率
    daily_volatility = clean_returns.std(ddof=1)

    annual_volatility = (
        daily_volatility * np.sqrt(trading_days)
    )

    # 5. 夏普比率
    daily_risk_free_rate = (
        (1 + annual_risk_free_rate)
        ** (1 / trading_days)
        - 1
    )

    excess_returns = (
        clean_returns - daily_risk_free_rate
    )

    if pd.isna(daily_volatility) or np.isclose(
        daily_volatility,
        0,
    ):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (
            excess_returns.mean()
            / daily_volatility
            * np.sqrt(trading_days)
        )

    # 6. 回撤序列和最大回撤
    rolling_max = nav.cummax()

    drawdown = (
        nav / rolling_max - 1
    )

    max_drawdown = drawdown.min()

    # 7. 卡玛比率
    if np.isclose(max_drawdown, 0):
        calmar_ratio = np.nan
    else:
        calmar_ratio = (
            annual_return / abs(max_drawdown)
        )
     

    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }



def calculate_buy_and_sell(result: pd.DataFrame) -> dict[str, float]:
    buy_count = (result["position_change"] == 1.0).sum()
    sell_count = (result["position_change"] == -1.0).sum()
    total_trade_count = buy_count + sell_count
    exposure = result["position"].mean()
    total_transaction_cost = result["transaction_cost"].sum()
    return{
        "评价交易日数量": result.shape[0],
        "持仓比例": exposure,
        "买入次数": buy_count,
        "卖出次数": sell_count,
        "总交易次数": total_trade_count,
        "交易成本简单加总": total_transaction_cost
    }



if __name__ == "__main__":
    symbol = "000001"
    df = pd.read_csv(f"{RAW_DIR}\{symbol}_qfq_daily.csv")
    result = ma_cross_backtest(df)
    check_if_valid(result)
    
    print(result[
    [
        "date",
        "close",
        "fast_ma",
        "slow_ma",
        "signal",
        "position",
        "turnover",
        "asset_return",
        "strategy_return"
    ]
    ].tail(30))