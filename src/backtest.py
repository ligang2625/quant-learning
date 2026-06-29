from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

project_root = project_root = Path(__file__).resolve().parent.parent
RAW_DIR = project_root / "data" / "raw"
PROCESSED_DIR = project_root / "data" / "processed"
REPORT_DIR = project_root / "reports"
BACKTEST_REPORT_DIR = REPORT_DIR / "ma_20_60_backtest"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def summarize_backtest_result(
    result: pd.DataFrame,
    slow_window: int = 60,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict:
    """
    对单只股票回测结果生成一行汇总。
    """

    required_cols = {
        "date",
        "symbol",
        "slow_ma",
        "position",
        "position_change",
        "strategy_return",
        "asset_return",
        "transaction_cost",
    }

    missing_cols = required_cols - set(result.columns)

    if missing_cols:
        raise ValueError(f"缺少必要字段：{sorted(missing_cols)}")

    data = result.copy()

    data["date"] = pd.to_datetime(data["date"])

    symbol = (
        data["symbol"]
        .dropna()
        .astype(str)
        .str.zfill(6)
        .iloc[0]
    )

    evaluation_mask = (
        data["slow_ma"]
        .shift(1)
        .notna()
    )

    evaluation_data = (
        data.loc[evaluation_mask]
        .copy()
        .reset_index(drop=True)
    )

    if evaluation_data.empty:
        raise ValueError(
            f"{symbol} 评价区间为空，请检查数据长度或 slow_window={slow_window}"
        )

    strategy_metrics = calculate_performance(
        evaluation_data["strategy_return"],
        annual_risk_free_rate=annual_risk_free_rate,
        trading_days=trading_days,
    )

    benchmark_metrics = calculate_performance(
        evaluation_data["asset_return"],
        annual_risk_free_rate=annual_risk_free_rate,
        trading_days=trading_days,
    )

    buy_count = int(
        (evaluation_data["position_change"] > 0).sum()
    )

    sell_count = int(
        (evaluation_data["position_change"] < 0).sum()
    )

    exposure = float(
        evaluation_data["position"].mean()
    )

    total_transaction_cost = float(
        evaluation_data["transaction_cost"].sum()
    )

    return {
        "symbol": symbol,
        "start_date": evaluation_data["date"].iloc[0],
        "end_date": evaluation_data["date"].iloc[-1],
        "trade_days": len(evaluation_data),

        "strategy_cumulative_return": strategy_metrics["cumulative_return"],
        "strategy_annual_return": strategy_metrics["annual_return"],
        "strategy_annual_volatility": strategy_metrics["annual_volatility"],
        "strategy_sharpe": strategy_metrics["sharpe_ratio"],
        "strategy_max_drawdown": strategy_metrics["max_drawdown"],
        "strategy_calmar": strategy_metrics["calmar_ratio"],

        "benchmark_cumulative_return": benchmark_metrics["cumulative_return"],
        "benchmark_annual_return": benchmark_metrics["annual_return"],
        "benchmark_annual_volatility": benchmark_metrics["annual_volatility"],
        "benchmark_sharpe": benchmark_metrics["sharpe_ratio"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "benchmark_calmar": benchmark_metrics["calmar_ratio"],

        "excess_annual_return": (
            strategy_metrics["annual_return"]
            - benchmark_metrics["annual_return"]
        ),
        "sharpe_diff": (
            strategy_metrics["sharpe_ratio"]
            - benchmark_metrics["sharpe_ratio"]
        ),
        "drawdown_improvement": (
            strategy_metrics["max_drawdown"]
            - benchmark_metrics["max_drawdown"]
        ),

        "exposure": exposure,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total_trade_count": buy_count + sell_count,
        "total_transaction_cost": total_transaction_cost,
    }


def format_batch_summary(
    batch_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    生成中文展示版汇总表。
    """

    display_df = batch_summary.copy()

    percent_cols = [
        "strategy_cumulative_return",
        "benchmark_cumulative_return",
        "strategy_annual_return",
        "benchmark_annual_return",
        "excess_annual_return",
        "strategy_annual_volatility",
        "benchmark_annual_volatility",
        "strategy_max_drawdown",
        "benchmark_max_drawdown",
        "drawdown_improvement",
        "exposure",
        "total_transaction_cost",
    ]

    for col in percent_cols:
        display_df[col] = display_df[col].map(
            lambda x: f"{x:.2%}"
        )

    ratio_cols = [
        "strategy_sharpe",
        "benchmark_sharpe",
        "sharpe_diff",
        "strategy_calmar",
        "benchmark_calmar",
    ]

    for col in ratio_cols:
        display_df[col] = display_df[col].map(
            lambda x: "NaN" if pd.isna(x) else f"{x:.3f}"
        )

    display_df["start_date"] = pd.to_datetime(
        display_df["start_date"]
    ).dt.date

    display_df["end_date"] = pd.to_datetime(
        display_df["end_date"]
    ).dt.date

    display_df = display_df.rename(
        columns={
            "symbol": "股票代码",
            "start_date": "开始日期",
            "end_date": "结束日期",
            "trade_days": "交易日数量",

            "strategy_cumulative_return": "策略累计收益",
            "strategy_annual_return": "策略年化收益",
            "strategy_annual_volatility": "策略年化波动",
            "strategy_sharpe": "策略夏普",
            "strategy_max_drawdown": "策略最大回撤",
            "strategy_calmar": "策略卡玛",

            "benchmark_cumulative_return": "买入持有累计收益",
            "benchmark_annual_return": "买入持有年化收益",
            "benchmark_annual_volatility": "买入持有年化波动",
            "benchmark_sharpe": "买入持有夏普",
            "benchmark_max_drawdown": "买入持有最大回撤",
            "benchmark_calmar": "买入持有卡玛",

            "excess_annual_return": "超额年化收益",
            "sharpe_diff": "夏普差值",
            "drawdown_improvement": "回撤改善",

            "exposure": "持仓比例",
            "buy_count": "买入次数",
            "sell_count": "卖出次数",
            "total_trade_count": "总交易次数",
            "total_transaction_cost": "交易成本简单加总",
        }
    )

    return display_df


def run_batch_ma_backtest(
    stock_list: list[str] | None = None,
    fast_window: int = 20,
    slow_window: int = 60,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
    save_result: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    批量运行 20/60 均线回测。

    Returns
    -------
    batch_summary:
        每只股票一行的绩效汇总表。
    batch_results:
        每只股票完整逐日回测结果。
    """

    if stock_list is None:
        stock_files = sorted(
            PROCESSED_DIR.glob("*_clean.csv")
        )

        stock_list = [
            file.stem.replace("_clean", "")
            for file in stock_files
        ]

    batch_results = {}
    summary_rows = []

    for symbol in stock_list:
        symbol = str(symbol).zfill(6)

        file_path = (
            PROCESSED_DIR
            / f"{symbol}_clean.csv"
        )

        if not file_path.exists():
            print(f"{symbol} 本地清洗文件不存在，跳过：{file_path}")
            continue

        print(f"正在回测：{symbol}")

        stock_df = pd.read_csv(
            file_path,
            dtype={"symbol": str},
        )

        backtest_result = ma_cross_backtest(
            df=stock_df,
            fast_window=fast_window,
            slow_window=slow_window,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )

        summary = summarize_backtest_result(
            result=backtest_result,
            slow_window=slow_window,
            annual_risk_free_rate=annual_risk_free_rate,
            trading_days=trading_days,
        )

        batch_results[symbol] = backtest_result
        summary_rows.append(summary)

        if save_result:
            backtest_result.to_csv(
                BACKTEST_REPORT_DIR / f"{symbol}_ma_{fast_window}_{slow_window}_backtest.csv",
                index=False,
                encoding="utf-8-sig",
            )

    if not summary_rows:
        return pd.DataFrame(), batch_results

    batch_summary = pd.DataFrame(summary_rows)

    batch_summary = (
        batch_summary
        .sort_values(
            "strategy_sharpe",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if save_result:
        raw_path = (
            REPORT_DIR
            / f"ma_{fast_window}_{slow_window}_batch_summary_raw.csv"
        )

        display_path = (
            REPORT_DIR
            / f"ma_{fast_window}_{slow_window}_batch_summary_display.csv"
        )

        batch_summary.to_csv(
            raw_path,
            index=False,
            encoding="utf-8-sig",
        )

        format_batch_summary(batch_summary).to_csv(
            display_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"原始批量回测结果已保存到：{raw_path}")
        print(f"展示版批量回测结果已保存到：{display_path}")

    return batch_summary, batch_results


def summarize_backtest_period(
    result: pd.DataFrame,
    period_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict:
    """
    对单只股票在指定时间区间内的回测结果进行绩效汇总。

    注意：
    1. 回测信号仍然基于完整历史数据生成；
    2. 这里只是在评价阶段切分时间区间；
    3. 交易次数使用原始 position_change，不在切片后重新 diff；
    4. 这样可以避免把区间开始时已有仓位误判为新买入。
    """

    required_cols = {
        "date",
        "symbol",
        "slow_ma",
        "position",
        "position_change",
        "strategy_return",
        "asset_return",
        "transaction_cost",
    }

    missing_cols = required_cols - set(result.columns)

    if missing_cols:
        raise ValueError(f"缺少必要字段：{sorted(missing_cols)}")

    data = result.copy()

    data["date"] = pd.to_datetime(data["date"])

    symbol = (
        data["symbol"]
        .dropna()
        .astype(str)
        .str.zfill(6)
        .iloc[0]
    )

    # 只保留策略真正可以开始交易后的区间
    evaluation_mask = (
        data["slow_ma"]
        .shift(1)
        .notna()
    )

    if start_date is not None:
        evaluation_mask &= (
            data["date"] >= pd.to_datetime(start_date)
        )

    if end_date is not None:
        evaluation_mask &= (
            data["date"] <= pd.to_datetime(end_date)
        )

    evaluation_data = (
        data.loc[evaluation_mask]
        .copy()
        .reset_index(drop=True)
    )

    if evaluation_data.empty:
        raise ValueError(
            f"{symbol} 在 {period_name} 的评价区间为空"
        )

    strategy_metrics = calculate_performance(
        returns=evaluation_data["strategy_return"],
        annual_risk_free_rate=annual_risk_free_rate,
        trading_days=trading_days,
    )

    benchmark_metrics = calculate_performance(
        returns=evaluation_data["asset_return"],
        annual_risk_free_rate=annual_risk_free_rate,
        trading_days=trading_days,
    )

    buy_count = int(
        (evaluation_data["position_change"] > 0).sum()
    )

    sell_count = int(
        (evaluation_data["position_change"] < 0).sum()
    )

    exposure = float(
        evaluation_data["position"].mean()
    )

    total_transaction_cost = float(
        evaluation_data["transaction_cost"].sum()
    )

    return {
        "period": period_name,
        "symbol": symbol,
        "start_date": evaluation_data["date"].iloc[0],
        "end_date": evaluation_data["date"].iloc[-1],
        "trade_days": len(evaluation_data),

        "strategy_cumulative_return": strategy_metrics["cumulative_return"],
        "strategy_annual_return": strategy_metrics["annual_return"],
        "strategy_annual_volatility": strategy_metrics["annual_volatility"],
        "strategy_sharpe": strategy_metrics["sharpe_ratio"],
        "strategy_max_drawdown": strategy_metrics["max_drawdown"],
        "strategy_calmar": strategy_metrics["calmar_ratio"],

        "benchmark_cumulative_return": benchmark_metrics["cumulative_return"],
        "benchmark_annual_return": benchmark_metrics["annual_return"],
        "benchmark_annual_volatility": benchmark_metrics["annual_volatility"],
        "benchmark_sharpe": benchmark_metrics["sharpe_ratio"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "benchmark_calmar": benchmark_metrics["calmar_ratio"],

        "excess_annual_return": (
            strategy_metrics["annual_return"]
            - benchmark_metrics["annual_return"]
        ),
        "sharpe_diff": (
            strategy_metrics["sharpe_ratio"]
            - benchmark_metrics["sharpe_ratio"]
        ),
        "drawdown_improvement": (
            strategy_metrics["max_drawdown"]
            - benchmark_metrics["max_drawdown"]
        ),

        "exposure": exposure,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "total_trade_count": buy_count + sell_count,
        "total_transaction_cost": total_transaction_cost,
    }


def run_ma_parameter_grid_search(
    stock_list: list[str],
    param_grid: list[tuple[int, int]],
    in_sample_end: str = "2024-12-31",
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """
    在样本内对多个均线参数组合进行批量测试。

    注意：
    1. 这里只用于样本内参数比较；
    2. 不使用样本外结果选择参数；
    3. 每个参数组合都会对 stock_list 中的所有股票进行回测；
    4. 返回每只股票、每组参数的一行样本内绩效。
    """

    grid_rows = []
    grid_results = {}

    for fast_window, slow_window in param_grid:
        print(f"\n正在测试参数：{fast_window}/{slow_window}")

        if fast_window >= slow_window:
            print(
                f"跳过非法参数：{fast_window}/{slow_window}"
            )
            continue

        for symbol in stock_list:
            symbol = str(symbol).zfill(6)

            file_path = (
                PROCESSED_DIR
                / f"{symbol}_clean.csv"
            )

            if not file_path.exists():
                print(f"{symbol} 文件不存在，跳过")
                continue

            stock_df = pd.read_csv(
                file_path,
                dtype={"symbol": str},
            )

            backtest_result = ma_cross_backtest(
                df=stock_df,
                fast_window=fast_window,
                slow_window=slow_window,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            )

            summary = summarize_backtest_period(
                result=backtest_result,
                period_name="in_sample",
                start_date=None,
                end_date=in_sample_end,
                annual_risk_free_rate=annual_risk_free_rate,
                trading_days=trading_days,
            )

            summary["fast_window"] = fast_window
            summary["slow_window"] = slow_window
            summary["ma_param"] = f"{fast_window}/{slow_window}"

            grid_rows.append(summary)

            grid_results[
                (symbol, fast_window, slow_window)
            ] = backtest_result

    grid_summary = pd.DataFrame(grid_rows)

    if grid_summary.empty:
        return grid_summary, grid_results

    grid_summary = (
        grid_summary
        .sort_values(
            [
                "fast_window",
                "slow_window",
                "symbol",
            ]
        )
        .reset_index(drop=True)
    )

    return grid_summary, grid_results


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