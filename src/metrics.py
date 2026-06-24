import numpy as np
import pandas as pd
import matplotlib as plt

def calc_stock_metrics(df: pd.DataFrame, price_col: str = "close", get_df: bool = False):
    """
    计算单只股票的核心收益风险指标。

    参数：
    df: 股票行情数据，至少包含 date、close、symbol 列
    price_col: 用于计算收益率的价格列，默认使用 close

    返回：
    dict: 包含累计收益、年化收益、年化波动率、最大回撤、夏普比率等指标
    """

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    data["daily_return"] = data[price_col].pct_change().fillna(0)
    data["net_value"] = (1 + data["daily_return"]).cumprod()

    cumulative_return = data["net_value"].iloc[-1] - 1

    trading_days = 252
    n_days = len(data)

    annual_return = (1 + cumulative_return) ** (trading_days / n_days) - 1
    annual_volatility = data["daily_return"].std() * np.sqrt(trading_days)

    data["rolling_max"] = data["net_value"].cummax()
    data["drawdown"] = data["net_value"] / data["rolling_max"] - 1

    max_drawdown = data["drawdown"].min()

    sharpe_ratio = (
        annual_return / annual_volatility
        if annual_volatility != 0
        else np.nan
    )

    max_drawdown_end_idx = data["drawdown"].idxmin()
    max_drawdown_start_idx = data.loc[:max_drawdown_end_idx, "net_value"].idxmax()

    result = {
        "symbol": data["symbol"].iloc[0] if "symbol" in data.columns else None,
        "start_date": data["date"].min(),
        "end_date": data["date"].max(),
        "trading_days": n_days,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_start_date": data.loc[max_drawdown_start_idx, "date"],
        "max_drawdown_end_date": data.loc[max_drawdown_end_idx, "date"],
        "max_drawdown_start_value": data.loc[max_drawdown_start_idx, "net_value"],
        "max_drawdown_end_value": data.loc[max_drawdown_end_idx, "net_value"],
    }

    return result if get_df else data

def print_metrics(metrics: dict):
    print(f"股票代码: {metrics['symbol']}")
    print(f"开始日期: {metrics['start_date'].date()}")
    print(f"结束日期: {metrics['end_date'].date()}")
    print(f"交易日数量: {metrics['trading_days']}")
    print(f"累计收益: {metrics['cumulative_return']:.2%}")
    print(f"年化收益: {metrics['annual_return']:.2%}")
    print(f"年化波动率: {metrics['annual_volatility']:.2%}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"最大回撤开始日期: {metrics['max_drawdown_start_date'].date()}")
    print(f"最大回撤结束日期: {metrics['max_drawdown_end_date'].date()}")
    print(f"回撤前最高净值: {metrics['max_drawdown_start_value']:.4f}")
    print(f"回撤最低净值: {metrics['max_drawdown_end_value']:.4f}")
    
def plot_metrics(data: pd.DataFrame):
    # 找到最大回撤最低点
    max_drawdown_end_idx = data["drawdown"].idxmin()
    max_drawdown_end_date = data.loc[max_drawdown_end_idx, "date"]
    max_drawdown_end_value = data.loc[max_drawdown_end_idx, "net_value"]

    # 在最大回撤最低点之前，找到历史最高点
    max_drawdown_start_idx = data.loc[:max_drawdown_end_idx, "net_value"].idxmax()
    max_drawdown_start_date = data.loc[max_drawdown_start_idx, "date"]
    max_drawdown_start_value = data.loc[max_drawdown_start_idx, "net_value"]
    
    plt.figure(figsize=(12, 5))
    plt.plot(data["date"], data["net_value"], label="Net Value")
    # 标记最大回撤起点
    plt.scatter(max_drawdown_start_date, max_drawdown_start_value, label="Drawdown Start")
    # 标记最大回撤终点
    plt.scatter(max_drawdown_end_date, max_drawdown_end_value, label="Drawdown End")

    plt.title("000001 Net Value Curve with Max Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Net Value")
    plt.grid(True)
    plt.legend()
    plt.show()


    plt.figure(figsize=(12, 5))
    plt.plot(data["date"], data["drawdown"])
    plt.title("000001 Drawdown Curve")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()