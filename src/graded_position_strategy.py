"""
双均线分级仓位策略。

第一版固定逻辑：

1. fast_ma > slow_ma：
   目标仓位为 1.0；

2. fast_ma <= slow_ma，但出现短期反弹：
   rebound_return > rebound_return_threshold
   且 close > fast_ma，
   目标仓位为 partial_position；

3. 其他情况：
   目标仓位为 0.0；

4. 当日收盘后生成目标仓位，
   下一交易日执行；

5. 交易成本按仓位变化绝对值计算。

本模块不负责参数网格搜索。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


try:
    from .backtest import (
        PROCESSED_DIR,
        REPORT_DIR,
        summarize_backtest_result,
        summarize_backtest_period,
    )
except ImportError:
    from backtest import (
        PROCESSED_DIR,
        REPORT_DIR,
        summarize_backtest_result,
        summarize_backtest_period,
    )


GRADED_POSITION_REPORT_DIR = (
    REPORT_DIR
    / "graded_position"
)


SignalVersion = Literal["v1", "v2"]


def _prepare_graded_price_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    清洗和排序单只股票数据。
    """
    required_columns = {
        "date",
        "close",
    }

    _require_columns(
        data=df,
        required_columns=required_columns,
        data_name="df",
    )

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["close"] = pd.to_numeric(
        data["close"],
        errors="raise",
    )

    if "symbol" in data.columns:
        data["symbol"] = (
            data["symbol"]
            .astype(str)
            .str.zfill(6)
        )

    if data["close"].isna().any():
        raise ValueError(
            "close 存在缺失值"
        )

    if (data["close"] <= 0).any():
        raise ValueError(
            "close 必须全部大于 0"
        )

    data = (
        data.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(
            "df 不能为空"
        )

    return data


def _add_graded_common_features(
    data: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    rebound_window: int,
    volatility_window: int,
    fast_ma_slope_window: int,
) -> pd.DataFrame:
    """
    计算 v1 和 v2 共用的指标。

    包括：

    - 股票收益；
    - 快慢均线；
    - 原始二元均线信号；
    - 短期反弹收益；
    - 波动调整反弹强度；
    - 快均线斜率；
    - 距最近一次均线退出的交易日数。
    """
    result = data.copy()

    result["asset_return"] = (
        result["close"].pct_change()
    )

    result["fast_ma"] = (
        result["close"]
        .rolling(
            window=fast_window,
            min_periods=fast_window,
        )
        .mean()
    )

    result["slow_ma"] = (
        result["close"]
        .rolling(
            window=slow_window,
            min_periods=slow_window,
        )
        .mean()
    )

    result["rebound_return"] = (
        result["close"]
        .pct_change(rebound_window)
    )

    result["daily_volatility"] = (
        result["asset_return"]
        .rolling(
            window=volatility_window,
            min_periods=volatility_window,
        )
        .std(ddof=1)
    )

    result["expected_rebound_move"] = (
        result["daily_volatility"]
        * np.sqrt(rebound_window)
    )

    result["rebound_score"] = (
        result["rebound_return"]
        / result[
            "expected_rebound_move"
        ].replace(
            0.0,
            np.nan,
        )
    )

    result["fast_ma_slope"] = (
        result["fast_ma"]
        / result["fast_ma"].shift(
            fast_ma_slope_window
        )
        - 1.0
    )

    result["binary_signal"] = (
        result["fast_ma"]
        > result["slow_ma"]
    )

    result.loc[
        result["slow_ma"].isna(),
        "binary_signal",
    ] = False

    result["binary_signal"] = (
        result["binary_signal"]
        .astype(bool)
    )

    previous_binary_signal = (
        result["binary_signal"]
        .shift(1)
        .fillna(False)
        .astype(bool)
    )

    result["binary_entry_event"] = (
        ~previous_binary_signal
        & result["binary_signal"]
    )

    result["binary_exit_event"] = (
        previous_binary_signal
        & ~result["binary_signal"]
    )

    result["days_since_exit"] = (
        _calculate_days_since_exit(
            result["binary_signal"]
        )
    )

    return result


def _calculate_days_since_exit(
    binary_signal: pd.Series,
) -> pd.Series:
    """
    计算距离最近一次二元均线退出的交易日数。

    退出当日：
        days_since_exit = 0

    退出后的下一交易日：
        days_since_exit = 1

    二元均线重新转为多头：
        days_since_exit = NaN
    """
    signal = (
        binary_signal
        .fillna(False)
        .astype(bool)
    )

    result = pd.Series(
        np.nan,
        index=signal.index,
        dtype=float,
    )

    previous_signal = False
    counter: int | None = None

    for index, current_signal in signal.items():
        exit_event = (
            previous_signal
            and not current_signal
        )

        if current_signal:
            counter = None

        elif exit_event:
            counter = 0
            result.loc[index] = 0.0

        elif counter is not None:
            counter += 1
            result.loc[index] = float(
                counter
            )

        previous_signal = current_signal

    return result


def build_graded_signal_v1(
    data: pd.DataFrame,
    partial_position: float = 0.3,
    rebound_return_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    第一版分级仓位信号。

    规则：

    1. fast_ma > slow_ma：
       signal = 1.0

    2. fast_ma <= slow_ma，
       rebound_return > threshold，
       close > fast_ma：
       signal = partial_position

    3. 其他：
       signal = 0.0
    """
    result = data.copy()

    required_columns = {
        "close",
        "fast_ma",
        "slow_ma",
        "binary_signal",
        "rebound_return",
    }

    _require_columns(
        data=result,
        required_columns=required_columns,
        data_name="data",
    )

    result["bearish_ma_condition"] = (
        ~result["binary_signal"]
        & result["slow_ma"].notna()
    )

    result["positive_rebound_condition"] = (
        result["rebound_return"]
        > rebound_return_threshold
    )

    result["price_above_fast_ma_condition"] = (
        result["close"]
        > result["fast_ma"]
    )

    result["partial_rebound_signal"] = (
        result["bearish_ma_condition"]
        & result[
            "positive_rebound_condition"
        ]
        & result[
            "price_above_fast_ma_condition"
        ]
    )

    result["signal"] = np.select(
        condlist=[
            result["binary_signal"],
            result[
                "partial_rebound_signal"
            ],
        ],
        choicelist=[
            1.0,
            partial_position,
        ],
        default=0.0,
    ).astype(float)

    result.loc[
        result["slow_ma"].isna(),
        "signal",
    ] = 0.0

    return result


def build_graded_signal_v2(
    data: pd.DataFrame,
    partial_position: float = 0.3,
    max_days_since_exit: int = 20,
    rebound_score_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    第二版分级仓位信号。

    规则：

    1. fast_ma > slow_ma：
       signal = 1.0

    2. fast_ma <= slow_ma，且满足：

       - 距离最近退出 1～max_days_since_exit 日；
       - close > fast_ma；
       - fast_ma_slope > 0；
       - rebound_score >= threshold；

       signal = partial_position

    3. 其他：
       signal = 0.0
    """
    result = data.copy()

    required_columns = {
        "close",
        "fast_ma",
        "slow_ma",
        "binary_signal",
        "days_since_exit",
        "fast_ma_slope",
        "rebound_score",
    }

    _require_columns(
        data=result,
        required_columns=required_columns,
        data_name="data",
    )

    result["bearish_ma_condition"] = (
        ~result["binary_signal"]
        & result["slow_ma"].notna()
    )

    result["recent_exit_condition"] = (
        result["days_since_exit"]
        .between(
            1,
            max_days_since_exit,
            inclusive="both",
        )
    )

    result["price_above_fast_ma_condition"] = (
        result["close"]
        > result["fast_ma"]
    )

    result["fast_ma_turning_up_condition"] = (
        result["fast_ma_slope"] > 0
    )

    result["strong_rebound_condition"] = (
        result["rebound_score"]
        >= rebound_score_threshold
    )

    result["partial_rebound_signal"] = (
        result["bearish_ma_condition"]
        & result[
            "recent_exit_condition"
        ]
        & result[
            "price_above_fast_ma_condition"
        ]
        & result[
            "fast_ma_turning_up_condition"
        ]
        & result[
            "strong_rebound_condition"
        ]
    )

    result["signal"] = np.select(
        condlist=[
            result["binary_signal"],
            result[
                "partial_rebound_signal"
            ],
        ],
        choicelist=[
            1.0,
            partial_position,
        ],
        default=0.0,
    ).astype(float)

    result.loc[
        result["slow_ma"].isna(),
        "signal",
    ] = 0.0

    return result


def _execute_graded_position_backtest(
    data: pd.DataFrame,
    partial_position: float,
    commission_rate: float,
    slippage_rate: float,
) -> pd.DataFrame:
    """
    对已经生成 signal 的数据执行统一回测。

    v1 和 v2 完全共享这一部分。
    """
    result = data.copy()

    result["signal_state"] = (
        _classify_position_state(
            position=result["signal"],
            partial_position=(
                partial_position
            ),
        )
    )

    # 当日收盘生成信号，下一交易日执行。
    result["position"] = (
        result["signal"]
        .shift(1)
        .fillna(0.0)
    )

    result["position_state"] = (
        _classify_position_state(
            position=result["position"],
            partial_position=(
                partial_position
            ),
        )
    )

    result["position_change"] = (
        result["position"]
        .diff()
        .fillna(
            result["position"]
        )
    )

    result["turnover"] = (
        result[
            "position_change"
        ].abs()
    )

    # 当前实际仓位对应的是前一日信号，
    # 因此诊断指标也向后移动一天。
    result["executed_days_since_exit"] = (
        result[
            "days_since_exit"
        ].shift(1)
    )

    result["executed_rebound_score"] = (
        result[
            "rebound_score"
        ].shift(1)
    )

    result["executed_fast_ma_slope"] = (
        result[
            "fast_ma_slope"
        ].shift(1)
    )

    benchmark_return = (
        result["asset_return"]
        .fillna(0.0)
    )

    one_way_cost = (
        commission_rate
        + slippage_rate
    )

    result["gross_strategy_return"] = (
        result["position"]
        * benchmark_return
    )

    result["transaction_cost"] = (
        result["turnover"]
        * one_way_cost
    )

    result["strategy_return"] = (
        result["gross_strategy_return"]
        - result["transaction_cost"]
    )

    result["strategy_nav"] = (
        1.0
        + result["strategy_return"]
    ).cumprod()

    result["benchmark_nav"] = (
        1.0
        + benchmark_return
    ).cumprod()

    result["strategy_rolling_max"] = (
        result["strategy_nav"]
        .cummax()
    )

    result["strategy_drawdown"] = (
        result["strategy_nav"]
        / result["strategy_rolling_max"]
        - 1.0
    )

    result["benchmark_rolling_max"] = (
        result["benchmark_nav"]
        .cummax()
    )

    result["benchmark_drawdown"] = (
        result["benchmark_nav"]
        / result["benchmark_rolling_max"]
        - 1.0
    )

    # 主动收益归因。
    result["active_return"] = (
        result["strategy_return"]
        - benchmark_return
    )

    result["timing_active_return"] = (
        result["gross_strategy_return"]
        - benchmark_return
    )

    result["cost_drag_return"] = (
        -result["transaction_cost"]
    )

    result["missed_upside_return"] = (
        np.where(
            benchmark_return > 0,
            result[
                "timing_active_return"
            ],
            0.0,
        )
    )

    result[
        "avoided_downside_return"
    ] = np.where(
        benchmark_return < 0,
        result["timing_active_return"],
        0.0,
    )

    result["strategy_log_return"] = (
        np.log1p(
            result["strategy_return"]
        )
    )

    result["benchmark_log_return"] = (
        np.log1p(
            benchmark_return
        )
    )

    result["active_log_return"] = (
        result["strategy_log_return"]
        - result[
            "benchmark_log_return"
        ]
    )

    return result


def graded_ma_backtest(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 40,
    rebound_window: int = 5,
    partial_position: float = 0.3,
    signal_version: SignalVersion = "v1",
    rebound_return_threshold: float = 0.0,
    volatility_window: int = 20,
    fast_ma_slope_window: int = 3,
    max_days_since_exit: int = 20,
    rebound_score_threshold: float = 1.0,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
) -> pd.DataFrame:
    """
    统一运行 v1 或 v2 分级仓位策略。

    signal_version
    --------------
    v1：
        宽松反弹条件。

    v2：
        近期离场限制
        + 波动调整反弹强度
        + 快均线斜率过滤。
    """
    _validate_positive_integer(
        fast_window,
        "fast_window",
    )

    _validate_positive_integer(
        slow_window,
        "slow_window",
    )

    _validate_positive_integer(
        rebound_window,
        "rebound_window",
    )

    _validate_positive_integer(
        volatility_window,
        "volatility_window",
    )

    _validate_positive_integer(
        fast_ma_slope_window,
        "fast_ma_slope_window",
    )

    _validate_positive_integer(
        max_days_since_exit,
        "max_days_since_exit",
    )

    if fast_window >= slow_window:
        raise ValueError(
            "fast_window 必须小于 slow_window"
        )

    if not 0 < partial_position < 1:
        raise ValueError(
            "partial_position 必须位于 0 和 1 之间"
        )

    if signal_version not in {
        "v1",
        "v2",
    }:
        raise ValueError(
            "signal_version 必须是 'v1' 或 'v2'"
        )

    if rebound_score_threshold <= 0:
        raise ValueError(
            "rebound_score_threshold 必须大于 0"
        )

    if commission_rate < 0:
        raise ValueError(
            "commission_rate 不能为负数"
        )

    if slippage_rate < 0:
        raise ValueError(
            "slippage_rate 不能为负数"
        )

    data = _prepare_graded_price_data(
        df
    )

    data = _add_graded_common_features(
        data=data,
        fast_window=fast_window,
        slow_window=slow_window,
        rebound_window=rebound_window,
        volatility_window=(
            volatility_window
        ),
        fast_ma_slope_window=(
            fast_ma_slope_window
        ),
    )

    if signal_version == "v1":
        data = build_graded_signal_v1(
            data=data,
            partial_position=(
                partial_position
            ),
            rebound_return_threshold=(
                rebound_return_threshold
            ),
        )

    else:
        data = build_graded_signal_v2(
            data=data,
            partial_position=(
                partial_position
            ),
            max_days_since_exit=(
                max_days_since_exit
            ),
            rebound_score_threshold=(
                rebound_score_threshold
            ),
        )

    data = (
        _execute_graded_position_backtest(
            data=data,
            partial_position=(
                partial_position
            ),
            commission_rate=(
                commission_rate
            ),
            slippage_rate=(
                slippage_rate
            ),
        )
    )

    data["signal_version"] = (
        signal_version
    )

    data["fast_window"] = (
        fast_window
    )

    data["slow_window"] = (
        slow_window
    )

    data["rebound_window"] = (
        rebound_window
    )

    data["partial_position_level"] = (
        partial_position
    )

    data["rebound_return_threshold"] = (
        rebound_return_threshold
    )

    data["volatility_window"] = (
        volatility_window
    )

    data["fast_ma_slope_window"] = (
        fast_ma_slope_window
    )

    data["max_days_since_exit"] = (
        max_days_since_exit
    )

    data["rebound_score_threshold"] = (
        rebound_score_threshold
    )

    _check_graded_backtest_result(
        data=data,
        partial_position=(
            partial_position
        ),
        commission_rate=(
            commission_rate
        ),
        slippage_rate=(
            slippage_rate
        ),
    )

    return data


def run_batch_graded_ma_backtest(
    stock_list: Sequence[str] | None = None,
    fast_window: int = 10,
    slow_window: int = 40,
    rebound_window: int = 5,
    partial_position: float = 0.3,
    signal_version: SignalVersion = "v1",
    rebound_return_threshold: float = 0.0,
    volatility_window: int = 20,
    fast_ma_slope_window: int = 3,
    max_days_since_exit: int = 20,
    rebound_score_threshold: float = 1.0,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
    save_result: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """
    统一批量运行 v1 或 v2 分级仓位策略。
    """
    if stock_list is None:
        stock_files = sorted(
            PROCESSED_DIR.glob(
                "*_clean.csv"
            )
        )

        stock_list = [
            file.stem.replace(
                "_clean",
                "",
            )
            for file in stock_files
        ]

    batch_results: dict[
        str,
        pd.DataFrame,
    ] = {}

    summary_rows: list[
        dict[str, Any]
    ] = []

    output_dir = (
        GRADED_POSITION_REPORT_DIR
        / signal_version
        / (
            f"ma_{fast_window}_"
            f"{slow_window}"
        )
    )

    if save_result:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    for raw_symbol in stock_list:
        symbol = str(
            raw_symbol
        ).zfill(6)

        file_path = (
            PROCESSED_DIR
            / f"{symbol}_clean.csv"
        )

        if not file_path.exists():
            print(
                f"{symbol} 文件不存在，跳过："
                f"{file_path}"
            )
            continue

        print(
            f"正在回测 {signal_version}："
            f"{symbol}"
        )

        stock_data = pd.read_csv(
            file_path,
            dtype={
                "symbol": str,
            },
        )

        result = graded_ma_backtest(
            df=stock_data,
            fast_window=fast_window,
            slow_window=slow_window,
            rebound_window=(
                rebound_window
            ),
            partial_position=(
                partial_position
            ),
            signal_version=(
                signal_version
            ),
            rebound_return_threshold=(
                rebound_return_threshold
            ),
            volatility_window=(
                volatility_window
            ),
            fast_ma_slope_window=(
                fast_ma_slope_window
            ),
            max_days_since_exit=(
                max_days_since_exit
            ),
            rebound_score_threshold=(
                rebound_score_threshold
            ),
            commission_rate=(
                commission_rate
            ),
            slippage_rate=(
                slippage_rate
            ),
        )

        summary = summarize_backtest_result(
            result=result,
            slow_window=slow_window,
            annual_risk_free_rate=(
                annual_risk_free_rate
            ),
            trading_days=trading_days,
        )

        evaluation_data = (
            _select_evaluation_data(
                result=result,
                start_date=None,
                end_date=None,
            )
        )

        position_state = (
            evaluation_data[
                "position_state"
            ]
        )

        signal_state = (
            evaluation_data[
                "signal_state"
            ]
        )

        partial_position_mask = (
            position_state == "partial"
        )

        partial_signal_mask = (
            signal_state == "partial"
        )

        previous_position_state = (
            position_state
            .shift(1)
            .fillna("cash")
        )

        partial_entry_mask = (
            partial_position_mask
            & (
                previous_position_state
                != "partial"
            )
        )

        partial_exit_mask = (
            (
                previous_position_state
                == "partial"
            )
            & ~partial_position_mask
        )

        summary.update(
            {
                "strategy_name": (
                    f"graded_{signal_version}"
                ),
                "signal_version": (
                    signal_version
                ),
                "fast_window": (
                    fast_window
                ),
                "slow_window": (
                    slow_window
                ),
                "rebound_window": (
                    rebound_window
                ),
                "partial_position": (
                    partial_position
                ),
                "rebound_return_threshold": (
                    rebound_return_threshold
                ),
                "volatility_window": (
                    volatility_window
                ),
                "fast_ma_slope_window": (
                    fast_ma_slope_window
                ),
                "max_days_since_exit": (
                    max_days_since_exit
                ),
                "rebound_score_threshold": (
                    rebound_score_threshold
                ),
                "full_position_rate": float(
                    (
                        position_state
                        == "full"
                    ).mean()
                ),
                "partial_position_rate": float(
                    partial_position_mask.mean()
                ),
                "cash_rate": float(
                    (
                        position_state
                        == "cash"
                    ).mean()
                ),
                "partial_signal_rate": float(
                    partial_signal_mask.mean()
                ),
                "partial_entry_count": int(
                    partial_entry_mask.sum()
                ),
                "partial_exit_count": int(
                    partial_exit_mask.sum()
                ),
                "total_turnover": float(
                    evaluation_data[
                        "turnover"
                    ].sum()
                ),
                "avg_rebound_score_at_partial_signal": (
                    _safe_mean(
                        evaluation_data.loc[
                            partial_signal_mask,
                            "rebound_score",
                        ]
                    )
                ),
                "avg_days_since_exit_at_partial_signal": (
                    _safe_mean(
                        evaluation_data.loc[
                            partial_signal_mask,
                            "days_since_exit",
                        ]
                    )
                ),
                "avg_fast_ma_slope_at_partial_signal": (
                    _safe_mean(
                        evaluation_data.loc[
                            partial_signal_mask,
                            "fast_ma_slope",
                        ]
                    )
                ),
            }
        )

        batch_results[symbol] = result
        summary_rows.append(summary)

        if save_result:
            result.to_csv(
                output_dir
                / (
                    f"{symbol}_"
                    f"{signal_version}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

    if not summary_rows:
        return (
            pd.DataFrame(),
            batch_results,
        )

    batch_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(
            "strategy_sharpe",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if save_result:
        batch_summary.to_csv(
            output_dir
            / "batch_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return (
        batch_summary,
        batch_results,
    )


def summarize_batch_period(
    batch_results: Mapping[
        str,
        pd.DataFrame,
    ],
    period_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    strategy_name: str = "strategy",
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    对批量回测结果在统一时间区间内生成绩效汇总。

    既可用于原始二元仓位结果，
    也可用于分级仓位结果。
    """
    if not batch_results:
        raise ValueError(
            "batch_results 不能为空"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for raw_symbol, result in (
        batch_results.items()
    ):
        symbol = str(
            raw_symbol
        ).zfill(6)

        try:
            summary = summarize_backtest_period(
                result=result,
                period_name=period_name,
                start_date=start_date,
                end_date=end_date,
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
                trading_days=trading_days,
            )
        except ValueError as error:
            print(
                f"跳过 {symbol}：{error}"
            )
            continue

        evaluation_data = (
            _select_evaluation_data(
                result=result,
                start_date=start_date,
                end_date=end_date,
            )
        )

        summary["strategy_name"] = (
            strategy_name
        )

        summary["total_turnover"] = float(
            evaluation_data[
                "position_change"
            ].abs().sum()
        )

        if (
            "position_state"
            in evaluation_data.columns
        ):
            summary[
                "full_position_rate"
            ] = float(
                (
                    evaluation_data[
                        "position_state"
                    ]
                    == "full"
                ).mean()
            )

            summary[
                "partial_position_rate"
            ] = float(
                (
                    evaluation_data[
                        "position_state"
                    ]
                    == "partial"
                ).mean()
            )

            summary["cash_rate"] = float(
                (
                    evaluation_data[
                        "position_state"
                    ]
                    == "cash"
                ).mean()
            )
        else:
            summary[
                "full_position_rate"
            ] = float(
                (
                    evaluation_data[
                        "position"
                    ]
                    > 0.5
                ).mean()
            )

            summary[
                "partial_position_rate"
            ] = 0.0

            summary["cash_rate"] = float(
                (
                    evaluation_data[
                        "position"
                    ]
                    <= 0.5
                ).mean()
            )

        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "strategy_sharpe",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def compare_strategy_summaries(
    binary_summary: pd.DataFrame,
    graded_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    对同一只股票上的二元仓位和分级仓位结果做成对比较。

    正数含义：

    annual_return_improvement > 0：
        分级仓位年化收益更高。

    sharpe_improvement > 0：
        分级仓位夏普更高。

    drawdown_improvement_vs_binary > 0：
        分级仓位最大回撤更接近 0。

    excess_return_improvement > 0：
        分级仓位相对买入持有的超额收益更高。
    """
    required_columns = {
        "symbol",
        "strategy_annual_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "strategy_calmar",
        "excess_annual_return",
        "exposure",
        "total_trade_count",
        "total_transaction_cost",
        "total_turnover",
    }

    _require_columns(
        binary_summary,
        required_columns,
        "binary_summary",
    )

    _require_columns(
        graded_summary,
        required_columns,
        "graded_summary",
    )

    binary = binary_summary.copy()
    graded = graded_summary.copy()

    binary["symbol"] = (
        binary["symbol"]
        .astype(str)
        .str.zfill(6)
    )

    graded["symbol"] = (
        graded["symbol"]
        .astype(str)
        .str.zfill(6)
    )

    comparison = binary.merge(
        graded,
        on="symbol",
        how="inner",
        suffixes=(
            "_binary",
            "_graded",
        ),
        validate="one_to_one",
    )

    if comparison.empty:
        raise ValueError(
            "两组结果没有共同股票"
        )

    comparison[
        "annual_return_improvement"
    ] = (
        comparison[
            "strategy_annual_return_graded"
        ]
        - comparison[
            "strategy_annual_return_binary"
        ]
    )

    comparison[
        "sharpe_improvement"
    ] = (
        comparison[
            "strategy_sharpe_graded"
        ]
        - comparison[
            "strategy_sharpe_binary"
        ]
    )

    comparison[
        "calmar_improvement"
    ] = (
        comparison[
            "strategy_calmar_graded"
        ]
        - comparison[
            "strategy_calmar_binary"
        ]
    )

    comparison[
        "drawdown_improvement_vs_binary"
    ] = (
        comparison[
            "strategy_max_drawdown_graded"
        ]
        - comparison[
            "strategy_max_drawdown_binary"
        ]
    )

    comparison[
        "excess_return_improvement"
    ] = (
        comparison[
            "excess_annual_return_graded"
        ]
        - comparison[
            "excess_annual_return_binary"
        ]
    )

    comparison["exposure_change"] = (
        comparison["exposure_graded"]
        - comparison["exposure_binary"]
    )

    comparison[
        "trade_count_change"
    ] = (
        comparison[
            "total_trade_count_graded"
        ]
        - comparison[
            "total_trade_count_binary"
        ]
    )

    comparison["turnover_change"] = (
        comparison[
            "total_turnover_graded"
        ]
        - comparison[
            "total_turnover_binary"
        ]
    )

    comparison[
        "transaction_cost_change"
    ] = (
        comparison[
            "total_transaction_cost_graded"
        ]
        - comparison[
            "total_transaction_cost_binary"
        ]
    )

    return comparison.sort_values(
        "sharpe_improvement",
        ascending=False,
    ).reset_index(drop=True)


def summarize_strategy_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    汇总分级仓位相对二元仓位的跨股票表现。
    """
    required_columns = {
        "symbol",
        "annual_return_improvement",
        "sharpe_improvement",
        "calmar_improvement",
        "drawdown_improvement_vs_binary",
        "excess_return_improvement",
        "exposure_change",
        "trade_count_change",
        "turnover_change",
        "transaction_cost_change",
    }

    _require_columns(
        comparison,
        required_columns,
        "comparison",
    )

    result = {
        "stock_count": int(
            comparison[
                "symbol"
            ].nunique()
        ),
        "avg_annual_return_improvement": float(
            comparison[
                "annual_return_improvement"
            ].mean()
        ),
        "median_annual_return_improvement": float(
            comparison[
                "annual_return_improvement"
            ].median()
        ),
        "annual_return_win_rate": float(
            (
                comparison[
                    "annual_return_improvement"
                ]
                > 0
            ).mean()
        ),
        "avg_sharpe_improvement": float(
            comparison[
                "sharpe_improvement"
            ].mean()
        ),
        "median_sharpe_improvement": float(
            comparison[
                "sharpe_improvement"
            ].median()
        ),
        "sharpe_win_rate": float(
            (
                comparison[
                    "sharpe_improvement"
                ]
                > 0
            ).mean()
        ),
        "avg_calmar_improvement": float(
            comparison[
                "calmar_improvement"
            ].mean()
        ),
        "calmar_win_rate": float(
            (
                comparison[
                    "calmar_improvement"
                ]
                > 0
            ).mean()
        ),
        "avg_drawdown_improvement": float(
            comparison[
                "drawdown_improvement_vs_binary"
            ].mean()
        ),
        "drawdown_win_rate": float(
            (
                comparison[
                    "drawdown_improvement_vs_binary"
                ]
                > 0
            ).mean()
        ),
        "avg_excess_return_improvement": float(
            comparison[
                "excess_return_improvement"
            ].mean()
        ),
        "excess_return_win_rate": float(
            (
                comparison[
                    "excess_return_improvement"
                ]
                > 0
            ).mean()
        ),
        "avg_exposure_change": float(
            comparison[
                "exposure_change"
            ].mean()
        ),
        "avg_trade_count_change": float(
            comparison[
                "trade_count_change"
            ].mean()
        ),
        "avg_turnover_change": float(
            comparison[
                "turnover_change"
            ].mean()
        ),
        "avg_transaction_cost_change": float(
            comparison[
                "transaction_cost_change"
            ].mean()
        ),
    }

    return pd.DataFrame(
        [result]
    )


def build_timing_attribution_detail(
    batch_results: Mapping[
        str,
        pd.DataFrame,
    ],
    strategy_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    为任意多头/空仓或分级仓位策略生成主动收益归因。

    可同时用于：

    - 原始二元 10/40；
    - 分级仓位 10/40。
    """
    if not batch_results:
        raise ValueError(
            "batch_results 不能为空"
        )

    frames: list[
        pd.DataFrame
    ] = []

    required_columns = {
        "date",
        "symbol",
        "asset_return",
        "position",
        "position_change",
        "gross_strategy_return",
        "transaction_cost",
        "strategy_return",
        "slow_ma",
    }

    for raw_symbol, result in (
        batch_results.items()
    ):
        _require_columns(
            result,
            required_columns,
            f"result[{raw_symbol}]",
        )

        data = _select_evaluation_data(
            result=result,
            start_date=start_date,
            end_date=end_date,
        )

        benchmark_return = (
            data["asset_return"]
            .fillna(0.0)
        )

        data["strategy_name"] = (
            strategy_name
        )

        data["active_return"] = (
            data["strategy_return"]
            - benchmark_return
        )

        data["timing_active_return"] = (
            data["gross_strategy_return"]
            - benchmark_return
        )

        data["cost_drag_return"] = (
            -data["transaction_cost"]
        )

        data["missed_upside_return"] = (
            np.where(
                (
                    benchmark_return > 0
                )
                & (
                    data["position"] < 1.0
                ),
                data[
                    "timing_active_return"
                ],
                0.0,
            )
        )

        data[
            "avoided_downside_return"
        ] = np.where(
            (
                benchmark_return < 0
            )
            & (
                data["position"] < 1.0
            ),
            data["timing_active_return"],
            0.0,
        )

        data["strategy_log_return"] = (
            np.log1p(
                data["strategy_return"]
            )
        )

        data["benchmark_log_return"] = (
            np.log1p(
                benchmark_return
            )
        )

        data["active_log_return"] = (
            data["strategy_log_return"]
            - data["benchmark_log_return"]
        )

        if "position_state" not in data:
            data["position_state"] = np.where(
                data["position"] >= 1.0,
                "full",
                np.where(
                    data["position"] > 0.0,
                    "partial",
                    "cash",
                ),
            )

        frames.append(data)

    detail = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_active = (
        detail["timing_active_return"]
        + detail["cost_drag_return"]
    )

    if not np.allclose(
        detail["active_return"],
        expected_active,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "主动收益归因不一致"
        )

    expected_timing = (
        detail["missed_upside_return"]
        + detail["avoided_downside_return"]
    )

    if not np.allclose(
        detail["timing_active_return"],
        expected_timing,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "择时收益归因不一致"
        )

    return (
        detail.sort_values(
            [
                "strategy_name",
                "symbol",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_timing_attribution(
    attribution_detail: pd.DataFrame,
    group_columns: Sequence[str] = (
        "strategy_name",
    ),
) -> pd.DataFrame:
    """
    汇总错过上涨、规避下跌和交易成本。
    """
    group_columns = list(
        group_columns
    )

    required_columns = {
        "date",
        "symbol",
        "active_return",
        "active_log_return",
        "missed_upside_return",
        "avoided_downside_return",
        "cost_drag_return",
        *group_columns,
    }

    _require_columns(
        attribution_detail,
        required_columns,
        "attribution_detail",
    )

    rows: list[
        dict[str, Any]
    ] = []

    grouped = attribution_detail.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for group_key, group in grouped:
        if not isinstance(
            group_key,
            tuple,
        ):
            group_key = (
                group_key,
            )

        missed_upside_loss = float(
            -group[
                "missed_upside_return"
            ].sum()
        )

        avoided_downside_benefit = float(
            group[
                "avoided_downside_return"
            ].sum()
        )

        transaction_cost_loss = float(
            -group[
                "cost_drag_return"
            ].sum()
        )

        total_loss = (
            missed_upside_loss
            + transaction_cost_loss
        )

        if np.isclose(
            total_loss,
            0.0,
        ):
            benefit_cost_ratio = np.nan
        else:
            benefit_cost_ratio = (
                avoided_downside_benefit
                / total_loss
            )

        row = dict(
            zip(
                group_columns,
                group_key,
            )
        )

        row.update(
            {
                "observation_count": int(
                    len(group)
                ),
                "stock_count": int(
                    group[
                        "symbol"
                    ].nunique()
                ),
                "active_return_sum": float(
                    group[
                        "active_return"
                    ].sum()
                ),
                "active_log_return_sum": float(
                    group[
                        "active_log_return"
                    ].sum()
                ),
                "missed_upside_loss": (
                    missed_upside_loss
                ),
                "avoided_downside_benefit": (
                    avoided_downside_benefit
                ),
                "transaction_cost_loss": (
                    transaction_cost_loss
                ),
                "net_timing_benefit": (
                    avoided_downside_benefit
                    - missed_upside_loss
                ),
                "benefit_cost_ratio": (
                    benefit_cost_ratio
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_position_state_performance(
    attribution_detail: pd.DataFrame,
    group_columns: Sequence[str] = (
        "strategy_name",
        "position_state",
    ),
) -> pd.DataFrame:
    """
    分析 full、partial、cash 三种实际仓位状态的收益贡献。
    """
    group_columns = list(
        group_columns
    )

    required_columns = {
        "date",
        "symbol",
        "position",
        "strategy_return",
        "asset_return",
        "active_return",
        "active_log_return",
        "transaction_cost",
        "missed_upside_return",
        "avoided_downside_return",
        *group_columns,
    }

    _require_columns(
        attribution_detail,
        required_columns,
        "attribution_detail",
    )

    result = (
        attribution_detail.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            observation_count=(
                "date",
                "count",
            ),
            stock_count=(
                "symbol",
                "nunique",
            ),
            average_position=(
                "position",
                "mean",
            ),
            strategy_mean_daily_return=(
                "strategy_return",
                "mean",
            ),
            benchmark_mean_daily_return=(
                "asset_return",
                "mean",
            ),
            active_mean_daily_return=(
                "active_return",
                "mean",
            ),
            active_return_sum=(
                "active_return",
                "sum",
            ),
            active_log_return_sum=(
                "active_log_return",
                "sum",
            ),
            missed_upside_sum=(
                "missed_upside_return",
                "sum",
            ),
            avoided_downside_sum=(
                "avoided_downside_return",
                "sum",
            ),
            transaction_cost_sum=(
                "transaction_cost",
                "sum",
            ),
        )
    )

    result["observation_rate"] = (
        result["observation_count"]
        / result.groupby(
            "strategy_name"
        )[
            "observation_count"
        ].transform("sum")
    )

    return result


def build_incremental_daily_detail(
    binary_results: Mapping[
        str,
        pd.DataFrame,
    ],
    graded_results: Mapping[
        str,
        pd.DataFrame,
    ],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    构建分级仓位相对原二元仓位的逐日增量收益。

    incremental_net_return > 0：
        当日分级仓位优于二元仓位。

    position_increment > 0：
        分级仓位比二元仓位持有更多仓位。
    """
    common_symbols = sorted(
        set(binary_results)
        & set(graded_results)
    )

    if not common_symbols:
        raise ValueError(
            "两组结果没有共同股票"
        )

    frames: list[
        pd.DataFrame
    ] = []

    columns = [
        "date",
        "symbol",
        "asset_return",
        "position",
        "position_change",
        "gross_strategy_return",
        "transaction_cost",
        "strategy_return",
        "slow_ma",
    ]

    for symbol in common_symbols:
        binary = _select_evaluation_data(
            result=binary_results[
                symbol
            ],
            start_date=start_date,
            end_date=end_date,
        )[columns]

        graded = _select_evaluation_data(
            result=graded_results[
                symbol
            ],
            start_date=start_date,
            end_date=end_date,
        )[columns]

        merged = binary.merge(
            graded,
            on=[
                "date",
                "symbol",
            ],
            how="inner",
            suffixes=(
                "_binary",
                "_graded",
            ),
            validate="one_to_one",
        )

        if merged.empty:
            continue

        binary_asset_return = (
            merged[
                "asset_return_binary"
            ].fillna(0.0)
        )

        graded_asset_return = (
            merged[
                "asset_return_graded"
            ].fillna(0.0)
        )

        if not np.allclose(
            binary_asset_return,
            graded_asset_return,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"{symbol} 两组回测的股票收益不一致"
            )

        merged["asset_return"] = (
            binary_asset_return
        )

        merged[
            "position_increment"
        ] = (
            merged[
                "position_graded"
            ]
            - merged[
                "position_binary"
            ]
        )

        merged[
            "incremental_gross_return"
        ] = (
            merged[
                "gross_strategy_return_graded"
            ]
            - merged[
                "gross_strategy_return_binary"
            ]
        )

        merged[
            "incremental_transaction_cost"
        ] = (
            merged[
                "transaction_cost_graded"
            ]
            - merged[
                "transaction_cost_binary"
            ]
        )

        merged[
            "incremental_net_return"
        ] = (
            merged[
                "strategy_return_graded"
            ]
            - merged[
                "strategy_return_binary"
            ]
        )

        merged[
            "incremental_log_return"
        ] = (
            np.log1p(
                merged[
                    "strategy_return_graded"
                ]
            )
            - np.log1p(
                merged[
                    "strategy_return_binary"
                ]
            )
        )

        tolerance = 1e-12

        added_exposure = (
            merged[
                "position_increment"
            ]
            > tolerance
        )

        reduced_exposure = (
            merged[
                "position_increment"
            ]
            < -tolerance
        )

        merged[
            "incremental_state"
        ] = np.select(
            condlist=[
                added_exposure
                & (
                    merged[
                        "asset_return"
                    ]
                    > 0
                ),
                added_exposure
                & (
                    merged[
                        "asset_return"
                    ]
                    < 0
                ),
                added_exposure,
                reduced_exposure,
            ],
            choicelist=[
                "added_exposure_up",
                "added_exposure_down",
                "added_exposure_flat",
                "reduced_exposure",
            ],
            default="same_exposure",
        )

        expected_incremental_gross = (
            merged[
                "position_increment"
            ]
            * merged[
                "asset_return"
            ]
        )

        if not np.allclose(
            merged[
                "incremental_gross_return"
            ],
            expected_incremental_gross,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"{symbol} 增量扣费前收益计算不一致"
            )

        expected_incremental_net = (
            merged[
                "incremental_gross_return"
            ]
            - merged[
                "incremental_transaction_cost"
            ]
        )

        if not np.allclose(
            merged[
                "incremental_net_return"
            ],
            expected_incremental_net,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                f"{symbol} 增量扣费后收益计算不一致"
            )

        frames.append(merged)

    if not frames:
        raise ValueError(
            "没有可比较的逐日记录"
        )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_incremental_effect(
    incremental_detail: pd.DataFrame,
    group_columns: Sequence[str] = (
        "incremental_state",
    ),
) -> pd.DataFrame:
    """
    汇总分级仓位相对二元仓位的增量效果。
    """
    group_columns = list(
        group_columns
    )

    required_columns = {
        "date",
        "symbol",
        "asset_return",
        "position_increment",
        "incremental_gross_return",
        "incremental_transaction_cost",
        "incremental_net_return",
        "incremental_log_return",
        *group_columns,
    }

    _require_columns(
        incremental_detail,
        required_columns,
        "incremental_detail",
    )

    result = (
        incremental_detail.groupby(
            group_columns,
            as_index=False,
            dropna=False,
        )
        .agg(
            observation_count=(
                "date",
                "count",
            ),
            stock_count=(
                "symbol",
                "nunique",
            ),
            average_position_increment=(
                "position_increment",
                "mean",
            ),
            average_asset_return=(
                "asset_return",
                "mean",
            ),
            incremental_gross_return_sum=(
                "incremental_gross_return",
                "sum",
            ),
            incremental_transaction_cost_sum=(
                "incremental_transaction_cost",
                "sum",
            ),
            incremental_net_return_sum=(
                "incremental_net_return",
                "sum",
            ),
            incremental_log_return_sum=(
                "incremental_log_return",
                "sum",
            ),
            positive_incremental_day_rate=(
                "incremental_net_return",
                lambda values: float(
                    (values > 0).mean()
                ),
            ),
        )
    )

    return result.sort_values(
        "incremental_log_return_sum",
        ascending=False,
    ).reset_index(drop=True)


def compare_period_summaries(
    baseline_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    """
    对同一批股票的两种策略进行成对比较。

    所有 diff 均为：
    candidate - baseline
    """
    metric_columns = [
        "strategy_annual_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "strategy_calmar",
        "excess_annual_return",
        "exposure",
        "total_trade_count",
        "total_transaction_cost",
        "total_turnover",
        "partial_position_rate",
    ]

    required_columns = {
        "symbol",
        *metric_columns,
    }

    for name, data in {
        baseline_name: baseline_summary,
        candidate_name: candidate_summary,
    }.items():
        missing_columns = (
            required_columns
            - set(data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{name} 缺少字段："
                f"{sorted(missing_columns)}"
            )

    baseline = baseline_summary[
        [
            "symbol",
            *metric_columns,
        ]
    ].copy()

    candidate = candidate_summary[
        [
            "symbol",
            *metric_columns,
        ]
    ].copy()

    comparison = baseline.merge(
        candidate,
        on="symbol",
        how="inner",
        suffixes=(
            f"_{baseline_name}",
            f"_{candidate_name}",
        ),
        validate="one_to_one",
    )

    comparison[
        "baseline_name"
    ] = baseline_name

    comparison[
        "candidate_name"
    ] = candidate_name

    for metric in metric_columns:
        comparison[
            f"{metric}_diff"
        ] = (
            comparison[
                f"{metric}_{candidate_name}"
            ]
            - comparison[
                f"{metric}_{baseline_name}"
            ]
        )

    return comparison


def summarize_pair_comparison(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    汇总候选策略相对基准策略的改善情况。
    """
    result = {
        "baseline_name": (
            comparison[
                "baseline_name"
            ].iloc[0]
        ),
        "candidate_name": (
            comparison[
                "candidate_name"
            ].iloc[0]
        ),
        "stock_count": (
            comparison[
                "symbol"
            ].nunique()
        ),
        "avg_annual_return_diff": (
            comparison[
                "strategy_annual_return_diff"
            ].mean()
        ),
        "median_annual_return_diff": (
            comparison[
                "strategy_annual_return_diff"
            ].median()
        ),
        "annual_return_win_rate": (
            comparison[
                "strategy_annual_return_diff"
            ].gt(0).mean()
        ),
        "avg_sharpe_diff": (
            comparison[
                "strategy_sharpe_diff"
            ].mean()
        ),
        "median_sharpe_diff": (
            comparison[
                "strategy_sharpe_diff"
            ].median()
        ),
        "sharpe_win_rate": (
            comparison[
                "strategy_sharpe_diff"
            ].gt(0).mean()
        ),
        "avg_drawdown_diff": (
            comparison[
                "strategy_max_drawdown_diff"
            ].mean()
        ),
        "drawdown_win_rate": (
            comparison[
                "strategy_max_drawdown_diff"
            ].gt(0).mean()
        ),
        "avg_calmar_diff": (
            comparison[
                "strategy_calmar_diff"
            ].mean()
        ),
        "calmar_win_rate": (
            comparison[
                "strategy_calmar_diff"
            ].gt(0).mean()
        ),
        "avg_excess_return_diff": (
            comparison[
                "excess_annual_return_diff"
            ].mean()
        ),
        "excess_return_win_rate": (
            comparison[
                "excess_annual_return_diff"
            ].gt(0).mean()
        ),
        "avg_exposure_diff": (
            comparison[
                "exposure_diff"
            ].mean()
        ),
        "avg_trade_count_diff": (
            comparison[
                "total_trade_count_diff"
            ].mean()
        ),
        "avg_turnover_diff": (
            comparison[
                "total_turnover_diff"
            ].mean()
        ),
        "avg_transaction_cost_diff": (
            comparison[
                "total_transaction_cost_diff"
            ].mean()
        ),
    }

    return pd.DataFrame(
        [result]
    )


def plot_stock_nav_comparison(
    binary_result: pd.DataFrame,
    graded_result: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Figure, Axes]:
    """
    绘制单只股票的二元仓位、分级仓位和买入持有净值。
    """
    binary = _select_evaluation_data(
        binary_result,
        start_date,
        end_date,
    )

    graded = _select_evaluation_data(
        graded_result,
        start_date,
        end_date,
    )

    merged = binary[
        [
            "date",
            "symbol",
            "asset_return",
            "strategy_return",
        ]
    ].merge(
        graded[
            [
                "date",
                "symbol",
                "strategy_return",
            ]
        ],
        on=[
            "date",
            "symbol",
        ],
        suffixes=(
            "_binary",
            "_graded",
        ),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "没有可绘制的共同数据"
        )

    symbol = str(
        merged["symbol"].iloc[0]
    ).zfill(6)

    binary_nav = (
        1.0
        + merged[
            "strategy_return_binary"
        ]
    ).cumprod()

    graded_nav = (
        1.0
        + merged[
            "strategy_return_graded"
        ]
    ).cumprod()

    benchmark_nav = (
        1.0
        + merged[
            "asset_return"
        ].fillna(0.0)
    ).cumprod()

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        merged["date"],
        binary_nav,
        label="Binary 10/40",
    )

    axis.plot(
        merged["date"],
        graded_nav,
        label="Graded 10/40",
    )

    axis.plot(
        merged["date"],
        benchmark_nav,
        label="Buy and hold",
        linestyle="--",
    )

    axis.axhline(
        1.0,
        linewidth=1.0,
        alpha=0.5,
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Net asset value")

    axis.set_title(
        f"{symbol} graded-position comparison"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    return figure, axis


def _check_graded_backtest_result(
    data: pd.DataFrame,
    partial_position: float,
    commission_rate: float,
    slippage_rate: float,
) -> None:
    """
    检查仓位、成本和归因计算。
    """
    allowed_positions = np.array(
        [
            0.0,
            partial_position,
            1.0,
        ]
    )

    valid_position = (
        data["position"]
        .apply(
            lambda value: np.isclose(
                value,
                allowed_positions,
            ).any()
        )
    )

    if not valid_position.all():
        raise ValueError(
            "存在非法仓位"
        )

    expected_position = (
        data["signal"]
        .shift(1)
        .fillna(0.0)
    )

    if not np.allclose(
        data["position"],
        expected_position,
    ):
        raise ValueError(
            "position 与延迟后的 signal 不一致"
        )

    expected_cost = (
        data["position_change"]
        .abs()
        * (
            commission_rate
            + slippage_rate
        )
    )

    if not np.allclose(
        data["transaction_cost"],
        expected_cost,
    ):
        raise ValueError(
            "交易成本计算不一致"
        )

    expected_active_return = (
        data["timing_active_return"]
        + data["cost_drag_return"]
    )

    if not np.allclose(
        data["active_return"],
        expected_active_return,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "主动收益分解不一致"
        )

    expected_timing_return = (
        data["missed_upside_return"]
        + data[
            "avoided_downside_return"
        ]
    )

    if not np.allclose(
        data["timing_active_return"],
        expected_timing_return,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "择时收益分解不一致"
        )


def _select_evaluation_data(
    result: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "slow_ma",
    }

    _require_columns(
        result,
        required_columns,
        "result",
    )

    data = result.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    evaluation_mask = (
        data["slow_ma"]
        .shift(1)
        .notna()
    )

    if start_date is not None:
        evaluation_mask &= (
            data["date"]
            >= pd.Timestamp(start_date)
        )

    if end_date is not None:
        evaluation_mask &= (
            data["date"]
            <= pd.Timestamp(end_date)
        )

    evaluation_data = (
        data.loc[evaluation_mask]
        .copy()
        .reset_index(drop=True)
    )

    if evaluation_data.empty:
        raise ValueError(
            "评价区间为空"
        )

    return evaluation_data


def _classify_position_state(
    position: pd.Series,
    partial_position: float,
) -> pd.Series:
    values = pd.to_numeric(
        position,
        errors="raise",
    )

    full_mask = np.isclose(
        values,
        1.0,
    )

    partial_mask = np.isclose(
        values,
        partial_position,
    )

    return pd.Series(
        np.select(
            condlist=[
                full_mask,
                partial_mask,
            ],
            choicelist=[
                "full",
                "partial",
            ],
            default="cash",
        ),
        index=position.index,
        dtype="object",
    )


def _validate_positive_integer(
    value: int,
    parameter_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (
                int,
                np.integer,
            ),
        )
    ):
        raise TypeError(
            f"{parameter_name} 必须是整数"
        )

    if value <= 0:
        raise ValueError(
            f"{parameter_name} 必须大于 0"
        )


def _require_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} 缺少字段："
            f"{sorted(missing_columns)}"
        )


def _safe_mean(
    values: pd.Series,
) -> float:
    clean_values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if clean_values.empty:
        return np.nan

    return float(
        clean_values.mean()
    )