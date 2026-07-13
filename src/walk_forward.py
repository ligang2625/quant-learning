"""
MA 策略滚动窗口验证与 Walk-Forward Analysis。

本模块只提供可复用函数，不负责：

1. 保存 CSV；
2. 调用 plt.show()；
3. 编写学习过程；
4. 使用最终保留测试集反向调整参数。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def generate_walk_forward_windows(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_months: int = 36,
    test_months: int = 12,
    step_months: int | None = None,
    window_type: Literal[
        "rolling",
        "expanding",
    ] = "rolling",
    include_partial_test: bool = False,
) -> pd.DataFrame:
    """
    生成 Walk-Forward 训练和测试窗口。

    Parameters
    ----------
    start_date:
        第一折训练期开始日期。

    end_date:
        Walk-Forward 开发阶段结束日期。

    train_months:
        训练窗口长度，单位为月。

    test_months:
        测试窗口长度，单位为月。

    step_months:
        每折向前移动的月份。

        None 表示使用 test_months，使测试窗口默认不重叠。

    window_type:
        rolling:
            固定长度滚动训练窗口。

        expanding:
            固定训练起点，逐步扩大训练窗口。

    include_partial_test:
        最后剩余区间不足完整测试窗口时，
        是否保留该不完整测试窗口。

    Returns
    -------
    pd.DataFrame
        每一行表示一折训练和测试窗口。
    """
    start = pd.Timestamp(
        start_date
    ).normalize()

    end = pd.Timestamp(
        end_date
    ).normalize()

    if pd.isna(start) or pd.isna(end):
        raise ValueError(
            "start_date 和 end_date 必须是有效日期"
        )

    if start >= end:
        raise ValueError(
            "start_date 必须早于 end_date"
        )

    if train_months <= 0:
        raise ValueError(
            "train_months 必须大于 0"
        )

    if test_months <= 0:
        raise ValueError(
            "test_months 必须大于 0"
        )

    if window_type not in {
        "rolling",
        "expanding",
    }:
        raise ValueError(
            "window_type 必须是 "
            "'rolling' 或 'expanding'"
        )

    if step_months is None:
        step_months = test_months

    if step_months <= 0:
        raise ValueError(
            "step_months 必须大于 0"
        )

    first_train_start = start
    train_start = start

    train_end = (
        train_start
        + pd.DateOffset(
            months=train_months
        )
        - pd.Timedelta(days=1)
    )

    rows: list[dict[str, object]] = []
    fold_id = 1

    while True:
        test_start = (
            train_end
            + pd.Timedelta(days=1)
        )

        planned_test_end = (
            test_start
            + pd.DateOffset(
                months=test_months
            )
            - pd.Timedelta(days=1)
        )

        if test_start > end:
            break

        if (
            planned_test_end > end
            and not include_partial_test
        ):
            break

        test_end = min(
            planned_test_end,
            end,
        )

        rows.append(
            {
                "fold_id": fold_id,
                "window_type": window_type,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_months": train_months,
                "test_months": test_months,
                "step_months": step_months,
                "is_partial_test": (
                    planned_test_end > end
                ),
            }
        )

        fold_id += 1

        train_end = (
            train_end
            + pd.DateOffset(
                months=step_months
            )
        )

        if window_type == "rolling":
            train_start = (
                train_start
                + pd.DateOffset(
                    months=step_months
                )
            )
        else:
            train_start = first_train_start

    windows = pd.DataFrame(rows)

    if windows.empty:
        raise ValueError(
            "没有生成 Walk-Forward 窗口，"
            "请缩短训练期或扩大日期范围"
        )

    return windows


def load_processed_stock_data(
    stock_list: Iterable[str],
    processed_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    读取股票清洗数据。

    为保证不同折之间的股票池一致，
    缺少文件时直接抛出异常，不静默跳过。
    """
    data_dir = (
        Path(processed_dir)
        if processed_dir is not None
        else DEFAULT_PROCESSED_DIR
    )

    stock_data: dict[
        str,
        pd.DataFrame,
    ] = {}

    for raw_symbol in stock_list:
        symbol = str(
            raw_symbol
        ).zfill(6)

        file_path = (
            data_dir
            / f"{symbol}_clean.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"{symbol} 清洗文件不存在："
                f"{file_path}"
            )

        data = pd.read_csv(
            file_path,
            dtype={
                "symbol": str,
            },
        )

        required_columns = {
            "date",
            "close",
        }

        missing_columns = (
            required_columns
            - set(data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{symbol} 缺少字段："
                f"{sorted(missing_columns)}"
            )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        data["close"] = pd.to_numeric(
            data["close"],
            errors="coerce",
        )

        data = (
            data
            .dropna(
                subset=[
                    "date",
                    "close",
                ]
            )
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        if data.empty:
            raise ValueError(
                f"{symbol} 没有有效数据"
            )

        if (data["close"] <= 0).any():
            raise ValueError(
                f"{symbol} close 必须全部大于 0"
            )

        data["symbol"] = symbol
        stock_data[symbol] = data

    if not stock_data:
        raise ValueError(
            "stock_list 不能为空"
        )

    return stock_data


def resolve_common_date_range(
    stock_data: dict[
        str,
        pd.DataFrame,
    ],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[
    pd.Timestamp,
    pd.Timestamp,
]:
    """
    计算所有股票共有的数据区间。

    这样可以避免不同 Walk-Forward 折
    覆盖不同股票数量。
    """
    if not stock_data:
        raise ValueError(
            "stock_data 不能为空"
        )

    common_start = max(
        data["date"].min()
        for data in stock_data.values()
    )

    common_end = min(
        data["date"].max()
        for data in stock_data.values()
    )

    resolved_start = (
        pd.Timestamp(
            start_date
        ).normalize()
        if start_date is not None
        else pd.Timestamp(
            common_start
        ).normalize()
    )

    resolved_end = (
        pd.Timestamp(
            end_date
        ).normalize()
        if end_date is not None
        else pd.Timestamp(
            common_end
        ).normalize()
    )

    if resolved_start < common_start:
        raise ValueError(
            "start_date 早于股票池共同数据起点："
            f"{common_start.date()}"
        )

    if resolved_end > common_end:
        raise ValueError(
            "end_date 晚于股票池共同数据终点："
            f"{common_end.date()}"
        )

    if resolved_start >= resolved_end:
        raise ValueError(
            "有效日期范围不足"
        )

    return (
        resolved_start,
        resolved_end,
    )


def precompute_ma_backtests(
    stock_data: dict[
        str,
        pd.DataFrame,
    ],
    parameter_grid: Iterable[
        tuple[int, int]
    ],
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
) -> dict[
    tuple[str, int, int],
    pd.DataFrame,
]:
    """
    预先运行所有股票和参数的完整历史回测。

    MA 和仓位计算只依赖当前及过去数据，
    因此可以先生成完整因果回测结果，
    再按每折训练期和测试期汇总。
    """
    ma_cross_backtest, _ = (
        _import_backtest_functions()
    )

    normalized_grid = (
        _normalize_parameter_grid(
            parameter_grid
        )
    )

    results: dict[
        tuple[str, int, int],
        pd.DataFrame,
    ] = {}

    for (
        fast_window,
        slow_window,
    ) in normalized_grid:
        for (
            symbol,
            stock_df,
        ) in stock_data.items():
            result = ma_cross_backtest(
                df=stock_df,
                fast_window=fast_window,
                slow_window=slow_window,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            )

            results[
                (
                    symbol,
                    fast_window,
                    slow_window,
                )
            ] = result

    return results


def run_ma_walk_forward(
    stock_list: Iterable[str],
    parameter_grid: Iterable[
        tuple[int, int]
    ],
    train_months: int = 36,
    test_months: int = 12,
    step_months: int | None = None,
    window_type: Literal[
        "rolling",
        "expanding",
    ] = "rolling",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    include_partial_test: bool = False,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
    robustness_metric: str = (
        "avg_strategy_sharpe"
    ),
    fast_radius: int | None = None,
    slow_radius: int | None = None,
    stability_penalty: float = 1.0,
    min_local_parameter_count: int = 9,
    processed_dir: str | Path | None = None,
    return_backtests: bool = False,
) -> dict[str, object]:
    """
    运行 MA Walk-Forward Analysis。

    每一折执行：

    1. 在训练期评价全部参数；
    2. 计算训练期参数局部稳健性；
    3. 选择训练期稳健参数；
    4. 在下一段测试期评价该参数。
    """
    parameter_grid = (
        _normalize_parameter_grid(
            parameter_grid
        )
    )

    stock_data = (
        load_processed_stock_data(
            stock_list=stock_list,
            processed_dir=processed_dir,
        )
    )

    (
        resolved_start,
        resolved_end,
    ) = resolve_common_date_range(
        stock_data=stock_data,
        start_date=start_date,
        end_date=end_date,
    )

    windows = (
        generate_walk_forward_windows(
            start_date=resolved_start,
            end_date=resolved_end,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            window_type=window_type,
            include_partial_test=(
                include_partial_test
            ),
        )
    )

    backtests = precompute_ma_backtests(
        stock_data=stock_data,
        parameter_grid=parameter_grid,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    (
        _,
        summarize_backtest_period,
    ) = _import_backtest_functions()

    (
        aggregate_ma_parameter_results,
        calculate_local_parameter_robustness,
        select_robust_ma_parameter,
    ) = _import_sensitivity_functions()

    train_detail_frames = []
    train_summary_frames = []
    train_robustness_frames = []

    selection_rows = []
    test_rows = []

    for window in windows.itertuples(
        index=False
    ):
        fold_train_rows = []

        for (
            fast_window,
            slow_window,
        ) in parameter_grid:
            for symbol in stock_data:
                result = backtests[
                    (
                        symbol,
                        fast_window,
                        slow_window,
                    )
                ]

                summary = (
                    summarize_backtest_period(
                        result=result,
                        period_name=(
                            f"fold_"
                            f"{window.fold_id}"
                            f"_train"
                        ),
                        start_date=(
                            window.train_start
                        ),
                        end_date=(
                            window.train_end
                        ),
                        annual_risk_free_rate=(
                            annual_risk_free_rate
                        ),
                        trading_days=trading_days,
                    )
                )

                summary.update(
                    {
                        "fold_id": (
                            window.fold_id
                        ),
                        "train_start": (
                            window.train_start
                        ),
                        "train_end": (
                            window.train_end
                        ),
                        "test_start": (
                            window.test_start
                        ),
                        "test_end": (
                            window.test_end
                        ),
                        "fast_window": (
                            fast_window
                        ),
                        "slow_window": (
                            slow_window
                        ),
                        "ma_param": (
                            f"{fast_window}/"
                            f"{slow_window}"
                        ),
                    }
                )

                fold_train_rows.append(
                    summary
                )

        fold_train_detail = pd.DataFrame(
            fold_train_rows
        )

        fold_parameter_summary = (
            aggregate_ma_parameter_results(
                fold_train_detail
            )
        )

        fold_robustness = (
            calculate_local_parameter_robustness(
                parameter_summary=(
                    fold_parameter_summary
                ),
                metric=robustness_metric,
                fast_radius=fast_radius,
                slow_radius=slow_radius,
                stability_penalty=(
                    stability_penalty
                ),
            )
        )

        (
            selected_fast,
            selected_slow,
        ) = select_robust_ma_parameter(
            robustness_summary=(
                fold_robustness
            ),
            min_local_parameter_count=(
                min_local_parameter_count
            ),
        )

        metadata = {
            "fold_id": window.fold_id,
            "window_type": (
                window.window_type
            ),
            "train_start": (
                window.train_start
            ),
            "train_end": (
                window.train_end
            ),
            "test_start": (
                window.test_start
            ),
            "test_end": (
                window.test_end
            ),
        }

        fold_parameter_summary = (
            fold_parameter_summary.assign(
                **metadata
            )
        )

        fold_robustness = (
            fold_robustness.assign(
                **metadata
            )
        )

        selected_record = (
            fold_robustness.loc[
                (
                    fold_robustness[
                        "fast_window"
                    ]
                    == selected_fast
                )
                & (
                    fold_robustness[
                        "slow_window"
                    ]
                    == selected_slow
                )
            ]
            .iloc[0]
        )

        selection_rows.append(
            {
                **metadata,
                "selected_fast_window": (
                    selected_fast
                ),
                "selected_slow_window": (
                    selected_slow
                ),
                "selected_ma_param": (
                    f"{selected_fast}/"
                    f"{selected_slow}"
                ),
                "train_avg_strategy_sharpe": (
                    selected_record[
                        "avg_strategy_sharpe"
                    ]
                ),
                "train_avg_excess_annual_return": (
                    selected_record[
                        "avg_excess_annual_return"
                    ]
                ),
                "train_return_win_rate": (
                    selected_record[
                        "return_win_rate"
                    ]
                ),
                "train_local_mean": (
                    selected_record[
                        "local_mean"
                    ]
                ),
                "train_local_std": (
                    selected_record[
                        "local_std"
                    ]
                ),
                "train_local_min": (
                    selected_record[
                        "local_min"
                    ]
                ),
                "train_robustness_score": (
                    selected_record[
                        "robustness_score"
                    ]
                ),
            }
        )

        for symbol in stock_data:
            selected_result = backtests[
                (
                    symbol,
                    selected_fast,
                    selected_slow,
                )
            ]

            test_summary = (
                summarize_backtest_period(
                    result=selected_result,
                    period_name=(
                        f"fold_"
                        f"{window.fold_id}"
                        f"_test"
                    ),
                    start_date=(
                        window.test_start
                    ),
                    end_date=(
                        window.test_end
                    ),
                    annual_risk_free_rate=(
                        annual_risk_free_rate
                    ),
                    trading_days=trading_days,
                )
            )

            test_summary.update(
                {
                    **metadata,
                    "selected_fast_window": (
                        selected_fast
                    ),
                    "selected_slow_window": (
                        selected_slow
                    ),
                    "selected_ma_param": (
                        f"{selected_fast}/"
                        f"{selected_slow}"
                    ),
                }
            )

            test_rows.append(
                test_summary
            )

        train_detail_frames.append(
            fold_train_detail
        )

        train_summary_frames.append(
            fold_parameter_summary
        )

        train_robustness_frames.append(
            fold_robustness
        )

    train_detail = pd.concat(
        train_detail_frames,
        ignore_index=True,
    )

    train_parameter_summary = pd.concat(
        train_summary_frames,
        ignore_index=True,
    )

    train_robustness_summary = pd.concat(
        train_robustness_frames,
        ignore_index=True,
    )

    selected_parameters = pd.DataFrame(
        selection_rows
    )

    test_detail = pd.DataFrame(
        test_rows
    )

    test_summary = (
        aggregate_walk_forward_test_results(
            test_detail=test_detail,
            parameter_column=(
                "selected_ma_param"
            ),
        )
    )

    parameter_frequency = (
        summarize_parameter_selection(
            selected_parameters
        )
    )

    output: dict[str, object] = {
        "windows": windows,
        "train_detail": train_detail,
        "train_parameter_summary": (
            train_parameter_summary
        ),
        "train_robustness_summary": (
            train_robustness_summary
        ),
        "selected_parameters": (
            selected_parameters
        ),
        "parameter_frequency": (
            parameter_frequency
        ),
        "test_detail": test_detail,
        "test_summary": test_summary,
    }

    if return_backtests:
        output["backtests"] = backtests

    return output


def run_fixed_parameter_validation(
    stock_list: Iterable[str],
    parameters: Iterable[
        tuple[int, int]
    ],
    windows: pd.DataFrame,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
    processed_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    在完全相同的测试窗口中评价固定参数。

    主要用于同期比较：

    - 10/40；
    - 20/60；
    - 每折动态选择的 Walk-Forward 参数。

    注意：
    固定 10/40 的历史滚动结果属于稳定性诊断，
    不能代替最终未使用数据上的独立验证。
    """
    required_columns = {
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    }

    _check_required_columns(
        data=windows,
        required_columns=(
            required_columns
        ),
        data_name="windows",
    )

    parameters = (
        _normalize_parameter_grid(
            parameters
        )
    )

    stock_data = (
        load_processed_stock_data(
            stock_list=stock_list,
            processed_dir=processed_dir,
        )
    )

    backtests = precompute_ma_backtests(
        stock_data=stock_data,
        parameter_grid=parameters,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    (
        _,
        summarize_backtest_period,
    ) = _import_backtest_functions()

    rows = []

    for (
        fast_window,
        slow_window,
    ) in parameters:
        ma_param = (
            f"{fast_window}/"
            f"{slow_window}"
        )

        for window in windows.itertuples(
            index=False
        ):
            for symbol in stock_data:
                result = backtests[
                    (
                        symbol,
                        fast_window,
                        slow_window,
                    )
                ]

                summary = (
                    summarize_backtest_period(
                        result=result,
                        period_name=(
                            f"fold_"
                            f"{window.fold_id}"
                            f"_fixed_test"
                        ),
                        start_date=(
                            window.test_start
                        ),
                        end_date=(
                            window.test_end
                        ),
                        annual_risk_free_rate=(
                            annual_risk_free_rate
                        ),
                        trading_days=trading_days,
                    )
                )

                summary.update(
                    {
                        "fold_id": (
                            window.fold_id
                        ),
                        "train_start": (
                            window.train_start
                        ),
                        "train_end": (
                            window.train_end
                        ),
                        "test_start": (
                            window.test_start
                        ),
                        "test_end": (
                            window.test_end
                        ),
                        "fast_window": (
                            fast_window
                        ),
                        "slow_window": (
                            slow_window
                        ),
                        "ma_param": (
                            ma_param
                        ),
                    }
                )

                rows.append(summary)

    detail = pd.DataFrame(rows)

    summary = (
        aggregate_walk_forward_test_results(
            test_detail=detail,
            parameter_column="ma_param",
        )
    )

    return {
        "detail": detail,
        "summary": summary,
    }


def aggregate_walk_forward_test_results(
    test_detail: pd.DataFrame,
    parameter_column: str,
) -> pd.DataFrame:
    """
    将每折、每只股票的测试结果汇总为：

        每折 × 每组参数一行
    """
    required_columns = {
        "fold_id",
        "test_start",
        "test_end",
        parameter_column,
        "symbol",
        "strategy_annual_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "excess_annual_return",
        "sharpe_diff",
        "drawdown_improvement",
        "total_trade_count",
    }

    _check_required_columns(
        data=test_detail,
        required_columns=(
            required_columns
        ),
        data_name="test_detail",
    )

    group_columns = [
        "fold_id",
        "test_start",
        "test_end",
        parameter_column,
    ]

    summary = (
        test_detail
        .groupby(
            group_columns,
            as_index=False,
        )
        .agg(
            stock_count=(
                "symbol",
                "nunique",
            ),
            avg_strategy_annual_return=(
                "strategy_annual_return",
                "mean",
            ),
            median_strategy_annual_return=(
                "strategy_annual_return",
                "median",
            ),
            avg_strategy_sharpe=(
                "strategy_sharpe",
                "mean",
            ),
            median_strategy_sharpe=(
                "strategy_sharpe",
                "median",
            ),
            avg_strategy_max_drawdown=(
                "strategy_max_drawdown",
                "mean",
            ),
            avg_excess_annual_return=(
                "excess_annual_return",
                "mean",
            ),
            median_excess_annual_return=(
                "excess_annual_return",
                "median",
            ),
            return_win_rate=(
                "excess_annual_return",
                _positive_rate,
            ),
            avg_sharpe_diff=(
                "sharpe_diff",
                "mean",
            ),
            sharpe_win_rate=(
                "sharpe_diff",
                _positive_rate,
            ),
            avg_drawdown_improvement=(
                "drawdown_improvement",
                "mean",
            ),
            drawdown_win_rate=(
                "drawdown_improvement",
                _positive_rate,
            ),
            avg_trade_count=(
                "total_trade_count",
                "mean",
            ),
        )
        .sort_values(
            [
                "fold_id",
                parameter_column,
            ]
        )
        .reset_index(drop=True)
    )

    return summary


def summarize_parameter_selection(
    selected_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """
    统计各组参数被训练窗口选中的次数。
    """
    required_columns = {
        "fold_id",
        "selected_fast_window",
        "selected_slow_window",
        "selected_ma_param",
    }

    _check_required_columns(
        data=selected_parameters,
        required_columns=(
            required_columns
        ),
        data_name="selected_parameters",
    )

    total_folds = (
        selected_parameters[
            "fold_id"
        ].nunique()
    )

    return (
        selected_parameters
        .groupby(
            [
                "selected_fast_window",
                "selected_slow_window",
                "selected_ma_param",
            ],
            as_index=False,
        )
        .agg(
            selected_count=(
                "fold_id",
                "count",
            ),
            first_selected_fold=(
                "fold_id",
                "min",
            ),
            last_selected_fold=(
                "fold_id",
                "max",
            ),
        )
        .assign(
            selected_rate=lambda data: (
                data["selected_count"]
                / total_folds
            )
        )
        .sort_values(
            [
                "selected_count",
                "selected_fast_window",
                "selected_slow_window",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def plot_walk_forward_metric(
    summary: pd.DataFrame,
    metric: str,
    parameter_column: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
) -> tuple[Figure, Axes]:
    """
    绘制每折测试指标。

    parameter_column=None:
        绘制单一 Walk-Forward 序列。

    parameter_column="ma_param":
        对比多组固定参数。
    """
    required_columns = {
        "fold_id",
        metric,
    }

    if parameter_column is not None:
        required_columns.add(
            parameter_column
        )

    _check_required_columns(
        data=summary,
        required_columns=(
            required_columns
        ),
        data_name="summary",
    )

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    if parameter_column is None:
        ordered = summary.sort_values(
            "fold_id"
        )

        axis.plot(
            ordered["fold_id"],
            ordered[metric],
            marker="o",
        )

    else:
        for (
            parameter,
            group,
        ) in summary.groupby(
            parameter_column
        ):
            ordered = group.sort_values(
                "fold_id"
            )

            axis.plot(
                ordered["fold_id"],
                ordered[metric],
                marker="o",
                label=str(parameter),
            )

        axis.legend(
            title="参数"
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
        alpha=0.5,
    )

    axis.set_xlabel(
        "Walk-Forward 折数"
    )

    axis.set_ylabel(
        ylabel or metric
    )

    axis.set_title(
        title
        or f"Walk-Forward：{metric}"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    return figure, axis


def plot_selected_parameters(
    selected_parameters: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """
    绘制每折训练期选择的快慢均线。
    """
    required_columns = {
        "fold_id",
        "selected_fast_window",
        "selected_slow_window",
    }

    _check_required_columns(
        data=selected_parameters,
        required_columns=(
            required_columns
        ),
        data_name="selected_parameters",
    )

    ordered = (
        selected_parameters
        .sort_values("fold_id")
    )

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    axis.plot(
        ordered["fold_id"],
        ordered[
            "selected_fast_window"
        ],
        marker="o",
        label="快均线",
    )

    axis.plot(
        ordered["fold_id"],
        ordered[
            "selected_slow_window"
        ],
        marker="o",
        label="慢均线",
    )

    axis.set_xlabel(
        "Walk-Forward 折数"
    )

    axis.set_ylabel(
        "均线窗口"
    )

    axis.set_title(
        "各折训练期选择的 MA 参数"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    return figure, axis


def _normalize_parameter_grid(
    parameter_grid: Iterable[
        tuple[int, int]
    ],
) -> list[tuple[int, int]]:
    normalized = set()

    for parameter in parameter_grid:
        if len(parameter) != 2:
            raise ValueError(
                "每组参数必须包含快线和慢线"
            )

        fast_window = int(
            parameter[0]
        )

        slow_window = int(
            parameter[1]
        )

        if (
            fast_window <= 0
            or slow_window <= 0
        ):
            raise ValueError(
                "均线窗口必须大于 0"
            )

        if fast_window >= slow_window:
            raise ValueError(
                "非法参数："
                f"{fast_window}/"
                f"{slow_window}"
            )

        normalized.add(
            (
                fast_window,
                slow_window,
            )
        )

    if not normalized:
        raise ValueError(
            "parameter_grid 不能为空"
        )

    return sorted(normalized)


def _positive_rate(
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
        (clean_values > 0).mean()
    )


def _check_required_columns(
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


def _import_backtest_functions():
    """
    同时兼容包导入和 notebook 将 src
    加入 sys.path 后的直接导入。
    """
    try:
        from .backtest import (
            ma_cross_backtest,
            summarize_backtest_period,
        )
    except ImportError:
        from backtest import (
            ma_cross_backtest,
            summarize_backtest_period,
        )

    return (
        ma_cross_backtest,
        summarize_backtest_period,
    )


def _import_sensitivity_functions():
    try:
        from .parameter_sensitivity import (
            aggregate_ma_parameter_results,
            calculate_local_parameter_robustness,
            select_robust_ma_parameter,
        )
    except ImportError:
        from parameter_sensitivity import (
            aggregate_ma_parameter_results,
            calculate_local_parameter_robustness,
            select_robust_ma_parameter,
        )

    return (
        aggregate_ma_parameter_results,
        calculate_local_parameter_robustness,
        select_robust_ma_parameter,
    )