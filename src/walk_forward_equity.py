"""
连续 Walk-Forward 净值构建与整体绩效评估。

主要功能：

1. 根据每折参数计划提取测试期逐日结果；
2. 在测试折边界按真实上一折末仓位重算交易成本；
3. 拼接连续样本外日收益；
4. 构建固定初始等权、多股票独立复利的组合净值；
5. 计算整体绩效和年度绩效；
6. 绘制净值与回撤曲线。
7.进行参数外部验证

本模块不负责：

1. 训练期参数搜索；
2. 保存 CSV 或图片；
3. 调用 plt.show()。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


Parameter = tuple[int, int]
BacktestKey = tuple[str, int, int]


def build_dynamic_schedule(
    selected_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """
    将每折选参结果整理为参数计划。

    selected_parameters 通常来自：
        run_ma_walk_forward()["selected_parameters"]
    """
    _require_columns(
        selected_parameters,
        {
            "fold_id",
            "selected_fast_window",
            "selected_slow_window",
        },
        "selected_parameters",
    )

    if selected_parameters.empty:
        raise ValueError("selected_parameters 不能为空")

    schedule = selected_parameters[
        [
            "fold_id",
            "selected_fast_window",
            "selected_slow_window",
        ]
    ].rename(
        columns={
            "selected_fast_window": "fast_window",
            "selected_slow_window": "slow_window",
        }
    )

    return _prepare_schedule(schedule)


def build_fixed_schedule(
    windows: pd.DataFrame,
    parameter: Parameter,
) -> pd.DataFrame:
    """
    为全部测试折生成同一组固定参数计划。
    """
    windows = _prepare_windows(windows)

    fast_window, slow_window = _normalize_parameter(
        parameter
    )

    schedule = windows[["fold_id"]].copy()

    schedule["fast_window"] = fast_window
    schedule["slow_window"] = slow_window

    return _prepare_schedule(schedule)


def build_continuous_stock_detail(
    backtests: Mapping[
        BacktestKey,
        pd.DataFrame,
    ],
    windows: pd.DataFrame,
    schedule: pd.DataFrame,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    initial_position: float = 0.0,
) -> pd.DataFrame:
    """
    拼接每只股票的连续测试期逐日结果。

    每折内部直接复用完整历史回测中的因果仓位。

    每折首个交易日需要重新计算：

        position_change
        transaction_cost
        gross_strategy_return
        strategy_return

    原因是新参数完整回测中的上一日仓位，
    不一定等于动态策略上一折的真实末仓位。

    第一折默认从 initial_position=0 开始。
    """
    windows = _prepare_windows(windows)

    schedule = _prepare_schedule(
        schedule=schedule,
        windows=windows,
    )

    backtests = _prepare_backtests(
        backtests
    )

    if commission_rate < 0:
        raise ValueError(
            "commission_rate 不能为负数"
        )

    if slippage_rate < 0:
        raise ValueError(
            "slippage_rate 不能为负数"
        )

    if not 0 <= initial_position <= 1:
        raise ValueError(
            "initial_position 必须位于 0 到 1 之间"
        )

    one_way_cost = (
        commission_rate
        + slippage_rate
    )

    symbols = sorted(
        {
            key[0]
            for key in backtests
        }
    )

    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        previous_position = float(
            initial_position
        )

        previous_date: pd.Timestamp | None = None
        previous_param: str | None = None

        for window in windows.itertuples(
            index=False
        ):
            selected = schedule.loc[
                schedule["fold_id"]
                == window.fold_id
            ].iloc[0]

            fast_window = int(
                selected["fast_window"]
            )

            slow_window = int(
                selected["slow_window"]
            )

            ma_param = str(
                selected["ma_param"]
            )

            key = (
                symbol,
                fast_window,
                slow_window,
            )

            if key not in backtests:
                raise KeyError(
                    f"缺少回测结果：{key}"
                )

            fold = backtests[key].loc[
                lambda data: data[
                    "date"
                ].between(
                    window.test_start,
                    window.test_end,
                    inclusive="both",
                )
            ].copy()

            if fold.empty:
                raise ValueError(
                    f"{symbol} 第 "
                    f"{window.fold_id} 折"
                    "测试期没有数据"
                )

            fold = (
                fold.sort_values("date")
                .reset_index(drop=True)
            )

            first_date = pd.Timestamp(
                fold.loc[0, "date"]
            )

            if (
                previous_date is not None
                and first_date <= previous_date
            ):
                raise ValueError(
                    f"{symbol} 测试日期发生重叠或倒序"
                )

            # 保存完整参数回测原本的边界计算，
            # 方便之后检查参数切换带来的差异。
            fold[
                "source_position_change"
            ] = fold[
                "position_change"
            ]

            fold[
                "source_transaction_cost"
            ] = fold[
                "transaction_cost"
            ]

            first_position = float(
                fold.loc[
                    0,
                    "position",
                ]
            )

            first_asset_return = float(
                pd.to_numeric(
                    pd.Series(
                        [
                            fold.loc[
                                0,
                                "asset_return",
                            ]
                        ]
                    ),
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[0]
            )

            boundary_change = (
                first_position
                - previous_position
            )

            boundary_cost = (
                abs(boundary_change)
                * one_way_cost
            )

            boundary_gross_return = (
                first_position
                * first_asset_return
            )

            fold.loc[
                0,
                "position_change",
            ] = boundary_change

            fold.loc[
                0,
                "transaction_cost",
            ] = boundary_cost

            fold.loc[
                0,
                "gross_strategy_return",
            ] = boundary_gross_return

            fold.loc[
                0,
                "strategy_return",
            ] = (
                boundary_gross_return
                - boundary_cost
            )

            fold["fold_id"] = int(
                window.fold_id
            )

            fold["test_start"] = (
                window.test_start
            )

            fold["test_end"] = (
                window.test_end
            )

            fold[
                "selected_fast_window"
            ] = fast_window

            fold[
                "selected_slow_window"
            ] = slow_window

            fold[
                "selected_ma_param"
            ] = ma_param

            fold["is_fold_start"] = False

            fold.loc[
                0,
                "is_fold_start",
            ] = True

            fold[
                "is_parameter_switch"
            ] = False

            if previous_param is not None:
                fold.loc[
                    0,
                    "is_parameter_switch",
                ] = (
                    previous_param
                    != ma_param
                )

            fold[
                "boundary_cost_adjustment"
            ] = (
                fold[
                    "transaction_cost"
                ]
                - fold[
                    "source_transaction_cost"
                ]
            )

            frames.append(fold)

            previous_position = float(
                fold[
                    "position"
                ].iloc[-1]
            )

            previous_date = pd.Timestamp(
                fold[
                    "date"
                ].iloc[-1]
            )

            previous_param = ma_param

    detail = pd.concat(
        frames,
        ignore_index=True,
    )

    detail["symbol"] = (
        detail["symbol"]
        .astype(str)
        .str.zfill(6)
    )

    detail = (
        detail.sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    if detail.duplicated(
        [
            "symbol",
            "date",
        ]
    ).any():
        raise ValueError(
            "连续明细中存在重复股票日期"
        )

    expected_strategy_return = (
        detail[
            "gross_strategy_return"
        ]
        - detail[
            "transaction_cost"
        ]
    )

    if not np.allclose(
        detail[
            "strategy_return"
        ],
        expected_strategy_return,
    ):
        raise ValueError(
            "strategy_return 与扣费后收益不一致"
        )

    detail[
        "stock_strategy_nav"
    ] = detail.groupby(
        "symbol"
    )[
        "strategy_return"
    ].transform(
        lambda values: (
            1.0 + values
        ).cumprod()
    )

    detail[
        "stock_benchmark_nav"
    ] = detail.groupby(
        "symbol"
    )[
        "asset_return"
    ].transform(
        lambda values: (
            1.0
            + values.fillna(0.0)
        ).cumprod()
    )

    return detail


def build_portfolio(
    stock_detail: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建固定初始等权组合。

    假设：

    1. 连续测试开始时，每只股票分配相同资金；
    2. 各股票子账户独立复利；
    3. 股票之间不进行每日等权再平衡；
    4. 组合净值等于各子账户净值的平均值。

    这种方式不会隐含额外的每日再平衡交易。
    """
    _require_columns(
        stock_detail,
        {
            "date",
            "symbol",
            "strategy_return",
            "asset_return",
            "position",
            "position_change",
            "transaction_cost",
            "boundary_cost_adjustment",
            "fold_id",
        },
        "stock_detail",
    )

    if stock_detail.empty:
        raise ValueError(
            "stock_detail 不能为空"
        )

    data = stock_detail.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.zfill(6)
    )

    if data.duplicated(
        [
            "symbol",
            "date",
        ]
    ).any():
        raise ValueError(
            "stock_detail 存在重复股票日期"
        )

    symbols = sorted(
        data["symbol"].unique()
    )

    strategy_returns = _pivot(
        data=data,
        column="strategy_return",
        symbols=symbols,
        fill="zero",
    )

    benchmark_returns = _pivot(
        data=data,
        column="asset_return",
        symbols=symbols,
        fill="zero",
    )

    positions = _pivot(
        data=data,
        column="position",
        symbols=symbols,
        fill="forward",
    )

    changes = _pivot(
        data=data,
        column="position_change",
        symbols=symbols,
        fill="zero",
    )

    costs = _pivot(
        data=data,
        column="transaction_cost",
        symbols=symbols,
        fill="zero",
    )

    boundary_adjustments = _pivot(
        data=data,
        column="boundary_cost_adjustment",
        symbols=symbols,
        fill="zero",
    )

    # 每只股票子账户在当日收益发生前的净值。
    strategy_prior_nav = (
        (
            1.0 + strategy_returns
        )
        .cumprod()
        .shift(1)
        .fillna(1.0)
    )

    benchmark_prior_nav = (
        (
            1.0 + benchmark_returns
        )
        .cumprod()
        .shift(1)
        .fillna(1.0)
    )

    strategy_sleeve_nav = (
        strategy_prior_nav
        * (
            1.0
            + strategy_returns
        )
    )

    benchmark_sleeve_nav = (
        benchmark_prior_nav
        * (
            1.0
            + benchmark_returns
        )
    )

    strategy_nav = (
        strategy_sleeve_nav
        .mean(axis=1)
    )

    benchmark_nav = (
        benchmark_sleeve_nav
        .mean(axis=1)
    )

    strategy_return = _returns_from_nav(
        strategy_nav
    )

    benchmark_return = _returns_from_nav(
        benchmark_nav
    )

    denominator = (
        strategy_prior_nav
        .sum(axis=1)
    )

    weighted_position = (
        (
            strategy_prior_nav
            * positions
        )
        .sum(axis=1)
        / denominator
    )

    weighted_turnover = (
        (
            strategy_prior_nav
            * changes.abs()
        )
        .sum(axis=1)
        / denominator
    )

    weighted_cost = (
        (
            strategy_prior_nav
            * costs
        )
        .sum(axis=1)
        / denominator
    )

    weighted_boundary_adjustment = (
        (
            strategy_prior_nav
            * boundary_adjustments
        )
        .sum(axis=1)
        / denominator
    )

    fold_by_date = (
        data.groupby("date")[
            "fold_id"
        ]
        .agg(
            lambda values: (
                values.mode().iloc[0]
            )
        )
    )

    portfolio = pd.DataFrame(
        {
            "date": strategy_nav.index,
            "fold_id": (
                strategy_nav.index
                .map(fold_by_date)
                .astype(int)
            ),
            "strategy_return": (
                strategy_return.values
            ),
            "benchmark_return": (
                benchmark_return.values
            ),
            "active_return": (
                strategy_return.values
                - benchmark_return.values
            ),
            "strategy_nav": (
                strategy_nav.values
            ),
            "benchmark_nav": (
                benchmark_nav.values
            ),
            "portfolio_position": (
                weighted_position.values
            ),
            "portfolio_turnover": (
                weighted_turnover.values
            ),
            "portfolio_transaction_cost": (
                weighted_cost.values
            ),
            "portfolio_boundary_cost_adjustment": (
                weighted_boundary_adjustment.values
            ),
            "trade_event_count": (
                changes.abs()
                .gt(0)
                .sum(axis=1)
                .astype(int)
                .values
            ),
            "stock_count": len(symbols),
        }
    )

    portfolio[
        "strategy_drawdown"
    ] = (
        portfolio[
            "strategy_nav"
        ]
        / portfolio[
            "strategy_nav"
        ].cummax()
        - 1.0
    )

    portfolio[
        "benchmark_drawdown"
    ] = (
        portfolio[
            "benchmark_nav"
        ]
        / portfolio[
            "benchmark_nav"
        ].cummax()
        - 1.0
    )

    return portfolio


def summarize_portfolio(
    portfolio: pd.DataFrame,
    strategy_name: str,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, Any]:
    """
    计算连续组合的整体绩效。
    """
    _require_columns(
        portfolio,
        {
            "date",
            "strategy_return",
            "benchmark_return",
            "portfolio_position",
            "portfolio_turnover",
            "portfolio_transaction_cost",
            "portfolio_boundary_cost_adjustment",
            "trade_event_count",
        },
        "portfolio",
    )

    calculate_performance = (
        _import_calculate_performance()
    )

    strategy = calculate_performance(
        returns=portfolio[
            "strategy_return"
        ],
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
        trading_days=trading_days,
    )

    benchmark = calculate_performance(
        returns=portfolio[
            "benchmark_return"
        ],
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
        trading_days=trading_days,
    )

    active_returns = (
        portfolio[
            "strategy_return"
        ]
        - portfolio[
            "benchmark_return"
        ]
    )

    tracking_error = (
        active_returns.std(ddof=1)
        * np.sqrt(trading_days)
    )

    if (
        pd.isna(tracking_error)
        or np.isclose(
            tracking_error,
            0.0,
        )
    ):
        information_ratio = np.nan
    else:
        information_ratio = (
            active_returns.mean()
            * trading_days
            / tracking_error
        )

    return {
        "strategy_name": strategy_name,
        "start_date": pd.Timestamp(
            portfolio[
                "date"
            ].iloc[0]
        ),
        "end_date": pd.Timestamp(
            portfolio[
                "date"
            ].iloc[-1]
        ),
        "trade_days": int(
            len(portfolio)
        ),
        "strategy_cumulative_return": (
            strategy[
                "cumulative_return"
            ]
        ),
        "strategy_annual_return": (
            strategy[
                "annual_return"
            ]
        ),
        "strategy_annual_volatility": (
            strategy[
                "annual_volatility"
            ]
        ),
        "strategy_sharpe": (
            strategy[
                "sharpe_ratio"
            ]
        ),
        "strategy_max_drawdown": (
            strategy[
                "max_drawdown"
            ]
        ),
        "strategy_calmar": (
            strategy[
                "calmar_ratio"
            ]
        ),
        "benchmark_cumulative_return": (
            benchmark[
                "cumulative_return"
            ]
        ),
        "benchmark_annual_return": (
            benchmark[
                "annual_return"
            ]
        ),
        "benchmark_annual_volatility": (
            benchmark[
                "annual_volatility"
            ]
        ),
        "benchmark_sharpe": (
            benchmark[
                "sharpe_ratio"
            ]
        ),
        "benchmark_max_drawdown": (
            benchmark[
                "max_drawdown"
            ]
        ),
        "benchmark_calmar": (
            benchmark[
                "calmar_ratio"
            ]
        ),
        "excess_annual_return": (
            strategy[
                "annual_return"
            ]
            - benchmark[
                "annual_return"
            ]
        ),
        "sharpe_diff": (
            strategy[
                "sharpe_ratio"
            ]
            - benchmark[
                "sharpe_ratio"
            ]
        ),
        "drawdown_improvement": (
            strategy[
                "max_drawdown"
            ]
            - benchmark[
                "max_drawdown"
            ]
        ),
        "tracking_error": tracking_error,
        "information_ratio": (
            information_ratio
        ),
        "average_exposure": float(
            portfolio[
                "portfolio_position"
            ].mean()
        ),
        "average_daily_turnover": float(
            portfolio[
                "portfolio_turnover"
            ].mean()
        ),
        "total_trade_count": int(
            portfolio[
                "trade_event_count"
            ].sum()
        ),
        "total_transaction_cost": float(
            portfolio[
                "portfolio_transaction_cost"
            ].sum()
        ),
        "total_boundary_cost_adjustment": float(
            portfolio[
                "portfolio_boundary_cost_adjustment"
            ].sum()
        ),
    }


def summarize_annual_returns(
    portfolio: pd.DataFrame,
    strategy_name: str,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    按自然年计算收益、夏普和年度内最大回撤。

    年度收益使用该年度内的实际累计收益，
    不对不完整年度做年化。
    """
    _require_columns(
        portfolio,
        {
            "date",
            "strategy_return",
            "benchmark_return",
        },
        "portfolio",
    )

    data = portfolio.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["year"] = (
        data["date"].dt.year
    )

    daily_risk_free_rate = (
        (
            1.0
            + annual_risk_free_rate
        )
        ** (
            1.0
            / trading_days
        )
        - 1.0
    )

    rows: list[dict[str, Any]] = []

    for (
        year,
        group,
    ) in data.groupby(
        "year",
        sort=True,
    ):
        strategy_returns = (
            group[
                "strategy_return"
            ]
        )

        benchmark_returns = (
            group[
                "benchmark_return"
            ]
        )

        strategy_nav = (
            1.0
            + strategy_returns
        ).cumprod()

        benchmark_nav = (
            1.0
            + benchmark_returns
        ).cumprod()

        strategy_year_return = float(
            strategy_nav.iloc[-1]
            - 1.0
        )

        benchmark_year_return = float(
            benchmark_nav.iloc[-1]
            - 1.0
        )

        rows.append(
            {
                "strategy_name": (
                    strategy_name
                ),
                "year": int(year),
                "trade_days": int(
                    len(group)
                ),
                "strategy_return": (
                    strategy_year_return
                ),
                "benchmark_return": (
                    benchmark_year_return
                ),
                "excess_return": (
                    strategy_year_return
                    - benchmark_year_return
                ),
                "strategy_sharpe": (
                    _sharpe(
                        returns=(
                            strategy_returns
                        ),
                        daily_risk_free_rate=(
                            daily_risk_free_rate
                        ),
                        trading_days=(
                            trading_days
                        ),
                    )
                ),
                "benchmark_sharpe": (
                    _sharpe(
                        returns=(
                            benchmark_returns
                        ),
                        daily_risk_free_rate=(
                            daily_risk_free_rate
                        ),
                        trading_days=(
                            trading_days
                        ),
                    )
                ),
                "strategy_max_drawdown": float(
                    (
                        strategy_nav
                        / strategy_nav.cummax()
                        - 1.0
                    ).min()
                ),
                "benchmark_max_drawdown": float(
                    (
                        benchmark_nav
                        / benchmark_nav.cummax()
                        - 1.0
                    ).min()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_equity_comparison(
    walk_forward_result: Mapping[
        str,
        Any,
    ],
    fixed_parameters: Iterable[
        Parameter
    ] = (
        (10, 40),
        (20, 60),
        (25, 110),
    ),
    dynamic_strategy_name: str = (
        "dynamic_rule_c"
    ),
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> dict[str, Any]:
    """
    构建规则 C 动态策略与多个固定参数的连续净值。

    walk_forward_result 必须由：

        run_ma_walk_forward(
            ...,
            return_backtests=True,
        )

    生成。
    """
    required_keys = {
        "windows",
        "selected_parameters",
        "backtests",
    }

    missing_keys = (
        required_keys
        - set(
            walk_forward_result
        )
    )

    if missing_keys:
        raise ValueError(
            "walk_forward_result 缺少："
            f"{sorted(missing_keys)}；"
            "请在 run_ma_walk_forward 中设置 "
            "return_backtests=True"
        )

    windows = _prepare_windows(
        walk_forward_result[
            "windows"
        ]
    )

    backtests = (
        walk_forward_result[
            "backtests"
        ]
    )

    schedules = {
        dynamic_strategy_name: (
            build_dynamic_schedule(
                walk_forward_result[
                    "selected_parameters"
                ]
            )
        )
    }

    for parameter in fixed_parameters:
        fast_window, slow_window = (
            _normalize_parameter(
                parameter
            )
        )

        strategy_name = (
            f"fixed_"
            f"{fast_window}_"
            f"{slow_window}"
        )

        schedules[
            strategy_name
        ] = build_fixed_schedule(
            windows=windows,
            parameter=parameter,
        )

    stock_details: dict[
        str,
        pd.DataFrame,
    ] = {}

    portfolios: dict[
        str,
        pd.DataFrame,
    ] = {}

    summaries: list[
        dict[str, Any]
    ] = []

    annual_summaries: list[
        pd.DataFrame
    ] = []

    for (
        strategy_name,
        schedule,
    ) in schedules.items():
        detail = (
            build_continuous_stock_detail(
                backtests=backtests,
                windows=windows,
                schedule=schedule,
                commission_rate=(
                    commission_rate
                ),
                slippage_rate=(
                    slippage_rate
                ),
            )
        )

        portfolio = build_portfolio(
            stock_detail=detail
        )

        stock_details[
            strategy_name
        ] = detail

        portfolios[
            strategy_name
        ] = portfolio

        summaries.append(
            summarize_portfolio(
                portfolio=portfolio,
                strategy_name=(
                    strategy_name
                ),
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
                trading_days=(
                    trading_days
                ),
            )
        )

        annual_summaries.append(
            summarize_annual_returns(
                portfolio=portfolio,
                strategy_name=(
                    strategy_name
                ),
                annual_risk_free_rate=(
                    annual_risk_free_rate
                ),
                trading_days=(
                    trading_days
                ),
            )
        )

    _check_same_benchmark(
        portfolios
    )

    summary = (
        pd.DataFrame(
            summaries
        )
        .sort_values(
            [
                "strategy_sharpe",
                "strategy_annual_return",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    annual_summary = pd.concat(
        annual_summaries,
        ignore_index=True,
    )

    return {
        "schedules": schedules,
        "stock_details": stock_details,
        "portfolios": portfolios,
        "summary": summary,
        "annual_summary": (
            annual_summary
        ),
    }


def plot_equity_curves(
    portfolios: Mapping[
        str,
        pd.DataFrame,
    ],
) -> tuple[Figure, Axes]:
    """
    绘制连续净值曲线。
    """
    if not portfolios:
        raise ValueError(
            "portfolios 不能为空"
        )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for (
        strategy_name,
        portfolio,
    ) in portfolios.items():
        axis.plot(
            portfolio["date"],
            portfolio[
                "strategy_nav"
            ],
            label=strategy_name,
        )

    benchmark = next(
        iter(
            portfolios.values()
        )
    )

    axis.plot(
        benchmark["date"],
        benchmark[
            "benchmark_nav"
        ],
        label=(
            "fixed_weight_buy_hold"
        ),
        linestyle="--",
    )

    axis.axhline(
        1.0,
        linewidth=1.0,
        alpha=0.5,
    )

    axis.set_xlabel("日期")
    axis.set_ylabel("净值")

    axis.set_title(
        "连续 Walk-Forward 净值对比"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    return figure, axis


def plot_drawdown_curves(
    portfolios: Mapping[
        str,
        pd.DataFrame,
    ],
) -> tuple[Figure, Axes]:
    """
    绘制连续回撤曲线。
    """
    if not portfolios:
        raise ValueError(
            "portfolios 不能为空"
        )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    for (
        strategy_name,
        portfolio,
    ) in portfolios.items():
        axis.plot(
            portfolio["date"],
            portfolio[
                "strategy_drawdown"
            ],
            label=strategy_name,
        )

    benchmark = next(
        iter(
            portfolios.values()
        )
    )

    axis.plot(
        benchmark["date"],
        benchmark[
            "benchmark_drawdown"
        ],
        label=(
            "fixed_weight_buy_hold"
        ),
        linestyle="--",
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
        alpha=0.5,
    )

    axis.set_xlabel("日期")
    axis.set_ylabel("回撤")

    axis.set_title(
        "连续 Walk-Forward 回撤对比"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    return figure, axis


def _prepare_windows(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        windows,
        {
            "fold_id",
            "test_start",
            "test_end",
        },
        "windows",
    )

    result = windows.copy()

    result["fold_id"] = (
        pd.to_numeric(
            result["fold_id"],
            errors="raise",
        )
        .astype(int)
    )

    result["test_start"] = (
        pd.to_datetime(
            result["test_start"],
            errors="raise",
        )
        .dt.normalize()
    )

    result["test_end"] = (
        pd.to_datetime(
            result["test_end"],
            errors="raise",
        )
        .dt.normalize()
    )

    result = (
        result.sort_values(
            "fold_id"
        )
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError(
            "windows 不能为空"
        )

    if result[
        "fold_id"
    ].duplicated().any():
        raise ValueError(
            "windows 中 fold_id 重复"
        )

    if (
        result["test_start"]
        > result["test_end"]
    ).any():
        raise ValueError(
            "存在非法测试窗口"
        )

    previous_end = (
        result["test_end"]
        .shift(1)
    )

    overlap = (
        previous_end.notna()
        & (
            result["test_start"]
            <= previous_end
        )
    )

    if overlap.any():
        raise ValueError(
            "测试窗口存在重叠"
        )

    expected_next_start = (
        previous_end
        + pd.Timedelta(days=1)
    )

    gap = (
        previous_end.notna()
        & (
            result["test_start"]
            != expected_next_start
        )
    )

    if gap.any():
        raise ValueError(
            "连续净值要求测试窗口首尾相接"
        )

    return result


def _prepare_schedule(
    schedule: pd.DataFrame,
    windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require_columns(
        schedule,
        {
            "fold_id",
            "fast_window",
            "slow_window",
        },
        "schedule",
    )

    result = schedule.copy()

    for column in [
        "fold_id",
        "fast_window",
        "slow_window",
    ]:
        result[column] = (
            pd.to_numeric(
                result[column],
                errors="raise",
            )
            .astype(int)
        )

    if result[
        "fold_id"
    ].duplicated().any():
        raise ValueError(
            "schedule 中 fold_id 重复"
        )

    invalid_parameter = (
        (
            result[
                "fast_window"
            ]
            <= 0
        )
        | (
            result[
                "slow_window"
            ]
            <= 0
        )
        | (
            result[
                "fast_window"
            ]
            >= result[
                "slow_window"
            ]
        )
    )

    if invalid_parameter.any():
        raise ValueError(
            "schedule 存在非法参数"
        )

    if windows is not None:
        windows = _prepare_windows(
            windows
        )

        if set(
            result["fold_id"]
        ) != set(
            windows["fold_id"]
        ):
            raise ValueError(
                "schedule 与 windows 折数不一致"
            )

    result["ma_param"] = (
        result[
            "fast_window"
        ].astype(str)
        + "/"
        + result[
            "slow_window"
        ].astype(str)
    )

    return (
        result.sort_values(
            "fold_id"
        )
        .reset_index(drop=True)
    )


def _prepare_backtests(
    backtests: Mapping[
        BacktestKey,
        pd.DataFrame,
    ],
) -> dict[
    BacktestKey,
    pd.DataFrame,
]:
    required_columns = {
        "date",
        "symbol",
        "asset_return",
        "position",
        "position_change",
        "gross_strategy_return",
        "transaction_cost",
        "strategy_return",
    }

    result: dict[
        BacktestKey,
        pd.DataFrame,
    ] = {}

    for (
        raw_key,
        raw_data,
    ) in backtests.items():
        if len(raw_key) != 3:
            raise ValueError(
                "backtests 键必须为 "
                "(symbol, fast_window, slow_window)"
            )

        symbol = str(
            raw_key[0]
        ).zfill(6)

        fast_window, slow_window = (
            _normalize_parameter(
                (
                    int(raw_key[1]),
                    int(raw_key[2]),
                )
            )
        )

        _require_columns(
            raw_data,
            required_columns,
            f"backtests[{raw_key}]",
        )

        data = raw_data.copy()

        data["date"] = pd.to_datetime(
            data["date"],
            errors="raise",
        )

        data["symbol"] = (
            data["symbol"]
            .astype(str)
            .str.zfill(6)
        )

        if data["symbol"].nunique() != 1:
            raise ValueError(
                f"backtests[{raw_key}] "
                "包含多只股票"
            )

        if (
            data["symbol"].iloc[0]
            != symbol
        ):
            raise ValueError(
                f"backtests[{raw_key}] "
                "的 symbol 与键不一致"
            )

        if data[
            "date"
        ].duplicated().any():
            raise ValueError(
                f"backtests[{raw_key}] "
                "存在重复日期"
            )

        numeric_columns = (
            required_columns
            - {
                "date",
                "symbol",
            }
        )

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="raise",
            )

        result[
            (
                symbol,
                fast_window,
                slow_window,
            )
        ] = (
            data.sort_values("date")
            .reset_index(drop=True)
        )

    if not result:
        raise ValueError(
            "backtests 不能为空"
        )

    return result


def _pivot(
    data: pd.DataFrame,
    column: str,
    symbols: list[str],
    fill: str,
) -> pd.DataFrame:
    matrix = (
        data.pivot(
            index="date",
            columns="symbol",
            values=column,
        )
        .sort_index()
        .reindex(
            columns=symbols
        )
    )

    if fill == "zero":
        return (
            matrix.fillna(0.0)
            .astype(float)
        )

    if fill == "forward":
        return (
            matrix.ffill()
            .fillna(0.0)
            .astype(float)
        )

    raise ValueError(
        f"未知填充方式：{fill}"
    )


def _returns_from_nav(
    nav: pd.Series,
) -> pd.Series:
    if nav.empty:
        raise ValueError(
            "nav 不能为空"
        )

    returns = nav.pct_change()

    returns.iloc[0] = (
        nav.iloc[0]
        - 1.0
    )

    return returns.astype(float)


def _sharpe(
    returns: pd.Series,
    daily_risk_free_rate: float,
    trading_days: int,
) -> float:
    volatility = returns.std(
        ddof=1
    )

    if (
        pd.isna(volatility)
        or np.isclose(
            volatility,
            0.0,
        )
    ):
        return np.nan

    return float(
        (
            returns
            - daily_risk_free_rate
        ).mean()
        / volatility
        * np.sqrt(
            trading_days
        )
    )


def _check_same_benchmark(
    portfolios: Mapping[
        str,
        pd.DataFrame,
    ],
) -> None:
    iterator = iter(
        portfolios.items()
    )

    first_name, first = next(
        iterator
    )

    reference = first[
        [
            "date",
            "benchmark_return",
        ]
    ].reset_index(drop=True)

    for (
        strategy_name,
        portfolio,
    ) in iterator:
        candidate = portfolio[
            [
                "date",
                "benchmark_return",
            ]
        ].reset_index(drop=True)

        try:
            pd.testing.assert_frame_equal(
                reference,
                candidate,
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                "基准序列不一致："
                f"{first_name} 与 "
                f"{strategy_name}"
            ) from error


def _normalize_parameter(
    parameter: Parameter,
) -> Parameter:
    if len(parameter) != 2:
        raise ValueError(
            "parameter 必须包含快线和慢线"
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
            "fast_window 必须小于 slow_window"
        )

    return (
        fast_window,
        slow_window,
    )


def _require_columns(
    data: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"{name} 缺少字段："
            f"{sorted(missing)}"
        )


def _import_calculate_performance():
    try:
        from .backtest import (
            calculate_performance,
        )
    except ImportError:
        from backtest import (
            calculate_performance,
        )

    return calculate_performance

def summarize_parameter_validation(
    validation_detail: pd.DataFrame,
) -> pd.DataFrame:
    """
    汇总每组固定参数在新股票池上的表现。
    """

    required_columns = {
        "ma_param",
        "symbol",
        "strategy_annual_return",
        "strategy_sharpe",
        "strategy_max_drawdown",
        "excess_annual_return",
        "sharpe_diff",
        "drawdown_improvement",
        "exposure",
        "total_trade_count",
        "total_transaction_cost",
    }

    missing_columns = (
        required_columns
        - set(validation_detail.columns)
    )

    if missing_columns:
        raise ValueError(
            f"缺少字段："
            f"{sorted(missing_columns)}"
        )

    result = (
        validation_detail
        .groupby(
            "ma_param",
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
            std_strategy_annual_return=(
                "strategy_annual_return",
                "std",
            ),

            avg_strategy_sharpe=(
                "strategy_sharpe",
                "mean",
            ),
            median_strategy_sharpe=(
                "strategy_sharpe",
                "median",
            ),
            std_strategy_sharpe=(
                "strategy_sharpe",
                "std",
            ),
            worst_strategy_sharpe=(
                "strategy_sharpe",
                "min",
            ),

            avg_excess_annual_return=(
                "excess_annual_return",
                "mean",
            ),
            median_excess_annual_return=(
                "excess_annual_return",
                "median",
            ),
            std_excess_annual_return=(
                "excess_annual_return",
                "std",
            ),
            worst_excess_annual_return=(
                "excess_annual_return",
                "min",
            ),

            avg_sharpe_diff=(
                "sharpe_diff",
                "mean",
            ),
            median_sharpe_diff=(
                "sharpe_diff",
                "median",
            ),

            avg_strategy_max_drawdown=(
                "strategy_max_drawdown",
                "mean",
            ),
            avg_drawdown_improvement=(
                "drawdown_improvement",
                "mean",
            ),
            worst_drawdown_improvement=(
                "drawdown_improvement",
                "min",
            ),

            avg_exposure=(
                "exposure",
                "mean",
            ),
            avg_trade_count=(
                "total_trade_count",
                "mean",
            ),
            avg_transaction_cost=(
                "total_transaction_cost",
                "mean",
            ),

            return_win_rate=(
                "excess_annual_return",
                lambda values: float(
                    (values > 0).mean()
                ),
            ),
            sharpe_win_rate=(
                "sharpe_diff",
                lambda values: float(
                    (values > 0).mean()
                ),
            ),
            drawdown_win_rate=(
                "drawdown_improvement",
                lambda values: float(
                    (values > 0).mean()
                ),
            ),
        )
        .sort_values(
            [
                "avg_strategy_sharpe",
                "avg_excess_annual_return",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return result