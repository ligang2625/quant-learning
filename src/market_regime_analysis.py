"""
市场状态与 MA 策略失效归因。

核心功能：

1. 使用当日之前的信息定义市场状态，避免未来函数；
2. 按趋势方向、趋势质量和波动状态分析策略表现；
3. 将相对买入持有的主动收益分解为：
   - 错过上涨；
   - 规避下跌；
   - 交易成本；
4. 分析策略失效集中在哪些股票和市场状态。

本模块不负责参数搜索，也不自动保存文件或显示图片。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


def add_market_regime_features(
    result: pd.DataFrame,
    trend_window: int = 60,
    volatility_window: int = 20,
    volatility_reference_window: int = 252,
    volatility_reference_min_periods: int = 60,
    efficiency_window: int = 20,
    direction_threshold: float = 1.0,
    efficiency_threshold: float = 0.35,
    start_date: str | None = None,
    end_date: str | None = None,
    exclude_warmup: bool = True,
) -> pd.DataFrame:
    """
    为单只股票逐日回测结果添加市场状态与归因字段。

    所有状态特征整体向后移动一天，因此第 t 日的状态
    只使用第 t-1 日及以前的信息。

    Parameters
    ----------
    result:
        ma_cross_backtest 的完整逐日结果。

    trend_window:
        判断中期趋势方向的窗口。

    volatility_window:
        计算近期年化波动率的窗口。

    volatility_reference_window:
        判断高低波动状态时，历史波动率基准的窗口。

    volatility_reference_min_periods:
        波动率基准最少需要的历史记录数。

    efficiency_window:
        价格路径效率计算窗口。

    direction_threshold:
        标准化趋势分数超过该值判定为上涨趋势，
        低于其负值判定为下跌趋势。

    efficiency_threshold:
        路径效率高于该值判定为 trending，
        否则判定为 choppy。

    start_date, end_date:
        最终分析区间。特征仍会使用区间之前的历史计算。

    exclude_warmup:
        是否排除慢均线尚未形成的回测预热期。
    """
    _validate_positive_int(
        trend_window,
        "trend_window",
    )
    _validate_positive_int(
        volatility_window,
        "volatility_window",
    )
    _validate_positive_int(
        volatility_reference_window,
        "volatility_reference_window",
    )
    _validate_positive_int(
        volatility_reference_min_periods,
        "volatility_reference_min_periods",
    )
    _validate_positive_int(
        efficiency_window,
        "efficiency_window",
    )

    if (
        volatility_reference_min_periods
        > volatility_reference_window
    ):
        raise ValueError(
            "volatility_reference_min_periods "
            "不能大于 volatility_reference_window"
        )

    if direction_threshold <= 0:
        raise ValueError(
            "direction_threshold 必须大于 0"
        )

    if not 0 <= efficiency_threshold <= 1:
        raise ValueError(
            "efficiency_threshold 必须位于 0 到 1 之间"
        )

    required_columns = {
        "date",
        "symbol",
        "close",
        "asset_return",
        "position",
        "position_change",
        "gross_strategy_return",
        "transaction_cost",
        "strategy_return",
    }

    _require_columns(
        data=result,
        required_columns=required_columns,
        data_name="result",
    )

    data = result.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.zfill(6)
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

    data = (
        data.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(
            "result 不能为空"
        )

    if data["symbol"].nunique() != 1:
        raise ValueError(
            "result 必须只包含一只股票"
        )

    if (data["close"] <= 0).any():
        raise ValueError(
            "close 必须全部大于 0"
        )

    if (
        data["strategy_return"]
        <= -1
    ).any():
        raise ValueError(
            "strategy_return 不能小于等于 -100%"
        )

    if (
        data["asset_return"]
        .dropna()
        .le(-1)
        .any()
    ):
        raise ValueError(
            "asset_return 不能小于等于 -100%"
        )

    asset_return = (
        data["asset_return"]
        .fillna(0.0)
    )

    # 截至当日的趋势收益。
    raw_trailing_return = (
        data["close"]
        .pct_change(trend_window)
    )

    # 截至当日的近期年化波动率。
    raw_annualized_volatility = (
        asset_return
        .rolling(
            volatility_window,
            min_periods=volatility_window,
        )
        .std(ddof=1)
        * np.sqrt(252)
    )

    # 每只股票自身历史波动率的滚动中位数。
    raw_volatility_reference = (
        raw_annualized_volatility
        .rolling(
            volatility_reference_window,
            min_periods=(
                volatility_reference_min_periods
            ),
        )
        .median()
    )

    # 路径效率：
    # 净价格变化 / 每日价格变化绝对值总和。
    raw_path_length = (
        data["close"]
        .diff()
        .abs()
        .rolling(
            efficiency_window,
            min_periods=efficiency_window,
        )
        .sum()
    )

    raw_net_change = (
        data["close"]
        .diff(efficiency_window)
        .abs()
    )

    raw_efficiency_ratio = (
        raw_net_change
        / raw_path_length.replace(
            0.0,
            np.nan,
        )
    )

    # 整体移动一天，避免用当日收盘判断当日状态。
    data["trailing_return"] = (
        raw_trailing_return.shift(1)
    )

    data["annualized_volatility"] = (
        raw_annualized_volatility.shift(1)
    )

    data["volatility_reference"] = (
        raw_volatility_reference.shift(1)
    )

    data["efficiency_ratio"] = (
        raw_efficiency_ratio.shift(1)
    )

    # 将趋势收益转换为波动率标准化趋势分数。
    expected_trend_move = (
        data["annualized_volatility"]
        / np.sqrt(252)
        * np.sqrt(trend_window)
    )

    data["trend_score"] = (
        data["trailing_return"]
        / expected_trend_move.replace(
            0.0,
            np.nan,
        )
    )

    data["market_direction"] = np.select(
        [
            data["trend_score"]
            >= direction_threshold,

            data["trend_score"]
            <= -direction_threshold,
        ],
        [
            "up",
            "down",
        ],
        default="sideways",
    )

    data["trend_quality"] = np.where(
        data["efficiency_ratio"]
        >= efficiency_threshold,
        "trending",
        "choppy",
    )

    data["volatility_state"] = np.where(
        data["annualized_volatility"]
        >= data["volatility_reference"],
        "high_vol",
        "low_vol",
    )

    regime_feature_columns = [
        "trailing_return",
        "annualized_volatility",
        "volatility_reference",
        "efficiency_ratio",
        "trend_score",
    ]

    valid_mask = (
        data[regime_feature_columns]
        .notna()
        .all(axis=1)
    )

    if (
        exclude_warmup
        and "slow_ma" in data.columns
    ):
        valid_mask &= (
            data["slow_ma"]
            .shift(1)
            .notna()
        )

    data = data.loc[
        valid_mask
    ].copy()

    # 在特征计算完成后才切分析区间，
    # 保留起始日期之前的历史用于状态计算。
    if start_date is not None:
        data = data.loc[
            data["date"]
            >= pd.Timestamp(start_date)
        ]

    if end_date is not None:
        data = data.loc[
            data["date"]
            <= pd.Timestamp(end_date)
        ]

    if data.empty:
        raise ValueError(
            "市场状态计算后没有有效记录，"
            "请检查日期范围或窗口设置"
        )

    data["market_regime"] = (
        data["market_direction"]
        + "|"
        + data["trend_quality"]
        + "|"
        + data["volatility_state"]
    )

    # -------------------------------------------------
    # 相对买入持有的主动收益分解
    # -------------------------------------------------

    data["active_return"] = (
        data["strategy_return"]
        - data["asset_return"].fillna(0.0)
    )

    data["timing_active_return"] = (
        data["gross_strategy_return"]
        - data["asset_return"].fillna(0.0)
    )

    data["cost_drag_return"] = (
        -data["transaction_cost"]
    )

    data["missed_upside_return"] = np.where(
        data["timing_active_return"] < 0,
        data["timing_active_return"],
        0.0,
    )

    data["avoided_downside_return"] = np.where(
        data["timing_active_return"] > 0,
        data["timing_active_return"],
        0.0,
    )

    data["strategy_log_return"] = np.log1p(
        data["strategy_return"]
    )

    data["benchmark_log_return"] = np.log1p(
        data["asset_return"].fillna(0.0)
    )

    data["active_log_return"] = (
        data["strategy_log_return"]
        - data["benchmark_log_return"]
    )

    position_in_market = (
        data["position"] > 0.5
    )

    benchmark_up = (
        data["asset_return"]
        .fillna(0.0)
        > 0
    )

    benchmark_down = (
        data["asset_return"]
        .fillna(0.0)
        < 0
    )

    data["timing_outcome"] = np.select(
        [
            position_in_market
            & benchmark_up,

            position_in_market
            & benchmark_down,

            (~position_in_market)
            & benchmark_up,

            (~position_in_market)
            & benchmark_down,
        ],
        [
            "captured_upside",
            "suffered_downside",
            "missed_upside",
            "avoided_downside",
        ],
        default="flat_market",
    )

    data[
        "underperformed_benchmark"
    ] = (
        data["active_return"] < 0
    )

    data["year"] = (
        data["date"].dt.year
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
        + data["avoided_downside_return"]
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

    return (
        data.reset_index(drop=True)
    )


def analyze_batch_market_regimes(
    batch_results: Mapping[
        str,
        pd.DataFrame,
    ],
    **feature_kwargs: Any,
) -> pd.DataFrame:
    """
    对 run_batch_ma_backtest 返回的 batch_results
    逐只股票添加市场状态和归因字段。
    """
    if not batch_results:
        raise ValueError(
            "batch_results 不能为空"
        )

    frames: list[pd.DataFrame] = []

    for raw_symbol, result in (
        batch_results.items()
    ):
        symbol = str(
            raw_symbol
        ).zfill(6)

        detail = (
            add_market_regime_features(
                result=result,
                **feature_kwargs,
            )
        )

        if (
            detail["symbol"].iloc[0]
            != symbol
        ):
            raise ValueError(
                f"batch_results 键 {symbol} "
                "与结果中的 symbol 不一致"
            )

        frames.append(detail)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined.duplicated(
        [
            "symbol",
            "date",
        ]
    ).any():
        raise ValueError(
            "批量市场状态明细存在重复股票日期"
        )

    return (
        combined.sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_regime_performance(
    regime_detail: pd.DataFrame,
    group_columns: Sequence[str] = (
        "market_direction",
        "trend_quality",
        "volatility_state",
    ),
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    按指定市场状态汇总条件表现。

    条件夏普将该状态下的所有日收益放在一起计算，
    主要用于不同市场状态之间的相对比较。
    """
    _validate_positive_int(
        trading_days,
        "trading_days",
    )

    group_columns = list(
        group_columns
    )

    required_columns = {
        "date",
        "symbol",
        "strategy_return",
        "asset_return",
        "active_return",
        "strategy_log_return",
        "benchmark_log_return",
        "active_log_return",
        "position",
        "position_change",
        "transaction_cost",
        "missed_upside_return",
        "avoided_downside_return",
        "cost_drag_return",
        "underperformed_benchmark",
        *group_columns,
    }

    _require_columns(
        data=regime_detail,
        required_columns=required_columns,
        data_name="regime_detail",
    )

    if regime_detail.empty:
        raise ValueError(
            "regime_detail 不能为空"
        )

    rows: list[
        dict[str, Any]
    ] = []

    grouped = regime_detail.groupby(
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

        strategy_returns = (
            group["strategy_return"]
        )

        benchmark_returns = (
            group["asset_return"]
            .fillna(0.0)
        )

        active_returns = (
            group["active_return"]
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
                    group["symbol"].nunique()
                ),

                "date_count": int(
                    group["date"].nunique()
                ),

                "strategy_mean_daily_return": float(
                    strategy_returns.mean()
                ),

                "benchmark_mean_daily_return": float(
                    benchmark_returns.mean()
                ),

                "active_mean_daily_return": float(
                    active_returns.mean()
                ),

                "conditional_strategy_sharpe": (
                    _annualized_sharpe(
                        strategy_returns,
                        trading_days,
                    )
                ),

                "conditional_benchmark_sharpe": (
                    _annualized_sharpe(
                        benchmark_returns,
                        trading_days,
                    )
                ),

                "conditional_active_information_ratio": (
                    _annualized_sharpe(
                        active_returns,
                        trading_days,
                    )
                ),

                "strategy_win_rate": float(
                    (
                        strategy_returns > 0
                    ).mean()
                ),

                "active_win_rate": float(
                    (
                        active_returns > 0
                    ).mean()
                ),

                "underperformance_rate": float(
                    group[
                        "underperformed_benchmark"
                    ].mean()
                ),

                "average_position": float(
                    group["position"].mean()
                ),

                "trade_event_count": int(
                    (
                        group[
                            "position_change"
                        ].abs() > 0
                    ).sum()
                ),

                "transaction_cost_sum": float(
                    group[
                        "transaction_cost"
                    ].sum()
                ),

                "strategy_log_return_sum": float(
                    group[
                        "strategy_log_return"
                    ].sum()
                ),

                "benchmark_log_return_sum": float(
                    group[
                        "benchmark_log_return"
                    ].sum()
                ),

                "active_log_return_sum": float(
                    group[
                        "active_log_return"
                    ].sum()
                ),

                "active_return_sum": float(
                    group[
                        "active_return"
                    ].sum()
                ),

                "missed_upside_sum": float(
                    group[
                        "missed_upside_return"
                    ].sum()
                ),

                "avoided_downside_sum": float(
                    group[
                        "avoided_downside_return"
                    ].sum()
                ),

                "cost_drag_sum": float(
                    group[
                        "cost_drag_return"
                    ].sum()
                ),
            }
        )

        rows.append(row)

    summary = pd.DataFrame(rows)

    return (
        summary.sort_values(
            [
                "active_log_return_sum",
                "observation_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def summarize_failure_attribution(
    regime_detail: pd.DataFrame,
    group_columns: Sequence[str] = (
        "symbol",
    ),
) -> pd.DataFrame:
    """
    汇总主动收益的三项来源。

    返回值中：

    missed_upside_loss:
        使用正数表示错过上涨造成的损失规模。

    avoided_downside_benefit:
        使用正数表示规避下跌带来的收益。

    transaction_cost_loss:
        使用正数表示交易成本损失。
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
        "timing_outcome",
        *group_columns,
    }

    _require_columns(
        data=regime_detail,
        required_columns=required_columns,
        data_name="regime_detail",
    )

    if regime_detail.empty:
        raise ValueError(
            "regime_detail 不能为空"
        )

    rows: list[
        dict[str, Any]
    ] = []

    grouped = regime_detail.groupby(
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

        missed_loss = float(
            -group[
                "missed_upside_return"
            ].sum()
        )

        avoided_benefit = float(
            group[
                "avoided_downside_return"
            ].sum()
        )

        cost_loss = float(
            -group[
                "cost_drag_return"
            ].sum()
        )

        active_return_sum = float(
            group["active_return"].sum()
        )

        total_loss = (
            missed_loss
            + cost_loss
        )

        if active_return_sum >= 0:
            primary_failure_driver = (
                "none"
            )
        elif missed_loss >= cost_loss:
            primary_failure_driver = (
                "missed_upside"
            )
        else:
            primary_failure_driver = (
                "transaction_cost"
            )

        benefit_cost_ratio = (
            avoided_benefit
            / total_loss
            if not np.isclose(
                total_loss,
                0.0,
            )
            else np.nan
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

                "active_return_sum": (
                    active_return_sum
                ),

                "active_log_return_sum": float(
                    group[
                        "active_log_return"
                    ].sum()
                ),

                "missed_upside_loss": (
                    missed_loss
                ),

                "avoided_downside_benefit": (
                    avoided_benefit
                ),

                "transaction_cost_loss": (
                    cost_loss
                ),

                "net_timing_benefit": (
                    avoided_benefit
                    - missed_loss
                ),

                "benefit_cost_ratio": (
                    benefit_cost_ratio
                ),

                "missed_upside_days": int(
                    (
                        group[
                            "timing_outcome"
                        ]
                        == "missed_upside"
                    ).sum()
                ),

                "avoided_downside_days": int(
                    (
                        group[
                            "timing_outcome"
                        ]
                        == "avoided_downside"
                    ).sum()
                ),

                "captured_upside_days": int(
                    (
                        group[
                            "timing_outcome"
                        ]
                        == "captured_upside"
                    ).sum()
                ),

                "suffered_downside_days": int(
                    (
                        group[
                            "timing_outcome"
                        ]
                        == "suffered_downside"
                    ).sum()
                ),

                "primary_failure_driver": (
                    primary_failure_driver
                ),
            }
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    return (
        result.sort_values(
            "active_log_return_sum",
            ascending=True,
        )
        .reset_index(drop=True)
    )


def summarize_timing_outcomes(
    regime_detail: pd.DataFrame,
    group_columns: Sequence[str] = (
        "timing_outcome",
    ),
) -> pd.DataFrame:
    """
    汇总捕获上涨、承受下跌、错过上涨和规避下跌。
    """
    group_columns = list(
        group_columns
    )

    required_columns = {
        "date",
        "symbol",
        "strategy_return",
        "asset_return",
        "active_return",
        "transaction_cost",
        *group_columns,
    }

    _require_columns(
        data=regime_detail,
        required_columns=required_columns,
        data_name="regime_detail",
    )

    result = (
        regime_detail.groupby(
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

            strategy_return_sum=(
                "strategy_return",
                "sum",
            ),

            benchmark_return_sum=(
                "asset_return",
                "sum",
            ),

            active_return_sum=(
                "active_return",
                "sum",
            ),

            transaction_cost_sum=(
                "transaction_cost",
                "sum",
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
        )
    )

    result["observation_rate"] = (
        result["observation_count"]
        / result[
            "observation_count"
        ].sum()
    )

    return (
        result.sort_values(
            "active_return_sum",
            ascending=True,
        )
        .reset_index(drop=True)
    )


def create_regime_matrix(
    regime_summary: pd.DataFrame,
    value_column: str,
    index_column: str = (
        "market_direction"
    ),
    columns_column: str = (
        "volatility_state"
    ),
    filters: Mapping[
        str,
        Any,
    ]
    | None = None,
) -> pd.DataFrame:
    """
    将状态汇总表转换为二维矩阵。
    """
    required_columns = {
        value_column,
        index_column,
        columns_column,
    }

    _require_columns(
        data=regime_summary,
        required_columns=required_columns,
        data_name="regime_summary",
    )

    data = regime_summary.copy()

    if filters:
        for column, value in (
            filters.items()
        ):
            if column not in data.columns:
                raise ValueError(
                    f"过滤字段不存在：{column}"
                )

            data = data.loc[
                data[column] == value
            ]

    if data.empty:
        raise ValueError(
            "过滤后没有状态记录"
        )

    if data.duplicated(
        [
            index_column,
            columns_column,
        ]
    ).any():
        raise ValueError(
            "当前过滤条件下矩阵单元格不唯一，"
            "请增加过滤条件"
        )

    matrix = (
        data.pivot(
            index=index_column,
            columns=columns_column,
            values=value_column,
        )
    )

    if (
        index_column
        == "market_direction"
    ):
        matrix = matrix.reindex(
            [
                "up",
                "sideways",
                "down",
            ]
        )

    if (
        columns_column
        == "volatility_state"
    ):
        matrix = matrix.reindex(
            columns=[
                "low_vol",
                "high_vol",
            ]
        )

    return matrix


def plot_regime_heatmap(
    regime_summary: pd.DataFrame,
    value_column: str = (
        "active_mean_daily_return"
    ),
    index_column: str = (
        "market_direction"
    ),
    columns_column: str = (
        "volatility_state"
    ),
    filters: Mapping[
        str,
        Any,
    ]
    | None = None,
    value_format: str = ".3%",
    title: str | None = None,
    cmap: str = "RdYlGn",
) -> tuple[Figure, Axes]:
    """
    绘制市场状态热力图。

    默认使用英文标题，避免中文字体缺失警告。
    """
    matrix = create_regime_matrix(
        regime_summary=(
            regime_summary
        ),
        value_column=value_column,
        index_column=index_column,
        columns_column=(
            columns_column
        ),
        filters=filters,
    )

    values = matrix.to_numpy(
        dtype=float
    )

    figure, axis = plt.subplots(
        figsize=(
            max(
                7.0,
                1.5
                * len(matrix.columns)
                + 3.0,
            ),
            max(
                4.5,
                0.9
                * len(matrix.index)
                + 2.5,
            ),
        )
    )

    image = axis.imshow(
        np.ma.masked_invalid(
            values
        ),
        aspect="auto",
        cmap=cmap,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        value_column
    )

    axis.set_xticks(
        np.arange(
            len(matrix.columns)
        )
    )

    axis.set_xticklabels(
        matrix.columns
    )

    axis.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    axis.set_yticklabels(
        matrix.index
    )

    axis.set_xlabel(
        columns_column
    )

    axis.set_ylabel(
        index_column
    )

    axis.set_title(
        title
        or (
            "Regime analysis: "
            f"{value_column}"
        )
    )

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            value = values[
                row_index,
                column_index,
            ]

            if np.isnan(value):
                continue

            red, green, blue, _ = (
                image.cmap(
                    image.norm(value)
                )
            )

            luminance = (
                0.2126 * red
                + 0.7152 * green
                + 0.0722 * blue
            )

            text_color = (
                "black"
                if luminance > 0.55
                else "white"
            )

            axis.text(
                column_index,
                row_index,
                format(
                    value,
                    value_format,
                ),
                ha="center",
                va="center",
                color=text_color,
            )

    figure.tight_layout()

    return (
        figure,
        axis,
    )


def plot_failure_decomposition(
    failure_summary: pd.DataFrame,
    label_column: str = "symbol",
    title: str = (
        "Active-return attribution"
    ),
) -> tuple[Figure, Axes]:
    """
    绘制错过上涨、规避下跌和交易成本归因图。
    """
    required_columns = {
        label_column,
        "missed_upside_loss",
        "avoided_downside_benefit",
        "transaction_cost_loss",
    }

    _require_columns(
        data=failure_summary,
        required_columns=required_columns,
        data_name="failure_summary",
    )

    if failure_summary.empty:
        raise ValueError(
            "failure_summary 不能为空"
        )

    data = failure_summary.copy()

    if (
        "active_log_return_sum"
        in data.columns
    ):
        data = data.sort_values(
            "active_log_return_sum"
        )
    else:
        data = data.sort_values(
            label_column
        )

    x_values = np.arange(
        len(data)
    )

    width = 0.25

    figure, axis = plt.subplots(
        figsize=(
            max(
                10.0,
                0.9 * len(data) + 4.0,
            ),
            6,
        )
    )

    axis.bar(
        x_values - width,
        -data[
            "missed_upside_loss"
        ],
        width,
        label="Missed upside",
    )

    axis.bar(
        x_values,
        data[
            "avoided_downside_benefit"
        ],
        width,
        label="Avoided downside",
    )

    axis.bar(
        x_values + width,
        -data[
            "transaction_cost_loss"
        ],
        width,
        label="Transaction cost",
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xticks(
        x_values
    )

    axis.set_xticklabels(
        data[
            label_column
        ].astype(str),
        rotation=45,
        ha="right",
    )

    axis.set_xlabel(
        label_column
    )

    axis.set_ylabel(
        "Return contribution"
    )

    axis.set_title(
        title
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    return (
        figure,
        axis,
    )


def _annualized_sharpe(
    returns: pd.Series,
    trading_days: int,
) -> float:
    clean_returns = (
        pd.to_numeric(
            returns,
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

    if len(clean_returns) < 2:
        return np.nan

    volatility = (
        clean_returns.std(
            ddof=1
        )
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
        clean_returns.mean()
        / volatility
        * np.sqrt(
            trading_days
        )
    )


def _validate_positive_int(
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

    if int(value) <= 0:
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