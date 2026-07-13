"""
MA 参数敏感性分析。

本模块只提供可复用函数，包括：

1. 生成双均线参数网格；
2. 汇总参数网格回测结果；
3. 计算参数局部稳定性；
4. 选择稳健参数候选；
5. 绘制参数热力图；
6. 绘制参数切片图；
7. 绘制表现与稳定性关系图；
8. 组织完整的参数敏感性分析流程。

数据展示、结果保存和学习过程应放在 notebook 中完成。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "avg_strategy_annual_return": "平均策略年化收益",
    "median_strategy_annual_return": "策略年化收益中位数",
    "std_strategy_annual_return": "策略年化收益标准差",
    "avg_excess_annual_return": "平均超额年化收益",
    "median_excess_annual_return": "超额年化收益中位数",
    "std_excess_annual_return": "超额年化收益标准差",
    "worst_excess_annual_return": "最差超额年化收益",
    "avg_strategy_sharpe": "平均策略夏普",
    "median_strategy_sharpe": "策略夏普中位数",
    "std_strategy_sharpe": "策略夏普标准差",
    "worst_strategy_sharpe": "最差策略夏普",
    "avg_sharpe_diff": "平均夏普差值",
    "avg_strategy_max_drawdown": "平均策略最大回撤",
    "avg_drawdown_improvement": "平均回撤改善",
    "return_win_rate": "超额收益胜率",
    "sharpe_win_rate": "夏普胜率",
    "drawdown_win_rate": "回撤改善胜率",
    "avg_exposure": "平均持仓比例",
    "avg_trade_count": "平均交易次数",
    "avg_transaction_cost": "平均交易成本",
}


def build_ma_parameter_grid(
    fast_windows: Iterable[int],
    slow_windows: Iterable[int],
    min_gap: int = 1,
) -> list[tuple[int, int]]:
    """
    生成合法的双均线参数组合。

    Parameters
    ----------
    fast_windows:
        快均线窗口集合。

    slow_windows:
        慢均线窗口集合。

    min_gap:
        慢均线与快均线之间的最小距离。

        参数组合必须满足：

            slow_window - fast_window >= min_gap

    Returns
    -------
    list[tuple[int, int]]
        合法的 (fast_window, slow_window) 参数组合。
    """
    if not isinstance(min_gap, int):
        raise TypeError("min_gap 必须是整数")

    if min_gap < 1:
        raise ValueError("min_gap 必须大于等于 1")

    fast_values = _normalize_windows(
        windows=fast_windows,
        parameter_name="fast_windows",
    )

    slow_values = _normalize_windows(
        windows=slow_windows,
        parameter_name="slow_windows",
    )

    parameter_grid = [
        (fast_window, slow_window)
        for fast_window in fast_values
        for slow_window in slow_values
        if slow_window - fast_window >= min_gap
    ]

    if not parameter_grid:
        raise ValueError(
            "没有生成合法参数组合，请检查均线窗口和 min_gap"
        )

    return parameter_grid


def aggregate_ma_parameter_results(
    grid_detail: pd.DataFrame,
) -> pd.DataFrame:
    """
    将参数网格回测明细汇总成每组参数一行。

    grid_detail 通常来自：

        backtest.run_ma_parameter_grid_search()

    输入明细中的每一行表示：

        一只股票 + 一组 MA 参数 + 样本内绩效

    Returns
    -------
    pd.DataFrame
        每组参数在整个股票池上的汇总结果。
    """
    data = _prepare_grid_detail(grid_detail)

    parameter_summary = (
        data.groupby(
            [
                "fast_window",
                "slow_window",
                "ma_param",
            ],
            as_index=False,
        )
        .agg(
            stock_count=("symbol", "nunique"),
            valid_sharpe_count=("strategy_sharpe", "count"),

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

            avg_sharpe_diff=(
                "sharpe_diff",
                "mean",
            ),
            avg_strategy_max_drawdown=(
                "strategy_max_drawdown",
                "mean",
            ),
            avg_drawdown_improvement=(
                "drawdown_improvement",
                "mean",
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
                _positive_rate,
            ),
            sharpe_win_rate=(
                "sharpe_diff",
                _positive_rate,
            ),
            drawdown_win_rate=(
                "drawdown_improvement",
                _positive_rate,
            ),
        )
        .sort_values(
            [
                "fast_window",
                "slow_window",
            ]
        )
        .reset_index(drop=True)
    )

    standard_deviation_columns = [
        "std_strategy_annual_return",
        "std_excess_annual_return",
        "std_strategy_sharpe",
    ]

    parameter_summary[standard_deviation_columns] = (
        parameter_summary[
            standard_deviation_columns
        ].fillna(0.0)
    )

    return parameter_summary


def calculate_local_parameter_robustness(
    parameter_summary: pd.DataFrame,
    metric: str = "avg_strategy_sharpe",
    fast_radius: int | None = None,
    slow_radius: int | None = None,
    stability_penalty: float = 1.0,
) -> pd.DataFrame:
    """
    计算每个参数点附近的局部稳定性。

    默认情况下，邻域半径使用参数网格中对应方向的
    最小正间隔。

    例如：

        fast_windows = [10, 15, 20, 25]
        slow_windows = [40, 50, 60, 70]

    自动得到：

        fast_radius = 5
        slow_radius = 10

    稳健性分数定义为：

        robustness_score
        = local_mean - stability_penalty * local_std

    其中：

    local_mean:
        参数邻域内的平均表现。

    local_std:
        参数邻域内的表现标准差。

    point_minus_local_mean:
        当前参数点表现减去邻域平均表现。
        该值特别大时，可能表示当前点是孤立尖峰。

    Parameters
    ----------
    parameter_summary:
        aggregate_ma_parameter_results 的返回结果。

    metric:
        用于分析参数稳定性的指标。

    fast_radius:
        快均线方向的邻域半径。
        None 表示自动推断。

    slow_radius:
        慢均线方向的邻域半径。
        None 表示自动推断。

    stability_penalty:
        对局部标准差的惩罚系数。

    Returns
    -------
    pd.DataFrame
        包含局部统计指标和稳健性分数的参数表。
    """
    required_columns = {
        "fast_window",
        "slow_window",
        "ma_param",
        metric,
    }

    _check_required_columns(
        data=parameter_summary,
        required_columns=required_columns,
        data_name="parameter_summary",
    )

    if parameter_summary.empty:
        raise ValueError("parameter_summary 不能为空")

    if stability_penalty < 0:
        raise ValueError("stability_penalty 不能小于 0")

    data = parameter_summary.copy()

    data[metric] = pd.to_numeric(
        data[metric],
        errors="coerce",
    )

    if data[metric].notna().sum() == 0:
        raise ValueError(f"指标 {metric} 没有有效数值")

    if fast_radius is None:
        fast_radius = _infer_grid_step(
            data["fast_window"]
        )

    if slow_radius is None:
        slow_radius = _infer_grid_step(
            data["slow_window"]
        )

    if not isinstance(fast_radius, int):
        raise TypeError("fast_radius 必须是整数")

    if not isinstance(slow_radius, int):
        raise TypeError("slow_radius 必须是整数")

    if fast_radius < 0 or slow_radius < 0:
        raise ValueError("邻域半径不能小于 0")

    local_statistics = []

    for row in data.itertuples(index=False):
        fast_window = int(row.fast_window)
        slow_window = int(row.slow_window)

        neighborhood_mask = (
            data["fast_window"]
            .sub(fast_window)
            .abs()
            <= fast_radius
        ) & (
            data["slow_window"]
            .sub(slow_window)
            .abs()
            <= slow_radius
        )

        neighborhood_values = (
            data.loc[
                neighborhood_mask,
                metric,
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .astype(float)
        )

        if neighborhood_values.empty:
            local_statistics.append(
                {
                    "local_parameter_count": 0,
                    "local_mean": np.nan,
                    "local_std": np.nan,
                    "local_min": np.nan,
                    "local_max": np.nan,
                    "local_range": np.nan,
                    "point_minus_local_mean": np.nan,
                    "is_local_peak": False,
                }
            )
            continue

        point_value = float(
            getattr(row, metric)
        )

        local_mean = float(
            neighborhood_values.mean()
        )

        local_std = float(
            neighborhood_values.std(ddof=0)
        )

        local_min = float(
            neighborhood_values.min()
        )

        local_max = float(
            neighborhood_values.max()
        )

        local_statistics.append(
            {
                "local_parameter_count": int(
                    len(neighborhood_values)
                ),
                "local_mean": local_mean,
                "local_std": local_std,
                "local_min": local_min,
                "local_max": local_max,
                "local_range": local_max - local_min,
                "point_minus_local_mean": (
                    point_value - local_mean
                ),
                "is_local_peak": bool(
                    np.isclose(
                        point_value,
                        local_max,
                    )
                ),
            }
        )

    local_statistics_df = pd.DataFrame(
        local_statistics
    )

    result = pd.concat(
        [
            data.reset_index(drop=True),
            local_statistics_df,
        ],
        axis=1,
    )

    result["robustness_metric"] = metric
    result["fast_radius"] = fast_radius
    result["slow_radius"] = slow_radius
    result["stability_penalty"] = stability_penalty

    result["robustness_score"] = (
        result["local_mean"]
        - stability_penalty * result["local_std"]
    )

    result = (
        result.sort_values(
            [
                "robustness_score",
                "local_std",
                metric,
            ],
            ascending=[
                False,
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    return result


def select_robust_ma_parameter(
    robustness_summary: pd.DataFrame,
    min_local_parameter_count: int = 9,
    min_return_win_rate: float = 0.6,
    min_avg_excess_return: float = 0.0,
    max_sharpe_gap: float = 0.20,
) -> tuple[int, int]:
    """
    先进行表现过滤，再从合格参数中选择稳健参数。

    max_sharpe_gap 表示：
    候选参数的平均夏普不能比本轮最高平均夏普
    低超过指定数值。
    """
    required_columns = {
        "fast_window",
        "slow_window",
        "avg_strategy_sharpe",
        "avg_excess_annual_return",
        "return_win_rate",
        "local_parameter_count",
        "local_mean",
        "local_std",
        "local_min",
        "robustness_score",
    }

    missing_columns = (
        required_columns
        - set(robustness_summary.columns)
    )

    if missing_columns:
        raise ValueError(
            f"robustness_summary 缺少字段："
            f"{sorted(missing_columns)}"
        )

    if robustness_summary.empty:
        raise ValueError(
            "robustness_summary 不能为空"
        )

    data = robustness_summary.copy()

    complete_neighborhood = data.loc[
        data["local_parameter_count"]
        >= min_local_parameter_count
    ].copy()

    if complete_neighborhood.empty:
        complete_neighborhood = data.copy()

    best_avg_sharpe = complete_neighborhood[
        "avg_strategy_sharpe"
    ].max()

    sharpe_threshold = (
        complete_neighborhood[
            "avg_strategy_sharpe"
        ].quantile(0.70)
    )

    candidates = complete_neighborhood.loc[
        (
            complete_neighborhood[
                "avg_strategy_sharpe"
            ] >= sharpe_threshold
        )
        & (
            complete_neighborhood[
                "avg_excess_annual_return"
            ] > 0
        )
        & (
            complete_neighborhood[
                "return_win_rate"
            ] >= 0.6
        )
    ].copy()

    # 过滤条件太严格时，退回完整邻域参数，
    # 避免程序无法继续运行。
    if candidates.empty:
        candidates = complete_neighborhood.copy()

    best_row = (
        candidates.sort_values(
            [
                "robustness_score",
                "local_min",
                "avg_strategy_sharpe",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    return (
        int(best_row["fast_window"]),
        int(best_row["slow_window"]),
    )


def create_parameter_matrix(
    parameter_summary: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    将参数长表转换为二维参数矩阵。

    行：
        fast_window

    列：
        slow_window

    单元格：
        metric
    """
    required_columns = {
        "fast_window",
        "slow_window",
        metric,
    }

    _check_required_columns(
        data=parameter_summary,
        required_columns=required_columns,
        data_name="parameter_summary",
    )

    if parameter_summary.empty:
        raise ValueError("parameter_summary 不能为空")

    matrix = parameter_summary.pivot(
        index="fast_window",
        columns="slow_window",
        values=metric,
    )

    return (
        matrix
        .sort_index()
        .sort_index(axis=1)
    )


def plot_parameter_heatmap(
    parameter_summary: pd.DataFrame,
    metric: str = "avg_strategy_sharpe",
    selected_parameter: tuple[int, int] | None = None,
    metric_label: str | None = None,
    value_format: str = ".2f",
    title: str | None = None,
    cmap: str = "viridis",
    annotate: bool = True,
) -> tuple[Figure, Axes]:
    """
    绘制快均线 × 慢均线参数热力图。

    函数不会调用 plt.show()，也不会自动保存图片。

    notebook 中可以使用：

        fig, ax = plot_parameter_heatmap(...)
        plt.show()

    或：

        fig.savefig(...)
    """
    matrix = create_parameter_matrix(
        parameter_summary=parameter_summary,
        metric=metric,
    )

    metric_display_name = _get_metric_label(
        metric=metric,
        custom_label=metric_label,
    )

    values = matrix.to_numpy(dtype=float)

    masked_values = np.ma.masked_invalid(
        values
    )

    figure_width = max(
        8.0,
        len(matrix.columns) + 3.0,
    )

    figure_height = max(
        5.5,
        len(matrix.index) * 0.7 + 2.5,
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    image = axis.imshow(
        masked_values,
        aspect="auto",
        origin="upper",
        cmap=cmap,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        metric_display_name
    )

    axis.set_xticks(
        np.arange(len(matrix.columns))
    )

    axis.set_xticklabels(
        matrix.columns
    )

    axis.set_yticks(
        np.arange(len(matrix.index))
    )

    axis.set_yticklabels(
        matrix.index
    )

    axis.set_xlabel("慢均线窗口")
    axis.set_ylabel("快均线窗口")

    axis.set_title(
        title
        or f"MA 参数敏感性：{metric_display_name}"
    )

    if annotate:
        _annotate_heatmap(
            axis=axis,
            image=image,
            values=values,
            value_format=value_format,
        )

    if selected_parameter is not None:
        _highlight_heatmap_parameter(
            axis=axis,
            matrix=matrix,
            selected_parameter=selected_parameter,
        )

    figure.tight_layout()

    return figure, axis


def plot_parameter_slice(
    parameter_summary: pd.DataFrame,
    metric: str,
    fixed_parameter: Literal["fast", "slow"],
    fixed_value: int,
    selected_parameter: tuple[int, int] | None = None,
    metric_label: str | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    固定一个均线参数，观察另一个参数变化时的指标曲线。

    fixed_parameter="fast":
        固定快均线，横轴为慢均线。

    fixed_parameter="slow":
        固定慢均线，横轴为快均线。
    """
    required_columns = {
        "fast_window",
        "slow_window",
        metric,
    }

    _check_required_columns(
        data=parameter_summary,
        required_columns=required_columns,
        data_name="parameter_summary",
    )

    if fixed_parameter == "fast":
        subset = parameter_summary.loc[
            parameter_summary["fast_window"]
            == fixed_value
        ].sort_values("slow_window")

        x_column = "slow_window"
        x_label = "慢均线窗口"
        fixed_label = f"快均线={fixed_value}"

    elif fixed_parameter == "slow":
        subset = parameter_summary.loc[
            parameter_summary["slow_window"]
            == fixed_value
        ].sort_values("fast_window")

        x_column = "fast_window"
        x_label = "快均线窗口"
        fixed_label = f"慢均线={fixed_value}"

    else:
        raise ValueError(
            "fixed_parameter 必须是 'fast' 或 'slow'"
        )

    if subset.empty:
        raise ValueError(
            f"没有找到 {fixed_label} 对应的参数结果"
        )

    metric_display_name = _get_metric_label(
        metric=metric,
        custom_label=metric_label,
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        subset[x_column],
        subset[metric],
        marker="o",
        linewidth=1.8,
    )

    axis.axhline(
        y=0.0,
        linewidth=1.0,
        alpha=0.5,
    )

    axis.set_xlabel(x_label)
    axis.set_ylabel(metric_display_name)

    axis.set_title(
        title
        or (
            f"{fixed_label} 时的参数切片："
            f"{metric_display_name}"
        )
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    if selected_parameter is not None:
        _highlight_slice_parameter(
            axis=axis,
            subset=subset,
            metric=metric,
            x_column=x_column,
            fixed_parameter=fixed_parameter,
            selected_parameter=selected_parameter,
        )

    figure.tight_layout()

    return figure, axis


def plot_performance_stability(
    robustness_summary: pd.DataFrame,
    selected_parameter: tuple[int, int] | None = None,
    metric_label: str | None = None,
    annotate: bool = True,
    title: str = "参数表现与局部稳定性",
) -> tuple[Figure, Axes]:
    """
    绘制参数邻域平均表现与邻域标准差。

    横轴越大：
        邻域平均表现越好。

    纵轴越低：
        邻域波动越小，参数越稳定。
    """
    required_columns = {
        "fast_window",
        "slow_window",
        "ma_param",
        "local_mean",
        "local_std",
        "robustness_metric",
    }

    _check_required_columns(
        data=robustness_summary,
        required_columns=required_columns,
        data_name="robustness_summary",
    )

    if robustness_summary.empty:
        raise ValueError(
            "robustness_summary 不能为空"
        )

    metric = str(
        robustness_summary[
            "robustness_metric"
        ].iloc[0]
    )

    metric_display_name = _get_metric_label(
        metric=metric,
        custom_label=metric_label,
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.scatter(
        robustness_summary["local_mean"],
        robustness_summary["local_std"],
        s=55,
        alpha=0.8,
    )

    if annotate:
        for row in robustness_summary.itertuples(
            index=False
        ):
            if (
                pd.isna(row.local_mean)
                or pd.isna(row.local_std)
            ):
                continue

            axis.annotate(
                row.ma_param,
                (
                    row.local_mean,
                    row.local_std,
                ),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    if selected_parameter is not None:
        selected_fast, selected_slow = (
            selected_parameter
        )

        selected_row = robustness_summary.loc[
            (
                robustness_summary["fast_window"]
                == selected_fast
            )
            & (
                robustness_summary["slow_window"]
                == selected_slow
            )
        ]

        if not selected_row.empty:
            axis.scatter(
                selected_row["local_mean"],
                selected_row["local_std"],
                s=160,
                marker="*",
                zorder=4,
            )

    axis.set_xlabel(
        f"邻域平均：{metric_display_name}"
    )

    axis.set_ylabel(
        "邻域标准差（越低越稳定）"
    )

    axis.set_title(title)

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    return figure, axis


def run_ma_parameter_sensitivity(
    stock_list: list[str],
    fast_windows: Iterable[int],
    slow_windows: Iterable[int],
    in_sample_end: str = "2024-12-31",
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0002,
    annual_risk_free_rate: float = 0.0,
    trading_days: int = 252,
    min_gap: int = 1,
    robustness_metric: str = "avg_strategy_sharpe",
    fast_radius: int | None = None,
    slow_radius: int | None = None,
    stability_penalty: float = 1.0,
    min_local_parameter_count: int = 4,
) -> dict[str, object]:
    """
    运行完整的 MA 参数敏感性计算流程。

    流程
    ----
    1. 生成参数网格；
    2. 调用现有的 MA 参数网格回测；
    3. 汇总每组参数的跨股票表现；
    4. 计算参数局部稳定性；
    5. 选择稳健参数候选。

    本函数不会：

    1. 自动保存 CSV；
    2. 自动保存图片；
    3. 自动调用 plt.show()；
    4. 使用样本外数据选择参数。

    Returns
    -------
    dict[str, object]

    parameter_grid:
        合法的参数组合。

    grid_detail:
        每只股票、每组参数的样本内绩效。

    grid_results:
        每只股票、每组参数的完整逐日回测结果。

    parameter_summary:
        每组参数的跨股票汇总结果。

    robustness_summary:
        每组参数的局部稳定性结果。

    robust_parameter:
        稳健参数候选。
    """
    if not stock_list:
        raise ValueError("stock_list 不能为空")

    parameter_grid = build_ma_parameter_grid(
        fast_windows=fast_windows,
        slow_windows=slow_windows,
        min_gap=min_gap,
    )

    run_grid_search = (
        _import_grid_search_function()
    )

    grid_detail, grid_results = run_grid_search(
        stock_list=stock_list,
        param_grid=parameter_grid,
        in_sample_end=in_sample_end,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        annual_risk_free_rate=annual_risk_free_rate,
        trading_days=trading_days,
    )

    if grid_detail.empty:
        raise ValueError(
            "参数网格回测没有产生结果，"
            "请检查股票文件和参数范围"
        )

    parameter_summary = (
        aggregate_ma_parameter_results(
            grid_detail=grid_detail
        )
    )

    robustness_summary = (
        calculate_local_parameter_robustness(
            parameter_summary=parameter_summary,
            metric=robustness_metric,
            fast_radius=fast_radius,
            slow_radius=slow_radius,
            stability_penalty=stability_penalty,
        )
    )

    robust_parameter = (
        select_robust_ma_parameter(
            robustness_summary=robustness_summary,
            min_local_parameter_count=(
                min_local_parameter_count
            ),
        )
    )

    return {
        "parameter_grid": parameter_grid,
        "grid_detail": grid_detail,
        "grid_results": grid_results,
        "parameter_summary": parameter_summary,
        "robustness_summary": robustness_summary,
        "robust_parameter": robust_parameter,
    }


def _prepare_grid_detail(
    grid_detail: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "fast_window",
        "slow_window",
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

    _check_required_columns(
        data=grid_detail,
        required_columns=required_columns,
        data_name="grid_detail",
    )

    if grid_detail.empty:
        raise ValueError("grid_detail 不能为空")

    data = grid_detail.copy()

    numeric_columns = (
        required_columns - {"symbol"}
    )

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "fast_window",
            "slow_window",
            "strategy_annual_return",
            "excess_annual_return",
        ]
    ).copy()

    if data.empty:
        raise ValueError(
            "grid_detail 清洗后没有有效记录"
        )

    data["fast_window"] = (
        data["fast_window"].astype(int)
    )

    data["slow_window"] = (
        data["slow_window"].astype(int)
    )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.zfill(6)
    )

    data["ma_param"] = (
        data["fast_window"].astype(str)
        + "/"
        + data["slow_window"].astype(str)
    )

    return data


def _normalize_windows(
    windows: Iterable[int],
    parameter_name: str,
) -> list[int]:
    normalized_values = []

    for value in windows:
        if (
            isinstance(value, bool)
            or not isinstance(
                value,
                (int, np.integer),
            )
        ):
            raise TypeError(
                f"{parameter_name} 中的窗口值必须是整数"
            )

        integer_value = int(value)

        if integer_value <= 0:
            raise ValueError(
                f"{parameter_name} 中的窗口值必须大于 0"
            )

        normalized_values.append(
            integer_value
        )

    unique_values = sorted(
        set(normalized_values)
    )

    if not unique_values:
        raise ValueError(
            f"{parameter_name} 不能为空"
        )

    return unique_values


def _positive_rate(
    values: pd.Series,
) -> float:
    clean_values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if clean_values.empty:
        return np.nan

    return float(
        (clean_values > 0).mean()
    )


def _infer_grid_step(
    values: pd.Series,
) -> int:
    unique_values = np.sort(
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    if len(unique_values) <= 1:
        return 0

    differences = np.diff(
        unique_values
    )

    positive_differences = differences[
        differences > 0
    ]

    if len(positive_differences) == 0:
        return 0

    return int(
        positive_differences.min()
    )


def _check_required_columns(
    data: pd.DataFrame,
    required_columns: set[str],
    data_name: str,
) -> None:
    missing_columns = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{data_name} 缺少必要字段："
            f"{sorted(missing_columns)}"
        )


def _get_metric_label(
    metric: str,
    custom_label: str | None,
) -> str:
    if custom_label is not None:
        return custom_label

    return METRIC_LABELS.get(
        metric,
        metric,
    )


def _annotate_heatmap(
    axis: Axes,
    image,
    values: np.ndarray,
    value_format: str,
) -> None:
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
                fontsize=9,
            )


def _highlight_heatmap_parameter(
    axis: Axes,
    matrix: pd.DataFrame,
    selected_parameter: tuple[int, int],
) -> None:
    selected_fast, selected_slow = (
        selected_parameter
    )

    if (
        selected_fast not in matrix.index
        or selected_slow not in matrix.columns
    ):
        return

    row_index = matrix.index.get_loc(
        selected_fast
    )

    column_index = matrix.columns.get_loc(
        selected_slow
    )

    axis.add_patch(
        Rectangle(
            (
                column_index - 0.5,
                row_index - 0.5,
            ),
            width=1,
            height=1,
            fill=False,
            linewidth=2.5,
        )
    )


def _highlight_slice_parameter(
    axis: Axes,
    subset: pd.DataFrame,
    metric: str,
    x_column: str,
    fixed_parameter: Literal["fast", "slow"],
    selected_parameter: tuple[int, int],
) -> None:
    selected_fast, selected_slow = (
        selected_parameter
    )

    if fixed_parameter == "fast":
        selected_x = selected_slow
    else:
        selected_x = selected_fast

    selected_row = subset.loc[
        subset[x_column] == selected_x
    ]

    if selected_row.empty:
        return

    selected_y = float(
        selected_row[metric].iloc[0]
    )

    axis.scatter(
        [selected_x],
        [selected_y],
        s=100,
        zorder=3,
    )

    axis.annotate(
        f"{selected_fast}/{selected_slow}",
        (
            selected_x,
            selected_y,
        ),
        textcoords="offset points",
        xytext=(6, 8),
    )


def _import_grid_search_function():
    """
    同时兼容两种导入方式。

    作为包导入：

        from src.parameter_sensitivity import ...

    将 src 加入 sys.path 后导入：

        from parameter_sensitivity import ...
    """
    try:
        from .backtest import (
            run_ma_parameter_grid_search,
        )
    except ImportError:
        from backtest import (
            run_ma_parameter_grid_search,
        )

    return run_ma_parameter_grid_search