from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt

from metrics import calc_stock_metrics
from get_single_stock import get_single_stock, clean_stock_data


# =========================
# 1. 基础配置
# =========================
project_root = project_root = Path(__file__).resolve().parent.parent
RAW_DIR = project_root / "data" / "raw"
PROCESSED_DIR = project_root / "data" / "processed"
REPORT_DIR = project_root / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)



# =========================
# 批量获取、清洗、计算指标
# =========================
def get_batch_stock(
    stock_list: list,
    period: str = "daily",
    start_date: str = "20210101",
    end_date: str | None = None,
    adjust: str = "qfq",
    clean: bool = True,
    save_to_local: bool = True):
    
    raw_df = pd.DataFrame()
    clean_df = pd.DataFrame()
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
        
    for symbol in stock_list:
        try:
            print(f"正在获取 {symbol}...")

            raw_df_single = get_single_stock(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )

            if raw_df_single is None or raw_df_single.empty:
                print(f"{symbol} 数据为空，跳过")
                continue

            raw_df = pd.concat([raw_df, raw_df_single], ignore_index=True)

            raw_path = RAW_DIR / f"{symbol}_{adjust}_{period}.csv"
            if save_to_local:
                raw_df_single.to_csv(raw_path, index=False, encoding="utf-8-sig")

            clean_df_single = clean_stock_data(raw_df_single, symbol)
            clean_df = pd.concat([clean_df, clean_df_single], ignore_index=True)

            clean_path = PROCESSED_DIR / f"{symbol}_clean.csv"
            if save_to_local:
                clean_df_single.to_csv(clean_path, index=False, encoding="utf-8-sig")

            time.sleep(1)

        except Exception as e:
            print(f"{symbol} 获取或处理失败：{e}")
        
    return clean_df if clean else raw_df

def batch_stock_analysis(stock_list: list):
    
    results = []

    for symbol in stock_list:
        file_path = PROCESSED_DIR / f"{symbol}_clean.csv"

        if not file_path.exists():
            print(f"{symbol} 本地文件不存在，跳过")
            continue

        stock_df = pd.read_csv(file_path)
        metrics = calc_stock_metrics(stock_df)
        results.append(metrics)

    if not results:
        return pd.DataFrame()
    result_df = pd.DataFrame(results)
    # 按夏普比率从高到低排序
    result_df = result_df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

    # 保存原始数值版结果
    raw_report_path = REPORT_DIR / "stock_metrics_ranking_raw.csv"
    result_df.to_csv(raw_report_path, index=False, encoding="utf-8-sig")

    # =========================
    # 6. 生成更易读的百分比展示版
    # =========================

    display_df = result_df.copy()

    percent_cols = [
        "cumulative_return",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
    ]

    for col in percent_cols:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")

    display_df["sharpe_ratio"] = display_df["sharpe_ratio"].map(lambda x: f"{x:.2f}")

    display_df["start_date"] = pd.to_datetime(display_df["start_date"]).dt.date
    display_df["end_date"] = pd.to_datetime(display_df["end_date"]).dt.date
    display_df["max_drawdown_start_date"] = pd.to_datetime(display_df["max_drawdown_start_date"]).dt.date
    display_df["max_drawdown_end_date"] = pd.to_datetime(display_df["max_drawdown_end_date"]).dt.date

    display_cols = [
        "symbol",
        "start_date",
        "end_date",
        "trading_days",
        "cumulative_return",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "max_drawdown_start_date",
        "max_drawdown_end_date",
    ]

    display_df = display_df[display_cols]

    display_report_path = REPORT_DIR / "stock_metrics_ranking_display.csv"
    display_df.to_csv(display_report_path, index=False, encoding="utf-8-sig")
    print(f"\n原始结果已保存到：{raw_report_path}")
    print(f"展示结果已保存到：{display_report_path}")
    return display_df


def batch_stock_plot(stock_list: list):
    PROCESSED_DIR = project_root / "data" / "processed"

    plt.figure(figsize=(12, 12))
    plt.subplot(211)
    for symbol in stock_list:
        file_path = PROCESSED_DIR / f"{symbol}_clean.csv"
        df_stock = pd.read_csv(file_path)

        df_stock["date"] = pd.to_datetime(df_stock["date"])
        df_stock = df_stock.sort_values("date").reset_index(drop=True)

        df_stock["daily_return"] = df_stock["close"].pct_change().fillna(0)
        df_stock["net_value"] = (1 + df_stock["daily_return"]).cumprod()

        plt.plot(df_stock["date"], df_stock["net_value"], label=symbol)

    plt.title("Net Value Comparison")
    plt.xlabel("Date")
    plt.ylabel("Net Value")
    plt.grid(True)
    plt.legend()

    plt.subplot(212)
    for symbol in stock_list:
        file_path = PROCESSED_DIR / f"{symbol}_clean.csv"
        df_stock = pd.read_csv(file_path)

        df_stock["date"] = pd.to_datetime(df_stock["date"])
        df_stock = df_stock.sort_values("date").reset_index(drop=True)

        df_stock["daily_return"] = df_stock["close"].pct_change().fillna(0)
        df_stock["net_value"] = (1 + df_stock["daily_return"]).cumprod()

        df_stock["rolling_max"] = df_stock["net_value"].cummax()
        df_stock["drawdown"] = df_stock["net_value"] / df_stock["rolling_max"] - 1

        plt.plot(df_stock["date"], df_stock["drawdown"], label=symbol)

    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    


if (__name__ == "__main__"):
    stock_list = ["000001", "000002", "600519", "600036", "300750"]
    start_date = "20210101"
    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    # =========================
    # 7. 打印结果
    # =========================
    display_df = batch_stock_analysis(stock_list=stock_list)
    print("\n风险收益排名表：")
    print(display_df)